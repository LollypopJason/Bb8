# Deploying to the robot (Jetson Nano + Teensy 4.0)

```
 development PC                         Jetson Nano (Python ≥ 3.6, numpy)          Teensy 4.0
 ──────────────                         ───────────────────────────────           ──────────
 train / tune in the sim                deploy/runtime/runner.py  100 Hz          bb8_mcu.ino
 python -m deploy.export_bundle  ──►    IMU (I2C) ─► estimator ─► PID/LQR/RL ─USB─► step pulses (50 kHz ISR)
   bundle.json, policy_mlp.npz,         safety, arming, teleop (UDP), CSV log ◄──── encoder counts, watchdog
   policy.onnx                                                                     │
 python -m deploy.sim_io   (same runtime against MuJoCo before touching hardware)  └─► STEP/DIR drivers ─► NEMA 17
```

**Why a microcontroller:** three steppers at 1/16 microstepping and 18.85 rad/s need about 9,600 step
pulses per second each, with microsecond timing. The encoders produce about 3,600 quadrature edges per
second each. Linux Python on the Nano can't time either reliably. The Teensy handles both; the Jetson
handles estimation, control and safety.

## 1. Export a bundle (development PC)

```bash
python -m deploy.export_bundle --out bundle --controller pid                 # start here
python -m deploy.export_bundle --out bundle --controller lqr --magnet 20     # LQR needs the preload
python -m deploy.export_bundle --out bundle --controller rl --policy runs/<run>/final.zip
```

The bundle always contains the PID gains and the LQR gain matrix, so you can choose the controller
with `--controller` on the robot. The LQR is designed through `SceneStepper` for the configured delay.
For a policy, the geometry and hardware profile are copied from the training run. The exporter warns
if the policy was trained on observations the robot can't reproduce (ground-truth ball sensors, or
the pre-fix observation ordering). Retrain such policies before deploying them.

**Check the bundle in software-in-the-loop first.** This runs the unmodified robot runtime against
MuJoCo, through a model of the firmware, with encoder quantisation and your motor/encoder signs:

```bash
python -m deploy.sim_io --bundle bundle --config robot.json --controller pid --seconds 20 --push 6 --randomize
```

## 2. Wiring (Teensy 4.0 defaults, edit the pin map at the top of `bb8_mcu.ino`)

| signal | Teensy pins | notes |
|---|---|---|
| STEP 0/1/2 | 2, 4, 6 | to the drivers' STEP inputs |
| DIR 0/1/2 | 3, 5, 7 | |
| ENABLE (all drivers) | 10 | active low (A4988 / DRV8825 / TMC2209) |
| encoder A/B, wheel 0/1/2 | 14/15, 16/17, 18/19 | **3.3 V only: Teensy 4 pins are not 5 V tolerant.** Use level shifters for 5 V encoders. |
| USB | Jetson USB port | serial device `/dev/ttyACM0` |
| BNO085 / MPU6050 | Jetson header pin 3 (SDA), 5 (SCL), 1 (3.3 V), 6 (GND) | I2C bus 1 |

* The drivers' microstep pins must match `microsteps` in `robot.json` (default 16).
* Set the driver current limit for the 17HS19-2004S1 (2.0 A rated). TMC2209 drivers run cooler and
  quieter than A4988s.
* Use 24 V motor power. Tie the ground of the motor supply, the Teensy and the Jetson together.
* Wheel *i* is the wheel at azimuth `bundle.json → robot.wheel_azimuth_deg[i]` (90°, 210°, 330°),
  measured counter-clockwise from the head's forward (+x) axis, viewed from above.

Flash the firmware with Arduino + Teensyduino: open `deploy/firmware/bb8_mcu/bb8_mcu.ino`, select
Teensy 4.0, then Upload. `bb8_protocol.h` is shared with the Python side and tested against it.

## 3. Jetson Nano setup (JetPack 4.6, Ubuntu 18.04, Python 3.6)

```bash
sudo apt update && sudo apt install -y python3-pip python3-numpy
pip3 install -r deploy/requirements-jetson.txt
sudo usermod -aG dialout,i2c $USER          # serial + I2C without sudo, then log out/in
sudo nvpmodel -m 0 && sudo jetson_clocks    # max clocks for a steady 100 Hz loop
```

Copy the repository (or just `deploy/`) plus your bundle to the Jetson. The runtime imports only
numpy, pyserial and the IMU driver: no MuJoCo, torch or gym.

* **BNO085:** uses Adafruit Blinka with `adafruit-circuitpython-bno08x`, reading the *game rotation
  vector* (gyro + accelerometer, no magnetometer). The steppers and the magnetic preload would corrupt
  a magnetometer. Recent Adafruit releases have dropped Python 3.6 support. If pip can't install them,
  run the runtime with `python3.8` (`sudo apt install python3.8`), or use the MPU6050 driver, which
  needs only smbus2.
* **MPU6050:** raw registers via smbus2, with an on-board complementary attitude filter.
* **onnxruntime** is optional (`--onnx`). By default the policy runs in numpy from `policy_mlp.npz`.

## 4. Bring-up checklist

1. `python3 -m deploy.runtime.runner write-config robot.json`, then edit the serial port, IMU type
   and `mount_euler_deg` (rotation from IMU axes to head axes: head x forward, z up).
2. **Signs, with the head OFF the ball:** run `python3 -m deploy.runtime.runner spin-test --config robot.json`.
   A positive command must turn each wheel so that **the point that touches the ball moves clockwise,
   viewed from above the head**. If a wheel turns the other way, flip its `motor_sign`. Then set
   `encoder_sign` to the suggested value. A wrong sign makes the robot fall straight away, and the
   SIL test `test_safety_faults` shows what that looks like.
3. **Calibrate:** hold the head upright on the ball, still, then run
   `python3 -m deploy.runtime.runner calibrate --config robot.json`. This writes the gyro bias and
   the level offset.
4. **Software in the loop** with your `robot.json` (step 1 of section 1).
5. **First balance with PID:** run
   `python3 -m deploy.runtime.runner run --bundle bundle --config robot.json --controller pid`.
   Hold the robot upright, press **Enter** to arm, and it arms once held within 3° for 1 s. Keep a hand
   near it. `x` + Enter disarms; Ctrl-C stops and disables the drivers. A CSV log goes to `logs/`.
6. **Teleop from a laptop:** `python3 -m deploy.runtime.teleop --host <jetson-ip>`. WASD/QE drive,
   space arms, x disarms. Without datagrams for 0.5 s the target returns to standstill.
7. Move on to LQR only with the magnetic preload (or a counterweight), and to RL only with a policy
   retrained on the current environment and checked in SIL.

## Safety behaviour

| condition | reaction |
|---|---|
| not armed | drivers disabled, zero rates |
| tilt > `tilt_cutoff_deg` (30°) | FAULT: drivers ramp to zero and disable; needs a new arm request |
| no MCU telemetry for `telemetry_timeout_s` (50 ms) | FAULT |
| firmware watchdog tripped | FAULT |
| Jetson stops sending for `watchdog_ms` (100 ms) | firmware ramps to zero and disables the drivers on its own |
| non-finite controller output | FAULT |

## What the simulation found (read before LQR / RL)

* **Traction limit without preload.** Each wheel carries only about 7 N of normal load from the head's
  weight, but the stepper can push about 15 N at the rim. An aggressive controller slips the wheels;
  wheel odometry, which assumes no slip, then reports motion that isn't happening, and the loop feeds
  on its own error. In simulation, LQR without preload fell in 12–25% of noisy, randomised episodes,
  while PID fell in none. With a 20 N preload both stayed up in every episode.
* **Odometry is the only ball velocity source** on the robot, so the simulator now uses the same
  estimator by default (`HardwareProfile.ball_state_source = "odometry"`). 1200 CPR at 100 Hz
  quantises wheel speed to about 0.5 rad/s, and a 40 ms low-pass (`odometry_velocity_tau`) is needed.
  Without it, LQR fell in every test episode.
* **Policies trained before this module** (for example the 1M-step demo run) used ground-truth ball
  sensors and a one-step-stale "previous action" in the observation. Retrain them.

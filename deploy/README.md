# Deploying to the robot: Jetson Nano + Arduino IoT IMU + stepper drivers on Jetson GPIO

```
 development PC                      Jetson Nano (JetPack 4.6, Python ≥ 3.6, numpy)
 ──────────────                      ───────────────────────────────────────────────────────────
 python -m deploy.export_bundle ──►  deploy/runtime/runner.py  (100 Hz)
   bundle.json, policy_mlp.npz         Arduino IMU frames ─► attitude fusion ─┐
 python -m deploy.sim_io  (same        step counts ─► wheel odometry ─────────┤
   runtime against MuJoCo)              ─► PID / LQR / RL ─► safety ─► step rates
                                                     │ Unix socket (bb8_protocol)
                                     jetson_stepgen/bb8_stepgen  (C, SCHED_FIFO, own CPU core)
                                       ─► sysfs GPIO ─► STEP / DIR / ENABLE ─► stepper drivers ─► NEMA 17
 Arduino Nano 33 IoT (on the head)
   firmware/bb8_imu: LSM6DS3 gyro + accel @ 416 Hz ─USB serial─► Jetson   (+ optional wheel encoders)
```

* **IMU:** the Arduino streams raw gyro and accelerometer samples with its own microsecond timestamps.
  The Jetson fuses every sample with a complementary filter (`imu.fusion_kp`), after removing the
  calibrated gyro bias. The LSM6DS3 has no magnetometer, which is what you want next to steppers and
  magnets: yaw drifts slowly, but balancing only uses yaw *rate*. The **Nano RP2040 Connect**
  (LSM6DSOX) runs the same sketch.
* **Steppers:** Python can't time step pulses, so `bb8_stepgen` does it. It's a small C daemon on
  one CPU core at real-time priority that writes STEP/DIR/ENABLE through sysfs GPIO, limits
  acceleration, and stops the motors if the runtime goes quiet for 100 ms or disconnects. It speaks
  the same protocol as the optional Teensy firmware (`firmware/bb8_mcu`), so the runtime doesn't care
  which one drives the motors.
* **Odometry:** wheel angles come from the commanded **step counts** by default. Steppers don't slip
  unless they stall. Encoders, if you fit them, can go on the Arduino (`odometry_source: imu_encoders`).

## 1. Export and check a bundle (development PC)

```bash
python -m deploy.export_bundle --out bundle --controller pid                 # start here
python -m deploy.export_bundle --out bundle --controller lqr --magnet 20     # LQR needs the preload
python -m deploy.export_bundle --out bundle --controller rl --policy runs/<run>/final.zip
python -m deploy.sim_io --bundle bundle --config robot.json --controller pid --seconds 20 --push 6 --randomize
```

`sim_io` runs the **unmodified robot runtime** against MuJoCo. It models the Arduino fusion at
500 Hz from noisy gyro and accel (so attitude errors during acceleration are included), the
daemon's step generation and acceleration limit, step-count odometry, and your motor signs. Results
over 5 randomised seeds, 15 s each, with 6 N pushes every 3 s:

| | no preload | 20 N magnetic preload |
|---|---|---|
| PID | 5/5 balanced, max tilt ≈ 8.5° | 5/5, ≈ 8.5° |
| LQR | **0/5** (wheels slip, odometry wrong) | 5/5, ≈ 6.4° |

## 2. Wiring

### Stepper drivers → Jetson Nano 40-pin header (defaults; any free GPIO works)

| signal | header pin | sysfs GPIO | |
|---|---|---|---|
| STEP 0 / 1 / 2 | 7 / 12 / 16 | 216 / 79 / 232 | `--step 216,79,232` |
| DIR 0 / 1 / 2 | 11 / 15 / 18 | 50 / 194 / 15 | `--dir 50,194,15` |
| ENABLE (all drivers) | 22 | 13 | `--enable 13`, active low |
| GND | 6, 9, 14, … | | shared with the driver logic and motor supply grounds |

* The sysfs numbers are those of the **Jetson Nano Developer Kit on JetPack 4.x**. Check them against
  the Jetson.GPIO pin table for your board, and use `sudo /opt/nvidia/jetson-io/jetson-io.py` to
  make sure none of those pins is configured for SPI or I2S.
* **Jetson GPIO is 3.3 V logic.** Power the drivers' logic supply (VDD on A4988/DRV8825, VIO on
  TMC2209) from **3.3 V**. With 5 V on VDD, an A4988 needs a high level of at least 3.5 V and won't
  see the Jetson's 3.3 V edges reliably. Motor power is 24 V.
* Set the drivers to **1/8 microstepping** to match `microsteps: 8` in `robot.json`. That's
  4,800 steps/s per motor at 18.85 rad/s. Finer microstepping means higher pulse rates and more jitter
  from Linux.
* Set the driver current limit for the 17HS19-2004S1 (2.0 A rated). TMC2209s are the quietest choice.
* Wheel *i* sits at azimuth `bundle.json → robot.wheel_azimuth_deg[i]` (90°, 210°, 330°), measured
  counter-clockwise from the head's forward (+x) axis, viewed from above.

### Arduino Nano 33 IoT → Jetson

* USB cable to the Jetson; it appears as `/dev/ttyACM0` (`imu.port`).
* **Mount the board rigidly on the head.** Enter its orientation in `imu.mount_euler_deg` as XYZ
  Euler angles in degrees, rotating the board's IMU axes into the head axes (head: x forward, z up).
  `[0, 0, 0]` means the board is flat with its IMU x axis pointing forward.
* Flash `deploy/firmware/bb8_imu/bb8_imu.ino` with the Arduino IDE (board package "Arduino SAMD"
  for the Nano 33 IoT, "Arduino Mbed OS Nano" for the RP2040 Connect). The LED blinks fast if no IMU
  is found and slowly while streaming.
* Optional encoders go on the Arduino: set `N_ENC` and the pins in the sketch, then set
  `odometry_source: imu_encoders`. **3.3 V signals only**; the SAMD21 isn't 5 V tolerant.

## 3. Jetson Nano setup

```bash
sudo apt update && sudo apt install -y python3-pip python3-numpy build-essential
pip3 install -r deploy/requirements-jetson.txt
sudo usermod -aG dialout $USER                       # Arduino serial without sudo; log out / in
cd deploy/jetson_stepgen && make && sudo make install  # bb8_stepgen + systemd unit
sudo systemctl enable --now bb8-stepgen               # edit the GPIO numbers in the unit first
sudo nvpmodel -m 0 && sudo jetson_clocks             # max clocks for steady timing
```

* For the best pulse timing, reserve CPU 3 for the daemon: add `isolcpus=3` to the `APPEND` line in
  `/boot/extlinux/extlinux.conf` and reboot. The daemon pins itself to `--cpu 3` (the default).
* `bb8_stepgen` needs root (or the `gpio` group plus `CAP_SYS_NICE`) for GPIO and real-time
  scheduling. The systemd unit runs it as root. Without root it warns and continues, with worse
  timing.
* Run it in the foreground for debugging with `sudo ./bb8_stepgen --step 216,79,232 --dir 50,194,15 --enable 13`.
  `./bb8_stepgen --sim` needs no GPIO and is for testing on any Linux PC.
* The runtime itself needs only numpy and pyserial (`smbus2` / Blinka only for the I2C IMU
  alternatives). By default the policy runs in numpy; `--onnx` uses onnxruntime if it's installed.

## 4. Bring-up checklist

1. `python3 -m deploy.runtime.runner write-config robot.json`, then edit `imu.port`,
   `imu.mount_euler_deg`, `microsteps` and the GPIO numbers in the daemon command line.
2. **Signs, with the head OFF the ball:** run `python3 -m deploy.runtime.runner spin-test --config robot.json`.
   Each wheel spins in turn. A positive command must turn the wheel so that **the point that touches
   the ball moves clockwise, viewed from above the head**. Flip `motor_sign` for any wheel that turns
   the other way. With encoders, then set `encoder_sign` to the suggested value.
3. **Calibrate:** hold the head upright on the ball and keep it still, then run
   `python3 -m deploy.runtime.runner calibrate --config robot.json`. This writes the gyro bias and
   the level offset. Any bias left over becomes a steady tilt error of bias / `fusion_kp`, so do this
   with the motors off and the robot still.
4. **Software in the loop** with your `robot.json` (section 1).
5. **First balance with PID:** run
   `python3 -m deploy.runtime.runner run --bundle bundle --config robot.json --controller pid`.
   Hold the robot upright and press **Enter**; it arms once held within 3° for 1 s. Keep a hand near
   it. `x` + Enter disarms; Ctrl-C stops and disables the drivers. A CSV log goes to `logs/`.
6. **Teleop from a laptop:** `python3 -m deploy.runtime.teleop --host <jetson-ip>`. WASD/QE drive,
   space arms, x disarms.
7. Move on to LQR only with the magnetic preload (or a counterweight), and to RL only with a policy
   retrained on the current environment and checked in SIL.

## Safety behaviour

| condition | reaction |
|---|---|
| not armed | drivers disabled, zero rates |
| tilt > `tilt_cutoff_deg` (30°) | FAULT: rates ramp to zero and the drivers are disabled; needs a new arm request |
| no IMU sample for `imu.timeout_s` (50 ms) | FAULT |
| no step-generator telemetry for 50 ms, or its watchdog tripped | FAULT |
| runtime stops sending for `watchdog_ms` (100 ms) or disconnects | `bb8_stepgen` ramps to zero and disables the drivers on its own |
| `bb8_stepgen` receives SIGINT / SIGTERM | STEP low, drivers disabled, then exit |
| non-finite controller output | FAULT |

## Notes from simulation

* **Traction limit without preload.** Each wheel carries only about 7 N of normal load from the head's
  weight, but the stepper can push about 15 N at the rim. The aggressive LQR slips the wheels, and
  odometry, which assumes no slip, corrupts the state estimate; see the table in section 1. PID
  tolerates this; LQR needs the preload.
* **Odometry is the only ball velocity source** on the robot, and the simulator uses the same
  estimator by default, with a 40 ms velocity low-pass (`odometry_velocity_tau`).
* **Policies trained before the deployment module** (for example the 1M-step demo run) used
  ground-truth ball sensors and a stale previous action in their observations. Retrain them. The
  exporter warns about this.

## Alternatives (still supported)

* `stepper_backend: teensy` with `firmware/bb8_mcu` on a Teensy 4.0 over USB: hardware-timed pulses,
  1/16 microstepping, encoders on the Teensy (`odometry_source: mcu_encoders`).
* `imu.type: bno085` (on-chip fusion, Adafruit Blinka) or `mpu6050` (smbus2) on Jetson I2C bus 1.

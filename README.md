# BB-8 Ball-Balancer Studio (MuJoCo)

A modular MuJoCo simulator and RL training studio built around a **BB-8 style
ball-balancer**: a layered hollow ball (air cavity / styrofoam / epoxy + sawdust)
driven by an omniwheel head unit that rides on top of it.

All dynamics run through `mujoco.MjModel` / `mujoco.MjData` / `mj_step`. There is
no placeholder physics.

## Project tree

```
Bb8/
├── requirements.txt
├── README.md
├── conftest.py
├── config/
│   └── composite_ball_params.py   # Module 1: dataclasses + physics calculators   [done]
├── models/
│   ├── mjcf_generator.py          # Module 2: MJCF builder, 5 examples, mesh import [done]
│   ├── runtime.py                 #   step-time helpers (contact alignment, magnet, pushes, delay line)
│   └── io_bus.py                  #   sensor/actuator bus (noise, bias drift, quantisation, latency)
├── tests/
│   ├── test_composite_ball.py
│   └── test_mjcf_generator.py
├── controllers/classical.py       # Module 3: omni mixers, cascaded PID, FD-LQR      [done]
├── envs/ball_balancer_env.py      # Module 4: Gymnasium env + domain randomisation   [done]
├── training/train_rl.py           # Module 5: SB3 PPO/SAC manager                    [done]
├── deploy/                        # Jetson Nano + Teensy deployment                  [done]
│   ├── export_bundle.py           #   freeze PID / LQR / policy + geometry into a bundle
│   ├── sim_io.py                  #   software-in-the-loop + pty firmware emulator
│   ├── runtime/                   #   robot side (Python 3.6, numpy): runner, estimator, controllers, IMU, MCU link
│   └── firmware/bb8_mcu/          #   Teensy 4.0 step generation, encoders, watchdog
└── app/
    ├── main_studio.py             # Module 6: FastAPI + WebSockets backend           [done]
    └── static/                    #   index.html, studio.js, studio.css, vendor/three (r160, MIT)
```

## Install & run

```bash
pip install -r requirements.txt
python -m pytest -q                                   # physics / controller / env / studio tests
python -m config.composite_ball_params                # ball calculator report
python -m models.mjcf_generator --list
python -m models.mjcf_generator --example bb8 --omni-mode subroller --magnet 20 --check 2 --out build/bb8
python -m mujoco.viewer --mjcf build/bb8/scene.xml     # inspect the written scene
```

> Scenes that use **contact-pair omniwheels** or the **magnet** must be stepped through
> `models.runtime.SceneStepper` (see below). The plain MuJoCo viewer still loads them,
> but it steps them with MuJoCo's default contact frames and no magnet force.

## Module 1: composite ball calculator

```python
from config.composite_ball_params import CompositeBallParams
ball = CompositeBallParams(inner_radius=0.200, foam_thickness=0.030, foam_density=35,
                           epoxy_thickness=0.004, epoxy_density=950)
# or CompositeBallParams.from_radii(0.200, 0.230, 0.234)
print(ball.report())
print(ball.mjcf_inertial_xml())
```

```
  layer             t [mm]  rho [kg/m3]     V [L]    m [kg]    I [kg m2]
  styrofoam          30.00         35.0    17.455    0.6109     0.018979
  epoxy_sawdust       4.00        950.0     2.706    2.5703     0.092240
  TOTAL              34.00                 53.671    3.1812     0.111219
  I/(m R2^2) = 0.6385  (solid sphere 0.400, thin shell 0.667)
```

Per layer: `m_i = 4/3 π ρ_i (R_i³ − R_{i−1}³)` and `I_i = 2/5 m_i (R_i⁵ − R_{i−1}⁵)/(R_i³ − R_{i−1}³)`.
The thin-wall limit is handled analytically. The totals are injected verbatim into
`<body name="composite_ball"><inertial pos="0 0 0" mass=… diaginertia="I I I"/>`, together with the
surface `friction`, `solref`, `solimp` and `condim`. Other contents of the module:

* `HeadUnitParams`: 3 or 4 omniwheels, zenith angle α ∈ [30°, 60°], CoM height, motors (torque limit,
  gear ratio, reflected rotor armature, damping), the omniwheel model, magnet preload and an internal
  counterweight. `wheel_geometry()` places each wheel; `ball_to_wheel_jacobian()` gives
  `ω_wheels = J Ω_ball,rel` with `J_i = −(R/r) a_iᵀ`.
* Parameter sets for every example, plus `SimulationParams` / `TerrainParams` (slope, bumps, heightfields).
* `scaled()` for domain randomisation, and JSON round-trip through `params_to_dict` / `params_from_dict`.

## Hardware profile (sim-to-real defaults)

`HardwareProfile` in `config/composite_ball_params.py` holds the motor, encoder, IMU and ball-sensor
noise, the pipeline latency and the loop rates. Every value can be changed in config and in the UI.

| item | default |
|------|---------|
| wheel drive | NEMA 17 17HS19-2004S1 @ 24 V, **direct drive**, 0.44 N·m pull-out torque, 18.85 rad/s (180 RPM), velocity (step-rate) servo `kv = 0.6`, pull-out torque falls as 1/ω above 300 RPM |
| wheel joint armature | 0.0008 kg·m² (the prototype spec; see note) |
| encoder | 1200 CPR on the output shaft, quantised |
| IMU | gyro white noise 0.005 rad/s, bias offset 0.01 rad/s plus a random walk, orientation noise 0.003 rad, accelerometer noise 0.03 m/s² |
| loop | physics 500 Hz (dt = 2 ms), control 100 Hz (`frame_skip = 5`), 10 ms pipeline delay (adjustable 0–40 ms) |

*Why direct drive:* with a 30:1 gearbox, 44 N·cm at the wheel would mean only 1.5 N·cm at the motor,
and 180 RPM at the wheel would mean 5,400 RPM at the motor. A NEMA 17 can't do either. Driving the
wheel directly gives both the 0.44 N·m and the 180 RPM wheel targets at 24 V.
*Why the 0.0008 kg·m² armature:* the bare rotor is 8.2e-6 kg·m². That inertia, plus a light
omniwheel, under a stiff step-rate servo at dt = 2 ms injects energy into the wheel–ball coupling,
because MuJoCo solves contacts with the explicit mass matrix while the servo gain is integrated
implicitly. 0.0008 kg·m² is about 10% of the ball load reflected to each wheel, removes that
artefact, and leaves the balance dynamics practically unchanged.

## Module 3: classical controllers (`controllers/classical.py`)

* **Mixers:** `KiwiMixer` implements the 120° matrix `ω = 1/r [−sinθ cosθ R] [vx vy ωz]ᵀ`.
  `BallWheelMixer` implements `ω_w = J (Ω_ball − Ω_head)` and `τ_w = J⁻ᵀ T`. `WheelCommandMapper`
  converts these to actuator `ctrl` values for velocity, torque or position actuators.
* **Reduced balance state** (`BallBotModel`), in the head's yaw frame:
  `[roll, pitch, roll_rate, pitch_rate, ex, ey, vx, vy, yaw_rate]`. Ball surface rotation and wheel
  angles are left out. `lift()` builds a full MuJoCo state that obeys the contact constraints: the ball
  rolls without slipping, the head rotates about the ball centre, and wheel speeds match through J.
* **Linearisation through `SceneStepper`:** `linearize_reduced` takes central finite differences of
  the 100 Hz map in reduced coordinates, saving and restoring state with
  `mj_getState`/`mj_setState(mjSTATE_INTEGRATION)`, so the aligned omniwheel friction and the magnet
  are part of (A, B). The delay's physics-step remainder enters as a separate `B_old`. `linearize_full`
  works on the full tangent state and matches `mujoco.mjd_transitionFD` to machine precision on plain
  scenes.
* **LQR:** `BallBotLQR.design(bundle, hardware)` solves the DARE on a model augmented for the delay
  (exact for any delay at physics-step resolution). It adds a rolling feedforward for velocity tracking
  and has separate default weights for velocity mode and torque mode. `FullStateLQR` handles the
  pendulums.
* **Cascaded PID:** the outer loop turns position and velocity error into a lean setpoint (≤ 0.12 rad).
  The inner loop runs PID on roll and pitch, with the derivative taken on the gyro. For the stepper
  servo the inner loop's output is integrated into a rate command, with anti-windup against saturation.
  Torque mode has its own default gains. `CartPolePID` handles Example 1.
* `BalanceTarget` covers station keeping, and velocity and yaw-rate tracking through a moving
  setpoint with acceleration-limited command ramps.
* `ClosedLoop` runs any controller with frame skip and the `DelayLine` pipeline delay.

Tested, all with the default 10 ms delay: LQR and PID station keeping through a 6–8 N push on BB-8
(both omniwheel modes, velocity and torque drives) and on the 1D and 2D balancers; velocity and yaw-rate
tracking; and cart-pole and spherical pendulum stabilisation.

**Example 4 change:** the `1d` variant is now a pitch-plane slice with two driven wheels at ±α. With a
single wheel on top, the head can pivot on that wheel's contact, which is a different, doubly unstable
system, so gains tuned on it wouldn't transfer to BB-8.

## Module 4: I/O bus and Gymnasium environment

**`models/io_bus.py`**
* `SensorBus` reads the scene's IMU (framequat, gyro, accelerometer), the 1200 CPR encoders and the
  ball channels. It adds gyro white noise, a per-episode bias plus a random walk, orientation and
  accelerometer noise, and quantises the encoders (wheel speed is then the count difference over the
  control period). `SensorBus.estimate()` turns a reading into the reduced balance state the
  controllers use.
* `ActuatorBus` applies the pipeline delay through a physics-step ring buffer, then saturation, then
  the NEMA 17 pull-out curve (each wheel's force range is updated from its speed).

**`envs/ball_balancer_env.py`**, registered as `BB8BallBalancer-v0`:

```python
from envs.ball_balancer_env import BallBalancerEnv, EnvConfig
env = BallBalancerEnv(EnvConfig(scene="bb8", task="velocity_tracking"))   # or planar_1d / planar_2d
obs, info = env.reset(seed=0)
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
info["reward_terms"]          # per-term reward breakdown
env.step_ctrl(ctrl)           # raw actuator ctrl, e.g. from CascadedPID / BallBotLQR
```

* Physics runs at 500 Hz through `SceneStepper` + `ActuatorBus`; the policy runs at 100 Hz.
  Observations come from `SensorBus`. Noise and latency switch on and off with
  `hardware.noise_enabled` / `latency_enabled`.
* **Observation:** roll, pitch, gyro (yaw frame), wheel speeds, ball velocity, velocity error,
  yaw-rate error, position error and the previous action.
* **Action:** with `body` (default) the policy commands the relative ball rotation about x, y and z,
  mixed through J. With `wheel` it commands each wheel directly.
* **Tasks:** `station_keeping` with random pushes (default 0.25 per second, 2–6 N for 50–120 ms), and
  `velocity_tracking` with resampled `(vx*, vy*, ωz*)` commands, ramped at an acceleration limit.
* **Reward** (`RewardWeights`): alive, tilt², velocity tracking `exp(−|v−v*|²/σ)`, position, yaw rate,
  angular-rate², ‖u‖², ‖u − u_prev‖², and a fall penalty.
* **Domain randomisation** is applied to the compiled model at every reset: epoxy shell mass ±15%,
  ball, floor and omniwheel friction, head CoM offset (±1 cm xy, ±1.5 cm z), head mass ±10%, and
  optionally the latency.
* Through `step_ctrl`, the cascaded PID and the LQR both balance BB-8 and the 1D slice for 12 s, with
  noise, 10 ms delay, domain randomisation and random pushes.

## Module 5: RL training (`training/train_rl.py`)

```bash
python -m training.train_rl --algo PPO --scene bb8 --task station_keeping --steps 2000000 --n-envs 8
python -m training.train_rl --algo SAC --scene planar_1d --vec-env dummy --n-envs 1 --steps 300000
```

```python
from training.train_rl import TrainingConfig, TrainingManager, PolicyRunner
mgr = TrainingManager(TrainingConfig(algo="PPO", total_timesteps=1_000_000))
mgr.hub.subscribe(lambda ev: print(ev))   # live events: started / episode / progress / train / checkpoint / finished
mgr.start()                               # background thread; mgr.stop() stops it cooperatively
policy = PolicyRunner.load(mgr.latest_onnx or mgr.latest_policy)   # obs -> action, for hot-swapping
```

* `SubprocVecEnv` uses `forkserver` (safe when started from a server thread). It falls back to
  `DummyVecEnv` for a single env. PPO uses `VecNormalize`.
* `MetricsHub` publishes every finished episode (return, length, fell, per-term reward sums), plus
  progress (throughput) and SB3 losses. The studio forwards these over its WebSocket.
* Each checkpoint writes `<tag>.zip`, `<tag>_vecnormalize.pkl` and `<tag>.onnx`. The ONNX graph
  contains the observation normalisation and the deterministic actor, and its actions match SB3's
  `predict` to within 1e-5.
* `evaluate()` scores either a policy or a classical controller on the same env.
* Measured here: about 900 env steps/s on 4 CPU cores, PPO updates included.

## Module 6: web studio (`app/main_studio.py` + `app/static`)

```bash
python -m app.main_studio                       # open http://localhost:8000
python -m app.main_studio --host 0.0.0.0 --port 8000 --runs runs   # reachable from other laptops on the LAN
```

MuJoCo runs on the server: 500 Hz physics and 100 Hz control in real time, with a 0.1–2× speed
control. The browser builds the Three.js scene from a descriptor of the compiled model, covering every
geom, the mesh vertex buffers and heightfields, so uploaded meshes, primitives, rollers and terrain all
appear automatically. Over `/ws` the browser receives body poses, contact-force and push arrows and the
telemetry at 30 Hz. Three.js 0.160 is bundled locally, so no CDN or internet is needed.

* **Scene:** all 5 examples, plus the omniwheel model (fast or sub-roller), the drive (stepper
  velocity or torque) and the variant. Toggles for showing the ball layers, collision geoms and force
  arrows, and for camera follow.
* **Composite ball:** edit R₀, layer thicknesses and densities, and surface μ to see a live per-layer
  mass and inertia table, then **Apply** to rebuild. The head unit (wheel count, α, mass, CoM, magnet,
  counterweight) and the terrain (slope, bumps) can be edited too, and an advanced JSON editor covers
  every parameter.
* **Controller:** Manual, Cascaded PID (live gain sliders), LQR (Q weights plus redesign, showing
  closed-loop ρ) or RL policy (pick any `.onnx`/`.zip` under `--runs` and hot-load it). WASD/QE drive
  the robot directly in manual mode and steer the balance target in PID, LQR and RL modes.
* **Disturbance:** ±X, ±Y or random pushes in newtons, with a set duration, on any body.
* **Sensors & latency:** noise, the pull-out curve, and a 0–40 ms delay.
* **Assets:** upload STL/OBJ meshes (trimesh mass properties, convex hulls), free or mounted on the
  head; add primitive props.
* **RL training:** start or stop PPO/SAC on the current scene and hardware; watch the progress bar and
  episode-return chart live; load the latest checkpoint into the 3D view.
* **Charts and logging:** head tilt, velocity versus target, wheel torques, and the reward breakdown.
  Record and download CSV or JSON logs.

REST endpoints (all JSON): `/api/scene`, `/api/status`, `/api/params`, `/api/ball/preview`,
`/api/ball/apply`, `/api/hardware`, `/api/controller`, `/api/teleop`, `/api/push`, `/api/sim`
(pause/speed/reset/record), `/api/step`, `/api/mesh` (multipart), `/api/primitive`,
`/api/extras/clear`, `/api/train/start|stop|status`, `/api/policies`, `/api/policy/load`,
`/api/log.csv`, `/api/log.json`.

**Results from the reference run:** 1M steps of PPO on BB-8 station keeping, with noise, 10 ms delay,
domain randomisation and the earlier, harsher pushes of 3–10 N. Episode length rose from about 90 to
about 1,500–1,860 of 2,000 steps. Over 10 randomised episodes the 500k checkpoint fell in 70% of them
and the 200k checkpoint in all of them. On the same harsh pushes, PID fell in 40% and LQR in 60%. With
the current default pushes (2–6 N), PID and LQR both had 0% falls. The PPO policy balances in the
studio, but it switches the wheel torques hard between their limits and drifts from its target. Treat
it as a working pipeline rather than a finished controller: train longer, or with action and
smoothness weights raised, before putting it on hardware.

## Deployment to the robot (`deploy/`)

See [`deploy/README.md`](deploy/README.md) for wiring, Jetson setup and the bring-up checklist.

```bash
python -m deploy.export_bundle --out bundle --controller pid                      # PC
python -m deploy.sim_io --bundle bundle --controller pid --push 6 --seconds 20    # PC: robot code vs MuJoCo
python3 -m deploy.runtime.runner run --bundle bundle --config robot.json          # Jetson Nano
```

The robot runtime reproduces the simulator's PID, LQR, observation and action mapping exactly; the
tests compare them to within 1e-12 (1e-5 for the network). It estimates the ball state from IMU and
wheel odometry, which the simulator now uses by default too. Tests cover Python 3.6 compatibility;
protocol interop with the C header; the firmware compiling on the host; the serial link over a pty
emulator (ramping, watchdog, spin test); software-in-the-loop balancing with pushes, mixed
motor/encoder signs and step-count odometry; teleop; and the safety faults (wrong motor sign,
telemetry loss, no arm request).

## Module 2: MJCF generator

| # | key | scene |
|---|-----|-------|
| 1 | `inverted_pendulum` | cart-pole, or a cart plus gimbal pendulum on an XY stage (`variant="spherical"`) |
| 2 | `single_omniwheel` | one omniwheel under a planar-constrained chassis (rolling vs lateral sliding) |
| 3 | `kiwi_drive` | 3 omniwheels at 120°. `meta.kinematics["mixing_matrix"]` is `1/r [−sinθ cosθ R]` |
| 4 | `planar_balancer` | `1d`: pitch-plane slice (two wheels at ±α, x–z plane). `2d`: 4 omniwheels, pitch + roll |
| 5 | `bb8` | composite ball + omniwheel head, magnet preload, counterweight, target marker |

```python
from models.mjcf_generator import build_example, SceneExtras, PrimitiveSpec, MeshImportSpec
from config.composite_ball_params import BB8Params, OmniwheelMode

p = BB8Params()
p.head.wheel.mode = OmniwheelMode.SUBROLLER   # or CONTACT_PAIR (default)
p.head.magnet.enabled = True
bundle = build_example("bb8", p, extras=SceneExtras(
    meshes=[MeshImportSpec("head_shell.stl", scale=0.001, parent="head", pos=(0, 0, 0.05))],
    primitives=[PrimitiveSpec("crate", size=(0.1, 0.1, 0.1), pos=(1, 0, 0.1), dynamic=True)]))
model, data, stepper = bundle.make()        # compiled, reset, ready
data.ctrl[:] = 0.0
stepper.push("head", force=[0, 15, 0], duration=0.1)   # disturbance tool
stepper.step(10)
bundle.write("build/bb8")                   # scene.xml + exported meshes
spec = bundle.to_spec()                     # mujoco.MjSpec for further editing
```

`bundle.meta` (`SceneMetadata`) lists every joint, actuator, sensor, IMU site and wheel (encoders,
torque sensors, axle and rolling directions), plus the kinematic matrices. Modules 3–6 use it, so they
never hard-code names.

### Omniwheel models

* **Sub-roller (high fidelity):** each hub has a driven hinge and `rollers_per_row × roller_rows`
  passive roller bodies (capsules or ellipsoids, staggered double rows) on tangential hinges.
* **Contact-pair (fast):** one sphere contact geom per wheel, plus `<pair friction="μ_t μ_lat spin roll roll">`.
  MuJoCo builds the contact tangent frame from the normal alone, so on its own the anisotropic friction
  would point in an arbitrary direction (see MuJoCo discussions #2517 and #2749). `SceneStepper` expands
  `mj_step` into its stages (`mj_fwdPosition` → realign each wheel contact frame so that tangent 1 is the
  rolling direction → `mj_makeConstraint`/`mj_projectConstraint` → velocity stage → `mj_step2`).

Both models are checked against the analytic kinematics in the tests, and both run at 500 Hz. In the
tests, sub-roller contacts show about 5% micro-slip. Measured during Module 2 (BB-8, 3 wheels):
sub-roller at dt = 1 ms cost about 0.05 s of wall time per simulated second, and contact-pair at
dt = 2 ms about 0.02 s, roughly a 2.5× speed-up. The Python-level frame alignment limits the gain,
so it is not the 10× the spec hoped for.

### Magnetic preload

An equal and opposite force pair acts between the head magnet site and the ball centre (or the
counterweight, when that is enabled), using a `constant` or `dipole` (`∝ d⁻⁴`) force law. By default it
is added to `qfrc_applied` right after the position stage of each step. `SceneStepper(...,
magnet_backend="passive_callback")` instead uses a real `mjcb_passive` hook. That hook is installed only
for the duration of `step()`, because the MuJoCo Python bindings cannot compile any model while a Python
passive callback is installed.

### Collision groups

| role | contype | conaffinity |
|------|---------|-------------|
| world / terrain | 1 | 1 |
| composite ball | 2 | 1 |
| ball rollers | 4 | 3 |
| ball pair-wheel sphere (plus an explicit pair with the ball) | 4 | 1 |
| ground rollers | 4 | 1 |
| chassis / head | 8 | 3 |
| props (default) | 1 | 1 |

### Mesh import

`MeshImporter` (trimesh) loads `.stl`, `.obj` and other formats, applies the unit scale, and computes
volume, centre of mass and the full inertia tensor. It falls back to the convex hull when a mesh isn't
watertight, and it regularises the tensor so MuJoCo accepts it. It then exports a visual mesh and a
convex hull (or a convex decomposition, when `vhacdx` is installed). A mesh can be static, free-floating,
or welded to a body such as `head`, in which case it adds its mass to that body.

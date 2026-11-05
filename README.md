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
│   └── runtime.py                 #   step-time helpers (contact alignment, magnet, pushes)
├── tests/
│   ├── test_composite_ball.py
│   └── test_mjcf_generator.py
├── controllers/classical.py       # Module 3: omni mixers, cascaded PID, FD-LQR      [done]
├── envs/ball_balancer_env.py      # Module 4: Gymnasium env + domain randomisation   [next]
├── training/train_rl.py           # Module 5: SB3 PPO/SAC manager                    [next]
└── app/main_studio.py             # Module 6: FastAPI + WebSockets + Three.js studio [next]
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

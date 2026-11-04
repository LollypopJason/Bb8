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
├── controllers/classical.py       # Module 3: omni mixer, cascaded PID, FD-LQR       [next]
├── envs/ball_balancer_env.py      # Module 4: Gymnasium env + domain randomisation   [next]
├── training/train_rl.py           # Module 5: SB3 PPO/SAC manager                    [next]
└── app/main_studio.py             # Module 6: FastAPI + WebSockets + Three.js studio [next]
```

## Install & run

```bash
pip install -r requirements.txt
python -m pytest -q                                   # 28 physics/integration tests
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

## Module 2: MJCF generator

| # | key | scene |
|---|-----|-------|
| 1 | `inverted_pendulum` | cart-pole, or a cart plus gimbal pendulum on an XY stage (`variant="spherical"`) |
| 2 | `single_omniwheel` | one omniwheel under a planar-constrained chassis (rolling vs lateral sliding) |
| 3 | `kiwi_drive` | 3 omniwheels at 120°. `meta.kinematics["mixing_matrix"]` is `1/r [−sinθ cosθ R]` |
| 4 | `planar_balancer` | `1d`: single wheel on the ball in the x–z plane. `2d`: 4 omniwheels, pitch + roll |
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

Both models are checked against the analytic kinematics in the tests. Measured on this machine (BB-8,
3 wheels): sub-roller at dt = 1 ms costs about 0.05 s of wall time per simulated second, and
contact-pair at dt = 2 ms about 0.02 s. That is roughly a 2.5× speed-up, much of it from the larger
stable timestep. The Python-level frame alignment limits the gain, so it is not the 10× the spec hoped
for.

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

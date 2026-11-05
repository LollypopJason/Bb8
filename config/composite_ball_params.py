"""Physical parameter dataclasses and calculators for the BB-8 ball-balancer studio.

This module is the single source of truth for every physical number that ends up
in a generated MJCF scene:

* :class:`CompositeBallParams` - the layered hollow sphere (air cavity, styrofoam
  base, epoxy + sawdust outer shell) with exact thick-walled-shell mass and
  inertia calculations.
* :class:`HeadUnitParams` - the omniwheel head unit that rides on top of the
  ball (geometry, motors, omniwheel model, magnetic preload / counterweight).
* Parameter sets for the progressive example scenes (pendulum, single
  omniwheel, kiwi drive, planar ball balancer) and global simulation settings
  (timestep, solver, terrain).

All quantities are SI (m, kg, s, N, rad) unless the field name says otherwise
(``*_deg`` fields are in degrees).

Thick-walled spherical shell between radii ``a < b`` with uniform density ``rho``::

    m = 4/3 * pi * rho * (b^3 - a^3)
    I = 2/5 * m * (b^5 - a^5) / (b^3 - a^3)

Concentric layers share the same centre, so total mass and inertia are plain
sums and the inertia tensor stays isotropic (``diag(I, I, I)``).
"""

from __future__ import annotations

import copy
import dataclasses
import enum
import json
import math
import typing
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "AIR_DENSITY",
    "ShellLayer",
    "LayerProperties",
    "ContactParams",
    "CompositeBallParams",
    "OmniwheelMode",
    "ControlMode",
    "RollerShape",
    "MotorParams",
    "nema17_direct_drive",
    "EncoderParams",
    "ImuNoiseParams",
    "BallSensorNoiseParams",
    "LatencyParams",
    "ControlLoopParams",
    "HardwareProfile",
    "OmniwheelParams",
    "MagneticPreloadParams",
    "CounterweightParams",
    "WheelGeometry",
    "HeadUnitParams",
    "BumpSpec",
    "HeightfieldSpec",
    "TerrainParams",
    "SimulationParams",
    "PendulumParams",
    "SingleOmniwheelParams",
    "KiwiDriveParams",
    "PlanarBalancerParams",
    "BB8Params",
    "thick_shell_mass",
    "thick_shell_inertia",
    "solid_cylinder_inertia",
    "solid_box_inertia",
    "params_to_dict",
    "params_from_dict",
    "save_params_json",
    "load_params_json",
]

AIR_DENSITY = 1.204  # kg/m^3 at 20 degC, sea level


# --------------------------------------------------------------------------- #
# Elementary rigid-body calculators
# --------------------------------------------------------------------------- #
def thick_shell_mass(density: float, r_inner: float, r_outer: float) -> float:
    """Mass of a uniform thick spherical shell between ``r_inner`` and ``r_outer``."""
    _check_radii(r_inner, r_outer)
    return 4.0 / 3.0 * math.pi * density * (r_outer**3 - r_inner**3)


def thick_shell_inertia(mass: float, r_inner: float, r_outer: float) -> float:
    """Moment of inertia (about any diameter) of a uniform thick spherical shell.

    ``I = 2/5 m (b^5 - a^5) / (b^3 - a^3)``. For a vanishing wall thickness the
    ratio is evaluated through its analytic limit ``5/3 b^2`` so the thin-shell
    result ``2/3 m b^2`` is recovered without division by zero.
    """
    _check_radii(r_inner, r_outer)
    a, b = float(r_inner), float(r_outer)
    if b <= 0.0:
        return 0.0
    rel = (b - a) / b
    if rel < 1e-9:
        ratio = 5.0 / 3.0 * b * b  # lim_{a->b} (b^5-a^5)/(b^3-a^3)
    else:
        ratio = (b**5 - a**5) / (b**3 - a**3)
    return 0.4 * mass * ratio


def solid_cylinder_inertia(mass: float, radius: float, length: float) -> Tuple[float, float, float]:
    """Principal inertia (Ixx, Iyy, Izz) of a solid cylinder whose axis is z."""
    i_axial = 0.5 * mass * radius**2
    i_trans = mass * (3.0 * radius**2 + length**2) / 12.0
    return (i_trans, i_trans, i_axial)


def solid_box_inertia(mass: float, sx: float, sy: float, sz: float) -> Tuple[float, float, float]:
    """Principal inertia of a solid box with full side lengths ``sx, sy, sz``."""
    return (
        mass * (sy**2 + sz**2) / 12.0,
        mass * (sx**2 + sz**2) / 12.0,
        mass * (sx**2 + sy**2) / 12.0,
    )


def _check_radii(r_inner: float, r_outer: float) -> None:
    if r_inner < 0.0:
        raise ValueError(f"inner radius must be >= 0, got {r_inner}")
    if r_outer < r_inner:
        raise ValueError(f"outer radius ({r_outer}) must be >= inner radius ({r_inner})")


def _fmt(v: float) -> str:
    """Compact, lossless-enough float formatting for MJCF attributes."""
    if v == 0.0:
        return "0"
    return f"{v:.9g}"


def _vec(values: Sequence[float]) -> str:
    return " ".join(_fmt(float(v)) for v in values)


# --------------------------------------------------------------------------- #
# Composite ball
# --------------------------------------------------------------------------- #
@dataclass
class ShellLayer:
    """One concentric layer of the composite ball."""

    name: str
    inner_radius: float
    outer_radius: float
    density: float
    rgba: Tuple[float, float, float, float] = (0.8, 0.8, 0.8, 1.0)

    @property
    def thickness(self) -> float:
        return self.outer_radius - self.inner_radius

    @property
    def volume(self) -> float:
        return 4.0 / 3.0 * math.pi * (self.outer_radius**3 - self.inner_radius**3)

    @property
    def mass(self) -> float:
        return thick_shell_mass(self.density, self.inner_radius, self.outer_radius)

    @property
    def inertia(self) -> float:
        return thick_shell_inertia(self.mass, self.inner_radius, self.outer_radius)

    def properties(self) -> "LayerProperties":
        return LayerProperties(
            name=self.name,
            inner_radius=self.inner_radius,
            outer_radius=self.outer_radius,
            thickness=self.thickness,
            density=self.density,
            volume=self.volume,
            mass=self.mass,
            inertia=self.inertia,
        )


@dataclass(frozen=True)
class LayerProperties:
    """Computed (read-only) properties of a single layer."""

    name: str
    inner_radius: float
    outer_radius: float
    thickness: float
    density: float
    volume: float
    mass: float
    inertia: float

    def as_dict(self) -> Dict[str, float]:
        return dataclasses.asdict(self)


@dataclass
class ContactParams:
    """MuJoCo contact parameters for a surface.

    ``friction`` is the MuJoCo geom triplet ``(sliding, torsional, rolling)``.
    ``solref = (timeconst, dampratio)`` and ``solimp = (dmin, dmax, width,
    midpoint, power)`` follow the MuJoCo solver-parameter conventions.
    """

    sliding_friction: float = 0.9
    torsional_friction: float = 0.005
    rolling_friction: float = 0.0002
    solref: Tuple[float, float] = (0.004, 1.0)
    solimp: Tuple[float, float, float, float, float] = (0.95, 0.99, 0.001, 0.5, 2.0)
    condim: int = 6
    margin: float = 0.0

    def validate(self) -> None:
        if self.condim not in (1, 3, 4, 6):
            raise ValueError(f"condim must be one of 1,3,4,6 (got {self.condim})")
        if min(self.sliding_friction, self.torsional_friction, self.rolling_friction) < 0:
            raise ValueError("friction coefficients must be non-negative")
        if self.solref[0] <= 0 or self.solref[1] <= 0:
            raise ValueError("solref (timeconst, dampratio) must be positive")
        dmin, dmax, width, mid, power = self.solimp
        if not (0.0 < dmin <= dmax < 1.0):
            raise ValueError("solimp requires 0 < dmin <= dmax < 1")
        if width <= 0 or not (0.0 < mid < 1.0) or power < 1.0:
            raise ValueError("solimp requires width > 0, 0 < midpoint < 1, power >= 1")

    @property
    def friction(self) -> Tuple[float, float, float]:
        return (self.sliding_friction, self.torsional_friction, self.rolling_friction)

    def mjcf_attrs(self) -> Dict[str, str]:
        return {
            "friction": _vec(self.friction),
            "solref": _vec(self.solref),
            "solimp": _vec(self.solimp),
            "condim": str(int(self.condim)),
            "margin": _fmt(self.margin),
        }


@dataclass
class CompositeBallParams:
    """Layered hollow ball: air cavity / styrofoam base / epoxy + sawdust shell.

    Radii::

        R0 = inner_radius                         (air cavity)
        R1 = R0 + foam_thickness                  (styrofoam outer radius)
        R2 = R1 + epoxy_thickness                 (outer surface)

    Use :meth:`from_radii` if you measured radii instead of thicknesses.
    """

    inner_radius: float = 0.200          # R0 [m]
    foam_thickness: float = 0.030        # t_foam [m]
    foam_density: float = 35.0           # rho_foam [kg/m^3]
    epoxy_thickness: float = 0.004       # t_epoxy [m]
    epoxy_density: float = 950.0         # rho_epoxy+sawdust [kg/m^3]
    include_air_mass: bool = False       # add the (tiny) mass of the trapped air
    air_density: float = AIR_DENSITY
    surface: ContactParams = field(
        default_factory=lambda: ContactParams(
            sliding_friction=0.9,
            torsional_friction=0.005,
            rolling_friction=0.0002,
            solref=(0.004, 1.0),
            solimp=(0.95, 0.99, 0.001, 0.5, 2.0),
            condim=6,
        )
    )
    # Visual colours (RGBA) used by the MJCF generator.
    foam_rgba: Tuple[float, float, float, float] = (0.95, 0.95, 0.92, 1.0)
    epoxy_rgba: Tuple[float, float, float, float] = (0.93, 0.90, 0.84, 1.0)
    accent_rgba: Tuple[float, float, float, float] = (0.95, 0.45, 0.10, 1.0)

    # ----------------------------------------------------------------- ctors
    @classmethod
    def from_radii(
        cls,
        r0: float,
        r1: float,
        r2: float,
        foam_density: float = 35.0,
        epoxy_density: float = 950.0,
        **kwargs: Any,
    ) -> "CompositeBallParams":
        """Build from measured radii ``R0 <= R1 <= R2`` instead of thicknesses."""
        if not (0.0 <= r0 <= r1 <= r2):
            raise ValueError(f"radii must satisfy 0 <= R0 <= R1 <= R2 (got {r0}, {r1}, {r2})")
        return cls(
            inner_radius=r0,
            foam_thickness=r1 - r0,
            foam_density=foam_density,
            epoxy_thickness=r2 - r1,
            epoxy_density=epoxy_density,
            **kwargs,
        )

    # ----------------------------------------------------------------- radii
    @property
    def r0(self) -> float:
        return self.inner_radius

    @property
    def r1(self) -> float:
        return self.inner_radius + self.foam_thickness

    @property
    def r2(self) -> float:
        return self.r1 + self.epoxy_thickness

    @property
    def outer_radius(self) -> float:
        return self.r2

    # ----------------------------------------------------------------- layers
    def layers(self) -> List[ShellLayer]:
        """Concentric layers from the inside out (zero-thickness layers kept)."""
        out: List[ShellLayer] = []
        if self.include_air_mass and self.r0 > 0.0:
            out.append(ShellLayer("air_cavity", 0.0, self.r0, self.air_density, (0.7, 0.85, 1.0, 0.15)))
        out.append(ShellLayer("styrofoam", self.r0, self.r1, self.foam_density, self.foam_rgba))
        out.append(ShellLayer("epoxy_sawdust", self.r1, self.r2, self.epoxy_density, self.epoxy_rgba))
        return out

    def layer_properties(self) -> List[LayerProperties]:
        return [layer.properties() for layer in self.layers()]

    # ----------------------------------------------------------------- totals
    @property
    def mass(self) -> float:
        return float(sum(layer.mass for layer in self.layers()))

    @property
    def inertia(self) -> float:
        """Scalar principal moment of inertia about the ball centre."""
        return float(sum(layer.inertia for layer in self.layers()))

    @property
    def inertia_tensor(self) -> np.ndarray:
        return np.eye(3) * self.inertia

    @property
    def radius_of_gyration(self) -> float:
        return math.sqrt(self.inertia / self.mass) if self.mass > 0 else 0.0

    @property
    def inertia_ratio(self) -> float:
        """``I / (m R2^2)``: 0.4 for a solid sphere, 0.667 for a thin shell."""
        return self.inertia / (self.mass * self.r2**2) if self.mass > 0 else 0.0

    @property
    def effective_rolling_mass(self) -> float:
        """Translational mass + rotational inertia reflected to the contact: ``m + I/R^2``."""
        return self.mass + self.inertia / self.r2**2

    @property
    def displaced_volume(self) -> float:
        return 4.0 / 3.0 * math.pi * self.r2**3

    # ----------------------------------------------------------------- utils
    def validate(self) -> None:
        if self.inner_radius < 0:
            raise ValueError("inner_radius must be >= 0")
        if self.foam_thickness < 0 or self.epoxy_thickness < 0:
            raise ValueError("layer thicknesses must be >= 0")
        if self.r2 <= 0:
            raise ValueError("ball outer radius must be > 0")
        if self.foam_density < 0 or self.epoxy_density < 0 or self.air_density < 0:
            raise ValueError("densities must be >= 0")
        if self.mass <= 0:
            raise ValueError("ball must have positive mass (check densities / thicknesses)")
        self.surface.validate()

    def scaled(
        self,
        epoxy_mass_scale: float = 1.0,
        foam_mass_scale: float = 1.0,
        friction_scale: float = 1.0,
    ) -> "CompositeBallParams":
        """Copy with scaled layer densities / friction (domain randomisation).

        Geometry is unchanged, so scaling a density scales that layer's mass
        and inertia by exactly the same factor.
        """
        new = copy.deepcopy(self)
        new.epoxy_density *= float(epoxy_mass_scale)
        new.foam_density *= float(foam_mass_scale)
        new.surface.sliding_friction *= float(friction_scale)
        return new

    def mjcf_inertial_attrs(self) -> Dict[str, str]:
        """Attributes for ``<inertial>`` of ``<body name="composite_ball">``."""
        i = self.inertia
        return {"pos": "0 0 0", "mass": _fmt(self.mass), "diaginertia": _vec((i, i, i))}

    def mjcf_inertial_xml(self) -> str:
        a = self.mjcf_inertial_attrs()
        return f'<inertial pos="{a["pos"]}" mass="{a["mass"]}" diaginertia="{a["diaginertia"]}"/>'

    def summary(self) -> Dict[str, Any]:
        return {
            "radii_m": {"R0": self.r0, "R1": self.r1, "R2": self.r2},
            "layers": [p.as_dict() for p in self.layer_properties()],
            "total_mass_kg": self.mass,
            "inertia_kgm2": self.inertia,
            "inertia_ratio_I_over_mR2": self.inertia_ratio,
            "radius_of_gyration_m": self.radius_of_gyration,
            "effective_rolling_mass_kg": self.effective_rolling_mass,
        }

    def report(self) -> str:
        """Human-readable table of the per-layer and total properties."""
        lines = [
            "Composite ball",
            f"  R0 = {self.r0 * 1e3:8.2f} mm   R1 = {self.r1 * 1e3:8.2f} mm   R2 = {self.r2 * 1e3:8.2f} mm",
            f"  {'layer':<15}{'t [mm]':>9}{'rho [kg/m3]':>13}{'V [L]':>10}{'m [kg]':>10}{'I [kg m2]':>13}",
        ]
        for p in self.layer_properties():
            lines.append(
                f"  {p.name:<15}{p.thickness * 1e3:9.2f}{p.density:13.1f}{p.volume * 1e3:10.3f}"
                f"{p.mass:10.4f}{p.inertia:13.6f}"
            )
        lines.append(
            f"  {'TOTAL':<15}{(self.r2 - self.r0) * 1e3:9.2f}{'':>13}{self.displaced_volume * 1e3:10.3f}"
            f"{self.mass:10.4f}{self.inertia:13.6f}"
        )
        lines.append(
            f"  I/(m R2^2) = {self.inertia_ratio:.4f}  (solid sphere 0.400, thin shell 0.667)"
        )
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Head unit, motors, omniwheels
# --------------------------------------------------------------------------- #
class OmniwheelMode(str, enum.Enum):
    """How omniwheels are modelled in MuJoCo."""

    SUBROLLER = "subroller"        # hub hinge + N passive roller bodies (high fidelity)
    CONTACT_PAIR = "contact_pair"  # one sphere geom + anisotropic friction <pair> (fast)


class ControlMode(str, enum.Enum):
    """Actuator type generated for a driven joint."""

    TORQUE = "motor"
    VELOCITY = "velocity"
    POSITION = "position"


class RollerShape(str, enum.Enum):
    CAPSULE = "capsule"
    ELLIPSOID = "ellipsoid"


@dataclass
class MotorParams:
    """Wheel drive model. Values are on the *motor* side of the gearbox.

    Default: a standard **NEMA 17 stepper (17HS19-2004S1, 2.0 A, 0.59 N m
    holding torque, rotor 82 g cm^2) at 24 V in direct drive**, limited to the
    prototype's wheel-side targets of 0.44 N m peak torque and 180 RPM
    (18.85 rad/s). A stepper is commanded by step rate, so the default actuator
    is a stiff velocity servo whose force range follows the pull-out torque
    curve (flat up to ``pullout_corner_speed``, then ``~1/omega``); if the load
    exceeds it the wheel falls behind, which approximates missed steps.

    ``wheel torque = gear_ratio * ctrl`` for torque mode. The reflected rotor
    inertia ``rotor_inertia * gear_ratio^2`` becomes the joint ``armature``.
    """

    control_mode: ControlMode = ControlMode.VELOCITY
    torque_limit: float = 0.44           # peak (pull-out) motor-shaft torque [N m]
    gear_ratio: float = 1.0              # direct drive
    rotor_inertia: float = 8.2e-6        # [kg m^2] motor-side (17HS19-2004S1 datasheet)
    # Joint armature used in simulation. ``None`` -> rotor_inertia * gear^2.
    # Default 0.0008 kg m^2 (prototype spec): the bare direct-drive rotor (8.2e-6)
    # plus a ~25 g cm^2 wheel is too light for a stiff stepper servo at a 2 ms
    # timestep (MuJoCo solves contacts with the explicit mass matrix, so the
    # implicitly-integrated servo then injects energy into the wheel/ball
    # coupling). 0.0008 is ~10 % of the ball load reflected to each wheel.
    armature_override: Optional[float] = 0.0008
    max_wheel_speed: float = 18.85       # [rad/s] wheel side (180 RPM, velocity ctrlrange)
    pullout_corner_speed: float = 31.4   # [rad/s] motor side: torque starts to drop (~300 RPM @ 24 V)
    joint_damping: float = 0.0005        # wheel joint viscous damping [N m s/rad]
    frictionloss: float = 0.0            # wheel joint Coulomb friction [N m]
    kv: float = 0.6                      # velocity-servo stiffness [N m s/rad] (wheel side)
    kp: float = 5.0                      # position-actuator gain [N m/rad] (wheel side)
    position_kv: float = 0.1             # damping for position actuator [N m s/rad]
    motor_model: str = "NEMA17 17HS19-2004S1 @ 24 V, direct drive"

    @property
    def wheel_torque_limit(self) -> float:
        return self.torque_limit * self.gear_ratio

    @property
    def armature(self) -> float:
        if self.armature_override is not None:
            return float(self.armature_override)
        return self.rotor_inertia * self.gear_ratio**2

    def pullout_wheel_torque(self, wheel_speed: Any) -> Any:
        """Available wheel-side torque at ``wheel_speed`` [rad/s] (works on arrays)."""
        w_motor = np.abs(np.asarray(wheel_speed, dtype=float)) * self.gear_ratio
        corner = max(self.pullout_corner_speed, 1e-9)
        scale = np.where(w_motor <= corner, 1.0, corner / np.maximum(w_motor, 1e-9))
        out = self.wheel_torque_limit * scale
        return float(out) if np.ndim(out) == 0 else out

    def validate(self) -> None:
        if self.torque_limit <= 0 or self.gear_ratio <= 0:
            raise ValueError("torque_limit and gear_ratio must be positive")
        if self.rotor_inertia < 0 or self.joint_damping < 0 or self.frictionloss < 0:
            raise ValueError("rotor_inertia, joint_damping and frictionloss must be >= 0")
        if self.max_wheel_speed <= 0 or self.pullout_corner_speed <= 0:
            raise ValueError("max_wheel_speed and pullout_corner_speed must be positive")
        if self.kv <= 0 or self.kp <= 0:
            raise ValueError("servo gains must be positive")
        if self.armature_override is not None and self.armature_override < 0:
            raise ValueError("armature_override must be >= 0")


def nema17_direct_drive(**overrides: Any) -> MotorParams:
    """The default stepper drive with optional field overrides."""
    return dataclasses.replace(MotorParams(), **overrides)


@dataclass
class EncoderParams:
    """Wheel encoder (counts per *output shaft* revolution)."""

    cpr: int = 1200
    quantize: bool = True

    @property
    def resolution(self) -> float:
        return 2.0 * math.pi / self.cpr

    def validate(self) -> None:
        if self.cpr < 4:
            raise ValueError("encoder cpr must be >= 4")


@dataclass
class ImuNoiseParams:
    """Head IMU noise (BNO085 / MPU6050 class)."""

    gyro_noise_std: float = 0.005            # white noise [rad/s]
    gyro_bias_std: float = 0.01              # per-episode constant offset std [rad/s]
    gyro_bias_walk: float = 0.001            # bias random walk [rad/s / sqrt(s)]
    orientation_noise_std: float = 0.003     # roll / pitch / yaw estimate noise [rad]
    accel_noise_std: float = 0.03            # [m/s^2]

    def validate(self) -> None:
        if min(self.gyro_noise_std, self.gyro_bias_std, self.gyro_bias_walk, self.orientation_noise_std,
               self.accel_noise_std) < 0:
            raise ValueError("IMU noise parameters must be >= 0")


@dataclass
class BallSensorNoiseParams:
    """Noise on the ball ground-truth / odometry channels."""

    position_std: float = 0.003              # [m]
    velocity_std: float = 0.01               # [m/s]
    angular_velocity_std: float = 0.02       # [rad/s]

    def validate(self) -> None:
        if min(self.position_std, self.velocity_std, self.angular_velocity_std) < 0:
            raise ValueError("ball sensor noise must be >= 0")


@dataclass
class LatencyParams:
    """Sensor -> controller -> actuator pipeline delay (ring buffer, physics-step resolution)."""

    delay: float = 0.010                     # [s], allowed range 0 .. 0.040

    def validate(self) -> None:
        if not (0.0 <= self.delay <= 0.040 + 1e-12):
            raise ValueError("pipeline delay must be within [0, 40] ms")


@dataclass
class ControlLoopParams:
    physics_timestep: float = 0.002          # 500 Hz
    control_rate_hz: float = 100.0           # policy / controller rate

    @property
    def control_dt(self) -> float:
        return 1.0 / self.control_rate_hz

    @property
    def frame_skip(self) -> int:
        return max(1, int(round(self.control_dt / self.physics_timestep)))

    def validate(self) -> None:
        if self.physics_timestep <= 0 or self.control_rate_hz <= 0:
            raise ValueError("timestep and control rate must be positive")
        if self.control_dt < self.physics_timestep - 1e-12:
            raise ValueError("control period must be at least one physics step")


@dataclass
class HardwareProfile:
    """Sim-to-real hardware model shared by the env, controllers and studio."""

    motor: MotorParams = field(default_factory=MotorParams)
    encoder: EncoderParams = field(default_factory=EncoderParams)
    imu: ImuNoiseParams = field(default_factory=ImuNoiseParams)
    ball_sensors: BallSensorNoiseParams = field(default_factory=BallSensorNoiseParams)
    latency: LatencyParams = field(default_factory=LatencyParams)
    loop: ControlLoopParams = field(default_factory=ControlLoopParams)
    noise_enabled: bool = True
    latency_enabled: bool = True
    torque_speed_curve: bool = True
    # "odometry": ball velocity / position from IMU + wheel encoders, exactly like the robot
    # (deploy.runtime.estimator); "sensor": noisy ground-truth ball channels.
    ball_state_source: str = "odometry"
    # low-pass time constant on the odometry ball velocity [s]: 1200 CPR at 100 Hz quantises
    # wheel speed to ~0.5 rad/s, which the LQR velocity gains would otherwise amplify
    odometry_velocity_tau: float = 0.04

    @property
    def delay_steps(self) -> int:
        """Pipeline delay in physics steps (0 when latency is disabled)."""
        if not self.latency_enabled:
            return 0
        return int(round(self.latency.delay / self.loop.physics_timestep))

    def validate(self) -> None:
        for part in (self.motor, self.encoder, self.imu, self.ball_sensors, self.latency, self.loop):
            part.validate()
        if self.ball_state_source not in ("odometry", "sensor"):
            raise ValueError("ball_state_source must be 'odometry' or 'sensor'")
        if not (0.0 <= self.odometry_velocity_tau <= 0.5):
            raise ValueError("odometry_velocity_tau must be within [0, 0.5] s")


@dataclass
class OmniwheelParams:
    """Geometry and contact model of one omniwheel."""

    mode: OmniwheelMode = OmniwheelMode.CONTACT_PAIR
    radius: float = 0.030
    width: float = 0.022
    hub_mass: float = 0.060
    # --- sub-roller mode
    rollers_per_row: int = 8
    roller_rows: int = 2
    roller_radius_ratio: float = 0.28    # roller radius / wheel radius
    roller_fill: float = 0.92            # fraction of the arc covered by each roller
    roller_mass: float = 0.003
    roller_shape: RollerShape = RollerShape.CAPSULE
    roller_damping: float = 2.0e-6
    roller_armature: float = 1.0e-7
    # --- contact-pair mode (also used as roller surface friction)
    mu_tangent: float = 1.2              # traction along rolling direction
    mu_lateral: float = 0.005            # sliding along the axle (free rollers)
    mu_torsional: float = 0.001
    mu_rolling: float = 0.0001
    solref: Tuple[float, float] = (0.004, 1.0)
    solimp: Tuple[float, float, float, float, float] = (0.95, 0.99, 0.001, 0.5, 2.0)
    rgba_hub: Tuple[float, float, float, float] = (0.15, 0.15, 0.17, 1.0)
    rgba_roller: Tuple[float, float, float, float] = (0.85, 0.15, 0.10, 1.0)

    @property
    def roller_radius(self) -> float:
        return self.radius * self.roller_radius_ratio

    @property
    def roller_center_radius(self) -> float:
        """Distance from hub axis to roller axis (roller surface touches ``radius``)."""
        return self.radius - self.roller_radius

    @property
    def roller_half_length(self) -> float:
        """Half-length of the roller's straight section / ellipsoid semi-axis."""
        arc = 2.0 * math.pi * self.roller_center_radius / self.rollers_per_row
        return 0.5 * arc * self.roller_fill

    @property
    def n_rollers(self) -> int:
        return self.rollers_per_row * self.roller_rows

    @property
    def total_mass(self) -> float:
        rollers = self.n_rollers * self.roller_mass if self.mode == OmniwheelMode.SUBROLLER else 0.0
        return self.hub_mass + rollers

    @property
    def pair_friction(self) -> Tuple[float, float, float, float, float]:
        """5-vector for ``<pair friction>``: tangent1 (rolling), tangent2 (axle), spin, roll, roll."""
        return (self.mu_tangent, self.mu_lateral, self.mu_torsional, self.mu_rolling, self.mu_rolling)

    def validate(self) -> None:
        if self.radius <= 0 or self.width <= 0 or self.hub_mass <= 0:
            raise ValueError("omniwheel radius, width and hub_mass must be positive")
        if self.rollers_per_row < 3 or self.roller_rows not in (1, 2):
            raise ValueError("need >= 3 rollers per row and 1 or 2 roller rows")
        if not (0.05 <= self.roller_radius_ratio <= 0.5):
            raise ValueError("roller_radius_ratio must be in [0.05, 0.5]")
        if not (0.2 <= self.roller_fill <= 1.0):
            raise ValueError("roller_fill must be in [0.2, 1.0]")
        if self.roller_mass <= 0:
            raise ValueError("roller_mass must be positive")
        if min(self.mu_tangent, self.mu_lateral, self.mu_torsional, self.mu_rolling) < 0:
            raise ValueError("friction coefficients must be >= 0")


@dataclass
class MagneticPreloadParams:
    """Magnetic attraction from a head magnet toward the ball centre / counterweight.

    Applied at runtime through a ``mjcb_passive`` callback as an equal and
    opposite force pair (head <-> ball or counterweight), so it adds normal
    load on the wheels without injecting net momentum.

    * ``constant``: ``|F| = force``
    * ``dipole``: ``|F| = force * (d_nominal / d)^4`` clipped to ``max_force``
    """

    enabled: bool = False
    force: float = 15.0                  # nominal attraction [N]
    model: str = "constant"              # "constant" | "dipole"
    max_force: float = 60.0              # clip for the dipole model [N]
    magnet_height: float = 0.010         # magnet height above the ball top in head frame [m]

    def validate(self) -> None:
        if self.force < 0 or self.max_force < 0:
            raise ValueError("magnetic forces must be >= 0")
        if self.model not in ("constant", "dipole"):
            raise ValueError("magnet model must be 'constant' or 'dipole'")


@dataclass
class CounterweightParams:
    """Optional pendulum mass inside the ball, hung from the ball centre on a ball joint."""

    enabled: bool = False
    mass: float = 1.0
    offset: float = 0.12                 # distance below the ball centre [m]
    radius: float = 0.04                 # visual radius [m]
    joint_damping: float = 0.02          # ball-joint damping coupling it to the shell

    def validate(self) -> None:
        if self.mass <= 0 or self.offset < 0 or self.radius <= 0:
            raise ValueError("counterweight mass/radius must be > 0 and offset >= 0")


@dataclass(frozen=True)
class WheelGeometry:
    """Placement of one omniwheel relative to the head-unit origin (ball top).

    Frames (all expressed in the head frame, which is aligned with the world
    when the head is upright):

    * ``normal`` - outward ball normal at the contact point.
    * ``rolling_dir`` - wheel rolling direction at the contact (azimuthal).
    * ``axle`` - wheel axis, ``normal x rolling_dir``. A positive wheel speed
      about ``axle`` moves the wheel rim at the contact along ``-rolling_dir``.
    * ``hub_rotation`` - columns ``[normal, rolling_dir, axle]`` (hub body frame;
      the drive hinge is the local z axis).
    """

    index: int
    azimuth: float                       # theta_i [rad]
    zenith: float                        # alpha [rad]
    normal: Tuple[float, float, float]
    rolling_dir: Tuple[float, float, float]
    axle: Tuple[float, float, float]
    contact_point: Tuple[float, float, float]   # relative to head origin
    center: Tuple[float, float, float]          # wheel centre relative to head origin
    hub_rotation: Tuple[Tuple[float, float, float], ...]

    @property
    def hub_quat(self) -> Tuple[float, float, float, float]:
        return rotation_matrix_to_quat(np.array(self.hub_rotation))

    @property
    def contact_frame_quat(self) -> Tuple[float, float, float, float]:
        """Site frame with x=rolling_dir, y=axle, z=normal (used for contact alignment)."""
        mat = np.column_stack([self.rolling_dir, self.axle, self.normal])
        return rotation_matrix_to_quat(mat)


def rotation_matrix_to_quat(mat: np.ndarray) -> Tuple[float, float, float, float]:
    """Proper rotation matrix (columns = body axes in parent) to unit quaternion (w, x, y, z)."""
    m = np.asarray(mat, dtype=float)
    if m.shape != (3, 3):
        raise ValueError("rotation matrix must be 3x3")
    if abs(np.linalg.det(m) - 1.0) > 1e-6:
        raise ValueError("matrix is not a proper rotation (det != 1)")
    tr = np.trace(m)
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2.0
        w, x, y, z = 0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        w, x, y, z = (m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        w, x, y, z = (m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        w, x, y, z = (m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s
    q = np.array([w, x, y, z])
    q /= np.linalg.norm(q)
    if q[0] < 0:
        q = -q
    return tuple(float(v) for v in q)


def wheel_frame(normal: Sequence[float], rolling_dir: Sequence[float]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Orthonormalise ``(normal, rolling_dir)`` and return ``(n, t, axle = n x t)``."""
    n = np.asarray(normal, dtype=float)
    n = n / np.linalg.norm(n)
    t = np.asarray(rolling_dir, dtype=float)
    t = t - n * float(t @ n)
    norm = np.linalg.norm(t)
    if norm < 1e-9:
        raise ValueError("rolling direction must not be parallel to the contact normal")
    t = t / norm
    return n, t, np.cross(n, t)


@dataclass
class HeadUnitParams:
    """The omniwheel head unit riding on top of the composite ball.

    The head frame origin sits at the top of the ball (``z = 2 R2`` when the
    ball rests on flat ground) and the wheels touch the ball at zenith angle
    ``alpha`` from the vertical, equally spaced in azimuth.
    """

    chassis_mass: float = 1.4                    # chassis + electronics + battery [kg]
    com_height: float = 0.11                     # CoM above the head origin (ball top) [m]
    chassis_radius: float = 0.13                 # used for inertia approximation & visuals
    chassis_height: float = 0.12
    diag_inertia_override: Optional[Tuple[float, float, float]] = None
    num_wheels: int = 3
    zenith_angle_deg: float = 45.0               # alpha in [30, 60]
    azimuth_offset_deg: float = 90.0             # azimuth of wheel 0
    wheel: OmniwheelParams = field(default_factory=OmniwheelParams)
    motor: MotorParams = field(default_factory=MotorParams)
    magnet: MagneticPreloadParams = field(default_factory=MagneticPreloadParams)
    counterweight: CounterweightParams = field(default_factory=CounterweightParams)
    rgba_body: Tuple[float, float, float, float] = (0.95, 0.95, 0.93, 1.0)
    rgba_accent: Tuple[float, float, float, float] = (0.95, 0.45, 0.10, 1.0)
    rgba_dark: Tuple[float, float, float, float] = (0.12, 0.12, 0.14, 1.0)

    @property
    def zenith_angle(self) -> float:
        return math.radians(self.zenith_angle_deg)

    @property
    def wheel_azimuths(self) -> List[float]:
        off = math.radians(self.azimuth_offset_deg)
        return [off + 2.0 * math.pi * i / self.num_wheels for i in range(self.num_wheels)]

    @property
    def chassis_inertia(self) -> Tuple[float, float, float]:
        if self.diag_inertia_override is not None:
            return tuple(float(v) for v in self.diag_inertia_override)  # type: ignore[return-value]
        return solid_cylinder_inertia(self.chassis_mass, self.chassis_radius, self.chassis_height)

    @property
    def total_mass(self) -> float:
        return self.chassis_mass + self.num_wheels * self.wheel.total_mass

    def validate(self) -> None:
        if self.chassis_mass <= 0:
            raise ValueError("chassis_mass must be positive")
        if self.num_wheels not in (3, 4):
            raise ValueError("num_wheels must be 3 or 4")
        if not (30.0 <= self.zenith_angle_deg <= 60.0):
            raise ValueError("zenith angle alpha must be within [30, 60] degrees")
        if self.chassis_radius <= 0 or self.chassis_height <= 0:
            raise ValueError("chassis dimensions must be positive")
        if self.diag_inertia_override is not None:
            a, b, c = self.diag_inertia_override
            if min(a, b, c) <= 0 or a + b < c or a + c < b or b + c < a:
                raise ValueError("diag_inertia_override must be positive and satisfy the triangle inequality")
        self.wheel.validate()
        self.motor.validate()
        self.magnet.validate()
        self.counterweight.validate()

    def wheel_geometry(self, ball_radius: float) -> List[WheelGeometry]:
        """Contact / centre / orientation of each wheel relative to the head origin."""
        alpha = self.zenith_angle
        r_w = self.wheel.radius
        out: List[WheelGeometry] = []
        for i, theta in enumerate(self.wheel_azimuths):
            n_raw = (math.sin(alpha) * math.cos(theta), math.sin(alpha) * math.sin(theta), math.cos(alpha))
            t_raw = (-math.sin(theta), math.cos(theta), 0.0)
            n, t, a = wheel_frame(n_raw, t_raw)
            top = np.array([0.0, 0.0, ball_radius])
            contact = ball_radius * n - top
            center = (ball_radius + r_w) * n - top
            rot = np.column_stack([n, t, a])
            out.append(
                WheelGeometry(
                    index=i,
                    azimuth=theta,
                    zenith=alpha,
                    normal=tuple(n),
                    rolling_dir=tuple(t),
                    axle=tuple(a),
                    contact_point=tuple(contact),
                    center=tuple(center),
                    hub_rotation=tuple(tuple(row) for row in rot),
                )
            )
        return out

    def ball_to_wheel_jacobian(self, ball_radius: float) -> np.ndarray:
        """Matrix ``J`` with ``omega_wheels = J @ Omega_ball_rel`` (no-slip along rolling dirs).

        ``Omega_ball_rel`` is the ball angular velocity relative to the head,
        expressed in the head frame. Derivation: no slip along ``t_i`` gives
        ``-r w_i = t_i . (Omega x R n_i)``, i.e. ``w_i = -(R/r) a_i . Omega``.
        """
        geos = self.wheel_geometry(ball_radius)
        scale = -ball_radius / self.wheel.radius
        return scale * np.array([g.axle for g in geos])

    def summary(self, ball_radius: float) -> Dict[str, Any]:
        return {
            "total_mass_kg": self.total_mass,
            "chassis_inertia": self.chassis_inertia,
            "wheel_torque_limit_Nm": self.motor.wheel_torque_limit,
            "armature_kgm2": self.motor.armature,
            "zenith_deg": self.zenith_angle_deg,
            "wheels": [
                {"azimuth_deg": math.degrees(g.azimuth), "center": g.center, "axle": g.axle}
                for g in self.wheel_geometry(ball_radius)
            ],
        }


# --------------------------------------------------------------------------- #
# Terrain & simulation
# --------------------------------------------------------------------------- #
@dataclass
class BumpSpec:
    """A static obstacle partially embedded in the floor."""

    type: str = "capsule"                         # capsule | sphere | box | cylinder
    pos: Tuple[float, float, float] = (1.0, 0.0, 0.0)
    size: Tuple[float, ...] = (0.015, 0.6)        # MuJoCo geom size convention
    euler_deg: Tuple[float, float, float] = (90.0, 0.0, 0.0)
    rgba: Tuple[float, float, float, float] = (0.45, 0.35, 0.25, 1.0)

    def validate(self) -> None:
        if self.type not in ("capsule", "sphere", "box", "cylinder"):
            raise ValueError(f"unsupported bump type {self.type!r}")


@dataclass
class HeightfieldSpec:
    """Procedural heightfield patch (ramp, sine waves, bumps or random terrain)."""

    name: str = "terrain"
    profile: str = "ramp"                         # ramp | sine | bumps | random
    pos: Tuple[float, float, float] = (2.0, 0.0, 0.0)
    half_size: Tuple[float, float] = (1.0, 1.0)   # x, y half extents [m]
    max_height: float = 0.08                      # [m]
    base_depth: float = 0.05                      # [m] (thickness below z=0)
    nrow: int = 64
    ncol: int = 64
    wavelength: float = 0.5                       # sine / bumps [m]
    seed: int = 0
    elevation: Optional[List[float]] = None       # explicit row-major data in [0, 1]

    def validate(self) -> None:
        if self.profile not in ("ramp", "sine", "bumps", "random"):
            raise ValueError(f"unsupported heightfield profile {self.profile!r}")
        if self.nrow < 2 or self.ncol < 2:
            raise ValueError("heightfield needs nrow, ncol >= 2")
        if self.max_height <= 0 or self.base_depth <= 0 or min(self.half_size) <= 0:
            raise ValueError("heightfield sizes must be positive")
        if self.elevation is not None and len(self.elevation) != self.nrow * self.ncol:
            raise ValueError("elevation must contain nrow * ncol values")

    def elevation_grid(self) -> np.ndarray:
        """Normalised (0..1) elevation grid of shape (nrow, ncol)."""
        if self.elevation is not None:
            grid = np.asarray(self.elevation, dtype=float).reshape(self.nrow, self.ncol)
        else:
            ys = np.linspace(-self.half_size[1], self.half_size[1], self.nrow)
            xs = np.linspace(-self.half_size[0], self.half_size[0], self.ncol)
            xx, yy = np.meshgrid(xs, ys)
            k = 2.0 * math.pi / max(self.wavelength, 1e-6)
            if self.profile == "ramp":
                grid = (xx - xx.min()) / (xx.max() - xx.min())
            elif self.profile == "sine":
                grid = 0.5 + 0.25 * np.sin(k * xx) + 0.25 * np.sin(k * yy)
            elif self.profile == "bumps":
                grid = np.clip(np.sin(k * xx) * np.sin(k * yy), 0.0, None)
            else:
                rng = np.random.default_rng(self.seed)
                raw = rng.random((self.nrow, self.ncol))
                kern = np.ones((3, 3)) / 9.0
                padded = np.pad(raw, 1, mode="edge")
                grid = sum(
                    kern[i, j] * padded[i : i + self.nrow, j : j + self.ncol] for i in range(3) for j in range(3)
                )
        lo, hi = float(grid.min()), float(grid.max())
        return (grid - lo) / (hi - lo) if hi > lo else np.zeros_like(grid)


@dataclass
class TerrainParams:
    floor_slope_deg: float = 0.0                  # tilt of the floor plane
    slope_axis: str = "y"                         # rotate floor about world x or y
    floor: ContactParams = field(
        default_factory=lambda: ContactParams(sliding_friction=1.0, torsional_friction=0.005,
                                              rolling_friction=0.0001, condim=3)
    )
    floor_half_size: float = 10.0
    bumps: List[BumpSpec] = field(default_factory=list)
    heightfields: List[HeightfieldSpec] = field(default_factory=list)

    def validate(self) -> None:
        if not (-30.0 <= self.floor_slope_deg <= 30.0):
            raise ValueError("floor slope must be within [-30, 30] degrees")
        if self.slope_axis not in ("x", "y"):
            raise ValueError("slope_axis must be 'x' or 'y'")
        self.floor.validate()
        for b in self.bumps:
            b.validate()
        for h in self.heightfields:
            h.validate()


@dataclass
class SimulationParams:
    timestep: float = 0.002
    integrator: str = "implicitfast"              # Euler | implicit | implicitfast | RK4
    cone: str = "elliptic"
    solver: str = "Newton"
    iterations: int = 100
    ls_iterations: int = 50
    noslip_iterations: int = 0
    impratio: float = 1.0
    gravity: Tuple[float, float, float] = (0.0, 0.0, -9.81)
    terrain: TerrainParams = field(default_factory=TerrainParams)
    offscreen_width: int = 1280
    offscreen_height: int = 720

    def validate(self) -> None:
        if not (1e-5 <= self.timestep <= 0.02):
            raise ValueError("timestep must be within [1e-5, 0.02] s")
        if self.integrator not in ("Euler", "implicit", "implicitfast", "RK4"):
            raise ValueError("unsupported integrator")
        if self.cone not in ("pyramidal", "elliptic"):
            raise ValueError("cone must be 'pyramidal' or 'elliptic'")
        if self.solver not in ("PGS", "CG", "Newton"):
            raise ValueError("solver must be PGS, CG or Newton")
        self.terrain.validate()


# --------------------------------------------------------------------------- #
# Example-scene parameters
# --------------------------------------------------------------------------- #
@dataclass
class PendulumParams:
    """Example 1: cart-pole (2D) or cart + gimbal pendulum on an XY stage (3D)."""

    variant: str = "cartpole"                     # cartpole | spherical
    cart_mass: float = 1.0
    cart_size: Tuple[float, float, float] = (0.10, 0.08, 0.04)   # half sizes
    rail_half_length: float = 1.5
    pole_mass: float = 0.2
    pole_length: float = 0.6
    pole_radius: float = 0.015
    tip_mass: float = 0.05
    force_limit: float = 20.0                     # [N]
    cart_damping: float = 0.1
    pole_damping: float = 0.002
    initial_angle_deg: float = 3.0

    def validate(self) -> None:
        if self.variant not in ("cartpole", "spherical"):
            raise ValueError("variant must be 'cartpole' or 'spherical'")
        if min(self.cart_mass, self.pole_mass, self.pole_length, self.pole_radius, self.force_limit) <= 0:
            raise ValueError("pendulum masses, sizes and force limit must be positive")


@dataclass
class SingleOmniwheelParams:
    """Example 2: one vertical omniwheel carrying a planar-constrained chassis."""

    chassis_mass: float = 1.5
    chassis_half_size: Tuple[float, float, float] = (0.06, 0.04, 0.03)
    free_yaw: bool = False                        # True: staggered roller rows can yaw the chassis
    wheel: OmniwheelParams = field(default_factory=lambda: OmniwheelParams(radius=0.04, width=0.03))
    motor: MotorParams = field(
        default_factory=MotorParams
    )

    def validate(self) -> None:
        if self.chassis_mass <= 0:
            raise ValueError("chassis_mass must be positive")
        self.wheel.validate()
        self.motor.validate()


@dataclass
class KiwiDriveParams:
    """Example 3: three omniwheels at 120 deg (holonomic kiwi drive)."""

    chassis_mass: float = 2.5
    chassis_radius: float = 0.15                  # R: centre-to-wheel distance [m]
    chassis_thickness: float = 0.03
    wheel_angles_deg: Tuple[float, float, float] = (90.0, 210.0, 330.0)
    wheel: OmniwheelParams = field(default_factory=lambda: OmniwheelParams(radius=0.035, width=0.025))
    motor: MotorParams = field(
        default_factory=MotorParams
    )

    @property
    def wheel_angles(self) -> List[float]:
        return [math.radians(a) for a in self.wheel_angles_deg]

    def mixing_matrix(self) -> np.ndarray:
        """``omega = M @ [vx, vy, wz]`` with ``M = 1/r [-sin th, cos th, R]`` (body frame)."""
        r = self.wheel.radius
        rows = [[-math.sin(t), math.cos(t), self.chassis_radius] for t in self.wheel_angles]
        return np.array(rows) / r

    def validate(self) -> None:
        if self.chassis_mass <= 0 or self.chassis_radius <= self.wheel.width:
            raise ValueError("chassis_mass must be > 0 and chassis_radius larger than the wheel width")
        if len(self.wheel_angles_deg) != 3:
            raise ValueError("kiwi drive needs exactly three wheel angles")
        self.wheel.validate()
        self.motor.validate()


@dataclass
class PlanarBalancerParams:
    """Example 4: planar-slice (1D) or decoupled 4-omniwheel (2D) balancer.

    ``1d`` keeps ball and head in the x-z plane with two driven wheels at
    +-``zenith_angle_deg`` (one "virtual wheel" in the pitch plane), so pitch
    gains tuned here transfer to the BB-8 inner loop.
    """

    variant: str = "1d"                           # "1d": x-z plane, "2d": pitch+roll, no yaw
    ball: CompositeBallParams = field(default_factory=CompositeBallParams)
    body_mass: float = 1.2
    com_height: float = 0.12
    body_inertia: Tuple[float, float, float] = (0.012, 0.012, 0.008)
    wheel_radius: float = 0.035
    wheel_width: float = 0.025
    wheel_mass: float = 0.08
    wheel_friction: float = 1.2
    zenith_angle_deg: float = 45.0                # wheel contact angle from vertical
    wheel: OmniwheelParams = field(default_factory=OmniwheelParams)  # 2d variant omniwheels
    motor: MotorParams = field(default_factory=MotorParams)
    initial_tilt_deg: float = 2.0

    def validate(self) -> None:
        if self.variant not in ("1d", "2d"):
            raise ValueError("variant must be '1d' or '2d'")
        if min(self.body_mass, self.com_height, self.wheel_radius, self.wheel_mass) <= 0:
            raise ValueError("body/wheel parameters must be positive")
        if not (30.0 <= self.zenith_angle_deg <= 60.0):
            raise ValueError("zenith angle must be within [30, 60] degrees")
        self.ball.validate()
        self.wheel.validate()
        self.motor.validate()


@dataclass
class BB8Params:
    """Example 5: full composite ball + omniwheel head unit (hero scene)."""

    ball: CompositeBallParams = field(default_factory=CompositeBallParams)
    head: HeadUnitParams = field(default_factory=HeadUnitParams)
    task_mode: str = "station_keeping"            # station_keeping | velocity_tracking
    initial_tilt_deg: Tuple[float, float] = (0.0, 0.0)   # head roll, pitch at reset
    target_marker: bool = True

    def validate(self) -> None:
        if self.task_mode not in ("station_keeping", "velocity_tracking"):
            raise ValueError("task_mode must be 'station_keeping' or 'velocity_tracking'")
        self.ball.validate()
        self.head.validate()


# --------------------------------------------------------------------------- #
# (De)serialisation helpers for the UI / config files
# --------------------------------------------------------------------------- #
def params_to_dict(obj: Any) -> Any:
    """Recursively convert parameter dataclasses (and enums / tuples) to JSON-safe data."""
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: params_to_dict(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    if isinstance(obj, enum.Enum):
        return obj.value
    if isinstance(obj, (list, tuple)):
        return [params_to_dict(v) for v in obj]
    if isinstance(obj, dict):
        return {k: params_to_dict(v) for k, v in obj.items()}
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def _coerce(tp: Any, value: Any) -> Any:
    if value is None:
        return None
    origin = typing.get_origin(tp)
    args = typing.get_args(tp)
    if origin is typing.Union:  # Optional[X]
        inner = [a for a in args if a is not type(None)]
        return _coerce(inner[0], value) if inner else value
    if isinstance(tp, type) and dataclasses.is_dataclass(tp):
        return params_from_dict(tp, value)
    if isinstance(tp, type) and issubclass(tp, enum.Enum):
        return tp(value)
    if origin in (tuple, typing.Tuple):
        if len(args) == 2 and args[1] is Ellipsis:
            return tuple(_coerce(args[0], v) for v in value)
        if args:
            return tuple(_coerce(a, v) for a, v in zip(args, value))
        return tuple(value)
    if origin in (list, typing.List):
        return [_coerce(args[0], v) for v in value] if args else list(value)
    if tp is float:
        return float(value)
    if tp is int:
        return int(value)
    if tp is bool:
        return bool(value)
    return value


def params_from_dict(cls: type, data: Dict[str, Any]) -> Any:
    """Inverse of :func:`params_to_dict`; unknown keys raise ``KeyError``."""
    hints = typing.get_type_hints(cls)
    names = {f.name for f in dataclasses.fields(cls)}
    unknown = set(data) - names
    if unknown:
        raise KeyError(f"unknown fields for {cls.__name__}: {sorted(unknown)}")
    kwargs = {k: _coerce(hints[k], v) for k, v in data.items()}
    return cls(**kwargs)


def save_params_json(obj: Any, path: str) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"type": type(obj).__name__, "params": params_to_dict(obj)}, fh, indent=2)


def load_params_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as fh:
        payload = json.load(fh)
    registry = {c.__name__: c for c in (
        CompositeBallParams, HeadUnitParams, SimulationParams, HardwareProfile, PendulumParams, SingleOmniwheelParams,
        KiwiDriveParams, PlanarBalancerParams, BB8Params,
    )}
    cls = registry[payload["type"]]
    return params_from_dict(cls, payload["params"])


if __name__ == "__main__":  # pragma: no cover - manual inspection helper
    ball = CompositeBallParams()
    ball.validate()
    print(ball.report())
    print("MJCF:", ball.mjcf_inertial_xml())
    head = HeadUnitParams()
    head.validate()
    print(json.dumps(head.summary(ball.r2), indent=2, default=float))

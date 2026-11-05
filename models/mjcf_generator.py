"""Programmatic MJCF generator for the BB-8 ball-balancer studio.

Every scene is produced from the parameter dataclasses in
:mod:`config.composite_ball_params` and returned as a :class:`SceneBundle`
(XML text + in-memory mesh assets + :class:`SceneMetadata` describing the
named joints, actuators, sensors and wheel kinematics for the controller, env
and GUI layers).

Scenes
------
1. ``inverted_pendulum``   - cart-pole (2D) or cart + gimbal pendulum on an XY stage (3D)
2. ``single_omniwheel``    - one omniwheel carrying a planar-constrained chassis
3. ``kiwi_drive``          - 3-wheel holonomic omni-drive robot
4. ``planar_balancer``     - 1D single-wheel ball balancer / 2D decoupled 4-omniwheel balancer
5. ``bb8``                 - composite ball + omniwheel head unit (hero scene)

Omniwheels support two models (:class:`~config.composite_ball_params.OmniwheelMode`):

* ``subroller``: driven hub hinge + ``N`` passive roller bodies with tangential hinges.
* ``contact_pair``: one spherical contact geom per wheel with an explicit
  ``<pair>`` carrying anisotropic friction ``(mu_tangent, mu_lateral, ...)``.
  The tangent frame of those contacts is aligned with the wheel at runtime
  by :class:`models.runtime.SceneStepper` (MuJoCo otherwise picks an arbitrary
  tangent frame), so stepping a contact-pair scene must go through the stepper.

Any scene can be extended with procedural primitives, terrain (slope, bumps,
heightfields) and imported ``.stl`` / ``.obj`` meshes (:class:`MeshImporter`).

Command line::

    python -m models.mjcf_generator --example bb8 --omni-mode subroller --out build/bb8 --check 2.0
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import io
import math
import os
import re
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import mujoco
import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:  # allow ``python models/mjcf_generator.py``
    sys.path.insert(0, _ROOT)

from config.composite_ball_params import (  # noqa: E402
    BB8Params,
    CompositeBallParams,
    ContactParams,
    ControlMode,
    HeadUnitParams,
    HeightfieldSpec,
    KiwiDriveParams,
    MotorParams,
    OmniwheelMode,
    OmniwheelParams,
    PendulumParams,
    PlanarBalancerParams,
    RollerShape,
    SimulationParams,
    SingleOmniwheelParams,
    TerrainParams,
    params_to_dict,
    rotation_matrix_to_quat,
    solid_box_inertia,
    solid_cylinder_inertia,
    wheel_frame,
)
from models.runtime import AlignedContact, MagnetLink, SceneStepper  # noqa: E402

__all__ = [
    "CollisionBits",
    "PrimitiveSpec",
    "MeshImportSpec",
    "ImportedMesh",
    "MeshImporter",
    "SceneExtras",
    "WheelMeta",
    "SceneMetadata",
    "SceneBundle",
    "MJCFBuilder",
    "build_inverted_pendulum",
    "build_single_omniwheel",
    "build_kiwi_drive",
    "build_planar_balancer",
    "build_bb8",
    "EXAMPLES",
    "build_example",
    "list_examples",
]


# --------------------------------------------------------------------------- #
# Formatting helpers
# --------------------------------------------------------------------------- #
def _f(v: float) -> str:
    v = float(v)
    if abs(v) < 1e-15:
        return "0"
    return f"{v:.9g}"


def _v(values: Iterable[float]) -> str:
    return " ".join(_f(x) for x in values)


def _attr(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return _f(value)
    if isinstance(value, (list, tuple, np.ndarray)):
        return _v(value)
    return str(value)


def _sub(parent: ET.Element, tag: str, **attrs: Any) -> ET.Element:
    """``ET.SubElement`` that drops ``None`` attributes and formats numbers/vectors."""
    clean = {k.rstrip("_"): _attr(v) for k, v in attrs.items() if v is not None}
    return ET.SubElement(parent, tag, clean)


def _quat_from_euler_deg(euler_deg: Sequence[float]) -> Tuple[float, float, float, float]:
    """Intrinsic XYZ Euler angles (MuJoCo default ``eulerseq="xyz"``) in degrees -> quaternion."""
    q = np.zeros(4)
    mujoco.mju_euler2Quat(q, np.radians(np.asarray(euler_deg, dtype=float)), "xyz")
    return tuple(float(x) for x in q)


def _axis_angle_quat(axis: Sequence[float], angle: float) -> np.ndarray:
    q = np.zeros(4)
    mujoco.mju_axisAngle2Quat(q, np.asarray(axis, dtype=float), float(angle))
    return q


def _quat_mul(a: Sequence[float], b: Sequence[float]) -> np.ndarray:
    out = np.zeros(4)
    mujoco.mju_mulQuat(out, np.asarray(a, dtype=float), np.asarray(b, dtype=float))
    return out


def _rot_vec(q: Sequence[float], v: Sequence[float]) -> np.ndarray:
    out = np.zeros(3)
    mujoco.mju_rotVecQuat(out, np.asarray(v, dtype=float), np.asarray(q, dtype=float))
    return out


def _safe_name(name: str) -> str:
    s = re.sub(r"[^A-Za-z0-9_]+", "_", name).strip("_")
    return s or "item"


# --------------------------------------------------------------------------- #
# Collision groups
# --------------------------------------------------------------------------- #
class CollisionBits:
    """``contype`` / ``conaffinity`` conventions shared by all generated scenes.

    MuJoCo collides two geoms if ``(ct1 & ca2) or (ct2 & ca1)``.

    =================  =======  ============  ==========================================
    role               contype  conaffinity   collides with
    =================  =======  ============  ==========================================
    world / terrain    1        1             ball, props, chassis, ground rollers
    composite ball     2        1             world, props, chassis, ball rollers
    ball rollers       4        3             ball (driving) and world (after a fall)
    ball pair wheel    4        1             world; the ball via an explicit <pair>
    ground rollers     4        1             world only
    chassis / head     8        3             world, ball, props
    props (default)    1        1             world, ball, chassis, props
    pair-only geoms    0        0             only explicit <pair> partners
    visual geoms       0        0             nothing
    =================  =======  ============  ==========================================
    """

    WORLD = (1, 1)
    BALL = (2, 1)
    BALL_ROLLER = (4, 3)
    BALL_PAIR_WHEEL = (4, 1)
    GROUND_ROLLER = (4, 1)
    CHASSIS = (8, 3)
    PROP = (1, 1)
    NONE = (0, 0)


VISUAL_GROUP = 1
COLLISION_GROUP = 3


# --------------------------------------------------------------------------- #
# Scene extras: primitives and meshes
# --------------------------------------------------------------------------- #
@dataclass
class PrimitiveSpec:
    """A procedural primitive added to a scene (static, free-floating or attached to a body)."""

    name: str
    type: str = "box"                                  # sphere | box | cylinder | capsule | ellipsoid
    size: Tuple[float, ...] = (0.05, 0.05, 0.05)       # MuJoCo geom size convention (half sizes)
    pos: Tuple[float, float, float] = (0.5, 0.0, 0.05)
    euler_deg: Optional[Tuple[float, float, float]] = None
    quat: Optional[Tuple[float, float, float, float]] = None
    dynamic: bool = False                              # True: own body with a free joint
    parent: str = "world"                              # body to attach to when not dynamic
    density: Optional[float] = 500.0                   # used when mass is None
    mass: Optional[float] = None
    rgba: Tuple[float, float, float, float] = (0.3, 0.6, 0.9, 1.0)
    contype: int = CollisionBits.PROP[0]
    conaffinity: int = CollisionBits.PROP[1]
    friction: Tuple[float, float, float] = (0.8, 0.005, 0.0001)
    condim: int = 3

    def validate(self) -> None:
        if self.type not in ("sphere", "box", "cylinder", "capsule", "ellipsoid"):
            raise ValueError(f"unsupported primitive type {self.type!r}")
        need = {"sphere": 1, "capsule": 2, "cylinder": 2, "box": 3, "ellipsoid": 3}[self.type]
        if len(self.size) < need or min(self.size[:need]) <= 0:
            raise ValueError(f"{self.type} needs {need} positive size values")
        if self.euler_deg is not None and self.quat is not None:
            raise ValueError("give either euler_deg or quat, not both")
        if self.mass is None and (self.density is None or self.density <= 0):
            raise ValueError("primitive needs a positive density or an explicit mass")


@dataclass
class MeshImportSpec:
    """A CAD mesh (.stl / .obj / .ply / .off) to import into the scene."""

    path: str
    name: Optional[str] = None
    scale: float = 1.0                                 # e.g. 0.001 for millimetre CAD exports
    density: float = 1240.0                            # PLA default [kg/m^3]
    mass: Optional[float] = None                       # override (inertia rescaled accordingly)
    parent: str = "world"                              # body name, e.g. "head" to mount on the head
    dynamic: bool = False                              # free joint (only for parent == "world")
    pos: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    euler_deg: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    collision: str = "hull"                            # hull | decompose | none
    max_hulls: int = 16
    rgba: Tuple[float, float, float, float] = (0.7, 0.7, 0.75, 1.0)
    contype: int = CollisionBits.PROP[0]
    conaffinity: int = CollisionBits.PROP[1]
    friction: Tuple[float, float, float] = (0.8, 0.005, 0.0001)

    def validate(self) -> None:
        if not os.path.isfile(self.path):
            raise FileNotFoundError(self.path)
        if self.scale <= 0 or self.density <= 0:
            raise ValueError("mesh scale and density must be positive")
        if self.mass is not None and self.mass <= 0:
            raise ValueError("mesh mass override must be positive")
        if self.collision not in ("hull", "decompose", "none"):
            raise ValueError("collision must be 'hull', 'decompose' or 'none'")
        if self.dynamic and self.parent != "world":
            raise ValueError("only world-level meshes can be dynamic (free joint)")


@dataclass
class ImportedMesh:
    """Result of :meth:`MeshImporter.load` (all values in the scaled mesh frame)."""

    name: str
    visual_file: str
    collision_files: List[str]
    assets: Dict[str, bytes]
    mass: float
    volume: float
    com: np.ndarray
    inertia: np.ndarray                                 # 3x3 about the CoM, mesh axes
    watertight: bool
    bounds: np.ndarray
    notes: List[str] = field(default_factory=list)

    @property
    def fullinertia(self) -> Tuple[float, float, float, float, float, float]:
        i = self.inertia
        return (i[0, 0], i[1, 1], i[2, 2], i[0, 1], i[0, 2], i[1, 2])

    def info(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "mass": self.mass,
            "volume": self.volume,
            "com": self.com.tolist(),
            "inertia": self.inertia.tolist(),
            "watertight": self.watertight,
            "bounds": self.bounds.tolist(),
            "collision_parts": len(self.collision_files),
            "notes": list(self.notes),
        }


class MeshImporter:
    """Loads CAD meshes with ``trimesh`` and computes collision hulls and mass properties."""

    def __init__(self) -> None:
        try:
            import trimesh  # noqa: F401
        except ImportError as exc:  # pragma: no cover
            raise ImportError("MeshImporter requires 'trimesh' (pip install trimesh)") from exc

    @staticmethod
    def _export_stl(mesh: Any) -> bytes:
        return mesh.export(file_type="stl")

    def load(self, spec: MeshImportSpec) -> ImportedMesh:
        import trimesh

        spec.validate()
        name = _safe_name(spec.name or os.path.splitext(os.path.basename(spec.path))[0])
        loaded = trimesh.load(spec.path, force="mesh", process=True)
        if not isinstance(loaded, trimesh.Trimesh) or len(loaded.faces) == 0:
            raise ValueError(f"{spec.path}: no triangle geometry found")
        mesh = loaded.copy()
        mesh.apply_scale(spec.scale)
        mesh.merge_vertices()
        notes: List[str] = []
        if not mesh.is_winding_consistent:
            mesh.fix_normals()
            notes.append("fixed inconsistent face winding")

        # Mass properties: use the closed mesh if possible, else its convex hull.
        source = mesh
        if not mesh.is_watertight:
            trimesh.repair.fill_holes(mesh)
        if not mesh.is_watertight or mesh.volume <= 0:
            source = mesh.convex_hull
            notes.append("mesh is not watertight: mass properties computed from its convex hull")
        source.density = spec.density
        volume = float(abs(source.volume))
        mass = float(abs(source.mass))
        com = np.asarray(source.center_mass, dtype=float)
        inertia = np.asarray(source.moment_inertia, dtype=float)
        if inertia[0, 0] < 0:
            inertia = -inertia
        if spec.mass is not None:
            inertia = inertia * (spec.mass / mass)
            mass = float(spec.mass)
            notes.append("mass overridden; inertia scaled to match")
        inertia = self._regularise_inertia(inertia, mass, mesh.extents)

        assets: Dict[str, bytes] = {}
        visual_file = f"{name}_visual.stl"
        assets[visual_file] = self._export_stl(mesh)

        parts: List[Any] = []
        if spec.collision == "hull":
            parts = [mesh.convex_hull]
        elif spec.collision == "decompose":
            try:
                pieces = mesh.convex_decomposition()
                if isinstance(pieces, trimesh.Trimesh):
                    pieces = [pieces]
                parts = [
                    p if isinstance(p, trimesh.Trimesh) else trimesh.Trimesh(**p) for p in pieces
                ][: spec.max_hulls]
                notes.append(f"convex decomposition produced {len(parts)} parts")
            except Exception as exc:  # backend (vhacd/coacd) missing or failed
                parts = [mesh.convex_hull]
                notes.append(f"convex decomposition unavailable ({type(exc).__name__}); using single hull")
        collision_files: List[str] = []
        for k, part in enumerate(parts):
            if len(part.faces) < 4 or part.volume <= 1e-12:
                notes.append(f"skipped degenerate collision part {k}")
                continue
            fname = f"{name}_col{k}.stl"
            assets[fname] = self._export_stl(part)
            collision_files.append(fname)

        return ImportedMesh(
            name=name,
            visual_file=visual_file,
            collision_files=collision_files,
            assets=assets,
            mass=mass,
            volume=volume,
            com=com,
            inertia=inertia,
            watertight=bool(mesh.is_watertight),
            bounds=np.asarray(mesh.bounds, dtype=float),
            notes=notes,
        )

    @staticmethod
    def _regularise_inertia(inertia: np.ndarray, mass: float, extents: np.ndarray) -> np.ndarray:
        """Symmetrise and make sure the tensor is positive definite and satisfies the triangle inequality."""
        inertia = 0.5 * (inertia + inertia.T)
        w, v = np.linalg.eigh(inertia)
        floor = max(1e-10, 1e-6 * mass * float(np.max(extents)) ** 2)
        w = np.maximum(w, floor)
        # Triangle inequality on principal moments.
        for i in range(3):
            others = w.sum() - w[i]
            if w[i] > others:
                w[i] = others
        return (v * w) @ v.T


@dataclass
class SceneExtras:
    """Optional additions applied to any example scene."""

    primitives: List[PrimitiveSpec] = field(default_factory=list)
    meshes: List[MeshImportSpec] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Metadata & bundle
# --------------------------------------------------------------------------- #
@dataclass
class WheelMeta:
    """Names and kinematics of one driven (omni)wheel."""

    index: int
    joint: str
    actuator: str
    hub_body: str
    mount_body: str
    frame_site: str                                    # x = rolling dir, y = axle, z = normal
    collision_geoms: List[str]
    target_geom: str
    radius: float
    azimuth: float
    axle: Tuple[float, float, float]                   # in mount-body frame
    rolling_dir: Tuple[float, float, float]
    normal: Tuple[float, float, float]
    encoder_pos_sensor: str
    encoder_vel_sensor: str
    torque_sensor: str


@dataclass
class SceneMetadata:
    """Everything downstream modules need to drive and observe a generated scene."""

    example: str
    title: str
    description: str
    timestep: float
    control_dt: float
    omni_mode: Optional[str] = None
    control_mode: Optional[str] = None
    bodies: Dict[str, str] = field(default_factory=dict)       # role -> body name
    joints: Dict[str, str] = field(default_factory=dict)       # role -> joint name
    actuators: List[str] = field(default_factory=list)
    sensors: List[str] = field(default_factory=list)
    imu_site: Optional[str] = None
    wheels: List[WheelMeta] = field(default_factory=list)
    aligned_contacts: List[AlignedContact] = field(default_factory=list)
    magnets: List[MagnetLink] = field(default_factory=list)
    kinematics: Dict[str, Any] = field(default_factory=dict)
    init_qpos: Dict[str, Union[float, Sequence[float]]] = field(default_factory=dict)   # joint -> value
    params: Dict[str, Any] = field(default_factory=dict)
    imported_meshes: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def requires_stepper(self) -> bool:
        return bool(self.aligned_contacts or self.magnets)

    def as_dict(self) -> Dict[str, Any]:
        return params_to_dict(self)


@dataclass
class SceneBundle:
    """Generated MJCF text, its assets, and its metadata."""

    name: str
    xml: str
    assets: Dict[str, bytes]
    meta: SceneMetadata

    def compile(self) -> mujoco.MjModel:
        return mujoco.MjModel.from_xml_string(self.xml, self.assets)

    def to_spec(self) -> "mujoco.MjSpec":
        """``mujoco.MjSpec`` of the scene for further programmatic editing."""
        try:
            return mujoco.MjSpec.from_string(self.xml, assets=self.assets)
        except TypeError:  # older bindings
            spec = mujoco.MjSpec()
            spec.from_string(self.xml)
            return spec

    def make(self) -> Tuple[mujoco.MjModel, mujoco.MjData, SceneStepper]:
        """Compile, create data, apply the initial state and return a ready stepper."""
        model = self.compile()
        data = mujoco.MjData(model)
        self.reset(model, data)
        stepper = SceneStepper(model, data, self.meta)
        stepper.forward()
        return model, data, stepper

    def reset(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        mujoco.mj_resetData(model, data)
        for jname, value in self.meta.init_qpos.items():
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
            if jid < 0:
                raise KeyError(f"init_qpos refers to unknown joint {jname!r}")
            adr = model.jnt_qposadr[jid]
            vals = np.atleast_1d(np.asarray(value, dtype=float))
            data.qpos[adr : adr + vals.size] = vals
        mujoco.mj_forward(model, data)

    def write(self, directory: str, filename: str = "scene.xml") -> str:
        """Write the XML and its mesh assets to ``directory``; returns the XML path."""
        os.makedirs(directory, exist_ok=True)
        for fname, blob in self.assets.items():
            with open(os.path.join(directory, fname), "wb") as fh:
                fh.write(blob)
        path = os.path.join(directory, filename)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(self.xml)
        return path


# --------------------------------------------------------------------------- #
# Low-level builder
# --------------------------------------------------------------------------- #
class MJCFBuilder:
    """Thin structured wrapper around an MJCF element tree."""

    def __init__(self, model_name: str, sim: SimulationParams) -> None:
        sim.validate()
        self.sim = sim
        self.assets: Dict[str, bytes] = {}
        self.root = ET.Element("mujoco", {"model": model_name})
        _sub(self.root, "compiler", angle="radian", autolimits=True, inertiafromgeom="auto",
             boundmass=1e-6, boundinertia=1e-10)
        _sub(
            self.root, "option",
            timestep=sim.timestep, integrator=sim.integrator, cone=sim.cone, solver=sim.solver,
            iterations=sim.iterations, ls_iterations=sim.ls_iterations,
            noslip_iterations=sim.noslip_iterations, impratio=sim.impratio, gravity=sim.gravity,
        )
        visual = _sub(self.root, "visual")
        _sub(visual, "global", offwidth=sim.offscreen_width, offheight=sim.offscreen_height,
             azimuth=135, elevation=-25)
        _sub(visual, "headlight", ambient=(0.35, 0.35, 0.35), diffuse=(0.5, 0.5, 0.5), specular=(0.1, 0.1, 0.1))
        _sub(visual, "quality", shadowsize=4096)
        _sub(visual, "map", znear=0.005, zfar=50)
        _sub(visual, "scale", contactwidth=0.02, contactheight=0.005, forcewidth=0.01, framelength=0.15,
             framewidth=0.006)
        self.default = _sub(self.root, "default")
        vis = _sub(self.default, "default", class_="visual")
        _sub(vis, "geom", contype=0, conaffinity=0, group=VISUAL_GROUP, density=0)
        col = _sub(self.default, "default", class_="collision")
        _sub(col, "geom", group=COLLISION_GROUP, rgba=(0.6, 0.6, 0.6, 0.35))
        self.asset = _sub(self.root, "asset")
        self.worldbody = _sub(self.root, "worldbody")
        self.contact = _sub(self.root, "contact")
        self.equality = _sub(self.root, "equality")
        self.actuator = _sub(self.root, "actuator")
        self.sensor = _sub(self.root, "sensor")
        self._names: set = set()
        self._add_common_assets()

    # ------------------------------------------------------------- naming
    def unique(self, name: str) -> str:
        base = _safe_name(name)
        out, k = base, 1
        while out in self._names:
            out = f"{base}_{k}"
            k += 1
        self._names.add(out)
        return out

    def claim(self, name: str) -> str:
        if name in self._names:
            raise ValueError(f"duplicate MJCF name {name!r}")
        self._names.add(name)
        return name

    # ------------------------------------------------------------- assets
    def _add_common_assets(self) -> None:
        a = self.asset
        _sub(a, "texture", type="skybox", builtin="gradient", rgb1=(0.42, 0.58, 0.78), rgb2=(0.05, 0.07, 0.12),
             width=512, height=3072)
        _sub(a, "texture", name="tex_grid", type="2d", builtin="checker", rgb1=(0.20, 0.23, 0.27),
             rgb2=(0.14, 0.16, 0.19), width=512, height=512, mark="edge", markrgb=(0.35, 0.38, 0.42))
        _sub(a, "material", name="mat_grid", texture="tex_grid", texrepeat=(8, 8), texuniform=True,
             reflectance=0.15)
        _sub(a, "texture", name="tex_ball", type="cube", builtin="checker", rgb1=(0.94, 0.92, 0.86),
             rgb2=(0.86, 0.82, 0.74), width=256, height=256, mark="cross", markrgb=(0.95, 0.45, 0.10))
        _sub(a, "material", name="mat_ball", texture="tex_ball", specular=0.4, shininess=0.4, reflectance=0.05)
        _sub(a, "material", name="mat_head", rgba=(0.95, 0.95, 0.93, 1), specular=0.5, shininess=0.6)
        _sub(a, "material", name="mat_accent", rgba=(0.95, 0.45, 0.10, 1), specular=0.3)
        _sub(a, "material", name="mat_dark", rgba=(0.12, 0.12, 0.14, 1), specular=0.6, shininess=0.8)
        _sub(a, "material", name="mat_metal", rgba=(0.62, 0.64, 0.68, 1), specular=0.9, shininess=0.9,
             reflectance=0.2)
        _sub(a, "material", name="mat_roller", rgba=(0.85, 0.15, 0.10, 1), specular=0.2)
        _sub(a, "material", name="mat_cart", rgba=(0.20, 0.45, 0.80, 1), specular=0.4)
        _sub(a, "material", name="mat_pole", rgba=(0.90, 0.70, 0.20, 1), specular=0.4)
        _sub(a, "material", name="mat_target", rgba=(0.10, 0.90, 0.40, 0.45))

    def add_mesh_asset(self, name: str, filename: str, blob: bytes) -> str:
        name = self.claim(name)
        self.assets[filename] = blob
        _sub(self.asset, "mesh", name=name, file=filename)
        return name

    # ------------------------------------------------------------- world
    def add_lights_and_cameras(self, track_body: Optional[str] = None) -> None:
        _sub(self.worldbody, "light", name=self.claim("sun"), directional=True, pos=(0, 0, 6),
             dir=(-0.3, 0.2, -1.0), diffuse=(0.8, 0.8, 0.78), specular=(0.2, 0.2, 0.2), castshadow=True)
        _sub(self.worldbody, "light", name=self.claim("fill"), directional=True, pos=(0, 0, 6),
             dir=(0.5, -0.4, -0.6), diffuse=(0.25, 0.27, 0.3), castshadow=False)
        _sub(self.worldbody, "camera", name=self.claim("overview"), pos=(2.2, -2.2, 1.6),
             xyaxes=(0.707, 0.707, 0, -0.35, 0.35, 0.87))

    def add_floor(self, terrain: TerrainParams) -> List[str]:
        """Floor plane (optionally sloped), bumps and heightfields. Returns terrain geom names."""
        names: List[str] = []
        euler = (0.0, 0.0, 0.0)
        if terrain.floor_slope_deg:
            ang = math.radians(terrain.floor_slope_deg)
            euler = (ang, 0.0, 0.0) if terrain.slope_axis == "x" else (0.0, ang, 0.0)
        floor = self.claim("floor")
        _sub(self.worldbody, "geom", name=floor, type="plane",
             size=(terrain.floor_half_size, terrain.floor_half_size, 0.05), euler=euler, material="mat_grid",
             contype=CollisionBits.WORLD[0], conaffinity=CollisionBits.WORLD[1], **terrain.floor.mjcf_attrs())
        names.append(floor)
        for k, bump in enumerate(terrain.bumps):
            gname = self.claim(f"bump_{k}")
            _sub(self.worldbody, "geom", name=gname, type=bump.type, size=bump.size, pos=bump.pos,
                 euler=np.radians(bump.euler_deg), rgba=bump.rgba,
                 contype=CollisionBits.WORLD[0], conaffinity=CollisionBits.WORLD[1], **terrain.floor.mjcf_attrs())
            names.append(gname)
        for hf in terrain.heightfields:
            names.append(self.add_heightfield(hf, terrain.floor))
        return names

    def add_heightfield(self, hf: HeightfieldSpec, contact: ContactParams) -> str:
        hf.validate()
        hname = self.claim(f"hf_{_safe_name(hf.name)}")
        grid = hf.elevation_grid()
        _sub(self.asset, "hfield", name=hname, nrow=hf.nrow, ncol=hf.ncol,
             size=(hf.half_size[0], hf.half_size[1], hf.max_height, hf.base_depth),
             elevation=" ".join(f"{z:.5f}" for z in grid.ravel()))
        gname = self.claim(f"{hname}_geom")
        _sub(self.worldbody, "geom", name=gname, type="hfield", hfield=hname, pos=hf.pos,
             rgba=(0.42, 0.40, 0.36, 1), contype=CollisionBits.WORLD[0], conaffinity=CollisionBits.WORLD[1],
             **contact.mjcf_attrs())
        return gname

    def to_xml(self) -> str:
        # Drop empty top-level sections to keep the file tidy.
        for tag in ("contact", "equality", "actuator", "sensor"):
            el = self.root.find(tag)
            if el is not None and len(el) == 0:
                self.root.remove(el)
        tree = ET.ElementTree(self.root)
        ET.indent(tree, space="  ")
        buf = io.StringIO()
        tree.write(buf, encoding="unicode", xml_declaration=False)
        return buf.getvalue() + "\n"


# --------------------------------------------------------------------------- #
# Shared component builders
# --------------------------------------------------------------------------- #
def _add_actuator(
    b: MJCFBuilder, joint: str, motor: MotorParams, name: Optional[str] = None,
) -> str:
    """Torque / velocity / position actuator on ``joint`` according to ``motor.control_mode``."""
    name = b.claim(name or f"act_{joint}")
    lim = motor.wheel_torque_limit
    mode = ControlMode(motor.control_mode)
    if mode == ControlMode.TORQUE:
        _sub(b.actuator, "motor", name=name, joint=joint, gear=(motor.gear_ratio,),
             ctrlrange=(-motor.torque_limit, motor.torque_limit))
    elif mode == ControlMode.VELOCITY:
        _sub(b.actuator, "velocity", name=name, joint=joint, kv=motor.kv,
             ctrlrange=(-motor.max_wheel_speed, motor.max_wheel_speed), forcerange=(-lim, lim))
    else:
        _sub(b.actuator, "position", name=name, joint=joint, kp=motor.kp, kv=motor.position_kv,
             forcerange=(-lim, lim))
    return name


def _add_joint_sensors(b: MJCFBuilder, joint: str, actuator: Optional[str], prefix: str) -> Tuple[str, str, str]:
    pos = b.claim(f"{prefix}_pos")
    vel = b.claim(f"{prefix}_vel")
    _sub(b.sensor, "jointpos", name=pos, joint=joint)
    _sub(b.sensor, "jointvel", name=vel, joint=joint)
    trq = ""
    if actuator:
        trq = b.claim(f"{prefix}_torque")
        _sub(b.sensor, "actuatorfrc", name=trq, actuator=actuator)
    return pos, vel, trq


def _add_imu(b: MJCFBuilder, body_el: ET.Element, prefix: str, pos: Sequence[float]) -> Tuple[str, List[str]]:
    site = b.claim(f"{prefix}_imu")
    _sub(body_el, "site", name=site, pos=pos, size=(0.008,), rgba=(0.1, 0.8, 0.3, 1), group=4)
    names = []
    for kind, suffix in (("accelerometer", "acc"), ("gyro", "gyro"), ("velocimeter", "vel")):
        n = b.claim(f"{prefix}_imu_{suffix}")
        _sub(b.sensor, kind, name=n, site=site)
        names.append(n)
    n = b.claim(f"{prefix}_imu_quat")
    _sub(b.sensor, "framequat", name=n, objtype="site", objname=site)
    names.append(n)
    return site, names


def _add_omniwheel(
    b: MJCFBuilder,
    mount_el: ET.Element,
    mount_body: str,
    name: str,
    center: Sequence[float],
    normal: Sequence[float],
    rolling_dir: Sequence[float],
    wheel: OmniwheelParams,
    motor: MotorParams,
    target_geoms: Sequence[str],
    roller_bits: Tuple[int, int],
    index: int,
    azimuth: float,
    pair_geom_bits: Tuple[int, int] = CollisionBits.NONE,
) -> WheelMeta:
    """Add one omniwheel (hub + drive hinge + rollers or pair contact) to ``mount_el``.

    ``normal`` points from the surface the wheel rides on toward the wheel
    centre; ``rolling_dir`` is the traction direction. Both are expressed in
    the mount body frame. The drive hinge axis is ``axle = normal x rolling_dir``.

    ``pair_geom_bits`` are the contype/conaffinity of the contact-pair sphere in
    addition to its explicit pairs (e.g. so a head-mounted wheel still lands on
    the floor after a fall). They must not overlap with ``target_geoms``.
    """
    n, t, a = wheel_frame(normal, rolling_dir)
    hub_quat = rotation_matrix_to_quat(np.column_stack([n, t, a]))
    site_quat = rotation_matrix_to_quat(np.column_stack([t, a, n]))
    mode = OmniwheelMode(wheel.mode)

    frame_site = b.claim(f"{name}_frame")
    _sub(mount_el, "site", name=frame_site, pos=center, quat=site_quat, size=(0.004,), group=5,
         rgba=(1, 1, 0, 1))

    hub_name = b.claim(f"{name}_hub")
    hub = _sub(mount_el, "body", name=hub_name, pos=center, quat=hub_quat)
    joint = b.claim(f"{name}_drive")
    _sub(hub, "joint", name=joint, type="hinge", axis=(0, 0, 1), damping=motor.joint_damping,
         armature=motor.armature, frictionloss=motor.frictionloss)
    r, w = wheel.radius, wheel.width
    _sub(hub, "inertial", pos=(0, 0, 0), mass=wheel.hub_mass,
         diaginertia=solid_cylinder_inertia(wheel.hub_mass, 0.8 * r, w))
    _sub(hub, "geom", class_="visual", type="cylinder", size=(0.55 * r, 0.5 * w), material="mat_metal")
    _sub(hub, "geom", class_="visual", type="cylinder", size=(0.18 * r, 0.62 * w), material="mat_dark")

    rows = wheel.roller_rows
    per_row = wheel.rollers_per_row
    r_roll = wheel.roller_radius
    arc = 2.0 * math.pi * wheel.roller_center_radius / per_row
    shape = RollerShape(wheel.roller_shape)
    if shape == RollerShape.CAPSULE:
        half_len = max(0.5 * arc * wheel.roller_fill - r_roll, 0.15 * arc)
        rho = math.sqrt(max((r - r_roll) ** 2 - half_len**2, 1e-12))   # envelope peak == r
        roller_size: Tuple[float, ...] = (r_roll, half_len)
    else:
        half_len = 0.5 * arc * wheel.roller_fill
        rho = r - r_roll
        roller_size = (r_roll, r_roll, half_len)
    row_z = [0.0] if rows == 1 else [-0.25 * w, 0.25 * w]

    collision_geoms: List[str] = []
    for row, z in enumerate(row_z):
        for k in range(per_row):
            phi = 2.0 * math.pi * k / per_row + (math.pi / per_row if row == 1 else 0.0)
            radial = np.array([math.cos(phi), math.sin(phi), 0.0])
            tangent = np.array([-math.sin(phi), math.cos(phi), 0.0])
            rot = np.column_stack([radial, [0.0, 0.0, -1.0], tangent])
            rq = rotation_matrix_to_quat(rot)
            rpos = rho * radial + np.array([0.0, 0.0, z])
            if mode == OmniwheelMode.SUBROLLER:
                rb = _sub(hub, "body", name=b.claim(f"{name}_roller_{row}_{k}"), pos=rpos, quat=rq)
                _sub(rb, "joint", name=b.claim(f"{name}_roller_{row}_{k}_spin"), type="hinge", axis=(0, 0, 1),
                     damping=wheel.roller_damping, armature=wheel.roller_armature)
                gname = b.claim(f"{name}_roller_{row}_{k}_geom")
                _sub(rb, "geom", name=gname, type=shape.value, size=roller_size, mass=wheel.roller_mass,
                     material="mat_roller", contype=roller_bits[0], conaffinity=roller_bits[1], condim=3,
                     friction=(wheel.mu_tangent, wheel.mu_torsional, wheel.mu_rolling),
                     solref=wheel.solref, solimp=wheel.solimp)
                collision_geoms.append(gname)
            else:
                _sub(hub, "geom", class_="visual", type=shape.value, size=roller_size, pos=rpos, quat=rq,
                     material="mat_roller")

    if mode == OmniwheelMode.CONTACT_PAIR:
        gname = b.claim(f"{name}_contact")
        _sub(hub, "geom", name=gname, type="sphere", size=(r,), contype=pair_geom_bits[0],
             conaffinity=pair_geom_bits[1], group=COLLISION_GROUP, density=0, rgba=(0.9, 0.2, 0.2, 0.25),
             friction=(wheel.mu_tangent, wheel.mu_torsional, wheel.mu_rolling), condim=3)
        collision_geoms.append(gname)
        for tg in target_geoms:
            _sub(b.contact, "pair", name=b.claim(f"pair_{name}_{tg}"), geom1=gname, geom2=tg, condim=3,
                 friction=wheel.pair_friction, solref=wheel.solref, solimp=wheel.solimp)
            b._pending_aligned.append(AlignedContact(gname, tg, frame_site, wheel.pair_friction))  # type: ignore[attr-defined]

    act = _add_actuator(b, joint, motor, name=f"{name}_motor")
    enc_pos, enc_vel, trq = _add_joint_sensors(b, joint, act, prefix=f"{name}_enc")
    return WheelMeta(
        index=index, joint=joint, actuator=act, hub_body=hub_name, mount_body=mount_body, frame_site=frame_site,
        collision_geoms=collision_geoms, target_geom=target_geoms[0] if target_geoms else "",
        radius=r, azimuth=azimuth, axle=tuple(float(x) for x in a), rolling_dir=tuple(float(x) for x in t),
        normal=tuple(float(x) for x in n), encoder_pos_sensor=enc_pos, encoder_vel_sensor=enc_vel,
        torque_sensor=trq,
    )


def _new_builder(name: str, sim: SimulationParams) -> MJCFBuilder:
    b = MJCFBuilder(name, sim)
    b._pending_aligned = []  # type: ignore[attr-defined]
    return b


def _apply_extras(b: MJCFBuilder, meta: SceneMetadata, extras: Optional[SceneExtras]) -> None:
    if extras is None:
        return
    bodies = {el.get("name"): el for el in b.worldbody.iter("body")}
    for prim in extras.primitives:
        prim.validate()
        orient: Dict[str, Any] = {}
        if prim.quat is not None:
            orient["quat"] = prim.quat
        elif prim.euler_deg is not None:
            orient["euler"] = np.radians(prim.euler_deg)
        mass_attr = {"mass": prim.mass} if prim.mass is not None else {"density": prim.density}
        gattrs = dict(type=prim.type, size=prim.size, rgba=prim.rgba, contype=prim.contype,
                      conaffinity=prim.conaffinity, friction=prim.friction, condim=prim.condim, **mass_attr)
        if prim.dynamic:
            body = _sub(b.worldbody, "body", name=b.claim(prim.name), pos=prim.pos, **orient)
            _sub(body, "freejoint", name=b.claim(f"{prim.name}_free"))
            _sub(body, "geom", name=b.claim(f"{prim.name}_geom"), **gattrs)
            meta.bodies[f"prop:{prim.name}"] = prim.name
        else:
            parent = b.worldbody if prim.parent == "world" else bodies.get(prim.parent)
            if parent is None:
                raise KeyError(f"primitive parent body {prim.parent!r} not found")
            _sub(parent, "geom", name=b.claim(prim.name), pos=prim.pos, **orient, **gattrs)

    if extras.meshes:
        importer = MeshImporter()
        for spec in extras.meshes:
            im = importer.load(spec)
            vis_mesh = b.add_mesh_asset(f"{im.name}_visual", im.visual_file, im.assets[im.visual_file])
            col_meshes = [
                b.add_mesh_asset(os.path.splitext(f)[0], f, im.assets[f]) for f in im.collision_files
            ]
            body_name = b.claim(im.name)
            parent = b.worldbody if spec.parent == "world" else bodies.get(spec.parent)
            if parent is None:
                raise KeyError(f"mesh parent body {spec.parent!r} not found")
            body = _sub(parent, "body", name=body_name, pos=spec.pos, euler=np.radians(spec.euler_deg))
            if spec.dynamic:
                _sub(body, "freejoint", name=b.claim(f"{im.name}_free"))
            if spec.dynamic or spec.parent != "world":
                _sub(body, "inertial", pos=im.com, mass=im.mass, fullinertia=im.fullinertia)
            _sub(body, "geom", class_="visual", type="mesh", mesh=vis_mesh, rgba=spec.rgba)
            if spec.collision != "none":
                for cm in col_meshes:
                    _sub(body, "geom", name=b.claim(f"{cm}_geom"), type="mesh", mesh=cm, group=COLLISION_GROUP,
                         contype=spec.contype, conaffinity=spec.conaffinity, friction=spec.friction,
                         rgba=(0.5, 0.5, 0.9, 0.3), density=0)
            meta.imported_meshes.append(im.info())
            meta.bodies[f"mesh:{im.name}"] = body_name


def _finish(b: MJCFBuilder, meta: SceneMetadata, extras: Optional[SceneExtras]) -> SceneBundle:
    _apply_extras(b, meta, extras)
    meta.aligned_contacts = list(b._pending_aligned)  # type: ignore[attr-defined]
    meta.sensors = [el.get("name") for el in b.sensor]
    meta.actuators = [el.get("name") for el in b.actuator]
    bundle = SceneBundle(name=meta.example, xml=b.to_xml(), assets=dict(b.assets), meta=meta)
    return bundle


def _default_sim(sim: Optional[SimulationParams], subroller: bool) -> SimulationParams:
    if sim is not None:
        return sim
    # 500 Hz for both omniwheel models (sub-roller scenes are also stable at 1 ms).
    return SimulationParams(timestep=0.002)


def _ground_height(terrain: TerrainParams, x: float = 0.0, y: float = 0.0) -> float:
    """Height of the (possibly sloped) floor plane above world point (x, y)."""
    ang = math.radians(terrain.floor_slope_deg)
    if terrain.slope_axis == "y":   # rotation about y: z = -x tan(ang)
        return -x * math.tan(ang)
    return y * math.tan(ang)        # rotation about x: z = y tan(ang)


def _floor_normal(terrain: TerrainParams) -> np.ndarray:
    ang = math.radians(terrain.floor_slope_deg)
    if terrain.slope_axis == "y":
        return np.array([math.sin(ang), 0.0, math.cos(ang)])
    return np.array([0.0, -math.sin(ang), math.cos(ang)])


# --------------------------------------------------------------------------- #
# Example 1: inverted pendulum
# --------------------------------------------------------------------------- #
def build_inverted_pendulum(
    params: Optional[PendulumParams] = None,
    sim: Optional[SimulationParams] = None,
    extras: Optional[SceneExtras] = None,
) -> SceneBundle:
    """Cart-pole (``variant='cartpole'``) or 3D gimbal pendulum on an XY cart (``'spherical'``)."""
    p = params or PendulumParams()
    p.validate()
    sim = _default_sim(sim, False)
    b = _new_builder(f"example1_{p.variant}", sim)
    b.add_lights_and_cameras()
    b.add_floor(sim.terrain)
    rail_z = 0.6
    meta = SceneMetadata(
        example="inverted_pendulum",
        title=f"Example 1: inverted pendulum ({p.variant})",
        description="Cart-actuated inverted pendulum for state-space / PID / RL sanity checks.",
        timestep=sim.timestep, control_dt=0.01, control_mode="motor",
    )
    L = p.rail_half_length
    _sub(b.worldbody, "geom", class_="visual", type="capsule", fromto=(-L, 0, rail_z, L, 0, rail_z), size=(0.012,),
         material="mat_metal")
    if p.variant == "spherical":
        _sub(b.worldbody, "geom", class_="visual", type="capsule", fromto=(0, -L, rail_z, 0, L, rail_z),
             size=(0.012,), material="mat_metal")
    for x in (-L, L):
        _sub(b.worldbody, "geom", class_="visual", type="cylinder", fromto=(x, 0, 0, x, 0, rail_z), size=(0.02,),
             material="mat_dark")

    cart = _sub(b.worldbody, "body", name=b.claim("cart"), pos=(0, 0, rail_z))
    slide_x = b.claim("cart_x")
    _sub(cart, "joint", name=slide_x, type="slide", axis=(1, 0, 0), range=(-L, L), damping=p.cart_damping,
         limited=True)
    joints = {"cart_x": slide_x}
    if p.variant == "spherical":
        slide_y = b.claim("cart_y")
        _sub(cart, "joint", name=slide_y, type="slide", axis=(0, 1, 0), range=(-L, L), damping=p.cart_damping,
             limited=True)
        joints["cart_y"] = slide_y
    sx, sy, sz = p.cart_size
    _sub(cart, "inertial", pos=(0, 0, 0), mass=p.cart_mass, diaginertia=solid_box_inertia(p.cart_mass, 2 * sx, 2 * sy, 2 * sz))
    _sub(cart, "geom", class_="visual", type="box", size=p.cart_size, material="mat_cart")

    pole = _sub(cart, "body", name=b.claim("pole"), pos=(0, 0, sz))
    hinge_y = b.claim("pole_pitch")
    _sub(pole, "joint", name=hinge_y, type="hinge", axis=(0, 1, 0), damping=p.pole_damping)
    joints["pole_pitch"] = hinge_y
    if p.variant == "spherical":
        hinge_x = b.claim("pole_roll")
        _sub(pole, "joint", name=hinge_x, type="hinge", axis=(1, 0, 0), damping=p.pole_damping)
        joints["pole_roll"] = hinge_x
    _sub(pole, "geom", name=b.claim("pole_geom"), type="capsule", fromto=(0, 0, 0, 0, 0, p.pole_length),
         size=(p.pole_radius,), mass=p.pole_mass, material="mat_pole", contype=0, conaffinity=0)
    _sub(pole, "geom", name=b.claim("pole_tip"), type="sphere", pos=(0, 0, p.pole_length),
         size=(2.2 * p.pole_radius,), mass=p.tip_mass, material="mat_accent", contype=0, conaffinity=0)
    imu_site, imu_sensors = _add_imu(b, pole, "pole", (0, 0, p.pole_length))

    acts = [_add_actuator(b, slide_x, MotorParams(control_mode=ControlMode.TORQUE, torque_limit=p.force_limit, gear_ratio=1.0), name="cart_force_x")]
    _add_joint_sensors(b, slide_x, acts[0], "cart_x")
    _add_joint_sensors(b, hinge_y, None, "pole_pitch")
    if p.variant == "spherical":
        acts.append(_add_actuator(b, joints["cart_y"], MotorParams(control_mode=ControlMode.TORQUE, torque_limit=p.force_limit, gear_ratio=1.0),
                                  name="cart_force_y"))
        _add_joint_sensors(b, joints["cart_y"], acts[1], "cart_y")
        _add_joint_sensors(b, joints["pole_roll"], None, "pole_roll")

    meta.bodies = {"cart": "cart", "pole": "pole"}
    meta.joints = joints
    meta.imu_site = imu_site
    meta.init_qpos = {hinge_y: math.radians(p.initial_angle_deg)}
    if p.variant == "spherical":
        meta.init_qpos[joints["pole_roll"]] = math.radians(0.5 * p.initial_angle_deg)
    meta.params = {"pendulum": params_to_dict(p), "sim": params_to_dict(sim)}
    return _finish(b, meta, extras)


# --------------------------------------------------------------------------- #
# Example 2: single omniwheel
# --------------------------------------------------------------------------- #
def build_single_omniwheel(
    params: Optional[SingleOmniwheelParams] = None,
    sim: Optional[SimulationParams] = None,
    extras: Optional[SceneExtras] = None,
) -> SceneBundle:
    """One omniwheel under a chassis constrained to planar motion (x, y, z, yaw)."""
    p = params or SingleOmniwheelParams()
    p.validate()
    sub = OmniwheelMode(p.wheel.mode) == OmniwheelMode.SUBROLLER
    sim = _default_sim(sim, sub)
    b = _new_builder("example2_single_omniwheel", sim)
    b.add_lights_and_cameras()
    terrain_geoms = b.add_floor(sim.terrain)
    meta = SceneMetadata(
        example="single_omniwheel",
        title="Example 2: single omniwheel on flat ground",
        description="Anisotropic rolling vs lateral sliding of one omniwheel; push the chassis sideways.",
        timestep=sim.timestep, control_dt=0.01, omni_mode=OmniwheelMode(p.wheel.mode).value,
        control_mode=ControlMode(p.motor.control_mode).value,
    )
    r = p.wheel.radius
    z0 = _ground_height(sim.terrain) + r + 1e-4
    chassis = _sub(b.worldbody, "body", name=b.claim("chassis"), pos=(0, 0, z0))
    joints = {}
    dofs = [("chassis_x", "slide", (1, 0, 0)), ("chassis_y", "slide", (0, 1, 0)), ("chassis_z", "slide", (0, 0, 1))]
    if p.free_yaw:
        dofs.append(("chassis_yaw", "hinge", (0, 0, 1)))
    for jn, jt, ax in dofs:
        joints[jn] = b.claim(jn)
        _sub(chassis, "joint", name=joints[jn], type=jt, axis=ax, damping=0.0)
    hx, hy, hz = p.chassis_half_size
    _sub(chassis, "inertial", pos=(0, 0, r + hz), mass=p.chassis_mass,
         diaginertia=solid_box_inertia(p.chassis_mass, 2 * hx, 2 * hy, 2 * hz))
    _sub(chassis, "geom", class_="visual", type="box", pos=(0, 0, r + hz), size=p.chassis_half_size,
         material="mat_cart")
    _sub(chassis, "geom", class_="visual", type="box", pos=(0, 0.65 * p.wheel.width + 0.004, 0.5 * (r + hz)),
         size=(0.01, 0.004, 0.5 * (r + hz) + hz), material="mat_metal")
    _sub(chassis, "geom", class_="visual", type="box", pos=(0, -0.65 * p.wheel.width - 0.004, 0.5 * (r + hz)),
         size=(0.01, 0.004, 0.5 * (r + hz) + hz), material="mat_metal")
    _sub(chassis, "geom", name=b.claim("chassis_collision"), type="box", pos=(0, 0, r + hz), size=p.chassis_half_size,
         contype=CollisionBits.CHASSIS[0], conaffinity=CollisionBits.CHASSIS[1], group=COLLISION_GROUP, density=0)
    imu_site, _ = _add_imu(b, chassis, "chassis", (0, 0, r + hz))
    wheel = _add_omniwheel(
        b, chassis, "chassis", "wheel0", center=(0, 0, 0), normal=(0, 0, 1), rolling_dir=(1, 0, 0), wheel=p.wheel,
        motor=p.motor, target_geoms=terrain_geoms, roller_bits=CollisionBits.GROUND_ROLLER, index=0, azimuth=0.0,
    )
    for role in ("chassis",):
        for kind in ("framepos", "framelinvel"):
            _sub(b.sensor, kind, name=b.claim(f"{role}_{kind[5:]}"), objtype="body", objname=role)
    meta.bodies = {"chassis": "chassis"}
    meta.joints = joints
    meta.imu_site = imu_site
    meta.wheels = [wheel]
    meta.kinematics = {
        "wheel_radius": r,
        "rolling_dir_body": wheel.rolling_dir,
        "axle_body": wheel.axle,
        "note": "slip = v_chassis . rolling_dir - r * omega_wheel; lateral motion along the axle is free.",
    }
    meta.params = {"single_omniwheel": params_to_dict(p), "sim": params_to_dict(sim)}
    return _finish(b, meta, extras)


# --------------------------------------------------------------------------- #
# Example 3: kiwi drive
# --------------------------------------------------------------------------- #
def build_kiwi_drive(
    params: Optional[KiwiDriveParams] = None,
    sim: Optional[SimulationParams] = None,
    extras: Optional[SceneExtras] = None,
) -> SceneBundle:
    """Three omniwheels at 120 deg on a free-floating chassis."""
    p = params or KiwiDriveParams()
    p.validate()
    sub = OmniwheelMode(p.wheel.mode) == OmniwheelMode.SUBROLLER
    sim = _default_sim(sim, sub)
    b = _new_builder("example3_kiwi_drive", sim)
    b.add_lights_and_cameras()
    terrain_geoms = b.add_floor(sim.terrain)
    meta = SceneMetadata(
        example="kiwi_drive",
        title="Example 3: 3-wheel kiwi holonomic drive",
        description="Holonomic (vx, vy, wz) control through the 120 deg omniwheel mixing matrix.",
        timestep=sim.timestep, control_dt=0.01, omni_mode=OmniwheelMode(p.wheel.mode).value,
        control_mode=ControlMode(p.motor.control_mode).value,
    )
    r = p.wheel.radius
    z0 = _ground_height(sim.terrain) + r + 1e-4
    chassis = _sub(b.worldbody, "body", name=b.claim("chassis"), pos=(0, 0, z0))
    free = b.claim("chassis_free")
    _sub(chassis, "freejoint", name=free)
    th = p.chassis_thickness
    plate_z = r + 0.5 * th
    _sub(chassis, "inertial", pos=(0, 0, plate_z), mass=p.chassis_mass,
         diaginertia=solid_cylinder_inertia(p.chassis_mass, 0.85 * p.chassis_radius, th))
    _sub(chassis, "geom", class_="visual", type="cylinder", pos=(0, 0, plate_z), size=(0.85 * p.chassis_radius, 0.5 * th),
         material="mat_cart")
    _sub(chassis, "geom", class_="visual", type="box", pos=(0.6 * p.chassis_radius, 0, plate_z + 0.5 * th + 0.003),
         size=(0.03, 0.008, 0.003), material="mat_accent")  # forward (+x) indicator
    _sub(chassis, "geom", name=b.claim("chassis_collision"), type="cylinder", pos=(0, 0, plate_z),
         size=(0.85 * p.chassis_radius, 0.5 * th), contype=CollisionBits.CHASSIS[0],
         conaffinity=CollisionBits.CHASSIS[1], group=COLLISION_GROUP, density=0)
    imu_site, _ = _add_imu(b, chassis, "chassis", (0, 0, plate_z))
    wheels = []
    for i, theta in enumerate(p.wheel_angles):
        e_r = np.array([math.cos(theta), math.sin(theta), 0.0])
        t = np.array([-math.sin(theta), math.cos(theta), 0.0])
        center = p.chassis_radius * e_r
        _sub(chassis, "geom", class_="visual", type="box", pos=center * 0.82 + np.array([0, 0, 0.5 * plate_z]),
             size=(0.012, 0.012, 0.5 * plate_z + 0.004), material="mat_metal",
             euler=(0, 0, theta))
        wheels.append(
            _add_omniwheel(
                b, chassis, "chassis", f"wheel{i}", center=center, normal=(0, 0, 1), rolling_dir=t, wheel=p.wheel,
                motor=p.motor, target_geoms=terrain_geoms, roller_bits=CollisionBits.GROUND_ROLLER, index=i,
                azimuth=theta,
            )
        )
    for kind in ("framepos", "framequat", "framelinvel", "frameangvel"):
        _sub(b.sensor, kind, name=b.claim(f"chassis_{kind[5:]}"), objtype="body", objname="chassis")
    meta.bodies = {"chassis": "chassis"}
    meta.joints = {"chassis_free": free}
    meta.imu_site = imu_site
    meta.wheels = wheels
    meta.kinematics = {
        "wheel_radius": r,
        "chassis_radius": p.chassis_radius,
        "wheel_angles": list(p.wheel_angles),
        "mixing_matrix": p.mixing_matrix().tolist(),
        "note": "omega = M @ [vx, vy, wz] (chassis frame); positive omega drives the rim contact along -t.",
    }
    meta.params = {"kiwi_drive": params_to_dict(p), "sim": params_to_dict(sim)}
    return _finish(b, meta, extras)


# --------------------------------------------------------------------------- #
# Composite ball (shared by examples 4 and 5)
# --------------------------------------------------------------------------- #
def _add_composite_ball(
    b: MJCFBuilder, parent: ET.Element, ball: CompositeBallParams, pos: Sequence[float],
    joint_kind: str = "free",
) -> Tuple[ET.Element, str, Dict[str, str]]:
    """``<body name="composite_ball">`` with exact layered inertia. Returns (body, collision geom, joints)."""
    ball.validate()
    body = _sub(parent, "body", name=b.claim("composite_ball"), pos=pos)
    joints: Dict[str, str] = {}
    if joint_kind == "free":
        joints["ball_free"] = b.claim("ball_free")
        _sub(body, "freejoint", name=joints["ball_free"])
    elif joint_kind == "planar_xz":
        for jn, jt, ax in (("ball_x", "slide", (1, 0, 0)), ("ball_z", "slide", (0, 0, 1)),
                           ("ball_pitch", "hinge", (0, 1, 0))):
            joints[jn] = b.claim(jn)
            _sub(body, "joint", name=joints[jn], type=jt, axis=ax)
    else:
        raise ValueError(f"unknown ball joint kind {joint_kind!r}")
    _sub(body, "inertial", **ball.mjcf_inertial_attrs())   # exact layered mass / inertia
    geom = b.claim("ball_geom")
    _sub(body, "geom", name=geom, type="sphere", size=(ball.r2,), material="mat_ball",
         contype=CollisionBits.BALL[0], conaffinity=CollisionBits.BALL[1], density=0, **ball.surface.mjcf_attrs())
    # BB-8 style decorative panels (visual only, make rotation visible).
    for k, nrm in enumerate([(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]):
        nv = np.array(nrm, dtype=float)
        rot = rotation_matrix_to_quat(np.column_stack(_basis_from_z(nv)))
        _sub(body, "geom", class_="visual", type="cylinder", pos=nv * (ball.r2 - 0.0035), quat=rot,
             size=(0.42 * ball.r2, 0.004), material="mat_accent" if k % 2 == 0 else "mat_metal")
        _sub(body, "geom", class_="visual", type="cylinder", pos=nv * (ball.r2 - 0.002), quat=rot,
             size=(0.26 * ball.r2, 0.0045), material="mat_head")
    # Show the layer boundaries as faint inner shells (visual group 2).
    _sub(body, "geom", type="sphere", size=(ball.r1,), contype=0, conaffinity=0, group=2, density=0,
         rgba=tuple(ball.foam_rgba[:3]) + (0.25,))
    if ball.r0 > 0:
        _sub(body, "geom", type="sphere", size=(ball.r0,), contype=0, conaffinity=0, group=2, density=0,
             rgba=(0.6, 0.8, 1.0, 0.15))
    _sub(body, "site", name=b.claim("ball_center"), pos=(0, 0, 0), size=(0.01,), group=4, rgba=(1, 0, 1, 1))
    for kind in ("framepos", "framequat", "framelinvel", "frameangvel"):
        _sub(b.sensor, kind, name=b.claim(f"ball_{kind[5:]}"), objtype="body", objname="composite_ball")
    return body, geom, joints


def _basis_from_z(z: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    z = z / np.linalg.norm(z)
    helper = np.array([1.0, 0.0, 0.0]) if abs(z[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    x = np.cross(helper, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return x, y, z


def _add_head_visuals(b: MJCFBuilder, head: ET.Element, hp: HeadUnitParams, wheel_centers: Sequence[np.ndarray]) -> None:
    rc, hc = hp.chassis_radius, hp.chassis_height
    base_z = 0.012
    _sub(head, "geom", class_="visual", type="cylinder", pos=(0, 0, base_z), size=(0.92 * rc, 0.006),
         material="mat_dark")
    _sub(head, "geom", class_="visual", type="ellipsoid", pos=(0, 0, base_z + 0.006),
         size=(rc, rc, 0.85 * hc), material="mat_head")
    _sub(head, "geom", class_="visual", type="cylinder", pos=(0, 0, base_z + 0.016), size=(1.002 * rc, 0.008),
         material="mat_accent")
    _sub(head, "geom", class_="visual", type="cylinder", pos=(0, 0, base_z + 0.03), size=(0.985 * rc, 0.004),
         material="mat_metal")
    eye_dir = np.array([1.0, 0.0, 0.55])
    eye_dir /= np.linalg.norm(eye_dir)
    eye_pos = np.array([0.0, 0.0, base_z + 0.006]) + eye_dir * np.array([rc, rc, 0.85 * hc]) * 0.97
    _sub(head, "geom", class_="visual", type="sphere", pos=eye_pos, size=(0.22 * rc,), material="mat_dark")
    _sub(head, "geom", class_="visual", type="sphere", pos=eye_pos + 0.08 * rc * eye_dir, size=(0.09 * rc,),
         rgba=(0.1, 0.1, 0.12, 1), material="mat_metal")
    _sub(head, "geom", class_="visual", type="capsule",
         fromto=(-0.2 * rc, 0.1 * rc, base_z + 0.8 * hc, -0.25 * rc, 0.12 * rc, base_z + 0.8 * hc + 0.10),
         size=(0.004,), material="mat_metal")
    for c in wheel_centers:
        c = np.asarray(c, dtype=float)
        top = np.array([0.55 * c[0], 0.55 * c[1], base_z])
        _sub(head, "geom", class_="visual", type="capsule", fromto=np.concatenate([top, c]), size=(0.007,),
             material="mat_metal")


# --------------------------------------------------------------------------- #
# Example 4: planar / decoupled ball balancer
# --------------------------------------------------------------------------- #
def build_planar_balancer(
    params: Optional[PlanarBalancerParams] = None,
    sim: Optional[SimulationParams] = None,
    extras: Optional[SceneExtras] = None,
) -> SceneBundle:
    """``1d``: pitch-plane slice (two wheels at +-alpha, x-z plane). ``2d``: 4 omniwheels, pitch & roll, no yaw."""
    p = params or PlanarBalancerParams()
    p.validate()
    sub = OmniwheelMode(p.wheel.mode) == OmniwheelMode.SUBROLLER
    sim = _default_sim(sim, sub)
    if sim.terrain.floor_slope_deg and p.variant == "1d" and sim.terrain.slope_axis == "x":
        raise ValueError("a 1D (x-z plane) balancer cannot use a floor sloped about the x axis")
    b = _new_builder(f"example4_planar_balancer_{p.variant}", sim)
    b.add_lights_and_cameras()
    b.add_floor(sim.terrain)
    R = p.ball.r2
    meta = SceneMetadata(
        example="planar_balancer",
        title=f"Example 4: {'1D planar-slice' if p.variant == '1d' else '2D decoupled'} ball balancer",
        description="Simplified balancer on the composite sphere for PID gain tuning before full 3D coupling.",
        timestep=sim.timestep, control_dt=0.01,
        omni_mode=OmniwheelMode(p.wheel.mode).value,
        control_mode=ControlMode(p.motor.control_mode).value,
    )
    nrm = _floor_normal(sim.terrain)
    ball_pos = np.array([0.0, 0.0, 0.0]) + nrm * R + np.array([0.0, 0.0, 1e-4])
    ball_body, ball_geom, ball_joints = _add_composite_ball(
        b, b.worldbody, p.ball, ball_pos, joint_kind="planar_xz" if p.variant == "1d" else "free")
    head_pos = ball_pos + np.array([0.0, 0.0, R])
    head = _sub(b.worldbody, "body", name=b.claim("head"), pos=head_pos)
    joints = dict(ball_joints)
    if p.variant == "1d":
        dofs = (("head_x", "slide", (1, 0, 0)), ("head_z", "slide", (0, 0, 1)), ("head_pitch", "hinge", (0, 1, 0)))
    else:
        dofs = (("head_x", "slide", (1, 0, 0)), ("head_y", "slide", (0, 1, 0)), ("head_z", "slide", (0, 0, 1)),
                ("head_roll", "hinge", (1, 0, 0)), ("head_pitch", "hinge", (0, 1, 0)))
    for jn, jt, ax in dofs:
        joints[jn] = b.claim(jn)
        _sub(head, "joint", name=joints[jn], type=jt, axis=ax)
    _sub(head, "inertial", pos=(0, 0, p.com_height), mass=p.body_mass, diaginertia=p.body_inertia)
    _sub(head, "geom", class_="visual", type="box", pos=(0, 0, p.com_height), size=(0.05, 0.05, p.com_height * 0.8),
         material="mat_head")
    _sub(head, "geom", class_="visual", type="sphere", pos=(0, 0, 2 * p.com_height), size=(0.03,),
         material="mat_accent")
    _sub(head, "geom", name=b.claim("head_collision"), type="box", pos=(0, 0, p.com_height + 0.04),
         size=(0.05, 0.05, 0.6 * p.com_height), contype=CollisionBits.CHASSIS[0],
         conaffinity=CollisionBits.CHASSIS[1], group=COLLISION_GROUP, density=0)
    imu_site, _ = _add_imu(b, head, "head", (0, 0, p.com_height))
    wheels: List[WheelMeta] = []
    if p.variant == "1d":
        # Pitch-plane slice of the ball-balancer: two driven wheels at +-alpha in the
        # x-z plane. Two contacts with x-components keep the head rotating about the
        # ball centre exactly like the 3/4-wheel head (a single top wheel would let
        # the head pivot on its contact - a different, double-inverted system).
        wp = dataclasses.replace(p.wheel, radius=p.wheel_radius, width=p.wheel_width, hub_mass=p.wheel_mass,
                                 mu_tangent=p.wheel_friction)
        alpha = math.radians(p.zenith_angle_deg)
        top = np.array([0.0, 0.0, R])
        for i, sgn in enumerate((1.0, -1.0)):
            n = np.array([sgn * math.sin(alpha), 0.0, math.cos(alpha)])
            t = np.array([math.cos(alpha), 0.0, -sgn * math.sin(alpha)])      # in-plane rolling direction
            center = (R + wp.radius) * n - top
            wheels.append(_add_omniwheel(
                b, head, "head", f"wheel{i}", center=center, normal=n, rolling_dir=t, wheel=wp, motor=p.motor,
                target_geoms=[ball_geom], roller_bits=CollisionBits.BALL_ROLLER,
                pair_geom_bits=CollisionBits.BALL_PAIR_WHEEL, index=i, azimuth=0.0 if sgn > 0 else math.pi,
            ))
            _sub(head, "geom", class_="visual", type="capsule",
                 fromto=np.concatenate([[0, 0, 0.012], center]), size=(0.006,), material="mat_metal")
        meta.init_qpos = {joints["head_pitch"]: math.radians(p.initial_tilt_deg)}
        J = [(-(R / wp.radius) * np.asarray(w.axle)).tolist() for w in wheels]
        meta.kinematics = {"wheel_radius": wp.radius, "ball_radius": R, "ball_to_wheel_jacobian": J,
                           "note": "both axles are +y: omega_w = -(R/r) (Omega_ball - Omega_head)_y"}
    else:
        hp = HeadUnitParams(num_wheels=4, zenith_angle_deg=p.zenith_angle_deg, azimuth_offset_deg=0.0,
                            wheel=p.wheel, motor=p.motor, chassis_mass=p.body_mass, com_height=p.com_height)
        geos = hp.wheel_geometry(R)
        for g in geos:
            wheels.append(_add_omniwheel(
                b, head, "head", f"wheel{g.index}", center=g.center, normal=g.normal, rolling_dir=g.rolling_dir,
                wheel=p.wheel, motor=p.motor, target_geoms=[ball_geom], roller_bits=CollisionBits.BALL_ROLLER,
                pair_geom_bits=CollisionBits.BALL_PAIR_WHEEL,
                index=g.index, azimuth=g.azimuth,
            ))
            _sub(head, "geom", class_="visual", type="capsule",
                 fromto=np.concatenate([[0, 0, 0.012], g.center]), size=(0.006,), material="mat_metal")
        meta.init_qpos = {joints["head_pitch"]: math.radians(p.initial_tilt_deg),
                          joints["head_roll"]: math.radians(0.5 * p.initial_tilt_deg)}
        meta.kinematics = {"wheel_radius": p.wheel.radius, "ball_radius": R,
                           "ball_to_wheel_jacobian": hp.ball_to_wheel_jacobian(R).tolist()}
    for kind in ("framepos", "framequat", "framelinvel", "frameangvel"):
        _sub(b.sensor, kind, name=b.claim(f"head_{kind[5:]}"), objtype="body", objname="head")
    meta.bodies = {"ball": "composite_ball", "head": "head"}
    meta.joints = joints
    meta.imu_site = imu_site
    meta.wheels = wheels
    meta.params = {"planar_balancer": params_to_dict(p), "sim": params_to_dict(sim),
                   "ball_summary": p.ball.summary()}
    return _finish(b, meta, extras)


# --------------------------------------------------------------------------- #
# Example 5: full BB-8
# --------------------------------------------------------------------------- #
def build_bb8(
    params: Optional[BB8Params] = None,
    sim: Optional[SimulationParams] = None,
    extras: Optional[SceneExtras] = None,
) -> SceneBundle:
    """Composite ball + 3/4-omniwheel head unit, magnet preload and optional internal counterweight."""
    p = params or BB8Params()
    p.validate()
    hp = p.head
    sub = OmniwheelMode(hp.wheel.mode) == OmniwheelMode.SUBROLLER
    sim = _default_sim(sim, sub)
    b = _new_builder("example5_bb8", sim)
    b.add_lights_and_cameras()
    b.add_floor(sim.terrain)
    R = p.ball.r2
    meta = SceneMetadata(
        example="bb8",
        title="Example 5: BB-8 composite ball-balancer",
        description=("Hollow + styrofoam + epoxy/sawdust ball driven by an omniwheel head unit; "
                     f"task mode: {p.task_mode}."),
        timestep=sim.timestep, control_dt=0.01, omni_mode=OmniwheelMode(hp.wheel.mode).value,
        control_mode=ControlMode(hp.motor.control_mode).value,
    )
    nrm = _floor_normal(sim.terrain)
    ball_pos = nrm * R + np.array([0.0, 0.0, 1e-4])
    ball_body, ball_geom, ball_joints = _add_composite_ball(b, b.worldbody, p.ball, ball_pos, "free")
    _sub(ball_body, "camera", name=b.claim("chase"), mode="trackcom", pos=(1.4, -1.4, 1.1),
         xyaxes=(0.707, 0.707, 0, -0.38, 0.38, 0.84))
    joints = dict(ball_joints)

    # Optional internal counterweight pendulum (inside the ball, hung from its centre).
    if hp.counterweight.enabled:
        cw = hp.counterweight
        cwb = _sub(ball_body, "body", name=b.claim("counterweight"), pos=(0, 0, 0))
        joints["counterweight_ball"] = b.claim("counterweight_ball")
        _sub(cwb, "joint", name=joints["counterweight_ball"], type="ball", damping=cw.joint_damping)
        i_cw = 0.4 * cw.mass * cw.radius**2
        _sub(cwb, "inertial", pos=(0, 0, -cw.offset), mass=cw.mass, diaginertia=(i_cw, i_cw, i_cw))
        _sub(cwb, "geom", class_="visual", type="sphere", pos=(0, 0, -cw.offset), size=(cw.radius,),
             material="mat_dark", group=2)
        _sub(cwb, "geom", class_="visual", type="capsule", fromto=(0, 0, 0, 0, 0, -cw.offset), size=(0.006,),
             material="mat_metal", group=2)
        _sub(cwb, "site", name=b.claim("counterweight_site"), pos=(0, 0, 0), size=(0.01,), group=4)

    # Head unit: tilted about the ball centre by the initial roll / pitch.
    roll, pitch = (math.radians(a) for a in p.initial_tilt_deg)
    q_tilt = _quat_mul(_axis_angle_quat((0, 1, 0), pitch), _axis_angle_quat((1, 0, 0), roll))
    head_pos = ball_pos + _rot_vec(q_tilt, (0, 0, R))
    head = _sub(b.worldbody, "body", name=b.claim("head"), pos=head_pos, quat=q_tilt)
    joints["head_free"] = b.claim("head_free")
    _sub(head, "freejoint", name=joints["head_free"])
    _sub(head, "inertial", pos=(0, 0, hp.com_height), mass=hp.chassis_mass, diaginertia=hp.chassis_inertia)
    geos = hp.wheel_geometry(R)
    _add_head_visuals(b, head, hp, [np.array(g.center) for g in geos])
    _sub(head, "geom", name=b.claim("head_collision"), type="ellipsoid",
         pos=(0, 0, 0.018 + 0.5 * hp.chassis_height),
         size=(hp.chassis_radius, hp.chassis_radius, 0.5 * hp.chassis_height), contype=CollisionBits.CHASSIS[0],
         conaffinity=CollisionBits.CHASSIS[1], group=COLLISION_GROUP, density=0)
    imu_site, _ = _add_imu(b, head, "head", (0, 0, hp.com_height))
    magnet_site = b.claim("head_magnet")
    _sub(head, "site", name=magnet_site, pos=(0, 0, hp.magnet.magnet_height), size=(0.012,), group=4,
         rgba=(0.9, 0.1, 0.1, 1))
    wheels = [
        _add_omniwheel(
            b, head, "head", f"wheel{g.index}", center=g.center, normal=g.normal, rolling_dir=g.rolling_dir,
            wheel=hp.wheel, motor=hp.motor, target_geoms=[ball_geom], roller_bits=CollisionBits.BALL_ROLLER,
                pair_geom_bits=CollisionBits.BALL_PAIR_WHEEL,
            index=g.index, azimuth=g.azimuth,
        )
        for g in geos
    ]
    for kind in ("framepos", "framequat", "framelinvel", "frameangvel"):
        _sub(b.sensor, kind, name=b.claim(f"head_{kind[5:]}"), objtype="body", objname="head")

    if p.target_marker:
        tgt = _sub(b.worldbody, "body", name=b.claim("target"), mocap=True, pos=(0, 0, 0.002))
        _sub(tgt, "geom", class_="visual", type="cylinder", size=(0.5 * R, 0.001), material="mat_target")
        _sub(tgt, "geom", class_="visual", type="capsule", fromto=(0, 0, 0.002, 0.6 * R, 0, 0.002), size=(0.006,),
             material="mat_target")
        meta.bodies["target"] = "target"

    if hp.magnet.enabled:
        target_body, target_site = ("counterweight", "counterweight_site") if hp.counterweight.enabled else (
            "composite_ball", "ball_center")
        meta.magnets = [MagnetLink(
            source_body="head", magnet_site=magnet_site, target_body=target_body, target_site=target_site,
            force=hp.magnet.force, model=hp.magnet.model, max_force=hp.magnet.max_force,
            nominal_distance=R + hp.magnet.magnet_height,
        )]

    meta.bodies.update({"ball": "composite_ball", "head": "head"})
    if hp.counterweight.enabled:
        meta.bodies["counterweight"] = "counterweight"
    meta.joints = joints
    meta.imu_site = imu_site
    meta.wheels = wheels
    meta.kinematics = {
        "ball_radius": R,
        "wheel_radius": hp.wheel.radius,
        "zenith_angle": hp.zenith_angle,
        "wheel_azimuths": hp.wheel_azimuths,
        "ball_to_wheel_jacobian": hp.ball_to_wheel_jacobian(R).tolist(),
        "note": "omega_wheels = J @ (Omega_ball - Omega_head) in head frame (no slip along rolling dirs).",
    }
    meta.params = {
        "bb8": params_to_dict(p),
        "sim": params_to_dict(sim),
        "ball_summary": p.ball.summary(),
        "head_summary": hp.summary(R),
        "task_mode": p.task_mode,
    }
    return _finish(b, meta, extras)


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ExampleInfo:
    key: str
    number: int
    title: str
    builder: Callable[..., SceneBundle]
    params_type: type


EXAMPLES: Dict[str, ExampleInfo] = {
    "inverted_pendulum": ExampleInfo("inverted_pendulum", 1, "Classic 2D & 3D inverted pendulum",
                                     build_inverted_pendulum, PendulumParams),
    "single_omniwheel": ExampleInfo("single_omniwheel", 2, "Single omniwheel on flat ground",
                                    build_single_omniwheel, SingleOmniwheelParams),
    "kiwi_drive": ExampleInfo("kiwi_drive", 3, "3-wheel kiwi holonomic omni-drive",
                              build_kiwi_drive, KiwiDriveParams),
    "planar_balancer": ExampleInfo("planar_balancer", 4, "1D/2D planar ball balancer",
                                   build_planar_balancer, PlanarBalancerParams),
    "bb8": ExampleInfo("bb8", 5, "Full BB-8 composite ball-balancer (hero scene)", build_bb8, BB8Params),
}


def _resolve_example(key: Union[str, int]) -> ExampleInfo:
    if isinstance(key, int) or (isinstance(key, str) and key.isdigit()):
        num = int(key)
        for info in EXAMPLES.values():
            if info.number == num:
                return info
        raise KeyError(f"no example number {num}")
    if key not in EXAMPLES:
        raise KeyError(f"unknown example {key!r}; choose from {list(EXAMPLES)}")
    return EXAMPLES[key]


def list_examples() -> List[Tuple[int, str, str]]:
    return [(i.number, i.key, i.title) for i in sorted(EXAMPLES.values(), key=lambda e: e.number)]


def build_example(
    key: Union[str, int],
    params: Any = None,
    sim: Optional[SimulationParams] = None,
    extras: Optional[SceneExtras] = None,
) -> SceneBundle:
    """Build example ``key`` (name or number 1-5) with optional parameter overrides."""
    info = _resolve_example(key)
    if params is not None and not isinstance(params, info.params_type):
        raise TypeError(f"{info.key} expects {info.params_type.__name__}, got {type(params).__name__}")
    return info.builder(copy.deepcopy(params) if params is not None else None, sim, extras)


def _set_omni_mode(params: Any, mode: OmniwheelMode) -> None:
    for attr in ("wheel",):
        if hasattr(params, attr):
            getattr(params, attr).mode = mode
    if hasattr(params, "head"):
        params.head.wheel.mode = mode


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Generate MuJoCo MJCF scenes for the BB-8 studio")
    ap.add_argument("--example", default="bb8", help="example name or number (1-5)")
    ap.add_argument("--list", action="store_true", help="list available examples")
    ap.add_argument("--omni-mode", choices=[m.value for m in OmniwheelMode], help="omniwheel model")
    ap.add_argument("--control-mode", choices=[m.value for m in ControlMode], help="wheel actuator type")
    ap.add_argument("--variant", help="pendulum: cartpole|spherical, planar balancer: 1d|2d")
    ap.add_argument("--magnet", type=float, help="enable BB-8 magnetic preload with this force [N]")
    ap.add_argument("--counterweight", type=float, help="enable BB-8 internal counterweight with this mass [kg]")
    ap.add_argument("--mesh", action="append", default=[], help="import a .stl/.obj mesh into the world")
    ap.add_argument("--mesh-scale", type=float, default=1.0)
    ap.add_argument("--slope", type=float, default=0.0, help="floor slope [deg] about y")
    ap.add_argument("--out", help="output directory for scene.xml + meshes")
    ap.add_argument("--print", action="store_true", help="print the XML")
    ap.add_argument("--check", type=float, default=0.0, metavar="SECONDS",
                    help="compile and simulate for SECONDS with zero control, then report")
    args = ap.parse_args(argv)

    if args.list:
        for num, key, title in list_examples():
            print(f"{num}. {key:<18} {title}")
        return 0

    info = _resolve_example(args.example)
    params = info.params_type()
    if args.omni_mode:
        _set_omni_mode(params, OmniwheelMode(args.omni_mode))
    if args.control_mode:
        motor = params.head.motor if hasattr(params, "head") else getattr(params, "motor", None)
        if motor is not None:
            motor.control_mode = ControlMode(args.control_mode)
    if args.variant and hasattr(params, "variant"):
        params.variant = args.variant
    if isinstance(params, BB8Params):
        if args.magnet is not None:
            params.head.magnet.enabled = True
            params.head.magnet.force = args.magnet
        if args.counterweight is not None:
            params.head.counterweight.enabled = True
            params.head.counterweight.mass = args.counterweight
    sim = None
    if args.slope:
        sub = OmniwheelMode(args.omni_mode or "contact_pair") == OmniwheelMode.SUBROLLER
        sim = _default_sim(None, sub)
        sim.terrain.floor_slope_deg = args.slope
    extras = SceneExtras(meshes=[MeshImportSpec(path=m, scale=args.mesh_scale, pos=(0.8, 0.0, 0.2), dynamic=True)
                                 for m in args.mesh]) if args.mesh else None

    bundle = info.builder(params, sim, extras)
    if args.print:
        print(bundle.xml)
    if args.out:
        path = bundle.write(args.out)
        print(f"wrote {path} ({len(bundle.assets)} mesh assets)")
    model, data, stepper = bundle.make()
    print(f"{bundle.meta.title}: nq={model.nq} nv={model.nv} nu={model.nu} nbody={model.nbody} "
          f"ngeom={model.ngeom} nsensor={model.nsensor} dt={model.opt.timestep}")
    ball = getattr(params, "ball", None)
    if ball is not None:
        print(ball.report())
    if args.check > 0:
        steps = int(round(args.check / model.opt.timestep))
        stepper.step(steps)
        ok = bool(np.all(np.isfinite(data.qpos)))
        print(f"simulated {data.time:.3f} s: finite={ok}, ncon={data.ncon}, warnings="
              f"{[int(w.number) for w in data.warning if w.number]}")
        for role, body in bundle.meta.bodies.items():
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body)
            if bid >= 0 and model.body_mocapid[bid] < 0:
                print(f"  {role:<14} pos = {np.round(data.xpos[bid], 4)}")
    stepper.close()
    return 0


if __name__ == "__main__":
    sys.exit(_main())

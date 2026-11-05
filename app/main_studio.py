"""BB-8 Studio: FastAPI + WebSockets backend for the browser (Three.js) simulator (Module 6).

Run::

    python -m app.main_studio                 # http://localhost:8000
    python -m app.main_studio --host 0.0.0.0 --port 8000 --runs runs

The physics runs here (MuJoCo, real time, 500 Hz physics / 100 Hz control).
The browser renders the scene from a *scene descriptor* (every visual geom,
mesh vertex buffers and heightfields of the compiled model) and receives body
poses, contact / push force arrows and telemetry over ``/ws`` at ~30 Hz.

Features: the 5 example scenes; live composite-ball editing (mass / inertia
preview, rebuild); omniwheel / drive / variant options; Manual (WASD/QE),
Cascaded PID, LQR and trained RL policy control with WASD steering of the
balance target; push disturbances in newtons; sensor noise / latency /
pull-out-curve toggles; STL/OBJ upload and primitive props; CSV/JSON logging;
background PPO/SAC training with live reward streaming and policy hot-loading.
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import csv
import io
import json
import math
import os
import re
import sys
import threading
import time
from collections import deque
from typing import Any, Deque, Dict, List, Optional

import mujoco
import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect  # noqa: E402
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402

from config.composite_ball_params import (  # noqa: E402
    BB8Params,
    CompositeBallParams,
    ControlMode,
    HardwareProfile,
    OmniwheelMode,
    PendulumParams,
    PlanarBalancerParams,
    SimulationParams,
    params_from_dict,
    params_to_dict,
)
from controllers.classical import (  # noqa: E402
    BallBotLQR,
    CartPolePID,
    CascadedPID,
    CascadedPIDGains,
    FullStateLQR,
    KiwiMixer,
    PIDGains,
)
from envs.ball_balancer_env import BallBalancerEnv, EnvConfig  # noqa: E402
from models.io_bus import ActuatorBus, SensorBus  # noqa: E402
from models.mjcf_generator import (  # noqa: E402
    EXAMPLES,
    MeshImportSpec,
    PrimitiveSpec,
    SceneBundle,
    SceneExtras,
    build_example,
)
from models.runtime import SceneStepper  # noqa: E402
from training.train_rl import MetricsHub, PolicyRunner, TrainingConfig, TrainingManager  # noqa: E402

STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
BALANCERS = ("bb8", "planar_balancer")
MODES = ("manual", "pid", "lqr", "rl", "off")
MESH_EXTENSIONS = (".stl", ".obj", ".ply", ".off")


# --------------------------------------------------------------------------- #
# Scene descriptor for the Three.js client
# --------------------------------------------------------------------------- #
_GEOM_TYPES = {
    int(mujoco.mjtGeom.mjGEOM_PLANE): "plane",
    int(mujoco.mjtGeom.mjGEOM_HFIELD): "hfield",
    int(mujoco.mjtGeom.mjGEOM_SPHERE): "sphere",
    int(mujoco.mjtGeom.mjGEOM_CAPSULE): "capsule",
    int(mujoco.mjtGeom.mjGEOM_ELLIPSOID): "ellipsoid",
    int(mujoco.mjtGeom.mjGEOM_CYLINDER): "cylinder",
    int(mujoco.mjtGeom.mjGEOM_BOX): "box",
    int(mujoco.mjtGeom.mjGEOM_MESH): "mesh",
}


def _r(values: Any, nd: int = 5) -> List[float]:
    return [round(float(v), nd) for v in np.asarray(values).ravel()]


def describe_scene(model: mujoco.MjModel, bundle: SceneBundle) -> Dict[str, Any]:
    """Everything the browser needs to build the 3D scene from the compiled model."""
    bodies = [{"id": i, "name": model.body(i).name, "parent": int(model.body_parentid[i]),
               "mocap": bool(model.body_mocapid[i] >= 0)} for i in range(model.nbody)]
    default_rgba = np.array([0.5, 0.5, 0.5, 1.0], dtype=np.float32)
    geoms, meshes, hfields = [], {}, {}
    for g in range(model.ngeom):
        gtype = _GEOM_TYPES.get(int(model.geom_type[g]))
        if gtype is None:
            continue
        rgba = model.geom_rgba[g].copy()
        mat = int(model.geom_matid[g])
        mat_name = model.material(mat).name if mat >= 0 else None
        if mat >= 0 and np.allclose(rgba, default_rgba):
            rgba = model.mat_rgba[mat]
        entry = {
            "id": g, "name": model.geom(g).name, "body": int(model.geom_bodyid[g]), "type": gtype,
            "size": _r(model.geom_size[g]), "pos": _r(model.geom_pos[g]), "quat": _r(model.geom_quat[g]),
            "rgba": _r(rgba, 3), "group": int(model.geom_group[g]), "material": mat_name,
            "collides": bool(model.geom_contype[g] or model.geom_conaffinity[g]),
        }
        if gtype == "mesh":
            mid = int(model.geom_dataid[g])
            entry["mesh"] = mid
            if mid not in meshes:
                va, vn = int(model.mesh_vertadr[mid]), int(model.mesh_vertnum[mid])
                fa, fn = int(model.mesh_faceadr[mid]), int(model.mesh_facenum[mid])
                meshes[mid] = {"vertices": _r(model.mesh_vert[va:va + vn], 5),
                               "faces": [int(x) for x in model.mesh_face[fa:fa + fn].ravel()]}
        if gtype == "hfield":
            hid = int(model.geom_dataid[g])
            entry["hfield"] = hid
            if hid not in hfields:
                nr, nc = int(model.hfield_nrow[hid]), int(model.hfield_ncol[hid])
                adr = int(model.hfield_adr[hid])
                hfields[hid] = {"nrow": nr, "ncol": nc, "size": _r(model.hfield_size[hid]),
                                "data": _r(model.hfield_data[adr:adr + nr * nc], 4)}
        geoms.append(entry)
    meta = bundle.meta
    return {
        "example": meta.example, "title": meta.title, "description": meta.description,
        "bodies": bodies, "geoms": geoms, "meshes": {str(k): v for k, v in meshes.items()},
        "hfields": {str(k): v for k, v in hfields.items()},
        "wheels": [{"name": w.hub_body, "frame_site": w.frame_site} for w in meta.wheels],
        "actuators": meta.actuators, "omni_mode": meta.omni_mode, "control_mode": meta.control_mode,
        "focus_body": meta.bodies.get("ball") or meta.bodies.get("chassis") or meta.bodies.get("cart"),
        "timestep": float(model.opt.timestep),
    }


# --------------------------------------------------------------------------- #
# Session
# --------------------------------------------------------------------------- #
def _safe_filename(name: str) -> str:
    base = os.path.basename(name)
    stem, ext = os.path.splitext(base)
    return (re.sub(r"[^A-Za-z0-9_-]+", "_", stem).strip("_") or "mesh") + ext.lower()


class StudioSession:
    """Owns the live simulation; all public methods are thread-safe."""

    def __init__(self, runs_dir: str = "runs", upload_dir: Optional[str] = None) -> None:
        self.lock = threading.RLock()
        self.runs_dir = os.path.abspath(runs_dir)
        self.upload_dir = os.path.abspath(upload_dir or os.path.join(self.runs_dir, "uploads"))
        self.example = "bb8"
        self.params: Any = BB8Params()
        self.sim = SimulationParams()
        self.hw = HardwareProfile()
        self.extras = SceneExtras()
        self.mode = "pid"
        self.pid_gains: Optional[CascadedPIDGains] = None
        self.lqr_q: Dict[str, float] = {}
        self.policy: Optional[PolicyRunner] = None
        self.policy_path: Optional[str] = None
        self.keys: set = set()
        self.teleop_speed = 0.3
        self.teleop_yaw = 0.6
        self.manual_gain = 0.5
        self.paused = False
        self.speed = 1.0
        self.recording = False
        self.record: List[Dict[str, Any]] = []
        self.telemetry: Deque[Dict[str, Any]] = deque(maxlen=6000)
        self._pending: Deque[Dict[str, Any]] = deque(maxlen=2000)
        self.events: Deque[Dict[str, Any]] = deque(maxlen=2000)
        self.hub = MetricsHub()
        self._event_seq = 0
        self.hub.subscribe(self._on_train_event)
        self.trainer: Optional[TrainingManager] = None
        self.version = 0
        self.fallen = False
        self.message = ""
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self.rebuild()

    def _on_train_event(self, ev: Dict[str, Any]) -> None:
        self._event_seq += 1
        self.events.append({"seq": self._event_seq, "type": "train", "event": ev})

    def events_since(self, seq: int) -> List[Dict[str, Any]]:
        return [e for e in list(self.events) if e["seq"] > seq]

    @property
    def event_seq(self) -> int:
        return self._event_seq

    # ------------------------------------------------------------- building
    @property
    def is_balancer(self) -> bool:
        return self.example in BALANCERS

    def _env_scene(self) -> str:
        if self.example == "bb8":
            return "bb8"
        return "planar_1d" if self.params.variant == "1d" else "planar_2d"

    def rebuild(self) -> None:
        with self.lock:
            sim = copy.deepcopy(self.sim)
            sim.timestep = self.hw.loop.physics_timestep
            if self.is_balancer:
                motor = self.params.head.motor if self.example == "bb8" else self.params.motor
                self.hw.motor = motor
            bundle = build_example(self.example, self.params, sim, self.extras)
            self.bundle = bundle
            self.env: Optional[BallBalancerEnv] = None
            if self.is_balancer:
                cfg = EnvConfig(scene=self._env_scene(), hardware=copy.deepcopy(self.hw), episode_seconds=1e9,
                                task="station_keeping")
                if self.example == "bb8":
                    cfg.bb8 = copy.deepcopy(self.params)
                else:
                    cfg.planar = copy.deepcopy(self.params)
                cfg.randomization.enabled = False
                cfg.pushes.enabled = False
                cfg.max_tilt_deg = 1e9           # the studio keeps simulating after a fall
                cfg.max_ball_distance = 1e9
                cfg.initial_tilt_deg = 0.0
                cfg.initial_rate = 0.0
                self.env = BallBalancerEnv(cfg, bundle=bundle)
                self.obs, _ = self.env.reset(seed=0)
                self.model, self.data, self.stepper = self.env.model, self.env.data, self.env.stepper
                self.control_dt = self.env.control_dt
                self.frame_skip = self.env.frame_skip
            else:
                self.model, self.data, self.stepper = bundle.make()
                self.frame_skip = max(1, int(round(self.hw.loop.control_dt / self.model.opt.timestep)))
                self.control_dt = self.frame_skip * self.model.opt.timestep
                hw = copy.deepcopy(self.hw)
                motor = getattr(self.params, "motor", None)
                if motor is not None:
                    hw.motor = motor
                else:
                    hw.torque_speed_curve = False
                self.actuators = ActuatorBus(self.model, bundle.meta, hw)
                self.sensors = SensorBus(self.model, bundle.meta, hw) if bundle.meta.imu_site else None
                self.physics_step = 0
            self.descriptor = describe_scene(self.model, bundle)
            self.kiwi = KiwiMixer.from_meta(bundle.meta) if self.example == "kiwi_drive" else None
            self.telemetry.clear()
            self._pending.clear()
            self.fallen = False
            self.version += 1
            self._design_controller()

    def _design_controller(self) -> None:
        self.controller = None
        mode = self.mode
        if mode in ("manual", "off"):
            return
        if mode == "rl":
            if not self.is_balancer:
                raise ValueError("RL policies are available for the ball-balancer scenes")
            if self.policy is None:
                raise ValueError("load a policy first")
            if self.policy.obs_dim != self.env.observation_space.shape[0] or self.policy.act_dim != self.env.n_act:
                raise ValueError(f"policy expects obs {self.policy.obs_dim}/act {self.policy.act_dim}, scene has "
                                 f"{self.env.observation_space.shape[0]}/{self.env.n_act}")
            return
        if self.is_balancer:
            if mode == "pid":
                self.controller = CascadedPID(self.model, self.bundle.meta, copy.deepcopy(self.pid_gains),
                                              dt=self.control_dt)
            else:
                self.controller = BallBotLQR.design(self.bundle, self.hw, model=self.bundle.compile(),
                                                    q_weights=self.lqr_q or None)
        elif self.example == "inverted_pendulum":
            if mode == "lqr":
                self.controller = FullStateLQR.design(self.bundle, frame_skip=self.frame_skip,
                                                      model=self.bundle.compile())
            elif self.params.variant == "cartpole":
                self.controller = CartPolePID(self.model, self.bundle.meta, dt=self.control_dt)
            else:
                raise ValueError("PID is available for the cart-pole variant; use LQR for the spherical pendulum")
        else:
            raise ValueError(f"{mode} control is not available for {self.example}; use manual")

    # ------------------------------------------------------------- settings
    def set_scene(self, example: str, omni_mode: Optional[str] = None, variant: Optional[str] = None,
                  control_mode: Optional[str] = None) -> None:
        if example not in EXAMPLES:
            raise ValueError(f"unknown example {example}")
        with self.lock:
            params = EXAMPLES[example].params_type()
            if omni_mode:
                for obj in (getattr(params, "wheel", None), getattr(getattr(params, "head", None), "wheel", None)):
                    if obj is not None:
                        obj.mode = OmniwheelMode(omni_mode)
            if variant and hasattr(params, "variant"):
                params.variant = variant
            if control_mode:
                motor = params.head.motor if hasattr(params, "head") else getattr(params, "motor", None)
                if motor is not None:
                    motor.control_mode = ControlMode(control_mode)
            params.validate()
            self.example, self.params = example, params
            self.sim = SimulationParams()
            self.extras = SceneExtras()
            self.mode = {"bb8": "pid", "planar_balancer": "pid", "inverted_pendulum": "lqr"}.get(example, "manual")
            self.rebuild()

    def set_params(self, params: Optional[Dict[str, Any]] = None, sim: Optional[Dict[str, Any]] = None,
                   hardware: Optional[Dict[str, Any]] = None) -> None:
        with self.lock:
            new_params = params_from_dict(EXAMPLES[self.example].params_type, params) if params is not None else None
            new_sim = params_from_dict(SimulationParams, sim) if sim is not None else None
            new_hw = params_from_dict(HardwareProfile, hardware) if hardware is not None else None
            for obj in (new_params, new_sim, new_hw):
                if obj is not None:
                    obj.validate()
            old = (self.params, self.sim, self.hw)
            self.params = new_params or self.params
            self.sim = new_sim or self.sim
            self.hw = new_hw or self.hw
            try:
                self.rebuild()
            except Exception:
                self.params, self.sim, self.hw = old
                self.rebuild()
                raise

    def set_ball(self, ball: Dict[str, Any]) -> Dict[str, Any]:
        with self.lock:
            if not hasattr(self.params, "ball"):
                raise ValueError("this scene has no composite ball")
            merged = params_to_dict(self.params.ball)
            merged.update(ball)
            new_ball = params_from_dict(CompositeBallParams, merged)
            new_ball.validate()
            params = copy.deepcopy(self.params)
            params.ball = new_ball
            self.set_params(params=params_to_dict(params))
            return new_ball.summary()

    def set_hardware(self, noise: Optional[bool] = None, latency_ms: Optional[float] = None,
                     latency_enabled: Optional[bool] = None, torque_curve: Optional[bool] = None) -> None:
        with self.lock:
            hw = copy.deepcopy(self.hw)
            if noise is not None:
                hw.noise_enabled = bool(noise)
            if latency_ms is not None:
                hw.latency.delay = float(latency_ms) / 1000.0
            if latency_enabled is not None:
                hw.latency_enabled = bool(latency_enabled)
            if torque_curve is not None:
                hw.torque_speed_curve = bool(torque_curve)
            hw.validate()
            self.hw = hw
            self.rebuild()

    def set_mode(self, mode: str, pid: Optional[Dict[str, float]] = None, lqr: Optional[Dict[str, float]] = None) -> None:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        with self.lock:
            if pid:
                g = copy.deepcopy(self.pid_gains) or (
                    CascadedPIDGains.torque_mode_defaults() if self.bundle.meta.control_mode == "motor"
                    else CascadedPIDGains())
                for k in ("kp", "ki", "kd"):
                    if k in pid:
                        setattr(g.inner, k, float(pid[k]))
                for k in ("outer_pos", "outer_vel", "outer_vel_ki", "max_tilt"):
                    if k in pid:
                        setattr(g, k, float(pid[k]))
                self.pid_gains = g
            if lqr:
                self.lqr_q.update({k: float(v) for k, v in lqr.items()})
            previous = self.mode
            self.mode = mode
            try:
                self._design_controller()
            except Exception:
                self.mode = previous
                self._design_controller()
                raise

    def pid_gains_dict(self) -> Dict[str, float]:
        g = self.pid_gains or (CascadedPIDGains.torque_mode_defaults() if self.bundle.meta.control_mode == "motor"
                               else CascadedPIDGains())
        return {"kp": g.inner.kp, "ki": g.inner.ki, "kd": g.inner.kd, "outer_pos": g.outer_pos,
                "outer_vel": g.outer_vel, "outer_vel_ki": g.outer_vel_ki, "max_tilt": g.max_tilt}

    def load_policy(self, path: str) -> Dict[str, Any]:
        path = os.path.abspath(path)
        if not (path.endswith(".onnx") or path.endswith(".zip")) or not os.path.isfile(path):
            raise ValueError("policy must be an existing .onnx or .zip file")
        if not path.startswith(self.runs_dir + os.sep):
            raise ValueError("policies are loaded from the runs directory only")
        runner = PolicyRunner.load(path)
        with self.lock:
            old = (self.policy, self.policy_path, self.mode)
            self.policy, self.policy_path = runner, path
            try:
                self.set_mode("rl")
            except Exception:
                self.policy, self.policy_path, self.mode = old
                raise
        return {"path": path, "obs_dim": runner.obs_dim, "act_dim": runner.act_dim}

    def push(self, body: str, force: List[float], duration: float) -> None:
        with self.lock:
            if mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body) < 0:
                raise ValueError(f"unknown body {body}")
            if not (0.0 < duration <= 2.0) or len(force) != 3 or np.linalg.norm(force) > 500:
                raise ValueError("push needs 3 force components (|F| <= 500 N) and 0 < duration <= 2 s")
            self.stepper.push(body, force, duration)

    def reset(self) -> None:
        with self.lock:
            if self.env is not None:
                self.obs, _ = self.env.reset(seed=0)
            else:
                self.bundle.reset(self.model, self.data)
                self.stepper.reset_state()
                self.actuators.reset()
                if self.sensors:
                    self.sensors.reset()
                self.physics_step = 0
            if self.controller is not None and hasattr(self.controller, "reset"):
                self.controller.reset()
            self.fallen = False

    # ------------------------------------------------------------- control
    def _key_axes(self) -> np.ndarray:
        k = self.keys
        return np.array([(("w" in k) - ("s" in k)), (("a" in k) - ("d" in k)), (("q" in k) - ("e" in k))],
                        dtype=float)

    def control_step(self) -> None:
        """Advance one control period (frame_skip physics steps)."""
        with self.lock:
            axes = self._key_axes()
            if self.env is not None:
                self._balancer_step(axes)
            else:
                self._generic_step(axes)

    def _balancer_step(self, axes: np.ndarray) -> None:
        env = self.env
        if self.mode != "manual":
            v = axes[:2] * self.teleop_speed
            if env.bbm.variant == "1d":
                v[1] = 0.0
            env.target.command(v, axes[2] * self.teleop_yaw if env.bbm.variant == "bb8" else 0.0)
        if self.mode in ("pid", "lqr"):
            ctrl = self.controller.compute(env.last_state, env.target)
            self.obs, reward, _, _, info = env.step_ctrl(ctrl)
        elif self.mode == "rl":
            self.obs, reward, _, _, info = env.step(self.policy(self.obs))
        elif self.mode == "manual":
            # W/S roll the ball toward +-x, A/D toward +-y, Q/E spin
            a = np.array([-axes[1], axes[0], -axes[2]]) * self.manual_gain
            self.obs, reward, _, _, info = env.step(a)
        else:
            self.obs, reward, _, _, info = env.step_ctrl(np.zeros(self.model.nu))
        s = env.last_state
        tilt = math.degrees(max(abs(s.roll), abs(s.pitch)))
        self.fallen = self.fallen or tilt > 35.0
        row = {
            "t": round(float(self.data.time), 4),
            "roll_deg": math.degrees(s.roll), "pitch_deg": math.degrees(s.pitch),
            "roll_rate": s.roll_rate, "pitch_rate": s.pitch_rate, "yaw_rate": s.yaw_rate,
            "ball_vx": s.vx, "ball_vy": s.vy, "target_vx": float(env.target.velocity[0]),
            "target_vy": float(env.target.velocity[1]), "target_wz": float(env.target.yaw_rate),
            "pos_err_x": s.ex, "pos_err_y": s.ey,
            "reward": float(reward),
            "reward_terms": {k: round(float(v), 5) for k, v in info["reward_terms"].items()},
            "wheel_torque": _r(self.data.actuator_force[env.actuators.wheel_act], 4),
            "wheel_speed": _r(env.last_reading.wheel_vel, 3),
            "ctrl": _r(self.data.ctrl, 4),
        }
        self._record(row)

    def _generic_step(self, axes: np.ndarray) -> None:
        m, d = self.model, self.data
        ctrl = np.zeros(m.nu)
        if self.example == "inverted_pendulum":
            if self.mode in ("pid", "lqr"):
                ctrl = self.controller.compute_full(d)
            elif self.mode == "manual":
                lim = m.actuator_ctrlrange[:, 1]
                ctrl[0] = axes[0] * lim[0] * 0.5
                if m.nu > 1:
                    ctrl[1] = axes[1] * lim[1] * 0.5
        elif self.mode == "manual":
            max_w = getattr(self.params, "motor").max_wheel_speed
            if self.kiwi is not None:
                w = self.kiwi.scaled_wheel_speeds(axes[0] * self.teleop_speed, axes[1] * self.teleop_speed,
                                                  axes[2] * 1.5, max_w)
            else:
                w = np.array([axes[0] * self.teleop_speed / self.bundle.meta.wheels[0].radius])
            ctrl[self.actuators.wheel_act] = w
        self.actuators.submit(ctrl, self.physics_step)
        for _ in range(self.frame_skip):
            self.actuators.apply(d, self.physics_step)
            self.stepper.step(1)
            self.physics_step += 1
        row: Dict[str, Any] = {"t": round(float(d.time), 4), "ctrl": _r(d.ctrl, 4),
                               "actuator_force": _r(d.actuator_force, 4)}
        if self.sensors is not None:
            r = self.sensors.read(d)
            yaw, pitch, roll = r.euler
            row.update(roll_deg=math.degrees(roll), pitch_deg=math.degrees(pitch), gyro=_r(r.gyro, 4),
                       wheel_speed=_r(r.wheel_vel, 3))
        if self.example == "inverted_pendulum":
            row["qpos"] = _r(d.qpos, 4)
        else:
            cid = m.body(self.bundle.meta.bodies["chassis"]).id
            vel = np.zeros(6)
            mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, cid, vel, 1)
            row.update(body_vx=float(vel[3]), body_vy=float(vel[4]), body_wz=float(vel[2]))
        self._record(row)

    def _record(self, row: Dict[str, Any]) -> None:
        self.telemetry.append(row)
        self._pending.append(row)
        if self.recording:
            self.record.append(row)

    # ------------------------------------------------------------- streaming
    def frame(self) -> Dict[str, Any]:
        with self.lock:
            m, d = self.model, self.data
            arrows = []
            buf = np.zeros(6)
            for i in range(d.ncon):
                c = d.contact[i]
                mujoco.mj_contactForce(m, d, i, buf)
                f_world = c.frame.reshape(3, 3).T @ buf[:3]
                if np.linalg.norm(f_world) > 0.2:
                    arrows.append({"kind": "contact", "p": _r(c.pos, 4), "f": _r(f_world, 3)})
            for p in self.stepper.pushes:
                arrows.append({"kind": "push", "p": _r(d.xipos[p.body_id], 4), "f": _r(p.force, 3)})
            telemetry = list(self._pending)
            self._pending.clear()
            frame = {
                "type": "frame", "version": self.version, "t": round(float(d.time), 4),
                "xpos": _r(d.xpos, 4), "xquat": _r(d.xquat, 5), "arrows": arrows,
                "telemetry": telemetry, "mode": self.mode, "paused": self.paused, "fallen": self.fallen,
                "recording": self.recording, "message": self.message,
            }
            if self.env is not None:
                frame["target"] = _r(self.env.target.position, 4)
        return frame

    def status(self) -> Dict[str, Any]:
        with self.lock:
            out = {
                "example": self.example, "mode": self.mode, "paused": self.paused, "speed": self.speed,
                "version": self.version, "params": params_to_dict(self.params), "sim": params_to_dict(self.sim),
                "hardware": params_to_dict(self.hw), "pid": self.pid_gains_dict() if self.is_balancer else None,
                "lqr_q": self.lqr_q, "policy": self.policy_path,
                "modes": self._available_modes(), "teleop_speed": self.teleop_speed,
                "recording": self.recording, "record_rows": len(self.record),
                "training": self.trainer.status() if self.trainer else None,
            }
            if hasattr(self.params, "ball"):
                out["ball_summary"] = self.params.ball.summary()
            if self.is_balancer and self.mode == "lqr" and self.controller is not None:
                out["lqr"] = {"spectral_radius": self.controller.design_data.spectral_radius,
                              "states": self.controller.design_data.state_names}
            return out

    def _available_modes(self) -> List[str]:
        if self.is_balancer:
            return ["manual", "pid", "lqr", "rl", "off"]
        if self.example == "inverted_pendulum":
            return ["manual", "lqr", "off"] + (["pid"] if self.params.variant == "cartpole" else [])
        return ["manual", "off"]

    # ------------------------------------------------------------- loop
    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="studio-sim", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)

    def _loop(self) -> None:
        next_t = time.perf_counter()
        while not self._stop.is_set():
            if self.paused:
                time.sleep(0.02)
                next_t = time.perf_counter()
                continue
            try:
                self.control_step()
                self.message = ""
            except Exception as exc:          # keep the server alive, show the problem in the UI
                self.message = f"{type(exc).__name__}: {exc}"
                with self.lock:
                    self.mode = "off"
                    self.controller = None
            next_t += self.control_dt / max(self.speed, 1e-3)
            delay = next_t - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            elif delay < -0.25:               # fell behind: do not try to catch up
                next_t = time.perf_counter()

    # ------------------------------------------------------------- logging
    def log_rows(self) -> List[Dict[str, Any]]:
        with self.lock:
            return list(self.record) if self.record else list(self.telemetry)

    def log_csv(self) -> str:
        rows = self.log_rows()
        flat_rows = []
        for row in rows:
            flat: Dict[str, Any] = {}
            for k, v in row.items():
                if isinstance(v, dict):
                    for kk, vv in v.items():
                        flat[f"{k}.{kk}"] = vv
                elif isinstance(v, list):
                    for i, vv in enumerate(v):
                        flat[f"{k}[{i}]"] = vv
                else:
                    flat[k] = v
            flat_rows.append(flat)
        cols: List[str] = []
        for r in flat_rows:
            for k in r:
                if k not in cols:
                    cols.append(k)
        out = io.StringIO()
        w = csv.DictWriter(out, fieldnames=cols)
        w.writeheader()
        w.writerows(flat_rows)
        return out.getvalue()

    # ------------------------------------------------------------- assets
    def add_mesh(self, filename: str, blob: bytes, scale: float, parent: str, dynamic: bool,
                 pos: List[float], density: float) -> Dict[str, Any]:
        name = _safe_filename(filename)
        if not name.endswith(MESH_EXTENSIONS):
            raise ValueError(f"mesh must be one of {MESH_EXTENSIONS}")
        if len(blob) > 50 * 1024 * 1024:
            raise ValueError("mesh file too large (50 MB max)")
        os.makedirs(self.upload_dir, exist_ok=True)
        path = os.path.join(self.upload_dir, name)
        with open(path, "wb") as fh:
            fh.write(blob)
        spec = MeshImportSpec(path=path, name=os.path.splitext(name)[0] + f"_{len(self.extras.meshes)}",
                              scale=scale, parent=parent, dynamic=dynamic, pos=tuple(pos), density=density)
        with self.lock:
            self.extras.meshes.append(spec)
            try:
                self.rebuild()
            except Exception:
                self.extras.meshes.pop()
                self.rebuild()
                raise
            return self.bundle.meta.imported_meshes[-1]

    def add_primitive(self, spec: Dict[str, Any]) -> None:
        prim = params_from_dict(PrimitiveSpec, spec)
        prim.validate()
        with self.lock:
            self.extras.primitives.append(prim)
            try:
                self.rebuild()
            except Exception:
                self.extras.primitives.pop()
                self.rebuild()
                raise

    def clear_extras(self) -> None:
        with self.lock:
            self.extras = SceneExtras()
            self.rebuild()

    # ------------------------------------------------------------- training
    def start_training(self, algo: str = "PPO", steps: int = 1_000_000, n_envs: int = 4, vec_env: str = "subproc",
                       task: str = "station_keeping") -> Dict[str, Any]:
        with self.lock:
            if not self.is_balancer:
                raise ValueError("training is available for the ball-balancer scenes")
            if self.trainer is not None and self.trainer.state == "running":
                raise ValueError("training already running")
            env_cfg = EnvConfig(scene=self._env_scene(), task=task, hardware=copy.deepcopy(self.hw))
            if self.example == "bb8":
                env_cfg.bb8 = copy.deepcopy(self.params)
            else:
                env_cfg.planar = copy.deepcopy(self.params)
            cfg = TrainingConfig(algo=algo, total_timesteps=int(steps), n_envs=int(n_envs), vec_env=vec_env,
                                 env=env_cfg, out_dir=self.runs_dir,
                                 checkpoint_every=max(10_000, int(steps) // 10))
            self.trainer = TrainingManager(cfg, self.hub)
            self.trainer.start(background=True)
            return self.trainer.status()

    def stop_training(self) -> Dict[str, Any]:
        if self.trainer is None:
            raise ValueError("no training run")
        self.trainer.stop(wait=False)
        return self.trainer.status()

    def list_policies(self) -> List[Dict[str, Any]]:
        out = []
        if not os.path.isdir(self.runs_dir):
            return out
        for root, _, files in os.walk(self.runs_dir):
            for f in files:
                if f.endswith(".onnx") or (f.endswith(".zip")):
                    p = os.path.join(root, f)
                    out.append({"path": p, "name": os.path.relpath(p, self.runs_dir), "mtime": os.path.getmtime(p)})
        return sorted(out, key=lambda e: -e["mtime"])


# --------------------------------------------------------------------------- #
# FastAPI app
# --------------------------------------------------------------------------- #
def create_app(session: Optional[StudioSession] = None, autostart: bool = True) -> FastAPI:
    from contextlib import asynccontextmanager

    sess = session or StudioSession()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if autostart:
            sess.start()
        yield
        sess.stop()
        if sess.trainer is not None and sess.trainer.state == "running":
            sess.trainer.stop(wait=False)

    app = FastAPI(title="BB-8 Studio", lifespan=lifespan)
    app.state.session = sess

    def guard(fn, *args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except (ValueError, KeyError, TypeError) as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    @app.get("/")
    def index():
        return FileResponse(os.path.join(STATIC_DIR, "index.html"))

    @app.get("/api/examples")
    def examples():
        return [{"key": i.key, "number": i.number, "title": i.title} for i in
                sorted(EXAMPLES.values(), key=lambda e: e.number)]

    @app.get("/api/scene")
    def scene():
        with sess.lock:
            return JSONResponse({**sess.descriptor, "version": sess.version})

    @app.post("/api/scene")
    def set_scene(body: Dict[str, Any]):
        guard(sess.set_scene, body.get("example", "bb8"), body.get("omni_mode"), body.get("variant"),
              body.get("control_mode"))
        return sess.status()

    @app.get("/api/status")
    def status():
        return sess.status()

    @app.post("/api/params")
    def set_params(body: Dict[str, Any]):
        guard(sess.set_params, body.get("params"), body.get("sim"), body.get("hardware"))
        return sess.status()

    @app.post("/api/ball/preview")
    def ball_preview(body: Dict[str, Any]):
        def run():
            base = params_to_dict(getattr(sess.params, "ball", CompositeBallParams()))
            base.update(body)
            ball = params_from_dict(CompositeBallParams, base)
            ball.validate()
            return {**ball.summary(), "report": ball.report()}
        return guard(run)

    @app.post("/api/ball/apply")
    def ball_apply(body: Dict[str, Any]):
        return guard(sess.set_ball, body)

    @app.post("/api/hardware")
    def hardware(body: Dict[str, Any]):
        guard(sess.set_hardware, body.get("noise"), body.get("latency_ms"), body.get("latency_enabled"),
              body.get("torque_curve"))
        return sess.status()

    @app.post("/api/controller")
    def controller(body: Dict[str, Any]):
        guard(sess.set_mode, body.get("mode", sess.mode), body.get("pid"), body.get("lqr"))
        return sess.status()

    @app.post("/api/teleop")
    def teleop(body: Dict[str, Any]):
        with sess.lock:
            if "keys" in body:
                sess.keys = {str(k).lower() for k in body["keys"]} & set("wasdqe")
            if "speed" in body:
                sess.teleop_speed = float(np.clip(body["speed"], 0.0, 1.0))
        return {"keys": sorted(sess.keys), "speed": sess.teleop_speed}

    @app.post("/api/push")
    def push(body: Dict[str, Any]):
        guard(sess.push, body.get("body", "head"), list(body.get("force", [10, 0, 0])),
              float(body.get("duration", 0.1)))
        return {"ok": True}

    @app.post("/api/sim")
    def sim_control(body: Dict[str, Any]):
        with sess.lock:
            if "paused" in body:
                sess.paused = bool(body["paused"])
            if "speed" in body:
                sess.speed = float(np.clip(body["speed"], 0.05, 4.0))
            if body.get("reset"):
                sess.reset()
            if "recording" in body:
                sess.recording = bool(body["recording"])
                if sess.recording:
                    sess.record = []
        return sess.status()

    @app.post("/api/step")
    def step(body: Dict[str, Any]):
        n = int(np.clip(body.get("n", 1), 1, 10_000))
        for _ in range(n):
            sess.control_step()
        return {"t": float(sess.data.time)}

    @app.get("/api/log.csv")
    def log_csv():
        return PlainTextResponse(sess.log_csv(), media_type="text/csv",
                                 headers={"Content-Disposition": "attachment; filename=bb8_log.csv"})

    @app.get("/api/log.json")
    def log_json():
        return JSONResponse(sess.log_rows(), headers={"Content-Disposition": "attachment; filename=bb8_log.json"})

    @app.post("/api/mesh")
    async def mesh(file: UploadFile = File(...), scale: float = Form(1.0), parent: str = Form("world"),
                   dynamic: bool = Form(True), x: float = Form(0.8), y: float = Form(0.0), z: float = Form(0.3),
                   density: float = Form(1240.0)):
        blob = await file.read()
        return guard(sess.add_mesh, file.filename or "mesh.stl", blob, scale, parent, dynamic and parent == "world",
                     [x, y, z], density)

    @app.post("/api/primitive")
    def primitive(body: Dict[str, Any]):
        guard(sess.add_primitive, body)
        return sess.status()

    @app.post("/api/extras/clear")
    def clear_extras():
        sess.clear_extras()
        return sess.status()

    @app.post("/api/train/start")
    def train_start(body: Dict[str, Any]):
        return guard(sess.start_training, body.get("algo", "PPO"), int(body.get("steps", 1_000_000)),
                     int(body.get("n_envs", 4)), body.get("vec_env", "subproc"),
                     body.get("task", "station_keeping"))

    @app.post("/api/train/stop")
    def train_stop():
        return guard(sess.stop_training)

    @app.get("/api/train/status")
    def train_status():
        return {"status": sess.trainer.status() if sess.trainer else None,
                "metrics": sess.hub.snapshot(last=300)}

    @app.get("/api/policies")
    def policies():
        return sess.list_policies()

    @app.post("/api/policy/load")
    def policy_load(body: Dict[str, Any]):
        path = body.get("path")
        if path == "latest":
            if sess.trainer is None or not (sess.trainer.latest_onnx or sess.trainer.latest_policy):
                raise HTTPException(status_code=400, detail="no checkpoint from the current run yet")
            path = sess.trainer.latest_onnx or sess.trainer.latest_policy
        return guard(sess.load_policy, str(path))

    @app.websocket("/ws")
    async def ws(websocket: WebSocket):
        await websocket.accept()
        sent_version = -1
        sent_seq = sess.event_seq

        async def receiver():
            while True:
                msg = await websocket.receive_json()
                if msg.get("type") == "keys":
                    with sess.lock:
                        sess.keys = {str(k).lower() for k in msg.get("keys", [])} & set("wasdqe")

        recv_task = asyncio.create_task(receiver())
        try:
            while True:
                if recv_task.done():
                    break
                if sess.version != sent_version:
                    with sess.lock:
                        desc = {**sess.descriptor, "version": sess.version}
                    sent_version = desc["version"]
                    await websocket.send_json({"type": "scene", "scene": desc, "status": sess.status()})
                await websocket.send_json(sess.frame())
                new = sess.events_since(sent_seq)
                if new:
                    sent_seq = new[-1]["seq"]
                    await websocket.send_json({"type": "events", "events": new[-200:]})
                await asyncio.sleep(1 / 30)
        except (WebSocketDisconnect, RuntimeError):
            pass
        finally:
            recv_task.cancel()

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    return app


def main(argv: Optional[List[str]] = None) -> int:
    import uvicorn

    ap = argparse.ArgumentParser(description="BB-8 MuJoCo studio (FastAPI + Three.js)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--runs", default="runs", help="directory for training runs and uploaded meshes")
    ap.add_argument("--example", default="bb8")
    args = ap.parse_args(argv)
    session = StudioSession(runs_dir=args.runs)
    if args.example != "bb8":
        session.set_scene(args.example)
    print(f"BB-8 Studio on http://{args.host}:{args.port}", flush=True)
    uvicorn.run(create_app(session), host=args.host, port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    sys.exit(main())

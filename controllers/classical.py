"""Classical controllers for the studio scenes (Module 3).

Contents
--------
* :class:`KiwiMixer` - holonomic ``(vx, vy, wz) -> wheel speeds`` (Example 3).
* :class:`BallWheelMixer` - ball-relative rotation / torque <-> omniwheel commands
  through ``omega_w = J (Omega_ball - Omega_head)`` (Examples 4 & 5).
* :class:`WheelCommandMapper` - wheel speeds / torques -> actuator ``ctrl`` for the
  scene's control mode (stepper velocity servo, torque or position).
* :class:`BallBotModel` - reduced balance coordinates around the top of the ball
  (``measure``) and a constraint-consistent ``lift`` back to a full MuJoCo state.
* :func:`linearize_full` / :func:`linearize_reduced` - central finite-difference
  linearisation **through** :class:`models.runtime.SceneStepper`, so aligned
  omniwheel friction and the magnetic preload are part of ``(A, B)``.
  ``linearize_full`` matches ``mujoco.mjd_transitionFD`` on plain scenes.
* :func:`dlqr`, :func:`augment_delay` - discrete LQR with exact pipeline delay
  (whole control steps plus a physics-step remainder).
* :class:`BallBotLQR`, :class:`CascadedPID` - balance controllers for the ball
  balancers; :class:`FullStateLQR`, :class:`CartPolePID` for Example 1.
* :class:`ClosedLoop` - runs a controller against a scene with frame skip and delay.

Balance state (all angles in rad, expressed in the head's yaw-aligned frame)::

    y = [roll, pitch, roll_rate, pitch_rate, ex, ey, vx, vy, yaw_rate]

``ex, ey`` are the ball position error to the target, ``vx, vy`` the ball
velocity. Absolute ball surface rotation and wheel angles are deliberately
excluded: they do not affect the balance dynamics.

Sign conventions (derived in ``config.composite_ball_params``): positive pitch
tilts the head top toward +x, positive roll toward -y. A positive relative ball
rotation about +y rolls the ball toward +x, which recovers a positive pitch.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Deque, Dict, List, Optional, Sequence, Tuple

import mujoco
import numpy as np
import scipy.linalg

from config.composite_ball_params import ControlMode, HardwareProfile
from models.mjcf_generator import SceneBundle, SceneMetadata
from models.runtime import DelayLine, SceneStepper

__all__ = [
    "STATE_NAMES",
    "KiwiMixer",
    "BallWheelMixer",
    "WheelCommandMapper",
    "BalanceTarget",
    "BalanceState",
    "BallBotModel",
    "linearize_full",
    "linearize_reduced",
    "dlqr",
    "augment_delay",
    "LQRDesign",
    "BallBotLQR",
    "PIDGains",
    "CascadedPIDGains",
    "CascadedPID",
    "FullStateLQR",
    "CartPolePID",
    "ClosedLoop",
]

STATE_NAMES: Tuple[str, ...] = ("roll", "pitch", "roll_rate", "pitch_rate", "ex", "ey", "vx", "vy", "yaw_rate")


def _rz(psi: float) -> np.ndarray:
    c, s = math.cos(psi), math.sin(psi)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _ry(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def _rx(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def _mat2quat(mat: np.ndarray) -> np.ndarray:
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.ascontiguousarray(mat, dtype=float).ravel())
    return q


def zyx_angles(mat: np.ndarray) -> Tuple[float, float, float]:
    """``(yaw, pitch, roll)`` of ``mat = Rz(yaw) Ry(pitch) Rx(roll)``."""
    m = np.asarray(mat, dtype=float).reshape(3, 3)
    yaw = math.atan2(m[1, 0], m[0, 0])
    pitch = math.asin(float(np.clip(-m[2, 0], -1.0, 1.0)))
    roll = math.atan2(m[2, 1], m[2, 2])
    return yaw, pitch, roll


# --------------------------------------------------------------------------- #
# Mixers
# --------------------------------------------------------------------------- #
class KiwiMixer:
    """``omega = M @ [vx, vy, wz]`` with ``M_i = 1/r [-sin th_i, cos th_i, R]`` (chassis frame)."""

    def __init__(self, mixing_matrix: Sequence[Sequence[float]]) -> None:
        self.M = np.asarray(mixing_matrix, dtype=float)
        self.M_pinv = np.linalg.pinv(self.M)

    @classmethod
    def from_meta(cls, meta: SceneMetadata) -> "KiwiMixer":
        return cls(meta.kinematics["mixing_matrix"])

    def wheel_speeds(self, vx: float, vy: float, wz: float) -> np.ndarray:
        return self.M @ np.array([vx, vy, wz], dtype=float)

    def body_velocity(self, wheel_speeds: Sequence[float]) -> np.ndarray:
        """Forward kinematics (least squares) ``[vx, vy, wz]`` from wheel speeds."""
        return self.M_pinv @ np.asarray(wheel_speeds, dtype=float)

    def scaled_wheel_speeds(self, vx: float, vy: float, wz: float, max_speed: float) -> np.ndarray:
        """Wheel speeds, uniformly scaled down if any would exceed ``max_speed``."""
        w = self.wheel_speeds(vx, vy, wz)
        peak = float(np.max(np.abs(w))) if w.size else 0.0
        return w * (max_speed / peak) if peak > max_speed > 0 else w


class BallWheelMixer:
    """Omniwheel mixer for wheels riding on a ball.

    ``omega_wheels = J @ Omega_rel`` where ``Omega_rel = Omega_ball - Omega_head``
    in the head frame, ``J_i = -(R / r_i) a_i^T``. By virtual work the wheel
    torques ``tau`` produce the torque ``T = J^T tau`` on the ball relative to
    the head (and ``-T`` on the head).
    """

    def __init__(self, jacobian: Sequence[Sequence[float]]) -> None:
        self.J = np.atleast_2d(np.asarray(jacobian, dtype=float))
        self.J_pinv = np.linalg.pinv(self.J)
        self.JT_pinv = np.linalg.pinv(self.J.T)

    @classmethod
    def from_meta(cls, meta: SceneMetadata) -> "BallWheelMixer":
        R = float(meta.kinematics["ball_radius"])
        rows = [-(R / w.radius) * np.asarray(w.axle, dtype=float) for w in meta.wheels]
        return cls(rows)

    def rel_rate_to_wheels(self, omega_rel: Sequence[float]) -> np.ndarray:
        return self.J @ np.asarray(omega_rel, dtype=float)

    def wheels_to_rel_rate(self, wheel_speeds: Sequence[float]) -> np.ndarray:
        return self.J_pinv @ np.asarray(wheel_speeds, dtype=float)

    def torque_to_wheels(self, torque: Sequence[float]) -> np.ndarray:
        """Least-norm wheel torques producing relative torque ``T`` on the ball."""
        return self.JT_pinv @ np.asarray(torque, dtype=float)

    def wheels_to_torque(self, wheel_torques: Sequence[float]) -> np.ndarray:
        return self.J.T @ np.asarray(wheel_torques, dtype=float)


class WheelCommandMapper:
    """Converts wheel-level intents into actuator ``ctrl`` values for the scene's control mode."""

    def __init__(self, model: mujoco.MjModel, meta: SceneMetadata) -> None:
        self.model = model
        self.mode = ControlMode(meta.control_mode or "motor")
        self.act_ids = np.array([mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, w.actuator)
                                 for w in meta.wheels], dtype=int)
        if np.any(self.act_ids < 0):
            raise KeyError("wheel actuators missing from model")
        self.gear = model.actuator_gear[self.act_ids, 0].copy()
        self.lo = model.actuator_ctrlrange[self.act_ids, 0].copy()
        self.hi = model.actuator_ctrlrange[self.act_ids, 1].copy()
        limited = model.actuator_ctrllimited[self.act_ids].astype(bool)
        self.lo[~limited] = -np.inf
        self.hi[~limited] = np.inf
        self.position_target = np.zeros(len(self.act_ids))

    def reset(self, data: Optional[mujoco.MjData] = None) -> None:
        self.position_target[:] = 0.0
        if data is not None and self.mode == ControlMode.POSITION:
            for k, aid in enumerate(self.act_ids):
                jid = self.model.actuator_trnid[aid, 0]
                self.position_target[k] = data.qpos[self.model.jnt_qposadr[jid]]

    def clip(self, ctrl: np.ndarray) -> np.ndarray:
        return np.clip(ctrl, self.lo, self.hi)

    def from_wheel_speeds(self, speeds: Sequence[float], dt: float) -> np.ndarray:
        speeds = np.asarray(speeds, dtype=float)
        if self.mode == ControlMode.VELOCITY:
            return self.clip(speeds)
        if self.mode == ControlMode.POSITION:
            self.position_target += speeds * dt
            return self.position_target.copy()
        raise ValueError("torque-mode actuators cannot follow wheel speed commands directly")

    def from_wheel_torques(self, torques: Sequence[float]) -> np.ndarray:
        if self.mode != ControlMode.TORQUE:
            raise ValueError("wheel torques require torque-mode (motor) actuators")
        return self.clip(np.asarray(torques, dtype=float) / self.gear)

    def write(self, data: mujoco.MjData, ctrl: np.ndarray) -> None:
        data.ctrl[self.act_ids] = ctrl


# --------------------------------------------------------------------------- #
# Reduced balance coordinates
# --------------------------------------------------------------------------- #
@dataclass
class BalanceTarget:
    """Station-keeping / velocity-tracking target (world frame).

    With ``track_position`` the target point moves with the commanded velocity,
    which turns velocity tracking into tracking of a moving set-point.
    """

    position: np.ndarray = field(default_factory=lambda: np.zeros(2))
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(2))
    yaw_rate: float = 0.0
    track_position: bool = True
    max_position_error: float = 0.3
    max_accel: float = 0.4                 # [m/s^2] ramp limit toward the commanded velocity
    max_yaw_accel: float = 2.0             # [rad/s^2]
    velocity_cmd: Optional[np.ndarray] = None
    yaw_rate_cmd: Optional[float] = None

    def command(self, velocity: Sequence[float], yaw_rate: float = 0.0) -> None:
        """Set a desired velocity; :meth:`advance` ramps the reference toward it."""
        self.velocity_cmd = np.asarray(velocity, dtype=float).copy()
        self.yaw_rate_cmd = float(yaw_rate)

    def advance(self, dt: float) -> None:
        if self.velocity_cmd is not None:
            dv = self.velocity_cmd - np.asarray(self.velocity, dtype=float)
            n = float(np.linalg.norm(dv))
            step = self.max_accel * dt
            self.velocity = self.velocity_cmd.copy() if n <= step else self.velocity + dv * (step / n)
        if self.yaw_rate_cmd is not None:
            dw = self.yaw_rate_cmd - self.yaw_rate
            self.yaw_rate += float(np.clip(dw, -self.max_yaw_accel * dt, self.max_yaw_accel * dt))
        self.position = np.asarray(self.position, dtype=float) + np.asarray(self.velocity, dtype=float) * dt

    def copy(self) -> "BalanceTarget":
        return BalanceTarget(np.array(self.position, dtype=float), np.array(self.velocity, dtype=float),
                             float(self.yaw_rate), self.track_position, self.max_position_error, self.max_accel,
                             self.max_yaw_accel, None if self.velocity_cmd is None else self.velocity_cmd.copy(),
                             self.yaw_rate_cmd)


@dataclass
class BalanceState:
    roll: float
    pitch: float
    roll_rate: float
    pitch_rate: float
    ex: float
    ey: float
    vx: float
    vy: float
    yaw_rate: float
    yaw: float = 0.0

    def vector(self) -> np.ndarray:
        return np.array([self.roll, self.pitch, self.roll_rate, self.pitch_rate, self.ex, self.ey, self.vx, self.vy,
                         self.yaw_rate])

    @classmethod
    def from_vector(cls, y: Sequence[float], yaw: float = 0.0) -> "BalanceState":
        return cls(*[float(v) for v in y[:9]], yaw=float(yaw))

    def error(self, target: BalanceTarget) -> np.ndarray:
        """Deviation vector used by the LQR: velocity errors relative to the (yaw-frame) target."""
        v_ref = _rz(-self.yaw)[:2, :2] @ np.asarray(target.velocity, dtype=float)
        e = self.vector()
        if not target.track_position:
            e[4:6] = 0.0
        else:
            n = float(np.linalg.norm(e[4:6]))
            if n > target.max_position_error:
                e[4:6] *= target.max_position_error / n
        e[6] -= v_ref[0]
        e[7] -= v_ref[1]
        e[8] -= target.yaw_rate
        return e


class BallBotModel:
    """Measures and lifts the reduced balance state of examples 4 (1d / 2d) and 5.

    ``measure`` works from ground truth (``data``). ``lift`` writes a full MuJoCo
    state for a reduced state that is consistent with the contact constraints:
    the ball rolls without slipping, the head is rotated about the ball centre
    (so the wheels stay on the ball) and wheel speeds satisfy
    ``omega_w = J (Omega_ball - Omega_head)``.
    """

    def __init__(self, model: mujoco.MjModel, meta: SceneMetadata) -> None:
        if meta.example not in ("bb8", "planar_balancer"):
            raise ValueError("BallBotModel supports the 'planar_balancer' and 'bb8' scenes")
        self.model = model
        self.meta = meta
        self.R = float(meta.kinematics["ball_radius"])
        self.mixer = BallWheelMixer.from_meta(meta)
        self.ball_id = model.body(meta.bodies["ball"]).id
        self.head_id = model.body(meta.bodies["head"]).id
        j = meta.joints
        self.variant = "bb8" if meta.example == "bb8" else ("1d" if "head_roll" not in j else "2d")
        self.active = {
            "1d": [1, 3, 4, 6],
            "2d": [0, 1, 2, 3, 4, 5, 6, 7],
            "bb8": list(range(9)),
        }[self.variant]
        self._jid = {k: model.joint(v).id for k, v in j.items()}
        self.wheel_dofs = np.array([model.jnt_dofadr[model.joint(w.joint).id] for w in meta.wheels], dtype=int)
        self.cw_jid = self._jid.get("counterweight_ball")
        self._scratch = mujoco.MjData(model)
        self.reference: Optional[np.ndarray] = None
        self.ref_ball = np.zeros(3)
        self.ref_arm = self.R

    # ----------------------------------------------------------- measurement
    def _refresh(self, data: mujoco.MjData) -> None:
        mujoco.mj_kinematics(self.model, data)
        mujoco.mj_comPos(self.model, data)
        mujoco.mj_comVel(self.model, data)

    def measure(self, data: mujoco.MjData, target: Optional[BalanceTarget] = None,
                refresh: bool = True) -> BalanceState:
        """Ground-truth reduced state (refreshes kinematics, does not change the physics state)."""
        if refresh:
            self._refresh(data)
        m = self.model
        H = data.xmat[self.head_id].reshape(3, 3)
        yaw, pitch, roll = zyx_angles(H)
        Rz_t = _rz(-yaw)
        vel = np.zeros(6)
        mujoco.mj_objectVelocity(m, data, mujoco.mjtObj.mjOBJ_XBODY, self.head_id, vel, 0)
        w_head = Rz_t @ vel[:3]
        mujoco.mj_objectVelocity(m, data, mujoco.mjtObj.mjOBJ_XBODY, self.ball_id, vel, 0)
        v_ball = Rz_t @ vel[3:]
        p = data.xpos[self.ball_id][:2]
        p_ref = np.zeros(2) if target is None else np.asarray(target.position, dtype=float)
        err = Rz_t[:2, :2] @ (p - p_ref)
        y = np.array([roll, pitch, w_head[0], w_head[1], err[0], err[1], v_ball[0], v_ball[1], w_head[2]])
        mask = np.zeros(9, dtype=bool)
        mask[self.active] = True
        y[~mask] = 0.0
        return BalanceState.from_vector(y, yaw=yaw)

    def relative_ball_rate(self, data: mujoco.MjData) -> np.ndarray:
        """``Omega_ball - Omega_head`` in the head frame (needs fresh kinematics)."""
        vel = np.zeros(6)
        mujoco.mj_objectVelocity(self.model, data, mujoco.mjtObj.mjOBJ_XBODY, self.ball_id, vel, 0)
        wb = vel[:3].copy()
        mujoco.mj_objectVelocity(self.model, data, mujoco.mjtObj.mjOBJ_XBODY, self.head_id, vel, 0)
        H = data.xmat[self.head_id].reshape(3, 3)
        return H.T @ (wb - vel[:3])

    # ----------------------------------------------------------- lifting
    def set_reference(self, data: mujoco.MjData) -> None:
        """Record ``data`` (upright, at rest) as the linearisation reference."""
        self._refresh(data)
        self.ref_ball = data.xpos[self.ball_id].copy()
        self.ref_arm = float(np.linalg.norm(data.xpos[self.head_id] - self.ref_ball))
        size = mujoco.mj_stateSize(self.model, mujoco.mjtState.mjSTATE_INTEGRATION)
        self.reference = np.zeros(size)
        mujoco.mj_getState(self.model, data, self.reference, mujoco.mjtState.mjSTATE_INTEGRATION)

    def settle_reference(self, data: mujoco.MjData, stepper: SceneStepper, seconds: float = 0.3) -> None:
        """Place the robot upright at rest, let contacts settle, and store it as reference."""
        m = self.model
        mujoco.mj_resetData(m, data)
        mujoco.mj_forward(m, data)
        self.ref_ball = data.xpos[self.ball_id].copy()
        self.ref_arm = self.R
        self.reference = None
        self.lift(data, np.zeros(9))
        data.ctrl[:] = 0.0
        stepper.reset_state()
        for _ in range(int(round(seconds / m.opt.timestep))):
            stepper.step(1)
        data.qvel[:] = 0.0
        if m.na:
            data.act[:] = 0.0
        self._refresh(data)
        self.ref_ball = data.xpos[self.ball_id].copy()
        self.ref_arm = float(np.linalg.norm(data.xpos[self.head_id] - self.ref_ball))
        self.lift(data, np.zeros(9))
        self.set_reference(data)

    def lift(self, data: mujoco.MjData, y: Sequence[float]) -> None:
        """Write the full state for reduced state ``y`` (yaw = 0) into ``data``."""
        m = self.model
        if self.reference is not None:
            mujoco.mj_setState(m, data, self.reference, mujoco.mjtState.mjSTATE_INTEGRATION)
        y = np.asarray(y, dtype=float)
        mask = np.zeros(9)
        mask[self.active] = 1.0
        y = y * mask
        roll, pitch, rr, pr, ex, ey, vx, vy, wz = y
        R = self.R
        c = self.ref_ball + np.array([ex, ey, 0.0])
        v_ball = np.array([vx, vy, 0.0])
        w_ball = np.array([-vy / R, vx / R, 0.0])
        Q = _ry(pitch) @ _rx(roll)
        arm = Q @ np.array([0.0, 0.0, self.ref_arm])
        p_head = c + arm
        w_head = np.array([rr, pr, wz])
        v_head = v_ball + np.cross(w_head, arm)
        jid = self._jid

        # ----- ball
        if "ball_free" in jid:
            qa, da = m.jnt_qposadr[jid["ball_free"]], m.jnt_dofadr[jid["ball_free"]]
            data.qpos[qa:qa + 3] = c
            qb = data.qpos[qa + 3:qa + 7].copy()
            Rb = np.zeros(9)
            mujoco.mju_quat2Mat(Rb, qb)
            data.qvel[da:da + 3] = v_ball
            data.qvel[da + 3:da + 6] = Rb.reshape(3, 3).T @ w_ball
        else:
            body_pos = m.body_pos[self.ball_id]
            data.qpos[m.jnt_qposadr[jid["ball_x"]]] = c[0] - body_pos[0]
            data.qpos[m.jnt_qposadr[jid["ball_z"]]] = c[2] - body_pos[2]
            data.qvel[m.jnt_dofadr[jid["ball_x"]]] = vx
            data.qvel[m.jnt_dofadr[jid["ball_z"]]] = 0.0
            data.qvel[m.jnt_dofadr[jid["ball_pitch"]]] = w_ball[1]

        # ----- head
        if "head_free" in jid:
            qa, da = m.jnt_qposadr[jid["head_free"]], m.jnt_dofadr[jid["head_free"]]
            data.qpos[qa:qa + 3] = p_head
            data.qpos[qa + 3:qa + 7] = _mat2quat(Q)
            data.qvel[da:da + 3] = v_head
            data.qvel[da + 3:da + 6] = Q.T @ w_head
        else:
            body_pos = m.body_pos[self.head_id]
            for axis_k, name in enumerate(("head_x", "head_y", "head_z")):
                if name in jid:
                    data.qpos[m.jnt_qposadr[jid[name]]] = p_head[axis_k] - body_pos[axis_k]
                    data.qvel[m.jnt_dofadr[jid[name]]] = v_head[axis_k]
            data.qpos[m.jnt_qposadr[jid["head_pitch"]]] = pitch
            data.qvel[m.jnt_dofadr[jid["head_pitch"]]] = pr
            if "head_roll" in jid:
                data.qpos[m.jnt_qposadr[jid["head_roll"]]] = roll
                data.qvel[m.jnt_dofadr[jid["head_roll"]]] = rr

        # ----- counterweight hangs still in the world
        if self.cw_jid is not None:
            da = m.jnt_dofadr[self.cw_jid]
            data.qvel[da:da + 3] = -w_ball

        # ----- wheels (no slip along rolling directions)
        mujoco.mj_kinematics(m, data)
        H = data.xmat[self.head_id].reshape(3, 3)
        data.qvel[self.wheel_dofs] = self.mixer.rel_rate_to_wheels(H.T @ (w_ball - w_head))


# --------------------------------------------------------------------------- #
# Finite-difference linearisation through SceneStepper
# --------------------------------------------------------------------------- #
def _state_get(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    buf = np.zeros(mujoco.mj_stateSize(model, mujoco.mjtState.mjSTATE_INTEGRATION))
    mujoco.mj_getState(model, data, buf, mujoco.mjtState.mjSTATE_INTEGRATION)
    return buf


def _state_set(model: mujoco.MjModel, data: mujoco.MjData, buf: np.ndarray) -> None:
    mujoco.mj_setState(model, data, buf, mujoco.mjtState.mjSTATE_INTEGRATION)


def linearize_full(
    stepper: SceneStepper,
    frame_skip: int = 1,
    eps: float = 1e-6,
) -> Tuple[np.ndarray, np.ndarray]:
    """Central FD of the ``frame_skip``-step map around the current state and ``ctrl``.

    Tangent-space state ``dx = [dq (nv), dv (nv), dact (na)]`` exactly like
    ``mujoco.mjd_transitionFD``; the stepper (contact alignment, magnets) is
    used for every rollout. The data state is restored afterwards.
    """
    m, d = stepper.model, stepper.data
    nv, na, nu = m.nv, m.na, m.nu
    nx = 2 * nv + na
    s0 = _state_get(m, d)
    qpos0, qvel0 = d.qpos.copy(), d.qvel.copy()
    act0 = d.act.copy()
    ctrl0 = d.ctrl.copy()

    def rollout(dx: np.ndarray, du: np.ndarray) -> np.ndarray:
        _state_set(m, d, s0)
        qpos = qpos0.copy()
        mujoco.mj_integratePos(m, qpos, dx[:nv], 1.0)
        d.qpos[:] = qpos
        d.qvel[:] = qvel0 + dx[nv:2 * nv]
        if na:
            d.act[:] = act0 + dx[2 * nv:]
        d.ctrl[:] = ctrl0 + du
        stepper.reset_state()
        stepper.step(frame_skip)
        out = np.zeros(nx)
        dq = np.zeros(nv)
        mujoco.mj_differentiatePos(m, dq, 1.0, ref_next_qpos, d.qpos)
        out[:nv] = dq
        out[nv:2 * nv] = d.qvel - ref_next_qvel
        if na:
            out[2 * nv:] = d.act - ref_next_act
        return out

    # nominal next state
    _state_set(m, d, s0)
    stepper.reset_state()
    stepper.step(frame_skip)
    ref_next_qpos, ref_next_qvel, ref_next_act = d.qpos.copy(), d.qvel.copy(), d.act.copy()

    A = np.zeros((nx, nx))
    B = np.zeros((nx, nu))
    zx, zu = np.zeros(nx), np.zeros(nu)
    for i in range(nx):
        e = zx.copy()
        e[i] = eps
        A[:, i] = (rollout(e, zu) - rollout(-e, zu)) / (2 * eps)
    for i in range(nu):
        e = zu.copy()
        e[i] = eps
        B[:, i] = (rollout(zx, e) - rollout(zx, -e)) / (2 * eps)
    _state_set(m, d, s0)
    mujoco.mj_forward(m, d)
    return A, B


def linearize_reduced(
    stepper: SceneStepper,
    bbm: BallBotModel,
    frame_skip: int,
    u0: Optional[np.ndarray] = None,
    delay_substeps: int = 0,
    eps: float = 1e-5,
    eps_u: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reduced-state linearisation ``y+ = A y + B_new u_new + B_old u_old`` around ``bbm.reference``.

    ``delay_substeps`` (< frame_skip) is the physics-step remainder of the
    pipeline delay: the first ``delay_substeps`` steps of the control period
    still apply the previous command ``u_old``. Returns ``(A, B_new, B_old)``
    on the active state components.
    """
    if bbm.reference is None:
        raise RuntimeError("call bbm.settle_reference(...) first")
    m, d = stepper.model, stepper.data
    if not (0 <= delay_substeps < frame_skip):
        raise ValueError("delay_substeps must be within [0, frame_skip)")
    nu = m.nu
    u0 = np.zeros(nu) if u0 is None else np.asarray(u0, dtype=float)
    eps_u = eps * 10 if eps_u is None else eps_u
    act = bbm.active
    ref_target = BalanceTarget(position=bbm.ref_ball[:2].copy())
    saved = _state_get(m, d)

    def f(y: np.ndarray, u_new: np.ndarray, u_old: np.ndarray) -> np.ndarray:
        full = np.zeros(9)
        full[act] = y
        bbm.lift(d, full)
        stepper.reset_state()
        for k in range(frame_skip):
            d.ctrl[:] = u_old if k < delay_substeps else u_new
            stepper.step(1)
        return bbm.measure(d, ref_target).vector()[act]

    n = len(act)
    y0 = np.zeros(n)
    A = np.zeros((n, n))
    Bn = np.zeros((n, nu))
    Bo = np.zeros((n, nu))
    for i in range(n):
        e = np.zeros(n)
        e[i] = eps
        A[:, i] = (f(y0 + e, u0, u0) - f(y0 - e, u0, u0)) / (2 * eps)
    for i in range(nu):
        e = np.zeros(nu)
        e[i] = eps_u
        Bn[:, i] = (f(y0, u0 + e, u0) - f(y0, u0 - e, u0)) / (2 * eps_u)
        if delay_substeps:
            Bo[:, i] = (f(y0, u0, u0 + e) - f(y0, u0, u0 - e)) / (2 * eps_u)
    _state_set(m, d, saved)
    mujoco.mj_forward(m, d)
    return A, Bn, Bo


# --------------------------------------------------------------------------- #
# LQR
# --------------------------------------------------------------------------- #
def dlqr(A: np.ndarray, B: np.ndarray, Q: np.ndarray, R: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Discrete LQR: returns ``(K, P)`` with ``u = -K x``."""
    P = scipy.linalg.solve_discrete_are(A, B, Q, R)
    K = np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)
    return K, P


def augment_delay(
    A: np.ndarray, B_new: np.ndarray, B_old: np.ndarray, whole_steps: int,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Augment with an input buffer for ``x+ = A x + B_new u_{k-m} + B_old u_{k-m-1}``.

    State ``z = [x, u_{k-1}, ..., u_{k-L}]``; returns ``(Az, Bz, L)``.
    """
    nx, nu = B_new.shape
    m = int(whole_steps)
    has_old = bool(np.any(B_old))
    L = m + (1 if has_old else 0)
    nz = nx + L * nu
    Az = np.zeros((nz, nz))
    Bz = np.zeros((nz, nu))
    Az[:nx, :nx] = A

    def add(Bmat: np.ndarray, lag: int) -> None:
        if lag == 0:
            Bz[:nx] += Bmat
        else:
            Az[:nx, nx + (lag - 1) * nu: nx + lag * nu] += Bmat

    add(B_new, m)
    if has_old:
        add(B_old, m + 1)
    if L >= 1:
        Bz[nx:nx + nu] = np.eye(nu)
        for s in range(1, L):
            Az[nx + s * nu: nx + (s + 1) * nu, nx + (s - 1) * nu: nx + s * nu] = np.eye(nu)
    return Az, Bz, L


@dataclass
class LQRDesign:
    A: np.ndarray
    B: np.ndarray
    K: np.ndarray
    Q: np.ndarray
    R: np.ndarray
    dt: float
    buffer_len: int
    state_names: List[str]
    input_names: List[str]

    @property
    def closed_loop_poles(self) -> np.ndarray:
        return np.linalg.eigvals(self.A - self.B @ self.K)

    @property
    def spectral_radius(self) -> float:
        return float(np.max(np.abs(self.closed_loop_poles)))

    def summary(self) -> Dict[str, Any]:
        return {
            "dt": self.dt,
            "buffer_len": self.buffer_len,
            "spectral_radius": self.spectral_radius,
            "open_loop_radius": float(np.max(np.abs(np.linalg.eigvals(self.A)))),
            "K": self.K.tolist(),
            "states": self.state_names,
            "inputs": self.input_names,
        }


DEFAULT_LQR_Q = {"roll": 400.0, "pitch": 400.0, "roll_rate": 10.0, "pitch_rate": 10.0, "ex": 300.0, "ey": 300.0,
                 "vx": 60.0, "vy": 60.0, "yaw_rate": 2.0}
# Torque mode saturates at 0.44 N m much earlier; softer position / velocity weights keep it robust.
DEFAULT_LQR_Q_TORQUE = dict(DEFAULT_LQR_Q, ex=30.0, ey=30.0, vx=10.0, vy=10.0)


class BallBotLQR:
    """LQR balance controller for the planar (1d/2d) balancer and the BB-8.

    Design runs on a private ``MjData`` so the live simulation is untouched::

        lqr = BallBotLQR.design(bundle, hardware)
        ctrl = lqr.compute(bbm.measure(data, target), target)
    """

    def __init__(self, design: LQRDesign, bbm: BallBotModel, nu: int, ctrl_lo: np.ndarray,
                 ctrl_hi: np.ndarray, mode: ControlMode = ControlMode.VELOCITY, yaw_feedforward: float = 1.1) -> None:
        self.design_data = design
        self.bbm = bbm
        self.nu = nu
        self.lo, self.hi = ctrl_lo, ctrl_hi
        self.mode = ControlMode(mode)
        self.yaw_feedforward = yaw_feedforward
        self.buffer: Deque[np.ndarray] = deque([np.zeros(nu) for _ in range(design.buffer_len)],
                                               maxlen=max(design.buffer_len, 1))

    @classmethod
    def design(
        cls,
        bundle: SceneBundle,
        hardware: Optional[HardwareProfile] = None,
        model: Optional[mujoco.MjModel] = None,
        q_weights: Optional[Dict[str, float]] = None,
        r_weight: Optional[float] = None,
        frame_skip: Optional[int] = None,
        delay_steps: Optional[int] = None,
    ) -> "BallBotLQR":
        hw = hardware or HardwareProfile()
        model = model or bundle.compile()
        meta = bundle.meta
        mode = ControlMode(meta.control_mode or "motor")
        if mode == ControlMode.POSITION:
            raise ValueError("LQR design needs torque or velocity actuators")
        data = mujoco.MjData(model)
        stepper = SceneStepper(model, data, meta)
        bbm = BallBotModel(model, meta)
        fs = frame_skip or max(1, int(round(hw.loop.control_dt / model.opt.timestep)))
        dsteps = hw.delay_steps if delay_steps is None else int(delay_steps)
        whole, rem = divmod(dsteps, fs)
        bbm.settle_reference(data, stepper)
        A, Bn, Bo = linearize_reduced(stepper, bbm, fs, delay_substeps=rem)
        names = [STATE_NAMES[i] for i in bbm.active]
        qw = dict(DEFAULT_LQR_Q if mode == ControlMode.VELOCITY else DEFAULT_LQR_Q_TORQUE)
        qw.update(q_weights or {})
        if r_weight is None:
            if mode == ControlMode.VELOCITY:
                r_weight = 2.0 / float(np.max(np.abs(model.actuator_ctrlrange[:, 1]))) ** 2 * 100
            else:
                r_weight = 1.0 / float(np.max(np.abs(model.actuator_ctrlrange[:, 1]))) ** 2
        Az, Bz, L = augment_delay(A, Bn, Bo, whole)
        nz = Az.shape[0]
        Q = np.zeros((nz, nz))
        Q[:len(names), :len(names)] = np.diag([qw[n] for n in names])
        Q[len(names):, len(names):] = np.eye(nz - len(names)) * 1e-6
        R = np.eye(model.nu) * r_weight
        K, _ = dlqr(Az, Bz, Q, R)
        dt = fs * model.opt.timestep
        design = LQRDesign(A=Az, B=Bz, K=K, Q=Q, R=R, dt=dt, buffer_len=L, state_names=names,
                           input_names=[model.actuator(i).name for i in range(model.nu)])
        lo = model.actuator_ctrlrange[:, 0].copy()
        hi = model.actuator_ctrlrange[:, 1].copy()
        unl = ~model.actuator_ctrllimited.astype(bool)
        lo[unl], hi[unl] = -np.inf, np.inf
        live_bbm = BallBotModel(model, meta)
        return cls(design, live_bbm, model.nu, lo, hi, mode=mode)

    @property
    def K(self) -> np.ndarray:
        return self.design_data.K

    def reset(self) -> None:
        for b in self.buffer:
            b[:] = 0.0

    def feedforward(self, state: BalanceState, target: BalanceTarget) -> np.ndarray:
        """Equilibrium command for the target motion (steady rolling needs wheel speed in velocity mode)."""
        if self.mode != ControlMode.VELOCITY:
            return np.zeros(self.nu)
        v = _rz(-state.yaw)[:2, :2] @ np.asarray(target.velocity, dtype=float)
        R = self.bbm.R
        wz = target.yaw_rate if self.bbm.variant == "bb8" else 0.0
        omega_rel = np.array([-v[1] / R, v[0] / R, -self.yaw_feedforward * wz])
        if self.bbm.variant == "1d":
            omega_rel[[0, 2]] = 0.0
        return self.bbm.mixer.rel_rate_to_wheels(omega_rel)

    def compute(self, state: BalanceState, target: BalanceTarget) -> np.ndarray:
        x = state.error(target)[self.bbm.active]
        z = np.concatenate([x] + list(self.buffer)[: self.design_data.buffer_len]) if self.design_data.buffer_len else x
        u_ff = self.feedforward(state, target)
        u = np.clip(u_ff - self.K @ z, self.lo, self.hi)
        if self.design_data.buffer_len:
            self.buffer.appendleft(u - u_ff)        # the model's input is the deviation from equilibrium
        return u


# --------------------------------------------------------------------------- #
# Cascaded PID
# --------------------------------------------------------------------------- #
@dataclass
class PIDGains:
    kp: float = 0.0
    ki: float = 0.0
    kd: float = 0.0
    i_limit: float = 1.0


@dataclass
class CascadedPIDGains:
    """Gains for :class:`CascadedPID`.

    Velocity (stepper) mode: inner-loop output is a relative ball angular
    acceleration [rad/s^2] integrated into the wheel speed command. Torque
    mode: output is the relative torque on the ball [N m].
    """

    inner: PIDGains = field(default_factory=lambda: PIDGains(kp=150.0, ki=20.0, kd=18.0, i_limit=0.2))
    outer_pos: float = 0.5              # rad of lean per m of position error
    outer_vel: float = 0.45             # rad of lean per m/s of velocity error
    outer_vel_ki: float = 0.05          # rad per (m/s * s)
    max_tilt: float = 0.12              # rad lean command limit
    yaw: PIDGains = field(default_factory=lambda: PIDGains(kp=0.6))
    yaw_feedforward: float = 1.1        # relative spin per unit head yaw rate (inertia ratio)
    rate_leak: float = 0.0              # optional leak on the integrated rate command [1/s]

    @classmethod
    def torque_mode_defaults(cls) -> "CascadedPIDGains":
        return cls(inner=PIDGains(kp=30.0, ki=5.0, kd=3.0, i_limit=0.2), outer_pos=0.25, outer_vel=0.18,
                   outer_vel_ki=0.05, max_tilt=0.12, yaw=PIDGains(kp=0.3), yaw_feedforward=0.0)


class CascadedPID:
    """Outer loop (ball position / velocity -> lean set-point), inner loop (lean -> wheel commands)."""

    def __init__(self, model: mujoco.MjModel, meta: SceneMetadata, gains: Optional[CascadedPIDGains] = None,
                 dt: float = 0.01) -> None:
        self.mapper = WheelCommandMapper(model, meta)
        self.mixer = BallWheelMixer.from_meta(meta)
        if gains is None:
            gains = (CascadedPIDGains.torque_mode_defaults() if self.mapper.mode == ControlMode.TORQUE
                     else CascadedPIDGains())
        self.gains = gains
        self.dt = dt
        self.variant = BallBotModel(model, meta).variant
        self.reset()

    def reset(self, data: Optional[mujoco.MjData] = None) -> None:
        self.i_tilt = np.zeros(2)
        self.i_vel = np.zeros(2)
        self.rate_cmd = np.zeros(3)
        self.tilt_ref = np.zeros(2)
        self.mapper.reset(data)

    def compute(self, state: BalanceState, target: BalanceTarget) -> np.ndarray:
        g, dt = self.gains, self.dt
        e = state.error(target)
        # Outer loop: position + velocity error -> lean set-point (yaw frame).
        self.i_vel = np.clip(self.i_vel + e[6:8] * dt, -1.0, 1.0)
        drive = g.outer_pos * e[4:6] + g.outer_vel * e[6:8] + g.outer_vel_ki * self.i_vel
        pitch_ref = float(np.clip(-drive[0], -g.max_tilt, g.max_tilt))
        roll_ref = float(np.clip(+drive[1], -g.max_tilt, g.max_tilt))
        self.tilt_ref = np.array([roll_ref, pitch_ref])
        # Inner loop on (roll, pitch); derivative on measurement.
        tilt = np.array([state.roll, state.pitch])
        rates = np.array([state.roll_rate, state.pitch_rate])
        err = tilt - self.tilt_ref
        lim = g.inner.i_limit
        self.i_tilt = np.clip(self.i_tilt + err * dt, -lim, lim)
        effort_xy = g.inner.kp * err + g.inner.ki * self.i_tilt + g.inner.kd * rates
        if self.variant == "1d":
            effort_xy[0] = 0.0
        yaw_err = target.yaw_rate - state.yaw_rate if self.variant == "bb8" else 0.0
        if self.mapper.mode == ControlMode.TORQUE:
            torque = np.array([effort_xy[0], effort_xy[1], -g.yaw.kp * yaw_err])
            return self.mapper.from_wheel_torques(self.mixer.torque_to_wheels(torque))
        # Velocity / position (stepper) mode: integrate acceleration into a rate command.
        self.rate_cmd[:2] += effort_xy * dt
        if g.rate_leak:
            self.rate_cmd[:2] *= max(0.0, 1.0 - g.rate_leak * dt)
        self.rate_cmd[2] = -(g.yaw_feedforward * target.yaw_rate + g.yaw.kp * yaw_err) if self.variant == "bb8" else 0.0
        speeds = self.mixer.rel_rate_to_wheels(self.rate_cmd)
        ctrl = self.mapper.from_wheel_speeds(speeds, dt)
        # Anti-windup: keep the internal command consistent with saturation.
        if self.mapper.mode == ControlMode.VELOCITY:
            self.rate_cmd[:2] = self.mixer.wheels_to_rel_rate(ctrl)[:2]
        return ctrl


# --------------------------------------------------------------------------- #
# Example 1: pendulum controllers
# --------------------------------------------------------------------------- #
class FullStateLQR:
    """LQR on the full tangent state around a reference (e.g. upright cart-pole).

    ``u = u0 - K [dq, dv]`` where ``dq`` is ``mj_differentiatePos(qpos_ref, qpos)``.
    """

    def __init__(self, model: mujoco.MjModel, qpos_ref: np.ndarray, K: np.ndarray, u0: np.ndarray,
                 design: LQRDesign) -> None:
        self.model = model
        self.qpos_ref = qpos_ref
        self.K = K
        self.u0 = u0
        self.design_data = design
        lo = model.actuator_ctrlrange[:, 0].copy()
        hi = model.actuator_ctrlrange[:, 1].copy()
        unl = ~model.actuator_ctrllimited.astype(bool)
        lo[unl], hi[unl] = -np.inf, np.inf
        self.lo, self.hi = lo, hi

    @classmethod
    def design(cls, bundle: SceneBundle, frame_skip: int = 5, q_diag: Optional[Sequence[float]] = None,
               r_diag: Optional[Sequence[float]] = None, model: Optional[mujoco.MjModel] = None) -> "FullStateLQR":
        model = model or bundle.compile()
        data = mujoco.MjData(model)
        mujoco.mj_resetData(model, data)          # qpos0 = upright for the pendulum scenes
        mujoco.mj_forward(model, data)
        stepper = SceneStepper(model, data, bundle.meta)
        A, B = linearize_full(stepper, frame_skip=frame_skip)
        nx = A.shape[0]
        Q = np.diag(q_diag) if q_diag is not None else np.eye(nx)
        if q_diag is None:
            for j in range(model.njnt):
                if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE:
                    Q[model.jnt_dofadr[j], model.jnt_dofadr[j]] = 50.0
        R = np.diag(r_diag) if r_diag is not None else np.eye(model.nu) * 0.01
        K, _ = dlqr(A, B, Q, R)
        design = LQRDesign(A=A, B=B, K=K, Q=Q, R=R, dt=frame_skip * model.opt.timestep, buffer_len=0,
                           state_names=[f"dof{i}" for i in range(nx)],
                           input_names=[model.actuator(i).name for i in range(model.nu)])
        return cls(model, model.qpos0.copy(), K, np.zeros(model.nu), design)

    def reset(self) -> None:
        pass

    def compute_full(self, data: mujoco.MjData) -> np.ndarray:
        m = self.model
        dq = np.zeros(m.nv)
        mujoco.mj_differentiatePos(m, dq, 1.0, self.qpos_ref, data.qpos)
        x = np.concatenate([dq, data.qvel, data.act])
        return np.clip(self.u0 - self.K @ x, self.lo, self.hi)


class CartPolePID:
    """Cascaded PID for the cart-pole: cart position -> pole angle set-point -> cart force."""

    def __init__(self, model: mujoco.MjModel, meta: SceneMetadata, dt: float = 0.01,
                 angle: PIDGains = PIDGains(kp=60.0, ki=0.0, kd=8.0),
                 position: PIDGains = PIDGains(kp=0.08, ki=0.0, kd=0.12), max_ref: float = 0.15) -> None:
        self.model = model
        self.dt = dt
        self.angle, self.position, self.max_ref = angle, position, max_ref
        j = meta.joints
        self.x_adr = model.jnt_qposadr[model.joint(j["cart_x"]).id]
        self.xv_adr = model.jnt_dofadr[model.joint(j["cart_x"]).id]
        self.th_adr = model.jnt_qposadr[model.joint(j["pole_pitch"]).id]
        self.thv_adr = model.jnt_dofadr[model.joint(j["pole_pitch"]).id]
        self.lim = model.actuator_ctrlrange[0, 1]
        self.reset()

    def reset(self) -> None:
        self.i_angle = 0.0

    def compute_full(self, data: mujoco.MjData, x_target: float = 0.0) -> np.ndarray:
        x, xd = data.qpos[self.x_adr], data.qvel[self.xv_adr]
        th, thd = data.qpos[self.th_adr], data.qvel[self.thv_adr]
        # theta > 0 (about +y) tilts the pole toward +x; to travel toward +x the pole must lean toward +x
        ref = float(np.clip(self.position.kp * (x_target - x) - self.position.kd * xd, -self.max_ref, self.max_ref))
        err = th - ref
        self.i_angle = float(np.clip(self.i_angle + err * self.dt, -1.0, 1.0))
        u = self.angle.kp * err + self.angle.ki * self.i_angle + self.angle.kd * thd
        out = np.zeros(self.model.nu)
        out[0] = np.clip(u, -self.lim, self.lim)
        return out


# --------------------------------------------------------------------------- #
# Closed-loop runner
# --------------------------------------------------------------------------- #
class ClosedLoop:
    """Runs a controller at ``control_dt`` with frame skip and a physics-step pipeline delay.

    ``policy(data, k) -> ctrl`` is called every control step ``k``.
    """

    def __init__(self, bundle: SceneBundle, hardware: Optional[HardwareProfile] = None,
                 model: Optional[mujoco.MjModel] = None) -> None:
        self.bundle = bundle
        self.hw = hardware or HardwareProfile()
        self.model = model or bundle.compile()
        self.data = mujoco.MjData(self.model)
        bundle.reset(self.model, self.data)
        self.stepper = SceneStepper(self.model, self.data, bundle.meta)
        self.frame_skip = max(1, int(round(self.hw.loop.control_dt / self.model.opt.timestep)))
        self.delay = DelayLine(self.model.nu, self.hw.delay_steps)
        self.step_count = 0

    def run(self, policy: Callable[[mujoco.MjData, int], np.ndarray], seconds: float,
            on_step: Optional[Callable[[mujoco.MjData, int], None]] = None) -> int:
        n_ctrl = int(round(seconds / (self.frame_skip * self.model.opt.timestep)))
        for k in range(n_ctrl):
            self.delay.submit(policy(self.data, k), self.step_count)
            for _ in range(self.frame_skip):
                self.data.ctrl[:] = self.delay.current(self.step_count)
                self.stepper.step(1)
                self.step_count += 1
            if on_step is not None:
                on_step(self.data, k)
        return n_ctrl

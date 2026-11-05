"""Input/Output bus: realistic sensors and actuators for the ball-balancer scenes.

* :class:`SensorBus` reads the MuJoCo sensors generated for a scene and adds the
  sim-to-real effects of :class:`config.composite_ball_params.HardwareProfile`:
  gyro white noise + per-episode bias + bias random walk, orientation noise,
  accelerometer noise, 1200 CPR encoder quantisation (wheel speed is then the
  finite difference of quantised counts), and ball ground-truth noise.
  :meth:`SensorBus.estimate` turns a reading into the reduced
  :class:`controllers.classical.BalanceState` the controllers use.
* :class:`ActuatorBus` applies the pipeline delay (ring buffer at physics-step
  resolution), the actuator saturation and the stepper pull-out torque/speed
  curve (by updating each wheel actuator's force range from its speed).

Both are used by the Gymnasium environment and by the studio so that a
controller sees identical signals in training, testing and the GUI.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import mujoco
import numpy as np

from config.composite_ball_params import ControlMode, HardwareProfile, MotorParams
from models.mjcf_generator import SceneMetadata
from models.runtime import DelayLine

__all__ = ["SensorReading", "SensorBus", "ActuatorBus", "quat_to_zyx"]


def quat_to_zyx(q: Sequence[float]) -> np.ndarray:
    """Unit quaternion (w, x, y, z) -> ``[yaw, pitch, roll]`` for ``R = Rz Ry Rx``."""
    mat = np.zeros(9)
    mujoco.mju_quat2Mat(mat, np.asarray(q, dtype=float))
    m = mat.reshape(3, 3)
    return np.array([
        math.atan2(m[1, 0], m[0, 0]),
        math.asin(float(np.clip(-m[2, 0], -1.0, 1.0))),
        math.atan2(m[2, 1], m[2, 2]),
    ])


def _rz(psi: float) -> np.ndarray:
    c, s = math.cos(psi), math.sin(psi)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


@dataclass
class SensorReading:
    """One (possibly noisy) snapshot of the sensor bus."""

    time: float
    imu_quat: np.ndarray                      # head orientation estimate (w, x, y, z)
    euler: np.ndarray                         # [yaw, pitch, roll] from imu_quat
    gyro: np.ndarray                          # head angular rate, IMU (head) frame [rad/s]
    accel: np.ndarray                         # specific force, IMU frame [m/s^2]
    wheel_pos: np.ndarray                     # encoder angle [rad] (quantised)
    wheel_vel: np.ndarray                     # wheel speed [rad/s]
    ball_pos: np.ndarray                      # world [m]
    ball_vel: np.ndarray                      # world [m/s]
    ball_angvel: np.ndarray                   # world [rad/s]
    gyro_bias: np.ndarray = field(default_factory=lambda: np.zeros(3))

    def as_dict(self) -> Dict[str, List[float]]:
        return {k: (np.asarray(v).tolist() if not isinstance(v, float) else v) for k, v in self.__dict__.items()}


class SensorBus:
    """Noisy, quantised view of the scene's sensors (see module docstring)."""

    def __init__(self, model: mujoco.MjModel, meta: SceneMetadata, hardware: Optional[HardwareProfile] = None,
                 rng: Optional[np.random.Generator] = None) -> None:
        self.model = model
        self.meta = meta
        self.hw = hardware or HardwareProfile()
        self.rng = rng or np.random.default_rng()
        if not meta.imu_site:
            raise ValueError("scene has no IMU site")
        base = meta.imu_site                    # e.g. "head_imu"
        self._adr = {}
        for key, name in (("acc", f"{base}_acc"), ("gyro", f"{base}_gyro"), ("quat", f"{base}_quat"),
                          ("ball_pos", "ball_pos"), ("ball_vel", "ball_linvel"), ("ball_angvel", "ball_angvel")):
            sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, name)
            self._adr[key] = (int(model.sensor_adr[sid]), int(model.sensor_dim[sid])) if sid >= 0 else None
        self._enc = []
        for w in meta.wheels:
            sp = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, w.encoder_pos_sensor)
            sv = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, w.encoder_vel_sensor)
            self._enc.append((int(model.sensor_adr[sp]), int(model.sensor_adr[sv])))
        self.n_wheels = len(self._enc)
        self.reset()

    # ------------------------------------------------------------------ state
    def reset(self, rng: Optional[np.random.Generator] = None) -> None:
        if rng is not None:
            self.rng = rng
        imu = self.hw.imu
        self.gyro_bias = (self.rng.normal(0.0, imu.gyro_bias_std, 3) if self.hw.noise_enabled
                          else np.zeros(3))
        self._last_counts: Optional[np.ndarray] = None
        self._last_time: Optional[float] = None
        self._last_vel = np.zeros(self.n_wheels)

    def _read(self, data: mujoco.MjData, key: str) -> np.ndarray:
        adr = self._adr.get(key)
        if adr is None:
            return np.zeros(4 if key == "quat" else 3)
        a, n = adr
        return data.sensordata[a:a + n].copy()

    # ------------------------------------------------------------------ read
    def read(self, data: mujoco.MjData) -> SensorReading:
        hw, rng = self.hw, self.rng
        noisy = hw.noise_enabled
        imu, bs = hw.imu, hw.ball_sensors
        t = float(data.time)
        dt = 0.0 if self._last_time is None else max(t - self._last_time, 0.0)

        quat = self._read(data, "quat")
        gyro = self._read(data, "gyro")
        acc = self._read(data, "acc")
        if noisy:
            if dt > 0:
                self.gyro_bias = self.gyro_bias + rng.normal(0.0, imu.gyro_bias_walk * math.sqrt(dt), 3)
            gyro = gyro + self.gyro_bias + rng.normal(0.0, imu.gyro_noise_std, 3)
            acc = acc + rng.normal(0.0, imu.accel_noise_std, 3)
            dq = np.zeros(4)
            mujoco.mju_axisAngle2Quat(dq, np.array([1.0, 0.0, 0.0]), 0.0)
            rot = rng.normal(0.0, imu.orientation_noise_std, 3)
            angle = float(np.linalg.norm(rot))
            if angle > 0:
                mujoco.mju_axisAngle2Quat(dq, rot / angle, angle)
                out = np.zeros(4)
                mujoco.mju_mulQuat(out, quat, dq)
                quat = out / np.linalg.norm(out)

        pos = np.array([data.sensordata[a] for a, _ in self._enc])
        vel_true = np.array([data.sensordata[v] for _, v in self._enc])
        if hw.encoder.quantize:
            res = hw.encoder.resolution
            counts = np.floor(pos / res)
            wheel_pos = counts * res
            if self._last_counts is None or dt <= 0:
                wheel_vel = self._last_vel if self._last_counts is not None else np.zeros(self.n_wheels)
            else:
                wheel_vel = (counts - self._last_counts) * res / dt
            self._last_counts = counts
        else:
            wheel_pos = pos
            wheel_vel = vel_true
        self._last_vel = wheel_vel

        ball_pos = self._read(data, "ball_pos")
        ball_vel = self._read(data, "ball_vel")
        ball_w = self._read(data, "ball_angvel")
        if noisy:
            ball_pos = ball_pos + rng.normal(0.0, bs.position_std, 3)
            ball_vel = ball_vel + rng.normal(0.0, bs.velocity_std, 3)
            ball_w = ball_w + rng.normal(0.0, bs.angular_velocity_std, 3)
        self._last_time = t
        return SensorReading(time=t, imu_quat=quat, euler=quat_to_zyx(quat), gyro=gyro, accel=acc,
                             wheel_pos=wheel_pos, wheel_vel=np.asarray(wheel_vel, dtype=float), ball_pos=ball_pos,
                             ball_vel=ball_vel, ball_angvel=ball_w, gyro_bias=self.gyro_bias.copy())

    # ------------------------------------------------------------------ estimate
    @staticmethod
    def estimate(reading: SensorReading, target_position: Sequence[float] = (0.0, 0.0),
                 active: Optional[Sequence[int]] = None):
        """Reduced balance state from a reading (IMU attitude + gyro + ball channels)."""
        from controllers.classical import BalanceState  # local import: controllers depend on models

        yaw, pitch, roll = reading.euler
        mat = np.zeros(9)
        mujoco.mju_quat2Mat(mat, reading.imu_quat)
        Rz_t = _rz(-yaw)
        w = Rz_t @ (mat.reshape(3, 3) @ reading.gyro)
        err = Rz_t[:2, :2] @ (reading.ball_pos[:2] - np.asarray(target_position, dtype=float))
        v = Rz_t @ reading.ball_vel
        y = np.array([roll, pitch, w[0], w[1], err[0], err[1], v[0], v[1], w[2]])
        if active is not None:
            mask = np.zeros(9, dtype=bool)
            mask[list(active)] = True
            y[~mask] = 0.0
        return BalanceState.from_vector(y, yaw=yaw)


class ActuatorBus:
    """Pipeline delay + saturation + stepper pull-out torque curve for the wheel actuators."""

    def __init__(self, model: mujoco.MjModel, meta: SceneMetadata, hardware: Optional[HardwareProfile] = None,
                 motor: Optional[MotorParams] = None) -> None:
        self.model = model
        self.meta = meta
        self.hw = hardware or HardwareProfile()
        self.motor = motor or self.hw.motor
        self.mode = ControlMode(meta.control_mode or "motor")
        self.delay = DelayLine(model.nu, self.hw.delay_steps)
        self.wheel_act = np.array([mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, w.actuator)
                                   for w in meta.wheels], dtype=int)
        self.wheel_dof = np.array([model.jnt_dofadr[model.actuator_trnid[a, 0]] for a in self.wheel_act], dtype=int)
        self.nominal_forcerange = model.actuator_forcerange.copy()
        lo = model.actuator_ctrlrange[:, 0].copy()
        hi = model.actuator_ctrlrange[:, 1].copy()
        unl = ~model.actuator_ctrllimited.astype(bool)
        lo[unl], hi[unl] = -np.inf, np.inf
        self.lo, self.hi = lo, hi
        self.last_applied = np.zeros(model.nu)

    def set_delay(self, delay_s: float) -> None:
        self.hw.latency.delay = float(delay_s)
        self.hw.latency.validate()
        current = self.delay.current(10**12)
        self.delay = DelayLine(self.model.nu, self.hw.delay_steps, initial=current)

    def reset(self, initial: Optional[Sequence[float]] = None) -> None:
        self.delay = DelayLine(self.model.nu, self.hw.delay_steps,
                               initial=np.zeros(self.model.nu) if initial is None else initial)
        self.model.actuator_forcerange[:] = self.nominal_forcerange
        self.last_applied[:] = 0.0

    def submit(self, ctrl: Sequence[float], step: int) -> None:
        self.delay.submit(np.clip(np.asarray(ctrl, dtype=float), self.lo, self.hi), step)

    def apply(self, data: mujoco.MjData, step: int) -> np.ndarray:
        """Write the delayed, limited command for physics step ``step`` into ``data.ctrl``."""
        ctrl = self.delay.current(step).copy()
        if self.hw.torque_speed_curve and self.wheel_act.size:
            tau = np.atleast_1d(self.motor.pullout_wheel_torque(data.qvel[self.wheel_dof]))
            if self.mode == ControlMode.TORQUE:
                gear = self.model.actuator_gear[self.wheel_act, 0]
                lim = tau / gear
                ctrl[self.wheel_act] = np.clip(ctrl[self.wheel_act], -lim, lim)
            else:
                self.model.actuator_forcerange[self.wheel_act, 0] = -tau
                self.model.actuator_forcerange[self.wheel_act, 1] = tau
        data.ctrl[:] = ctrl
        self.last_applied = ctrl
        return ctrl

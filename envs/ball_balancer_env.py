"""Gymnasium environment for the ball balancers (Module 4).

``BallBalancerEnv`` wraps the BB-8 (or the planar 1d / 2d balancer) scene:

* physics at ``hardware.loop.physics_timestep`` (500 Hz), policy at
  ``control_rate_hz`` (100 Hz, ``frame_skip = 5``), every physics step through
  :class:`models.runtime.SceneStepper` (aligned omniwheel contacts, magnet,
  pushes) and :class:`models.io_bus.ActuatorBus` (delay ring buffer,
  saturation, stepper pull-out curve);
* observations from :class:`models.io_bus.SensorBus` (IMU noise / bias drift,
  1200 CPR encoders, ball sensor noise) - toggle with ``hardware.noise_enabled``
  and ``hardware.latency_enabled``. With ``hardware.ball_state_source="odometry"``
  (default) ball velocity / position come from wheel odometry through the exact
  estimator the robot runs (``deploy.runtime.estimator``);
* tasks ``station_keeping`` (hold the origin under random pushes) and
  ``velocity_tracking`` (follow resampled ``(vx*, vy*, wz*)`` commands);
* reward terms with tunable weights (:class:`RewardWeights`), returned per
  step in ``info["reward_terms"]``;
* domain randomisation applied directly to the compiled model at every reset:
  epoxy shell mass (+-15 %), ball/floor friction, omniwheel traction, head CoM
  offset and head mass.

Action (``Box(-1, 1)``):

* ``action_mode="body"`` (default, 3 values): relative ball rotation command
  about head x / y / z, mixed to the wheels through ``J`` (velocity drives:
  rate in units of ``max_body_rate``; torque drives: torque in units of
  ``max_body_torque``);
* ``action_mode="wheel"``: one value per wheel actuator, scaled to its ``ctrlrange``.

Observation (float32)::

    [roll, pitch, gyro_yawframe(3), wheel_vel / max_wheel_speed (n_wheels),
     ball_vel_yawframe(2), vel_error(2), yaw_rate_error, pos_error(2), prev_action(n_act)]
"""

from __future__ import annotations

import copy
import dataclasses
import math
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from config.composite_ball_params import (
    BB8Params,
    ControlMode,
    HardwareProfile,
    OmniwheelMode,
    PlanarBalancerParams,
    SimulationParams,
)
from controllers.classical import BalanceState, BalanceTarget, BallBotModel, BallWheelMixer
from models.io_bus import ActuatorBus, SensorBus
from models.mjcf_generator import SceneBundle, build_bb8, build_planar_balancer
from models.runtime import SceneStepper
from deploy.runtime.estimator import OdometryEstimator
from deploy.runtime.mathutil import quat_to_mat

__all__ = ["RewardWeights", "DomainRandomization", "PushConfig", "EnvConfig", "BallBalancerEnv", "ENV_ID"]

ENV_ID = "BB8BallBalancer-v0"


@dataclass
class RewardWeights:
    alive: float = 1.0                  # survival bonus per control step
    tilt: float = 20.0                  # * (roll^2 + pitch^2)
    velocity_tracking: float = 1.0      # * exp(-|v - v*|^2 / velocity_sigma)
    velocity_sigma: float = 0.05
    position: float = 0.5               # * exp(-|e|^2 / position_sigma)   (station keeping)
    position_sigma: float = 0.05
    yaw_rate_tracking: float = 0.3      # * exp(-(wz - wz*)^2 / yaw_sigma)
    yaw_sigma: float = 0.2
    angular_velocity: float = 0.05      # * (roll_rate^2 + pitch_rate^2)
    action: float = 0.02                # * |u_t|^2
    smoothness: float = 0.1             # * |u_t - u_{t-1}|^2
    fall: float = 50.0                  # one-off penalty on termination


@dataclass
class DomainRandomization:
    enabled: bool = True
    epoxy_mass_scale: float = 0.15      # +- fraction of the epoxy/sawdust shell mass
    ball_friction: Tuple[float, float] = (0.7, 1.1)   # multipliers
    floor_friction: Tuple[float, float] = (0.7, 1.2)
    wheel_friction: Tuple[float, float] = (0.8, 1.2)  # omniwheel traction (mu_tangent)
    head_com_offset_xy: float = 0.01    # [m] +- uniform
    head_com_offset_z: float = 0.015    # [m] +- uniform
    head_mass_scale: float = 0.10       # +- fraction
    latency_range: Optional[Tuple[float, float]] = None   # e.g. (0.0, 0.02) to randomise the delay [s]


@dataclass
class PushConfig:
    enabled: bool = True
    rate: float = 0.25                  # expected pushes per second
    force: Tuple[float, float] = (2.0, 6.0)             # [N]  (raise for a harder curriculum)
    duration: Tuple[float, float] = (0.05, 0.12)        # [s]
    body: str = "head"


@dataclass
class EnvConfig:
    scene: str = "bb8"                  # bb8 | planar_1d | planar_2d
    bb8: BB8Params = field(default_factory=BB8Params)
    planar: PlanarBalancerParams = field(default_factory=PlanarBalancerParams)
    hardware: HardwareProfile = field(default_factory=HardwareProfile)
    task: str = "station_keeping"       # station_keeping | velocity_tracking
    episode_seconds: float = 20.0
    action_mode: str = "body"           # body | wheel
    max_body_rate: float = 3.0          # [rad/s] relative ball rate at |a| = 1 (velocity drives)
    max_body_torque: float = 3.0        # [N m] relative torque at |a| = 1 (torque drives)
    max_tilt_deg: float = 30.0
    max_ball_distance: float = 4.0
    initial_tilt_deg: float = 3.0
    initial_rate: float = 0.1
    command_interval: Tuple[float, float] = (3.0, 6.0)  # [s] between velocity commands
    command_vx: Tuple[float, float] = (-0.4, 0.4)
    command_vy: Tuple[float, float] = (-0.4, 0.4)
    command_wz: Tuple[float, float] = (-0.6, 0.6)
    rewards: RewardWeights = field(default_factory=RewardWeights)
    randomization: DomainRandomization = field(default_factory=DomainRandomization)
    pushes: PushConfig = field(default_factory=PushConfig)

    def validate(self) -> None:
        if self.scene not in ("bb8", "planar_1d", "planar_2d"):
            raise ValueError("scene must be bb8, planar_1d or planar_2d")
        if self.task not in ("station_keeping", "velocity_tracking"):
            raise ValueError("task must be station_keeping or velocity_tracking")
        if self.action_mode not in ("body", "wheel"):
            raise ValueError("action_mode must be body or wheel")
        self.hardware.validate()


class BallBalancerEnv(gym.Env):
    """See module docstring."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 50}

    def __init__(self, config: Optional[EnvConfig] = None, render_mode: Optional[str] = None,
                 bundle: Optional[SceneBundle] = None) -> None:
        super().__init__()
        self.cfg = copy.deepcopy(config) if config is not None else EnvConfig()
        self.cfg.validate()
        self.render_mode = render_mode
        hw = self.cfg.hardware
        self.bundle = bundle or self._build_bundle()
        self.model = self.bundle.compile()
        self.data = mujoco.MjData(self.model)
        self.meta = self.bundle.meta
        self.stepper = SceneStepper(self.model, self.data, self.meta)
        self.bbm = BallBotModel(self.model, self.meta)
        self.mixer = BallWheelMixer.from_meta(self.meta)
        self.mode = ControlMode(self.meta.control_mode or "motor")
        if self.mode == ControlMode.POSITION and self.cfg.action_mode == "body":
            raise ValueError("body actions need velocity or torque wheel drives")
        self.frame_skip = max(1, int(round(hw.loop.control_dt / self.model.opt.timestep)))
        self.control_dt = self.frame_skip * self.model.opt.timestep
        self.sensors = SensorBus(self.model, self.meta, hw)
        self.actuators = ActuatorBus(self.model, self.meta, hw, motor=self._motor_params())
        # Robot-identical odometry (deploy.runtime.estimator) or noisy ground-truth ball sensors
        self.odometry = (OdometryEstimator(self.mixer.J, self.bbm.R, self.bbm.active, hw.odometry_velocity_tau)
                         if hw.ball_state_source == "odometry" else None)
        self.n_wheels = len(self.meta.wheels)
        self.n_act = 3 if self.cfg.action_mode == "body" else self.model.nu
        self.action_space = spaces.Box(-1.0, 1.0, shape=(self.n_act,), dtype=np.float32)
        obs_dim = 2 + 3 + self.n_wheels + 2 + 2 + 1 + 2 + self.n_act
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(obs_dim,), dtype=np.float32)
        self._nominal = self._capture_nominal()
        self.bbm.settle_reference(self.data, self.stepper)
        self._max_steps = int(round(self.cfg.episode_seconds / self.control_dt))
        self.target = BalanceTarget()
        self.prev_action = np.zeros(self.n_act)
        self.step_count = 0
        self.physics_step = 0
        self._next_command = 0.0
        self._renderer = None
        self.last_reading = None
        self.last_state: Optional[BalanceState] = None

    # ------------------------------------------------------------------ setup
    def _build_bundle(self) -> SceneBundle:
        cfg, hw = self.cfg, self.cfg.hardware
        sim = SimulationParams(timestep=hw.loop.physics_timestep)
        if cfg.scene == "bb8":
            p = copy.deepcopy(cfg.bb8)
            p.task_mode = cfg.task
            return build_bb8(p, sim)
        p = copy.deepcopy(cfg.planar)
        p.variant = "1d" if cfg.scene == "planar_1d" else "2d"
        return build_planar_balancer(p, sim)

    def _motor_params(self):
        return self.cfg.bb8.head.motor if self.cfg.scene == "bb8" else self.cfg.planar.motor

    def _capture_nominal(self) -> Dict[str, Any]:
        m = self.model
        ball = self.bbm.ball_id
        head = self.bbm.head_id
        ball_geom = m.geom("ball_geom").id
        floor = m.geom("floor").id
        roller_geoms = [m.geom(g).id for w in self.meta.wheels for g in w.collision_geoms]
        return {
            "ball_mass": float(m.body_mass[ball]),
            "ball_inertia": m.body_inertia[ball].copy(),
            "head_mass": float(m.body_mass[head]),
            "head_inertia": m.body_inertia[head].copy(),
            "head_ipos": m.body_ipos[head].copy(),
            "ball_friction": m.geom_friction[ball_geom].copy(),
            "floor_friction": m.geom_friction[floor].copy(),
            "roller_friction": m.geom_friction[roller_geoms].copy(),
            "pair_friction": m.pair_friction.copy(),
            "ids": (ball, head, ball_geom, floor, roller_geoms),
        }

    def _ball_params(self):
        return self.cfg.bb8.ball if self.cfg.scene == "bb8" else self.cfg.planar.ball

    def _randomize(self) -> Dict[str, float]:
        m, nom, dr = self.model, self._nominal, self.cfg.randomization
        ball, head, ball_geom, floor, rollers = nom["ids"]
        # restore nominal
        m.body_mass[ball] = nom["ball_mass"]
        m.body_inertia[ball] = nom["ball_inertia"]
        m.body_mass[head] = nom["head_mass"]
        m.body_inertia[head] = nom["head_inertia"]
        m.body_ipos[head] = nom["head_ipos"]
        m.geom_friction[ball_geom] = nom["ball_friction"]
        m.geom_friction[floor] = nom["floor_friction"]
        m.geom_friction[rollers] = nom["roller_friction"]
        m.pair_friction[:] = nom["pair_friction"]
        info: Dict[str, float] = {}
        if not dr.enabled:
            mujoco.mj_setConst(m, self.data)
            return info
        rng = self.np_random
        scale = 1.0 + rng.uniform(-dr.epoxy_mass_scale, dr.epoxy_mass_scale)
        ball_p = self._ball_params().scaled(epoxy_mass_scale=scale)
        m.body_mass[ball] = ball_p.mass
        m.body_inertia[ball] = ball_p.inertia
        fb = rng.uniform(*dr.ball_friction)
        ff = rng.uniform(*dr.floor_friction)
        fw = rng.uniform(*dr.wheel_friction)
        m.geom_friction[ball_geom, 0] = nom["ball_friction"][0] * fb
        m.geom_friction[floor, 0] = nom["floor_friction"][0] * ff
        m.geom_friction[rollers, 0] = nom["roller_friction"][:, 0] * fw
        if m.npair:
            m.pair_friction[:, 0] = nom["pair_friction"][:, 0] * fw
        hs = 1.0 + rng.uniform(-dr.head_mass_scale, dr.head_mass_scale)
        m.body_mass[head] = nom["head_mass"] * hs
        m.body_inertia[head] = nom["head_inertia"] * hs
        off = np.array([rng.uniform(-dr.head_com_offset_xy, dr.head_com_offset_xy),
                        rng.uniform(-dr.head_com_offset_xy, dr.head_com_offset_xy),
                        rng.uniform(-dr.head_com_offset_z, dr.head_com_offset_z)])
        m.body_ipos[head] = nom["head_ipos"] + off
        if dr.latency_range is not None:
            self.actuators.set_delay(float(rng.uniform(*dr.latency_range)))
        mujoco.mj_setConst(m, self.data)
        info.update(epoxy_scale=scale, ball_mass=float(ball_p.mass), ball_friction=fb, floor_friction=ff,
                    wheel_friction=fw, head_mass_scale=hs, head_com_dx=off[0], head_com_dy=off[1],
                    head_com_dz=off[2], delay=self.cfg.hardware.latency.delay)
        return info

    # ------------------------------------------------------------------ gym API
    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        super().reset(seed=seed)
        rng = self.np_random
        self.stepper.reset_state()
        mujoco.mj_resetData(self.model, self.data)
        dr_info = self._randomize()
        tilt = math.radians(self.cfg.initial_tilt_deg)
        y0 = np.zeros(9)
        y0[0:2] = rng.uniform(-tilt, tilt, 2)
        y0[2:4] = rng.uniform(-self.cfg.initial_rate, self.cfg.initial_rate, 2)
        if options and "initial_state" in options:
            y0 = np.asarray(options["initial_state"], dtype=float)
        self.bbm.lift(self.data, y0)
        mujoco.mj_forward(self.model, self.data)
        self.target = BalanceTarget(position=self.data.xpos[self.bbm.ball_id][:2].copy())
        self._next_command = 0.0 if self.cfg.task == "velocity_tracking" else np.inf
        self.sensors.reset(rng)
        self.actuators.reset()
        self.prev_action = np.zeros(self.n_act)
        self.step_count = 0
        self.physics_step = 0
        self._update_command()
        if self.odometry is not None:
            self.odometry.reset(position=self.data.xpos[self.bbm.ball_id][:2].copy())
        reading = self.sensors.read(self.data)
        obs = self._observation(reading, dt=0.0)
        return obs, {"randomization": dr_info}

    def _update_command(self) -> None:
        t = self.step_count * self.control_dt
        if t + 1e-9 < self._next_command:
            return
        c, rng = self.cfg, self.np_random
        v = np.array([rng.uniform(*c.command_vx), rng.uniform(*c.command_vy)])
        wz = rng.uniform(*c.command_wz) if self.cfg.scene == "bb8" else 0.0
        if self.cfg.scene == "planar_1d":
            v[1] = 0.0
        self.target.command(v, wz)
        self._next_command = t + rng.uniform(*c.command_interval)

    def _maybe_push(self) -> None:
        pc = self.cfg.pushes
        if not pc.enabled or pc.rate <= 0:
            return
        rng = self.np_random
        if rng.random() < pc.rate * self.control_dt:
            ang = rng.uniform(0, 2 * math.pi)
            if self.cfg.scene == "planar_1d":
                ang = 0.0 if rng.random() < 0.5 else math.pi
            f = rng.uniform(*pc.force)
            self.stepper.push(pc.body, [f * math.cos(ang), f * math.sin(ang), 0.0], rng.uniform(*pc.duration))

    def action_to_ctrl(self, action: np.ndarray) -> np.ndarray:
        a = np.clip(np.asarray(action, dtype=float), -1.0, 1.0)
        m = self.model
        if self.cfg.action_mode == "wheel":
            lo, hi = m.actuator_ctrlrange[:, 0], m.actuator_ctrlrange[:, 1]
            return lo + (a + 1.0) * 0.5 * (hi - lo)
        cmd = a.copy()
        variant = self.bbm.variant
        if variant == "1d":
            cmd[[0, 2]] = 0.0
        elif variant == "2d":
            cmd[2] = 0.0
        ctrl = np.zeros(m.nu)
        act_ids = self.actuators.wheel_act
        if self.mode == ControlMode.VELOCITY:
            ctrl[act_ids] = self.mixer.rel_rate_to_wheels(cmd * self.cfg.max_body_rate)
        else:
            gear = m.actuator_gear[act_ids, 0]
            ctrl[act_ids] = self.mixer.torque_to_wheels(cmd * self.cfg.max_body_torque) / gear
        lo, hi = m.actuator_ctrlrange[:, 0], m.actuator_ctrlrange[:, 1]
        lim = m.actuator_ctrllimited.astype(bool)
        ctrl[lim] = np.clip(ctrl[lim], lo[lim], hi[lim])
        return ctrl

    def _estimate(self, reading, dt: float) -> BalanceState:
        if self.odometry is not None:
            s = self.odometry.update(quat_to_mat(reading.imu_quat), reading.gyro, reading.wheel_vel, dt,
                                     self.target.position)
            return BalanceState.from_vector(s.vector(), yaw=s.yaw)
        return SensorBus.estimate(reading, self.target.position, self.bbm.active)

    def _observation(self, reading, dt: Optional[float] = None) -> np.ndarray:
        state = self._estimate(reading, self.control_dt if dt is None else dt)
        self.last_reading, self.last_state = reading, state
        err = state.error(self.target)
        max_w = max(self._motor_params().max_wheel_speed, 1e-6)
        obs = np.concatenate([
            [state.roll, state.pitch],
            [state.roll_rate, state.pitch_rate, state.yaw_rate],
            reading.wheel_vel / max_w,
            [state.vx, state.vy],
            err[6:8],
            [err[8]],
            err[4:6],
            self.prev_action,
        ])
        return obs.astype(np.float32)

    def _reward(self, truth: BalanceState, action: np.ndarray) -> Tuple[float, Dict[str, float]]:
        w = self.cfg.rewards
        e = truth.error(self.target)
        vel_err2 = float(e[6] ** 2 + e[7] ** 2)
        terms = {
            "alive": w.alive,
            "tilt": -w.tilt * float(truth.roll ** 2 + truth.pitch ** 2),
            "velocity_tracking": w.velocity_tracking * math.exp(-vel_err2 / w.velocity_sigma),
            "position": (w.position * math.exp(-float(e[4] ** 2 + e[5] ** 2) / w.position_sigma)
                         if self.cfg.task == "station_keeping" else 0.0),
            "yaw_rate_tracking": (w.yaw_rate_tracking * math.exp(-float(e[8] ** 2) / w.yaw_sigma)
                                  if self.cfg.scene == "bb8" else 0.0),
            "angular_velocity": -w.angular_velocity * float(truth.roll_rate ** 2 + truth.pitch_rate ** 2),
            "action": -w.action * float(np.sum(action ** 2)),
            "smoothness": -w.smoothness * float(np.sum((action - self.prev_action) ** 2)),
        }
        return float(sum(terms.values())), terms

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=float).reshape(self.n_act), -1.0, 1.0)
        return self._advance(self.action_to_ctrl(action), action)

    def step_ctrl(self, ctrl):
        """Step with raw actuator ``ctrl`` (classical controllers through the same I/O pipeline).

        The reward's action terms then see zeros; everything else is identical to :meth:`step`.
        """
        return self._advance(np.asarray(ctrl, dtype=float), np.zeros(self.n_act))

    def _advance(self, ctrl: np.ndarray, action: np.ndarray):
        self._update_command()
        self._maybe_push()
        self.actuators.submit(ctrl, self.physics_step)
        for _ in range(self.frame_skip):
            self.actuators.apply(self.data, self.physics_step)
            self.stepper.step(1)
            self.physics_step += 1
        self.target.advance(self.control_dt)
        self.step_count += 1

        truth = self.bbm.measure(self.data, self.target)
        reward, terms = self._reward(truth, action)
        ball_dist = float(np.linalg.norm(self.data.xpos[self.bbm.ball_id][:2]))
        tilt = math.degrees(max(abs(truth.roll), abs(truth.pitch)))
        finite = bool(np.all(np.isfinite(self.data.qpos)) and np.all(np.isfinite(self.data.qvel)))
        terminated = (not finite) or tilt > self.cfg.max_tilt_deg or ball_dist > self.cfg.max_ball_distance
        if terminated:
            reward -= self.cfg.rewards.fall
            terms["fall"] = -self.cfg.rewards.fall
        truncated = self.step_count >= self._max_steps
        self.prev_action = action          # the observation carries the action just applied
        reading = self.sensors.read(self.data) if finite else self.last_reading
        obs = self._observation(reading) if finite else np.zeros(self.observation_space.shape, np.float32)
        info = {
            "reward_terms": terms,
            "tilt_deg": tilt,
            "state": truth.vector(),
            "target_velocity": np.asarray(self.target.velocity, dtype=float).copy(),
            "target_yaw_rate": float(self.target.yaw_rate),
            "ctrl": ctrl,
        }
        return obs, reward, bool(terminated), bool(truncated), info

    def render(self):
        if self.render_mode != "rgb_array":
            return None
        if self._renderer is None:
            try:
                self._renderer = mujoco.Renderer(self.model, 480, 640)
            except Exception as exc:  # no OpenGL context (headless server)
                raise RuntimeError("offscreen rendering needs an OpenGL backend (set MUJOCO_GL=egl/osmesa); "
                                   "use the web studio for visualisation") from exc
        self._renderer.update_scene(self.data, camera="chase" if self.cfg.scene == "bb8" else -1)
        return self._renderer.render()

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
        self.stepper.close()


def _register() -> None:
    if ENV_ID not in gym.registry:
        gym.register(id=ENV_ID, entry_point="envs.ball_balancer_env:BallBalancerEnv")


_register()

"""Software-in-the-loop for the robot runtime (development PC).

* :class:`FirmwareModel` - Python mirror of ``firmware/bb8_mcu/bb8_mcu.ino``:
  acceleration-limited step rates, step counters, watchdog, enable logic.
* :class:`SimIO` - the robot runtime's IO interface backed by the MuJoCo BB-8
  (``envs.ball_balancer_env``): IMU samples from the noisy sensor bus, encoder
  counts quantised to ``encoder_cpr`` with the configured signs, step-rate
  commands through :class:`FirmwareModel` into the stepper servo model, and
  disabled drivers leave the wheels free. The unmodified
  :class:`deploy.runtime.runner.BalanceRunner` runs against it::

      python -m deploy.sim_io --bundle bundle/ --seconds 20 --push 6

* :class:`PtyMCU` - the firmware protocol on a pseudo-terminal, so
  :class:`deploy.runtime.mcu.SerialMCU` can be exercised over a real serial
  device without hardware.
"""

from __future__ import annotations

import argparse
import copy
import math
import os
import sys
import threading
import time
from typing import List, Optional, Sequence

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from deploy.runtime import protocol as P  # noqa: E402
from deploy.runtime.bundle import Bundle  # noqa: E402
from deploy.runtime.config import RobotConfig  # noqa: E402
from deploy.runtime.imu import ComplementaryFilter, ImuSample  # noqa: E402
from deploy.runtime.runner import BalanceRunner  # noqa: E402


class FirmwareModel:
    """Behaviour of the microcontroller firmware (kept in sync with bb8_mcu.ino)."""

    def __init__(self, n: int, max_accel: float = 150000.0, watchdog_ms: int = 100) -> None:
        self.n = n
        self.max_accel = float(max_accel)          # steps/s^2
        self.watchdog_us = int(watchdog_ms) * 1000
        self.target = np.zeros(n)
        self.rate = np.zeros(n)
        self.position = np.zeros(n)                # fractional steps
        self.enabled = False
        self.enable_req = False
        self.watchdog = False
        self.configured = False
        self.last_cmd_us: Optional[int] = None
        self.seq_echo = 0

    def on_cfg(self, cfg: P.Cfg) -> None:
        self.max_accel, self.watchdog_us, self.configured = float(cfg.max_accel), int(cfg.watchdog_ms) * 1000, True

    def on_cmd(self, cmd: P.Cmd, now_us: int) -> None:
        k = min(len(cmd.step_rates), self.n)
        self.target[:] = 0.0
        self.target[:k] = cmd.step_rates[:k]
        self.enable_req = cmd.enable
        self.watchdog = False
        self.last_cmd_us = now_us
        self.seq_echo = cmd.seq

    def update(self, now_us: int, dt: float) -> None:
        if self.last_cmd_us is not None and now_us - self.last_cmd_us > self.watchdog_us:
            self.watchdog = True
            self.enable_req = False
            self.target[:] = 0.0
        if self.enable_req:
            self.enabled = True
        step = self.max_accel * dt
        self.rate += np.clip(self.target - self.rate, -step, step)
        if not self.enable_req and np.all(self.rate == 0.0):
            self.enabled = False
        if not self.enabled:
            self.rate[:] = 0.0
        self.position += self.rate * dt

    @property
    def flags(self) -> int:
        return ((P.TF_ENABLED if self.enabled else 0) | (P.TF_WATCHDOG if self.watchdog else 0)
                | (P.TF_CONFIGURED if self.configured else 0))


class SimIO:
    """IO backend for :class:`BalanceRunner` on the MuJoCo BB-8."""

    def __init__(self, bundle: Bundle, cfg: RobotConfig, env_config=None, seed: int = 0,
                 detent_torque: float = 0.02, hold_until_armed: bool = True,
                 residual_gyro_bias_std: float = 0.002) -> None:
        """``hold_until_armed``: like an operator, keep the robot upright and still until the
        drivers are first enabled (the unpowered head would otherwise fall off the ball).
        ``residual_gyro_bias_std``: gyro bias left after calibration [rad/s]; with Jetson-side
        fusion it becomes a steady tilt error of bias / fusion_kp."""
        from config.composite_ball_params import params_from_dict
        from envs.ball_balancer_env import BallBalancerEnv, EnvConfig

        if env_config is None:
            env_config = EnvConfig()
            env_config.bb8 = params_from_dict(type(env_config.bb8), bundle.data["params"]["bb8"])
            env_config.hardware = params_from_dict(type(env_config.hardware), bundle.data["params"]["hardware"])
            env_config.randomization.enabled = False
            env_config.pushes.enabled = False
        env_config = copy.deepcopy(env_config)
        env_config.max_tilt_deg = 1e9                 # the runner's safety logic decides, not the env
        env_config.max_ball_distance = 1e9
        env_config.episode_seconds = 1e9
        env_config.initial_tilt_deg = min(env_config.initial_tilt_deg, 1.0)   # held roughly upright ...
        self.env = BallBalancerEnv(env_config)
        self.env.reset(seed=seed)
        if hold_until_armed:
            self.env.data.qvel[:] = 0.0                                        # ... and still
            if self.env.model.na:
                self.env.data.act[:] = 0.0
        self.cfg = cfg
        self.n = bundle.n_wheels
        self.fw = FirmwareModel(self.n, cfg["max_accel"] * cfg.steps_per_rad, cfg["watchdog_ms"])
        self.fw.configured = True
        self.motor_sign = np.asarray(cfg["motor_sign"], dtype=float)
        self.enc_sign = np.asarray(cfg["encoder_sign"], dtype=float)
        self.detent = detent_torque
        self.holding = hold_until_armed
        m = self.env.model
        self.wheel_qadr = np.array([m.jnt_qposadr[m.joint(w.joint).id] for w in self.env.meta.wheels])
        self.dt = self.env.control_dt
        self.physics_dt = m.opt.timestep
        # IMU model: Jetson-side fusion of raw gyro/accel (arduino, mpu6050) at physics rate,
        # or the sensor bus' orientation-noise model (bno085, fused on-chip)
        self.fusion = cfg["imu"]["type"] in ("arduino", "mpu6050")
        hw = self.env.cfg.hardware
        self.rng = np.random.default_rng(seed + 17)
        noisy = hw.noise_enabled
        self.imu_noise = (hw.imu.gyro_noise_std if noisy else 0.0, hw.imu.accel_noise_std if noisy else 0.0)
        # gyro bias left after `runner calibrate` (the raw bias, hw.imu.gyro_bias_std, is calibrated out)
        self.residual_bias = self.rng.normal(0.0, residual_gyro_bias_std, 3) if noisy else np.zeros(3)
        self.filter = ComplementaryFilter(kp=cfg["imu"]["fusion_kp"])
        base = self.env.meta.imu_site
        sid = lambda name: (m.sensor_adr[m.sensor(name).id], 3)          # noqa: E731
        self._gyro_adr, self._acc_adr = sid(base + "_gyro")[0], sid(base + "_acc")[0]
        self._last_gyro, self._last_acc = np.zeros(3), np.array([0.0, 0.0, 9.81])
        import mujoco as _mj

        _mj.mj_forward(m, self.env.data)
        self._imu_sample(held=hold_until_armed)
        self.frame_skip = self.env.frame_skip
        self.pushes: List[tuple] = []          # (time, body, force, duration)
        self._last_us = self._now_us()
        self.log: List[dict] = []

    # ------------------------------------------------------------- runner IO API
    def _now_us(self) -> int:
        return int(round(self.env.data.time * 1e6))

    def now(self) -> float:
        return float(self.env.data.time)

    def _imu_sample(self, held: bool = False) -> None:
        sd = self.env.data.sensordata
        gn, an = self.imu_noise
        if held:      # held still by hand: the IMU feels only gravity (frozen sim data may hold contact accels)
            H = self.env.data.xmat[self.env.bbm.head_id].reshape(3, 3)
            g_true, a_true = np.zeros(3), H.T @ np.array([0.0, 0.0, 9.81])
        else:
            g_true, a_true = sd[self._gyro_adr:self._gyro_adr + 3], sd[self._acc_adr:self._acc_adr + 3]
        g = g_true + self.residual_bias + self.rng.normal(0.0, gn, 3)
        a = a_true + self.rng.normal(0.0, an, 3)
        self._last_gyro, self._last_acc = g, a
        self.filter.update(g, a, self.physics_dt)     # the residual (post-calibration) bias stays in

    def odometry(self, telem):
        src = self.cfg["odometry_source"]
        return (telem.steps if src == "steps" else telem.encoders), telem.mcu_us

    def read(self):
        env = self.env
        if self.fusion:
            imu = ImuSample(self.filter.q.copy(), self._last_gyro.copy(), self._last_acc.copy(), float(env.data.time))
        else:
            r = env.sensors.read(env.data)
            imu = ImuSample(r.imu_quat, r.gyro, r.accel, r.time)
        angles = env.data.qpos[self.wheel_qadr]
        enc = np.floor(self.enc_sign * angles * self.cfg.counts_per_rad).astype(np.int64)
        steps = np.floor(self.fw.position).astype(np.int64)
        telem = P.Telem(self.fw.seq_echo, self._now_us() & 0xFFFFFFFF, self.fw.flags,
                        [int(v) for v in enc], [int(v) for v in steps])
        return imu, telem

    def write(self, step_rates: Sequence[float], enable: bool) -> None:
        self.fw.on_cmd(P.Cmd(0, bool(enable), list(step_rates)), self._now_us())

    def wait_tick(self) -> None:
        env = self.env
        m, d = env.model, env.data
        for t_push, body, force, dur in list(self.pushes):
            if t_push <= d.time < t_push + self.dt:
                env.stepper.push(body, force, dur)
                self.pushes.remove((t_push, body, force, dur))
        if self.holding:
            for _ in range(self.frame_skip):
                self.fw.update(self._now_us(), self.physics_dt)
                if self.fw.enabled:
                    self.holding = False              # released: physics takes over
                    break
                d.time += self.physics_dt             # held still by hand: time passes, nothing moves
                if self.fusion:
                    self._imu_sample(held=True)
            if self.holding:
                self.log.append({"t": float(d.time), "tilt_deg": 0.0, "ball_xy": d.xpos[env.bbm.ball_id][:2].copy(),
                                 "enabled": False})
                return
        for _ in range(self.frame_skip):
            self.fw.update(self._now_us(), self.physics_dt)
            wheel_cmd = self.fw.rate * self.motor_sign / self.cfg.steps_per_rad
            ctrl = np.zeros(m.nu)
            ctrl[env.actuators.wheel_act] = wheel_cmd
            env.actuators.submit(ctrl, env.physics_step)
            env.actuators.apply(d, env.physics_step)
            if not self.fw.enabled:                       # unpowered stepper: only detent torque
                m.actuator_forcerange[env.actuators.wheel_act, 0] = -self.detent
                m.actuator_forcerange[env.actuators.wheel_act, 1] = self.detent
            env.stepper.step(1)
            env.physics_step += 1
            if self.fusion:
                self._imu_sample()
        truth = env.bbm.measure(d)
        self.log.append({"t": float(d.time), "tilt_deg": math.degrees(max(abs(truth.roll), abs(truth.pitch))),
                         "ball_xy": d.xpos[env.bbm.ball_id][:2].copy(), "enabled": self.fw.enabled})

    def close(self) -> None:
        pass

    def schedule_push(self, t: float, force, body: str = "head", duration: float = 0.1) -> None:
        self.pushes.append((float(t), body, list(force), float(duration)))


def run_sil(bundle_dir: str, controller: Optional[str] = None, seconds: float = 10.0,
            cfg: Optional[RobotConfig] = None, pushes=(), env_config=None, seed: int = 0, auto_arm: bool = True):
    """Run the robot runtime against the simulator. Returns (runner, io)."""
    bundle = Bundle.load(bundle_dir)
    cfg = cfg or RobotConfig({"motor_sign": [1] * bundle.n_wheels, "encoder_sign": [1] * bundle.n_wheels})
    cfg.validate(bundle.n_wheels)
    io = SimIO(bundle, cfg, env_config, seed)
    for t, f in pushes:
        io.schedule_push(t, f)
    runner = BalanceRunner(bundle, cfg, io, controller, auto_arm=auto_arm)
    runner.run(max_ticks=int(round(seconds / io.dt)), status_every=0)
    return runner, io


class PtyMCU:
    """Firmware emulator on a pseudo-terminal (``self.port`` is the device path for pyserial)."""

    def __init__(self, n: int = 3, telem_period_s: float = 0.02, steps_to_counts: float = 1200 / 3200) -> None:
        import tty

        self.master, slave = os.openpty()
        tty.setraw(self.master)
        tty.setraw(slave)
        self.port = os.ttyname(slave)
        self._slave = slave
        self.fw = FirmwareModel(n)
        self.parser = P.StreamParser()
        self.telem_period = telem_period_s
        self.k_enc = steps_to_counts
        self._stop = threading.Event()
        self._t0 = time.monotonic()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _us(self) -> int:
        return int((time.monotonic() - self._t0) * 1e6)

    def _send_telem(self) -> None:
        steps = [int(math.floor(p)) for p in self.fw.position]
        enc = [int(math.floor(p * self.k_enc)) for p in self.fw.position]
        os.write(self.master, P.encode_telem(self.fw.seq_echo, self._us() & 0xFFFFFFFF, self.fw.flags, enc, steps))

    def _run(self) -> None:
        import select

        last = self._us()
        last_telem = 0.0
        while not self._stop.is_set():
            r, _, _ = select.select([self.master], [], [], 0.001)
            now = self._us()
            self.fw.update(now, (now - last) * 1e-6)
            last = now
            if r:
                try:
                    data = os.read(self.master, 4096)
                except OSError:
                    break
                for ftype, payload in self.parser.feed(data):
                    try:
                        msg = P.decode(ftype, payload)
                    except ValueError:
                        continue
                    if isinstance(msg, P.Cfg):
                        self.fw.on_cfg(msg)
                    elif isinstance(msg, P.Cmd):
                        self.fw.on_cmd(msg, now)
                        self._send_telem()
                        last_telem = time.monotonic()
            if time.monotonic() - last_telem > self.telem_period:
                self._send_telem()
                last_telem = time.monotonic()

    def close(self) -> None:
        self._stop.set()
        self.thread.join(timeout=1)
        os.close(self.master)
        os.close(self._slave)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Run the robot runtime against the MuJoCo BB-8 (software in the loop)")
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--config", help="robot config JSON (signs / steps / encoder resolution)")
    ap.add_argument("--controller", choices=["pid", "lqr", "rl"])
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--push", type=float, default=0.0, help="push the head with this force [N] every 4 s")
    ap.add_argument("--randomize", action="store_true", help="domain randomisation of the simulated robot")
    args = ap.parse_args(argv)
    bundle = Bundle.load(args.bundle)
    cfg = RobotConfig.load(args.config) if args.config else None
    env_config = None
    if args.randomize:
        from config.composite_ball_params import params_from_dict
        from envs.ball_balancer_env import EnvConfig

        env_config = EnvConfig()
        env_config.bb8 = params_from_dict(type(env_config.bb8), bundle.data["params"]["bb8"])
        env_config.hardware = params_from_dict(type(env_config.hardware), bundle.data["params"]["hardware"])
        env_config.pushes.enabled = False
    pushes = [(t, [args.push * math.cos(t), args.push * math.sin(t), 0.0])
              for t in np.arange(3.0, args.seconds, 4.0)] if args.push else []
    t0 = time.time()
    runner, io = run_sil(args.bundle, args.controller, args.seconds, cfg, pushes, env_config)
    tilts = [row["tilt_deg"] for row in io.log]
    print(f"controller={runner.ctrl.name} final mode={runner.mode} {runner.fault_reason}")
    print(f"max tilt {max(tilts):.2f} deg, final ball position {io.log[-1]['ball_xy'].round(3)} m, "
          f"{len(io.log)} ticks in {time.time() - t0:.1f} s wall")
    return 0 if runner.mode == "ARMED" else 1


if __name__ == "__main__":
    sys.exit(main())

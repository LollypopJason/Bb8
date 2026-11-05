"""BB-8 balance runner for the Jetson Nano (Python >= 3.6, numpy + pyserial).

Hardware (default): Arduino Nano 33 IoT IMU on USB (``firmware/bb8_imu``) and stepper
drivers on Jetson GPIO driven by the ``jetson_stepgen/bb8_stepgen`` daemon. A Teensy
(``firmware/bb8_mcu``) and I2C IMUs (BNO085 / MPU6050) are supported alternatives.

Usage on the robot::

    python3 -m deploy.runtime.runner write-config robot.json        # then edit ports / signs / IMU
    python3 -m deploy.runtime.runner calibrate --config robot.json  # robot held upright and still
    sudo ./deploy/jetson_stepgen/bb8_stepgen &                      # or the systemd service
    python3 -m deploy.runtime.runner spin-test --config robot.json  # robot OFF the ball: checks signs
    python3 -m deploy.runtime.runner run --bundle bundle/ --config robot.json --controller pid

Control loop (``control_rate_hz``, default 100 Hz):

1. newest MCU telemetry (encoder counts, MCU clock) + IMU sample
2. wheel speeds from count deltas over the MCU clock, balance state from
   :class:`deploy.runtime.estimator.OdometryEstimator`
3. safety state machine (DISARMED -> ARMED -> FAULT), teleop target update
4. controller (PID / LQR / RL policy) -> wheel speeds -> signed step rates -> MCU

Safety: the motors are only enabled while ARMED. Arming needs an arm request
(``--auto-arm``, Enter on the console or a teleop "arm" message) and the head
held within ``arm_tilt_deg`` for ``arm_hold_s``. Any tilt beyond
``tilt_cutoff_deg``, stale MCU telemetry, or a firmware watchdog trip drops to
FAULT (motors disabled); re-arming requires a new arm request. The firmware
independently stops the motors if commands stop for ``watchdog_ms``.
"""

import argparse
import csv
import json
import math
import os
import select
import sys
import time

import numpy as np

from .bundle import Bundle
from .config import RobotConfig
from .controllers import Target, make_controller
from .estimator import OdometryEstimator, WheelSpeedEstimator

DISARMED, ARMED, FAULT = "DISARMED", "ARMED", "FAULT"


# --------------------------------------------------------------------------- IO backends
class HardwareIO(object):
    """Real robot: step generator (Jetson GPIO daemon or USB MCU) + IMU, real-time tick scheduling."""

    def __init__(self, cfg, n_wheels, mcu=None, imu=None):
        from .imu import make_imu
        from .mcu import make_motor_link

        self.cfg = cfg
        self.n = n_wheels
        self.mcu = mcu or make_motor_link(cfg, n_wheels)
        self.imu = imu or make_imu(cfg)
        self.period = 1.0 / cfg["control_rate_hz"]
        self.next_t = None
        self.overruns = 0

    def start(self):
        self.mcu.configure()
        self.mcu.send([0.0] * self.n, False)
        self.mcu.wait_for_telemetry()
        self.next_t = time.monotonic()

    def read(self):
        from .imu import ImuTimeout

        telem = self.mcu.poll()
        fresh = self.mcu.telem_time is not None and time.monotonic() - self.mcu.telem_time < \
            self.cfg["safety"]["telemetry_timeout_s"]
        try:
            imu = self.imu.read()
        except ImuTimeout:
            imu = None
        return imu, (telem if fresh else None)

    def odometry(self, telem):
        """(counts, microsecond clock) of the configured odometry source."""
        src = self.cfg["odometry_source"]
        if src == "imu_encoders":
            return self.imu.raw.encoders, self.imu.raw.encoder_us
        return (telem.encoders if src == "mcu_encoders" else telem.steps), telem.mcu_us

    def write(self, step_rates, enable):
        self.mcu.send(step_rates, enable)

    def wait_tick(self):
        self.next_t += self.period
        delay = self.next_t - time.monotonic()
        if delay > 0:
            time.sleep(delay)
        else:
            self.overruns += 1
            if delay < -0.1:
                self.next_t = time.monotonic()

    def now(self):
        return time.monotonic()

    def close(self):
        self.mcu.close()


# --------------------------------------------------------------------------- teleop
class UdpTeleop(object):
    """JSON datagrams: {"vx": m/s, "vy": m/s, "wz": rad/s, "arm": bool, "disarm": bool}."""

    def __init__(self, port, max_speed, max_yaw_rate, timeout_s):
        import socket

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(("0.0.0.0", port))
        self.sock.setblocking(False)
        self.max_speed, self.max_yaw, self.timeout = max_speed, max_yaw_rate, timeout_s
        self.cmd = (0.0, 0.0, 0.0)
        self.last = 0.0

    def poll(self, now):
        events = []
        while True:
            try:
                data, _ = self.sock.recvfrom(1024)
            except (BlockingIOError, OSError):
                break
            try:
                msg = json.loads(data.decode("utf-8"))
            except ValueError:
                continue
            vx = float(np.clip(msg.get("vx", 0.0), -self.max_speed, self.max_speed))
            vy = float(np.clip(msg.get("vy", 0.0), -self.max_speed, self.max_speed))
            wz = float(np.clip(msg.get("wz", 0.0), -self.max_yaw, self.max_yaw))
            self.cmd, self.last = (vx, vy, wz), now
            for key in ("arm", "disarm"):
                if msg.get(key):
                    events.append(key)
        if now - self.last > self.timeout:
            self.cmd = (0.0, 0.0, 0.0)
        return self.cmd, events


# --------------------------------------------------------------------------- runner
class BalanceRunner(object):
    def __init__(self, bundle, cfg, io, controller=None, policy_backend="numpy", auto_arm=False, log_path=None,
                 teleop=None, console=False):
        cfg.validate(bundle.n_wheels)
        self.bundle, self.cfg, self.io = bundle, cfg, io
        self.n = bundle.n_wheels
        self.ctrl = make_controller(bundle, controller, policy_backend)
        self.target = Target.from_bundle(bundle)
        tau = float(bundle.section("robot").get("odometry_velocity_tau", 0.0))
        self.estimator = OdometryEstimator(bundle.J, bundle.R, bundle.active, tau)
        use_enc = cfg["odometry_source"] != "steps"
        k = cfg.counts_per_rad if use_enc else cfg.steps_per_rad
        self.wheels = WheelSpeedEstimator(k, cfg["encoder_sign"] if use_enc else cfg["motor_sign"])
        self.motor_sign = np.asarray(cfg["motor_sign"], dtype=float)
        lim = cfg["safety"]["max_wheel_speed"]
        self.max_wheel_speed = min(bundle.max_wheel_speed, lim) if lim else bundle.max_wheel_speed
        self.dt = 1.0 / cfg["control_rate_hz"]
        if abs(self.dt - bundle.control_dt) > 1e-6:
            raise ValueError("config control rate (%.1f Hz) differs from the bundle's design rate (%.1f Hz)"
                             % (1.0 / self.dt, 1.0 / bundle.control_dt))
        self.mode = DISARMED
        self.arm_requested = bool(auto_arm)
        self.upright_since = None
        self.fault_reason = ""
        self.teleop = teleop
        self.console = console
        self.ticks = 0
        self.state = None
        self.last_cmd = np.zeros(self.n)
        self.log_rows = []
        self.log_path = log_path
        self._log_file = None
        self._log_writer = None

    # ------------------------------------------------------------- helpers
    def step_rates(self, wheel_speeds):
        w = np.clip(np.asarray(wheel_speeds, dtype=float), -self.max_wheel_speed, self.max_wheel_speed)
        return (self.motor_sign * w * self.cfg.steps_per_rad).tolist()

    def _arm(self):
        self.mode = ARMED
        self.fault_reason = ""
        self.estimator.reset()
        self.target = Target.from_bundle(self.bundle)
        self.ctrl.reset()

    def _disarm(self, reason):
        if self.mode == ARMED:
            self.mode = FAULT
            self.fault_reason = reason
        self.arm_requested = False
        self.upright_since = None

    def request_arm(self):
        self.arm_requested = True

    def _console_events(self):
        if not self.console:
            return []
        try:
            r, _, _ = select.select([sys.stdin], [], [], 0)
        except (ValueError, OSError):
            return []
        if r:
            line = sys.stdin.readline().strip().lower()
            return ["disarm"] if line in ("x", "d", "disarm", "stop") else ["arm"]
        return []

    # ------------------------------------------------------------- loop
    def tick(self):
        try:
            self._tick()
        finally:
            self.ticks += 1

    def _tick(self):
        imu, telem = self.io.read()
        now = self.io.now()
        events = self._console_events()
        if self.teleop is not None:
            (vx, vy, wz), ev = self.teleop.poll(now)
            events += ev
            self.target.command([vx, vy], wz)
        if "disarm" in events:
            self.mode = DISARMED                      # operator stop is not a fault
            self.fault_reason = "operator"
            self.arm_requested = False
            self.upright_since = None
        if "arm" in events:
            self.request_arm()

        if telem is None or imu is None:
            self._disarm("no telemetry from the step generator" if telem is None else "no IMU data")
            self.io.write([0.0] * self.n, False)
            self._log(now, None, np.zeros(self.n))
            return
        counts, clock_us = self.io.odometry(telem)
        if len(counts) != self.n:
            self._disarm("odometry source reports %d wheels, expected %d" % (len(counts), self.n))
            self.io.write([0.0] * self.n, False)
            return
        speeds = self.wheels.update(counts, clock_us)
        state = self.estimator.update(imu.rot, imu.gyro, speeds, self.dt, self.target.position)
        self.state = state
        tilt = state.tilt_deg
        safety = self.cfg["safety"]

        if self.mode == ARMED:
            if tilt > safety["tilt_cutoff_deg"]:
                self._disarm("tilt %.1f deg > cutoff" % tilt)
            elif telem.watchdog:
                self._disarm("firmware watchdog tripped")
        else:
            if tilt < safety["arm_tilt_deg"]:
                self.upright_since = self.upright_since if self.upright_since is not None else now
            else:
                self.upright_since = None
            if (self.arm_requested and self.upright_since is not None
                    and now - self.upright_since >= safety["arm_hold_s"]):
                self._arm()
                state = self.estimator.update(imu.rot, imu.gyro, speeds, 0.0, self.target.position)

        if self.mode == ARMED:
            cmd = np.asarray(self.ctrl.compute(state, self.target, speeds), dtype=float)
            if not np.all(np.isfinite(cmd)):
                self._disarm("controller produced non-finite output")
                cmd = np.zeros(self.n)
            self.io.write(self.step_rates(cmd), self.mode == ARMED)
            self.target.advance(self.dt)
        else:
            cmd = np.zeros(self.n)
            self.io.write([0.0] * self.n, False)
        self.last_cmd = cmd
        self._log(now, state, speeds)

    def _log(self, now, state, speeds):
        row = {"t": round(now, 4), "mode": self.mode, "fault": self.fault_reason}
        if state is not None:
            for name, v in zip(("roll", "pitch", "roll_rate", "pitch_rate", "ex", "ey", "vx", "vy", "yaw_rate"),
                               state.vector()):
                row[name] = round(float(v), 5)
        for i in range(self.n):
            row["w%d" % i] = round(float(speeds[i]), 4)
            row["cmd%d" % i] = round(float(self.last_cmd[i]), 4)
        row["target_vx"], row["target_vy"] = [round(float(v), 4) for v in self.target.velocity]
        self.log_rows.append(row)
        if len(self.log_rows) > 100000:
            del self.log_rows[:50000]
        if self.log_path:
            if self._log_writer is None:
                d = os.path.dirname(self.log_path)
                if d and not os.path.isdir(d):
                    os.makedirs(d)
                self._log_file = open(self.log_path, "w")
                fields = ["t", "mode", "fault", "roll", "pitch", "roll_rate", "pitch_rate", "ex", "ey", "vx", "vy",
                          "yaw_rate"] + ["w%d" % i for i in range(self.n)] + ["cmd%d" % i for i in range(self.n)] + \
                         ["target_vx", "target_vy"]
                self._log_writer = csv.DictWriter(self._log_file, fieldnames=fields)
                self._log_writer.writeheader()
            self._log_writer.writerow(row)

    def run(self, max_ticks=None, status_every=1.0):
        if hasattr(self.io, "start"):
            self.io.start()
        last_status = self.io.now()
        try:
            while max_ticks is None or self.ticks < max_ticks:
                self.tick()
                self.io.wait_tick()
                now = self.io.now()
                if status_every and now - last_status >= status_every and self.console:
                    last_status = now
                    s = self.state
                    print("[%s] tilt %5.1f deg  v (%+.2f, %+.2f) m/s  %s" % (
                        self.mode, s.tilt_deg if s else float("nan"), s.vx if s else 0.0, s.vy if s else 0.0,
                        self.fault_reason), flush=True)
        except KeyboardInterrupt:
            pass
        finally:
            try:
                self.io.write([0.0] * self.n, False)
            finally:
                if self._log_file:
                    self._log_file.close()
                self.io.close()


# --------------------------------------------------------------------------- CLI helpers
def calibrate(cfg, seconds=3.0, imu_raw=None):
    """Average gyro (bias) and attitude (level offset) while the robot is held upright and still."""
    from .imu import make_imu
    from .mathutil import zyx_angles

    imu = imu_raw or make_imu(cfg).raw
    if hasattr(imu, "gyro_bias"):
        imu.gyro_bias = np.zeros(3)          # measure the raw bias
    gyros, rolls, pitches = [], [], []
    from .mathutil import euler_xyz_deg_to_mat, quat_to_mat

    R_head_imu = euler_xyz_deg_to_mat(*cfg["imu"]["mount_euler_deg"])
    t0 = time.monotonic()
    while time.monotonic() - t0 < seconds:
        q, g, a, _ = imu.read()
        gyros.append(g)
        _, p, r = zyx_angles(quat_to_mat(q).dot(R_head_imu.T))
        rolls.append(r)
        pitches.append(p)
        time.sleep(0.01)
    bias = np.mean(gyros, axis=0)
    cfg.data["imu"]["gyro_bias"] = [round(float(b), 6) for b in bias]
    cfg.data["imu"]["level_offset_deg"] = [round(math.degrees(float(np.mean(rolls))), 3),
                                           round(math.degrees(float(np.mean(pitches))), 3)]
    return cfg


def spin_test(cfg, n_wheels, speed=1.0, seconds=1.0, mcu=None, imu_raw=None):
    """Spin each wheel at +speed rad/s (robot OFF the ball); report step and encoder deltas.

    With odometry from steps there is no encoder: only check visually that each wheel turns the
    right way (see deploy/README.md) and fix ``motor_sign``.
    """
    from .mcu import make_motor_link

    mcu = mcu or make_motor_link(cfg, n_wheels)
    src = cfg["odometry_source"]
    if src == "imu_encoders" and imu_raw is None:
        from .imu import ArduinoImu

        imu_raw = ArduinoImu(cfg["imu"]["port"], cfg["imu"]["baud"])
        imu_raw.wait_ready()

    def counts(telem):
        if src == "imu_encoders":
            imu_raw.read()
            return list(imu_raw.encoders)
        return list(telem.encoders if src == "mcu_encoders" else telem.steps)

    mcu.configure()
    mcu.send([0.0] * n_wheels, False)
    mcu.wait_for_telemetry()
    results = []
    for i in range(n_wheels):
        rates = [0.0] * n_wheels
        rates[i] = cfg["motor_sign"][i] * speed * cfg.steps_per_rad
        start = mcu.wait_for_telemetry()
        c0 = counts(start)
        t_end = time.monotonic() + seconds
        while time.monotonic() < t_end:
            mcu.send(rates, True)
            mcu.poll()
            time.sleep(0.01)
        mcu.send([0.0] * n_wheels, True)
        end = mcu.wait_for_telemetry()
        c1 = counts(end)
        d_steps = end.steps[i] - start.steps[i]
        res = {"wheel": i, "step_delta": d_steps}
        if src != "steps":
            d_enc = c1[i] - c0[i]
            res["encoder_delta"] = d_enc
            res["suggested_encoder_sign"] = int(np.sign(d_enc) * np.sign(d_steps * cfg["motor_sign"][i]) or 1)
        results.append(res)
    mcu.send([0.0] * n_wheels, False)
    return results


def main(argv=None):
    ap = argparse.ArgumentParser(description="BB-8 robot runtime (Jetson Nano)")
    sub = ap.add_subparsers(dest="cmd")
    w = sub.add_parser("write-config", help="write a default robot config")
    w.add_argument("path")
    c = sub.add_parser("calibrate", help="gyro bias + level offset (hold the robot upright and still)")
    c.add_argument("--config", required=True)
    c.add_argument("--seconds", type=float, default=3.0)
    s = sub.add_parser("spin-test", help="spin each wheel briefly with the robot OFF the ball")
    s.add_argument("--config", required=True)
    s.add_argument("--wheels", type=int, default=3)
    r = sub.add_parser("run", help="balance")
    r.add_argument("--bundle", required=True)
    r.add_argument("--config", required=True)
    r.add_argument("--controller", choices=["pid", "lqr", "rl"])
    r.add_argument("--onnx", action="store_true", help="evaluate the policy with onnxruntime instead of numpy")
    r.add_argument("--auto-arm", action="store_true", help="arm automatically once held upright")
    r.add_argument("--log", default=None, help="CSV log path (default: <log_dir>/run_<time>.csv)")
    r.add_argument("--no-teleop", action="store_true")
    args = ap.parse_args(argv)

    if args.cmd == "write-config":
        RobotConfig().save(args.path)
        print("wrote", args.path)
        return 0
    if args.cmd == "calibrate":
        cfg = RobotConfig.load(args.config)
        print("hold the robot upright and still for %.0f s ..." % args.seconds)
        calibrate(cfg, args.seconds)
        cfg.save(args.config)
        print("gyro_bias", cfg["imu"]["gyro_bias"], "level_offset_deg", cfg["imu"]["level_offset_deg"])
        return 0
    if args.cmd == "spin-test":
        cfg = RobotConfig.load(args.config)
        print("robot must be OFF the ball. Spinning each wheel at +1 rad/s for 1 s ...")
        for res in spin_test(cfg, args.wheels):
            print(res)
        print("Each wheel must turn so that its ball-contact point moves CLOCKWISE seen from above the head; "
              "fix motor_sign first, then (with encoders) set encoder_sign to the suggestion.")
        return 0
    if args.cmd == "run":
        cfg = RobotConfig.load(args.config)
        bundle = Bundle.load(args.bundle)
        tp = cfg["teleop"]
        teleop = None if args.no_teleop else UdpTeleop(tp["udp_port"], tp["max_speed"], tp["max_yaw_rate"],
                                                       tp["timeout_s"])
        log = args.log or os.path.join(cfg["log_dir"], time.strftime("run_%Y%m%d_%H%M%S.csv"))
        io = HardwareIO(cfg, bundle.n_wheels)
        runner = BalanceRunner(bundle, cfg, io, args.controller, "onnx" if args.onnx else "numpy", args.auto_arm,
                               log, teleop, console=True)
        print("controller %s | press Enter to arm (x + Enter to disarm), Ctrl-C to stop" % runner.ctrl.name)
        runner.run()
        return 0
    ap.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())

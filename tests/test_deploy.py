"""Deployment module tests: protocol, firmware, robot runtime parity, serial link, SIL balance, safety."""

import math
import os
import shutil
import subprocess
import sys
import time

import mujoco
import numpy as np
import pytest

from config.composite_ball_params import BB8Params, ControlMode, HardwareProfile
from controllers.classical import BalanceState, BalanceTarget, BallBotLQR, CascadedPID
from deploy.export_bundle import build_bundle
from deploy.runtime import protocol as P
from deploy.runtime.bundle import Bundle
from deploy.runtime.config import RobotConfig
from deploy.runtime.controllers import LQRController, PIDController, PolicyController, Target
from deploy.runtime.estimator import State, WheelSpeedEstimator
from deploy.runtime.imu import ComplementaryFilter, MountedImu
from deploy.runtime.runner import ARMED, FAULT, BalanceRunner, calibrate, spin_test
from deploy.sim_io import PtyMCU, SimIO, run_sil

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIRMWARE = os.path.join(ROOT, "deploy", "firmware")


@pytest.fixture(scope="module")
def bundles(tmp_path_factory):
    base = tmp_path_factory.mktemp("bundles")
    plain = str(base / "plain")
    build_bundle(plain, "pid")
    p = BB8Params()
    p.head.magnet.enabled = True
    p.head.magnet.force = 20.0
    magnet = str(base / "magnet")
    build_bundle(magnet, "lqr", params=p)
    return {"plain": plain, "magnet": magnet}


# --------------------------------------------------------------------------- python 3.6 / dependencies
def test_runtime_is_python36_compatible():
    vermin = shutil.which("vermin")
    if vermin is None:
        pytest.skip("vermin not installed")
    out = subprocess.run([vermin, "-t=3.6-", "--no-tips", "--violations", "--eval-annotations",
                          os.path.join(ROOT, "deploy", "runtime")], capture_output=True, text=True)
    assert out.returncode == 0, out.stdout + out.stderr


def test_runtime_does_not_import_simulation_stack():
    code = ("import sys, deploy.runtime.runner, deploy.runtime.controllers, deploy.runtime.imu, "
            "deploy.runtime.mcu, deploy.runtime.teleop; "
            "bad = [m for m in ('mujoco', 'torch', 'gymnasium', 'stable_baselines3', 'scipy', 'models', 'envs') "
            "if m in sys.modules]; print(bad); sys.exit(1 if bad else 0)")
    out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True)
    assert out.returncode == 0, out.stdout + out.stderr


# --------------------------------------------------------------------------- protocol
def test_protocol_roundtrip_and_resync():
    cmd = P.encode_cmd(513, [1000.5, -2000.0, 0.0], True)
    tel = P.encode_telem(7, 123456789, P.TF_ENABLED | P.TF_CONFIGURED, [1, -2, 2 ** 31 - 1], [4, 5, -6])
    parser = P.StreamParser()
    corrupted = bytearray(cmd)
    corrupted[6] ^= 0xFF
    frames = parser.feed(b"\x00\xAA\x13" + bytes(corrupted) + cmd[:5]) + parser.feed(cmd[5:] + tel)
    assert parser.crc_errors == 1 and len(frames) == 2
    c = P.decode(*frames[0])
    assert c.seq == 513 and c.enable and c.step_rates == pytest.approx([1000.5, -2000.0, 0.0])
    t = P.decode(*frames[1])
    assert t.encoders == [1, -2, 2 ** 31 - 1] and t.steps == [4, 5, -6] and t.enabled and not t.watchdog
    with pytest.raises(ValueError):
        P.decode(P.T_CMD, b"\x00\x00\x01\x05")


@pytest.fixture(scope="module")
def proto_bin(tmp_path_factory):
    if shutil.which("g++") is None:
        pytest.skip("g++ not available")
    exe = str(tmp_path_factory.mktemp("cc") / "proto_test")
    subprocess.run(["g++", "-std=c++11", "-Wall", "-Wextra", "-Werror", "-O2", "-o", exe,
                    os.path.join(FIRMWARE, "protocol_host_test.cpp")], check=True, cwd=FIRMWARE)
    return exe


def test_c_protocol_interop(proto_bin):
    frames = (P.encode_cmd(65535, [12345.25, -0.5], False) + b"\x55\xAA\x00" + P.encode_cfg(150000.0, 100, 3)).hex()
    out = subprocess.run([proto_bin], input="FRAME %s\nTELEM 9 4294967295 5 3 -1 2 2147483647 10 -20 30\n" % frames,
                         capture_output=True, text=True, check=True).stdout.splitlines()
    assert out[0].split() == ["CMD", "65535", "0", "2", "12345.2500", "-0.5000"]
    assert out[1].split() == ["CFG", "150000.0", "100", "3"]
    assert out[2] == "FRAMES 2 CRCERR 0"
    parser = P.StreamParser()
    (ftype, payload), = parser.feed(bytes.fromhex(out[3]))
    t = P.decode(ftype, payload)
    assert (t.seq_echo, t.mcu_us, t.flags) == (9, 4294967295, 5)
    assert t.encoders == [-1, 2, 2147483647] and t.steps == [10, -20, 30]


def test_firmware_compiles_against_host_stubs(tmp_path):
    if shutil.which("g++") is None:
        pytest.skip("g++ not available")
    exe = str(tmp_path / "fw")
    subprocess.run(["g++", "-std=c++11", "-Wall", "-Wextra", "-Werror", "-Ihost_stubs", "-Ibb8_mcu", "-x", "c++",
                    "host_stubs/main.cpp", "-o", exe], check=True, cwd=FIRMWARE)
    subprocess.run([exe], check=True)


# --------------------------------------------------------------------------- runtime parity with the simulator
def _random_states(rng, n=40):
    for _ in range(n):
        y = rng.normal(0, 1, 9) * np.array([0.05, 0.05, 0.3, 0.3, 0.2, 0.2, 0.3, 0.3, 0.4])
        yield y, float(rng.uniform(-3, 3))


def _targets(rng):
    a = BalanceTarget()
    b = Target(a.max_position_error, a.max_accel, a.max_yaw_accel)
    v, w = rng.normal(0, 0.2, 2), float(rng.normal(0, 0.3))
    a.command(v, w)
    b.command(v, w)
    return a, b


def test_pid_and_lqr_match_simulator(bundles):
    from models.mjcf_generator import build_bb8

    bundle = Bundle.load(bundles["plain"])
    sim_bundle = build_bb8(BB8Params())
    model = sim_bundle.compile()
    rng = np.random.default_rng(0)
    sim_pid, rob_pid = CascadedPID(model, sim_bundle.meta, dt=bundle.control_dt), PIDController(bundle)
    sim_lqr = BallBotLQR.design(sim_bundle, HardwareProfile(), model=model)
    rob_lqr = LQRController(bundle)
    assert np.allclose(sim_lqr.K, rob_lqr.K)
    t_sim, t_rob = _targets(rng)
    for y, yaw in _random_states(rng):
        s_sim, s_rob = BalanceState.from_vector(y, yaw), State(y, yaw)
        assert np.allclose(s_sim.error(t_sim), s_rob.error(t_rob))
        assert np.allclose(sim_pid.compute(s_sim, t_sim), rob_pid.compute(s_rob, t_rob), atol=1e-12)
        assert np.allclose(sim_lqr.compute(s_sim, t_sim), rob_lqr.compute(s_rob, t_rob), atol=1e-12)
        t_sim.advance(0.01)
        t_rob.advance(0.01)
        assert np.allclose(t_sim.position, t_rob.position) and t_sim.yaw_rate == pytest.approx(t_rob.yaw_rate)


@pytest.fixture(scope="module")
def policy_bundle(tmp_path_factory):
    from training.train_rl import TrainingConfig, TrainingManager

    runs = tmp_path_factory.mktemp("runs")
    cfg = TrainingConfig(total_timesteps=1024, n_envs=1, vec_env="dummy", out_dir=str(runs), checkpoint_every=0)
    cfg.ppo.n_steps = 512
    mgr = TrainingManager(cfg)
    mgr.start(background=False)
    assert mgr.state == "finished", mgr.error
    out = str(tmp_path_factory.mktemp("bundle_rl"))
    data = build_bundle(out, "rl", policy_path=mgr.latest_policy)
    assert data["warnings"] == []
    return out, mgr.latest_policy


def test_policy_network_and_io_match_simulator(policy_bundle):
    from envs.ball_balancer_env import BallBalancerEnv, EnvConfig
    from training.train_rl import PolicyRunner

    path, zip_path = policy_bundle
    bundle = Bundle.load(path)
    rob_np, rob_onnx = PolicyController(bundle, "numpy"), PolicyController(bundle, "onnx")
    sb3 = PolicyRunner.load(zip_path)
    env = BallBalancerEnv(EnvConfig())
    obs, _ = env.reset(seed=2)
    rob_np.reset()
    for _ in range(25):
        assert np.allclose(rob_np.net(obs), sb3(obs), atol=1e-5)
        assert np.allclose(rob_onnx.net(obs), sb3(obs), atol=1e-5)
        action = sb3(obs)
        ctrl = env.action_to_ctrl(action)
        assert np.allclose(rob_np.action_to_wheels(action), ctrl[env.actuators.wheel_act])
        obs, _, term, _, _ = env.step(action)
        # the robot builds the same observation from the same estimate / encoder speeds / target
        s = env.last_state
        t = Target(env.target.max_position_error, env.target.max_accel, env.target.max_yaw_accel)
        t.position, t.velocity, t.yaw_rate = env.target.position, env.target.velocity, env.target.yaw_rate
        rob_np.prev_action = np.asarray(action, dtype=float)
        rob_obs = rob_np.observation(State(s.vector(), s.yaw), t, env.last_reading.wheel_vel)
        assert np.allclose(rob_obs, obs, atol=1e-6)
        if term:
            break


def test_wheel_speed_estimator_wraps():
    est = WheelSpeedEstimator(counts_per_rad=1200 / (2 * math.pi), signs=[1, -1])
    est.update([2 ** 31 - 10, 0], 2 ** 32 - 5000)
    w = est.update([-(2 ** 31) + 110, -120], 5000)            # int32 counts and u32 micros both wrap
    assert w[0] == pytest.approx(120 / 1200 * 2 * math.pi / 0.01)
    assert w[1] == pytest.approx(120 / 1200 * 2 * math.pi / 0.01)


# --------------------------------------------------------------------------- IMU helpers
def test_imu_mounting_and_complementary_filter():
    class Raw:
        def read(self):
            return np.array([1.0, 0, 0, 0]), np.array([0.1, 0.0, 0.0]), np.array([0, 0, 9.81]), 0.0

    m = MountedImu(Raw(), mount_euler_deg=(0, 0, 90), gyro_bias=(0.05, 0, 0))
    s = m.read()
    assert np.allclose(s.gyro, [0.0, 0.05, 0.0], atol=1e-12)    # IMU x axis points along head y
    f = ComplementaryFilter(kp=2.0)
    tilt = math.radians(10)
    accel = 9.81 * np.array([-math.sin(tilt), 0.0, math.cos(tilt)])   # pitched 10 deg, at rest
    f.update([0, 0, 0], [0, 0, 9.81], 0.01)
    for _ in range(600):
        q = f.update([0, 0, 0], accel, 0.01)
    from deploy.runtime.mathutil import quat_to_mat, zyx_angles

    _, pitch, roll = zyx_angles(quat_to_mat(q))
    assert pitch == pytest.approx(tilt, abs=2e-3) and abs(roll) < 1e-3


def test_calibrate_with_fake_imu():
    class Raw:
        def read(self):
            q = np.array([math.cos(0.01), 0.0, math.sin(0.01), 0.0])       # 0.02 rad pitch
            return q, np.array([0.01, -0.02, 0.003]), np.array([0, 0, 9.81]), 0.0

    cfg = calibrate(RobotConfig(), seconds=0.05, imu_raw=Raw())
    assert cfg["imu"]["gyro_bias"] == pytest.approx([0.01, -0.02, 0.003])
    assert cfg["imu"]["level_offset_deg"][1] == pytest.approx(math.degrees(0.02), abs=1e-3)


def test_config_validation(bundles):
    with pytest.raises(ValueError):
        RobotConfig({"motor_sign": [1, 2, 1]})
    with pytest.raises(KeyError):
        RobotConfig({"no_such_key": 1})
    bundle = Bundle.load(bundles["plain"])
    with pytest.raises(ValueError):
        BalanceRunner(bundle, RobotConfig({"motor_sign": [1, 1]}), io=None)
    with pytest.raises(ValueError):
        BalanceRunner(bundle, RobotConfig({"control_rate_hz": 200.0}), io=None)


# --------------------------------------------------------------------------- serial link
def test_serial_mcu_over_pty_ramp_and_watchdog():
    pytest.importorskip("serial")
    from deploy.runtime.mcu import SerialMCU

    emu = PtyMCU(n=3)
    try:
        mcu = SerialMCU(emu.port, n_motors=3, max_accel_steps=20000.0, watchdog_ms=100)
        mcu.configure()
        mcu.send([0.0, 0.0, 0.0], False)
        t = mcu.wait_for_telemetry()
        assert not t.enabled
        t0 = time.monotonic()
        while time.monotonic() - t0 < 0.3:
            mcu.send([4000.0, -2000.0, 0.0], True)
            mcu.poll()
            time.sleep(0.01)
        t = mcu.wait_for_telemetry()
        assert t.enabled and emu.fw.configured
        assert emu.fw.rate[0] == pytest.approx(4000.0) and emu.fw.rate[1] == pytest.approx(-2000.0)
        assert t.steps[0] > 300 and t.steps[1] < -150 and t.encoders[0] > 0      # ramped (20k steps/s^2) and moving
        time.sleep(0.4)                                                           # host goes silent
        t = mcu.wait_for_telemetry()
        assert t.watchdog and not t.enabled
        # the spin test drives each wheel and reports the encoder direction
        cfg = RobotConfig({"serial_port": emu.port})
        res = spin_test(cfg, 3, speed=1.0, seconds=0.2, mcu=mcu)
        assert all(r["step_delta"] > 0 and r["suggested_encoder_sign"] == 1 for r in res)
        mcu.close()
    finally:
        emu.close()


# --------------------------------------------------------------------------- software in the loop
def _max_tilt(io, after=0.0):
    return max(r["tilt_deg"] for r in io.log if r["t"] >= after)


@pytest.mark.parametrize("signs", [
    {},
    {"motor_sign": [1, -1, 1], "encoder_sign": [-1, 1, -1]},
    {"use_encoders": False, "motor_sign": [-1, 1, 1], "encoder_sign": [1, 1, 1]},
])
def test_sil_pid_balances_with_pushes(bundles, signs):
    cfg = RobotConfig(dict({"motor_sign": [1, 1, 1], "encoder_sign": [1, 1, 1]}, **signs))
    pushes = [(3.0, [6.0, 0.0, 0.0]), (6.0, [0.0, -6.0, 0.0])]
    runner, io = run_sil(bundles["plain"], "pid", 10.0, cfg, pushes)
    assert runner.mode == ARMED, runner.fault_reason
    assert _max_tilt(io) < 12.0
    assert np.linalg.norm(io.log[-1]["ball_xy"]) < 0.25


def test_sil_lqr_with_magnet(bundles):
    runner, io = run_sil(bundles["magnet"], "lqr", 10.0, pushes=[(3.0, [6.0, 0.0, 0.0]), (6.0, [0.0, 6.0, 0.0])])
    assert runner.mode == ARMED, runner.fault_reason
    assert _max_tilt(io) < 12.0


def test_sil_teleop_moves_target(bundles):
    class FakeTeleop:
        def poll(self, now):
            return ((0.25, 0.0, 0.0) if now > 2.0 else (0.0, 0.0, 0.0)), []

    bundle = Bundle.load(bundles["plain"])
    cfg = RobotConfig()
    io = SimIO(bundle, cfg)
    runner = BalanceRunner(bundle, cfg, io, "pid", auto_arm=True, teleop=FakeTeleop())
    runner.run(max_ticks=800, status_every=0)
    assert runner.mode == ARMED
    assert runner.state.vx == pytest.approx(0.25, abs=0.06)
    assert io.log[-1]["ball_xy"][0] > 0.5


def test_safety_faults(bundles):
    bundle = Bundle.load(bundles["plain"])
    # wrong motor direction on one wheel: the robot falls and the runner disarms
    cfg = RobotConfig()
    io = SimIO(bundle, RobotConfig({"motor_sign": [1, -1, 1]}))
    runner = BalanceRunner(bundle, cfg, io, "pid", auto_arm=True)
    runner.run(max_ticks=600, status_every=0)
    assert runner.mode == FAULT and "tilt" in runner.fault_reason
    assert not io.fw.enabled                                  # firmware ramped down and disabled the drivers

    # microcontroller goes silent
    class Silent(SimIO):
        def read(self):
            imu, telem = super().read()
            return imu, (None if self.env.data.time > 2.0 else telem)

    io = Silent(bundle, cfg)
    runner = BalanceRunner(bundle, cfg, io, "pid", auto_arm=True)
    runner.run(max_ticks=300, status_every=0)
    assert runner.mode == FAULT and "telemetry" in runner.fault_reason

    # no arm request -> motors never enabled
    io = SimIO(bundle, cfg)
    runner = BalanceRunner(bundle, cfg, io, "pid", auto_arm=False)
    runner.run(max_ticks=200, status_every=0)
    assert runner.mode == "DISARMED" and not io.fw.enabled


def test_export_rejects_torque_drive(tmp_path):
    p = BB8Params()
    p.head.motor.control_mode = ControlMode.TORQUE
    with pytest.raises(ValueError):
        build_bundle(str(tmp_path / "b"), "pid", params=p)

"""Module 3 tests: mixers, linearisation, LQR / PID closed-loop behaviour."""

import mujoco
import numpy as np
import pytest

from config.composite_ball_params import (
    BB8Params,
    ControlMode,
    HardwareProfile,
    KiwiDriveParams,
    OmniwheelMode,
    PendulumParams,
    PlanarBalancerParams,
)
from controllers.classical import (
    BalanceTarget,
    BallBotLQR,
    BallBotModel,
    BallWheelMixer,
    CartPolePID,
    CascadedPID,
    ClosedLoop,
    FullStateLQR,
    KiwiMixer,
    augment_delay,
    dlqr,
    linearize_full,
    linearize_reduced,
)
from models.mjcf_generator import build_example
from models.runtime import DelayLine, SceneStepper


# --------------------------------------------------------------------------- helpers
def _balance_run(key, params, kind, push=6.0, seconds=8.0, velocity=None, yaw_rate=0.0, hardware=None):
    hw = hardware or HardwareProfile()
    bundle = build_example(key, params)
    model = bundle.compile()
    loop = ClosedLoop(bundle, hw, model)
    bbm = BallBotModel(model, bundle.meta)
    target = BalanceTarget()
    if kind == "lqr":
        ctrl = BallBotLQR.design(bundle, hw, model=model)
    else:
        ctrl = CascadedPID(model, bundle.meta, dt=hw.loop.control_dt)
    log = []

    def policy(data, k):
        if velocity is not None and k == 100:
            target.command(velocity, yaw_rate)
        target.advance(hw.loop.control_dt)
        if push and k == 200:
            loop.stepper.push("head", [push, 0.0, 0.0], 0.1)
        s = bbm.measure(data, target)
        log.append(np.r_[s.vector(), s.error(target)])
        return ctrl.compute(s, target)

    loop.run(policy, seconds)
    return np.array(log)


# --------------------------------------------------------------------------- mixers
def test_kiwi_mixer_roundtrip():
    mix = KiwiMixer(KiwiDriveParams().mixing_matrix())
    cmd = np.array([0.2, -0.1, 0.7])
    assert np.allclose(mix.body_velocity(mix.wheel_speeds(*cmd)), cmd)
    w = mix.scaled_wheel_speeds(5.0, 0.0, 0.0, max_speed=10.0)
    assert np.max(np.abs(w)) == pytest.approx(10.0)


def test_ball_mixer_power_duality():
    bundle = build_example("bb8")
    mix = BallWheelMixer.from_meta(bundle.meta)
    assert np.allclose(mix.J, bundle.meta.kinematics["ball_to_wheel_jacobian"])
    omega = np.array([0.3, -0.2, 0.5])
    torque = np.array([0.4, 0.1, -0.2])
    wheels_w = mix.rel_rate_to_wheels(omega)
    wheels_t = mix.torque_to_wheels(torque)
    assert np.dot(wheels_t, wheels_w) == pytest.approx(np.dot(torque, omega))   # virtual work
    assert np.allclose(mix.wheels_to_rel_rate(wheels_w), omega)


# --------------------------------------------------------------------------- linearisation
@pytest.mark.parametrize("variant", ["cartpole", "spherical"])
def test_linearize_full_matches_mjd_transitionFD(variant):
    bundle = build_example(1, PendulumParams(variant=variant))
    m = bundle.compile()
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    A, B = linearize_full(SceneStepper(m, d, bundle.meta), frame_skip=1)
    A2 = np.zeros((2 * m.nv, 2 * m.nv))
    B2 = np.zeros((2 * m.nv, m.nu))
    mujoco.mjd_transitionFD(m, d, 1e-6, True, A2, B2, None, None)
    assert np.allclose(A, A2, atol=1e-8)
    assert np.allclose(B, B2, atol=1e-8)


@pytest.mark.parametrize("key,params", [
    ("bb8", BB8Params()),
    ("planar_balancer", PlanarBalancerParams(variant="1d")),
    ("planar_balancer", PlanarBalancerParams(variant="2d")),
])
def test_lift_measure_roundtrip_and_physical_open_loop(key, params):
    bundle = build_example(key, params)
    m = bundle.compile()
    d = mujoco.MjData(m)
    st = SceneStepper(m, d, bundle.meta)
    bbm = BallBotModel(m, bundle.meta)
    bbm.settle_reference(d, st)
    y = np.array([0.02, -0.03, 0.1, -0.2, 0.05, -0.04, 0.3, -0.1, 0.5])
    bbm.lift(d, y)
    meas = bbm.measure(d, BalanceTarget(position=bbm.ref_ball[:2])).vector()
    mask = np.zeros(9)
    mask[bbm.active] = 1.0
    assert np.allclose(meas, y * mask, atol=5e-4)
    A, _, _ = linearize_reduced(st, bbm, frame_skip=5)
    rho = np.max(np.abs(np.linalg.eigvals(A)))
    assert 1.0 < rho < 1.1          # tipping instability of a few 1/s, no numerical artefacts


def test_linearisation_sees_magnet_through_stepper():
    def pitch_row(enabled):
        p = BB8Params()
        p.head.magnet.enabled = enabled
        p.head.magnet.force = 30.0
        p.head.magnet.model = "dipole"
        bundle = build_example("bb8", p)
        m = bundle.compile()
        d = mujoco.MjData(m)
        st = SceneStepper(m, d, bundle.meta)
        bbm = BallBotModel(m, bundle.meta)
        bbm.settle_reference(d, st)
        A, Bn, _ = linearize_reduced(st, bbm, frame_skip=5)
        return A[3], Bn[3]
    a0, b0 = pitch_row(False)
    a1, b1 = pitch_row(True)
    assert np.max(np.abs(a1 - a0)) > 1e-3 or np.max(np.abs(b1 - b0)) > 1e-3


def test_delay_augmentation_matches_direct_simulation():
    rng = np.random.default_rng(0)
    A = np.array([[1.02, 0.01], [0.3, 0.97]])
    Bn = np.array([[0.0], [0.05]])
    Bo = np.array([[0.0], [0.02]])
    m = 2
    Az, Bz, L = augment_delay(A, Bn, Bo, m)
    assert L == 3
    us = rng.normal(size=(30, 1))
    x = np.array([0.1, -0.2])
    z = np.concatenate([x, np.zeros(L)])
    hist = [np.zeros(1)] * (m + 2)
    for k in range(30):
        hist = [us[k]] + hist[:-1]                     # hist[j] = u_{k-j}
        x = A @ x + Bn @ hist[m] + Bo @ hist[m + 1]
        z = Az @ z + Bz @ us[k]
        assert np.allclose(z[:2], x)
    K, _ = dlqr(Az, Bz, np.eye(Az.shape[0]), np.eye(1))
    assert np.max(np.abs(np.linalg.eigvals(Az - Bz @ K))) < 1.0


def test_delay_line():
    dl = DelayLine(1, delay_steps=3)
    dl.submit([1.0], 0)
    assert dl.current(2)[0] == 0.0
    assert dl.current(3)[0] == 1.0
    dl.submit([2.0], 5)
    assert dl.current(7)[0] == 1.0 and dl.current(8)[0] == 2.0


# --------------------------------------------------------------------------- closed loop
def _assert_balanced(log, max_tilt_deg=15.0, final_pos=0.05):
    assert np.degrees(np.max(np.abs(log[:, :2]))) < max_tilt_deg
    assert np.all(np.abs(log[-1, 13:15]) < final_pos)          # position error at the end
    assert np.all(np.abs(log[-1, :2]) < np.radians(1.0))


@pytest.mark.parametrize("kind", ["lqr", "pid"])
@pytest.mark.parametrize("mode", list(OmniwheelMode))
def test_bb8_station_keeping_with_delay_and_push(kind, mode):
    p = BB8Params(initial_tilt_deg=(2.0, 3.0))
    p.head.wheel.mode = mode
    _assert_balanced(_balance_run("bb8", p, kind, push=8.0))


@pytest.mark.parametrize("kind", ["lqr", "pid"])
@pytest.mark.parametrize("variant", ["1d", "2d"])
def test_planar_balancer(kind, variant):
    _assert_balanced(_balance_run("planar_balancer", PlanarBalancerParams(variant=variant), kind))


@pytest.mark.parametrize("kind", ["lqr", "pid"])
def test_bb8_torque_mode(kind):
    p = BB8Params()
    p.head.motor.control_mode = ControlMode.TORQUE
    _assert_balanced(_balance_run("bb8", p, kind))


@pytest.mark.parametrize("kind", ["lqr", "pid"])
def test_bb8_velocity_tracking(kind):
    log = _balance_run("bb8", BB8Params(), kind, push=0.0, seconds=6.0, velocity=[0.3, -0.2])
    assert np.degrees(np.max(np.abs(log[:, :2]))) < 12.0
    assert np.allclose(log[-1, 6:8], [0.3, -0.2], atol=0.05)     # yaw stays ~0, so yaw frame == world


@pytest.mark.parametrize("kind", ["lqr", "pid"])
def test_bb8_yaw_rate_tracking(kind):
    log = _balance_run("bb8", BB8Params(), kind, push=0.0, seconds=6.0, velocity=[0.0, 0.0], yaw_rate=0.5)
    assert log[-1, 8] == pytest.approx(0.5, abs=0.1)


def test_bb8_zero_latency_lqr():
    hw = HardwareProfile()
    hw.latency_enabled = False
    _assert_balanced(_balance_run("bb8", BB8Params(), "lqr", hardware=hw))


@pytest.mark.parametrize("variant", ["cartpole", "spherical"])
def test_pendulum_lqr(variant):
    bundle = build_example(1, PendulumParams(variant=variant))
    m = bundle.compile()
    loop = ClosedLoop(bundle, HardwareProfile(), m)
    lqr = FullStateLQR.design(bundle, frame_skip=loop.frame_skip, model=m)
    log = []

    def policy(d, k):
        if k == 200:
            loop.stepper.push("pole", [3.0, 0.0, 0.0], 0.05)
        log.append(d.qpos.copy())
        return lqr.compute_full(d)

    loop.run(policy, 8.0)
    log = np.array(log)
    assert np.all(np.abs(log[-1]) < 0.02)


def test_cartpole_pid():
    bundle = build_example(1)
    m = bundle.compile()
    loop = ClosedLoop(bundle, HardwareProfile(), m)
    pid = CartPolePID(m, bundle.meta, dt=loop.hw.loop.control_dt)
    log = []

    def policy(d, k):
        if k == 200:
            loop.stepper.push("pole", [3.0, 0.0, 0.0], 0.05)
        log.append(d.qpos.copy())
        return pid.compute_full(d)

    loop.run(policy, 10.0)
    log = np.array(log)
    assert np.max(np.abs(log[:, 1])) < 0.3
    assert np.all(np.abs(log[-1]) < 0.03)

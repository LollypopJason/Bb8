"""Module 4 tests: sensor/actuator bus and the Gymnasium environment."""

import math

import gymnasium as gym
import mujoco
import numpy as np
import pytest
from stable_baselines3.common.env_checker import check_env

from config.composite_ball_params import ControlMode, HardwareProfile
from controllers.classical import BallBotLQR, CascadedPID
from envs.ball_balancer_env import ENV_ID, BallBalancerEnv, EnvConfig
from models.io_bus import ActuatorBus, SensorBus
from models.mjcf_generator import build_example


def _quiet_hw(**kw):
    hw = HardwareProfile()
    hw.noise_enabled = False
    hw.encoder.quantize = False
    for k, v in kw.items():
        setattr(hw, k, v)
    return hw


# --------------------------------------------------------------------------- sensor bus
def test_sensor_bus_without_noise_matches_ground_truth():
    bundle = build_example("bb8")
    m, d, st = bundle.make()
    st.push("head", [5, 2, 0], 0.2)
    st.step(100)
    st.forward()                      # fresh sensordata for the current state
    bus = SensorBus(m, bundle.meta, _quiet_hw())
    r = bus.read(d)
    head = m.body("head").id
    assert np.allclose(r.imu_quat, d.xquat[head], atol=1e-9) or np.allclose(r.imu_quat, -d.xquat[head], atol=1e-9)
    w = d.sensordata[m.sensor_adr[m.sensor("head_imu_gyro").id]:][:3]
    assert np.allclose(r.gyro, w)
    from controllers.classical import BallBotModel
    truth = BallBotModel(m, bundle.meta).measure(d).vector()
    est = SensorBus.estimate(r).vector()
    assert np.allclose(est, truth, atol=1e-6)


def test_encoder_quantisation_and_noise_levels():
    bundle = build_example("bb8")
    m, d, st = bundle.make()                       # upright, at rest: the gyro only sees noise
    hw = HardwareProfile()
    bus = SensorBus(m, bundle.meta, hw, rng=np.random.default_rng(0))
    qadr = m.jnt_qposadr[m.joint("wheel0_drive").id]
    res = 2 * math.pi / hw.encoder.cpr
    gyros, speeds = [], []
    for k in range(1, 401):
        d.time = 0.01 * k
        d.qpos[qadr] = 3.0 * d.time                # wheel turning at 3 rad/s
        mujoco.mj_forward(m, d)
        r = bus.read(d)
        gyros.append(r.gyro - bus.gyro_bias)
        speeds.append(r.wheel_vel[0])
        assert np.allclose(np.round(r.wheel_pos / res), r.wheel_pos / res, atol=1e-6)
    g = np.array(gyros)
    assert np.std(g[:, 0]) == pytest.approx(hw.imu.gyro_noise_std, rel=0.3)
    speeds = np.array(speeds[1:])
    assert np.mean(speeds) == pytest.approx(3.0, rel=0.02)
    assert set(np.round(speeds / (res / 0.01)).astype(int)) <= {4, 5, 6}   # quantised count rate


def test_gyro_bias_random_walk_and_reset():
    bundle = build_example("bb8")
    m, d, st = bundle.make()
    hw = HardwareProfile()
    hw.imu.gyro_bias_walk = 0.05
    bus = SensorBus(m, bundle.meta, hw, rng=np.random.default_rng(1))
    b0 = bus.gyro_bias.copy()
    for _ in range(100):
        st.step(5)
        bus.read(d)
    assert np.linalg.norm(bus.gyro_bias - b0) > 1e-3
    bus.reset(np.random.default_rng(2))
    assert np.linalg.norm(bus.gyro_bias) < 0.1


# --------------------------------------------------------------------------- actuator bus
def test_actuator_bus_delay_and_pullout_curve():
    bundle = build_example("bb8")
    m, d, st = bundle.make()
    hw = HardwareProfile()
    hw.latency.delay = 0.012          # 6 physics steps
    bus = ActuatorBus(m, bundle.meta, hw)
    assert bus.delay.delay_steps == 6
    bus.submit(np.full(m.nu, 5.0), 0)
    for k in range(6):
        assert np.all(bus.apply(d, k) == 0.0)
    assert np.all(bus.apply(d, 6) == 5.0)
    # pull-out torque drops above the corner speed
    dof = m.jnt_dofadr[m.joint("wheel0_drive").id]
    d.qvel[dof] = 2 * hw.motor.pullout_corner_speed
    bus.apply(d, 7)
    a = m.actuator("wheel0_motor").id
    assert m.actuator_forcerange[a, 1] == pytest.approx(0.5 * hw.motor.wheel_torque_limit)
    bus.reset()
    assert m.actuator_forcerange[a, 1] == pytest.approx(hw.motor.wheel_torque_limit)


# --------------------------------------------------------------------------- env
@pytest.mark.parametrize("scene", ["bb8", "planar_1d", "planar_2d"])
def test_env_api(scene):
    env = BallBalancerEnv(EnvConfig(scene=scene))
    check_env(env, warn=False, skip_render_check=True)
    obs, info = env.reset(seed=3)
    assert obs.shape == env.observation_space.shape
    assert set(info["randomization"]) >= {"epoxy_scale", "ball_friction", "wheel_friction", "head_com_dx"}
    steps = 0
    terminated = truncated = False
    while not (terminated or truncated):
        obs, r, terminated, truncated, info = env.step(np.zeros(env.n_act))
        steps += 1
    assert terminated and steps < 500          # an unactuated balancer falls
    assert info["reward_terms"]["fall"] < 0
    env.close()


def test_env_seeding_is_deterministic():
    def rollout():
        env = BallBalancerEnv(EnvConfig())
        obs, _ = env.reset(seed=11)
        out = [obs]
        for _ in range(20):
            out.append(env.step(np.full(env.n_act, 0.1))[0])
        return np.array(out)
    assert np.array_equal(rollout(), rollout())


def test_domain_randomization_ranges():
    env = BallBalancerEnv(EnvConfig())
    epoxy = [p for p in env.cfg.bb8.ball.layer_properties() if p.name == "epoxy_sawdust"][0].mass
    nominal = env.cfg.bb8.ball.mass
    masses = []
    for s in range(30):
        _, info = env.reset(seed=s)
        masses.append(env.model.body_mass[env.bbm.ball_id])
        assert abs(info["randomization"]["head_com_dx"]) <= 0.01
    masses = np.array(masses)
    assert np.all(np.abs(masses - nominal) <= 0.15 * epoxy + 1e-9)
    assert masses.std() > 0.01
    env.cfg.randomization.enabled = False
    env.reset(seed=0)
    assert env.model.body_mass[env.bbm.ball_id] == pytest.approx(nominal)


def test_velocity_tracking_task_commands():
    env = BallBalancerEnv(EnvConfig(task="velocity_tracking"))
    env.reset(seed=4)
    cmds = set()
    for _ in range(700):
        env.step_ctrl(np.zeros(env.model.nu))
        cmds.add(tuple(np.round(env.target.velocity_cmd, 4)))
    assert len(cmds) >= 2


def test_gym_make():
    env = gym.make(ENV_ID)
    obs, _ = env.reset(seed=0)
    assert env.observation_space.contains(obs)
    env.close()


@pytest.mark.parametrize("scene", ["bb8", "planar_1d"])
@pytest.mark.parametrize("kind", ["pid", "lqr"])
def test_classical_controllers_through_noisy_env(scene, kind):
    cfg = EnvConfig(scene=scene, episode_seconds=12.0)
    cfg.pushes.rate = 0.3
    env = BallBalancerEnv(cfg)
    env.reset(seed=5)
    if kind == "pid":
        ctrl = CascadedPID(env.model, env.meta, dt=env.control_dt)
    else:
        ctrl = BallBotLQR.design(env.bundle, cfg.hardware)
    done = False
    steps = 0
    max_tilt = 0.0
    while not done:
        u = ctrl.compute(env.last_state, env.target)
        _, _, term, trunc, info = env.step_ctrl(u)
        max_tilt = max(max_tilt, info["tilt_deg"])
        done = term or trunc
        steps += 1
    assert not term, f"fell after {steps} steps"
    assert max_tilt < 20.0                     # random pushes up to 10 N

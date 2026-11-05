"""Module 5 tests: training manager, live metrics, checkpoints, ONNX export, playback."""

import os

import numpy as np
import pytest

from controllers.classical import CascadedPID
from envs.ball_balancer_env import BallBalancerEnv, EnvConfig
from training.train_rl import MetricsHub, PolicyRunner, TrainingConfig, TrainingManager, evaluate


def _short_cfg(tmp_path, algo, vec="dummy", steps=None):
    cfg = TrainingConfig(algo=algo, total_timesteps=steps or (2048 if algo == "PPO" else 400), n_envs=2,
                         vec_env=vec, out_dir=str(tmp_path), checkpoint_every=1024 if algo == "PPO" else 0,
                         progress_every=256)
    cfg.ppo.n_steps = 512
    cfg.sac.learning_starts = 100
    cfg.sac.buffer_size = 10_000
    return cfg


@pytest.mark.parametrize("algo", ["PPO", "SAC"])
def test_background_training_metrics_checkpoints_and_onnx(tmp_path, algo):
    events = []
    hub = MetricsHub()
    hub.subscribe(events.append)
    mgr = TrainingManager(_short_cfg(tmp_path, algo), hub)
    mgr.start(background=True)
    assert mgr.wait(timeout=600)
    assert mgr.state == "finished", mgr.error
    kinds = {e["type"] for e in events}
    assert {"started", "progress", "episode", "checkpoint", "finished"} <= kinds
    ep = next(e for e in events if e["type"] == "episode")
    assert ep["length"] > 0 and "tilt" in ep["terms"]
    final = mgr.checkpoints[-1]
    assert os.path.isfile(final["zip"]) and os.path.isfile(final["onnx"])
    if algo == "PPO":
        assert len(mgr.checkpoints) >= 2                         # periodic + final
    # ONNX (normalisation baked in) == SB3 predict on the same observations
    zip_runner = PolicyRunner.load(final["zip"])
    onnx_runner = PolicyRunner.load(final["onnx"])
    env = BallBalancerEnv(EnvConfig())
    obs, _ = env.reset(seed=0)
    for _ in range(30):
        a_zip, a_onnx = zip_runner(obs), onnx_runner(obs)
        assert np.allclose(a_zip, a_onnx, atol=1e-5)
        obs, _, term, trunc, _ = env.step(a_zip)
        if term or trunc:
            obs, _ = env.reset()
    assert onnx_runner.obs_dim == env.observation_space.shape[0]


def test_subproc_training_and_stop(tmp_path):
    cfg = _short_cfg(tmp_path, "PPO", vec="subproc", steps=10_000_000)
    mgr = TrainingManager(cfg)
    mgr.start()
    import time

    deadline = time.time() + 300
    while mgr.timesteps < 1000 and time.time() < deadline and mgr.state == "running":
        time.sleep(0.5)
    mgr.stop(timeout=300)
    assert mgr.state == "stopped", mgr.error
    assert 0 < mgr.timesteps < cfg.total_timesteps
    assert os.path.isfile(mgr.latest_policy)


def test_evaluate_policy_and_controller():
    cfg = EnvConfig(episode_seconds=3.0)
    zero = evaluate(lambda obs: np.zeros(3), cfg, episodes=2)
    pid = evaluate(None, cfg, episodes=2,
                   controller_factory=lambda env: CascadedPID(env.model, env.meta, dt=env.control_dt))
    assert zero["fall_rate"] == 1.0
    assert pid["fall_rate"] == 0.0
    assert pid["mean_return"] > zero["mean_return"]

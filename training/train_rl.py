"""Reinforcement-learning training manager (Module 5).

* :class:`TrainingConfig` - PPO / SAC hyper-parameters, vector-env layout and
  the :class:`envs.ball_balancer_env.EnvConfig` to train on.
* :class:`MetricsHub` - thread-safe store + publish/subscribe of live training
  events (episode reward / length / reward-term breakdown, throughput, losses,
  checkpoints). The studio backend subscribes to it and forwards events over
  WebSockets.
* :class:`TrainingManager` - runs ``model.learn`` in a background thread with
  ``SubprocVecEnv`` (or ``DummyVecEnv``), ``VecNormalize`` for PPO, periodic
  checkpoints (``.zip`` + normaliser ``.pkl`` + ``.onnx``) and cooperative stop.
* :func:`export_onnx` - deterministic actor (observation normalisation baked
  in) to ONNX; :class:`PolicyRunner` loads ``.zip`` or ``.onnx`` policies for
  playback / hot-swapping in the 3D studio.
* :func:`evaluate` - episodes of a policy or classical controller on the env.

Command line::

    python -m training.train_rl --algo PPO --scene bb8 --task station_keeping \\
        --steps 2000000 --n-envs 8 --out runs
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import json
import math
import os
import pickle
import sys
import threading
import time
import traceback
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Deque, Dict, List, Optional, Sequence

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from config.composite_ball_params import params_from_dict, params_to_dict  # noqa: E402
from envs.ball_balancer_env import BallBalancerEnv, EnvConfig  # noqa: E402

__all__ = [
    "TrainingConfig",
    "MetricsHub",
    "TrainingManager",
    "make_vec_env",
    "export_onnx",
    "PolicyRunner",
    "evaluate",
]


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
@dataclass
class PPOParams:
    n_steps: int = 1024                  # per environment
    batch_size: int = 2048
    n_epochs: int = 10
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    log_std_init: float = -1.0


@dataclass
class SACParams:
    buffer_size: int = 1_000_000
    batch_size: int = 256
    learning_rate: float = 3e-4
    learning_starts: int = 10_000
    gamma: float = 0.99
    tau: float = 0.005
    train_freq: int = 1
    gradient_steps: int = 1


@dataclass
class TrainingConfig:
    algo: str = "PPO"                    # PPO | SAC
    total_timesteps: int = 2_000_000
    n_envs: int = 8
    vec_env: str = "subproc"             # subproc | dummy
    start_method: str = "forkserver"     # subproc start method (forkserver/spawn are thread-safe)
    env: EnvConfig = field(default_factory=EnvConfig)
    seed: int = 0
    out_dir: str = "runs"
    run_name: Optional[str] = None
    checkpoint_every: int = 100_000      # timesteps
    net_arch: List[int] = field(default_factory=lambda: [256, 256])
    normalize_obs: bool = True
    device: str = "cpu"
    ppo: PPOParams = field(default_factory=PPOParams)
    sac: SACParams = field(default_factory=SACParams)
    progress_every: int = 2_000          # timesteps between progress events

    def validate(self) -> None:
        if self.algo not in ("PPO", "SAC"):
            raise ValueError("algo must be PPO or SAC")
        if self.vec_env not in ("subproc", "dummy"):
            raise ValueError("vec_env must be subproc or dummy")
        if self.n_envs < 1 or self.total_timesteps < 1:
            raise ValueError("n_envs and total_timesteps must be >= 1")
        self.env.validate()


# --------------------------------------------------------------------------- #
# Live metrics
# --------------------------------------------------------------------------- #
class MetricsHub:
    """Thread-safe event store with subscribers (callables receiving each event dict)."""

    def __init__(self, max_events: int = 20_000) -> None:
        self._lock = threading.Lock()
        self.events: Deque[Dict[str, Any]] = deque(maxlen=max_events)
        self.episodes: Deque[Dict[str, Any]] = deque(maxlen=max_events)
        self._subscribers: List[Callable[[Dict[str, Any]], None]] = []
        self.latest: Dict[str, Any] = {}

    def publish(self, event: Dict[str, Any]) -> None:
        event = dict(event)
        event.setdefault("wall_time", time.time())
        with self._lock:
            self.events.append(event)
            if event.get("type") == "episode":
                self.episodes.append(event)
            self.latest[event.get("type", "event")] = event
            subs = list(self._subscribers)
        for cb in subs:
            try:
                cb(event)
            except Exception:  # a broken subscriber must never stop training
                pass

    def subscribe(self, cb: Callable[[Dict[str, Any]], None]) -> None:
        with self._lock:
            self._subscribers.append(cb)

    def unsubscribe(self, cb: Callable[[Dict[str, Any]], None]) -> None:
        with self._lock:
            if cb in self._subscribers:
                self._subscribers.remove(cb)

    def snapshot(self, last: int = 500) -> Dict[str, Any]:
        with self._lock:
            eps = list(self.episodes)[-last:]
            return {"episodes": eps, "latest": dict(self.latest)}


def _make_callback(hub: MetricsHub, manager: "TrainingManager"):
    from stable_baselines3.common.callbacks import BaseCallback

    class LiveMetricsCallback(BaseCallback):
        """Publishes episodes / progress / losses, saves checkpoints, honours stop requests."""

        def __init__(self) -> None:
            super().__init__()
            self._term_sums: Optional[List[Dict[str, float]]] = None
            self._last_progress = 0
            self._last_ckpt = 0
            self._last_train = -10**12
            self._t0 = time.time()

        def _on_training_start(self) -> None:
            self._term_sums = [dict() for _ in range(self.training_env.num_envs)]
            self._t0 = time.time()
            hub.publish({"type": "started", "run": manager.run_name, "algo": manager.cfg.algo,
                         "total_timesteps": manager.cfg.total_timesteps})

        def _on_step(self) -> bool:
            infos = self.locals.get("infos", [])
            for i, info in enumerate(infos):
                sums = self._term_sums[i]
                for k, v in info.get("reward_terms", {}).items():
                    sums[k] = sums.get(k, 0.0) + float(v)
                ep = info.get("episode")
                if ep is not None:
                    hub.publish({"type": "episode", "timesteps": int(self.num_timesteps),
                                 "reward": float(ep["r"]), "length": int(ep["l"]),
                                 "fell": bool(info.get("reward_terms", {}).get("fall", 0.0) < 0),
                                 "terms": {k: v for k, v in sums.items()}})
                    self._term_sums[i] = {}
            n = self.num_timesteps
            if n - self._last_progress >= manager.cfg.progress_every:
                self._last_progress = n
                dt = max(time.time() - self._t0, 1e-6)
                manager.timesteps = n
                hub.publish({"type": "progress", "timesteps": int(n), "fps": n / dt,
                             "fraction": n / manager.cfg.total_timesteps})
            if manager.cfg.checkpoint_every and n - self._last_ckpt >= manager.cfg.checkpoint_every:
                self._last_ckpt = n
                manager._save(self.model, f"step_{n}")
            return not manager._stop.is_set()

        def _on_rollout_end(self) -> None:
            if self.num_timesteps - self._last_train < manager.cfg.progress_every:
                return
            self._last_train = self.num_timesteps
            values = {k: float(v) for k, v in self.logger.name_to_value.items() if np.isscalar(v)}
            if values:
                hub.publish({"type": "train", "timesteps": int(self.num_timesteps), "values": values})

    return LiveMetricsCallback()


# --------------------------------------------------------------------------- #
# Vector envs
# --------------------------------------------------------------------------- #
def _env_factory(env_cfg_dict: Dict[str, Any], seed: int):
    def _init():
        from stable_baselines3.common.monitor import Monitor

        cfg = params_from_dict(EnvConfig, env_cfg_dict)
        env = Monitor(BallBalancerEnv(cfg))
        env.reset(seed=seed)
        env.action_space.seed(seed)
        return env

    return _init


def make_vec_env(env_cfg: EnvConfig, n_envs: int, seed: int = 0, kind: str = "subproc",
                 start_method: str = "forkserver"):
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

    cfg_dict = params_to_dict(env_cfg)
    fns = [_env_factory(cfg_dict, seed + i) for i in range(n_envs)]
    if kind == "subproc" and n_envs > 1:
        return SubprocVecEnv(fns, start_method=start_method)
    return DummyVecEnv(fns)


# --------------------------------------------------------------------------- #
# ONNX export & policy playback
# --------------------------------------------------------------------------- #
def export_onnx(model: Any, path: str, obs_rms: Any = None, clip_obs: float = 10.0, epsilon: float = 1e-8,
                opset: int = 17) -> str:
    """Export the deterministic actor of a PPO / SAC model to ONNX (``obs -> action in [-1, 1]``).

    Use ``opset=13`` for older onnxruntime builds (e.g. Jetson Nano / JetPack 4.x).
    """
    import torch

    policy = model.policy
    obs_dim = int(np.prod(model.observation_space.shape))
    mean = torch.zeros(obs_dim) if obs_rms is None else torch.as_tensor(obs_rms.mean, dtype=torch.float32)
    var = torch.ones(obs_dim) if obs_rms is None else torch.as_tensor(obs_rms.var, dtype=torch.float32)
    use_norm = obs_rms is not None

    class Actor(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.policy = policy
            self.register_buffer("mean", mean)
            self.register_buffer("std", torch.sqrt(var + epsilon))

        def forward(self, obs: torch.Tensor) -> torch.Tensor:
            if use_norm:
                obs = torch.clamp((obs - self.mean) / self.std, -clip_obs, clip_obs)
            if hasattr(self.policy, "actor"):                       # SAC
                action = self.policy.actor(obs, deterministic=True)
            else:                                                   # PPO / A2C
                features = self.policy.extract_features(obs, self.policy.pi_features_extractor)
                latent_pi = self.policy.mlp_extractor.forward_actor(features)
                action = self.policy.action_net(latent_pi)
            return torch.clamp(action, -1.0, 1.0)

    actor = Actor().eval()
    dummy = torch.zeros(1, obs_dim, dtype=torch.float32)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with torch.no_grad():
        torch.onnx.export(actor, dummy, path, input_names=["obs"], output_names=["action"],
                          dynamic_axes={"obs": {0: "batch"}, "action": {0: "batch"}}, opset_version=opset,
                          dynamo=False)
    return path


class PolicyRunner:
    """Deterministic policy from an SB3 ``.zip`` (+ ``vecnormalize.pkl``) or an ``.onnx`` file."""

    def __init__(self, act_fn: Callable[[np.ndarray], np.ndarray], source: str, obs_dim: int, act_dim: int) -> None:
        self._act = act_fn
        self.source = source
        self.obs_dim = obs_dim
        self.act_dim = act_dim

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "PolicyRunner":
        if path.endswith(".onnx"):
            import onnxruntime as ort

            sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
            inp = sess.get_inputs()[0]
            obs_dim = int(inp.shape[1])
            act_dim = int(sess.get_outputs()[0].shape[1])

            def act(obs: np.ndarray) -> np.ndarray:
                x = np.asarray(obs, dtype=np.float32).reshape(1, -1)
                return sess.run(None, {inp.name: x})[0][0].astype(float)

            return cls(act, path, obs_dim, act_dim)
        from stable_baselines3 import PPO, SAC

        with open(path, "rb"):
            pass
        meta_path = os.path.join(os.path.dirname(path), "run.json")
        algo = "PPO"
        if os.path.isfile(meta_path):
            with open(meta_path) as fh:
                algo = json.load(fh).get("algo", "PPO")
        model = (SAC if algo == "SAC" else PPO).load(path, device=device)
        norm_path = path[:-4] + "_vecnormalize.pkl"
        obs_rms = None
        if os.path.isfile(norm_path):
            with open(norm_path, "rb") as fh:
                obs_rms = pickle.load(fh)

        def act(obs: np.ndarray) -> np.ndarray:
            x = np.asarray(obs, dtype=np.float32)
            if obs_rms is not None:
                x = np.clip((x - obs_rms.mean) / np.sqrt(obs_rms.var + 1e-8), -10.0, 10.0).astype(np.float32)
            a, _ = model.predict(x, deterministic=True)
            return np.clip(np.asarray(a, dtype=float), -1.0, 1.0)

        return cls(act, path, int(np.prod(model.observation_space.shape)), int(np.prod(model.action_space.shape)))

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        return self._act(obs)


# --------------------------------------------------------------------------- #
# Training manager
# --------------------------------------------------------------------------- #
class TrainingManager:
    """Background PPO / SAC training with live metrics and checkpoints."""

    def __init__(self, config: Optional[TrainingConfig] = None, hub: Optional[MetricsHub] = None) -> None:
        self.cfg = copy.deepcopy(config) if config is not None else TrainingConfig()
        self.cfg.validate()
        self.hub = hub or MetricsHub()
        self.run_name = self.cfg.run_name or time.strftime(f"{self.cfg.algo.lower()}_%Y%m%d_%H%M%S")
        self.run_dir = os.path.join(self.cfg.out_dir, self.run_name)
        self.state = "idle"
        self.error: Optional[str] = None
        self.timesteps = 0
        self.checkpoints: List[Dict[str, str]] = []
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._vec = None

    # ----------------------------------------------------------- control
    def start(self, background: bool = True) -> "TrainingManager":
        if self.state == "running":
            raise RuntimeError("training already running")
        self._stop.clear()
        self.state = "running"
        if background:
            self._thread = threading.Thread(target=self._run, name=f"train-{self.run_name}", daemon=True)
            self._thread.start()
        else:
            self._run()
        return self

    def stop(self, wait: bool = True, timeout: Optional[float] = None) -> None:
        self._stop.set()
        if wait:
            self.wait(timeout)

    def wait(self, timeout: Optional[float] = None) -> bool:
        if self._thread is not None:
            self._thread.join(timeout)
            return not self._thread.is_alive()
        return True

    def status(self) -> Dict[str, Any]:
        return {"run": self.run_name, "state": self.state, "algo": self.cfg.algo, "timesteps": self.timesteps,
                "total_timesteps": self.cfg.total_timesteps, "error": self.error,
                "checkpoints": list(self.checkpoints), "run_dir": self.run_dir}

    @property
    def latest_policy(self) -> Optional[str]:
        return self.checkpoints[-1]["zip"] if self.checkpoints else None

    @property
    def latest_onnx(self) -> Optional[str]:
        return self.checkpoints[-1].get("onnx") if self.checkpoints else None

    # ----------------------------------------------------------- internals
    def _build_model(self, vec):
        from stable_baselines3 import PPO, SAC

        c = self.cfg
        if c.algo == "PPO":
            p = c.ppo
            batch = min(p.batch_size, p.n_steps * c.n_envs)
            return PPO("MlpPolicy", vec, n_steps=p.n_steps, batch_size=batch, n_epochs=p.n_epochs,
                       learning_rate=p.learning_rate, gamma=p.gamma, gae_lambda=p.gae_lambda,
                       clip_range=p.clip_range, ent_coef=p.ent_coef, vf_coef=p.vf_coef,
                       max_grad_norm=p.max_grad_norm, seed=c.seed, device=c.device, verbose=0,
                       policy_kwargs=dict(net_arch=dict(pi=list(c.net_arch), vf=list(c.net_arch)),
                                          log_std_init=p.log_std_init))
        p = c.sac
        return SAC("MlpPolicy", vec, buffer_size=p.buffer_size, batch_size=p.batch_size,
                   learning_rate=p.learning_rate, learning_starts=p.learning_starts, gamma=p.gamma, tau=p.tau,
                   train_freq=p.train_freq, gradient_steps=p.gradient_steps, seed=c.seed, device=c.device,
                   verbose=0, policy_kwargs=dict(net_arch=list(c.net_arch)))

    def _obs_rms(self):
        from stable_baselines3.common.vec_env import VecNormalize

        return self._vec.obs_rms if isinstance(self._vec, VecNormalize) else None

    def _save(self, model, tag: str) -> Dict[str, str]:
        os.makedirs(self.run_dir, exist_ok=True)
        base = os.path.join(self.run_dir, tag)
        model.save(base + ".zip")
        entry = {"tag": tag, "zip": base + ".zip", "timesteps": str(int(model.num_timesteps))}
        rms = self._obs_rms()
        if rms is not None:
            with open(base + "_vecnormalize.pkl", "wb") as fh:
                pickle.dump(rms, fh)
        try:
            entry["onnx"] = export_onnx(model, base + ".onnx", rms)
        except Exception as exc:  # export problems must not kill training
            entry["onnx_error"] = f"{type(exc).__name__}: {exc}"
        self.checkpoints.append(entry)
        self.hub.publish({"type": "checkpoint", **entry})
        return entry

    def _run(self) -> None:
        from stable_baselines3.common.vec_env import VecNormalize

        c = self.cfg
        try:
            os.makedirs(self.run_dir, exist_ok=True)
            with open(os.path.join(self.run_dir, "run.json"), "w") as fh:
                json.dump({"algo": c.algo, "config": params_to_dict(c)}, fh, indent=2)
            if c.vec_env == "subproc" and c.n_envs > 1:
                import torch

                # leave the CPU cores to the simulation workers
                torch.set_num_threads(max(1, (os.cpu_count() or 2) - c.n_envs))
            vec = make_vec_env(c.env, c.n_envs, c.seed, c.vec_env, c.start_method)
            if c.normalize_obs:
                vec = VecNormalize(vec, norm_obs=True, norm_reward=(c.algo == "PPO"), clip_obs=10.0,
                                   gamma=c.ppo.gamma if c.algo == "PPO" else c.sac.gamma)
            self._vec = vec
            model = self._build_model(vec)
            callback = _make_callback(self.hub, self)
            model.learn(total_timesteps=c.total_timesteps, callback=callback, progress_bar=False)
            self.timesteps = int(model.num_timesteps)
            self._save(model, "final")
            self.state = "stopped" if self._stop.is_set() else "finished"
        except Exception as exc:
            self.state = "failed"
            self.error = "".join(traceback.format_exception(exc))
            self.hub.publish({"type": "error", "error": str(exc)})
        finally:
            if self._vec is not None:
                try:
                    self._vec.close()
                except Exception:
                    pass
            self.hub.publish({"type": "finished", "state": self.state, "timesteps": self.timesteps})


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
def evaluate(policy: Optional[Callable[[np.ndarray], np.ndarray]], env_cfg: Optional[EnvConfig] = None,
             episodes: int = 5, seed: int = 1000, controller_factory: Optional[Callable[[BallBalancerEnv], Any]] = None
             ) -> Dict[str, float]:
    """Mean return / length / fall rate of a policy (obs -> action) or a classical controller."""
    env = BallBalancerEnv(env_cfg or EnvConfig())
    returns, lengths, falls = [], [], 0
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        ctrl = controller_factory(env) if controller_factory else None
        done, R, n = False, 0.0, 0
        while not done:
            if ctrl is not None:
                obs, r, term, trunc, _ = env.step_ctrl(ctrl.compute(env.last_state, env.target))
            else:
                obs, r, term, trunc, _ = env.step(policy(obs))
            R += r
            n += 1
            done = term or trunc
        falls += int(term)
        returns.append(R)
        lengths.append(n)
    env.close()
    return {"mean_return": float(np.mean(returns)), "mean_length": float(np.mean(lengths)),
            "fall_rate": falls / episodes, "episodes": episodes}


def _main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Train a BB-8 balancing policy with Stable-Baselines3")
    ap.add_argument("--algo", choices=["PPO", "SAC"], default="PPO")
    ap.add_argument("--scene", choices=["bb8", "planar_1d", "planar_2d"], default="bb8")
    ap.add_argument("--task", choices=["station_keeping", "velocity_tracking"], default="station_keeping")
    ap.add_argument("--steps", type=int, default=2_000_000)
    ap.add_argument("--n-envs", type=int, default=8)
    ap.add_argument("--vec-env", choices=["subproc", "dummy"], default="subproc")
    ap.add_argument("--no-noise", action="store_true", help="disable sensor noise")
    ap.add_argument("--no-latency", action="store_true", help="disable the pipeline delay")
    ap.add_argument("--no-dr", action="store_true", help="disable domain randomisation")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs")
    ap.add_argument("--name")
    args = ap.parse_args(argv)

    env_cfg = EnvConfig(scene=args.scene, task=args.task)
    env_cfg.hardware.noise_enabled = not args.no_noise
    env_cfg.hardware.latency_enabled = not args.no_latency
    env_cfg.randomization.enabled = not args.no_dr
    cfg = TrainingConfig(algo=args.algo, total_timesteps=args.steps, n_envs=args.n_envs, vec_env=args.vec_env,
                         env=env_cfg, seed=args.seed, out_dir=args.out, run_name=args.name)
    mgr = TrainingManager(cfg)

    def printer(ev: Dict[str, Any]) -> None:
        if ev["type"] == "progress":
            eps = list(mgr.hub.episodes)[-20:]
            mean_r = np.mean([e["reward"] for e in eps]) if eps else float("nan")
            mean_l = np.mean([e["length"] for e in eps]) if eps else float("nan")
            print(f"[{ev['timesteps']:>9d}] {ev['fps']:7.0f} sps  return(20)={mean_r:8.1f}  len={mean_l:6.0f}",
                  flush=True)
        elif ev["type"] in ("checkpoint", "finished", "error"):
            print(ev, flush=True)

    mgr.hub.subscribe(printer)
    mgr.start(background=True)
    try:
        while mgr.state == "running":
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("stopping...", flush=True)
        mgr.stop()
    mgr.wait()
    print(json.dumps(mgr.status(), indent=2))
    return 0 if mgr.state in ("finished", "stopped") else 1


if __name__ == "__main__":
    sys.exit(_main())

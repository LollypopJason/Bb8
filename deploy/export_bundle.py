"""Freeze a controller / trained policy into a robot deployment bundle (development PC).

    python -m deploy.export_bundle --out bundle/ --controller pid
    python -m deploy.export_bundle --out bundle/ --controller lqr --magnet 20
    python -m deploy.export_bundle --out bundle/ --controller rl --policy runs/ppo_x/final.zip

The bundle directory contains::

    bundle.json        robot geometry (J, radii, limits), PID gains, LQR gain matrix
                       (designed through SceneStepper for the configured delay), target limits,
                       policy layout; always includes PID and LQR so the robot can switch
    policy_mlp.npz     actor weights + observation normaliser (numpy inference on the robot)
    policy.onnx        same actor for onnxruntime (opset 13 for JetPack 4.x)

For ``--policy`` the robot geometry, hardware profile and parameters are taken
from the training run (``run.json``) so the bundle matches what the policy saw.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import pickle
import sys
from typing import Any, Dict, Optional

import mujoco
import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from config.composite_ball_params import (  # noqa: E402
    BB8Params,
    ControlMode,
    HardwareProfile,
    SimulationParams,
    load_params_json,
    params_from_dict,
    params_to_dict,
)
from controllers.classical import BalanceTarget, BallBotLQR, BallBotModel, BallWheelMixer, CascadedPIDGains  # noqa: E402
from deploy.runtime.bundle import FORMAT, VERSION  # noqa: E402
from envs.ball_balancer_env import EnvConfig  # noqa: E402
from models.mjcf_generator import build_bb8  # noqa: E402


def extract_mlp(model: Any) -> Dict[str, Any]:
    """Actor layers of an SB3 PPO / SAC model as numpy arrays + activations."""
    import torch

    policy = model.policy
    if hasattr(policy, "actor"):                       # SAC
        seq, head, output = policy.actor.latent_pi, policy.actor.mu, "tanh"
    else:                                              # PPO / A2C
        seq, head, output = policy.mlp_extractor.policy_net, policy.action_net, "clip"
    layers = []
    for mod in list(seq) + [head]:
        if isinstance(mod, torch.nn.Linear):
            layers.append([mod.weight.detach().cpu().numpy().astype(np.float64),
                           mod.bias.detach().cpu().numpy().astype(np.float64), "none"])
        elif isinstance(mod, torch.nn.Tanh):
            layers[-1][2] = "tanh"
        elif isinstance(mod, torch.nn.ReLU):
            layers[-1][2] = "relu"
        else:
            raise ValueError(f"unsupported layer in the actor: {mod}")
    return {"layers": layers, "output": output}


def build_bundle(out_dir: str, controller: str = "pid", params: Optional[BB8Params] = None,
                 hardware: Optional[HardwareProfile] = None, policy_path: Optional[str] = None,
                 lqr_q: Optional[Dict[str, float]] = None, max_body_rate: Optional[float] = None) -> Dict[str, Any]:
    warnings = []
    run_cfg = None
    algo = "PPO"
    if policy_path:
        run_json = os.path.join(os.path.dirname(os.path.abspath(policy_path)), "run.json")
        if not os.path.isfile(run_json):
            raise FileNotFoundError(f"{run_json} not found: export needs the training run's configuration")
        with open(run_json) as fh:
            run = json.load(fh)
        algo = run.get("algo", "PPO")
        env_dict = dict(run["config"]["env"])
        hw_dict = dict(env_dict.get("hardware", {}))
        if "ball_state_source" not in hw_dict:          # runs from before the odometry / prev-action fixes
            warnings.append("policy was trained before the odometry estimator and prev-action fixes: its "
                            "observations differ from the robot's - retrain before deploying")
            hw_dict["ball_state_source"] = "sensor"
            env_dict["hardware"] = hw_dict
        run_cfg = params_from_dict(EnvConfig, env_dict)
        if run_cfg.scene != "bb8":
            raise ValueError("only BB-8 policies can be deployed to the robot")
        if run_cfg.action_mode != "body":
            raise ValueError("only body-mode policies are supported on the robot")
        params = params or run_cfg.bb8
        hardware = hardware or run_cfg.hardware
        max_body_rate = max_body_rate or run_cfg.max_body_rate
        if run_cfg.hardware.ball_state_source != "odometry":
            warnings.append("policy was trained with ground-truth ball sensors; the robot uses wheel odometry")
    params = params or BB8Params()
    hw = hardware or HardwareProfile()
    if ControlMode(params.head.motor.control_mode) != ControlMode.VELOCITY:
        raise ValueError("the robot runtime drives steppers: the BB-8 motor control_mode must be 'velocity'")
    sim = SimulationParams(timestep=hw.loop.physics_timestep)
    bundle = build_bb8(params, sim)
    model = bundle.compile()
    meta = bundle.meta
    bbm = BallBotModel(model, meta)
    mixer = BallWheelMixer.from_meta(meta)
    hw.motor = params.head.motor
    lqr = BallBotLQR.design(bundle, hw, model=model, q_weights=lqr_q)
    pid = CascadedPIDGains()
    tgt = BalanceTarget()
    frame_skip = max(1, int(round(hw.loop.control_dt / model.opt.timestep)))
    control_dt = frame_skip * model.opt.timestep

    os.makedirs(out_dir, exist_ok=True)
    data: Dict[str, Any] = {
        "format": FORMAT,
        "version": VERSION,
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
        "mujoco": mujoco.__version__,
        "controller": controller,
        "robot": {
            "variant": "bb8",
            "ball_radius": bbm.R,
            "wheel_radius": params.head.wheel.radius,
            "n_wheels": len(meta.wheels),
            "J": mixer.J.tolist(),
            "active": list(bbm.active),
            "wheel_azimuth_deg": [float(np.degrees(w.azimuth)) for w in meta.wheels],
            "wheel_axle_head": [list(w.axle) for w in meta.wheels],
            "zenith_deg": params.head.zenith_angle_deg,
            "max_wheel_speed": params.head.motor.max_wheel_speed,
            "wheel_torque_limit": params.head.motor.wheel_torque_limit,
            "control_dt": control_dt,
            "design_delay_s": hw.latency.delay if hw.latency_enabled else 0.0,
            "yaw_feedforward": lqr.yaw_feedforward,
            "odometry_velocity_tau": hw.odometry_velocity_tau,
            "magnet": params_to_dict(params.head.magnet),
            "ball": params.ball.summary(),
        },
        "target": {"max_position_error": tgt.max_position_error, "max_accel": tgt.max_accel,
                   "max_yaw_accel": tgt.max_yaw_accel},
        "pid": {"kp": pid.inner.kp, "ki": pid.inner.ki, "kd": pid.inner.kd, "i_limit": pid.inner.i_limit,
                "outer_pos": pid.outer_pos, "outer_vel": pid.outer_vel, "outer_vel_ki": pid.outer_vel_ki,
                "max_tilt": pid.max_tilt, "yaw_kp": pid.yaw.kp, "yaw_feedforward": pid.yaw_feedforward,
                "rate_leak": pid.rate_leak},
        "lqr": {"K": lqr.K.tolist(), "buffer_len": lqr.design_data.buffer_len,
                "state_names": lqr.design_data.state_names, "spectral_radius": lqr.design_data.spectral_radius,
                "q_weights": lqr_q or {}},
        "policy": None,
        "params": {"bb8": params_to_dict(params), "hardware": params_to_dict(hw)},
        "warnings": warnings,
    }

    if policy_path:
        from stable_baselines3 import PPO, SAC
        from training.train_rl import export_onnx

        sb3 = (SAC if algo == "SAC" else PPO).load(policy_path, device="cpu")
        norm_path = policy_path[:-4] + "_vecnormalize.pkl"
        rms = None
        if os.path.isfile(norm_path):
            with open(norm_path, "rb") as fh:
                rms = pickle.load(fh)
        mlp = extract_mlp(sb3)
        arrays: Dict[str, np.ndarray] = {}
        for i, (W, b, _) in enumerate(mlp["layers"]):
            arrays[f"W{i}"], arrays[f"b{i}"] = W, b
        if rms is not None:
            arrays["obs_mean"] = np.asarray(rms.mean, dtype=np.float64)
            arrays["obs_var"] = np.asarray(rms.var, dtype=np.float64)
        np.savez(os.path.join(out_dir, "policy_mlp.npz"), **arrays)
        export_onnx(sb3, os.path.join(out_dir, "policy.onnx"), rms, opset=13)
        obs_dim = int(np.prod(sb3.observation_space.shape))
        act_dim = int(np.prod(sb3.action_space.shape))
        expected = 2 + 3 + len(meta.wheels) + 2 + 2 + 1 + 2 + 3
        if obs_dim != expected or act_dim != 3:
            raise ValueError(f"policy obs/act ({obs_dim}/{act_dim}) do not match a body-mode BB-8 "
                             f"({expected}/3)")
        data["policy"] = {
            "algo": algo, "source": os.path.abspath(policy_path), "mlp": "policy_mlp.npz", "onnx": "policy.onnx",
            "obs_dim": obs_dim, "act_dim": act_dim, "action_mode": "body",
            "max_body_rate": float(max_body_rate or EnvConfig().max_body_rate),
            "activations": [layer[2] for layer in mlp["layers"]], "output": mlp["output"],
            "clip_obs": 10.0, "epsilon": 1e-8,
            "ball_state_source": run_cfg.hardware.ball_state_source,
        }
    elif controller == "rl":
        raise ValueError("--controller rl needs --policy")

    with open(os.path.join(out_dir, "bundle.json"), "w") as fh:
        json.dump(data, fh, indent=2)
    return data


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Export a BB-8 deployment bundle for the robot")
    ap.add_argument("--out", required=True)
    ap.add_argument("--controller", choices=["pid", "lqr", "rl"], default="pid", help="default controller on the robot")
    ap.add_argument("--policy", help="SB3 .zip checkpoint (with *_vecnormalize.pkl and run.json next to it)")
    ap.add_argument("--params", help="BB8Params JSON (config.composite_ball_params.save_params_json)")
    ap.add_argument("--magnet", type=float, help="enable the magnetic preload with this force [N]")
    ap.add_argument("--delay-ms", type=float, help="pipeline delay the LQR is designed for")
    args = ap.parse_args(argv)
    params = load_params_json(args.params) if args.params else None
    if args.magnet is not None:
        params = params or BB8Params()
        params.head.magnet.enabled = True
        params.head.magnet.force = args.magnet
    hw = None
    if args.delay_ms is not None:
        hw = HardwareProfile()
        hw.latency.delay = args.delay_ms / 1000.0
    data = build_bundle(args.out, args.controller, params, hw, args.policy)
    print(f"bundle written to {args.out}: controller={data['controller']}, wheels={data['robot']['n_wheels']}, "
          f"LQR rho={data['lqr']['spectral_radius']:.4f}, policy={'yes' if data['policy'] else 'no'}")
    for w in data["warnings"]:
        print("WARNING:", w)
    return 0


if __name__ == "__main__":
    sys.exit(main())

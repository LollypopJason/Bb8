"""Robot-side controllers built from a deployment bundle (numpy only, Python 3.6).

Each class reproduces its simulator counterpart exactly (verified by
``tests/test_deploy.py``):

* :class:`Target`          <-> ``controllers.classical.BalanceTarget``
* :class:`PIDController`   <-> ``controllers.classical.CascadedPID`` (stepper / velocity mode)
* :class:`LQRController`   <-> ``controllers.classical.BallBotLQR``
* :class:`PolicyController` <-> ``envs.ball_balancer_env.BallBalancerEnv`` observation + action
  mapping, with the network evaluated in numpy (:class:`MLP`) or onnxruntime.

All controllers output **wheel speeds in rad/s** (signed about the simulator's
wheel axles); the runner converts them to step rates.
"""

import numpy as np

from .mathutil import rz


class Target(object):
    def __init__(self, max_position_error=0.3, max_accel=0.4, max_yaw_accel=2.0, track_position=True):
        self.position = np.zeros(2)
        self.velocity = np.zeros(2)
        self.yaw_rate = 0.0
        self.track_position = track_position
        self.max_position_error = max_position_error
        self.max_accel = max_accel
        self.max_yaw_accel = max_yaw_accel
        self.velocity_cmd = None
        self.yaw_rate_cmd = None

    @classmethod
    def from_bundle(cls, bundle):
        t = bundle.section("target")
        return cls(t["max_position_error"], t["max_accel"], t["max_yaw_accel"])

    def command(self, velocity, yaw_rate=0.0):
        self.velocity_cmd = np.asarray(velocity, dtype=float).copy()
        self.yaw_rate_cmd = float(yaw_rate)

    def advance(self, dt):
        if self.velocity_cmd is not None:
            dv = self.velocity_cmd - np.asarray(self.velocity, dtype=float)
            n = float(np.linalg.norm(dv))
            step = self.max_accel * dt
            self.velocity = self.velocity_cmd.copy() if n <= step else self.velocity + dv * (step / n)
        if self.yaw_rate_cmd is not None:
            dw = self.yaw_rate_cmd - self.yaw_rate
            self.yaw_rate += float(np.clip(dw, -self.max_yaw_accel * dt, self.max_yaw_accel * dt))
        self.position = np.asarray(self.position, dtype=float) + np.asarray(self.velocity, dtype=float) * dt


class _Mixer(object):
    def __init__(self, bundle):
        self.J = bundle.J
        self.J_pinv = np.linalg.pinv(self.J)
        self.lim = bundle.max_wheel_speed

    def clip(self, w):
        return np.clip(w, -self.lim, self.lim)


class PIDController(object):
    name = "pid"

    def __init__(self, bundle, gains=None):
        self.bundle = bundle
        self.g = dict(bundle.section("pid"))
        if gains:
            self.g.update(gains)
        self.dt = bundle.control_dt
        self.mix = _Mixer(bundle)
        self.variant = bundle.variant
        self.reset()

    def reset(self):
        self.i_tilt = np.zeros(2)
        self.i_vel = np.zeros(2)
        self.rate_cmd = np.zeros(3)
        self.tilt_ref = np.zeros(2)

    def compute(self, state, target, wheel_speeds=None):
        g, dt = self.g, self.dt
        e = state.error(target)
        self.i_vel = np.clip(self.i_vel + e[6:8] * dt, -1.0, 1.0)
        drive = g["outer_pos"] * e[4:6] + g["outer_vel"] * e[6:8] + g["outer_vel_ki"] * self.i_vel
        pitch_ref = float(np.clip(-drive[0], -g["max_tilt"], g["max_tilt"]))
        roll_ref = float(np.clip(+drive[1], -g["max_tilt"], g["max_tilt"]))
        self.tilt_ref = np.array([roll_ref, pitch_ref])
        tilt = np.array([state.roll, state.pitch])
        rates = np.array([state.roll_rate, state.pitch_rate])
        err = tilt - self.tilt_ref
        lim = g["i_limit"]
        self.i_tilt = np.clip(self.i_tilt + err * dt, -lim, lim)
        effort = g["kp"] * err + g["ki"] * self.i_tilt + g["kd"] * rates
        if self.variant == "1d":
            effort[0] = 0.0
        yaw_err = target.yaw_rate - state.yaw_rate if self.variant == "bb8" else 0.0
        self.rate_cmd[:2] += effort * dt
        if g["rate_leak"]:
            self.rate_cmd[:2] *= max(0.0, 1.0 - g["rate_leak"] * dt)
        self.rate_cmd[2] = (-(g["yaw_feedforward"] * target.yaw_rate + g["yaw_kp"] * yaw_err)
                            if self.variant == "bb8" else 0.0)
        speeds = self.mix.clip(self.mix.J.dot(self.rate_cmd))
        self.rate_cmd[:2] = self.mix.J_pinv.dot(speeds)[:2]       # anti-windup (as in the simulator)
        return speeds


class LQRController(object):
    name = "lqr"

    def __init__(self, bundle):
        sec = bundle.section("lqr")
        self.K = np.array(sec["K"], dtype=float)
        self.active = list(bundle.active)
        self.L = int(sec["buffer_len"])
        self.R = bundle.R
        self.variant = bundle.variant
        self.yaw_ff = float(bundle.section("robot").get("yaw_feedforward", 1.1))
        self.mix = _Mixer(bundle)
        self.nu = self.mix.J.shape[0]
        self.reset()

    def reset(self):
        self.buffer = [np.zeros(self.nu) for _ in range(self.L)]

    def feedforward(self, state, target):
        v = rz(-state.yaw)[:2, :2].dot(np.asarray(target.velocity, dtype=float))
        wz = target.yaw_rate if self.variant == "bb8" else 0.0
        omega = np.array([-v[1] / self.R, v[0] / self.R, -self.yaw_ff * wz])
        if self.variant == "1d":
            omega[[0, 2]] = 0.0
        return self.mix.J.dot(omega)

    def compute(self, state, target, wheel_speeds=None):
        x = state.error(target)[self.active]
        z = np.concatenate([x] + self.buffer) if self.L else x
        u_ff = self.feedforward(state, target)
        u = self.mix.clip(u_ff - self.K.dot(z))
        if self.L:
            self.buffer = [u - u_ff] + self.buffer[:-1]
        return u


class MLP(object):
    """Feed-forward actor exported from Stable-Baselines3 (obs normalisation included)."""

    def __init__(self, npz_path, spec):
        data = np.load(npz_path)
        self.mean = data["obs_mean"] if "obs_mean" in data else None
        self.std = np.sqrt(data["obs_var"] + spec.get("epsilon", 1e-8)) if "obs_var" in data else None
        self.clip_obs = float(spec.get("clip_obs", 10.0))
        self.layers = [(data["W%d" % i], data["b%d" % i], act) for i, act in enumerate(spec["activations"])]
        self.output = spec["output"]

    def __call__(self, obs):
        x = np.asarray(obs, dtype=np.float64)
        if self.mean is not None:
            x = np.clip((x - self.mean) / self.std, -self.clip_obs, self.clip_obs)
        for W, b, act in self.layers:
            x = W.dot(x) + b
            if act == "tanh":
                x = np.tanh(x)
            elif act == "relu":
                x = np.maximum(x, 0.0)
        if self.output == "tanh":
            x = np.tanh(x)
        return np.clip(x, -1.0, 1.0)


class OnnxPolicy(object):
    def __init__(self, path):
        import onnxruntime as ort   # optional dependency on the robot

        self.sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        self.inp = self.sess.get_inputs()[0].name

    def __call__(self, obs):
        x = np.asarray(obs, dtype=np.float32).reshape(1, -1)
        return np.clip(self.sess.run(None, {self.inp: x})[0][0].astype(float), -1.0, 1.0)


class PolicyController(object):
    """Observation building and action mapping of ``BallBalancerEnv`` (body action mode)."""

    name = "rl"

    def __init__(self, bundle, backend="numpy"):
        self.spec = bundle.section("policy")
        if self.spec["action_mode"] != "body":
            raise ValueError("only body-mode policies are supported on the robot")
        if backend == "onnx":
            self.net = OnnxPolicy(bundle.path(self.spec["onnx"]))
        else:
            self.net = MLP(bundle.path(self.spec["mlp"]), self.spec)
        self.mix = _Mixer(bundle)
        self.variant = bundle.variant
        self.max_body_rate = float(self.spec["max_body_rate"])
        self.n_act = int(self.spec["act_dim"])
        self.reset()

    def reset(self):
        self.prev_action = np.zeros(self.n_act)

    def observation(self, state, target, wheel_speeds):
        err = state.error(target)
        return np.concatenate([
            [state.roll, state.pitch],
            [state.roll_rate, state.pitch_rate, state.yaw_rate],
            np.asarray(wheel_speeds, dtype=float) / max(self.mix.lim, 1e-6),
            [state.vx, state.vy],
            err[6:8],
            [err[8]],
            err[4:6],
            self.prev_action,
        ]).astype(np.float32)

    def action_to_wheels(self, action):
        cmd = np.clip(np.asarray(action, dtype=float), -1.0, 1.0).copy()
        if self.variant == "1d":
            cmd[[0, 2]] = 0.0
        elif self.variant == "2d":
            cmd[2] = 0.0
        return self.mix.clip(self.mix.J.dot(cmd * self.max_body_rate))

    def compute(self, state, target, wheel_speeds):
        obs = self.observation(state, target, wheel_speeds)
        action = self.net(obs)
        self.prev_action = np.asarray(action, dtype=float)
        return self.action_to_wheels(action)


def make_controller(bundle, name=None, policy_backend="numpy"):
    name = name or bundle.default_controller
    if name == "pid":
        return PIDController(bundle)
    if name == "lqr":
        return LQRController(bundle)
    if name == "rl":
        return PolicyController(bundle, policy_backend)
    raise ValueError("unknown controller %r" % name)

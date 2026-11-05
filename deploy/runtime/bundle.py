"""Load a deployment bundle (``bundle.json`` + ``policy_mlp.npz`` / ``policy.onnx``)."""

import json
import os

import numpy as np

FORMAT = "bb8-bundle"
VERSION = 1


class Bundle(object):
    def __init__(self, data, directory="."):
        if data.get("format") != FORMAT or data.get("version") != VERSION:
            raise ValueError("not a bb8 bundle (format %r version %r)" % (data.get("format"), data.get("version")))
        self.data = data
        self.dir = directory
        r = data["robot"]
        self.variant = r["variant"]
        self.R = float(r["ball_radius"])
        self.J = np.array(r["J"], dtype=float)
        self.n_wheels = self.J.shape[0]
        self.max_wheel_speed = float(r["max_wheel_speed"])
        self.control_dt = float(r["control_dt"])
        self.active = list(r["active"])

    @classmethod
    def load(cls, directory):
        with open(os.path.join(directory, "bundle.json")) as fh:
            return cls(json.load(fh), directory)

    @property
    def default_controller(self):
        return self.data["controller"]

    def section(self, name):
        if name not in self.data or self.data[name] is None:
            raise KeyError("bundle has no %r section" % name)
        return self.data[name]

    def path(self, rel):
        return os.path.join(self.dir, rel)

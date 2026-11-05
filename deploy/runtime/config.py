"""Robot hardware configuration (JSON file on the Jetson). Python 3.6 compatible."""

import copy
import json
import math

DEFAULT_CONFIG = {
    "serial_port": "/dev/ttyACM0",       # Teensy USB serial
    "baud": 1000000,                     # ignored by USB-CDC, used for hardware UART
    "motor_steps_per_rev": 200,          # 1.8 deg stepper
    "microsteps": 16,
    "gear_ratio": 1.0,                   # direct drive
    "motor_sign": [1, 1, 1],             # +1 if a positive step rate turns the wheel positively about its sim axle
    "use_encoders": True,                # False: odometry from the commanded step counts
    "encoder_cpr": 1200,                 # decoded counts per wheel revolution
    "encoder_sign": [1, 1, 1],
    "max_accel": 300.0,                  # wheel acceleration limit in the firmware [rad/s^2]
    "watchdog_ms": 100,                  # firmware stops the motors without commands for this long
    "control_rate_hz": 100.0,
    "imu": {
        "type": "bno085",                # bno085 | mpu6050
        "i2c_bus": 1,                    # Jetson Nano header pins 3/5 = I2C bus 1
        "address": None,                 # None: driver default (0x4A BNO085, 0x68 MPU6050)
        "mount_euler_deg": [0.0, 0.0, 0.0],   # IMU axes -> head axes (head: x forward, z up)
        "gyro_bias": [0.0, 0.0, 0.0],    # rad/s, filled in by `calibrate`
        "level_offset_deg": [0.0, 0.0],  # roll, pitch of the IMU when the head is truly upright
    },
    "safety": {
        "tilt_cutoff_deg": 30.0,         # disarm beyond this tilt
        "arm_tilt_deg": 3.0,             # must be within this tilt ...
        "arm_hold_s": 1.0,               # ... for this long before arming
        "telemetry_timeout_s": 0.05,     # disarm if the MCU goes silent
        "max_wheel_speed": None,         # None: use the bundle's limit
    },
    "teleop": {"udp_port": 5005, "max_speed": 0.3, "max_yaw_rate": 0.6, "timeout_s": 0.5},
    "log_dir": "logs",
}


def _merge(base, override):
    out = copy.deepcopy(base)
    for k, v in override.items():
        if k not in out:
            raise KeyError("unknown config key %r" % k)
        if isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


class RobotConfig(object):
    def __init__(self, data=None):
        self.data = _merge(DEFAULT_CONFIG, data or {})
        self.validate()

    @classmethod
    def load(cls, path):
        with open(path) as fh:
            return cls(json.load(fh))

    def save(self, path):
        with open(path, "w") as fh:
            json.dump(self.data, fh, indent=2)

    def __getitem__(self, key):
        return self.data[key]

    @property
    def steps_per_rad(self):
        d = self.data
        return d["motor_steps_per_rev"] * d["microsteps"] * d["gear_ratio"] / (2.0 * math.pi)

    @property
    def counts_per_rad(self):
        return self.data["encoder_cpr"] / (2.0 * math.pi)

    def validate(self, n_wheels=None):
        d = self.data
        if d["motor_steps_per_rev"] <= 0 or d["microsteps"] <= 0 or d["gear_ratio"] <= 0:
            raise ValueError("steps / microsteps / gear must be positive")
        if d["encoder_cpr"] < 4:
            raise ValueError("encoder_cpr must be >= 4")
        if d["imu"]["type"] not in ("bno085", "mpu6050"):
            raise ValueError("imu.type must be bno085 or mpu6050")
        if not (10.0 <= d["control_rate_hz"] <= 1000.0):
            raise ValueError("control_rate_hz must be within 10..1000")
        for key in ("motor_sign", "encoder_sign"):
            if any(s not in (-1, 1) for s in d[key]):
                raise ValueError("%s entries must be +1 or -1" % key)
        if n_wheels is not None:
            for key in ("motor_sign", "encoder_sign"):
                if len(d[key]) != n_wheels:
                    raise ValueError("%s needs %d entries (one per wheel)" % (key, n_wheels))
        s = d["safety"]
        if not (0 < s["arm_tilt_deg"] < s["tilt_cutoff_deg"] <= 60):
            raise ValueError("need 0 < arm_tilt_deg < tilt_cutoff_deg <= 60")

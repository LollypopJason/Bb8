"""IMU drivers for the Jetson Nano (I2C bus 1 on header pins 3 = SDA, 5 = SCL).

Every driver returns an :class:`ImuSample` in the **head frame** (x forward,
y left, z up when upright): orientation quaternion (w, x, y, z) world<-head,
body rate [rad/s] and specific force [m/s^2]. The mounting rotation,
gyro bias and level offset come from the robot config (see ``calibrate``).

* ``bno085`` - Adafruit CircuitPython driver (``adafruit-circuitpython-bno08x``
  via Blinka). Uses the *game rotation vector* (gyro + accel fusion, no
  magnetometer): the stepper motors and the magnetic preload would corrupt
  a magnetometer heading.
* ``mpu6050`` - raw registers over ``smbus2`` with an on-Jetson Mahony-style
  complementary filter (gyro 500 dps, accel 4 g, DLPF 44 Hz).
"""

import math
import time

import numpy as np

from .mathutil import euler_xyz_deg_to_mat, mat_to_quat, quat_to_mat

GRAVITY = 9.80665


class ImuSample(object):
    __slots__ = ("quat", "gyro", "accel", "t")

    def __init__(self, quat, gyro, accel, t):
        self.quat, self.gyro, self.accel, self.t = quat, gyro, accel, t

    @property
    def rot(self):
        return quat_to_mat(self.quat)


class MountedImu(object):
    """Applies the mounting rotation, gyro bias and level offset to a raw IMU."""

    def __init__(self, raw, mount_euler_deg=(0, 0, 0), gyro_bias=(0, 0, 0), level_offset_deg=(0, 0)):
        self.raw = raw
        self.R_head_imu = euler_xyz_deg_to_mat(*mount_euler_deg)       # imu vectors -> head frame
        self.gyro_bias = np.asarray(gyro_bias, dtype=float)              # in IMU frame
        r, p = [math.radians(v) for v in level_offset_deg]
        cr, sr, cp, sp = math.cos(r), math.sin(r), math.cos(p), math.sin(p)
        # rotation of the "true upright" head relative to what the IMU reports
        self.R_level = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]]).dot(
            np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]]))

    def read(self):
        q_imu, gyro, accel, t = self.raw.read()
        R_world_imu = quat_to_mat(q_imu)
        R_world_head = R_world_imu.dot(self.R_head_imu.T).dot(self.R_level.T)
        g = self.R_head_imu.dot(np.asarray(gyro, dtype=float) - self.gyro_bias)
        a = self.R_head_imu.dot(np.asarray(accel, dtype=float))
        return ImuSample(mat_to_quat(R_world_head), g, a, t)


class Bno085(object):
    def __init__(self, i2c_bus=1, address=None, rate_hz=200):
        import board                     # Adafruit Blinka (Jetson supported)
        import busio
        from adafruit_bno08x import (BNO_REPORT_ACCELEROMETER, BNO_REPORT_GAME_ROTATION_VECTOR,
                                     BNO_REPORT_GYROSCOPE)
        from adafruit_bno08x.i2c import BNO08X_I2C

        if i2c_bus != 1:
            raise ValueError("the Blinka driver uses the default I2C bus (Jetson bus 1)")
        i2c = busio.I2C(board.SCL, board.SDA, frequency=400000)
        self.dev = BNO08X_I2C(i2c, address=address or 0x4A)
        interval = int(1e6 / rate_hz)
        for rep in (BNO_REPORT_GAME_ROTATION_VECTOR, BNO_REPORT_GYROSCOPE, BNO_REPORT_ACCELEROMETER):
            self.dev.enable_feature(rep, interval)

    def read(self):
        i, j, k, real = self.dev.game_quaternion
        gyro = self.dev.gyro
        accel = self.dev.acceleration
        return np.array([real, i, j, k]), np.array(gyro), np.array(accel), time.monotonic()


class ComplementaryFilter(object):
    """Mahony-style attitude filter: integrates the gyro, corrects roll/pitch toward gravity."""

    def __init__(self, kp=1.0, ki=0.0):
        self.kp, self.ki = kp, ki
        self.q = np.array([1.0, 0.0, 0.0, 0.0])
        self.integral = np.zeros(3)
        self.initialised = False

    def update(self, gyro, accel, dt):
        gyro = np.asarray(gyro, dtype=float)
        a = np.asarray(accel, dtype=float)
        n = np.linalg.norm(a)
        if not self.initialised and n > 1e-6:
            # start level with the measured gravity direction
            roll = math.atan2(a[1], a[2])
            pitch = math.atan2(-a[0], math.hypot(a[1], a[2]))
            cr, sr, cp, sp = math.cos(roll), math.sin(roll), math.cos(pitch), math.sin(pitch)
            R = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]]).dot(np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]]))
            self.q = mat_to_quat(R)
            self.initialised = True
            return self.q
        if n > 1e-6 and 0.5 * GRAVITY < n < 1.5 * GRAVITY:      # trust the accelerometer only near 1 g
            R = quat_to_mat(self.q)
            v = R.T.dot(np.array([0.0, 0.0, 1.0]))               # expected gravity direction (body)
            err = np.cross(a / n, v)
            self.integral += self.ki * err * dt
            gyro = gyro + self.kp * err + self.integral
        w, x, y, z = self.q
        gx, gy, gz = gyro
        dq = 0.5 * np.array([-x * gx - y * gy - z * gz, w * gx + y * gz - z * gy, w * gy - x * gz + z * gx,
                             w * gz + x * gy - y * gx])
        self.q = self.q + dq * dt
        self.q /= np.linalg.norm(self.q)
        return self.q


class Mpu6050(object):
    PWR_MGMT_1, CONFIG, GYRO_CONFIG, ACCEL_CONFIG, ACCEL_XOUT_H = 0x6B, 0x1A, 0x1B, 0x1C, 0x3B

    def __init__(self, i2c_bus=1, address=None):
        from smbus2 import SMBus

        self.addr = address or 0x68
        self.bus = SMBus(i2c_bus)
        self.bus.write_byte_data(self.addr, self.PWR_MGMT_1, 0x01)       # wake, PLL with X gyro
        self.bus.write_byte_data(self.addr, self.CONFIG, 0x03)           # DLPF 44 Hz
        self.bus.write_byte_data(self.addr, self.GYRO_CONFIG, 0x08)      # +-500 dps
        self.bus.write_byte_data(self.addr, self.ACCEL_CONFIG, 0x08)     # +-4 g
        self.filter = ComplementaryFilter()
        self.last_t = None

    def read(self):
        raw = self.bus.read_i2c_block_data(self.addr, self.ACCEL_XOUT_H, 14)
        vals = [((raw[i] << 8) | raw[i + 1]) for i in range(0, 14, 2)]
        vals = [v - 65536 if v > 32767 else v for v in vals]
        accel = np.array(vals[0:3]) / 8192.0 * GRAVITY
        gyro = np.radians(np.array(vals[4:7]) / 65.5)
        t = time.monotonic()
        dt = 0.01 if self.last_t is None else min(max(t - self.last_t, 1e-4), 0.05)
        self.last_t = t
        q = self.filter.update(gyro, accel, dt)
        return q.copy(), gyro, accel, t


def make_imu(cfg):
    imu_cfg = cfg["imu"]
    raw = Bno085(imu_cfg["i2c_bus"], imu_cfg["address"]) if imu_cfg["type"] == "bno085" else \
        Mpu6050(imu_cfg["i2c_bus"], imu_cfg["address"])
    return MountedImu(raw, imu_cfg["mount_euler_deg"], imu_cfg["gyro_bias"], imu_cfg["level_offset_deg"])

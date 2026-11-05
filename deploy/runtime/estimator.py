"""Balance-state estimation on the robot (and, optionally, in the simulator).

Inputs are what the hardware actually measures: head orientation and body rate
from the IMU, and wheel speeds from the encoders (or step counts). The ball
velocity is reconstructed from wheel odometry, assuming no slip:

    Omega_rel  = J^+ omega_wheels                       (head frame)
    Omega_ball = H Omega_rel + omega_head               (world frame)
    v_ball     = Omega_ball x (0, 0, R)                 (rolling on flat ground)

and integrated for the ball position (relative to where the robot was armed).
The state vector matches ``controllers.classical.BalanceState`` exactly::

    [roll, pitch, roll_rate, pitch_rate, ex, ey, vx, vy, yaw_rate]   (head yaw frame)
"""

import math

import numpy as np

from .mathutil import rz, zyx_angles


class State(object):
    __slots__ = ("roll", "pitch", "roll_rate", "pitch_rate", "ex", "ey", "vx", "vy", "yaw_rate", "yaw")

    def __init__(self, vec, yaw=0.0):
        (self.roll, self.pitch, self.roll_rate, self.pitch_rate, self.ex, self.ey, self.vx, self.vy,
         self.yaw_rate) = [float(v) for v in vec]
        self.yaw = float(yaw)

    def vector(self):
        return np.array([self.roll, self.pitch, self.roll_rate, self.pitch_rate, self.ex, self.ey, self.vx,
                         self.vy, self.yaw_rate])

    def error(self, target):
        """Identical to ``BalanceState.error``."""
        v_ref = rz(-self.yaw)[:2, :2].dot(np.asarray(target.velocity, dtype=float))
        e = self.vector()
        if not target.track_position:
            e[4:6] = 0.0
        else:
            n = float(np.linalg.norm(e[4:6]))
            if n > target.max_position_error:
                e[4:6] *= target.max_position_error / n
        e[6] -= v_ref[0]
        e[7] -= v_ref[1]
        e[8] -= target.yaw_rate
        return e

    @property
    def tilt_deg(self):
        return math.degrees(max(abs(self.roll), abs(self.pitch)))


class OdometryEstimator(object):
    def __init__(self, J, ball_radius, active=None, velocity_tau=0.0):
        """``velocity_tau`` [s]: first-order low-pass on the odometry velocity (0 = off)."""
        self.velocity_tau = float(velocity_tau)
        self.J = np.atleast_2d(np.asarray(J, dtype=float))
        self.J_pinv = np.linalg.pinv(self.J)
        self.R = float(ball_radius)
        self.mask = np.ones(9, dtype=bool)
        if active is not None:
            self.mask[:] = False
            self.mask[list(active)] = True
        self.reset()

    def reset(self, position=(0.0, 0.0)):
        self.position = np.array(position, dtype=float)
        self.velocity = np.zeros(2)

    def update(self, head_rot, gyro_head, wheel_speeds, dt, target_position=(0.0, 0.0)):
        """``head_rot``: world<-head rotation; ``gyro_head``: body rate in head frame [rad/s]."""
        H = np.asarray(head_rot, dtype=float)
        yaw, pitch, roll = zyx_angles(H)
        w_world = H.dot(np.asarray(gyro_head, dtype=float))
        omega_rel = self.J_pinv.dot(np.asarray(wheel_speeds, dtype=float))
        omega_ball = H.dot(omega_rel) + w_world
        v_raw = np.array([omega_ball[1] * self.R, -omega_ball[0] * self.R])
        if self.velocity_tau > 0 and dt > 0:
            a = dt / (self.velocity_tau + dt)
            v = self.velocity + a * (v_raw - self.velocity)
        else:
            v = v_raw
        self.velocity = v
        if dt > 0:
            self.position = self.position + v * dt
        Rz_t = rz(-yaw)
        w = Rz_t.dot(w_world)
        err = Rz_t[:2, :2].dot(self.position - np.asarray(target_position, dtype=float))
        vy = Rz_t[:2, :2].dot(v)
        y = np.array([roll, pitch, w[0], w[1], err[0], err[1], vy[0], vy[1], w[2]])
        y[~self.mask] = 0.0
        return State(y, yaw)


class WheelSpeedEstimator(object):
    """Wheel speed from quantised counts and the MCU's microsecond clock (32-bit wrap safe)."""

    def __init__(self, counts_per_rad, signs):
        self.k = 1.0 / float(counts_per_rad)
        self.signs = np.asarray(signs, dtype=float)
        self.last_counts = None
        self.last_us = None
        self.speeds = np.zeros(len(signs))
        self.angles = np.zeros(len(signs))

    def update(self, counts, mcu_us):
        counts = np.asarray(counts, dtype=np.int64)
        if self.last_counts is not None:
            dt = ((mcu_us - self.last_us) & 0xFFFFFFFF) * 1e-6
            if dt > 0:
                dc = counts - self.last_counts
                dc = (dc + 2 ** 31) % 2 ** 32 - 2 ** 31        # int32 wrap
                self.speeds = self.signs * dc * self.k / dt
        self.angles = self.signs * counts * self.k
        self.last_counts, self.last_us = counts, mcu_us
        return self.speeds

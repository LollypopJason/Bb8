"""Host side of the microcontroller link (pyserial)."""

import time

from . import protocol as P


class SerialMCU(object):
    """Sends CMD frames, keeps the newest TELEM frame. Non-blocking reads."""

    def __init__(self, port, baud=1000000, n_motors=3, max_accel_steps=150000.0, watchdog_ms=100, ser=None):
        if ser is None:
            import serial

            ser = serial.Serial(port, baud, timeout=0, write_timeout=0.05)
        self.ser = ser
        self.n = n_motors
        self.parser = P.StreamParser()
        self.seq = 0
        self.telem = None
        self.telem_time = None
        self.telem_count = 0
        self.max_accel_steps = max_accel_steps
        self.watchdog_ms = watchdog_ms

    def configure(self):
        self.ser.write(P.encode_cfg(self.max_accel_steps, self.watchdog_ms, self.n))

    def send(self, step_rates, enable):
        self.seq = (self.seq + 1) & 0xFFFF
        self.ser.write(P.encode_cmd(self.seq, step_rates, enable))

    def poll(self):
        """Read everything available; returns the newest telemetry (or the previous one)."""
        waiting = getattr(self.ser, "in_waiting", 0)
        data = self.ser.read(max(int(waiting), 1))
        if data:
            for ftype, payload in self.parser.feed(data):
                if ftype != P.T_TELEM:
                    continue
                try:
                    self.telem = P.decode(ftype, payload)
                except ValueError:
                    self.parser.crc_errors += 1
                    continue
                self.telem_time = time.monotonic()
                self.telem_count += 1
        return self.telem

    def wait_for_telemetry(self, timeout=2.0):
        t0 = time.monotonic()
        start = self.telem_count
        while time.monotonic() - t0 < timeout:
            self.poll()
            if self.telem_count > start:
                return self.telem
            time.sleep(0.002)
        raise TimeoutError("no telemetry from the microcontroller")

    def close(self):
        try:
            self.send([0.0] * self.n, False)
        except (OSError, ValueError):
            pass
        finally:
            self.ser.close()


def make_motor_link(cfg, n_motors):
    """SerialMCU for the configured stepper back end (Jetson GPIO daemon or USB microcontroller)."""
    kw = dict(n_motors=n_motors, max_accel_steps=cfg["max_accel"] * cfg.steps_per_rad,
              watchdog_ms=cfg["watchdog_ms"])
    if cfg["stepper_backend"] == "jetson_gpio":
        from .link import SocketLink

        return SerialMCU(None, ser=SocketLink(cfg["stepgen_socket"]), **kw)
    return SerialMCU(cfg["serial_port"], cfg["baud"], **kw)

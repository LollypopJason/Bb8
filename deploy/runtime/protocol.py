"""Binary serial protocol between the Jetson (host) and the microcontroller.

Frame::

    0xAA 0x55 | type (u8) | len (u8) | payload (len bytes) | crc16 (u16, little endian)

CRC-16/CCITT-FALSE (poly 0x1021, init 0xFFFF) over ``type, len, payload``.
All multi-byte fields are little endian. Identical definitions live in
``deploy/firmware/bb8_mcu/bb8_protocol.h``.

==========  ======  ==================================================================
type        dir     payload
==========  ======  ==================================================================
0x01 CMD    host>   u16 seq, u8 flags (bit0 enable), u8 n, f32 step_rate[n] [steps/s]
0x02 CFG    host>   f32 max_accel [steps/s^2], u16 watchdog_ms, u8 n
0x81 TELEM  >host   u16 seq_echo, u32 mcu_us, u8 flags, u8 n, i32 encoder[n], i32 steps[n]
==========  ======  ==================================================================

TELEM flags: bit0 drivers enabled, bit1 watchdog tripped, bit2 configured.
"""

import struct

SYNC = b"\xAA\x55"
T_CMD = 0x01
T_CFG = 0x02
T_TELEM = 0x81
MAX_MOTORS = 4

FLAG_ENABLE = 0x01
TF_ENABLED = 0x01
TF_WATCHDOG = 0x02
TF_CONFIGURED = 0x04


def crc16(data, crc=0xFFFF):
    for byte in bytearray(data):
        crc ^= byte << 8
        for _ in range(8):
            crc = ((crc << 1) ^ 0x1021) & 0xFFFF if crc & 0x8000 else (crc << 1) & 0xFFFF
    return crc


def frame(ftype, payload):
    if len(payload) > 255:
        raise ValueError("payload too long")
    body = struct.pack("<BB", ftype, len(payload)) + payload
    return SYNC + body + struct.pack("<H", crc16(body))


def encode_cmd(seq, step_rates, enable):
    n = len(step_rates)
    if not 1 <= n <= MAX_MOTORS:
        raise ValueError("1..4 motors")
    payload = struct.pack("<HBB", seq & 0xFFFF, FLAG_ENABLE if enable else 0, n)
    payload += struct.pack("<%df" % n, *[float(r) for r in step_rates])
    return frame(T_CMD, payload)


def encode_cfg(max_accel, watchdog_ms, n):
    return frame(T_CFG, struct.pack("<fHB", float(max_accel), int(watchdog_ms), int(n)))


def encode_telem(seq_echo, mcu_us, flags, encoders, steps):
    n = len(encoders)
    payload = struct.pack("<HIBB", seq_echo & 0xFFFF, mcu_us & 0xFFFFFFFF, flags, n)
    payload += struct.pack("<%di" % n, *encoders) + struct.pack("<%di" % n, *steps)
    return frame(T_TELEM, payload)


class Cmd(object):
    __slots__ = ("seq", "enable", "step_rates")

    def __init__(self, seq, enable, step_rates):
        self.seq, self.enable, self.step_rates = seq, enable, step_rates


class Cfg(object):
    __slots__ = ("max_accel", "watchdog_ms", "n")

    def __init__(self, max_accel, watchdog_ms, n):
        self.max_accel, self.watchdog_ms, self.n = max_accel, watchdog_ms, n


class Telem(object):
    __slots__ = ("seq_echo", "mcu_us", "flags", "encoders", "steps")

    def __init__(self, seq_echo, mcu_us, flags, encoders, steps):
        self.seq_echo, self.mcu_us, self.flags = seq_echo, mcu_us, flags
        self.encoders, self.steps = encoders, steps

    @property
    def enabled(self):
        return bool(self.flags & TF_ENABLED)

    @property
    def watchdog(self):
        return bool(self.flags & TF_WATCHDOG)


def decode(ftype, payload):
    """Decode a payload into Cmd / Cfg / Telem; raises ValueError on malformed data."""
    try:
        if ftype == T_CMD:
            seq, flags, n = struct.unpack_from("<HBB", payload)
            if not 1 <= n <= MAX_MOTORS or len(payload) != 4 + 4 * n:
                raise ValueError("bad CMD length")
            return Cmd(seq, bool(flags & FLAG_ENABLE), list(struct.unpack_from("<%df" % n, payload, 4)))
        if ftype == T_CFG:
            if len(payload) != 7:
                raise ValueError("bad CFG length")
            return Cfg(*struct.unpack("<fHB", payload))
        if ftype == T_TELEM:
            seq, us, flags, n = struct.unpack_from("<HIBB", payload)
            if not 1 <= n <= MAX_MOTORS or len(payload) != 8 + 8 * n:
                raise ValueError("bad TELEM length")
            enc = list(struct.unpack_from("<%di" % n, payload, 8))
            steps = list(struct.unpack_from("<%di" % n, payload, 8 + 4 * n))
            return Telem(seq, us, flags, enc, steps)
    except struct.error as exc:
        raise ValueError(str(exc))
    raise ValueError("unknown frame type 0x%02x" % ftype)


class StreamParser(object):
    """Incremental frame parser that resynchronises after noise / CRC errors."""

    def __init__(self):
        self.buf = bytearray()
        self.crc_errors = 0
        self.frames = 0

    def feed(self, data):
        self.buf.extend(data)
        out = []
        while True:
            i = self.buf.find(SYNC)
            if i < 0:
                del self.buf[:-1]          # keep a possible half sync byte
                return out
            if i:
                del self.buf[:i]
            if len(self.buf) < 6:
                return out
            length = self.buf[3]
            total = 2 + 2 + length + 2
            if len(self.buf) < total:
                return out
            body = bytes(self.buf[2:4 + length])
            (crc_rx,) = struct.unpack_from("<H", self.buf, 4 + length)
            if crc16(body) != crc_rx:
                self.crc_errors += 1
                del self.buf[:2]           # skip this sync and search again
                continue
            del self.buf[:total]
            self.frames += 1
            out.append((body[0], body[2:]))

"""Byte-stream link to the ``bb8_stepgen`` daemon (Unix socket) with a pyserial-like interface,
so :class:`deploy.runtime.mcu.SerialMCU` talks to it exactly like to a USB microcontroller."""

import socket
import time


class SocketLink(object):
    def __init__(self, path, connect_timeout=5.0):
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        t0 = time.monotonic()
        while True:
            try:
                self.sock.connect(path)
                break
            except (FileNotFoundError, ConnectionRefusedError):
                if time.monotonic() - t0 > connect_timeout:
                    raise ConnectionError("bb8_stepgen is not running (no socket at %s)" % path)
                time.sleep(0.05)
        self.sock.setblocking(False)
        self.in_waiting = 4096

    def write(self, data):
        self.sock.setblocking(True)
        try:
            self.sock.sendall(data)
        finally:
            self.sock.setblocking(False)

    def read(self, n=4096):
        try:
            data = self.sock.recv(n)
        except (BlockingIOError, InterruptedError):
            return b""
        if data == b"":
            raise ConnectionError("bb8_stepgen closed the connection")
        return data

    def close(self):
        self.sock.close()

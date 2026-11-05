"""Keyboard teleop client (run on a laptop on the same network as the Jetson).

    python3 -m deploy.runtime.teleop --host jetson.local

W/S = +-x, A/D = +-y, Q/E = yaw, space = arm, x = disarm, Esc = quit.
Sends JSON datagrams to the runner's UDP port at 20 Hz; the runner zeroes the
command if datagrams stop arriving (``teleop.timeout_s``).
"""

import argparse
import curses
import json
import socket
import time


def _loop(stdscr, host, port, speed, yaw):
    stdscr.nodelay(True)
    curses.cbreak()
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    held = {}
    hold_s = 0.25                     # terminal key repeat keeps a key "held"
    while True:
        now = time.time()
        while True:
            ch = stdscr.getch()
            if ch == -1:
                break
            if ch == 27:
                sock.sendto(json.dumps({"disarm": True}).encode(), (host, port))
                return
            key = chr(ch).lower() if 0 <= ch < 256 else ""
            msg = {}
            if key == " ":
                msg["arm"] = True
            elif key == "x":
                msg["disarm"] = True
            elif key in "wasdqe" and key:
                held[key] = now
            if msg:
                sock.sendto(json.dumps(msg).encode(), (host, port))
        active = {k for k, t in held.items() if now - t < hold_s}
        vx = speed * (("w" in active) - ("s" in active))
        vy = speed * (("a" in active) - ("d" in active))
        wz = yaw * (("q" in active) - ("e" in active))
        sock.sendto(json.dumps({"vx": vx, "vy": vy, "wz": wz}).encode(), (host, port))
        stdscr.erase()
        stdscr.addstr(0, 0, "BB-8 teleop -> %s:%d   W/S A/D Q/E move, space arm, x disarm, Esc quit" % (host, port))
        stdscr.addstr(2, 0, "vx %+.2f m/s   vy %+.2f m/s   wz %+.2f rad/s" % (vx, vy, wz))
        stdscr.refresh()
        time.sleep(0.05)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--host", required=True)
    ap.add_argument("--port", type=int, default=5005)
    ap.add_argument("--speed", type=float, default=0.3)
    ap.add_argument("--yaw", type=float, default=0.6)
    args = ap.parse_args(argv)
    curses.wrapper(_loop, args.host, args.port, args.speed, args.yaw)


if __name__ == "__main__":
    main()

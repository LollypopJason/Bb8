"""Deployment of BB-8 controllers / policies to the real robot (Jetson Nano + Teensy).

* ``deploy.export_bundle`` (development PC, needs MuJoCo): freezes a controller or a
  trained policy plus the robot geometry into a self-contained bundle directory.
* ``deploy.runtime`` (Jetson Nano, Python >= 3.6, numpy only): reads the IMU and the
  microcontroller, estimates the balance state, runs PID / LQR / RL and streams step
  rates to the microcontroller with safety interlocks.
* ``deploy.firmware.bb8_mcu`` (Teensy 4.x): step pulse generation, encoder counting,
  watchdog.
* ``deploy.sim_io`` (development PC): runs the unmodified robot runtime against the
  MuJoCo simulation (software-in-the-loop) and emulates the microcontroller on a pty.
"""

"""Deployment of BB-8 controllers / policies to the real robot.

Default hardware: Jetson Nano (control), Arduino Nano 33 IoT on the head (IMU over USB),
stepper drivers on Jetson GPIO driven by the ``jetson_stepgen`` daemon. A Teensy step
generator and I2C IMUs are supported alternatives.


* ``deploy.export_bundle`` (development PC, needs MuJoCo): freezes a controller or a
  trained policy plus the robot geometry into a self-contained bundle directory.
* ``deploy.runtime`` (Jetson Nano, Python >= 3.6, numpy only): fuses the IMU, estimates
  the balance state, runs PID / LQR / RL and streams step rates to the step generator
  with safety interlocks.
* ``deploy.jetson_stepgen`` (C daemon): real-time STEP/DIR/ENABLE on Jetson GPIO.
* ``deploy.firmware.bb8_imu`` (Arduino Nano 33 IoT / RP2040 Connect): 416 Hz IMU stream.
* ``deploy.firmware.bb8_mcu`` (Teensy 4.x, alternative): step generation, encoders.
* ``deploy.sim_io`` (development PC): runs the unmodified robot runtime against the
  MuJoCo simulation (software-in-the-loop) and emulates the microcontroller on a pty.
"""

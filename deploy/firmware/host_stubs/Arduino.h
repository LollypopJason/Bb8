// Minimal Arduino / Teensyduino API stubs so bb8_mcu.ino can be compiled on a PC (syntax and
// type checking only - see tests/test_deploy.py). Not used on the microcontroller.
#pragma once
#include <cmath>
#include <cstddef>
#include <cstdint>
#define HIGH 1
#define LOW 0
#define OUTPUT 1
#define INPUT_PULLUP 2
#define CHANGE 3
#define LED_BUILTIN 13
inline void pinMode(uint8_t, uint8_t) {}
inline void digitalWriteFast(uint8_t, uint8_t) {}
inline uint8_t digitalReadFast(uint8_t) { return 0; }
inline uint8_t digitalPinToInterrupt(uint8_t p) { return p; }
inline void attachInterrupt(uint8_t, void (*)(), int) {}
inline void noInterrupts() {}
inline void interrupts() {}
inline uint32_t micros() { static uint32_t t = 0; return t += 100; }
inline uint32_t millis() { return micros() / 1000; }
inline void delay(uint32_t) {}
inline void digitalWrite(uint8_t, uint8_t) {}
inline int digitalRead(uint8_t) { return 0; }
struct IntervalTimer { bool begin(void (*)(), float) { return true; } void priority(uint8_t) {} };
struct SerialStub {
  void begin(unsigned long) {}
  int available() { return 0; }
  int read() { return -1; }
  size_t write(const uint8_t*, size_t n) { return n; }
};
#ifndef BB8_CUSTOM_SERIAL
extern SerialStub Serial;
#endif

// Fake LSM6DS3 on the I2C bus (WHO_AM_I 0x69, data always ready, gravity on +z)
struct WireStub {
  uint8_t reg = 0, pending = 0;
  void begin() {}
  void setClock(uint32_t) {}
  void beginTransmission(uint8_t) { pending = 0xFF; }
  size_t write(uint8_t v) { if (pending == 0xFF) { reg = v; pending = 0; } return 1; }
  uint8_t endTransmission(bool = true) { return 0; }
  uint8_t requestFrom(uint8_t, uint8_t n) { pending = n; return n; }
  int available() { return pending; }
  int read() {
    if (!pending) return -1;
    pending--;
    uint8_t r = reg++;
    if (r == 0x0F) return 0x69;
    if (r == 0x1E) return 0x03;
    if (r == 0x2C) return 0x00;                      // accel z low byte
    if (r == 0x2D) return 0x20;                      // accel z = 0x2000 = 8192 LSB = 1 g at 4 g scale
    return 0;
  }
};
extern WireStub Wire;

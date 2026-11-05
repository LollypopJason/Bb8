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
inline void pinMode(uint8_t, uint8_t) {}
inline void digitalWriteFast(uint8_t, uint8_t) {}
inline uint8_t digitalReadFast(uint8_t) { return 0; }
inline uint8_t digitalPinToInterrupt(uint8_t p) { return p; }
inline void attachInterrupt(uint8_t, void (*)(), int) {}
inline void noInterrupts() {}
inline void interrupts() {}
inline uint32_t micros() { static uint32_t t = 0; return t += 100; }
struct IntervalTimer { bool begin(void (*)(), float) { return true; } void priority(uint8_t) {} };
struct SerialStub {
  void begin(unsigned long) {}
  int available() { return 0; }
  int read() { return -1; }
  size_t write(const uint8_t*, size_t n) { return n; }
};
extern SerialStub Serial;

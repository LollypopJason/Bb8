// Host build driver for the IMU sketch: prints the IMU frames it would send (hex).
#include <cstdio>
#define BB8_CUSTOM_SERIAL
#include "Arduino.h"
struct PrintSerial {
  void begin(unsigned long) {}
  size_t write(const uint8_t* b, size_t n) { for (size_t i = 0; i < n; i++) printf("%02x", b[i]); printf("\n"); return n; }
};
static PrintSerial Serial;
WireStub Wire;
#include "../bb8_imu/bb8_imu.ino"
int main() { setup(); for (int i = 0; i < 3; i++) loop(); return 0; }

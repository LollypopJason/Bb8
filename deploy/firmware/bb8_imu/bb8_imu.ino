// BB-8 head IMU firmware for the Arduino Nano 33 IoT (LSM6DS3) or Nano RP2040 Connect (LSM6DSOX).
//
// Streams raw gyro + accelerometer samples at 416 Hz to the Jetson Nano over USB serial as
// bb8_protocol.h IMU frames (rad/s, m/s^2, sensor axes, micros() timestamp). Attitude fusion,
// gyro-bias and mounting corrections run on the Jetson (deploy/runtime/imu.py::ArduinoImu).
// Optionally also counts up to 3 quadrature wheel encoders (set N_ENC / pins below).
//
// Mount the board rigidly on the head and enter its orientation in robot.json (imu.mount_euler_deg).

#include <Arduino.h>
#include <Wire.h>
#include "bb8_protocol.h"

// ----------------------------------------------------------------------------- options (edit)
static const uint8_t IMU_ADDR = 0x6A;                 // on-board IMU on both boards
static const uint8_t N_ENC = 0;                       // 0..3 wheel encoders on this board
static const uint8_t ENC_A_PIN[3] = {2, 9, 11};       // Nano 33 IoT interrupt-capable pins
static const uint8_t ENC_B_PIN[3] = {3, 10, 13};

// ----------------------------------------------------------------------------- LSM6DS3 / LSM6DSOX
static const uint8_t REG_WHO_AM_I = 0x0F, REG_CTRL1_XL = 0x10, REG_CTRL2_G = 0x11, REG_CTRL3_C = 0x12,
                     REG_STATUS = 0x1E, REG_OUTX_L_G = 0x22;
static const float GYRO_SCALE = 0.0175f * 3.14159265f / 180.0f;   // 500 dps: 17.5 mdps/LSB -> rad/s
static const float ACCEL_SCALE = 0.122e-3f * 9.80665f;            // 4 g: 0.122 mg/LSB -> m/s^2

static void writeReg(uint8_t reg, uint8_t val) {
  Wire.beginTransmission(IMU_ADDR); Wire.write(reg); Wire.write(val); Wire.endTransmission();
}
static uint8_t readReg(uint8_t reg) {
  Wire.beginTransmission(IMU_ADDR); Wire.write(reg); Wire.endTransmission(false);
  Wire.requestFrom(IMU_ADDR, (uint8_t)1);
  return Wire.available() ? (uint8_t)Wire.read() : 0;
}
static bool readBlock(uint8_t reg, uint8_t* buf, uint8_t n) {
  Wire.beginTransmission(IMU_ADDR); Wire.write(reg); Wire.endTransmission(false);
  if (Wire.requestFrom(IMU_ADDR, n) != n) return false;
  for (uint8_t i = 0; i < n; i++) buf[i] = (uint8_t)Wire.read();
  return true;
}

static bool imuInit() {
  uint8_t who = readReg(REG_WHO_AM_I);
  if (who != 0x69 && who != 0x6A && who != 0x6C) return false;     // DS3, DS3TR-C, DSOX
  writeReg(REG_CTRL3_C, 0x01);                                      // software reset
  delay(20);
  writeReg(REG_CTRL3_C, 0x44);                                      // block data update, auto-increment
  writeReg(REG_CTRL1_XL, 0x68);                                     // accel 416 Hz, +-4 g
  writeReg(REG_CTRL2_G, 0x64);                                      // gyro 416 Hz, +-500 dps
  return true;
}

// ----------------------------------------------------------------------------- encoders
static volatile int32_t encCount[3];
static volatile uint8_t encState[3];
static const int8_t QDEC[16] = {0, -1, 1, 0, 1, 0, 0, -1, -1, 0, 0, 1, 0, 1, -1, 0};
static inline void encUpdate(uint8_t i) {
  uint8_t s = (uint8_t)((digitalRead(ENC_A_PIN[i]) << 1) | digitalRead(ENC_B_PIN[i]));
  encCount[i] += QDEC[(encState[i] << 2) | s];
  encState[i] = s;
}
static void enc0() { encUpdate(0); }
static void enc1() { encUpdate(1); }
static void enc2() { encUpdate(2); }
static void (*const ENC_ISR[3])() = {enc0, enc1, enc2};

// ----------------------------------------------------------------------------- main
static bool imuOk = false;
static uint16_t seq = 0;

void setup() {
  Serial.begin(1000000);                 // USB CDC: baud is ignored
  Wire.begin();
  Wire.setClock(400000);
  pinMode(LED_BUILTIN, OUTPUT);
  imuOk = imuInit();
  for (uint8_t i = 0; i < N_ENC; i++) {
    pinMode(ENC_A_PIN[i], INPUT_PULLUP);
    pinMode(ENC_B_PIN[i], INPUT_PULLUP);
    encState[i] = (uint8_t)((digitalRead(ENC_A_PIN[i]) << 1) | digitalRead(ENC_B_PIN[i]));
    attachInterrupt(digitalPinToInterrupt(ENC_A_PIN[i]), ENC_ISR[i], CHANGE);
    attachInterrupt(digitalPinToInterrupt(ENC_B_PIN[i]), ENC_ISR[i], CHANGE);
  }
}

void loop() {
  if (!imuOk) {                          // no IMU found: blink and retry
    digitalWrite(LED_BUILTIN, (millis() / 100) % 2);
    static uint32_t lastTry = 0;
    if (millis() - lastTry > 1000) { lastTry = millis(); imuOk = imuInit(); }
    return;
  }
  uint8_t status = readReg(REG_STATUS);
  if ((status & 0x03) != 0x03) return;   // wait for both accel (XLDA) and gyro (GDA) data
  uint8_t raw[12];
  if (!readBlock(REG_OUTX_L_G, raw, 12)) return;
  uint32_t us = micros();
  float gyro[3], accel[3];
  for (int i = 0; i < 3; i++) {
    gyro[i] = (int16_t)(raw[2 * i] | (raw[2 * i + 1] << 8)) * GYRO_SCALE;
    accel[i] = (int16_t)(raw[6 + 2 * i] | (raw[7 + 2 * i] << 8)) * ACCEL_SCALE;
  }
  int32_t enc[3];
  noInterrupts();
  for (uint8_t i = 0; i < N_ENC; i++) enc[i] = encCount[i];
  interrupts();
  uint8_t out[6 + 31 + 4 * BB8_MAX_MOTORS];
  size_t n = bb8_encode_imu(seq++, us, gyro, accel, N_ENC, enc, out);
  Serial.write(out, n);
  digitalWrite(LED_BUILTIN, (seq >> 8) & 1);   // ~1.6 Hz heartbeat while streaming
}

// BB-8 wheel controller firmware for Teensy 4.0 / 4.1 (Arduino + Teensyduino).
//
// * Receives step-rate commands from the Jetson Nano over USB serial (bb8_protocol.h).
// * Generates STEP/DIR pulses for up to 4 stepper drivers (A4988 / DRV8825 / TMC2209 in STEP/DIR
//   mode) from a 50 kHz IntervalTimer with phase accumulators -> up to 25 kHz steps per motor.
// * Limits acceleration (CFG max_accel, steps/s^2) so the steppers do not stall.
// * Counts quadrature encoders (x4 decoding) on interrupts.
// * Watchdog: without a CMD for watchdog_ms the motors ramp to zero and the drivers are disabled.
// * Replies to every CMD with TELEM (encoder counts, step counters, micros()) and sends TELEM
//   every 20 ms while idle.
//
// Behaviour is mirrored by deploy/sim_io.py::FirmwareModel for software-in-the-loop tests.

#include <Arduino.h>
#include "bb8_protocol.h"

// ----------------------------------------------------------------------------- pin map (edit)
static const uint8_t N_MOTORS = 3;
static const uint8_t STEP_PIN[BB8_MAX_MOTORS] = {2, 4, 6, 8};
static const uint8_t DIR_PIN[BB8_MAX_MOTORS] = {3, 5, 7, 9};
static const uint8_t ENABLE_PIN = 10;            // shared, active LOW (A4988 / DRV8825 / TMC2209 EN)
static const uint8_t ENC_A_PIN[BB8_MAX_MOTORS] = {14, 16, 18, 20};
static const uint8_t ENC_B_PIN[BB8_MAX_MOTORS] = {15, 17, 19, 21};
static const bool USE_ENCODERS = true;

// ----------------------------------------------------------------------------- state
static const uint32_t TICK_HZ = 50000;            // step generator ISR rate
static IntervalTimer stepTimer;

static volatile uint32_t phase[BB8_MAX_MOTORS];
static volatile uint32_t phaseInc[BB8_MAX_MOTORS];   // rate * 2^32 / TICK_HZ (unsigned magnitude)
static volatile int8_t dirSign[BB8_MAX_MOTORS];
static volatile int32_t stepCount[BB8_MAX_MOTORS];
static volatile bool pulseHigh[BB8_MAX_MOTORS];
static volatile int32_t encCount[BB8_MAX_MOTORS];
static volatile uint8_t encState[BB8_MAX_MOTORS];

static float targetRate[BB8_MAX_MOTORS];          // steps/s (signed)
static float currentRate[BB8_MAX_MOTORS];
static float maxAccel = 150000.0f;                // steps/s^2
static uint32_t watchdogUs = 100000;
static uint32_t lastCmdUs = 0;
static bool haveCmd = false, enableReq = false, enabled = false, watchdogTripped = false, configured = false;
static uint16_t seqEcho = 0;
static bb8_parser_t parser;

// ----------------------------------------------------------------------------- ISRs
static void stepIsr() {
  for (uint8_t i = 0; i < N_MOTORS; i++) {
    if (pulseHigh[i]) { digitalWriteFast(STEP_PIN[i], LOW); pulseHigh[i] = false; continue; }  // >= 20 us pulse
    uint32_t before = phase[i];
    phase[i] = before + phaseInc[i];
    if (phase[i] < before) {                       // accumulator wrapped: one step
      digitalWriteFast(STEP_PIN[i], HIGH);
      pulseHigh[i] = true;
      stepCount[i] += dirSign[i];
    }
  }
}

static const int8_t QDEC[16] = {0, -1, 1, 0, 1, 0, 0, -1, -1, 0, 0, 1, 0, 1, -1, 0};
static inline void encUpdate(uint8_t i) {
  uint8_t s = (uint8_t)((digitalReadFast(ENC_A_PIN[i]) << 1) | digitalReadFast(ENC_B_PIN[i]));
  encCount[i] += QDEC[(encState[i] << 2) | s];
  encState[i] = s;
}
static void enc0() { encUpdate(0); }
static void enc1() { encUpdate(1); }
static void enc2() { encUpdate(2); }
static void enc3() { encUpdate(3); }
static void (*const ENC_ISR[BB8_MAX_MOTORS])() = {enc0, enc1, enc2, enc3};

// ----------------------------------------------------------------------------- helpers
static void applyRate(uint8_t i, float rate) {
  int8_t d = rate >= 0 ? 1 : -1;
  float mag = fabsf(rate);
  if (mag > TICK_HZ / 2) mag = TICK_HZ / 2;        // two ISR ticks per step (pulse high + low)
  uint32_t inc = (uint32_t)((double)mag * 4294967296.0 / TICK_HZ);
  if (d != dirSign[i]) digitalWriteFast(DIR_PIN[i], d > 0 ? HIGH : LOW);   // DIR set >= 20 us before the next step
  noInterrupts();
  dirSign[i] = d;
  phaseInc[i] = inc;
  interrupts();
}

static void setDrivers(bool on) {
  enabled = on;
  digitalWriteFast(ENABLE_PIN, on ? LOW : HIGH);
}

static void sendTelem() {
  int32_t enc[BB8_MAX_MOTORS], steps[BB8_MAX_MOTORS];
  noInterrupts();
  for (uint8_t i = 0; i < N_MOTORS; i++) { enc[i] = encCount[i]; steps[i] = stepCount[i]; }
  interrupts();
  uint8_t flags = (enabled ? BB8_TF_ENABLED : 0) | (watchdogTripped ? BB8_TF_WATCHDOG : 0) |
                  (configured ? BB8_TF_CONFIGURED : 0);
  uint8_t out[6 + 8 + 8 * BB8_MAX_MOTORS];
  size_t n = bb8_encode_telem(seqEcho, micros(), flags, N_MOTORS, USE_ENCODERS ? enc : steps, steps, out);
  Serial.write(out, n);
}

static void handleFrame() {
  if (parser.type == BB8_T_CMD) {
    bb8_cmd_t c;
    if (!bb8_decode_cmd(parser.payload, parser.len, &c)) return;
    for (uint8_t i = 0; i < N_MOTORS; i++) targetRate[i] = i < c.n ? c.rate[i] : 0.0f;
    enableReq = c.enable;
    seqEcho = c.seq;
    lastCmdUs = micros();
    haveCmd = true;
    watchdogTripped = false;
    sendTelem();
  } else if (parser.type == BB8_T_CFG) {
    bb8_cfg_t c;
    if (!bb8_decode_cfg(parser.payload, parser.len, &c)) return;
    if (c.max_accel > 0) maxAccel = c.max_accel;
    watchdogUs = (uint32_t)c.watchdog_ms * 1000UL;
    configured = true;
  }
}

// ----------------------------------------------------------------------------- setup / loop
void setup() {
  Serial.begin(1000000);                           // USB: baud is ignored
  pinMode(ENABLE_PIN, OUTPUT);
  setDrivers(false);
  for (uint8_t i = 0; i < N_MOTORS; i++) {
    pinMode(STEP_PIN[i], OUTPUT);
    pinMode(DIR_PIN[i], OUTPUT);
    digitalWriteFast(STEP_PIN[i], LOW);
    digitalWriteFast(DIR_PIN[i], HIGH);
    dirSign[i] = 1;
    if (USE_ENCODERS) {
      pinMode(ENC_A_PIN[i], INPUT_PULLUP);
      pinMode(ENC_B_PIN[i], INPUT_PULLUP);
      encState[i] = (uint8_t)((digitalReadFast(ENC_A_PIN[i]) << 1) | digitalReadFast(ENC_B_PIN[i]));
      attachInterrupt(digitalPinToInterrupt(ENC_A_PIN[i]), ENC_ISR[i], CHANGE);
      attachInterrupt(digitalPinToInterrupt(ENC_B_PIN[i]), ENC_ISR[i], CHANGE);
    }
  }
  bb8_parser_init(&parser);
  stepTimer.begin(stepIsr, 1000000.0f / TICK_HZ);
  stepTimer.priority(32);
}

void loop() {
  static uint32_t lastUpdate = micros(), lastTelem = 0;
  while (Serial.available()) {
    if (bb8_parser_feed(&parser, (uint8_t)Serial.read())) handleFrame();
  }
  uint32_t now = micros();
  if (now - lastUpdate < 1000) return;             // 1 kHz rate / enable update
  float dt = (now - lastUpdate) * 1e-6f;
  lastUpdate = now;

  if (haveCmd && now - lastCmdUs > watchdogUs) {   // host went silent
    watchdogTripped = true;
    enableReq = false;
    for (uint8_t i = 0; i < N_MOTORS; i++) targetRate[i] = 0.0f;
  }
  if (enableReq && !enabled) setDrivers(true);
  bool allZero = true;
  float step = maxAccel * dt;
  for (uint8_t i = 0; i < N_MOTORS; i++) {
    float d = targetRate[i] - currentRate[i];
    currentRate[i] += d > step ? step : (d < -step ? -step : d);
    if (!enabled) currentRate[i] = 0.0f;
    if (currentRate[i] != 0.0f) allZero = false;
    applyRate(i, currentRate[i]);
  }
  if (!enableReq && enabled && allZero) setDrivers(false);
  if (now - lastTelem > 20000) { lastTelem = now; sendTelem(); }
}

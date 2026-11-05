// BB-8 host <-> microcontroller serial protocol (C/C++, header only).
// Must stay identical to deploy/runtime/protocol.py.
//
// Frame: 0xAA 0x55 | type u8 | len u8 | payload[len] | crc16 u16 (little endian)
// CRC-16/CCITT-FALSE (poly 0x1021, init 0xFFFF) over type, len, payload.
//   0x01 CMD   host->mcu  u16 seq, u8 flags (bit0 enable), u8 n, f32 step_rate[n]   (steps/s)
//   0x02 CFG   host->mcu  f32 max_accel (steps/s^2), u16 watchdog_ms, u8 n
//   0x81 TELEM mcu->host  u16 seq_echo, u32 mcu_us, u8 flags, u8 n, i32 encoder[n], i32 steps[n]
#ifndef BB8_PROTOCOL_H
#define BB8_PROTOCOL_H

#include <stdint.h>
#include <string.h>

#define BB8_SYNC0 0xAA
#define BB8_SYNC1 0x55
#define BB8_T_CMD 0x01
#define BB8_T_CFG 0x02
#define BB8_T_TELEM 0x81
#define BB8_MAX_MOTORS 4
#define BB8_MAX_PAYLOAD 255
#define BB8_FLAG_ENABLE 0x01
#define BB8_TF_ENABLED 0x01
#define BB8_TF_WATCHDOG 0x02
#define BB8_TF_CONFIGURED 0x04

static inline uint16_t bb8_crc16_update(uint16_t crc, uint8_t byte) {
  crc ^= (uint16_t)byte << 8;
  for (int i = 0; i < 8; i++) crc = (crc & 0x8000) ? (uint16_t)((crc << 1) ^ 0x1021) : (uint16_t)(crc << 1);
  return crc;
}

static inline uint16_t bb8_crc16(const uint8_t* data, size_t len) {
  uint16_t crc = 0xFFFF;
  for (size_t i = 0; i < len; i++) crc = bb8_crc16_update(crc, data[i]);
  return crc;
}

// little-endian helpers (independent of the host byte order)
static inline void bb8_put_u16(uint8_t* p, uint16_t v) { p[0] = v & 0xFF; p[1] = v >> 8; }
static inline void bb8_put_u32(uint8_t* p, uint32_t v) { for (int i = 0; i < 4; i++) p[i] = (v >> (8 * i)) & 0xFF; }
static inline uint16_t bb8_get_u16(const uint8_t* p) { return (uint16_t)(p[0] | (p[1] << 8)); }
static inline uint32_t bb8_get_u32(const uint8_t* p) {
  return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}
static inline float bb8_get_f32(const uint8_t* p) { uint32_t u = bb8_get_u32(p); float f; memcpy(&f, &u, 4); return f; }
static inline void bb8_put_f32(uint8_t* p, float f) { uint32_t u; memcpy(&u, &f, 4); bb8_put_u32(p, u); }

// Wraps a payload into a frame; returns the frame length (out needs len + 6 bytes).
static inline size_t bb8_frame(uint8_t type, const uint8_t* payload, uint8_t len, uint8_t* out) {
  out[0] = BB8_SYNC0; out[1] = BB8_SYNC1; out[2] = type; out[3] = len;
  memcpy(out + 4, payload, len);
  uint16_t crc = bb8_crc16(out + 2, (size_t)len + 2);
  bb8_put_u16(out + 4 + len, crc);
  return (size_t)len + 6;
}

typedef struct { uint16_t seq; uint8_t enable; uint8_t n; float rate[BB8_MAX_MOTORS]; } bb8_cmd_t;
typedef struct { float max_accel; uint16_t watchdog_ms; uint8_t n; } bb8_cfg_t;

static inline int bb8_decode_cmd(const uint8_t* p, uint8_t len, bb8_cmd_t* c) {
  if (len < 4) return 0;
  c->seq = bb8_get_u16(p); c->enable = p[2] & BB8_FLAG_ENABLE; c->n = p[3];
  if (c->n < 1 || c->n > BB8_MAX_MOTORS || len != 4 + 4 * c->n) return 0;
  for (int i = 0; i < c->n; i++) c->rate[i] = bb8_get_f32(p + 4 + 4 * i);
  return 1;
}

static inline int bb8_decode_cfg(const uint8_t* p, uint8_t len, bb8_cfg_t* c) {
  if (len != 7) return 0;
  c->max_accel = bb8_get_f32(p); c->watchdog_ms = bb8_get_u16(p + 4); c->n = p[6];
  return 1;
}

// Builds a TELEM frame into out (needs 6 + 8 + 8n bytes); returns the frame length.
static inline size_t bb8_encode_telem(uint16_t seq, uint32_t us, uint8_t flags, uint8_t n, const int32_t* enc,
                                      const int32_t* steps, uint8_t* out) {
  uint8_t p[8 + 8 * BB8_MAX_MOTORS];
  bb8_put_u16(p, seq); bb8_put_u32(p + 2, us); p[6] = flags; p[7] = n;
  for (int i = 0; i < n; i++) bb8_put_u32(p + 8 + 4 * i, (uint32_t)enc[i]);
  for (int i = 0; i < n; i++) bb8_put_u32(p + 8 + 4 * n + 4 * i, (uint32_t)steps[i]);
  return bb8_frame(BB8_T_TELEM, p, (uint8_t)(8 + 8 * n), out);
}

// Incremental parser: feed bytes, returns 1 when a valid frame is in (type, payload, len).
typedef struct {
  uint8_t state, type, len, idx;
  uint8_t payload[BB8_MAX_PAYLOAD];
  uint16_t crc_rx;
  uint32_t crc_errors;
} bb8_parser_t;

static inline void bb8_parser_init(bb8_parser_t* ps) { memset(ps, 0, sizeof(*ps)); }

static inline int bb8_parser_feed(bb8_parser_t* ps, uint8_t b) {
  switch (ps->state) {
    case 0: ps->state = (b == BB8_SYNC0) ? 1 : 0; return 0;
    case 1: ps->state = (b == BB8_SYNC1) ? 2 : (b == BB8_SYNC0 ? 1 : 0); return 0;
    case 2: ps->type = b; ps->state = 3; return 0;
    case 3: ps->len = b; ps->idx = 0; ps->state = (b == 0) ? 5 : 4; return 0;
    case 4: ps->payload[ps->idx++] = b; if (ps->idx >= ps->len) ps->state = 5; return 0;
    case 5: ps->crc_rx = b; ps->state = 6; return 0;
    case 6: {
      ps->crc_rx |= (uint16_t)b << 8;
      ps->state = 0;
      uint16_t crc = bb8_crc16_update(bb8_crc16_update(0xFFFF, ps->type), ps->len);
      for (int i = 0; i < ps->len; i++) crc = bb8_crc16_update(crc, ps->payload[i]);
      if (crc == ps->crc_rx) return 1;
      ps->crc_errors++;
      return 0;
    }
  }
  ps->state = 0;
  return 0;
}

#endif  // BB8_PROTOCOL_H

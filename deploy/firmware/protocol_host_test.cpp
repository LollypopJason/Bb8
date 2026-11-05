// Host-side test of bb8_protocol.h (built by tests/test_deploy.py with g++).
// stdin: hex lines "CMD <hex frame>" / "CFG <hex frame>" -> decodes and prints values;
// "TELEM seq us flags n e0..en s0..sn" -> prints the encoded frame as hex.
#include <cstdio>
#include <cstring>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include "bb8_mcu/bb8_protocol.h"

static std::vector<uint8_t> unhex(const std::string& s) {
  std::vector<uint8_t> out;
  for (size_t i = 0; i + 1 < s.size(); i += 2) out.push_back((uint8_t)std::stoi(s.substr(i, 2), nullptr, 16));
  return out;
}

int main() {
  std::string line;
  while (std::getline(std::cin, line)) {
    std::istringstream in(line);
    std::string kind;
    in >> kind;
    if (kind == "FRAME") {
      std::string hex;
      in >> hex;
      bb8_parser_t ps;
      bb8_parser_init(&ps);
      int got = 0;
      for (uint8_t b : unhex(hex)) {
        if (!bb8_parser_feed(&ps, b)) continue;
        got++;
        if (ps.type == BB8_T_CMD) {
          bb8_cmd_t c;
          if (!bb8_decode_cmd(ps.payload, ps.len, &c)) { printf("BADCMD\n"); continue; }
          printf("CMD %u %u %u", c.seq, c.enable, c.n);
          for (int i = 0; i < c.n; i++) printf(" %.4f", c.rate[i]);
          printf("\n");
        } else if (ps.type == BB8_T_CFG) {
          bb8_cfg_t c;
          if (!bb8_decode_cfg(ps.payload, ps.len, &c)) { printf("BADCFG\n"); continue; }
          printf("CFG %.1f %u %u\n", c.max_accel, c.watchdog_ms, c.n);
        }
      }
      printf("FRAMES %d CRCERR %u\n", got, ps.crc_errors);
    } else if (kind == "IMU") {
      unsigned seq, n; unsigned long us; float g[3], a[3]; int32_t enc[4];
      in >> seq >> us >> g[0] >> g[1] >> g[2] >> a[0] >> a[1] >> a[2] >> n;
      for (unsigned i = 0; i < n; i++) in >> enc[i];
      uint8_t out[64];
      size_t len = bb8_encode_imu((uint16_t)seq, (uint32_t)us, g, a, (uint8_t)n, enc, out);
      for (size_t i = 0; i < len; i++) printf("%02x", out[i]);
      printf("\n");
    } else if (kind == "TELEM") {
      unsigned seq, flags, n; unsigned long us;
      in >> seq >> us >> flags >> n;
      int32_t enc[4], steps[4];
      for (unsigned i = 0; i < n; i++) in >> enc[i];
      for (unsigned i = 0; i < n; i++) in >> steps[i];
      uint8_t out[64];
      size_t len = bb8_encode_telem((uint16_t)seq, (uint32_t)us, (uint8_t)flags, (uint8_t)n, enc, steps, out);
      for (size_t i = 0; i < len; i++) printf("%02x", out[i]);
      printf("\n");
    }
    fflush(stdout);
  }
  return 0;
}

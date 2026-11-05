// bb8_stepgen - real-time STEP/DIR pulse generator for stepper drivers wired to Jetson Nano GPIO.
//
// Speaks the bb8_protocol.h CMD / CFG / TELEM frames on a Unix stream socket, exactly like the
// Teensy firmware does over USB, so the Python runtime (deploy/runtime) is unchanged:
//   CMD   step rates [steps/s] + enable  ->  acceleration-limited rates, step pulses, DIR, ENABLE
//   TELEM step counters (also reported in the encoder field), flags, microsecond clock
// Safety: without a CMD for watchdog_ms, or when the client disconnects, the motors ramp to zero
// and the drivers are disabled. SIGINT / SIGTERM disable the drivers before exiting.
//
// Timing: one thread, SCHED_FIFO, pinned to one CPU core, busy-polling CLOCK_MONOTONIC while any
// motor turns. Pulse edges come from sysfs GPIO writes (pre-opened value files, ~2-5 us each on the
// Nano). Use the default 1/8 microstepping or coarser: 18.85 rad/s is then 4800 steps/s per motor.
//
// Build:  make            (on the Jetson; no dependencies)
// Run:    sudo ./bb8_stepgen --step 216,79,232 --dir 50,194,15 --enable 13
//         ./bb8_stepgen --sim  (no GPIO, for testing on a PC)
#define _GNU_SOURCE
#include <errno.h>
#include <fcntl.h>
#include <math.h>
#include <sched.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <time.h>
#include <unistd.h>

#include "bb8_protocol.h"

#define MAXM BB8_MAX_MOTORS

typedef struct {
  int sim;
  const char* gpio_root;
  int step_gpio[MAXM], dir_gpio[MAXM], n;
  int enable_gpio, enable_active_low;
  int step_fd[MAXM], dir_fd[MAXM], enable_fd;
} gpio_t;

static volatile sig_atomic_t g_stop = 0;
static void on_signal(int sig) { (void)sig; g_stop = 1; }

static int64_t now_ns(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (int64_t)ts.tv_sec * 1000000000LL + ts.tv_nsec;
}

static int write_file(const char* path, const char* text) {
  int fd = open(path, O_WRONLY);
  if (fd < 0) return -1;
  ssize_t n = write(fd, text, strlen(text));
  close(fd);
  return n == (ssize_t)strlen(text) ? 0 : -1;
}

static int gpio_open(gpio_t* g, int num) {
  if (g->sim) return -1;
  char path[256], buf[16];
  snprintf(path, sizeof path, "%s/gpio%d/value", g->gpio_root, num);
  if (access(path, F_OK) != 0) {
    snprintf(path, sizeof path, "%s/export", g->gpio_root);
    snprintf(buf, sizeof buf, "%d", num);
    write_file(path, buf);                        // EBUSY if already exported: fine
  }
  snprintf(path, sizeof path, "%s/gpio%d/direction", g->gpio_root, num);
  for (int i = 0; i < 100 && write_file(path, "out") != 0; i++) usleep(10000);   // wait for udev
  snprintf(path, sizeof path, "%s/gpio%d/value", g->gpio_root, num);
  int fd = open(path, O_WRONLY);
  if (fd < 0) { fprintf(stderr, "bb8_stepgen: cannot open %s: %s\n", path, strerror(errno)); exit(2); }
  return fd;
}

static inline void gpio_set(int fd, int v) {
  if (fd >= 0) { ssize_t r = pwrite(fd, v ? "1" : "0", 1, 0); (void)r; }
}

static void gpio_setup(gpio_t* g) {
  for (int i = 0; i < g->n; i++) {
    g->step_fd[i] = gpio_open(g, g->step_gpio[i]);
    g->dir_fd[i] = gpio_open(g, g->dir_gpio[i]);
    gpio_set(g->step_fd[i], 0);
    gpio_set(g->dir_fd[i], 1);
  }
  g->enable_fd = g->enable_gpio >= 0 ? gpio_open(g, g->enable_gpio) : -1;
}

static void drivers(gpio_t* g, int on) { gpio_set(g->enable_fd, g->enable_active_low ? !on : on); }

static int parse_list(const char* s, int* out) {
  int n = 0;
  while (*s && n < MAXM) {
    out[n++] = atoi(s);
    const char* c = strchr(s, ',');
    if (!c) break;
    s = c + 1;
  }
  return n;
}

typedef struct {
  double target[MAXM], rate[MAXM];
  int64_t period_ns[MAXM], next_step[MAXM], low_at[MAXM];
  int dir[MAXM], high[MAXM];
  int32_t steps[MAXM];
  double max_accel;
  int64_t watchdog_ns, last_cmd;
  int have_cmd, enable_req, enabled, watchdog, configured;
  uint16_t seq_echo;
} motors_t;

static void send_telem(int fd, const motors_t* m, int n, int64_t t0) {
  if (fd < 0) return;
  uint8_t flags = (m->enabled ? BB8_TF_ENABLED : 0) | (m->watchdog ? BB8_TF_WATCHDOG : 0) |
                  (m->configured ? BB8_TF_CONFIGURED : 0);
  uint8_t out[6 + 8 + 8 * MAXM];
  uint32_t us = (uint32_t)((now_ns() - t0) / 1000);
  size_t len = bb8_encode_telem(m->seq_echo, us, flags, (uint8_t)n, m->steps, m->steps, out);
  ssize_t r = send(fd, out, len, MSG_DONTWAIT | MSG_NOSIGNAL);
  (void)r;
}

static void apply_rate(gpio_t* g, motors_t* m, int i, double rate, int64_t now, int64_t dir_setup_ns) {
  int d = rate >= 0 ? 1 : -1;
  if (d != m->dir[i]) {
    gpio_set(g->dir_fd[i], d > 0);
    m->dir[i] = d;
    if (m->next_step[i] < now + dir_setup_ns) m->next_step[i] = now + dir_setup_ns;
  }
  double mag = fabs(rate);
  int64_t old = m->period_ns[i];
  m->period_ns[i] = mag < 0.5 ? 0 : (int64_t)(1e9 / mag);
  if (m->period_ns[i] && !old) m->next_step[i] = now + dir_setup_ns;                 // start
  else if (m->period_ns[i] && old) m->next_step[i] += m->period_ns[i] - old;          // keep the phase
}

int main(int argc, char** argv) {
  gpio_t g;
  memset(&g, 0, sizeof g);
  g.gpio_root = "/sys/class/gpio";
  g.n = 3;
  int defaults_step[3] = {216, 79, 232}, defaults_dir[3] = {50, 194, 15};   // header pins 7,12,16 / 11,15,18
  memcpy(g.step_gpio, defaults_step, sizeof defaults_step);
  memcpy(g.dir_gpio, defaults_dir, sizeof defaults_dir);
  g.enable_gpio = 13;                                                        // header pin 22
  g.enable_active_low = 1;
  const char* sock_path = "/tmp/bb8_stepgen.sock";
  int pulse_us = 3, dir_setup_us = 5, cpu = 3, prio = 80;
  for (int i = 1; i < argc; i++) {
    const char* a = argv[i];
    const char* v = i + 1 < argc ? argv[i + 1] : "";
    if (!strcmp(a, "--sim")) g.sim = 1;
    else if (!strcmp(a, "--socket")) { sock_path = v; i++; }
    else if (!strcmp(a, "--gpio-root")) { g.gpio_root = v; i++; }
    else if (!strcmp(a, "--step")) { g.n = parse_list(v, g.step_gpio); i++; }
    else if (!strcmp(a, "--dir")) { parse_list(v, g.dir_gpio); i++; }
    else if (!strcmp(a, "--enable")) { g.enable_gpio = atoi(v); i++; }
    else if (!strcmp(a, "--enable-active-high")) g.enable_active_low = 0;
    else if (!strcmp(a, "--pulse-us")) { pulse_us = atoi(v); i++; }
    else if (!strcmp(a, "--dir-setup-us")) { dir_setup_us = atoi(v); i++; }
    else if (!strcmp(a, "--cpu")) { cpu = atoi(v); i++; }
    else if (!strcmp(a, "--rt-prio")) { prio = atoi(v); i++; }
    else { fprintf(stderr, "usage: %s [--sim] [--socket P] [--gpio-root D] [--step a,b,c] [--dir a,b,c] "
                           "[--enable n|-1] [--enable-active-high] [--pulse-us 3] [--dir-setup-us 5] [--cpu 3] "
                           "[--rt-prio 80]\n", argv[0]); return 1; }
  }
  if (g.n < 1 || g.n > MAXM) { fprintf(stderr, "bb8_stepgen: 1..4 motors\n"); return 1; }

  signal(SIGINT, on_signal);
  signal(SIGTERM, on_signal);
  signal(SIGPIPE, SIG_IGN);
  if (!g.sim) {
    if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) perror("bb8_stepgen: mlockall (continuing)");
    struct sched_param sp = {.sched_priority = prio};
    if (sched_setscheduler(0, SCHED_FIFO, &sp) != 0) perror("bb8_stepgen: SCHED_FIFO needs root (continuing)");
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(cpu, &set);
    if (sched_setaffinity(0, sizeof set, &set) != 0) perror("bb8_stepgen: cpu affinity (continuing)");
  }
  gpio_setup(&g);
  drivers(&g, 0);

  int lsock = socket(AF_UNIX, SOCK_STREAM | SOCK_NONBLOCK, 0);
  struct sockaddr_un addr;
  memset(&addr, 0, sizeof addr);
  addr.sun_family = AF_UNIX;
  strncpy(addr.sun_path, sock_path, sizeof addr.sun_path - 1);
  unlink(sock_path);
  if (lsock < 0 || bind(lsock, (struct sockaddr*)&addr, sizeof addr) != 0 || listen(lsock, 1) != 0) {
    perror("bb8_stepgen: socket");
    return 2;
  }
  chmod(sock_path, 0666);
  fprintf(stderr, "bb8_stepgen: %d motors, %s, socket %s\n", g.n, g.sim ? "SIM (no GPIO)" : g.gpio_root, sock_path);

  motors_t m;
  memset(&m, 0, sizeof m);
  for (int i = 0; i < g.n; i++) m.dir[i] = 1;
  m.max_accel = 150000.0;
  m.watchdog_ns = 100000000LL;
  bb8_parser_t ps;
  bb8_parser_init(&ps);
  int client = -1;
  const int64_t t0 = now_ns(), pulse_ns = (int64_t)pulse_us * 1000, dir_setup_ns = (int64_t)dir_setup_us * 1000;
  int64_t next_ctrl = t0, last_ctrl = t0, last_telem = t0;

  while (!g_stop) {
    int64_t now = now_ns();
    // --------------------------------------------------------------- step pulses
    for (int i = 0; i < g.n; i++) {
      if (m.high[i] && now >= m.low_at[i]) { gpio_set(g.step_fd[i], 0); m.high[i] = 0; }
      if (m.period_ns[i] && !m.high[i] && now >= m.next_step[i]) {
        gpio_set(g.step_fd[i], 1);
        m.high[i] = 1;
        m.low_at[i] = now + pulse_ns;
        m.steps[i] += m.dir[i];
        m.next_step[i] += m.period_ns[i];
        if (m.next_step[i] < now) m.next_step[i] = now + m.period_ns[i];   // fell behind: no step bursts
      }
    }
    if (now < next_ctrl) {
      int busy = 0;
      for (int i = 0; i < g.n; i++) busy |= m.period_ns[i] != 0 || m.high[i];
      if (!busy) usleep(100);                      // idle: do not burn the core
      continue;
    }
    // --------------------------------------------------------------- 1 kHz: socket, watchdog, ramps
    next_ctrl = now + 1000000;
    double dt = (now - last_ctrl) * 1e-9;
    last_ctrl = now;
    if (client < 0) {
      client = accept4(lsock, NULL, NULL, SOCK_NONBLOCK);
      if (client >= 0) bb8_parser_init(&ps);
    }
    if (client >= 0) {
      uint8_t buf[512];
      ssize_t r = recv(client, buf, sizeof buf, MSG_DONTWAIT);
      if (r == 0 || (r < 0 && errno != EAGAIN && errno != EWOULDBLOCK)) {    // client gone: stop safely
        close(client);
        client = -1;
        m.enable_req = 0;
        for (int i = 0; i < g.n; i++) m.target[i] = 0.0;
      }
      for (ssize_t k = 0; k < r; k++) {
        if (!bb8_parser_feed(&ps, buf[k])) continue;
        if (ps.type == BB8_T_CMD) {
          bb8_cmd_t c;
          if (!bb8_decode_cmd(ps.payload, ps.len, &c)) continue;
          for (int i = 0; i < g.n; i++) m.target[i] = i < c.n ? c.rate[i] : 0.0;
          m.enable_req = c.enable;
          m.seq_echo = c.seq;
          m.last_cmd = now;
          m.have_cmd = 1;
          m.watchdog = 0;
          send_telem(client, &m, g.n, t0);
          last_telem = now;
        } else if (ps.type == BB8_T_CFG) {
          bb8_cfg_t c;
          if (!bb8_decode_cfg(ps.payload, ps.len, &c)) continue;
          if (c.max_accel > 0) m.max_accel = c.max_accel;
          m.watchdog_ns = (int64_t)c.watchdog_ms * 1000000LL;
          m.configured = 1;
        }
      }
    }
    if (m.have_cmd && now - m.last_cmd > m.watchdog_ns) {
      m.watchdog = 1;
      m.enable_req = 0;
      for (int i = 0; i < g.n; i++) m.target[i] = 0.0;
    }
    if (m.enable_req && !m.enabled) { drivers(&g, 1); m.enabled = 1; }
    int all_zero = 1;
    double step = m.max_accel * dt;
    for (int i = 0; i < g.n; i++) {
      double d = m.target[i] - m.rate[i];
      m.rate[i] += d > step ? step : (d < -step ? -step : d);
      if (!m.enabled) m.rate[i] = 0.0;
      if (m.rate[i] != 0.0) all_zero = 0;
      apply_rate(&g, &m, i, m.rate[i], now, dir_setup_ns);
    }
    if (!m.enable_req && m.enabled && all_zero) { drivers(&g, 0); m.enabled = 0; }
    if (now - last_telem > 20000000LL) { send_telem(client, &m, g.n, t0); last_telem = now; }
  }
  for (int i = 0; i < g.n; i++) gpio_set(g.step_fd[i], 0);
  drivers(&g, 0);
  if (client >= 0) close(client);
  close(lsock);
  unlink(sock_path);
  fprintf(stderr, "bb8_stepgen: stopped, drivers disabled\n");
  return 0;
}

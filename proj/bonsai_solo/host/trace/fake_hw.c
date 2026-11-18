/* Emulated Bonsai accelerator hardware for the x86 trace harness.

   Physical regions (scratch, K/V cache, model, GPIO) live in anonymous
   memory. A thread emulates the GPIO toggle/ack protocol and DONE edges and
   appends every register write to $FAKE_HW_TRACE. At exit the scratch region
   is written to $FAKE_HW_SCRATCH so the solo build can preload it. Results
   (argmax tokens, residual sums) are not computed. */
#define _GNU_SOURCE
#include <errno.h>
#include <pthread.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>

#define FAKE_MEM_FD 1000
#define GPIO_PHYS 0x40000000ull
#define DDR_PHYS 0x80000000ull
#define DDR_BYTES 0x40000000ull
#define SCRATCH_PHYS 0xbe000000ull
#define SCRATCH_BYTES (32ull << 20)

#define STATUS_PRESENT 0x80000000u
#define STATUS_DONE 0x10000000u
#define STATUS_ACK 0x08000000u
#define STATUS_PACKED_GENERATION 0x04000000u
#define STATUS_SEQUENCE_IDLE 0x02d44000u

static volatile uint32_t *gpio;
static uint8_t *ddr;
static FILE *trace;
static pthread_mutex_t trace_lock = PTHREAD_MUTEX_INITIALIZER;

static void *real_mmap(void *a, size_t l, int p, int f, int fd, off_t o) {
  return (void *)syscall(SYS_mmap, a, l, p, f, fd, o);
}

static void ensure_regions(void) {
  if (gpio != NULL) return;
  gpio = real_mmap(NULL, 4096, PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  ddr = real_mmap(NULL, DDR_BYTES, PROT_READ | PROT_WRITE,
                  MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
  if (gpio == MAP_FAILED || ddr == MAP_FAILED) abort();
  gpio[2] = STATUS_PRESENT | STATUS_PACKED_GENERATION | STATUS_SEQUENCE_IDLE;
}

static void log_line(const char *format, ...) {
  if (trace == NULL) return;
  pthread_mutex_lock(&trace_lock);
  va_list args;
  va_start(args, format);
  vfprintf(trace, format, args);
  va_end(args);
  fflush(trace);
  pthread_mutex_unlock(&trace_lock);
}

static void *emulator(void *unused) {
  (void)unused;
  uint32_t last = __atomic_load_n(&gpio[0], __ATOMIC_ACQUIRE);
  for (;;) {
    uint32_t value = __atomic_load_n(&gpio[0], __ATOMIC_ACQUIRE);
    if (value == last) {
      __builtin_ia32_pause();
      continue;
    }
    last = value;
    uint32_t status = gpio[2];
    if ((value & 0x20000000u) != 0u) {
      uint32_t index = (value >> 25) & 7u;
      uint32_t data = value & 0x01ffffffu;
      log_line("W %u 0x%07x\n", index, data);
      if (index == 5u) {
        log_line("START %u\n", data);
        status ^= STATUS_DONE;
      }
    }
    status = (status & ~STATUS_ACK) | (value & 0x80000000u ? STATUS_ACK : 0u);
    __atomic_store_n(&gpio[2], status, __ATOMIC_RELEASE);
  }
  return NULL;
}

static void dump_scratch(void) {
  const char *path = getenv("FAKE_HW_SCRATCH");
  if (path == NULL || ddr == NULL) return;
  FILE *out = fopen(path, "wb");
  if (out == NULL) return;
  fwrite(ddr + (SCRATCH_PHYS - DDR_PHYS), 1, SCRATCH_BYTES, out);
  fclose(out);
  const char *model_path = getenv("FAKE_HW_MODEL_OUT");
  const char *model_size = getenv("FAKE_HW_MODEL_BYTES");
  if (model_path == NULL || model_size == NULL) return;
  out = fopen(model_path, "wb");
  if (out == NULL) return;
  fwrite(ddr + (0xaf000000ull - DDR_PHYS), 1, strtoull(model_size, NULL, 0),
         out);
  fclose(out);
}

__attribute__((constructor)) static void fake_hw_start(void) {
  ensure_regions();
  const char *path = getenv("FAKE_HW_TRACE");
  if (path != NULL) trace = fopen(path, "w");
  const char *model = getenv("FAKE_HW_MODEL");
  if (model != NULL) {
    FILE *in = fopen(model, "rb");
    if (in == NULL) abort();
    size_t total = fread(ddr + (0xaf000000ull - DDR_PHYS), 1, 1ull << 28, in);
    fclose(in);
    fprintf(stderr, "fake_hw: preloaded %zu model bytes\n", total);
  }
  pthread_t thread;
  pthread_create(&thread, NULL, emulator, NULL);
  pthread_detach(thread);
  atexit(dump_scratch);
}

void fake_hw_marker(const char *name) { log_line("MARK %s\n", name); }

int fake_open(const char *path, int flags, ...) {
  if (strcmp(path, "/dev/mem") == 0) return FAKE_MEM_FD;
  char redirected[4096];
  const char *run_dir = getenv("FAKE_HW_RUN_DIR");
  if (run_dir != NULL && strncmp(path, "/run/", 5) == 0) {
    snprintf(redirected, sizeof(redirected), "%s/%s", run_dir, path + 5);
    path = redirected;
  }
  mode_t mode = 0;
  if ((flags & O_CREAT) != 0) {
    va_list args;
    va_start(args, flags);
    mode = (mode_t)va_arg(args, int);
    va_end(args);
  }
  return (int)syscall(SYS_openat, AT_FDCWD, path, flags, mode);
}

void *fake_mmap(void *address, size_t length, int protection, int flags,
                int fd, off_t offset) {
  if (fd != FAKE_MEM_FD)
    return real_mmap(address, length, protection, flags, fd, offset);
  ensure_regions();
  uint64_t physical = (uint64_t)offset;
  if (physical == GPIO_PHYS) return (void *)gpio;
  if (physical >= DDR_PHYS && physical + length <= DDR_PHYS + DDR_BYTES)
    return ddr + (physical - DDR_PHYS);
  errno = EINVAL;
  return MAP_FAILED;
}

int fake_munmap(void *address, size_t length) {
  uint8_t *byte = address;
  if ((void *)byte == (void *)gpio ||
      (byte >= ddr && byte < ddr + DDR_BYTES))
    return 0;
  return (int)syscall(SYS_munmap, address, length);
}

#define _GNU_SOURCE

#include <errno.h>
#include <fcntl.h>
#include <sched.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdatomic.h>
#include <string.h>
#include <sys/file.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#include "bonsai_model_manifest.h"

#define BONSAI_MODEL_PHYS UINT64_C(0xaf000000)
#define BONSAI_GPIO_PHYS UINT64_C(0x40000000)
#define BONSAI_SCRATCH_PHYS UINT64_C(0xbe000000)
#define BONSAI_SCRATCH_BYTES (2u * 1024u * 1024u)
#define BONSAI_INPUT_OFFSET 0x0f000000u
#define BONSAI_OUTPUT_OFFSET 0x0f010000u
#define BONSAI_OUTPUT_LOCAL 0x00010000u
#define BONSAI_STATUS_PRESENT 0x80000000u
#define BONSAI_STATUS_ERROR 0x20000000u
#define BONSAI_STATUS_DONE 0x10000000u

struct block_q8_0 {
  uint16_t d;
  int8_t qs[32];
};

_Static_assert(sizeof(struct block_q8_0) == 34u,
               "llama.cpp Q8_0 layout changed");

struct model_stage_stamp {
  uint64_t device;
  uint64_t inode;
  uint64_t size;
  int64_t mtime_seconds;
  int64_t mtime_nanoseconds;
};

static atomic_flag accelerator_lock = ATOMIC_FLAG_INIT;
static volatile uint32_t *accelerator_gpio;
static volatile uint8_t *accelerator_scratch;
static uint32_t accelerator_toggle;
static int accelerator_initialized;
static int accelerator_available;
static const struct bonsai_tensor_spec *cached_tensor;
static uintptr_t cached_tensor_address;
static uintptr_t cached_row_address;
static uint32_t cached_row;
static const void *cached_input;
static uint64_t cached_signature[3];
static uintptr_t model_mapping_base;
static int model_mapping_base_valid;
static int whole_matrix_reported;
static int hardware_argmax_reported;
static int profile_enabled;
static int model_slot_lock = -1;
static uint64_t profile_calls;
static uint64_t profile_columns;
static uint64_t profile_rows;
static uint64_t profile_weights;
static uint64_t profile_pack_ns;
static uint64_t profile_hardware_ns;
static uint64_t profile_output_ns;
static struct block_q8_0 quantized_input[6144u / 32u]
    __attribute__((aligned(64)));

void quantize_row_q8_0(const float *input, void *output, int64_t count);
void bonsai_quantize_q8_to_scratch(const float *input,
                                   volatile uint64_t *output,
                                   int64_t count) __attribute__((weak));

void __real_ggml_vec_dot_q1_0_q8_0(int n, float *s, size_t bs,
                                   const void *vx, size_t bx,
                                   const void *vy, size_t by, int nrc);

static void accelerator_fence(void) {
  __asm__ volatile("fence iorw, iorw" ::: "memory");
}

static uint64_t profile_now(void) {
  struct timespec now;
  if (clock_gettime(CLOCK_MONOTONIC_RAW, &now) != 0) return 0u;
  return (uint64_t)now.tv_sec * UINT64_C(1000000000) +
         (uint64_t)now.tv_nsec;
}

static void report_profile(void) {
  if (!profile_enabled || profile_calls == 0u) return;
  fprintf(stderr,
          "Bonsai FPGA profile: calls=%llu columns=%llu rows=%llu weights=%llu "
          "pack=%.3fms hardware=%.3fms output=%.3fms\n",
          (unsigned long long)profile_calls,
          (unsigned long long)profile_columns,
          (unsigned long long)profile_rows,
          (unsigned long long)profile_weights,
          (double)profile_pack_ns / 1000000.0,
          (double)profile_hardware_ns / 1000000.0,
          (double)profile_output_ns / 1000000.0);
}

static void lock_accelerator(void) {
  while (atomic_flag_test_and_set_explicit(&accelerator_lock,
                                            memory_order_acquire))
    __asm__ volatile("nop");
}

static void unlock_accelerator(void) {
  atomic_flag_clear_explicit(&accelerator_lock, memory_order_release);
}

static int stamps_equal(const struct model_stage_stamp *a,
                        const struct model_stage_stamp *b) {
  return a->device == b->device && a->inode == b->inode &&
         a->size == b->size && a->mtime_seconds == b->mtime_seconds &&
         a->mtime_nanoseconds == b->mtime_nanoseconds;
}

static int staged_model_matches(int source, const uint8_t *mapping,
                                size_t size) {
  enum { SAMPLE_BYTES = 64u * 1024u };
  uint8_t *sample = malloc(SAMPLE_BYTES);
  if (sample == NULL || size < SAMPLE_BYTES * 2u) {
    free(sample);
    return 0;
  }
  ssize_t first = pread(source, sample, SAMPLE_BYTES, 0);
  int matches = first == (ssize_t)SAMPLE_BYTES &&
                memcmp(sample, mapping, SAMPLE_BYTES) == 0;
  if (matches) {
    off_t offset = (off_t)(size - SAMPLE_BYTES);
    ssize_t last = pread(source, sample, SAMPLE_BYTES, offset);
    matches = last == (ssize_t)SAMPLE_BYTES &&
              memcmp(sample, mapping + (size_t)offset, SAMPLE_BYTES) == 0;
  }
  free(sample);
  return matches;
}

static int stage_model(int memory) {
  const char *path = getenv("BONSAI_MODEL");
  if (path == NULL || path[0] == '\0')
    path = "/root/models/Bonsai-1.7B.gguf";
  const char *prestaged_env = getenv("BONSAI_MODEL_PRESTAGED");
  int prestaged = prestaged_env != NULL && prestaged_env[0] != '\0' &&
                  strcmp(prestaged_env, "0") != 0;
  int source = -1;
  struct stat st = {0};
  if (!prestaged) {
    source = open(path, O_RDONLY);
    if (source < 0) {
      perror("Bonsai model open");
      return -1;
    }
    if (fstat(source, &st) != 0) {
      perror("Bonsai model stat");
      (void)close(source);
      return -1;
    }
    if (!S_ISREG(st.st_mode) ||
        (uint64_t)st.st_size != BONSAI_MODEL_FILE_SIZE) {
      fprintf(stderr, "Bonsai model stat mismatch: size=%llu expected=%llu\n",
              (unsigned long long)st.st_size,
              (unsigned long long)BONSAI_MODEL_FILE_SIZE);
      (void)close(source);
      return -1;
    }
  }
  int lock = open("/run/fpga-model.lock", O_RDWR | O_CREAT, 0600);
  if (lock < 0 || flock(lock, LOCK_EX | LOCK_NB) != 0) {
    if (lock >= 0 && (errno == EWOULDBLOCK || errno == EAGAIN))
      fprintf(stderr, "Bonsai model slot is in use; exit the other model runtime first\n");
    else
      perror("Bonsai model lock");
    if (source >= 0) (void)close(source);
    if (lock >= 0) (void)close(lock);
    return -1;
  }
  void *mapping = mmap(NULL, (size_t)BONSAI_MODEL_FILE_SIZE,
                       PROT_READ | PROT_WRITE, MAP_SHARED, memory,
                       (off_t)BONSAI_MODEL_PHYS);
  if (mapping == MAP_FAILED) {
    perror("Bonsai model DDR mapping");
    if (source >= 0) (void)close(source);
    (void)close(lock);
    return -1;
  }

  struct model_stage_stamp expected = {
      .device = (uint64_t)st.st_dev,
      .inode = (uint64_t)st.st_ino,
      .size = (uint64_t)st.st_size,
      .mtime_seconds = (int64_t)st.st_mtim.tv_sec,
      .mtime_nanoseconds = (int64_t)st.st_mtim.tv_nsec,
  };
  if (prestaged)
    (void)unlink("/run/fpga-model.stamp");
  struct model_stage_stamp actual;
  int stamp = open("/run/fpga-model.stamp", O_RDONLY);
  ssize_t stamp_size = stamp >= 0 ? read(stamp, &actual, sizeof(actual)) : -1;
  if (stamp >= 0) (void)close(stamp);
  int staged = !prestaged && stamp_size == (ssize_t)sizeof(actual) &&
               stamps_equal(&actual, &expected);
  if (!prestaged && !staged)
    staged = staged_model_matches(source, mapping,
                                  (size_t)BONSAI_MODEL_FILE_SIZE);
  if (!prestaged && !staged) {
    fprintf(stderr, "Staging Bonsai weights from SD:   0%%");
    size_t copied = 0;
    size_t next_report = 16u * 1024u * 1024u;
    while (copied < (size_t)BONSAI_MODEL_FILE_SIZE) {
      size_t remaining = (size_t)BONSAI_MODEL_FILE_SIZE - copied;
      size_t request = remaining < 1024u * 1024u ? remaining :
                       1024u * 1024u;
      ssize_t received = read(source, (uint8_t *)mapping + copied, request);
      if (received < 0 && errno == EINTR) continue;
      if (received <= 0) {
        perror("Bonsai model read");
        (void)munmap(mapping, (size_t)BONSAI_MODEL_FILE_SIZE);
        (void)close(source);
        (void)close(lock);
        return -1;
      }
      copied += (size_t)received;
      if (copied >= next_report || copied == (size_t)BONSAI_MODEL_FILE_SIZE) {
        fprintf(stderr, "\rStaging Bonsai weights from SD: %3zu%%",
                copied * 100u / (size_t)BONSAI_MODEL_FILE_SIZE);
        next_report = copied + 16u * 1024u * 1024u;
      }
    }
    accelerator_fence();
    stamp = open("/run/fpga-model.stamp",
                 O_WRONLY | O_CREAT | O_TRUNC, 0600);
    if (stamp < 0 || write(stamp, &expected, sizeof(expected)) !=
                       (ssize_t)sizeof(expected)) {
      perror("Bonsai model stamp");
      (void)munmap(mapping, (size_t)BONSAI_MODEL_FILE_SIZE);
      (void)close(source);
      (void)close(lock);
      return -1;
    }
    (void)close(stamp);
    fprintf(stderr, "\n");
  } else if (!prestaged &&
             (stamp_size != (ssize_t)sizeof(actual) ||
              !stamps_equal(&actual, &expected))) {
    /* A stage-2 boot can preload the model before /run exists. Record the
       verified contents so subsequent launches need no DDR sampling. */
    stamp = open("/run/fpga-model.stamp",
                 O_WRONLY | O_CREAT | O_TRUNC, 0600);
    if (stamp >= 0) {
      (void)write(stamp, &expected, sizeof(expected));
      (void)close(stamp);
    }
  }
  const uint8_t *bytes = mapping;
  uint8_t header[4] = {bytes[0], bytes[1], bytes[2], bytes[3]};
  int valid = header[0] == 'G' && header[1] == 'G' && header[2] == 'U' &&
              header[3] == 'F';
  (void)munmap(mapping, (size_t)BONSAI_MODEL_FILE_SIZE);
  if (source >= 0) (void)close(source);
  if (valid) {
    /* Retain the exclusive lock until process exit so another model runtime
       cannot overwrite the shared accelerator-visible DDR slot mid-run. */
    model_slot_lock = lock;
  } else {
    fprintf(stderr, "Bonsai staged model header mismatch: %02x %02x %02x %02x\n",
            header[0], header[1], header[2], header[3]);
    (void)flock(lock, LOCK_UN);
    (void)close(lock);
  }
  return valid ? 0 : -1;
}

static void initialize_accelerator(void) {
  accelerator_initialized = 1;
  profile_enabled = getenv("BONSAI_PROFILE") != NULL;
  if (profile_enabled) (void)atexit(report_profile);
  if (getenv("BONSAI_ACCELERATOR") != NULL &&
      strcmp(getenv("BONSAI_ACCELERATOR"), "0") == 0)
    return;
  int memory = open("/dev/mem", O_RDWR | O_SYNC);
  if (memory < 0) return;
  if (stage_model(memory) != 0) {
    (void)close(memory);
    fprintf(stderr, "Bonsai FPGA acceleration disabled: model staging failed\n");
    return;
  }
  void *gpio = mmap(NULL, 4096u, PROT_READ | PROT_WRITE, MAP_SHARED,
                    memory, (off_t)BONSAI_GPIO_PHYS);
  void *scratch = mmap(NULL, BONSAI_SCRATCH_BYTES,
                       PROT_READ | PROT_WRITE, MAP_SHARED, memory,
                       (off_t)BONSAI_SCRATCH_PHYS);
  (void)close(memory);
  if (gpio == MAP_FAILED || scratch == MAP_FAILED) {
    fprintf(stderr, "Bonsai FPGA acceleration disabled: mapping failed\n");
    return;
  }
  accelerator_gpio = gpio;
  accelerator_scratch = scratch;
  accelerator_gpio[1] = 0u;
  accelerator_fence();
  if ((accelerator_gpio[2] & BONSAI_STATUS_PRESENT) == 0u) {
    fprintf(stderr, "Bonsai FPGA acceleration disabled: accelerator absent\n");
    return;
  }
  accelerator_toggle = accelerator_gpio[0] & 0x80000000u;
  accelerator_available = 1;
}

int bonsai_accelerator_prepare(void) {
  lock_accelerator();
  if (!accelerator_initialized)
    initialize_accelerator();
  int available = accelerator_available;
  unlock_accelerator();
  return available;
}

static int pointer_file_offset(const void *pointer, uint64_t *offset_out) {
  FILE *maps = fopen("/proc/self/maps", "r");
  if (maps == NULL) return -1;
  uintptr_t address = (uintptr_t)pointer;
  char line[1024];
  int found = -1;
  while (fgets(line, sizeof(line), maps) != NULL) {
    unsigned long start;
    unsigned long end;
    unsigned long offset;
    if (sscanf(line, "%lx-%lx %*4s %lx", &start, &end, &offset) == 3 &&
        address >= (uintptr_t)start && address < (uintptr_t)end) {
      *offset_out = (uint64_t)offset + address - (uintptr_t)start;
      found = 0;
      break;
    }
  }
  (void)fclose(maps);
  return found;
}

static const struct bonsai_tensor_spec *find_tensor(uint64_t file_offset,
                                                     uint32_t n,
                                                     uint32_t *row_out) {
  for (size_t i = 0; i < BONSAI_TENSOR_COUNT; ++i) {
    const struct bonsai_tensor_spec *tensor = &bonsai_tensors[i];
    if (tensor->ne0 != n) continue;
    uint64_t row_bytes = (uint64_t)(n / 128u) * 18u;
    uint64_t bytes = row_bytes * tensor->ne1;
    if (file_offset >= tensor->offset && file_offset < tensor->offset + bytes) {
      uint64_t delta = file_offset - tensor->offset;
      if (delta % row_bytes != 0u) return NULL;
      *row_out = (uint32_t)(delta / row_bytes);
      return tensor;
    }
  }
  return NULL;
}

static const struct bonsai_tensor_spec *find_pointer_tensor(const void *pointer,
                                                             uint32_t n,
                                                             uint32_t *row_out,
                                                             uintptr_t *start_out) {
  uintptr_t address = (uintptr_t)pointer;
  if (cached_tensor != NULL && cached_tensor->ne0 == n) {
    uint64_t row_bytes = (uint64_t)(n / 128u) * 18u;
    if (address == cached_row_address) {
      *row_out = cached_row;
      *start_out = cached_tensor_address;
      return cached_tensor;
    }
    if (cached_row + 1u < cached_tensor->ne1 &&
        address == cached_row_address + row_bytes) {
      *row_out = cached_row + 1u;
      *start_out = cached_tensor_address;
      return cached_tensor;
    }
    if (address >= cached_tensor_address) {
      uint64_t delta = address - cached_tensor_address;
      if (delta < row_bytes * cached_tensor->ne1 &&
          delta % row_bytes == 0u) {
        *row_out = (uint32_t)(delta / row_bytes);
        *start_out = cached_tensor_address;
        return cached_tensor;
      }
    }
  }

  uint64_t file_offset;
  if (model_mapping_base_valid && address >= model_mapping_base &&
      address - model_mapping_base < BONSAI_MODEL_FILE_SIZE) {
    file_offset = address - model_mapping_base;
  } else {
    if (pointer_file_offset(pointer, &file_offset) != 0) return NULL;
    if (file_offset >= BONSAI_MODEL_FILE_SIZE) return NULL;
    model_mapping_base = address - (uintptr_t)file_offset;
    model_mapping_base_valid = 1;
  }
  const struct bonsai_tensor_spec *tensor = find_tensor(file_offset, n, row_out);
  if (tensor == NULL) return NULL;
  uint64_t row_bytes = (uint64_t)(n / 128u) * 18u;
  *start_out = address - (uintptr_t)(*row_out * row_bytes);
  return tensor;
}

static uint64_t load_u64(const uint8_t *data) {
  uint64_t value;
  memcpy(&value, data, sizeof(value));
  return value;
}

static void input_signature(const void *input, uint32_t n,
                            uint64_t signature[3]) {
  const uint8_t *bytes = input;
  size_t size = (size_t)(n / 32u) * sizeof(struct block_q8_0);
  signature[0] = load_u64(bytes);
  signature[1] = load_u64(bytes + size / 2u);
  signature[2] = load_u64(bytes + size - sizeof(uint64_t));
}

static void write_register(uint32_t index, uint32_t value) {
  accelerator_toggle ^= 0x80000000u;
  accelerator_gpio[0] = accelerator_toggle | 0x20000000u |
                        ((index & 7u) << 25) | (value & 0x01ffffffu);
  accelerator_fence();
  for (volatile uint32_t settle = 0; settle < 64u; ++settle)
    __asm__ volatile("nop");
}

static void copy_q8_to_scratch(const struct block_q8_0 *input,
                               uint32_t blocks) {
  /* blocks is a multiple of four, so the 34-byte-per-block total is a
     multiple of eight and 64-bit uncached stores need no tail. */
  const uint8_t *source = (const uint8_t *)input;
  uint32_t bytes = blocks * (uint32_t)sizeof(*input);
  volatile uint64_t *destination =
      (volatile uint64_t *)accelerator_scratch;
  for (uint32_t offset = 0; offset < bytes; offset += 8u) {
    uint64_t word;
    memcpy(&word, source + offset, sizeof(word));
    destination[offset / 8u] = word;
  }
}

static int run_tensor(const struct bonsai_tensor_spec *tensor,
                      const struct block_q8_0 *input) {
  uint32_t blocks = tensor->ne0 / 32u;
  copy_q8_to_scratch(input, blocks);
  accelerator_fence();
  uint32_t old_done = accelerator_gpio[2] & BONSAI_STATUS_DONE;
  write_register(0u, (uint32_t)(tensor->offset >> 3));
  write_register(1u, BONSAI_INPUT_OFFSET >> 3);
  write_register(2u, BONSAI_OUTPUT_OFFSET >> 3);
  write_register(3u, tensor->ne0);
  write_register(4u, tensor->ne1);
  write_register(5u, 3u);
  uint32_t status;
  do {
    status = accelerator_gpio[2];
  } while ((status & BONSAI_STATUS_DONE) == old_done);
  accelerator_fence();
  return (status & BONSAI_STATUS_ERROR) == 0u ? 0 : -1;
}

static uint64_t submit_columns;
static uint64_t submit_rows;
static uint64_t submit_done_ns;
static uint32_t submit_output_first;
static uint32_t submit_status;
void bonsai_accelerate_release(void);

/* Runs a whole-matrix matvec and leaves the raw Q30 results in the scratch
   region. On success returns 1 with the accelerator lock still held: the
   caller reads results with bonsai_accelerate_read_rows (safe from multiple
   threads of this process) and must finish with bonsai_accelerate_release.
   On failure returns 0 with the lock released. */
static int debug_enabled = -1;

static void debug_step(const char *step) {
  if (debug_enabled < 0)
    debug_enabled = getenv("BONSAI_DEBUG") != NULL;
  if (debug_enabled) {
    fprintf(stderr, "[bonsai] %s\n", step);
    fflush(stderr);
  }
}

static int bonsai_accelerate_f32_matvec_submit_at(
    const void *weights, uint32_t columns, uint32_t rows, const float *input,
    uint32_t output_first, uint32_t mode) {
  if (columns == 0u || columns > 6144u || (columns % 128u) != 0u)
    return 0;
  if ((uint64_t)BONSAI_OUTPUT_LOCAL +
          ((uint64_t)output_first + rows) * sizeof(int64_t) >
      BONSAI_SCRATCH_BYTES)
    return 0;
  lock_accelerator();
  if (!accelerator_initialized) initialize_accelerator();
  if (!accelerator_available) {
    debug_step("submit: accelerator unavailable");
    unlock_accelerator();
    return 0;
  }

  uint32_t first_row = 0u;
  uintptr_t tensor_address;
  const struct bonsai_tensor_spec *tensor =
      find_pointer_tensor(weights, columns, &first_row, &tensor_address);
  if (tensor == NULL || first_row != 0u || tensor->ne1 != rows) {
    if (debug_enabled < 0) debug_enabled = getenv("BONSAI_DEBUG") != NULL;
    if (debug_enabled) {
      uint64_t file_offset = UINT64_MAX;
      (void)pointer_file_offset(weights, &file_offset);
      fprintf(stderr,
              "[bonsai] submit declined: ptr=%p cols=%u rows=%u "
              "file_offset=%llu tensor=%s first_row=%u\n",
              weights, columns, rows,
              (unsigned long long)file_offset,
              tensor == NULL ? "none" : "found", first_row);
      fflush(stderr);
    }
    unlock_accelerator();
    return 0;
  }
  debug_step("submit: tensor matched, packing");

  uint64_t pack_start = profile_enabled ? profile_now() : 0u;
  /* Consecutive tensors of one graph layer (attn q/k/v, ffn up/gate) share
     the same activation vector; skip requantizing and re-copying it. The
     first float is compared in addition to the pointer so a buffer reused
     with new contents cannot alias a stale packing. */
  static const float *packed_input_pointer;
  static uint32_t packed_input_columns;
  static float packed_input_probe[3];
  if (packed_input_pointer != input || packed_input_columns != columns ||
      packed_input_probe[0] != input[0] ||
      packed_input_probe[1] != input[columns / 2u] ||
      packed_input_probe[2] != input[columns - 1u]) {
    if (bonsai_quantize_q8_to_scratch != NULL) {
      bonsai_quantize_q8_to_scratch(
          input, (volatile uint64_t *)accelerator_scratch, columns);
    } else {
      quantize_row_q8_0(input, quantized_input, columns);
      copy_q8_to_scratch(quantized_input, columns / 32u);
    }
    accelerator_fence();
    packed_input_pointer = input;
    packed_input_columns = columns;
    packed_input_probe[0] = input[0];
    packed_input_probe[1] = input[columns / 2u];
    packed_input_probe[2] = input[columns - 1u];
  }
  if (profile_enabled) profile_pack_ns += profile_now() - pack_start;
  uint64_t hardware_start = profile_enabled ? profile_now() : 0u;
  uint32_t old_done = accelerator_gpio[2] & BONSAI_STATUS_DONE;
  write_register(0u, (uint32_t)(tensor->offset >> 3));
  write_register(1u, BONSAI_INPUT_OFFSET >> 3);
  write_register(2u, (BONSAI_OUTPUT_OFFSET >> 3) + output_first);
  write_register(3u, tensor->ne0);
  write_register(4u, tensor->ne1);
  write_register(5u, mode);
  if (debug_enabled > 0) {
    static int dumped;
    if (!dumped) {
      dumped = 1;
      fprintf(stderr,
              "[drv] regs: w=%llu in=%u out=%u ne0=%u ne1=%u gpio0=%08x "
              "scratch_rb=%02x%02x%02x%02x%02x%02x%02x%02x q8d=%04x\n",
              (unsigned long long)(tensor->offset >> 3),
              BONSAI_INPUT_OFFSET >> 3, BONSAI_OUTPUT_OFFSET >> 3,
              tensor->ne0, tensor->ne1, accelerator_gpio[0],
              accelerator_scratch[0], accelerator_scratch[1],
              accelerator_scratch[2], accelerator_scratch[3],
              accelerator_scratch[4], accelerator_scratch[5],
              accelerator_scratch[6], accelerator_scratch[7],
              (unsigned)*(const uint16_t *)quantized_input);
      fflush(stderr);
    }
  }
  debug_step("submit: command written, polling");
  uint32_t status;
  do {
    status = accelerator_gpio[2];
  } while ((status & BONSAI_STATUS_DONE) == old_done);
  accelerator_fence();
  if (debug_enabled > 0) {
    static int dumped2;
    if (!dumped2) {
      dumped2 = 1;
      const volatile int64_t *raw =
          (const volatile int64_t *)(accelerator_scratch + BONSAI_OUTPUT_LOCAL);
      fprintf(stderr, "[drv] status=%08x raw[0..3]=%lld %lld %lld %lld\n",
              status, (long long)raw[0], (long long)raw[1],
              (long long)raw[2], (long long)raw[3]);
      fflush(stderr);
    }
  }
  debug_step("submit: done observed");
  if (profile_enabled)
    profile_hardware_ns += profile_now() - hardware_start;
  if ((status & BONSAI_STATUS_ERROR) != 0u) {
    accelerator_available = 0;
    fprintf(stderr, "Bonsai FPGA acceleration disabled: command failed\n");
    unlock_accelerator();
    return 0;
  }

  if (!whole_matrix_reported) {
    whole_matrix_reported = 1;
    fprintf(stderr, "Bonsai FPGA whole-matrix path enabled\n");
  }
  cached_tensor = tensor;
  cached_tensor_address = tensor_address;
  submit_columns = columns;
  submit_rows = rows;
  submit_output_first = output_first;
  submit_status = status;
  submit_done_ns = profile_enabled ? profile_now() : 0u;
  return 1;
}

int bonsai_accelerate_f32_matvec_submit(const void *weights, uint32_t columns,
                                        uint32_t rows, const float *input) {
  return bonsai_accelerate_f32_matvec_submit_at(weights, columns, rows, input,
                                                0u, 3u);
}

int bonsai_accelerate_f32_matvec_store(const void *weights, uint32_t columns,
                                       uint32_t rows, const float *input,
                                       uint32_t output_first) {
  if (!bonsai_accelerate_f32_matvec_submit_at(
          weights, columns, rows, input, output_first, 3u))
    return 0;
  bonsai_accelerate_release();
  return 1;
}

int bonsai_accelerate_f32_matvec_argmax(const void *weights,
                                        uint32_t columns, uint32_t rows,
                                        const float *input,
                                        uint32_t *best_token) {
  lock_accelerator();
  if (!accelerator_initialized) initialize_accelerator();
  int supported = accelerator_available &&
                  (accelerator_gpio[2] & UINT32_C(0x08000000)) != 0u;
  unlock_accelerator();
  if (!supported || !bonsai_accelerate_f32_matvec_submit_at(
          weights, columns, rows, input, 0u, 4u))
    return 0;
  if (!hardware_argmax_reported) {
    hardware_argmax_reported = 1;
    fprintf(stderr, "Bonsai FPGA hardware argmax enabled\n");
  }
  *best_token = (submit_status >> 8) & UINT32_C(0x0003ffff);
  bonsai_accelerate_release();
  return 1;
}

/* This FPU's 64-bit integer converts (fcvt.s.l) are unreliable, so build
   the float from 32-bit halves, which convert correctly. */
static inline float s64_q30_to_float(int64_t value) {
  int32_t high = (int32_t)(value >> 32);
  uint32_t low = (uint32_t)value;
  return (float)high * 4.0f + (float)low * (1.0f / 1073741824.0f);
}

void bonsai_accelerate_read_rows(float *output, uint32_t first,
                                 uint32_t count) {
  const volatile int64_t *hardware_output =
      (const volatile int64_t *)(accelerator_scratch + BONSAI_OUTPUT_LOCAL) +
      submit_output_first;
  for (uint32_t row = first; row < first + count; ++row)
    output[row] = s64_q30_to_float(hardware_output[row]);
}

void bonsai_accelerate_read_stored_rows(float *output, uint32_t source_first,
                                        uint32_t count) {
  const volatile int64_t *hardware_output =
      (const volatile int64_t *)(accelerator_scratch + BONSAI_OUTPUT_LOCAL) +
      source_first;
  for (uint32_t row = 0; row < count; ++row)
    output[row] = s64_q30_to_float(hardware_output[row]);
}

const volatile int64_t *bonsai_accelerate_stored_output(void) {
  return (const volatile int64_t *)(accelerator_scratch + BONSAI_OUTPUT_LOCAL);
}

void bonsai_accelerate_argmax_rows(uint32_t first, uint32_t count,
                                   int64_t *best_raw, uint32_t *best_token) {
  const volatile int64_t *hardware_output =
      (const volatile int64_t *)(accelerator_scratch + BONSAI_OUTPUT_LOCAL) +
      submit_output_first;
  int64_t score = INT64_MIN;
  uint32_t token = first;
  for (uint32_t row = first; row < first + count; ++row) {
    int64_t value = hardware_output[row];
    if (value > score) {
      score = value;
      token = row;
    }
  }
  *best_raw = score;
  *best_token = token;
}

void bonsai_accelerate_release(void) {
  if (profile_enabled) {
    profile_output_ns += profile_now() - submit_done_ns;
    ++profile_calls;
    profile_columns += submit_columns;
    profile_rows += submit_rows;
    profile_weights += submit_columns * submit_rows;
  }
  unlock_accelerator();
}

int bonsai_accelerate_f32_matvec(const void *weights, uint32_t columns,
                                 uint32_t rows, const float *input,
                                 float *output) {
  if (!bonsai_accelerate_f32_matvec_submit(weights, columns, rows, input))
    return 0;
  bonsai_accelerate_read_rows(output, 0u, rows);
  bonsai_accelerate_release();
  return 1;
}

void __wrap_ggml_vec_dot_q1_0_q8_0(int n, float *s, size_t bs,
                                   const void *vx, size_t bx,
                                   const void *vy, size_t by, int nrc) {
  if (nrc != 1 || n <= 0 || n > 6144 || (n % 128) != 0) {
    __real_ggml_vec_dot_q1_0_q8_0(n, s, bs, vx, bx, vy, by, nrc);
    return;
  }
  lock_accelerator();
  if (!accelerator_initialized) initialize_accelerator();
  if (!accelerator_available) {
    unlock_accelerator();
    __real_ggml_vec_dot_q1_0_q8_0(n, s, bs, vx, bx, vy, by, nrc);
    return;
  }

  uint32_t row;
  uintptr_t tensor_address;
  const struct bonsai_tensor_spec *tensor =
      find_pointer_tensor(vx, (uint32_t)n, &row, &tensor_address);
  if (tensor == NULL) {
    unlock_accelerator();
    __real_ggml_vec_dot_q1_0_q8_0(n, s, bs, vx, bx, vy, by, nrc);
    return;
  }
  cached_row_address = (uintptr_t)vx;
  cached_row = row;

  uint64_t signature[3];
  input_signature(vy, (uint32_t)n, signature);
  if (cached_tensor != tensor || cached_input != vy ||
      cached_signature[0] != signature[0] ||
      cached_signature[1] != signature[1] ||
      cached_signature[2] != signature[2]) {
    if (run_tensor(tensor, vy) != 0) {
      accelerator_available = 0;
      fprintf(stderr, "Bonsai FPGA acceleration disabled: command failed\n");
      unlock_accelerator();
      __real_ggml_vec_dot_q1_0_q8_0(n, s, bs, vx, bx, vy, by, nrc);
      return;
    }
    cached_tensor = tensor;
    cached_tensor_address = tensor_address;
    cached_input = vy;
    memcpy(cached_signature, signature, sizeof(cached_signature));
  } else if (cached_tensor_address != tensor_address) {
    cached_tensor_address = tensor_address;
  }
  const volatile int64_t *output =
      (const volatile int64_t *)(accelerator_scratch + BONSAI_OUTPUT_LOCAL);
  *s = s64_q30_to_float(output[row]);
  unlock_accelerator();
}

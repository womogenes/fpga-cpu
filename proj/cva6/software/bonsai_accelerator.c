#define _GNU_SOURCE

#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
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
#define BONSAI_SCRATCH_BYTES (32u * 1024u * 1024u)
#define BONSAI_INPUT_OFFSET 0x0f000000u
#define BONSAI_OUTPUT_OFFSET 0x0f010000u
#define BONSAI_OUTPUT_LOCAL 0x00010000u
#define BONSAI_STATUS_PRESENT 0x80000000u
#define BONSAI_STATUS_ERROR 0x20000000u
#define BONSAI_STATUS_DONE 0x10000000u
#define BONSAI_STATUS_BUS_REQUEST 0x40000000u
#define BONSAI_STATUS_COMMAND_ACK 0x08000000u
#define BONSAI_STATUS_PACKED_GENERATION 0x04000000u
#define BONSAI_STATUS_VECTOR_PACK 0x02000000u
#define BONSAI_STATUS_ATTENTION 0x01000000u
#define BONSAI_STATUS_ATTENTION_PACK 0x00800000u
#define BONSAI_NORM_SOURCE_LOCAL 0x00001000u
#define BONSAI_ATTENTION_TABLE_LOCAL 0x00200000u
#define BONSAI_ATTENTION_KV_LOCAL 0x00240000u
#define BONSAI_CACHED_INPUT_BYTES 40960u
#define BONSAI_RESIDUAL_SUM_LOCAL 0x00280000u
#define BONSAI_NORM_SLOT_FALLBACK_BASE 0x00300000u
#define BONSAI_NORM_SLOT_BYTES 0x00006000u
#define BONSAI_NORM_OUTPUT_LOCAL 0x00000000u
#define BONSAI_NORM_RESIDUAL_LOCAL 0x00002000u
#define BONSAI_NORM_WEIGHT_LOCAL 0x00004000u
#define BONSAI_NORM_SLOT_COUNT 60u

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

static volatile uint32_t accelerator_lock;
static volatile uint32_t *accelerator_gpio;
static volatile uint8_t *accelerator_scratch;
static uint8_t *cached_input_buffer;
static volatile uint8_t *accelerator_input;
static uint32_t accelerator_input_words;
static int accelerator_input_absolute;
static const float *norm_slot_weights[BONSAI_NORM_SLOT_COUNT];
static uint32_t norm_slot_count;
static volatile uint64_t *resident_norm_source;
static const float *resident_norm_weight;
static int resident_norm_valid;
static int forced_output_active;
static uint32_t forced_output_words;
static const volatile int64_t *forced_output_pointer;
enum {
  OUTPUT_SLOT_BYTES = 96u * 1024u,
  /* Eight slots span two complete four-matvec transformer-layer cycles. Even
     the worker that only reads the large FFN result consumes 24 KiB from the
     intervening slot before an address is reused, deterministically evicting
     every old line from its private 8 KiB D-cache. */
  OUTPUT_SLOT_COUNT = 8u,
  OUTPUT_RING_BYTES = OUTPUT_SLOT_BYTES * OUTPUT_SLOT_COUNT,
};
static uint8_t *cached_output_buffer;
static uint32_t cached_output_words;
static uint32_t cached_output_slot = OUTPUT_SLOT_COUNT - 1u;
static const volatile int64_t *active_output;
static uint32_t active_output_words;
static int accelerator_output_absolute;
static int accelerator_output_cached_ring;
static uint32_t accelerator_toggle;
static int accelerator_initialized;
static int accelerator_available;
static int accelerator_packed_model;
static int accelerator_float_output;
static int accelerator_attention_available;
static int accelerator_output_compact;
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
static uint64_t profile_ffn_pack_ns;
static uint64_t profile_norm_pack_ns;
static uint64_t profile_ffn_pack_calls;
static uint64_t profile_norm_pack_calls;
static const float *packed_input_pointer;
static uint32_t packed_input_columns;
static float packed_input_probe[3];
struct profile_shape {
  uint32_t columns;
  uint32_t rows;
  uint64_t calls;
  uint64_t pack_ns;
  uint64_t hardware_ns;
  uint64_t output_ns;
};
static struct profile_shape profile_shapes[8];
static struct profile_shape *active_profile_shape;
static struct block_q8_0 quantized_input[6144u / 32u]
    __attribute__((aligned(64)));

void quantize_row_q8_0(const float *input, void *output, int64_t count);
void bonsai_quantize_q8_to_scratch(const float *input,
                                   volatile uint64_t *output,
                                   int64_t count) __attribute__((weak));
float bonsai_reliable_resident_sum_squares(
    const volatile uint32_t *values, uint32_t count);
float bonsai_fast_resident_sum_squares(
    const volatile uint32_t *values, uint32_t count);

void __real_ggml_vec_dot_q1_0_q8_0(int n, float *s, size_t bs,
                                   const void *vx, size_t bx,
                                   const void *vy, size_t by, int nrc);

static void accelerator_fence(void) {
#ifdef __riscv
  __asm__ volatile("fence iorw, iorw" ::: "memory");
#else
  __atomic_thread_fence(__ATOMIC_SEQ_CST);
#endif
}

static void initialize_cached_input(void) {
  enum { ARENA_BYTES = 16u * 1024u * 1024u };
  accelerator_input = accelerator_scratch;
  accelerator_input_words = BONSAI_INPUT_OFFSET >> 3;
  accelerator_input_absolute = 0;
  const char *cached_env = getenv("BONSAI_CACHED_INPUT");
  if (cached_env == NULL || strcmp(cached_env, "1") != 0) return;
  if ((accelerator_gpio[2] & BONSAI_STATUS_PACKED_GENERATION) == 0u) return;
  uint8_t *arena = mmap(NULL, ARENA_BYTES, PROT_READ | PROT_WRITE,
                        MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (arena == MAP_FAILED) return;
  for (size_t offset = 0; offset < ARENA_BYTES; offset += 4096u)
    arena[offset] = 0u;
  uint64_t entries[ARENA_BYTES / 4096u];
  int pagemap = open("/proc/self/pagemap", O_RDONLY);
  uintptr_t first_page = (uintptr_t)arena / 4096u;
  ssize_t received = pagemap >= 0 ?
      pread(pagemap, entries, sizeof(entries),
            (off_t)(first_page * sizeof(entries[0]))) : -1;
  if (pagemap >= 0) (void)close(pagemap);
  uint64_t first = 0u;
  if (received == (ssize_t)sizeof(entries)) {
    for (size_t page = 0;
         page + BONSAI_CACHED_INPUT_BYTES / 4096u <= ARENA_BYTES / 4096u;
         ++page) {
      uint64_t first_entry = entries[page];
      uint64_t first_frame =
          first_entry & UINT64_C(0x007fffffffffffff);
      int contiguous = (first_entry & UINT64_C(0x8000000000000000)) != 0u &&
                       first_frame != 0u;
      for (size_t span = 1u;
           contiguous && span < BONSAI_CACHED_INPUT_BYTES / 4096u; ++span) {
        uint64_t entry = entries[page + span];
        contiguous = (entry & UINT64_C(0x8000000000000000)) != 0u &&
            (entry & UINT64_C(0x007fffffffffffff)) == first_frame + span;
      }
      if (contiguous) {
        first = first_frame * 4096u;
        size_t offset = page * 4096u;
        cached_input_buffer = arena + offset;
        break;
      }
    }
  }
  if (cached_input_buffer == NULL ||
      mlock(cached_input_buffer, BONSAI_CACHED_INPUT_BYTES) != 0) {
    cached_input_buffer = NULL;
    (void)munmap(arena, ARENA_BYTES);
    return;
  }
  accelerator_input = cached_input_buffer;
  /* CVA6's DDR window starts at 0x80000000, while this accelerator's
     30-bit AXI port carries the window-relative address. */
  accelerator_input_words = (uint32_t)((first & UINT64_C(0x3fffffff)) >> 3);
  accelerator_input_absolute = 1;
}

static void initialize_cached_output(void) {
  active_output =
      (const volatile int64_t *)(accelerator_scratch + BONSAI_OUTPUT_LOCAL);
  accelerator_output_absolute = 0;
  accelerator_output_cached_ring = 0;

  const char *cached_env = getenv("BONSAI_CACHED_OUTPUT");
  if (cached_env == NULL || strcmp(cached_env, "1") != 0) return;

  /* Anonymous pages must be faulted in before pagemap reveals their physical
     frames.  That necessarily leaves zero-filled lines in the CVA6 caches,
     and accelerator DMA cannot invalidate those lines.  Keep this path only
     as an explicit diagnostic; the reserved /dev/mem ring below begins with
     no process-owned cache lines and rotates far beyond the private-cache
     capacity before reuse. */
  if ((accelerator_gpio[2] & BONSAI_STATUS_PACKED_GENERATION) != 0u &&
      strcmp(cached_env, "anonymous") == 0) {
    enum { ARENA_BYTES = 32u * 1024u * 1024u };
    uint8_t *arena = mmap(NULL, ARENA_BYTES, PROT_READ | PROT_WRITE,
                          MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (arena != MAP_FAILED) {
      for (size_t offset = 0; offset < ARENA_BYTES; offset += 4096u)
        arena[offset] = 0u;
      const size_t page_count = ARENA_BYTES / 4096u;
      const size_t needed_pages = OUTPUT_RING_BYTES / 4096u;
      uint64_t *entries = malloc(page_count * sizeof(*entries));
      int pagemap = open("/proc/self/pagemap", O_RDONLY);
      uintptr_t virtual_page = (uintptr_t)arena / 4096u;
      ssize_t received = entries != NULL && pagemap >= 0 ?
          pread(pagemap, entries, page_count * sizeof(*entries),
                (off_t)(virtual_page * sizeof(*entries))) : -1;
      if (pagemap >= 0) (void)close(pagemap);
      uint64_t first_physical = 0u;
      size_t selected_offset = 0u;
      size_t run = 0u;
      uint64_t previous_frame = 0u;
      if (received == (ssize_t)(page_count * sizeof(*entries))) {
        for (size_t page = 0; page < page_count; ++page) {
          uint64_t entry = entries[page];
          uint64_t frame = entry & UINT64_C(0x007fffffffffffff);
          int present = (entry & UINT64_C(0x8000000000000000)) != 0u &&
                        frame != 0u;
          if (present && run != 0u && frame == previous_frame + 1u)
            ++run;
          else
            run = present ? 1u : 0u;
          previous_frame = frame;
          if (run == needed_pages) {
            size_t first_page = page + 1u - needed_pages;
            selected_offset = first_page * 4096u;
            first_physical = (frame + 1u - needed_pages) * 4096u;
            cached_output_buffer = arena + selected_offset;
            break;
          }
        }
      }
      free(entries);
      if (cached_output_buffer != NULL &&
          mlock(cached_output_buffer, OUTPUT_RING_BYTES) == 0) {
        cached_output_words = (uint32_t)(
            (first_physical & UINT64_C(0x3fffffff)) >> 3);
        if (selected_offset != 0u)
          (void)munmap(arena, selected_offset);
        size_t selected_end = selected_offset + OUTPUT_RING_BYTES;
        if (selected_end < ARENA_BYTES)
          (void)munmap(arena + selected_end, ARENA_BYTES - selected_end);
        active_output = (const volatile int64_t *)cached_output_buffer;
        accelerator_output_absolute = 1;
        accelerator_output_cached_ring = 1;
        return;
      }
      cached_output_buffer = NULL;
      (void)munmap(arena, ARENA_BYTES);
    }
  }

  /* The scratch reserved-memory range is below high_memory. Opening /dev/mem
     without O_SYNC therefore creates a normal cacheable alias on Linux,
     while the original mapping remains uncached for control/input fallback.
     Rotating through eight 96 KiB slots prevents DMA from ever overwriting a
     line that can still reside in any worker's private 8 KiB cache. */
  int memory = open("/dev/mem", O_RDWR);
  if (memory < 0) return;
  void *mapping = mmap(NULL, OUTPUT_RING_BYTES, PROT_READ | PROT_WRITE,
                       MAP_SHARED, memory,
                       (off_t)(BONSAI_SCRATCH_PHYS + BONSAI_OUTPUT_LOCAL));
  (void)close(memory);
  if (mapping == MAP_FAILED) return;
  cached_output_buffer = mapping;
  cached_output_words = BONSAI_OUTPUT_OFFSET >> 3;
  active_output = (const volatile int64_t *)cached_output_buffer;
  accelerator_output_cached_ring = 1;
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
  fprintf(stderr,
          "Bonsai FPGA transforms: ffn_pack=%.3fms/%llu "
          "norm_pack=%.3fms/%llu\n",
          (double)profile_ffn_pack_ns / 1000000.0,
          (unsigned long long)profile_ffn_pack_calls,
          (double)profile_norm_pack_ns / 1000000.0,
          (unsigned long long)profile_norm_pack_calls);
  for (size_t i = 0; i < sizeof(profile_shapes) / sizeof(profile_shapes[0]);
       ++i) {
    const struct profile_shape *shape = &profile_shapes[i];
    if (shape->calls == 0u) continue;
    fprintf(stderr,
            "Bonsai FPGA shape: %ux%u calls=%llu pack=%.3fms "
            "hardware=%.3fms output=%.3fms\n",
            shape->columns, shape->rows,
            (unsigned long long)shape->calls,
            (double)shape->pack_ns / 1000000.0,
            (double)shape->hardware_ns / 1000000.0,
            (double)shape->output_ns / 1000000.0);
  }
}

static struct profile_shape *find_profile_shape(uint32_t columns,
                                                uint32_t rows) {
  for (size_t i = 0; i < sizeof(profile_shapes) / sizeof(profile_shapes[0]);
       ++i) {
    if (profile_shapes[i].calls != 0u &&
        profile_shapes[i].columns == columns &&
        profile_shapes[i].rows == rows)
      return &profile_shapes[i];
  }
  for (size_t i = 0; i < sizeof(profile_shapes) / sizeof(profile_shapes[0]);
       ++i) {
    if (profile_shapes[i].calls == 0u) {
      profile_shapes[i].columns = columns;
      profile_shapes[i].rows = rows;
      return &profile_shapes[i];
    }
  }
  return NULL;
}

static void lock_accelerator(void) {
  uint32_t prior;
  do {
#ifdef __riscv
    __asm__ volatile("amoswap.w.aq %0, %2, (%1)"
                     : "=r"(prior) : "r"(&accelerator_lock), "r"(1u)
                     : "memory");
#else
    prior = __atomic_exchange_n(&accelerator_lock, 1u, __ATOMIC_ACQUIRE);
#endif
  } while (prior != 0u);
}

static void unlock_accelerator(void) {
#ifdef __riscv
  __asm__ volatile("amoswap.w.rl zero, zero, (%0)" ::
                   "r"(&accelerator_lock) : "memory");
#else
  __atomic_store_n(&accelerator_lock, 0u, __ATOMIC_RELEASE);
#endif
}

static int stamps_equal(const struct model_stage_stamp *a,
                        const struct model_stage_stamp *b) {
  return a->device == b->device && a->inode == b->inode &&
         a->size == b->size && a->mtime_seconds == b->mtime_seconds &&
         a->mtime_nanoseconds == b->mtime_nanoseconds;
}

enum {
  FAST_STAGE_WINDOW_BYTES = 4u * 1024u * 1024u,
  FAST_STAGE_WORKERS = 4u,
};

struct fast_stage_worker {
  const uint8_t *source;
  uint8_t *destination;
  size_t size;
};

static void *fast_stage_worker_main(void *argument) {
  const struct fast_stage_worker *worker = argument;
  (void)memcpy(worker->destination, worker->source, worker->size);
  return NULL;
}

static int fast_stage_model(int source, uint8_t *mapping, size_t begin,
                            size_t end, size_t total) {
  uint8_t *buffer = NULL;
  int allocation = posix_memalign((void **)&buffer, 4096u,
                                  FAST_STAGE_WINDOW_BYTES);
  if (allocation != 0) {
    errno = allocation;
    perror("Bonsai fast staging buffer");
    return -1;
  }
  if (lseek(source, (off_t)begin, SEEK_SET) < 0) {
    perror("Bonsai fast staging seek");
    free(buffer);
    return -1;
  }
  fprintf(stderr, "Streaming Bonsai weights from SD: %3zu%%",
          begin * 100u / total);
  size_t offset = begin;
  while (offset < end) {
    size_t window = end - offset;
    if (window > FAST_STAGE_WINDOW_BYTES) window = FAST_STAGE_WINDOW_BYTES;
    size_t received = 0u;
    while (received < window) {
      ssize_t result = read(source, buffer + received, window - received);
      if (result < 0 && errno == EINTR) continue;
      if (result <= 0) {
        if (result == 0) errno = EIO;
        perror("Bonsai fast staging read");
        free(buffer);
        return -1;
      }
      received += (size_t)result;
    }

    pthread_t threads[FAST_STAGE_WORKERS];
    struct fast_stage_worker workers[FAST_STAGE_WORKERS];
    size_t part = (window + FAST_STAGE_WORKERS - 1u) / FAST_STAGE_WORKERS;
    part = (part + 4095u) & ~(size_t)4095u;
    unsigned created = 0u;
    int failed = 0;
    for (unsigned index = 0u; index < FAST_STAGE_WORKERS; ++index) {
      size_t worker_begin = (size_t)index * part;
      size_t worker_end = worker_begin + part;
      if (worker_begin > window) worker_begin = window;
      if (worker_end > window) worker_end = window;
      workers[index] = (struct fast_stage_worker){
          .source = buffer + worker_begin,
          .destination = mapping + offset + worker_begin,
          .size = worker_end - worker_begin,
      };
      int error = pthread_create(&threads[index], NULL,
                                 fast_stage_worker_main, &workers[index]);
      if (error != 0) {
        errno = error;
        perror("Bonsai fast staging worker");
        failed = 1;
        break;
      }
      ++created;
    }
    for (unsigned index = 0u; index < created; ++index) {
      int error = pthread_join(threads[index], NULL);
      if (error != 0) {
        errno = error;
        perror("Bonsai fast staging join");
        failed = 1;
      }
    }
    if (failed) {
      free(buffer);
      return -1;
    }
    (void)posix_fadvise(source, (off_t)offset, (off_t)window,
                        POSIX_FADV_DONTNEED);
    offset += window;
    fprintf(stderr, "\rStreaming Bonsai weights from SD: %3zu%%",
            offset * 100u / total);
  }
  accelerator_fence();
  fprintf(stderr, "\n");
  free(buffer);
  return 0;
}

static int stage_model(int memory) {
  const char *path = getenv("BONSAI_MODEL");
  if (path == NULL || path[0] == '\0')
    path = "/root/models/Bonsai-1.7B.gguf";
  const char *prestaged_env = getenv("BONSAI_MODEL_PRESTAGED");
  int prestaged = prestaged_env != NULL && prestaged_env[0] != '\0' &&
                  strcmp(prestaged_env, "0") != 0;
  int source = -1;
  int source_packed = 0;
  size_t resume_bytes = 0u;
  size_t copy_end = 0u;
  struct stat st = {0};
  size_t stage_bytes = (size_t)BONSAI_MODEL_FILE_SIZE;
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
    uint8_t source_magic[4] = {0};
    ssize_t magic_size = pread(source, source_magic, sizeof(source_magic), 0);
    source_packed = magic_size == (ssize_t)sizeof(source_magic) &&
                    source_magic[0] == 'B' && source_magic[1] == 'G' &&
                    source_magic[2] == 'U' && source_magic[3] == 'F';
    if (!S_ISREG(st.st_mode) || st.st_size <= 0 ||
        ((!source_packed || (uint64_t)st.st_size > BONSAI_MODEL_FILE_SIZE) &&
         (uint64_t)st.st_size != BONSAI_MODEL_FILE_SIZE)) {
      fprintf(stderr, "Bonsai model stat mismatch: size=%llu expected=%llu\n",
              (unsigned long long)st.st_size,
              (unsigned long long)BONSAI_MODEL_FILE_SIZE);
      (void)close(source);
      return -1;
    }
    stage_bytes = (size_t)st.st_size;
    copy_end = stage_bytes;
    const char *resume_env = getenv("BONSAI_STAGE_RESUME_MB");
    if (resume_env != NULL) {
      unsigned long resume_mb = strtoul(resume_env, NULL, 10);
      if (resume_mb <= stage_bytes / (1024u * 1024u))
        resume_bytes = (size_t)resume_mb * 1024u * 1024u;
    }
    const char *end_env = getenv("BONSAI_STAGE_END_MB");
    if (end_env != NULL) {
      unsigned long end_mb = strtoul(end_env, NULL, 10);
      size_t requested_end = (size_t)end_mb * 1024u * 1024u;
      if (requested_end > resume_bytes && requested_end < copy_end)
        copy_end = requested_end;
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
  int model_memory = memory;
  if (getenv("BONSAI_CACHED_STAGE") != NULL) {
    int cached_memory = open("/dev/mem", O_RDWR);
    if (cached_memory >= 0) model_memory = cached_memory;
  }
  /* A prestaged launch only needs to validate the resident model's magic.
     Mapping all 248 MB here builds tens of thousands of device PTEs, then
     map_model() immediately builds the real cacheable inference mapping a
     second time. Keep this validation mapping to a single page. */
  size_t mapping_bytes = prestaged ? 4096u :
      (size_t)BONSAI_MODEL_FILE_SIZE;
  void *mapping = mmap(NULL, mapping_bytes,
                       PROT_READ | PROT_WRITE, MAP_SHARED, model_memory,
                       (off_t)BONSAI_MODEL_PHYS);
  if (model_memory != memory) (void)close(model_memory);
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
  const uint8_t *mapped_bytes = mapping;
  int mapped_header_standard = mapped_bytes[0] == 'G' &&
                            mapped_bytes[1] == 'G' &&
                            mapped_bytes[2] == 'U' &&
                            mapped_bytes[3] == 'F';
  int mapped_header_packed = mapped_bytes[0] == 'B' &&
                             mapped_bytes[1] == 'G' &&
                             mapped_bytes[2] == 'U' &&
                             mapped_bytes[3] == 'F';
  int staged_header_matches = source_packed ? mapped_header_packed :
                                              mapped_header_standard;
  int staged = prestaged ? (mapped_header_standard || mapped_header_packed) :
      stamp_size == (ssize_t)sizeof(actual) &&
      stamps_equal(&actual, &expected) && staged_header_matches;
  if (resume_bytes != 0u) staged = 0;
  int record_stamp = 0;
  if (!prestaged && !staged) {
    const char *fast_stage_env = getenv("BONSAI_FAST_STAGE");
    int fast_stage = fast_stage_env != NULL && fast_stage_env[0] != '\0' &&
                     strcmp(fast_stage_env, "0") != 0;
    if (fast_stage) {
      if (fast_stage_model(source, mapping, resume_bytes, copy_end,
                           stage_bytes) != 0) {
        (void)munmap(mapping, mapping_bytes);
        (void)close(source);
        (void)close(lock);
        return -1;
      }
      record_stamp = copy_end == stage_bytes;
    } else {
    /* The reserved-DDR mapping broadcasts coherence invalidations for every
       repaired store. Use sequential 64 KiB reads and yield after each chunk
       so staging remains responsive without paying a syscall per page. */
    enum { STAGE_CHUNK_BYTES = 64u * 1024u };
    uint8_t *stage_buffer = aligned_alloc(64u, STAGE_CHUNK_BYTES);
    if (stage_buffer == NULL) {
      perror("Bonsai model staging buffer");
      (void)munmap(mapping, mapping_bytes);
      (void)close(source);
      (void)close(lock);
      return -1;
    }
    size_t repaired_bytes = 0u;
    int overwrite_stage = getenv("BONSAI_STAGE_OVERWRITE") != NULL;
    fprintf(stderr, "Verifying/restoring Bonsai weights from SD: %3zu%%",
            resume_bytes * 100u / stage_bytes);
    size_t copied = resume_bytes;
    if (resume_bytes != 0u &&
        lseek(source, (off_t)resume_bytes, SEEK_SET) < 0) {
      perror("Bonsai model resume seek");
      (void)munmap(mapping, mapping_bytes);
      (void)close(source);
      (void)close(lock);
      return -1;
    }
    size_t next_report = copied + 16u * 1024u * 1024u;
    while (copied < copy_end) {
      size_t remaining = copy_end - copied;
      size_t request = remaining < STAGE_CHUNK_BYTES ? remaining :
                       STAGE_CHUNK_BYTES;
      ssize_t received = read(source, stage_buffer, request);
      if (received < 0 && errno == EINTR) continue;
      if (received <= 0) {
        perror("Bonsai model read");
        free(stage_buffer);
        (void)munmap(mapping, mapping_bytes);
        (void)close(source);
        (void)close(lock);
        return -1;
      }
      volatile uint64_t *destination =
          (volatile uint64_t *)((uint8_t *)mapping + copied);
      size_t words = (size_t)received / sizeof(*destination);
      const uint64_t *source_words = (const uint64_t *)stage_buffer;
      for (size_t word = 0; word < words; ++word) {
        uint64_t wanted = source_words[word];
        if (overwrite_stage || destination[word] != wanted) {
          destination[word] = wanted;
          repaired_bytes += sizeof(*destination);
        }
      }
      for (size_t byte = words * sizeof(*destination);
           byte < (size_t)received; ++byte) {
        if (overwrite_stage ||
            ((volatile uint8_t *)destination)[byte] != stage_buffer[byte]) {
          ((volatile uint8_t *)destination)[byte] = stage_buffer[byte];
          ++repaired_bytes;
        }
      }
      copied += (size_t)received;
      (void)sched_yield();
      if (copied >= next_report || copied == copy_end) {
        fprintf(stderr, "\rVerifying/restoring Bonsai weights from SD: %3zu%%",
                copied * 100u / stage_bytes);
        next_report = copied + 16u * 1024u * 1024u;
      }
    }
    free(stage_buffer);
    accelerator_fence();
    record_stamp = 1;
    fprintf(stderr, " (%zu bytes repaired)\n", repaired_bytes);
    }
  }
  const uint8_t *bytes = mapping;
  uint8_t header[4] = {bytes[0], bytes[1], bytes[2], bytes[3]};
  int packed = header[0] == 'B' && header[1] == 'G' &&
               header[2] == 'U' && header[3] == 'F';
  int valid = packed || (header[0] == 'G' && header[1] == 'G' &&
                         header[2] == 'U' && header[3] == 'F');
  accelerator_packed_model = packed;
  if (valid && record_stamp) {
    stamp = open("/run/fpga-model.stamp",
                 O_WRONLY | O_CREAT | O_TRUNC, 0600);
    if (stamp >= 0) {
      (void)write(stamp, &expected, sizeof(expected));
      (void)close(stamp);
    } else {
      perror("Bonsai model stamp");
    }
  }
  (void)munmap(mapping, mapping_bytes);
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
  /* The platform synchronizes accelerator status[31:8] into GPIO input and
     uses input[7:0] for the physical switches.  Bit 26 is the generation
     marker for the packed-model/float-output/transform-capable gateware; do
     not mistake switch bits 0..2 for accelerator feature flags. */
  /* This gateware converts every Q1 accumulator to FP32 before writing the
     low output word.  The generation bit describes that output protocol as
     well as packed-weight support; it is not conditional on which weight
     layout the current model uses. */
  accelerator_float_output =
      (accelerator_gpio[2] & BONSAI_STATUS_PACKED_GENERATION) != 0u;
  /* Bits 25:8 report the winning token after an argmax command, so the
     attention feature bits occupying the top of that field are not stable
     across runtime invocations.  The packed-generation marker is invariant
     and identifies this transform/attention-capable gateware generation. */
  accelerator_attention_available = accelerator_float_output;
  if (getenv("BONSAI_Q30_OUTPUT") != NULL)
    accelerator_float_output = 0;
  if (accelerator_packed_model &&
      (accelerator_gpio[2] & BONSAI_STATUS_PACKED_GENERATION) == 0u) {
    fprintf(stderr,
            "Bonsai FPGA acceleration disabled: packed model unsupported\n");
    return;
  }
  /* The GPIO output register can retain a command toggle across a soft FPGA
     reload even though the accelerator synchronizer resets to zero. Drive
     both phases once so software and hardware start from a known phase. */
  accelerator_gpio[0] = UINT32_C(0xae000000);
  accelerator_fence();
  (void)usleep(100u);
  accelerator_gpio[0] = UINT32_C(0x2e000000);
  accelerator_fence();
  (void)usleep(100u);
  accelerator_toggle = 0u;
  initialize_cached_input();
  initialize_cached_output();
  if (profile_enabled)
    fprintf(stderr, "Bonsai FPGA input: %s address=0x%08x output=%s\n",
            accelerator_input_absolute ? "cached" : "uncached",
            (uint32_t)(accelerator_input_absolute ?
                accelerator_input_words << 3 : BONSAI_SCRATCH_PHYS),
            accelerator_output_cached_ring ? "cached-ring" : "uncached");
  accelerator_available = 1;
}

volatile uint64_t *bonsai_accelerator_packed_input(void) {
  if (!accelerator_initialized) initialize_accelerator();
  const char *packed_input = getenv("BONSAI_PACKED_INPUT");
  if (packed_input != NULL && strcmp(packed_input, "0") == 0) return NULL;
  return (volatile uint64_t *)accelerator_input;
}

void bonsai_accelerator_mark_packed_input(const float *input,
                                          uint32_t columns) {
  packed_input_pointer = input;
  packed_input_columns = columns;
  packed_input_probe[0] = input[0];
  packed_input_probe[1] = input[columns / 2u];
  packed_input_probe[2] = input[columns - 1u];
  accelerator_fence();
}

int bonsai_accelerator_prepare(void) {
  lock_accelerator();
  if (!accelerator_initialized)
    initialize_accelerator();
  int available = accelerator_available;
  unlock_accelerator();
  return available;
}

int bonsai_accelerator_has_float_output(void) {
  if (!accelerator_initialized) initialize_accelerator();
  return accelerator_float_output;
}

const uint8_t *bonsai_accelerator_map_staged_model(size_t *size_out) {
  if (!accelerator_initialized) initialize_accelerator();
  if (!accelerator_available) return NULL;

  /* A non-O_SYNC /dev/mem mapping is cacheable on this CVA6 Linux port.
     The model slot is immutable while this process holds model_slot_lock, so
     CPU metadata, norms, and embedding rows can safely use the fast alias
     while the accelerator reads the same physical DDR through AXI. */
  int model_flags = O_RDWR;
  const char *uncached_model = getenv("BONSAI_UNCACHED_MODEL");
  if (uncached_model != NULL && strcmp(uncached_model, "0") != 0)
    model_flags |= O_SYNC;
  int memory = open("/dev/mem", model_flags);
  if (memory < 0) return NULL;
  size_t model_size = (size_t)BONSAI_MODEL_FILE_SIZE;
  int full_mapping = getenv("BONSAI_FULL_MODEL_MAPPING") != NULL ||
                     getenv("BONSAI_TRACE") != NULL;
  void *mapping;
  if (full_mapping) {
    mapping = mmap(NULL, model_size, PROT_READ, MAP_SHARED, memory,
                   (off_t)BONSAI_MODEL_PHYS);
  } else {
    /* Keep tensor pointers equal to base + GGUF offset, but avoid creating
       device PTEs for the roughly 200 MB of accelerator-only Q1 weights.
       CVA6's /dev/mem mapper eagerly constructs those PTEs and previously
       spent about eleven seconds here on every fresh runtime. */
    mapping = mmap(NULL, model_size, PROT_NONE,
                   MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (mapping != MAP_FAILED) {
      long page_value = sysconf(_SC_PAGESIZE);
      size_t page = page_value > 0 ? (size_t)page_value : 4096u;
      const size_t page_mask = page - 1u;
      /* The header supplies the packed-model magic consumed by model_init. */
      if (mmap(mapping, page, PROT_READ, MAP_SHARED | MAP_FIXED, memory,
               (off_t)BONSAI_MODEL_PHYS) == MAP_FAILED) {
        (void)munmap(mapping, model_size);
        mapping = MAP_FAILED;
      }
      for (size_t i = 0u;
           mapping != MAP_FAILED && i < BONSAI_MODEL_ENTRY_COUNT; ++i) {
        const struct bonsai_model_entry *entry = &bonsai_model_entries[i];
        size_t bytes = 0u;
        if (entry->type == 0u) {
          bytes = (size_t)entry->ne0 * entry->ne1 * sizeof(float);
        } else if (strcmp(entry->name, "token_embd.weight") == 0) {
          bytes = (size_t)(entry->ne0 / 128u) * 18u * entry->ne1;
        }
        if (bytes == 0u) continue;
        size_t begin = (size_t)entry->offset & ~page_mask;
        size_t end = ((size_t)entry->offset + bytes + page_mask) & ~page_mask;
        if (end > model_size) {
          (void)munmap(mapping, model_size);
          mapping = MAP_FAILED;
          break;
        }
        void *address = (uint8_t *)mapping + begin;
        if (mmap(address, end - begin, PROT_READ, MAP_SHARED | MAP_FIXED,
                 memory, (off_t)(BONSAI_MODEL_PHYS + begin)) == MAP_FAILED) {
          (void)munmap(mapping, model_size);
          mapping = MAP_FAILED;
          break;
        }
      }
    }
  }
  (void)close(memory);
  if (mapping == MAP_FAILED) return NULL;
  model_mapping_base = (uintptr_t)mapping;
  model_mapping_base_valid = 1;
  *size_out = model_size;
  return (const uint8_t *)mapping;
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
  (void)accelerator_gpio[0];
  /* Each register write is distinguished only by the toggle bit.  Wait until
     the DDR-domain synchronizer publishes the same phase in status bit 27;
     this is the acknowledgement that makes back-to-back writes lossless. */
  accelerator_fence();
  uint32_t expected = accelerator_toggle != 0u ?
      BONSAI_STATUS_COMMAND_ACK : 0u;
  while ((accelerator_gpio[2] & BONSAI_STATUS_COMMAND_ACK) != expected)
    __asm__ volatile("nop");
}

static int start_command(uint32_t mode, uint32_t old_done) {
  (void)old_done;
  /* write_register waits for the DDR-domain command acknowledgement.  Once
     that phase matches, retransmitting register 5 is actively unsafe: the
     first launch may already have left IDLE, and a duplicate launch then
     raises the accelerator's busy-command error. */
  write_register(5u, mode);
  return 1;
}

static void report_accelerator_error(const char *operation, uint32_t status) {
  uint32_t diagnostic = (status >> 8) & UINT32_C(0x0003ffff);
  fprintf(stderr,
          "Bonsai FPGA %s error: status=%08x outer_state=%u "
          "packer_state=%u reader_busy=%u reader_done=%u ffn_busy=%u "
          "ffn_done=%u group_drain=%u block_valid=%u writer_active=%u "
          "writes_outstanding=%u\n",
          operation, status, (diagnostic >> 12) & 0x3fu,
          (diagnostic >> 8) & 0x0fu, (diagnostic >> 7) & 1u,
          (diagnostic >> 6) & 1u, (diagnostic >> 5) & 1u,
          (diagnostic >> 4) & 1u, (diagnostic >> 3) & 1u,
          (diagnostic >> 2) & 1u, (diagnostic >> 1) & 1u,
          diagnostic & 1u);
  fflush(stderr);
}

static void copy_q8_to_scratch(const struct block_q8_0 *input,
                               uint32_t blocks) {
  /* blocks is a multiple of four, so the 34-byte-per-block total is a
     multiple of eight and 64-bit uncached stores need no tail. */
  const uint8_t *source = (const uint8_t *)input;
  uint32_t bytes = blocks * (uint32_t)sizeof(*input);
  volatile uint64_t *destination =
      (volatile uint64_t *)accelerator_input;
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
  if (!start_command(3u, old_done)) return -1;
  uint32_t status;
  do {
    status = accelerator_gpio[2];
  } while ((status & BONSAI_STATUS_DONE) == old_done);
  accelerator_fence();
  if ((status & BONSAI_STATUS_ERROR) != 0u)
    report_accelerator_error("tensor", status);
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
  uint64_t output_capacity = OUTPUT_SLOT_BYTES / sizeof(int64_t);
  if (mode == 3u && output_first == 0u &&
      rows <= (BONSAI_SCRATCH_BYTES - BONSAI_OUTPUT_LOCAL) / sizeof(int64_t))
    output_capacity =
        (BONSAI_SCRATCH_BYTES - BONSAI_OUTPUT_LOCAL) / sizeof(int64_t);
  if (mode != 4u && mode != 6u &&
      (uint64_t)output_first + rows > output_capacity)
    return 0;
  const char *compact_env = getenv("BONSAI_COMPACT_OUTPUT");
  int compact_output = compact_env != NULL &&
      strcmp(compact_env, "1") == 0 && mode == 3u && rows <= 6144u &&
      (rows & 1u) == 0u && (output_first & 1u) == 0u;
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
  if (tensor == NULL || first_row >= tensor->ne1 ||
      rows > tensor->ne1 - first_row) {
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
  int cpu_packed_input = 0;
  active_profile_shape = profile_enabled ?
      find_profile_shape(columns, rows) : NULL;
  /* Consecutive tensors of one graph layer (attn q/k/v, ffn up/gate) share
     the same activation vector; skip requantizing and re-copying it. The
     first float is compared in addition to the pointer so a buffer reused
     with new contents cannot alias a stale packing. */
  if (packed_input_pointer != input || packed_input_columns != columns ||
      packed_input_probe[0] != input[0] ||
      packed_input_probe[1] != input[columns / 2u] ||
      packed_input_probe[2] != input[columns - 1u]) {
    const char *reference_pack = getenv("BONSAI_REFERENCE_PACK");
    if (bonsai_quantize_q8_to_scratch != NULL &&
        (reference_pack == NULL || strcmp(reference_pack, "0") == 0)) {
      bonsai_quantize_q8_to_scratch(
          input, (volatile uint64_t *)accelerator_input, columns);
    } else {
      quantize_row_q8_0(input, quantized_input, columns);
      if (getenv("BONSAI_MATVEC_TRACE") != NULL) {
        uint64_t local_fingerprint = UINT64_C(0xcbf29ce484222325);
        const uint64_t *local_words = (const uint64_t *)quantized_input;
        uint32_t local_word_count = (columns / 128u) * 17u;
        for (uint32_t word = 0; word < local_word_count; ++word) {
          local_fingerprint ^= local_words[word];
          local_fingerprint *= UINT64_C(0x100000001b3);
        }
        fprintf(stderr, "Bonsai local packed hash=%016llx\n",
                (unsigned long long)local_fingerprint);
      }
      copy_q8_to_scratch(quantized_input, columns / 32u);
    }
    accelerator_fence();
    cpu_packed_input = 1;
    packed_input_pointer = input;
    packed_input_columns = columns;
    packed_input_probe[0] = input[0];
    packed_input_probe[1] = input[columns / 2u];
    packed_input_probe[2] = input[columns - 1u];
  }
  if (profile_enabled) {
    uint64_t elapsed = profile_now() - pack_start;
    profile_pack_ns += elapsed;
    if (active_profile_shape != NULL) active_profile_shape->pack_ns += elapsed;
  }
  /* The custom CPU/DDR bridge can acknowledge an uncached FENCE while its
     final scratch stores are still draining. Read both ends back through
     the CPU path before handing exclusive DDR ownership to the accelerator. */
  uint32_t packed_words = (columns / 128u) * 17u;
  const volatile uint64_t *drain_input =
      (const volatile uint64_t *)accelerator_input;
  if (cpu_packed_input) {
    volatile uint64_t drain_value = 0u;
    for (uint32_t word = 0; word < packed_words; ++word)
      drain_value ^= drain_input[word];
    (void)drain_value;
    accelerator_fence();
  }
  int trace_matvec = getenv("BONSAI_MATVEC_TRACE") != NULL &&
                     mode != 4u && mode != 6u;
  uint64_t source_fingerprint = UINT64_C(0xcbf29ce484222325);
  uint64_t input_fingerprint = UINT64_C(0xcbf29ce484222325);
  if (trace_matvec) {
    for (uint32_t item = 0; item < columns; ++item) {
      uint32_t bits;
      memcpy(&bits, input + item, sizeof(bits));
      source_fingerprint ^= bits;
      source_fingerprint *= UINT64_C(0x100000001b3);
    }
    for (uint32_t word = 0; word < packed_words; ++word) {
      input_fingerprint ^= drain_input[word];
      input_fingerprint *= UINT64_C(0x100000001b3);
    }
  }
  uint64_t hardware_start = profile_enabled ? profile_now() : 0u;
  if (accelerator_packed_model && mode >= 3u && mode <= 4u) mode += 2u;
  uint64_t row_bytes = (uint64_t)(columns / 128u) * 18u;
  uint32_t old_done = accelerator_gpio[2] & BONSAI_STATUS_DONE;
  write_register(0u, (uint32_t)((tensor->offset + first_row * row_bytes) >> 3));
  write_register(1u, accelerator_input_words & UINT32_C(0x01ffffff));
  uint32_t output_words = BONSAI_OUTPUT_OFFSET >> 3;
  int command_output_absolute = accelerator_output_absolute;
  if (forced_output_active) {
    output_words = forced_output_words;
    active_output = forced_output_pointer;
    command_output_absolute = 1;
    /* The residual command streams the compact matvec output immediately
       followed by the resident FP32 vector: 8 KiB + 8 KiB. */
    compact_output = 1;
  } else if (accelerator_output_cached_ring) {
    if (output_first == 0u)
      cached_output_slot = (cached_output_slot + 1u) % OUTPUT_SLOT_COUNT;
    output_words = cached_output_words +
        cached_output_slot * (OUTPUT_SLOT_BYTES / 8u);
    active_output = (const volatile int64_t *)(cached_output_buffer +
        cached_output_slot * OUTPUT_SLOT_BYTES);
  } else {
    /* A fused residual command temporarily redirects active_output to its
       resident norm slot. Restore the ordinary uncached scratch view before
       every subsequent command; the cacheable ring path does this naturally
       when it selects its next slot. */
    active_output = (const volatile int64_t *)(
        accelerator_scratch + BONSAI_OUTPUT_LOCAL);
  }
  active_output_words = output_words;
  write_register(6u, (accelerator_input_words >> 25) |
                      (accelerator_input_absolute ? 4u : 0u) |
                      ((output_words >> 25) << 3) |
                      (command_output_absolute ? 32u : 0u) |
                      (compact_output ? 64u : 0u));
  write_register(2u, (output_words & UINT32_C(0x01ffffff)) +
                      (compact_output ? output_first / 2u : output_first));
  write_register(3u, columns);
  write_register(4u, rows);
  if (!start_command(mode, old_done)) {
    fprintf(stderr, "Bonsai FPGA acceleration disabled: start command lost\n");
    accelerator_available = 0;
    unlock_accelerator();
    return 0;
  }
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
              accelerator_input[0], accelerator_input[1],
              accelerator_input[2], accelerator_input[3],
              accelerator_input[4], accelerator_input[5],
              accelerator_input[6], accelerator_input[7],
              (unsigned)*(const uint16_t *)quantized_input);
      fflush(stderr);
    }
  }
  debug_step("submit: command written, polling");
  uint32_t status;
  do {
    status = accelerator_gpio[2];
  } while ((status & BONSAI_STATUS_DONE) == old_done);
  if (debug_enabled > 0) {
    static int dumped2;
    if (!dumped2) {
      dumped2 = 1;
      const volatile int64_t *raw = active_output;
      fprintf(stderr, "[drv] status=%08x raw[0..3]=%lld %lld %lld %lld\n",
              status, (long long)raw[0], (long long)raw[1],
              (long long)raw[2], (long long)raw[3]);
      fflush(stderr);
    }
  }
  debug_step("submit: done observed");
  if (profile_enabled) {
    uint64_t elapsed = profile_now() - hardware_start;
    profile_hardware_ns += elapsed;
    if (active_profile_shape != NULL)
      active_profile_shape->hardware_ns += elapsed;
  }
  if ((status & BONSAI_STATUS_ERROR) != 0u) {
    /* A DDR read response can fail transiently under concurrent CVA6/SD
       traffic.  The command is idempotent: retry it with the same retained
       registers before disabling acceleration. */
    int recovered = 0;
    for (uint32_t retry = 0; retry < 2u && !recovered; ++retry) {
      report_accelerator_error("matvec", status);
      fprintf(stderr,
              "Bonsai FPGA matvec retry: mode=%u shape=%ux%u retry=%u\n",
              mode, columns, rows, retry + 1u);
      old_done = status & BONSAI_STATUS_DONE;
      if (!start_command(mode, old_done)) break;
      do {
        status = accelerator_gpio[2];
      } while ((status & BONSAI_STATUS_DONE) == old_done);
      recovered = (status & BONSAI_STATUS_ERROR) == 0u;
    }
    if (!recovered) {
      accelerator_available = 0;
      fprintf(stderr, "Bonsai FPGA acceleration disabled: command failed\n");
      unlock_accelerator();
      return 0;
    }
  }

  if (trace_matvec) {
    static uint32_t trace_call;
    uint64_t fingerprint = UINT64_C(0xcbf29ce484222325);
    const volatile int64_t *raw = active_output +
        (compact_output ? output_first / 2u : output_first);
    uint32_t first_bits = 0u;
    for (uint32_t row = 0; row < rows; ++row) {
      uint32_t bits = compact_output ?
          ((const volatile uint32_t *)raw)[row] : (uint32_t)raw[row];
      if (row == 0u) first_bits = bits;
      fingerprint ^= bits;
      fingerprint *= UINT64_C(0x100000001b3);
    }
    fprintf(stderr,
            "Bonsai matvec %u tensor=%llu shape=%ux%u output=%u "
            "source=%016llx in=%016llx first=%016llx hash=%016llx\n",
            trace_call++, (unsigned long long)tensor->offset, columns, rows,
            output_first, (unsigned long long)source_fingerprint,
            (unsigned long long)input_fingerprint,
            (unsigned long long)first_bits,
            (unsigned long long)fingerprint);
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
  accelerator_output_compact = compact_output;
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
                  (accelerator_gpio[2] & BONSAI_STATUS_PACKED_GENERATION) != 0u;
  const char *hardware_argmax = getenv("BONSAI_HW_ARGMAX");
  if (hardware_argmax != NULL && strcmp(hardware_argmax, "0") == 0)
    supported = 0;
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

static inline float s64_q30_to_float(int64_t value) {
  if (accelerator_float_output) {
    uint32_t bits = (uint32_t)value;
    float result;
    memcpy(&result, &bits, sizeof(result));
    return result;
  }
  /* This FPU's 64-bit integer converts (fcvt.s.l) are unreliable, so build
     the exact Q30 value from 32-bit halves, which convert correctly. */
  int32_t high = (int32_t)(value >> 32);
  uint32_t low = (uint32_t)value;
  return (float)high * 4.0f + (float)low * (1.0f / 1073741824.0f);
}

static inline float accelerator_output_at(const volatile int64_t *output,
                                          uint32_t row) {
  if (accelerator_float_output) {
    uint32_t bits = accelerator_output_compact ?
        ((const volatile uint32_t *)output)[row] : (uint32_t)output[row];
    float result;
    memcpy(&result, &bits, sizeof(result));
    return result;
  }
  return s64_q30_to_float(output[row]);
}

void bonsai_accelerate_read_rows(float *output, uint32_t first,
                                 uint32_t count) {
  const volatile int64_t *hardware_output =
      active_output + (accelerator_output_compact ?
                       submit_output_first / 2u : submit_output_first);
  for (uint32_t row = first; row < first + count; ++row)
    output[row] = accelerator_output_at(hardware_output, row);
}

float bonsai_accelerate_read_rows_add_sum(float *residual, uint32_t first,
                                          uint32_t count) {
  const volatile int64_t *hardware_output =
      active_output + (accelerator_output_compact ?
                       submit_output_first / 2u : submit_output_first);
  float sum = 0.0f;
  for (uint32_t row = first; row < first + count; ++row) {
    float value = residual[row] + accelerator_output_at(hardware_output, row);
    residual[row] = value;
    sum += value * value;
  }
  return sum;
}

void bonsai_accelerate_read_stored_rows(float *output, uint32_t source_first,
                                        uint32_t count) {
  const volatile int64_t *hardware_output =
      active_output + (accelerator_output_compact ?
                       source_first / 2u : source_first);
  for (uint32_t row = 0; row < count; ++row)
    output[row] = accelerator_output_at(hardware_output, row);
}

const volatile int64_t *bonsai_accelerate_stored_output(void) {
  return active_output;
}

int bonsai_accelerator_has_compact_output(void) {
  return accelerator_output_compact;
}

int bonsai_accelerator_has_attention(void) {
  if (!accelerator_initialized) initialize_accelerator();
  const char *hardware_attention = getenv("BONSAI_HW_ATTENTION");
  return accelerator_available && accelerator_attention_available &&
      (hardware_attention == NULL || strcmp(hardware_attention, "0") != 0);
}

void bonsai_accelerator_output_fence(void) {
  /* DONE follows every AXI write response. Order that device read before
     cacheable ring reads; the gateware's write-through configuration drains a
     FENCE without performing a whole-cache flush. */
#ifdef __riscv
  __asm__ volatile("fence i,r" ::: "memory");
#else
  __atomic_thread_fence(__ATOMIC_ACQUIRE);
#endif
}

int bonsai_accelerator_pack_ffn(uint32_t elements) {
  if (elements != 6144u) return 0;
  const char *hardware_transforms = getenv("BONSAI_HW_TRANSFORMS");
  const char *hardware_ffn = getenv("BONSAI_HW_FFN");
  if ((hardware_ffn != NULL && strcmp(hardware_ffn, "0") == 0) ||
      ((hardware_ffn == NULL || strcmp(hardware_ffn, "1") != 0) &&
       (hardware_transforms == NULL ||
        strcmp(hardware_transforms, "1") != 0)))
    return 0;
  uint64_t profile_start = profile_enabled ? profile_now() : 0u;
  lock_accelerator();
  if (!accelerator_initialized) initialize_accelerator();
  if (!accelerator_available ||
      (accelerator_gpio[2] & BONSAI_STATUS_PACKED_GENERATION) == 0u) {
    unlock_accelerator();
    return 0;
  }

  uint32_t source_words = BONSAI_OUTPUT_OFFSET >> 3;
  int source_absolute = 0;
  if (accelerator_output_cached_ring) {
    source_words = cached_output_words +
        cached_output_slot * (OUTPUT_SLOT_BYTES / 8u);
    source_absolute = accelerator_output_absolute;
  }
  uint32_t destination_words = accelerator_input_words;
  uint32_t old_done = accelerator_gpio[2] & BONSAI_STATUS_DONE;
  write_register(1u, source_words & UINT32_C(0x01ffffff));
  write_register(2u, destination_words & UINT32_C(0x01ffffff));
  write_register(6u, (source_words >> 25) |
                      (source_absolute ? 4u : 0u) |
                      ((destination_words >> 25) << 3) |
                      (accelerator_input_absolute ? 32u : 0u) |
                      (accelerator_output_compact ? 64u : 0u));
  write_register(3u, elements);
  write_register(4u, 1u);
  if (!start_command(7u, old_done)) {
    accelerator_available = 0;
    unlock_accelerator();
    return 0;
  }
  uint32_t status;
  do {
    status = accelerator_gpio[2];
  } while ((status & BONSAI_STATUS_DONE) == old_done);
  accelerator_fence();
  if ((status & BONSAI_STATUS_ERROR) != 0u) {
    accelerator_available = 0;
    report_accelerator_error("FFN packing", status);
    unlock_accelerator();
    return 0;
  }
  if (profile_enabled) {
    profile_ffn_pack_ns += profile_now() - profile_start;
    ++profile_ffn_pack_calls;
  }
  unlock_accelerator();
  return 1;
}

static volatile uint64_t *norm_source_for_weight(const float *weight,
                                                  uint32_t elements,
                                                  uint32_t *slot_out) {
  uint32_t slot = 0u;
  while (slot < norm_slot_count && norm_slot_weights[slot] != weight) ++slot;
  int cached_weight = slot < norm_slot_count;
  if (!cached_weight && norm_slot_count < BONSAI_NORM_SLOT_COUNT) {
    slot = norm_slot_count++;
    norm_slot_weights[slot] = weight;
  }
  uint32_t source_offset = slot < BONSAI_NORM_SLOT_COUNT ?
      BONSAI_NORM_SLOT_FALLBACK_BASE + slot * BONSAI_NORM_SLOT_BYTES +
          BONSAI_NORM_RESIDUAL_LOCAL :
      BONSAI_NORM_SOURCE_LOCAL;
  volatile uint64_t *source =
      (volatile uint64_t *)(accelerator_scratch + source_offset);
  if (!cached_weight) {
    for (uint32_t i = 0u; i < elements; i += 2u) {
      uint64_t pair;
      memcpy(&pair, weight + i, sizeof(pair));
      source[elements / 2u + i / 2u] = pair;
    }
  }
  if (slot_out != NULL) *slot_out = slot;
  return source;
}

int bonsai_accelerator_pack_norm(const float *input, const float *weight,
                                 float scale, uint32_t elements) {
  if (elements != 2048u) return 0;
  const char *hardware_transforms = getenv("BONSAI_HW_TRANSFORMS");
  const char *hardware_norm = getenv("BONSAI_HW_NORM");
  if ((hardware_norm != NULL && strcmp(hardware_norm, "0") == 0) ||
      ((hardware_norm == NULL || strcmp(hardware_norm, "1") != 0) &&
       (hardware_transforms == NULL ||
        strcmp(hardware_transforms, "1") != 0)))
    return 0;
  uint64_t profile_start = profile_enabled ? profile_now() : 0u;
  lock_accelerator();
  if (!accelerator_initialized) initialize_accelerator();
  if (!accelerator_available ||
      (accelerator_gpio[2] & BONSAI_STATUS_PACKED_GENERATION) == 0u) {
    unlock_accelerator();
    return 0;
  }

  uint32_t slot;
  volatile uint64_t *source = norm_source_for_weight(weight, elements, &slot);
  for (uint32_t i = 0; i < elements; i += 2u) {
    uint64_t pair;
    memcpy(&pair, input + i, sizeof(pair));
    source[i / 2u] = pair;
  }
  accelerator_fence();

  uint32_t scale_bits;
  memcpy(&scale_bits, &scale, sizeof(scale_bits));
  uint32_t source_words = BONSAI_INPUT_OFFSET / 8u +
      (uint32_t)((const volatile uint8_t *)source - accelerator_scratch) / 8u;
  uint32_t old_done = accelerator_gpio[2] & BONSAI_STATUS_DONE;
  write_register(1u, source_words & UINT32_C(0x01ffffff));
  write_register(2u, accelerator_input_words & UINT32_C(0x01ffffff));
  write_register(6u, (source_words >> 25) |
                      ((accelerator_input_words >> 25) << 3) |
                      (accelerator_input_absolute ? 32u : 0u) |
                      ((scale_bits >> 25) << 7));
  write_register(7u, scale_bits & UINT32_C(0x01ffffff));
  write_register(3u, elements);
  write_register(4u, 2u);
  if (!start_command(7u, old_done)) {
    accelerator_available = 0;
    unlock_accelerator();
    return 0;
  }
  uint32_t status;
  do {
    status = accelerator_gpio[2];
  } while ((status & BONSAI_STATUS_DONE) == old_done);
  accelerator_fence();
  if ((status & BONSAI_STATUS_ERROR) != 0u) {
    accelerator_available = 0;
    report_accelerator_error("norm packing", status);
    unlock_accelerator();
    return 0;
  }
  if (profile_enabled) {
    profile_norm_pack_ns += profile_now() - profile_start;
    ++profile_norm_pack_calls;
  }
  if (slot < BONSAI_NORM_SLOT_COUNT) {
    resident_norm_source = source;
    resident_norm_weight = weight;
    resident_norm_valid = 1;
  } else {
    resident_norm_valid = 0;
  }
  unlock_accelerator();
  return 1;
}

int bonsai_accelerator_pack_resident_norm(const float *weight, float scale,
                                          uint32_t elements) {
  if (elements != 2048u || !resident_norm_valid ||
      resident_norm_weight != weight)
    return 0;
  const char *hardware_residual = getenv("BONSAI_HW_RESIDUAL");
  if (hardware_residual == NULL || strcmp(hardware_residual, "1") != 0)
    return 0;
  uint64_t profile_start = profile_enabled ? profile_now() : 0u;
  lock_accelerator();
  if (!accelerator_initialized) initialize_accelerator();
  if (!accelerator_available ||
      (accelerator_gpio[2] & BONSAI_STATUS_PACKED_GENERATION) == 0u) {
    unlock_accelerator();
    return 0;
  }

  accelerator_fence();
  uint32_t scale_bits;
  memcpy(&scale_bits, &scale, sizeof(scale_bits));
  uint32_t source_words = BONSAI_INPUT_OFFSET / 8u +
      (uint32_t)((const volatile uint8_t *)resident_norm_source -
                 accelerator_scratch) / 8u;
  uint32_t old_done = accelerator_gpio[2] & BONSAI_STATUS_DONE;
  write_register(1u, source_words & UINT32_C(0x01ffffff));
  write_register(2u, accelerator_input_words & UINT32_C(0x01ffffff));
  write_register(6u, (source_words >> 25) |
                      ((accelerator_input_words >> 25) << 3) |
                      (accelerator_input_absolute ? 32u : 0u) |
                      ((scale_bits >> 25) << 7));
  write_register(7u, scale_bits & UINT32_C(0x01ffffff));
  write_register(3u, elements);
  write_register(4u, 2u);
  if (!start_command(7u, old_done)) {
    accelerator_available = 0;
    unlock_accelerator();
    return 0;
  }
  uint32_t status;
  do {
    status = accelerator_gpio[2];
  } while ((status & BONSAI_STATUS_DONE) == old_done);
  accelerator_fence();
  if ((status & BONSAI_STATUS_ERROR) != 0u) {
    accelerator_available = 0;
    report_accelerator_error("resident norm packing", status);
    unlock_accelerator();
    return 0;
  }
  if (profile_enabled) {
    profile_norm_pack_ns += profile_now() - profile_start;
    ++profile_norm_pack_calls;
  }
  unlock_accelerator();
  return 1;
}

int bonsai_accelerate_f32_matvec_residual(
    const void *weights, uint32_t columns, uint32_t rows, const float *input,
    const float *next_norm_weight, float *sum_out) {
  static uint32_t residual_trace_matvec[2048];
  static uint32_t residual_trace_residual[2048];
  static int residual_trace_done;
  const char *hardware_residual = getenv("BONSAI_HW_RESIDUAL");
  if (hardware_residual == NULL || strcmp(hardware_residual, "1") != 0 ||
      rows != 2048u || !resident_norm_valid)
    return 0;

  uint32_t next_slot;
  volatile uint64_t *next_source =
      norm_source_for_weight(next_norm_weight, rows, &next_slot);
  if (next_slot >= BONSAI_NORM_SLOT_COUNT) return 0;
  /* A newly encountered norm writes 8 KiB of weights through the uncached
     scratch mapping. Retire those stores before the matvec takes exclusive
     DDR ownership; the resident norm command will consume them immediately
     after the fused residual command. */
  accelerator_fence();

  uintptr_t current_residual_offset =
      (uintptr_t)((const volatile uint8_t *)resident_norm_source -
                  accelerator_scratch);
  if (current_residual_offset < BONSAI_NORM_RESIDUAL_LOCAL) return 0;
  uintptr_t current_output_offset =
      current_residual_offset - BONSAI_NORM_RESIDUAL_LOCAL;
  int residual_trace = getenv("BONSAI_HW_RESIDUAL_TRACE") != NULL &&
      !residual_trace_done;
  if (residual_trace) {
    const volatile uint32_t *source =
        (const volatile uint32_t *)resident_norm_source;
    for (uint32_t index = 0u; index < rows; ++index)
      residual_trace_residual[index] = source[index];
  }
  forced_output_words = (uint32_t)(
      ((BONSAI_SCRATCH_PHYS + current_output_offset) &
       UINT64_C(0x3fffffff)) >> 3);
  forced_output_pointer =
      (const volatile int64_t *)(accelerator_scratch + current_output_offset);
  forced_output_active = 1;
  int submitted = bonsai_accelerate_f32_matvec_submit_at(
      weights, columns, rows, input, 0u, 3u);
  forced_output_active = 0;
  if (!submitted) return 0;
  if (residual_trace) {
    const volatile uint32_t *source = (const volatile uint32_t *)(
        accelerator_scratch + current_output_offset);
    bonsai_accelerator_output_fence();
    for (uint32_t index = 0u; index < rows; ++index)
      residual_trace_matvec[index] = source[index];
  }

  uint32_t input_words = (uint32_t)(
      ((BONSAI_SCRATCH_PHYS + current_output_offset) &
       UINT64_C(0x3fffffff)) >> 3);
  uintptr_t next_residual_offset =
      (uintptr_t)((const volatile uint8_t *)next_source -
                  accelerator_scratch);
  uint32_t output_words = (uint32_t)(
      ((BONSAI_SCRATCH_PHYS + next_residual_offset) &
       UINT64_C(0x3fffffff)) >> 3);
  uint32_t old_done = accelerator_gpio[2] & BONSAI_STATUS_DONE;
  write_register(1u, input_words & UINT32_C(0x01ffffff));
  write_register(2u, output_words & UINT32_C(0x01ffffff));
  uint32_t address_config = (input_words >> 25) | 4u |
                            ((output_words >> 25) << 3) | 32u;
  write_register(6u, address_config);
  write_register(3u, rows);
  write_register(4u, 5u);
  if (!start_command(7u, old_done)) {
    accelerator_available = 0;
    resident_norm_valid = 0;
    bonsai_accelerate_release();
    return 0;
  }
  uint32_t status;
  do {
    status = accelerator_gpio[2];
  } while ((status & BONSAI_STATUS_DONE) == old_done);
  accelerator_fence();
  if ((status & BONSAI_STATUS_ERROR) != 0u) {
    accelerator_available = 0;
    resident_norm_valid = 0;
    report_accelerator_error("residual add/sum", status);
    bonsai_accelerate_release();
    return 0;
  }

  // The low 18 bits are already present in the completion status. Select the
  // upper 14 bits through the acknowledged control register and reconstruct
  // the scalar without racing the CPU and accelerator DDR ports.
  uint32_t sum_bits = (status >> 8) & UINT32_C(0x0003ffff);
  write_register(6u, address_config | UINT32_C(0x01000000));
  uint32_t sum_status = accelerator_gpio[2];
  sum_bits |= ((sum_status >> 8) & UINT32_C(0x00003fff)) << 18;
  if (getenv("BONSAI_CHECK_HW_RESIDUAL_SUM") != NULL) {
    float checked_sum = bonsai_reliable_resident_sum_squares(
        (const volatile uint32_t *)next_source, rows);
    uint32_t checked_sum_bits;
    memcpy(&checked_sum_bits, &checked_sum, sizeof(checked_sum_bits));
    fprintf(stderr, "Bonsai initial hardware residual sum=%08x CPU=%08x\n",
            sum_bits, checked_sum_bits);
  }
  if (getenv("BONSAI_REPEAT_HW_RESIDUAL") != NULL) {
    static uint32_t reference[2048];
    const volatile uint32_t *actual =
        (const volatile uint32_t *)next_source;
    for (uint32_t index = 0u; index < rows; ++index)
      reference[index] = actual[index];
    uint32_t reference_sum = *(const volatile uint32_t *)(
        accelerator_scratch + BONSAI_RESIDUAL_SUM_LOCAL);
    for (uint32_t pass = 0u; pass < 1u; ++pass) {
      old_done = status & BONSAI_STATUS_DONE;
      if (!start_command(7u, old_done)) break;
      do {
        status = accelerator_gpio[2];
      } while ((status & BONSAI_STATUS_DONE) == old_done);
      accelerator_fence();
      uint32_t mismatches = 0u;
      uint32_t first = rows;
      for (uint32_t index = 0u; index < rows; ++index) {
        if (actual[index] != reference[index]) {
          if (first == rows) first = index;
          ++mismatches;
        }
      }
      uint32_t observed_sum = *(const volatile uint32_t *)(
          accelerator_scratch + BONSAI_RESIDUAL_SUM_LOCAL);
      if (mismatches != 0u || reference_sum != observed_sum)
        fprintf(stderr,
                "Bonsai repeated hardware residual pass=%u mismatches=%u "
                "first=%u sum=%08x/%08x\n",
                pass + 1u, mismatches, first, reference_sum, observed_sum);
    }
  }

  if (getenv("BONSAI_CPU_RESIDUAL_SUM") != NULL) {
    if (getenv("BONSAI_FAST_CPU_RESIDUAL_SUM") != NULL)
      *sum_out = bonsai_fast_resident_sum_squares(
          (const volatile uint32_t *)next_source, rows);
    else
      *sum_out = bonsai_reliable_resident_sum_squares(
          (const volatile uint32_t *)next_source, rows);
    if (getenv("BONSAI_REPEAT_HW_RESIDUAL") != NULL) {
      uint32_t cpu_sum_bits;
      memcpy(&cpu_sum_bits, sum_out, sizeof(cpu_sum_bits));
      fprintf(stderr, "Bonsai CPU residual sum=%08x hardware=%08x\n",
              cpu_sum_bits, sum_bits);
    }
  } else {
    memcpy(sum_out, &sum_bits, sizeof(*sum_out));
  }
  if (residual_trace) {
      const volatile uint32_t *actual =
          (const volatile uint32_t *)next_source;
      uint32_t mismatches = 0u;
      uint32_t first_mismatch = rows;
      for (uint32_t index = 0u; index < rows; ++index) {
        uint32_t matvec_bits = residual_trace_matvec[index];
        uint32_t residual_bits = residual_trace_residual[index];
        uint32_t actual_bits = actual[index];
        float matvec_value;
        float residual_value;
        float expected_value;
        uint32_t expected_bits;
        memcpy(&matvec_value, &matvec_bits, sizeof(matvec_value));
        memcpy(&residual_value, &residual_bits, sizeof(residual_value));
        expected_value = matvec_value + residual_value;
        memcpy(&expected_bits, &expected_value, sizeof(expected_bits));
        if (expected_bits != actual_bits) {
          if (first_mismatch == rows) first_mismatch = index;
          ++mismatches;
        }
        if (index < 8u)
          fprintf(stderr,
                  "Bonsai residual trace index=%u matvec=%08x residual=%08x "
                  "expected=%08x actual=%08x\n",
                  index, matvec_bits, residual_bits, expected_bits,
                  actual_bits);
      }
      fprintf(stderr,
              "Bonsai residual trace mismatches=%u first=%u sum=%08x\n",
              mismatches, first_mismatch, sum_bits);
      residual_trace_done = 1;
  }
  resident_norm_source = next_source;
  resident_norm_weight = next_norm_weight;
  resident_norm_valid = 1;
  bonsai_accelerate_release();
  return 1;
}

int bonsai_accelerator_pack_vector(const float *input, uint32_t elements) {
  static float unit_weight[2048] __attribute__((aligned(64)));
  static int unit_weight_initialized;
  if (elements != 2048u) return 0;
  const char *hardware_transforms = getenv("BONSAI_HW_TRANSFORMS");
  const char *hardware_norm = getenv("BONSAI_HW_NORM");
  if ((hardware_norm != NULL && strcmp(hardware_norm, "0") == 0) ||
      ((hardware_norm == NULL || strcmp(hardware_norm, "1") != 0) &&
       (hardware_transforms == NULL ||
        strcmp(hardware_transforms, "1") != 0)))
    return 0;
  uint64_t profile_start = profile_enabled ? profile_now() : 0u;
  lock_accelerator();
  if (!accelerator_initialized) initialize_accelerator();
  if (!accelerator_available ||
      (accelerator_gpio[2] & BONSAI_STATUS_PACKED_GENERATION) == 0u) {
    unlock_accelerator();
    return 0;
  }
  if ((accelerator_gpio[2] & BONSAI_STATUS_VECTOR_PACK) == 0u) {
    unlock_accelerator();
    if (!unit_weight_initialized) {
      for (uint32_t i = 0; i < elements; ++i) unit_weight[i] = 1.0f;
      unit_weight_initialized = 1;
    }
    return bonsai_accelerator_pack_norm(input, unit_weight, 1.0f, elements);
  }

  uint64_t *source = (uint64_t *)(accelerator_input +
                                  BONSAI_NORM_SOURCE_LOCAL);
  for (uint32_t i = 0; i < elements; i += 2u) {
    uint64_t pair;
    memcpy(&pair, input + i, sizeof(pair));
    source[i / 2u] = pair;
  }
  accelerator_fence();

  uint32_t source_words = accelerator_input_words +
      BONSAI_NORM_SOURCE_LOCAL / 8u;
  uint32_t old_done = accelerator_gpio[2] & BONSAI_STATUS_DONE;
  write_register(1u, source_words & UINT32_C(0x01ffffff));
  write_register(2u, accelerator_input_words & UINT32_C(0x01ffffff));
  write_register(6u, (source_words >> 25) |
                      (accelerator_input_absolute ? 4u : 0u) |
                      ((accelerator_input_words >> 25) << 3) |
                      (accelerator_input_absolute ? 32u : 0u) |
                      ((UINT32_C(0x3f800000) >> 25) << 7));
  write_register(7u, UINT32_C(0x3f800000) & UINT32_C(0x01ffffff));
  write_register(3u, elements);
  write_register(4u, 3u);
  if (!start_command(7u, old_done)) {
    accelerator_available = 0;
    unlock_accelerator();
    return 0;
  }
  uint32_t status;
  do {
    status = accelerator_gpio[2];
  } while ((status & BONSAI_STATUS_DONE) == old_done);
  accelerator_fence();
  if ((status & BONSAI_STATUS_ERROR) != 0u) {
    accelerator_available = 0;
    report_accelerator_error("vector packing", status);
    unlock_accelerator();
    return 0;
  }
  if (profile_enabled) {
    profile_norm_pack_ns += profile_now() - profile_start;
    ++profile_norm_pack_calls;
  }
  unlock_accelerator();
  return 1;
}

int bonsai_accelerator_attention(uint32_t layer, uint32_t position,
                                 const int32_t *q_weight,
                                 const int32_t *k_weight,
                                 const int32_t *rope_cos,
                                 const int32_t *rope_sin,
                                 float *probe_output) {
  if (!accelerator_initialized) initialize_accelerator();
  if (!accelerator_available || layer >= 28u || position >= 256u ||
      !bonsai_accelerator_has_attention())
    return 0;

  lock_accelerator();
  volatile uint64_t *table = (volatile uint64_t *)(accelerator_scratch +
      BONSAI_ATTENTION_TABLE_LOCAL);
  for (uint32_t i = 0; i < 64u; ++i) {
    uint64_t pair;
    memcpy(&pair, q_weight + i * 2u, sizeof(pair));
    table[i] = pair;
    memcpy(&pair, k_weight + i * 2u, sizeof(pair));
    table[64u + i] = pair;
  }
  for (uint32_t i = 0; i < 32u; ++i) {
    uint64_t pair;
    memcpy(&pair, rope_cos + i * 2u, sizeof(pair));
    table[128u + i] = pair;
    memcpy(&pair, rope_sin + i * 2u, sizeof(pair));
    table[160u + i] = pair;
  }
  accelerator_fence();

  (void)probe_output;
  uint32_t old_done = accelerator_gpio[2] & BONSAI_STATUS_DONE;
  write_register(1u, active_output_words & UINT32_C(0x01ffffff));
  write_register(2u, accelerator_input_words & UINT32_C(0x01ffffff));
  write_register(6u, (active_output_words >> 25) |
                      (accelerator_output_absolute ? 4u : 0u) |
                      ((accelerator_input_words >> 25) << 3) |
                      (accelerator_input_absolute ? 32u : 0u));
  write_register(3u, (layer << 8) | position);
  write_register(4u, 4u);
  if (!start_command(7u, old_done)) {
    accelerator_available = 0;
    unlock_accelerator();
    return 0;
  }
  uint32_t status;
  do {
    status = accelerator_gpio[2];
  } while ((status & BONSAI_STATUS_DONE) == old_done);
  if ((status & BONSAI_STATUS_ERROR) != 0u) {
    report_accelerator_error("attention", status);
    accelerator_available = 0;
    unlock_accelerator();
    return 0;
  }

  accelerator_fence();
  unlock_accelerator();
  return 1;
}

const volatile uint16_t *bonsai_accelerator_attention_kv(
    uint32_t layer, uint32_t position) {
  if (!accelerator_initialized) initialize_accelerator();
  uint64_t offset = BONSAI_ATTENTION_KV_LOCAL +
      (((uint64_t)layer * 256u + position) << 12);
  if (!accelerator_available || layer >= 28u || position >= 256u ||
      offset + 4096u > BONSAI_SCRATCH_BYTES)
    return NULL;
  accelerator_fence();
  return (const volatile uint16_t *)(accelerator_scratch + offset);
}

void bonsai_accelerate_argmax_rows(uint32_t first, uint32_t count,
                                   int64_t *best_raw, uint32_t *best_token) {
  const volatile int64_t *hardware_output =
      active_output + (accelerator_output_compact ?
                       submit_output_first / 2u : submit_output_first);
  int64_t score = INT64_MIN;
  uint32_t token = first;
  for (uint32_t row = first; row < first + count; ++row) {
    int64_t rank;
    if (accelerator_float_output) {
      float value = accelerator_output_at(hardware_output, row);
      uint32_t bits;
      memcpy(&bits, &value, sizeof(bits));
      /* Map finite IEEE-754 values to a signed integer with the same order.
         Negative magnitudes run in the opposite bitwise direction. */
      rank = (bits & UINT32_C(0x80000000)) != 0u ?
          -(int64_t)(bits & UINT32_C(0x7fffffff)) : (int64_t)bits;
    } else {
      rank = hardware_output[row];
    }
    if (rank > score) {
      score = rank;
      token = row;
    }
  }
  *best_raw = score;
  *best_token = token;
}

void bonsai_accelerate_release(void) {
  if (profile_enabled) {
    uint64_t elapsed = profile_now() - submit_done_ns;
    profile_output_ns += elapsed;
    if (active_profile_shape != NULL) {
      active_profile_shape->output_ns += elapsed;
      ++active_profile_shape->calls;
    }
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
      active_output;
  *s = s64_q30_to_float(output[row]);
  unlock_accelerator();
}

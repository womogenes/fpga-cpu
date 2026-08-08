// Included by the standalone LFM runtime after its model and quantization
// types have been declared. The reserved model and scratch regions are outside
// Linux RAM, so /dev/mem maps them uncached and avoids CPU/DMA coherency work.

#include <string.h>

#define LFM_GPIO_PHYS 0x40000000ul
#define LFM_GPIO_BYTES 4096ul
#define LFM_SCRATCH_PHYS 0xbe000000ul
#define LFM_SCRATCH_BYTES (2ul * 1024ul * 1024ul)
#define LFM_MODEL_PHYS 0xaf000000ul
#define LFM_INPUT_OFFSET 0x0f000000u
#define LFM_OUTPUT_OFFSET 0x0f010000u
#define LFM_OUTPUT_LOCAL 0x00010000u
#define LFM_STATUS_PRESENT 0x80000000u
#define LFM_STATUS_BUSY 0x40000000u
#define LFM_STATUS_ERROR 0x20000000u
#define LFM_STATUS_DONE 0x10000000u

static volatile uint32_t *lfm_accel_gpio;
static volatile uint8_t *lfm_accel_scratch;
static uint32_t lfm_accel_toggle;
static uint32_t lfm_accel_loaded_revision = UINT32_MAX;
static const struct block_q8_0 *lfm_accel_pack_q8;
static uint32_t lfm_accel_pack_blocks;
static volatile const int64_t *lfm_accel_raw_output;
static float *lfm_accel_float_output;
static int lfm_accel_profile;
static uint64_t lfm_profile_pack_ns;
static uint64_t lfm_profile_q4_ns;
static uint64_t lfm_profile_output_ns;
static uint64_t lfm_profile_q6_pack_ns;
static uint64_t lfm_profile_q6_ns;
static uint32_t lfm_profile_q4_calls;
#ifdef LFM_ACCELERATOR_VERIFY
static uint32_t lfm_accel_verify_q4_calls;
#endif

static void lfm_accel_fence(void) {
  __asm__ volatile("fence iorw, iorw" ::: "memory");
}

static void lfm_accel_init(void) {
  if (lfm_accel_gpio != NULL) return;
  int memory = open("/dev/mem", O_RDWR | O_SYNC);
  if (memory < 0) fail("accelerator cannot open /dev/mem");
  void *gpio = mmap(NULL, LFM_GPIO_BYTES, PROT_READ | PROT_WRITE,
                    MAP_SHARED, memory, LFM_GPIO_PHYS);
  void *scratch = mmap(NULL, LFM_SCRATCH_BYTES, PROT_READ | PROT_WRITE,
                       MAP_SHARED, memory, LFM_SCRATCH_PHYS);
  (void)close(memory);
  if (gpio == MAP_FAILED || scratch == MAP_FAILED)
    fail("accelerator cannot map registers or scratch DDR");
  lfm_accel_gpio = (volatile uint32_t *)gpio;
  lfm_accel_scratch = (volatile uint8_t *)scratch;
  lfm_accel_gpio[1] = 0u;
  lfm_accel_fence();
  if ((lfm_accel_gpio[2] & LFM_STATUS_PRESENT) == 0u)
    fail("LFM accelerator is absent from this bitstream");
  lfm_accel_toggle = lfm_accel_gpio[0] & 0x80000000u;
  lfm_accel_profile = getenv("LFM_PROFILE") != NULL;
}

static void lfm_accel_register(uint32_t index, uint32_t value) {
  lfm_accel_toggle ^= 0x80000000u;
  lfm_accel_gpio[0] = lfm_accel_toggle | 0x20000000u |
                      ((index & 7u) << 25) | (value & 0x01ffffffu);
  lfm_accel_fence();
  /* The command word crosses from the 100 MHz GPIO clock into the 200 MHz DDR
     clock through a two-flop toggle synchronizer. Keep successive toggle
     edges separated long enough that neither can be hidden by that CDC. */
  for (volatile uint32_t settle = 0; settle < 64u; ++settle)
    __asm__ volatile("nop");
}

static uint32_t lfm_accel_weight_words(const uint8_t *data) {
  uintptr_t offset = (uintptr_t)(data - g_model.base);
  if ((offset & 7u) != 0u || offset >= 256u * 1024u * 1024u)
    fail("accelerator tensor offset is invalid");
  return (uint32_t)(offset >> 3);
}

static uint32_t lfm_accel_run(const uint8_t *weights, uint32_t input_size,
                              uint32_t output_size, uint32_t mode) {
  lfm_accel_init();
  uint32_t old_done = lfm_accel_gpio[2] & LFM_STATUS_DONE;
  lfm_accel_register(0u, lfm_accel_weight_words(weights));
  lfm_accel_register(1u, LFM_INPUT_OFFSET >> 3);
  lfm_accel_register(2u, LFM_OUTPUT_OFFSET >> 3);
  lfm_accel_register(3u, input_size);
  lfm_accel_register(4u, output_size);
  lfm_accel_fence();
  lfm_accel_register(5u, mode);

  uint32_t status;
  do {
    status = lfm_accel_gpio[2];
  } while ((status & LFM_STATUS_DONE) == old_done);
  lfm_accel_fence();
  if ((status & LFM_STATUS_ERROR) != 0u)
    fail("LFM accelerator reported an AXI or command error");
  return status;
}

static void lfm_accel_pack_q8_range(uint32_t begin, uint32_t end) {
  /* Two 34-byte accelerator blocks occupy 68 bytes, so every pair starts on
     a 32-bit boundary. Assemble each pair in cache and issue 17 aligned
     stores instead of 68 uncached byte transactions. */
  union {
    uint8_t bytes[68];
    uint32_t words[17];
  } packed;
  for (uint32_t pair = begin; pair < end; ++pair) {
    uint32_t first = pair * 2u;
    uint32_t count = lfm_accel_pack_blocks - first;
    if (count > 2u) count = 2u;
    /* The one-block tail writes nine words; clear its two padding bytes. */
    if (count == 1u) packed.words[8] = 0u;
    for (uint32_t i = 0; i < count; ++i) {
      uint32_t block = first + i;
      uint32_t offset = i * 34u;
      fp16_t scale = float_to_half(lfm_accel_pack_q8[block].d);
      packed.bytes[offset] = (uint8_t)scale;
      packed.bytes[offset + 1u] = (uint8_t)(scale >> 8);
      memcpy(&packed.bytes[offset + 2u], lfm_accel_pack_q8[block].qs, 32u);
    }
    volatile uint32_t *dst =
        (volatile uint32_t *)(lfm_accel_scratch + first * 34u);
    uint32_t words = (count * 34u + 3u) / 4u;
    for (uint32_t word = 0; word < words; ++word)
      dst[word] = packed.words[word];
  }
}

#if !defined(LFM_ACCELERATOR_VERIFY)
static const float *lfm_accel_quantize_input;
static uint32_t lfm_accel_quantize_blocks;

static void lfm_accel_quantize_q8_range(uint32_t begin, uint32_t end) {
  union {
    uint8_t bytes[68];
    uint32_t words[17];
  } packed;
  for (uint32_t pair = begin; pair < end; ++pair) {
    uint32_t first = pair * 2u;
    uint32_t count = lfm_accel_quantize_blocks - first;
    if (count > 2u) count = 2u;
    if (count == 1u) packed.words[8] = 0u;
    for (uint32_t i = 0; i < count; ++i) {
      uint32_t block = first + i;
      const float *values = lfm_accel_quantize_input + block * 32u;
      float amax = 0.0f;
      for (uint32_t j = 0; j < 32u; ++j) {
        float a = abs_f(values[j]);
        if (a > amax) amax = a;
      }
      float d = amax / 127.0f;
      float id = d != 0.0f ? 1.0f / d : 0.0f;
      uint32_t offset = i * 34u;
      fp16_t scale = float_to_half(d);
      packed.bytes[offset] = (uint8_t)scale;
      packed.bytes[offset + 1u] = (uint8_t)(scale >> 8);
      for (uint32_t j = 0; j < 32u; ++j) {
        float q = values[j] * id;
        int value = (int)(q + (q >= 0.0f ? 0.5f : -0.5f));
        if (value > 127) value = 127;
        if (value < -127) value = -127;
        packed.bytes[offset + 2u + j] = (uint8_t)(int8_t)value;
      }
    }
    volatile uint32_t *destination =
        (volatile uint32_t *)(lfm_accel_scratch + first * 34u);
    uint32_t words = (count * 34u + 3u) / 4u;
    for (uint32_t word = 0; word < words; ++word)
      destination[word] = packed.words[word];
  }
}

static void lfm_accel_quantize_q8_0(const float *x, uint32_t n,
                                    uint32_t revision) {
  lfm_accel_init();
  uint64_t start = lfm_accel_profile != 0 ? read_ticks() : 0u;
  lfm_accel_quantize_input = x;
  lfm_accel_quantize_blocks = n / 32u;
  parallel_range(lfm_accel_quantize_q8_range,
                 (lfm_accel_quantize_blocks + 1u) / 2u);
  lfm_accel_fence();
  lfm_accel_loaded_revision = revision;
  if (lfm_accel_profile != 0)
    lfm_profile_pack_ns += read_ticks() - start;
}
#endif

static void lfm_accel_output_range(uint32_t begin, uint32_t end) {
  const float q30 = 1.0f / 1073741824.0f;
  for (uint32_t row = begin; row < end; ++row)
    lfm_accel_float_output[row] = (float)lfm_accel_raw_output[row] * q30;
}

static void parallel_q4(float *out, const struct tensor *weight,
                        const struct block_q8_0 *input) {
  lfm_accel_init();
  uint32_t blocks = weight->ne0 / 32u;
  if (lfm_accel_loaded_revision != g_q8_revision) {
    uint64_t pack_start = lfm_accel_profile != 0 ? read_ticks() : 0u;
    lfm_accel_pack_q8 = input;
    lfm_accel_pack_blocks = blocks;
    parallel_range(lfm_accel_pack_q8_range, (blocks + 1u) / 2u);
    lfm_accel_fence();
    lfm_accel_loaded_revision = g_q8_revision;
    if (lfm_accel_profile != 0)
      lfm_profile_pack_ns += read_ticks() - pack_start;
  }
#ifdef LFM_ACCELERATOR_VERIFY
  if (lfm_accel_verify_q4_calls < 5u || lfm_accel_verify_q4_calls == 9u ||
      lfm_accel_verify_q4_calls == 16u) {
    uint32_t bytes = blocks * 34u;
    uint32_t mismatches = 0;
    uint32_t first = 0;
    for (uint32_t block = 0; block < blocks; ++block) {
      fp16_t scale = float_to_half(input[block].d);
      for (uint32_t j = 0; j < 34u; ++j) {
        uint8_t expected = j == 0u ? (uint8_t)scale :
                           j == 1u ? (uint8_t)(scale >> 8) :
                           (uint8_t)input[block].qs[j - 2u];
        uint32_t i = block * 34u + j;
        if (lfm_accel_scratch[i] != expected) {
          if (mismatches == 0u) first = i;
          ++mismatches;
        }
      }
    }
    fprintf(stderr, "VERIFY Q4 call=%u scratch_bytes=%u mismatches=%u first=%u\n",
            lfm_accel_verify_q4_calls, bytes, mismatches, first);
    if (lfm_accel_verify_q4_calls == 9u) {
      int dump = open("/tmp/lfm-call9-input.bin",
                      O_WRONLY | O_CREAT | O_TRUNC, 0600);
      if (dump >= 0) {
        for (uint32_t block = 0; block < blocks; ++block) {
          fp16_t scale = float_to_half(input[block].d);
          (void)write(dump, &scale, sizeof(scale));
          (void)write(dump, input[block].qs, sizeof(input[block].qs));
        }
        (void)close(dump);
      }
    }
    lfm_accel_fence();
  }
#endif
  uint64_t q4_start = lfm_accel_profile != 0 ? read_ticks() : 0u;
  (void)lfm_accel_run(weight->data, weight->ne0, weight->ne1, 1u);
  if (lfm_accel_profile != 0) {
    lfm_profile_q4_ns += read_ticks() - q4_start;
    ++lfm_profile_q4_calls;
  }
  volatile const int64_t *raw =
      (volatile const int64_t *)(lfm_accel_scratch + LFM_OUTPUT_LOCAL);
  uint64_t output_start = lfm_accel_profile != 0 ? read_ticks() : 0u;
  lfm_accel_raw_output = raw;
  lfm_accel_float_output = out;
  parallel_range(lfm_accel_output_range, weight->ne1);
  if (lfm_accel_profile != 0)
    lfm_profile_output_ns += read_ticks() - output_start;
#ifdef LFM_ACCELERATOR_VERIFY
  if (lfm_accel_verify_q4_calls == 9u) {
    int dump = open("/tmp/lfm-call9-output.bin",
                    O_WRONLY | O_CREAT | O_TRUNC, 0600);
    if (dump >= 0) {
      (void)write(dump, (const void *)raw,
                  weight->ne1 * (uint32_t)sizeof(*raw));
      (void)close(dump);
    }
  }
  {
    const struct block_q4_0 *rows = (const struct block_q4_0 *)weight->data;
    uint32_t row_blocks = weight->ne0 / 32u;
    float maximum_error = 0.0f;
    uint32_t maximum_row = 0;
    float maximum_hardware = 0.0f;
    float maximum_reference = 0.0f;
    for (uint32_t row = 0; row < weight->ne1; ++row) {
      float reference = dot_q4_q8(rows + row * row_blocks, input, weight->ne0);
      float error = abs_f(out[row] - reference);
      if (error > maximum_error) {
        maximum_error = error;
        maximum_row = row;
        maximum_hardware = out[row];
        maximum_reference = reference;
      }
    }
    if (maximum_error > 0.0001f || lfm_accel_verify_q4_calls < 5u ||
        lfm_accel_verify_q4_calls == 9u ||
        lfm_accel_verify_q4_calls == 16u ||
        lfm_accel_verify_q4_calls == 64u) {
      fprintf(stderr,
              "VERIFY Q4 call=%u rows=%u max_row=%u hw=%.9g ref=%.9g error=%.9g\n",
              lfm_accel_verify_q4_calls, weight->ne1, maximum_row,
              maximum_hardware, maximum_reference, maximum_error);
    }
  }
  ++lfm_accel_verify_q4_calls;
#endif
}

static uint32_t parallel_q6_argmax(const struct block_q6_k *rows,
                                   const struct block_q8_k *input) {
  lfm_accel_init();
  uint64_t pack_start = lfm_accel_profile != 0 ? read_ticks() : 0u;
  union {
    uint8_t bytes[(LFM_DIM / 256u) * 258u];
    uint32_t words[((LFM_DIM / 256u) * 258u + 3u) / 4u];
  } packed = { .bytes = {0} };
  for (uint32_t block = 0; block < LFM_DIM / 256u; ++block) {
    uint32_t offset = block * 258u;
    fp16_t scale = float_to_half(input[block].d);
    packed.bytes[offset] = (uint8_t)scale;
    packed.bytes[offset + 1u] = (uint8_t)(scale >> 8);
    memcpy(&packed.bytes[offset + 2u], input[block].qs, 256u);
  }
  volatile uint32_t *dst = (volatile uint32_t *)lfm_accel_scratch;
  for (uint32_t word = 0; word < sizeof(packed.words) / sizeof(packed.words[0]);
       ++word)
    dst[word] = packed.words[word];
  lfm_accel_fence();
  if (lfm_accel_profile != 0)
    lfm_profile_q6_pack_ns += read_ticks() - pack_start;
#ifdef LFM_ACCELERATOR_VERIFY
  uint32_t input_mismatches = 0;
  uint32_t input_first = 0;
  for (uint32_t block = 0; block < LFM_DIM / 256u; ++block) {
    fp16_t scale = float_to_half(input[block].d);
    for (uint32_t j = 0; j < 258u; ++j) {
      uint8_t expected = j == 0u ? (uint8_t)scale :
                         j == 1u ? (uint8_t)(scale >> 8) :
                         (uint8_t)input[block].qs[j - 2u];
      uint32_t i = block * 258u + j;
      if (lfm_accel_scratch[i] != expected) {
        if (input_mismatches == 0u) input_first = i;
        ++input_mismatches;
      }
    }
  }
  fprintf(stderr, "VERIFY Q6 input mismatches=%u first=%u\n",
          input_mismatches, input_first);
  lfm_accel_fence();
#endif
  uint64_t q6_start = lfm_accel_profile != 0 ? read_ticks() : 0u;
  uint32_t status = lfm_accel_run((const uint8_t *)rows, LFM_DIM,
                                  LFM_VOCAB, 2u);
  if (lfm_accel_profile != 0)
    lfm_profile_q6_ns += read_ticks() - q6_start;
  uint32_t hardware_best = (status >> 8) & 0xffffu;
#ifdef LFM_ACCELERATOR_VERIFY
  static uint32_t verified;
  static uint32_t q6_calls;
  fprintf(stderr, "VERIFY Q6 call=%u hw=%u\n", q6_calls, hardware_best);
  if (verified < 2u) {
    uint32_t reference_best = 0;
    float reference_score = -3.402823466e38f;
    uint32_t row_blocks = LFM_DIM / 256u;
    for (uint32_t row = 0; row < LFM_VOCAB; ++row) {
      float score = dot_q6_q8k(rows + row * row_blocks, input, LFM_DIM);
      if (score > reference_score) {
        reference_score = score;
        reference_best = row;
      }
    }
    fprintf(stderr, "VERIFY Q6 call=%u hw=%u ref=%u score=%.9g\n",
            q6_calls, hardware_best, reference_best, reference_score);
    ++verified;
  }
  ++q6_calls;
#endif
  if (lfm_accel_profile != 0) {
    fprintf(stderr,
            "LFM profile: q4_calls=%u pack=%.3fms q4=%.3fms output=%.3fms q6_pack=%.3fms q6=%.3fms\n",
            lfm_profile_q4_calls, (double)lfm_profile_pack_ns / 1000000.0,
            (double)lfm_profile_q4_ns / 1000000.0,
            (double)lfm_profile_output_ns / 1000000.0,
            (double)lfm_profile_q6_pack_ns / 1000000.0,
            (double)lfm_profile_q6_ns / 1000000.0);
    lfm_profile_pack_ns = 0u;
    lfm_profile_q4_ns = 0u;
    lfm_profile_output_ns = 0u;
    lfm_profile_q6_pack_ns = 0u;
    lfm_profile_q6_ns = 0u;
    lfm_profile_q4_calls = 0u;
  }
  return hardware_best;
}

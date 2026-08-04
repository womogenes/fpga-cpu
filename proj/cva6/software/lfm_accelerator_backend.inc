// Included by the standalone LFM runtime after its model and quantization
// types have been declared. The reserved model and scratch regions are outside
// Linux RAM, so /dev/mem maps them uncached and avoids CPU/DMA coherency work.

#define LFM_GPIO_PHYS 0x40000000ul
#define LFM_GPIO_BYTES 4096ul
#define LFM_SCRATCH_PHYS 0xbe000000ul
#define LFM_SCRATCH_BYTES (1024ul * 1024ul)
#define LFM_MODEL_PHYS 0xb0000000ul
#define LFM_INPUT_OFFSET 0x0e000000u
#define LFM_OUTPUT_OFFSET 0x0e010000u
#define LFM_OUTPUT_LOCAL 0x00010000u
#define LFM_STATUS_PRESENT 0x80000000u
#define LFM_STATUS_BUSY 0x40000000u
#define LFM_STATUS_ERROR 0x20000000u
#define LFM_STATUS_DONE 0x10000000u

static volatile uint32_t *lfm_accel_gpio;
static volatile uint8_t *lfm_accel_scratch;
static uint32_t lfm_accel_toggle;

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
}

static void lfm_accel_register(uint32_t index, uint32_t value) {
  lfm_accel_toggle ^= 0x80000000u;
  lfm_accel_gpio[0] = lfm_accel_toggle | 0x20000000u |
                      ((index & 7u) << 25) | (value & 0x01ffffffu);
  lfm_accel_fence();
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
  lfm_accel_register(4u, output_size & 0xffffu);
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

static void parallel_q4(float *out, const struct tensor *weight,
                        const struct block_q8_0 *input) {
  lfm_accel_init();
  uint32_t blocks = weight->ne0 / 32u;
  volatile uint8_t *dst = lfm_accel_scratch;
  for (uint32_t block = 0; block < blocks; ++block) {
    fp16_t scale = float_to_half(input[block].d);
    *dst++ = (uint8_t)scale;
    *dst++ = (uint8_t)(scale >> 8);
    for (uint32_t j = 0; j < 32u; ++j)
      *dst++ = (uint8_t)input[block].qs[j];
  }
  lfm_accel_fence();
  (void)lfm_accel_run(weight->data, weight->ne0, weight->ne1, 1u);
  volatile const int64_t *raw =
      (volatile const int64_t *)(lfm_accel_scratch + LFM_OUTPUT_LOCAL);
  const float q40 = 1.0f / 1099511627776.0f;
  for (uint32_t row = 0; row < weight->ne1; ++row)
    out[row] = (float)raw[row] * q40;
}

static uint32_t parallel_q6_argmax(const struct block_q6_k *rows,
                                   const struct block_q8_k *input) {
  lfm_accel_init();
  volatile uint8_t *dst = lfm_accel_scratch;
  for (uint32_t block = 0; block < LFM_DIM / 256u; ++block) {
    fp16_t scale = float_to_half(input[block].d);
    *dst++ = (uint8_t)scale;
    *dst++ = (uint8_t)(scale >> 8);
    for (uint32_t j = 0; j < 256u; ++j)
      *dst++ = (uint8_t)input[block].qs[j];
  }
  lfm_accel_fence();
  uint32_t status = lfm_accel_run((const uint8_t *)rows, LFM_DIM,
                                  LFM_VOCAB, 2u);
  return (status >> 8) & 0xffffu;
}

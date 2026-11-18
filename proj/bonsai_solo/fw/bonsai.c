/* Bonsai-1.7B token loop, loaded at 0x1000 and started by the monitor's 'J'.
   DDR must already hold the compact model (0xaf000000), the runtime scratch
   image (0xbe000000) and the per-token layer-0 sum table (SUM_TABLE).

   Each token issues exactly the CVA6 runtime's production sequence
   (host/trace/production_trace_2tok.txt): layer-0 norm pack, one 28-layer
   sequencer command, output norm pack, hardware argmax.

   Protocol (little-endian):
     'G' n_prompt n_generate token[n_prompt]
         -> for every generated token: 'T' token cycles[6]
         -> 'E' when finished
     'Z' reset position to 0      -> 'Z'
     'Q' return to the monitor */
#include "bonsai_math.h"
#include "hw.h"

#define MODEL_PHYS 0xaf000000u
#define SCRATCH_PHYS 0xbe000000u
#define SUM_TABLE (SCRATCH_PHYS + 0x01000000u)
#define EMBEDDING_OFFSET 5945120u
#define EMBEDDING_ROW_BYTES 288u
#define VOCAB 151669u
#define DIM 2048u
#define LAYERS 28u
#define MAX_CONTEXT 2048u

#define INPUT_WORDS (0x0f000000u >> 3)
#define OUTPUT_WORDS (0x0f010000u >> 3)
#define LAYER0_SLOT_RESIDUAL 0x000d2000u /* norm slot 0 + 0x2000 */
#define OUTPUT_SLOT_RESIDUAL 0x00228000u /* norm slot 56 + 0x2000 */
#define DESCRIPTOR_LOCAL 0x00290000u

#define STATUS_ERROR 0x20000000u
#define STATUS_DONE 0x10000000u
#define STATUS_ACK 0x08000000u

static uint32_t toggle;

static void write_register(uint32_t index, uint32_t value) {
  toggle ^= 0x80000000u;
  ACCEL_COMMAND = toggle | 0x20000000u | ((index & 7u) << 25) |
                  (value & 0x01ffffffu);
  uint32_t expected = toggle ? STATUS_ACK : 0u;
  while ((ACCEL_STATUS & STATUS_ACK) != expected) {
  }
}

/* Returns the final status, or 0 with the error bit reported to the host. */
static uint32_t run(uint32_t mode, uint32_t old_done) {
  write_register(5u, mode);
  uint32_t status;
  do {
    status = ACCEL_STATUS;
  } while ((status & STATUS_DONE) == old_done);
  return status;
}

static uint32_t old_done(void) { return ACCEL_STATUS & STATUS_DONE; }

static void fail(uint32_t stage, uint32_t status) {
  uart_putc('X');
  uart_put_u32(stage);
  uart_put_u32(status);
}

/* Norm + Q8 pack of a resident FP32 vector (accelerator mode 7, ne1 = 2). */
static int pack_norm(uint32_t residual_local, uint32_t scale_bits,
                     uint32_t stage) {
  uint32_t source_words = INPUT_WORDS + residual_local / 8u;
  uint32_t done = old_done();
  write_register(1u, source_words & 0x01ffffffu);
  write_register(2u, INPUT_WORDS & 0x01ffffffu);
  write_register(6u, (source_words >> 25) | ((INPUT_WORDS >> 25) << 3) |
                     ((scale_bits >> 25) << 7));
  write_register(7u, scale_bits & 0x01ffffffu);
  write_register(3u, DIM);
  write_register(4u, 2u);
  uint32_t status = run(7u, done);
  if (status & STATUS_ERROR) {
    fail(stage, status);
    return 0;
  }
  return 1;
}

/* Expand the token's packed Q1 embedding row into the layer-0 residual. */
static void embed(uint32_t token) {
  uint32_t row = MODEL_PHYS + EMBEDDING_OFFSET + token * EMBEDDING_ROW_BYTES;
  uint32_t packed[EMBEDDING_ROW_BYTES / 4u];
  for (uint32_t beat = 0; beat < EMBEDDING_ROW_BYTES / 32u; ++beat) {
    beat_read(row + 32u * beat);
    for (uint32_t word = 0; word < 8u; ++word)
      packed[8u * beat + word] = BEAT_WORD(word);
  }
  const uint8_t *bytes = (const uint8_t *)packed;
  uint32_t destination = SCRATCH_PHYS + LAYER0_SLOT_RESIDUAL;
  BEAT_STROBE = 0xffffffffu;
  for (uint32_t block = 0; block < 16u; ++block) {
    const uint8_t *group = bytes + (block / 8u) * 144u;
    const uint8_t *scale = group + (block % 8u) * 2u;
    const uint8_t *signs = group + 16u + (block % 8u) * 16u;
    uint32_t positive =
        half_to_float_bits((uint16_t)(scale[0] | (scale[1] << 8)));
    uint32_t negative = positive ^ 0x80000000u;
    for (uint32_t beat = 0; beat < 16u; ++beat) {
      uint32_t bits = signs[beat];
      for (uint32_t word = 0; word < 8u; ++word)
        BEAT_WORD(word) = ((bits >> word) & 1u) ? positive : negative;
      BEAT_ADDRESS = (destination + (block * 128u + beat * 8u) * 4u) &
                     0x3fffffffu;
      BEAT_CONTROL = BEAT_WRITE;
    }
  }
}

static uint32_t ddr_word(uint32_t address) {
  return *(volatile uint32_t *)address;
}

static uint32_t forward(uint32_t token, uint32_t position,
                        uint32_t profile[6]) {
  uint64_t t0 = cycles();
  embed(token);
  uint32_t sum_bits = ddr_word(SUM_TABLE + 4u * token);
  uint32_t scale_bits = rms_scale_bits(sum_bits, 11u);
  uint64_t t1 = cycles();
  if (!pack_norm(LAYER0_SLOT_RESIDUAL, scale_bits, 1u)) return 0xffffffffu;
  uint64_t t2 = cycles();

  /* All 28 layers in one sequencer command (accelerator mode 7, ne1 = 6). */
  uint32_t descriptor_address = (SCRATCH_PHYS + DESCRIPTOR_LOCAL) & 0x3fffffffu;
  uint32_t descriptor_words = descriptor_address >> 3;
  uint32_t incoming_sum_bits = 0u;
  uint32_t address_config = (descriptor_words >> 25) | 4u |
      ((incoming_sum_bits >> 25) << 7) | (1u << 14);
  uint32_t done = old_done();
  write_register(0u, descriptor_words & 0x01ffffffu);
  write_register(1u, 0u);
  write_register(2u, 0u);
  write_register(3u, (0u << 17) | (LAYERS << 11) | position);
  write_register(4u, 6u);
  write_register(6u, address_config);
  write_register(7u, incoming_sum_bits & 0x01ffffffu);
  uint32_t status = run(7u, done);
  if (status & STATUS_ERROR) {
    fail(2u, status);
    return 0xffffffffu;
  }
  uint32_t final_sum_bits = (status >> 8) & 0x0003ffffu;
  write_register(6u, address_config | 0x01000000u);
  final_sum_bits |= ((ACCEL_STATUS >> 8) & 0x00003fffu) << 18;
  uint64_t t3 = cycles();

  if (!pack_norm(OUTPUT_SLOT_RESIDUAL, rms_scale_bits(final_sum_bits, 11u),
                 3u))
    return 0xffffffffu;
  uint64_t t4 = cycles();

  /* Hardware argmax over the tied embedding (packed Q1 argmax, mode 6). */
  done = old_done();
  write_register(0u, EMBEDDING_OFFSET >> 3);
  write_register(1u, INPUT_WORDS & 0x01ffffffu);
  write_register(6u, 0u);
  write_register(2u, OUTPUT_WORDS & 0x01ffffffu);
  write_register(3u, DIM);
  write_register(4u, VOCAB);
  status = run(6u, done);
  if (status & STATUS_ERROR) {
    fail(4u, status);
    return 0xffffffffu;
  }
  uint64_t t5 = cycles();
  profile[0] = (uint32_t)(t1 - t0); /* embedding + layer-0 scale */
  profile[1] = (uint32_t)(t2 - t1); /* layer-0 norm pack */
  profile[2] = (uint32_t)(t3 - t2); /* 28-layer sequencer */
  profile[3] = (uint32_t)(t4 - t3); /* output norm pack */
  profile[4] = (uint32_t)(t5 - t4); /* argmax */
  profile[5] = (uint32_t)(t5 - t0);
  return (status >> 8) & 0x0003ffffu;
}

__attribute__((section(".text.start"))) void app_main(void) {
  static uint32_t position;
  static uint32_t initialized;
  if (initialized != 0x50504153u) {
    /* Drive both command phases once so firmware and hardware agree. */
    ACCEL_COMMAND = 0xae000000u;
    for (volatile int i = 0; i < 1000; ++i) {
    }
    ACCEL_COMMAND = 0x2e000000u;
    for (volatile int i = 0; i < 1000; ++i) {
    }
    toggle = 0u;
    position = 0u;
    initialized = 0x50504153u;
  }
  LEDS = 0x08u;
  for (;;) {
    uint8_t command = uart_getc();
    if (command == 'Q') return;
    if (command == 'Z') {
      position = 0u;
      uart_putc('Z');
      continue;
    }
    if (command != 'G') {
      uart_putc('?');
      continue;
    }
    uint32_t prompt_count = uart_get_u32();
    uint32_t generate = uart_get_u32();
    uint32_t token = 0u;
    uint32_t profile[6];
    int ok = 1;
    for (uint32_t i = 0; i < prompt_count; ++i) {
      uint32_t prompt_token = uart_get_u32();
      if (!ok) continue;
      if (position >= MAX_CONTEXT || prompt_token >= VOCAB) {
        ok = 0;
        continue;
      }
      token = forward(prompt_token, position++, profile);
      if (token == 0xffffffffu) ok = 0;
    }
    /* The prediction after the last prompt token is the first output. */
    for (uint32_t i = 0; ok && i < generate; ++i) {
      uart_putc('T');
      uart_put_u32(token);
      for (int j = 0; j < 6; ++j) uart_put_u32(profile[j]);
      if (i + 1u == generate || position >= MAX_CONTEXT) break;
      token = forward(token, position++, profile);
      if (token == 0xffffffffu) ok = 0;
    }
    uart_putc('E');
  }
}

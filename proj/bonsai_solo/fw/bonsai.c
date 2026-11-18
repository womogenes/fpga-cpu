/* Bonsai-1.7B token loop, loaded at 0x1000 and started by the monitor's 'J'.
   DDR must already hold the compact model (0xaf000000), the runtime scratch
   image (0xbe000000) and the per-token layer-0 sum table (SUM_TABLE).

   Each token issues the CVA6 runtime's validated command-by-command layer
   path (host/trace/production_trace_legacy.txt): per layer a norm pack,
   fused QKV, attention, o-proj + residual, norm pack, fused up/gate, FFN
   pack, down + residual; then the output norm and hardware argmax. The
   one-command layer sequencer is not used: its retained-QKV attention mode
   produces zero K/V (found in full-system simulation).

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

/* 30-bit physical byte address of a scratch offset, in 8-byte words. */
static uint32_t scratch_words(uint32_t local) {
  return ((SCRATCH_PHYS + local) & 0x3fffffffu) >> 3;
}

/* The 57 norm slots: 0-49 from 0xd0000, 50-56 from 0x202000, 24 KiB each:
   output workspace at +0, FP32 residual at +0x2000, norm weight at +0x4000.
   Layer l uses slot 2l (attention norm) and 2l+1 (FFN norm); slot 56 holds
   the output norm. */
static uint32_t slot_base(uint32_t slot) {
  return slot < 50u ? 0x000d0000u + slot * 0x6000u
                    : 0x00202000u + (slot - 50u) * 0x6000u;
}

static uint32_t layer_weights[LAYERS][4]; /* qkv, attn_out, up/gate, down */

/* Packed Q1 matvec (mode 5) writing compact FP32 rows. */
static int matvec(uint32_t weight_words, uint32_t config, uint32_t output_words,
                  uint32_t columns, uint32_t rows, uint32_t stage) {
  uint32_t done = old_done();
  write_register(0u, weight_words);
  write_register(1u, INPUT_WORDS & 0x01ffffffu);
  write_register(6u, config);
  write_register(2u, output_words & 0x01ffffffu);
  write_register(3u, columns);
  write_register(4u, rows);
  uint32_t status = run(5u, done);
  if (status & STATUS_ERROR) {
    fail(stage, status);
    return 0;
  }
  return 1;
}

/* Projection followed by the chained residual add into next_residual.
   Returns the FP32 sum of squares of the new residual, or 0xffffffff. */
static uint32_t matvec_residual(uint32_t weight_words, uint32_t columns,
                                uint32_t workspace_local,
                                uint32_t next_residual_local, uint32_t stage) {
  uint32_t output_words = scratch_words(workspace_local);
  uint32_t chain_words = scratch_words(next_residual_local);
  uint32_t done = old_done();
  write_register(0u, weight_words);
  write_register(1u, INPUT_WORDS & 0x01ffffffu);
  write_register(7u, chain_words & 0x01ffffffu);
  write_register(6u, (INPUT_WORDS >> 25) | ((output_words >> 25) << 3) | 32u |
                     64u | (1u << 10) | (((chain_words >> 25) & 3u) << 11) |
                     (1u << 13));
  write_register(2u, output_words & 0x01ffffffu);
  write_register(3u, columns);
  write_register(4u, DIM);
  uint32_t status = run(5u, done);
  if (status & STATUS_ERROR) {
    fail(stage, status);
    return 0xffffffffu;
  }
  uint32_t sum_bits = (status >> 8) & 0x0003ffffu;
  write_register(6u, (output_words >> 25) | 4u | ((chain_words >> 25) << 3) |
                     32u | 0x01000000u);
  sum_bits |= ((ACCEL_STATUS >> 8) & 0x00003fffu) << 18;
  return sum_bits;
}

#define ATTENTION_TABLE_LOCAL 0x00200000u
#define QK_TABLE_LOCAL 0x00291000u
#define ROPE_TABLE_LOCAL 0x00300000u

static void copy_beats(uint32_t destination, uint32_t source, uint32_t beats) {
  BEAT_STROBE = 0xffffffffu;
  for (uint32_t beat = 0; beat < beats; ++beat) {
    beat_read(source + 32u * beat);
    BEAT_ADDRESS = (destination + 32u * beat) & 0x3fffffffu;
    BEAT_CONTROL = BEAT_WRITE;
  }
}

/* The attention command reads a 1,536-byte table: the layer's fixed-point
   Q/K norm weights (1,024 bytes) then the position's RoPE cos/sin (512). */
static void stage_attention_table(uint32_t layer, uint32_t position) {
  uint32_t table = SCRATCH_PHYS + ATTENTION_TABLE_LOCAL;
  copy_beats(table, SCRATCH_PHYS + QK_TABLE_LOCAL + 1024u * layer, 32u);
  copy_beats(table + 1024u, SCRATCH_PHYS + ROPE_TABLE_LOCAL + 512u * position,
             16u);
}

/* Accelerator mode 7 sub-commands (attention, FFN packing). */
static int transform(uint32_t input_words, uint32_t config, uint32_t ne0,
                     uint32_t ne1, uint32_t stage) {
  uint32_t done = old_done();
  write_register(1u, input_words & 0x01ffffffu);
  write_register(2u, INPUT_WORDS & 0x01ffffffu);
  write_register(6u, config);
  write_register(3u, ne0);
  write_register(4u, ne1);
  uint32_t status = run(7u, done);
  if (status & STATUS_ERROR) {
    fail(stage, status);
    return 0;
  }
  return 1;
}

static void load_layer_weights(void) {
  for (uint32_t layer = 0; layer < LAYERS; ++layer)
    for (uint32_t i = 0; i < 4u; ++i)
      layer_weights[layer][i] =
          ddr_word(SCRATCH_PHYS + DESCRIPTOR_LOCAL + 64u * layer + 8u + 4u * i);
}

/* One token through the CVA6 runtime's validated eight-command layer path
   (host/trace/production_trace_legacy.txt). */
static uint32_t forward(uint32_t token, uint32_t position,
                        uint32_t profile[6]) {
  uint64_t t0 = cycles();
  embed(token);
  uint32_t sum_bits = ddr_word(SUM_TABLE + 4u * token);
  uint64_t t1 = cycles();
  uint64_t norm_cycles = 0u;
  for (uint32_t layer = 0; layer < LAYERS; ++layer) {
    uint32_t attn_slot = slot_base(2u * layer);
    uint32_t ffn_slot = slot_base(2u * layer + 1u);
    uint32_t next_slot = slot_base(2u * layer + 2u);
    uint64_t n0 = cycles();
    if (!pack_norm(attn_slot + 0x2000u, rms_scale_bits(sum_bits, 11u),
                   0x10u + layer))
      return 0xffffffffu;
    norm_cycles += cycles() - n0;
    if (!matvec(layer_weights[layer][0], 64u, OUTPUT_WORDS, DIM, 4096u,
                0x40u + layer))
      return 0xffffffffu;
    stage_attention_table(layer, position);
    if (!transform(OUTPUT_WORDS, 0u, (layer << 11) | position, 4u,
                   0x60u + layer))
      return 0xffffffffu;
    sum_bits = matvec_residual(layer_weights[layer][1], DIM, attn_slot,
                               ffn_slot + 0x2000u, 0x80u + layer);
    if (sum_bits == 0xffffffffu) return sum_bits;
    n0 = cycles();
    if (!pack_norm(ffn_slot + 0x2000u, rms_scale_bits(sum_bits, 11u),
                   0xa0u + layer))
      return 0xffffffffu;
    norm_cycles += cycles() - n0;
    if (!matvec(layer_weights[layer][2], 64u | (1u << 14), OUTPUT_WORDS, DIM,
                12288u, 0xc0u + layer))
      return 0xffffffffu;
    if (!transform(OUTPUT_WORDS, 64u | (1u << 14), 6144u, 1u, 0xe0u + layer))
      return 0xffffffffu;
    sum_bits = matvec_residual(layer_weights[layer][3], 6144u, ffn_slot,
                               next_slot + 0x2000u, 0x100u + layer);
    if (sum_bits == 0xffffffffu) return sum_bits;
  }
  uint64_t t3 = cycles();

  if (!pack_norm(OUTPUT_SLOT_RESIDUAL, rms_scale_bits(sum_bits, 11u), 3u))
    return 0xffffffffu;
  uint64_t t4 = cycles();

  /* Hardware argmax over the tied embedding (packed Q1 argmax, mode 6). */
  uint32_t done = old_done();
  write_register(0u, EMBEDDING_OFFSET >> 3);
  write_register(1u, INPUT_WORDS & 0x01ffffffu);
  write_register(6u, 0u);
  write_register(2u, OUTPUT_WORDS & 0x01ffffffu);
  write_register(3u, DIM);
  write_register(4u, VOCAB);
  uint32_t status = run(6u, done);
  if (status & STATUS_ERROR) {
    fail(4u, status);
    return 0xffffffffu;
  }
  uint64_t t5 = cycles();
  profile[0] = (uint32_t)(t1 - t0);          /* embedding */
  profile[1] = (uint32_t)norm_cycles;        /* 56 norm packs */
  profile[2] = (uint32_t)(t3 - t1 - norm_cycles); /* layer kernels */
  profile[3] = (uint32_t)(t4 - t3);          /* output norm pack */
  profile[4] = (uint32_t)(t5 - t4);          /* argmax */
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
    load_layer_weights();
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

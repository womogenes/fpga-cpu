#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "bonsai_model_manifest.h"

struct block_q8_0 {
  uint16_t d;
  int8_t qs[32];
};

const uint8_t *bonsai_accelerator_map_staged_model(size_t *size_out);
volatile uint64_t *bonsai_accelerator_packed_input(void);
void bonsai_accelerator_mark_packed_input(const float *input,
                                          uint32_t columns);
int bonsai_accelerate_f32_matvec(const void *weights, uint32_t columns,
                                 uint32_t rows, const float *input,
                                 float *output);

/* These are fallback references for accelerator object paths this diagnostic
   never takes. */
void quantize_row_q8_0(const float *input, void *output, int64_t count) {
  (void)input;
  (void)output;
  (void)count;
  abort();
}

void __real_ggml_vec_dot_q1_0_q8_0(int n, float *s, size_t bs,
                                   const void *vx, size_t bx,
                                   const void *vy, size_t by, int nrc) {
  (void)n;
  (void)s;
  (void)bs;
  (void)vx;
  (void)bx;
  (void)vy;
  (void)by;
  (void)nrc;
  abort();
}

static float half_to_float(uint16_t value) {
  uint32_t sign = (uint32_t)(value & 0x8000u) << 16;
  uint32_t exponent = (value >> 10) & 0x1fu;
  uint32_t fraction = value & 0x03ffu;
  uint32_t bits;
  if (exponent == 0u) {
    if (fraction == 0u) {
      bits = sign;
    } else {
      int shift = 0;
      while ((fraction & 0x0400u) == 0u) {
        fraction <<= 1;
        ++shift;
      }
      fraction &= 0x03ffu;
      bits = sign | ((uint32_t)(113 - shift) << 23) | (fraction << 13);
    }
  } else if (exponent == 31u) {
    bits = sign | 0x7f800000u | (fraction << 13);
  } else {
    bits = sign | ((exponent + 112u) << 23) | (fraction << 13);
  }
  float result;
  memcpy(&result, &bits, sizeof(result));
  return result;
}

static float expected_segment(const uint8_t *row, uint32_t q8_block) {
  uint32_t q1_block = q8_block / 4u;
  uint32_t quarter = q8_block % 4u;
  const uint8_t *group = row + (q1_block / 4u) * 72u;
  const uint8_t *scale_bits = group + (q1_block % 4u) * 2u;
  const uint8_t *signs = group + 8u + (q1_block % 4u) * 16u;
  uint16_t scale_raw;
  memcpy(&scale_raw, scale_bits, sizeof(scale_raw));
  int sum = 0;
  for (uint32_t item = quarter * 32u; item < quarter * 32u + 32u; ++item)
    sum += ((signs[item / 8u] >> (item % 8u)) & 1u) != 0u ? 1 : -1;
  return half_to_float(scale_raw) * (float)sum;
}

static void copy_packed(volatile uint64_t *destination,
                        const struct block_q8_0 *source) {
  const uint8_t *bytes = (const uint8_t *)source;
  for (uint32_t offset = 0; offset < 192u * 34u; offset += 8u) {
    uint64_t word;
    memcpy(&word, bytes + offset, sizeof(word));
    destination[offset / 8u] = word;
  }
  __asm__ volatile("fence iorw, iorw" ::: "memory");
}

static int run_probe(const char *label, const uint8_t *weights,
                     volatile uint64_t *packed, float *dummy,
                     struct block_q8_0 *input, int scale_probe) {
  static const uint32_t samples[] = {
      0u, 1u, 3u, 4u, 15u, 16u, 31u, 32u, 47u, 48u, 63u, 64u,
      95u, 96u, 127u, 128u, 159u, 160u, 191u,
  };
  int failures = 0;
  printf("%s probe:\n", label);
  for (size_t sample = 0; sample < sizeof(samples) / sizeof(samples[0]);
       ++sample) {
    uint32_t selected = samples[sample];
    for (uint32_t block = 0; block < 192u; ++block) {
      input[block].d = scale_probe ? 0u : 0x3c00u;
      memset(input[block].qs, scale_probe ? 1 : 0,
             sizeof(input[block].qs));
    }
    if (scale_probe)
      input[selected].d = 0x3c00u;
    else
      memset(input[selected].qs, 1, sizeof(input[selected].qs));
    copy_packed(packed, input);
    bonsai_accelerator_mark_packed_input(dummy, 6144u);
    float actual = 0.0f;
    if (!bonsai_accelerate_f32_matvec(weights, 6144u, 1u, dummy, &actual)) {
      fprintf(stderr, "accelerator rejected %s block %u\n", label, selected);
      return -1;
    }
    float expected = expected_segment(weights, selected);
    float error = fabsf(actual - expected);
    float tolerance = 0.001f + fabsf(expected) * 0.001f;
    int pass = isfinite(actual) && error <= tolerance;
    failures += !pass;
    printf("  block %3u expected=% .7g actual=% .7g %s\n",
           selected, expected, actual, pass ? "PASS" : "FAIL");
  }
  return failures;
}

static int run_full_matrix_probe(const uint8_t *weights,
                                 volatile uint64_t *packed, float *dummy,
                                 struct block_q8_0 *input) {
  for (uint32_t block = 0; block < 192u; ++block) {
    input[block].d = 0x3c00u;
    memset(input[block].qs, 1, sizeof(input[block].qs));
  }
  copy_packed(packed, input);
  bonsai_accelerator_mark_packed_input(dummy, 6144u);
  float *actual = calloc(2048u, sizeof(*actual));
  float *repeat = calloc(2048u, sizeof(*repeat));
  if (actual == NULL || repeat == NULL) return -1;
  if (!bonsai_accelerate_f32_matvec(weights, 6144u, 2048u, dummy, actual)) {
    fprintf(stderr, "accelerator rejected full-matrix probe\n");
    free(actual);
    return -1;
  }
  if (!bonsai_accelerate_f32_matvec(weights, 6144u, 2048u, dummy, repeat)) {
    fprintf(stderr, "accelerator rejected repeated full-matrix probe\n");
    free(repeat);
    free(actual);
    return -1;
  }
  int repeat_failures = 0;
  for (uint32_t row = 0; row < 2048u; ++row) {
    uint32_t first_bits;
    uint32_t repeat_bits;
    memcpy(&first_bits, &actual[row], sizeof(first_bits));
    memcpy(&repeat_bits, &repeat[row], sizeof(repeat_bits));
    if (first_bits != repeat_bits) {
      if (repeat_failures < 16)
        printf("  repeat row %4u first=%08x second=%08x FAIL\n",
               row, first_bits, repeat_bits);
      ++repeat_failures;
    }
  }
  printf("full-matrix repeatability: failures=%d/2048\n", repeat_failures);
  int failures = 0;
  for (uint32_t row = 0; row < 2048u; ++row) {
    float single = 0.0f;
    if (!bonsai_accelerate_f32_matvec(weights + row * 864u, 6144u, 1u,
                                      dummy, &single)) {
      fprintf(stderr, "accelerator rejected single row %u\n", row);
      free(repeat);
      free(actual);
      return -1;
    }
    uint32_t batch_bits;
    uint32_t single_bits;
    memcpy(&batch_bits, &actual[row], sizeof(batch_bits));
    memcpy(&single_bits, &single, sizeof(single_bits));
    if (batch_bits != single_bits) {
      if (failures < 16)
        printf("  row %4u batch=%08x single=%08x FAIL\n",
               row, batch_bits, single_bits);
      ++failures;
    }
  }
  printf("full-matrix versus single-row: failures=%d/2048\n", failures);
  free(repeat);
  free(actual);
  return failures + repeat_failures;
}

static int run_prefix_probe(const uint8_t *weights,
                            volatile uint64_t *packed, float *dummy,
                            struct block_q8_0 *input) {
  static const uint32_t prefixes[] = {
      1u, 2u, 3u, 4u, 8u, 12u, 16u, 17u, 24u, 32u, 40u, 47u, 48u,
  };
  int failures = 0;
  printf("active-Q1-prefix probe:\n");
  for (size_t sample = 0; sample < sizeof(prefixes) / sizeof(prefixes[0]);
       ++sample) {
    uint32_t q1_prefix = prefixes[sample];
    float expected = 0.0f;
    for (uint32_t block = 0; block < 192u; ++block) {
      input[block].d = 0x3c00u;
      int active = block < q1_prefix * 4u;
      memset(input[block].qs, active ? 1 : 0, sizeof(input[block].qs));
      if (active) expected += expected_segment(weights, block);
    }
    copy_packed(packed, input);
    bonsai_accelerator_mark_packed_input(dummy, 6144u);
    float actual = 0.0f;
    if (!bonsai_accelerate_f32_matvec(weights, 6144u, 1u, dummy, &actual))
      return -1;
    float tolerance = 0.002f + fabsf(expected) * 0.001f;
    int pass = isfinite(actual) && fabsf(actual - expected) <= tolerance;
    failures += !pass;
    printf("  prefix %2u expected=% .7g actual=% .7g %s\n",
           q1_prefix, expected, actual, pass ? "PASS" : "FAIL");
  }
  return failures;
}

int main(void) {
  if (BONSAI_TENSOR_COUNT <= 5u || bonsai_tensors[5].ne0 != 6144u) {
    fprintf(stderr, "unexpected model manifest\n");
    return 2;
  }
  size_t model_size = 0u;
  const uint8_t *model = bonsai_accelerator_map_staged_model(&model_size);
  if (model == NULL || model_size < bonsai_tensors[5].offset + 864u) {
    fprintf(stderr, "could not map prestaged model\n");
    return 2;
  }
  volatile uint64_t *packed = bonsai_accelerator_packed_input();
  if (packed == NULL) {
    fprintf(stderr, "packed accelerator input unavailable\n");
    return 2;
  }
  float *dummy = calloc(6144u, sizeof(*dummy));
  struct block_q8_0 *input = calloc(192u, sizeof(*input));
  if (dummy == NULL || input == NULL) return 2;
  const uint8_t *weights = model + bonsai_tensors[5].offset;
  int data_failures = run_probe("wide-data", weights, packed, dummy, input, 0);
  int scale_failures = run_probe("scale", weights, packed, dummy, input, 1);
  int prefix_failures = run_prefix_probe(weights, packed, dummy, input);
  int matrix_failures = run_full_matrix_probe(weights, packed, dummy, input);
  printf("summary: wide-data failures=%d scale failures=%d "
         "prefix failures=%d matrix failures=%d\n",
         data_failures, scale_failures, prefix_failures, matrix_failures);
  return data_failures == 0 && scale_failures == 0 && prefix_failures == 0 &&
                 matrix_failures == 0
             ? 0
             : 1;
}

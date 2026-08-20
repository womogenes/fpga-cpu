#include <stdint.h>
#include <string.h>

enum {
  BONSAI_HEADS = 16u,
  BONSAI_KV_HEADS = 8u,
  BONSAI_HEAD_DIM = 128u,
  BONSAI_KV_DIM = BONSAI_KV_HEADS * BONSAI_HEAD_DIM,
  BONSAI_MAX_CONTEXT = 256u,
};

static uint32_t positive_q16_to_f32_bits(uint32_t value) {
  if (value == 0u) return 0u;
  uint32_t leading = 31u - (uint32_t)__builtin_clz(value);
  uint32_t significand = value << (23u - leading);
  int exponent = (int)leading - 16 + 127;
  return ((uint32_t)exponent << 23) |
         (significand & UINT32_C(0x007fffff));
}

static uint32_t signed_q16_to_f32_bits(int64_t value) {
  if (value == 0) return 0u;
  uint32_t sign = value < 0 ? UINT32_C(0x80000000) : 0u;
  uint64_t magnitude = value < 0 ? (uint64_t)-value : (uint64_t)value;
  if (magnitude > UINT32_C(16777216))
    magnitude = UINT32_C(16777216);
  return sign | positive_q16_to_f32_bits((uint32_t)magnitude);
}

static uint32_t exp_q16(int32_t value) {
  if (value <= -INT32_C(1310720)) return 0u;
  if (value >= 0) return 65536u;
  int64_t scaled = (int64_t)value * INT64_C(94548);
  int32_t n = -(int32_t)((-scaled + INT64_C(2147483648)) >> 32);
  int64_t r = ((int64_t)value << 14) -
              (int64_t)n * INT64_C(744261118);
  int64_t r2 = (r * r) >> 30;
  int64_t polynomial = INT64_C(1073741824) + r;
  int64_t power = r2;
  polynomial += power / 2;
  power = (power * r) >> 30;
  polynomial += power / 6;
  power = (power * r) >> 30;
  polynomial += power / 24;
  power = (power * r) >> 30;
  polynomial += power / 120;
  if (n < 0) polynomial >>= -n;
  else polynomial <<= n;
  if (polynomial <= 0) return 0u;
  uint64_t rounded = (uint64_t)(polynomial + (INT64_C(1) << 13)) >> 14;
  return rounded > 65536u ? 65536u : (uint32_t)rounded;
}

static uint64_t divide_u64_reciprocal(uint64_t numerator,
                                      uint64_t denominator,
                                      uint64_t reciprocal) {
  uint64_t quotient =
      (uint64_t)(((__uint128_t)numerator * reciprocal) >> 64);
  if (quotient * denominator > numerator) --quotient;
  return quotient;
}

static void output_sums(float *output, const int64_t *sums,
                        uint64_t denominator) {
  uint64_t reciprocal = UINT64_MAX / denominator + 1u;
  uint64_t rounding = denominator / 2u;
  for (uint32_t j = 0u; j < BONSAI_HEAD_DIM; ++j) {
    int64_t sum = sums[j];
    uint64_t magnitude = sum >= 0 ? (uint64_t)sum : (uint64_t)-sum;
    int64_t result = (int64_t)divide_u64_reciprocal(
        magnitude + rounding, denominator, reciprocal);
    if (sum < 0) result = -result;
    uint32_t bits = signed_q16_to_f32_bits(result);
    memcpy(output + j, &bits, sizeof(bits));
  }
}

/* Evaluate the two query heads belonging to each GQA key/value head
   together. The arithmetic order within every dot product, softmax, and
   value accumulation stays identical to the scalar reference. */
void bonsai_attention_gqa_range(float *attention, const int32_t *query,
                                const int32_t *keys, const int32_t *values,
                                uint32_t begin, uint32_t end,
                                uint32_t position) {
  if (position == 0u) {
    for (uint32_t h = begin; h < end; ++h) {
      uint32_t kvh = h / (BONSAI_HEADS / BONSAI_KV_HEADS);
      const int32_t *value = values + kvh * BONSAI_HEAD_DIM;
      for (uint32_t j = 0u; j < BONSAI_HEAD_DIM; ++j) {
        uint32_t bits = signed_q16_to_f32_bits(value[j]);
        memcpy(attention + h * BONSAI_HEAD_DIM + j, &bits, sizeof(bits));
      }
    }
    return;
  }

  for (uint32_t h = begin; h < end; h += 2u) {
    uint32_t pair_end = h + 2u < end ? h + 2u : end;
    int32_t scores[2][BONSAI_MAX_CONTEXT];
    int32_t maxima[2] = {INT32_MIN, INT32_MIN};
    uint32_t kvh = h / (BONSAI_HEADS / BONSAI_KV_HEADS);
    for (uint32_t t = 0u; t <= position; ++t) {
      const int32_t *key = keys + (size_t)t * BONSAI_KV_DIM +
                           kvh * BONSAI_HEAD_DIM;
      for (uint32_t local = 0u; h + local < pair_end; ++local) {
        const int32_t *q = query + (h + local) * BONSAI_HEAD_DIM;
        int64_t dot = 0;
        for (uint32_t j = 0u; j < BONSAI_HEAD_DIM; ++j)
          dot += (int64_t)q[j] * key[j];
        int64_t scaled = dot * INT64_C(5793);
        int32_t score = scaled >= 0 ?
            (int32_t)((scaled + (INT64_C(1) << 31)) >> 32) :
            -(int32_t)((-scaled + (INT64_C(1) << 31)) >> 32);
        scores[local][t] = score;
        if (score > maxima[local]) maxima[local] = score;
      }
    }
    uint64_t denominators[2] = {0u, 0u};
    for (uint32_t local = 0u; h + local < pair_end; ++local) {
      for (uint32_t t = 0u; t <= position; ++t) {
        scores[local][t] =
            (int32_t)exp_q16(scores[local][t] - maxima[local]);
        denominators[local] += (uint32_t)scores[local][t];
      }
    }
    int64_t sums[2][BONSAI_HEAD_DIM] = {{0}};
    for (uint32_t t = 0u; t <= position; ++t) {
      const int32_t *value = values + (size_t)t * BONSAI_KV_DIM +
                             kvh * BONSAI_HEAD_DIM;
      int64_t score0 = scores[0][t];
      int64_t score1 = pair_end == h + 2u ? scores[1][t] : 0;
      for (uint32_t j = 0u; j < BONSAI_HEAD_DIM; ++j) {
        sums[0][j] += score0 * value[j];
        if (pair_end == h + 2u) sums[1][j] += score1 * value[j];
      }
    }
    output_sums(attention + h * BONSAI_HEAD_DIM, sums[0], denominators[0]);
    if (pair_end == h + 2u)
      output_sums(attention + (h + 1u) * BONSAI_HEAD_DIM, sums[1],
                  denominators[1]);
  }
}

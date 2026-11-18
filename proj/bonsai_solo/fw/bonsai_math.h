/* Bit-exact integer float helpers shared by the controller firmware and the
   host table builder. Copied from the CVA6 runtime (software/bonsai_main.c),
   which already avoided the FPU; any change here breaks exact agreement with
   the validated CVA6 token outputs. */
#ifndef BONSAI_MATH_H
#define BONSAI_MATH_H

#include <stdint.h>

static uint32_t half_to_float_bits(uint16_t h) {
  uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
  uint32_t exponent = (h >> 10) & 0x1fu;
  uint32_t mantissa = h & 0x3ffu;
  uint32_t bits;
  if (exponent == 0u) {
    if (mantissa == 0u) {
      bits = sign;
    } else {
      exponent = 127u - 15u + 1u;
      while ((mantissa & 0x400u) == 0u) {
        mantissa <<= 1;
        exponent--;
      }
      mantissa &= 0x3ffu;
      bits = sign | (exponent << 23) | (mantissa << 13);
    }
  } else if (exponent == 0x1fu) {
    bits = sign | 0x7f800000u | (mantissa << 13);
  } else {
    bits = sign | ((exponent - 15u + 127u) << 23) | (mantissa << 13);
  }
  return bits;
}

static uint32_t multiply_f32_bits(uint32_t a, uint32_t b) {
  uint32_t sign = (a ^ b) & UINT32_C(0x80000000);
  uint32_t exponent_a = (a >> 23) & 0xffu;
  uint32_t exponent_b = (b >> 23) & 0xffu;
  if (exponent_a == 0u || exponent_b == 0u) return sign;
  if (exponent_a == 0xffu || exponent_b == 0xffu)
    return sign | UINT32_C(0x7f800000);
  uint32_t significand_a = (a & UINT32_C(0x007fffff)) |
                           UINT32_C(0x00800000);
  uint32_t significand_b = (b & UINT32_C(0x007fffff)) |
                           UINT32_C(0x00800000);
  uint64_t product = (uint64_t)significand_a * significand_b;
  uint32_t normalize = (uint32_t)(product >> 47);
  uint32_t shift = 23u + normalize;
  uint32_t significand = (uint32_t)(product >> shift);
  uint64_t remainder = product & ((UINT64_C(1) << shift) - 1u);
  uint64_t halfway = UINT64_C(1) << (shift - 1u);
  if (remainder > halfway ||
      (remainder == halfway && (significand & 1u) != 0u))
    ++significand;
  int exponent = (int)exponent_a + (int)exponent_b - 127 +
                 (int)normalize;
  if (significand == UINT32_C(0x01000000)) {
    significand >>= 1;
    ++exponent;
  }
  if (exponent <= 0) return sign;
  if (exponent >= 255) return sign | UINT32_C(0x7f800000);
  return sign | ((uint32_t)exponent << 23) |
         (significand & UINT32_C(0x007fffff));
}

static uint32_t add_f32_bits(uint32_t a, uint32_t b) {
  uint32_t magnitude_a = a & UINT32_C(0x7fffffff);
  uint32_t magnitude_b = b & UINT32_C(0x7fffffff);
  if (magnitude_a == 0u) return b;
  if (magnitude_b == 0u) return a;
  uint32_t exponent_a = magnitude_a >> 23;
  uint32_t exponent_b = magnitude_b >> 23;
  if (exponent_a == 0xffu) return a;
  if (exponent_b == 0xffu) return b;
  if (magnitude_a < magnitude_b) {
    uint32_t temporary = a;
    a = b;
    b = temporary;
    temporary = magnitude_a;
    magnitude_a = magnitude_b;
    magnitude_b = temporary;
    temporary = exponent_a;
    exponent_a = exponent_b;
    exponent_b = temporary;
  }
  uint64_t significand_a =
      ((uint64_t)((magnitude_a & UINT32_C(0x007fffff)) |
                  UINT32_C(0x00800000))) << 32;
  uint64_t significand_b =
      ((uint64_t)((magnitude_b & UINT32_C(0x007fffff)) |
                  UINT32_C(0x00800000))) << 32;
  uint32_t difference = exponent_a - exponent_b;
  if (difference >= 63u) {
    significand_b = 1u;
  } else if (difference != 0u) {
    uint64_t discarded = significand_b &
        ((UINT64_C(1) << difference) - 1u);
    significand_b = (significand_b >> difference) | (discarded != 0u);
  }
  uint64_t result;
  uint32_t sign = a & UINT32_C(0x80000000);
  if (((a ^ b) & UINT32_C(0x80000000)) == 0u) {
    result = significand_a + significand_b;
    if ((result & (UINT64_C(1) << 56)) != 0u) {
      result = (result >> 1) | (result & 1u);
      ++exponent_a;
    }
  } else {
    result = significand_a - significand_b;
    if (result == 0u) return 0u;
    while ((result & (UINT64_C(1) << 55)) == 0u && exponent_a > 1u) {
      result <<= 1;
      --exponent_a;
    }
  }
  uint32_t significand = (uint32_t)(result >> 32);
  uint64_t remainder = result & UINT64_C(0xffffffff);
  if (remainder > UINT64_C(0x80000000) ||
      (remainder == UINT64_C(0x80000000) && (significand & 1u) != 0u))
    ++significand;
  if (significand == UINT32_C(0x01000000)) {
    significand >>= 1;
    ++exponent_a;
  }
  if (exponent_a >= 255u) return sign | UINT32_C(0x7f800000);
  return sign | (exponent_a << 23) |
         (significand & UINT32_C(0x007fffff));
}

static uint64_t integer_sqrt_u64(uint64_t value) {
  uint64_t result = 0u;
  uint64_t bit = UINT64_C(1) << 62;
  while (bit > value) bit >>= 2;
  while (bit != 0u) {
    if (value >= result + bit) {
      value -= result + bit;
      result = (result >> 1) + bit;
    } else {
      result >>= 1;
    }
    bit >>= 2;
  }
  return result;
}

static uint32_t add_positive_normal_f32_bits(uint32_t a, uint32_t b) {
  uint32_t exponent_a = a >> 23;
  uint32_t exponent_b = b >> 23;
  uint32_t significand_a = (a & UINT32_C(0x007fffff)) |
                           UINT32_C(0x00800000);
  uint32_t significand_b = (b & UINT32_C(0x007fffff)) |
                           UINT32_C(0x00800000);
  if (exponent_a < exponent_b) {
    uint32_t temporary = exponent_a;
    exponent_a = exponent_b;
    exponent_b = temporary;
    temporary = significand_a;
    significand_a = significand_b;
    significand_b = temporary;
  }
  uint32_t difference = exponent_a - exponent_b;
  uint64_t aligned_a = (uint64_t)significand_a << 3;
  uint64_t aligned_b = (uint64_t)significand_b << 3;
  if (difference >= 28u) {
    aligned_b = aligned_b != 0u;
  } else if (difference != 0u) {
    uint64_t discarded = aligned_b & ((UINT64_C(1) << difference) - 1u);
    aligned_b = (aligned_b >> difference) | (discarded != 0u);
  }
  uint64_t total = aligned_a + aligned_b;
  if ((total & (UINT64_C(1) << 27)) != 0u) {
    total = (total >> 1) | (total & 1u);
    ++exponent_a;
  }
  uint32_t significand = (uint32_t)(total >> 3);
  uint32_t remainder = (uint32_t)(total & 7u);
  if (remainder > 4u || (remainder == 4u && (significand & 1u) != 0u))
    ++significand;
  if (significand == UINT32_C(0x01000000)) {
    significand >>= 1;
    ++exponent_a;
  }
  return (exponent_a << 23) | (significand & UINT32_C(0x007fffff));
}

#define BONSAI_EPS_BITS UINT32_C(0x358637bd) /* 1e-6f */

/* 1/sqrt(sum / 2^shift + epsilon), returned as FP32 bits. */
static uint32_t rms_scale_bits(uint32_t sum_bits, uint32_t shift) {
  uint32_t epsilon_bits = BONSAI_EPS_BITS;
  uint32_t exponent = (sum_bits >> 23) & 0xffu;
  uint32_t mean_bits;
  if (exponent > shift) {
    mean_bits = sum_bits - (shift << 23);
  } else {
    uint32_t significand = (sum_bits & UINT32_C(0x007fffff)) |
                           UINT32_C(0x00800000);
    uint32_t down = shift + 1u - exponent;
    mean_bits = down < 24u ? significand >> down : 0u;
  }
  uint32_t x_bits = mean_bits == 0u ? epsilon_bits :
      add_positive_normal_f32_bits(mean_bits, epsilon_bits);
  exponent = (x_bits >> 23) & 0xffu;
  uint32_t significand = (x_bits & UINT32_C(0x007fffff)) |
                         UINT32_C(0x00800000);
  int unbiased = (int)exponent - 127;
  int half = unbiased >= 0 ? unbiased / 2 : -((-unbiased + 1) / 2);
  uint32_t odd = (uint32_t)(unbiased - 2 * half);
  uint64_t radicand = (UINT64_C(1) << 63) /
                      ((uint64_t)significand << odd);
  uint32_t root = (uint32_t)integer_sqrt_u64(radicand);
  uint32_t leading = 31u - (uint32_t)__builtin_clz(root);
  uint32_t output_significand = root << (23u - leading);
  int output_exponent = (int)leading - 20 - half + 127;
  return ((uint32_t)output_exponent << 23) |
         (output_significand & UINT32_C(0x007fffff));
}

/* Sequential sum of squares of one embedding row, in the CVA6 runtime's
   reliable_sum_squares order. Each element is +/-scale of its block. */
__attribute__((unused)) static uint32_t embedding_sum_squares_bits(const uint8_t *row) {
  uint32_t sum_bits = 0u;
  for (uint32_t block = 0u; block < 16u; ++block) {
    const uint8_t *group = row + (block / 8u) * 144u;
    const uint8_t *scale = group + (block % 8u) * 2u;
    uint32_t d = half_to_float_bits((uint16_t)(scale[0] | (scale[1] << 8)));
    /* (+d)^2 == (-d)^2 bit for bit, so the sign bits do not matter. */
    uint32_t square = multiply_f32_bits(d, d);
    for (uint32_t i = 0u; i < 128u; ++i)
      sum_bits = add_f32_bits(sum_bits, square);
  }
  return sum_bits;
}

#endif

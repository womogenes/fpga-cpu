#include <stdint.h>
#include <stdio.h>
#include <string.h>

static int64_t fp32_bits_to_q16(uint32_t bits) {
  uint32_t magnitude = bits & UINT32_C(0x7fffffff);
  uint32_t exponent_field = magnitude >> 23;
  if (exponent_field == 0u) return 0;
  uint64_t significand =
      (magnitude & UINT32_C(0x007fffff)) | UINT32_C(0x00800000);
  int shift = (int)exponent_field - 127 - 7;
  uint64_t fixed = shift >= 0 ? significand << shift :
      (-shift >= 32 ? 0u : significand >> -shift);
  return (bits >> 31) != 0u ? -(int64_t)fixed : (int64_t)fixed;
}

static uint32_t sigmoid_q16(int64_t value) {
  static const uint16_t table[65] = {
    22, 28, 36, 47, 60, 77, 98, 126, 162, 208, 267, 342, 439, 562,
    720, 922, 1179, 1506, 1921, 2446, 3108, 3938, 4971, 6249, 7812,
    9702, 11955, 14595, 17625, 21025, 24743, 28693, 32768, 36843,
    40793, 44511, 47911, 50941, 53581, 55834, 57724, 59287, 60565,
    61598, 62428, 63090, 63615, 64030, 64357, 64614, 64816, 64974,
    65097, 65194, 65269, 65328, 65374, 65410, 65438, 65459, 65476,
    65489, 65500, 65508, 65514,
  };
  if (value <= -INT64_C(524288)) return 0u;
  if (value >= INT64_C(524288)) return 65536u;
  uint32_t position = (uint32_t)(value + INT64_C(524288));
  uint32_t index = position >> 14;
  uint32_t fraction = position & UINT32_C(0x3fff);
  uint32_t lower = table[index];
  int32_t difference = (int32_t)table[index + 1u] - (int32_t)lower;
  return lower + (uint32_t)((difference * (int32_t)fraction) >> 14);
}

static uint32_t positive_q16_to_f32_bits(uint32_t value) {
  if (value == 0u) return 0u;
  uint32_t leading = 31u - (uint32_t)__builtin_clz(value);
  uint32_t significand = value << (23u - leading);
  int exponent = (int)leading - 16 + 127;
  return ((uint32_t)exponent << 23) |
         (significand & UINT32_C(0x007fffff));
}

static uint32_t multiply_f32_bits(uint32_t a, uint32_t b) {
  uint32_t sign = (a ^ b) & UINT32_C(0x80000000);
  uint32_t exponent_a = (a >> 23) & 0xffu;
  uint32_t exponent_b = (b >> 23) & 0xffu;
  if (exponent_a == 0u || exponent_b == 0u) return sign;
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
  return sign | ((uint32_t)exponent << 23) |
         (significand & UINT32_C(0x007fffff));
}

int main(void) {
  const uint32_t up = UINT32_C(0xbf1dac17);
  const uint32_t gate = UINT32_C(0xbe656f7e);
  int64_t gate_q16 = fp32_bits_to_q16(gate);
  uint32_t sigmoid = sigmoid_q16(gate_q16);
  uint32_t sigmoid_bits = positive_q16_to_f32_bits(sigmoid);
  uint32_t gated = multiply_f32_bits(gate, sigmoid_bits);
  uint32_t result = multiply_f32_bits(up, gated);
  printf("gate_q16=%lld sigmoid=%u sigmoid_bits=%08x gated=%08x result=%08x\n",
         (long long)gate_q16, sigmoid, sigmoid_bits, gated, result);
  return 0;
}

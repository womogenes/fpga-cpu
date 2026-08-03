#include <stdint.h>

union float_bits {
  float value;
  uint32_t bits;
};

float fabsf(float value) {
  union float_bits converted = {.value = value};
  converted.bits &= 0x7fffffffu;
  return converted.value;
}

float copysignf(float magnitude, float sign) {
  union float_bits magnitude_bits = {.value = magnitude};
  union float_bits sign_bits = {.value = sign};
  magnitude_bits.bits =
      (magnitude_bits.bits & 0x7fffffffu) | (sign_bits.bits & 0x80000000u);
  return magnitude_bits.value;
}

float nanf(const char* tag) {
  (void)tag;
  union float_bits result = {.bits = 0x7fc00000u};
  return result.value;
}

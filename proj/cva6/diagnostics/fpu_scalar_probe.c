#define _GNU_SOURCE

#include <pthread.h>
#include <sched.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

static uint32_t bits(float value) {
  uint32_t result;
  memcpy(&result, &value, sizeof(result));
  return result;
}

static float from_bits(uint32_t value) {
  float result;
  memcpy(&result, &value, sizeof(result));
  return result;
}

static float fp_add(float a, float b) {
  float result;
  __asm__ volatile("fadd.s %0, %1, %2" : "=f"(result) : "f"(a), "f"(b));
  return result;
}

static float fp_mul(float a, float b) {
  float result;
  __asm__ volatile("fmul.s %0, %1, %2" : "=f"(result) : "f"(a), "f"(b));
  return result;
}

static float fp_madd(float a, float b, float c) {
  float result;
  __asm__ volatile("fmadd.s %0, %1, %2, %3"
                   : "=f"(result)
                   : "f"(a), "f"(b), "f"(c));
  return result;
}

int main(void) {
  cpu_set_t cpus;
  CPU_ZERO(&cpus);
  CPU_SET(0, &cpus);
  if (pthread_setaffinity_np(pthread_self(), sizeof(cpus), &cpus) != 0)
    return 2;

  const float a = from_bits(UINT32_C(0x3fc00000));
  const float b = from_bits(UINT32_C(0x40100000));
  const float c = from_bits(UINT32_C(0xbf000000));
  uint32_t add_first = 0;
  uint32_t mul_first = 0;
  uint32_t madd_first = 0;
  unsigned add_bad = 0;
  unsigned mul_bad = 0;
  unsigned madd_bad = 0;

  for (unsigned i = 0; i < 32; ++i) {
    uint32_t add = bits(fp_add(a, b));
    uint32_t mul = bits(fp_mul(a, b));
    uint32_t madd = bits(fp_madd(a, b, c));
    if (i == 0) {
      add_first = add;
      mul_first = mul;
      madd_first = madd;
    } else {
      add_bad += add != add_first;
      mul_bad += mul != mul_first;
      madd_bad += madd != madd_first;
    }
    printf("%02u add=%08x mul=%08x madd=%08x\n", i, add, mul, madd);
  }
  printf("expected add=40700000 mul=40580000 madd=40380000; mismatches=%u/%u/%u\n",
         add_bad, mul_bad, madd_bad);

  unsigned identity_bad = 0;
  uint32_t state = UINT32_C(0x12345678);
  const float zero = from_bits(UINT32_C(0x00000000));
  const float one = from_bits(UINT32_C(0x3f800000));
  for (unsigned i = 0; i < 100000; ++i) {
    state = state * UINT32_C(1664525) + UINT32_C(1013904223);
    uint32_t input_bits = UINT32_C(0x3f000000) | (state & UINT32_C(0x007fffff));
    float input = from_bits(input_bits);
    identity_bad += bits(fp_add(input, zero)) != input_bits;
    identity_bad += bits(fp_mul(input, one)) != input_bits;
    identity_bad += bits(fp_madd(input, one, zero)) != input_bits;
  }
  printf("identity operations: mismatches=%u/300000\n", identity_bad);
  return add_bad || mul_bad || madd_bad || identity_bad ||
         add_first != UINT32_C(0x40700000) || mul_first != UINT32_C(0x40580000) ||
         madd_first != UINT32_C(0x40380000);
}

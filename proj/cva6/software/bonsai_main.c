/* Minimal Qwen3-1.7B (Bonsai) decoder for the CVA6 FPGA workstation.
 *
 * All 197 Q1_0 matrices run on the FPGA matrix-vector accelerator through
 * the whole-matrix submit/read interface in bonsai_accelerator.c; the small
 * f32 norm vectors are read from the memory-mapped GGUF. The scalar side
 * follows the LFM2.5 decoder: a persistent pthread worker pool covers
 * attention heads, logit argmax, and accelerator result conversion, and a
 * TOKEN/EVAL line protocol serves an interactive chat client. */

#define _GNU_SOURCE

#include <fcntl.h>
#include <linux/futex.h>
#include <math.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>

#include "bonsai_model_manifest.h"
#include "bonsai_rope_manifest.h"

#define BONSAI_LAYERS 28u
#define BONSAI_DIM 2048u
#define BONSAI_HEADS 16u
#define BONSAI_KV_HEADS 8u
#define BONSAI_HEAD_DIM 128u
#define BONSAI_KV_DIM (BONSAI_KV_HEADS * BONSAI_HEAD_DIM)
#define BONSAI_FF_DIM 6144u
#define BONSAI_VOCAB 151669u
#define BONSAI_EPS 1e-6f
#define BONSAI_MAX_CONTEXT 256u
#define BONSAI_MAX_WORKERS 4u

#define BONSAI_ROPE_BASE 1000000.0f
#define BONSAI_YARN_FACTOR 4.0f
#define BONSAI_YARN_ORIG_CTX 8192.0f
#define BONSAI_YARN_BETA_FAST 32.0f
#define BONSAI_YARN_BETA_SLOW 1.0f

#define GGUF_TYPE_F32 0u
#define GGUF_TYPE_Q1_0 41u

#ifndef BONSAI_HOST_STUB
int bonsai_accelerate_f32_matvec_submit(const void *weights, uint32_t columns,
                                        uint32_t rows, const float *input);
void bonsai_accelerate_read_rows(float *output, uint32_t first,
                                 uint32_t count);
float bonsai_accelerate_read_rows_add_sum(float *residual, uint32_t first,
                                          uint32_t count);
void bonsai_accelerate_release(void);
int bonsai_accelerate_f32_matvec_store(const void *weights, uint32_t columns,
                                       uint32_t rows, const float *input,
                                       uint32_t output_first);
int bonsai_accelerate_f32_matvec_argmax(const void *weights,
                                        uint32_t columns, uint32_t rows,
                                        const float *input,
                                        uint32_t *best_token);
void bonsai_accelerate_read_stored_rows(float *output, uint32_t source_first,
                                        uint32_t count);
const volatile int64_t *bonsai_accelerate_stored_output(void);
void bonsai_accelerate_argmax_rows(uint32_t first, uint32_t count,
                                   int64_t *best_raw, uint32_t *best_token);
int bonsai_accelerator_prepare(void);
int bonsai_accelerator_has_float_output(void);
int bonsai_accelerator_has_compact_output(void);
int bonsai_accelerator_has_attention(void);
volatile uint64_t *bonsai_accelerator_packed_input(void);
void bonsai_accelerator_mark_packed_input(const float *input,
                                          uint32_t columns);
void bonsai_accelerator_output_fence(void);
int bonsai_accelerator_pack_ffn(uint32_t elements);
int bonsai_accelerator_pack_norm(const float *input, const float *weight,
                                 float scale, uint32_t elements);
int bonsai_accelerator_pack_resident_norm(const float *weight, float scale,
                                          uint32_t elements);
int bonsai_accelerate_f32_matvec_residual(
    const void *weights, uint32_t columns, uint32_t rows, const float *input,
    const float *next_norm_weight, float *sum_out);
int bonsai_accelerator_pack_vector(const float *input, uint32_t elements);
int bonsai_accelerator_attention(uint32_t layer, uint32_t position,
                                 const int32_t *q_weight,
                                 const int32_t *k_weight,
                                 const int32_t *rope_cos,
                                 const int32_t *rope_sin,
                                 float *probe_output);
const volatile uint16_t *bonsai_accelerator_attention_kv(
    uint32_t layer, uint32_t position);
const uint8_t *bonsai_accelerator_map_staged_model(size_t *size_out);
#else
int bonsai_accelerate_f32_matvec_submit(const void *weights, uint32_t columns,
                                        uint32_t rows, const float *input);
int bonsai_accelerate_f32_matvec_argmax(const void *weights,
                                        uint32_t columns, uint32_t rows,
                                        const float *input,
                                        uint32_t *best_token);
void bonsai_accelerate_read_rows(float *output, uint32_t first,
                                 uint32_t count);
void bonsai_accelerate_release(void);
void bonsai_accelerate_argmax_rows(uint32_t first, uint32_t count,
                                   int64_t *best_raw, uint32_t *best_token);
#endif

typedef uint16_t fp16_t;

void quantize_row_q8_0(const float *x, void *vy, int64_t k);
void bonsai_attention_gqa_range(float *attention, const int32_t *query,
                                const int32_t *keys, const int32_t *values,
                                uint32_t begin, uint32_t end,
                                uint32_t position);

/* Polynomial exp and inline round keep every hot float operation on the
   single-precision hardware; libm's generic paths promote through soft
   double on this F-only ABI and cost tens of microseconds per call. */

#ifdef BONSAI_HOST_STUB
static int bonsai_accelerator_prepare(void) {
  return 1;
}
static int bonsai_accelerator_has_float_output(void) {
  return 0;
}
static int bonsai_accelerator_has_compact_output(void) {
  return 0;
}
static int bonsai_accelerator_has_attention(void) {
  return 0;
}
#endif

/* This board's FPU divide path is unreliable (fdiv.s can return NaN), so
   every hot-path division goes through Newton iterations that need only
   multiply and subtract. rsqrt_f returns 1/sqrt(x) directly for the RMS
   norms; rcp_f returns 1/x. */
static inline float rsqrt_f(float x) {
  union { float f; uint32_t u; } v = {x};
  if (x <= 0.0f) return 0.0f;
  v.u = 0x5f3759dfu - (v.u >> 1);
  float y = v.f;
  y = y * (1.5f - 0.5f * x * y * y);
  y = y * (1.5f - 0.5f * x * y * y);
  y = y * (1.5f - 0.5f * x * y * y);
  return y;
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

/* Compute 1/sqrt(sum / 2^shift + epsilon) using only integer operations.
   This avoids both the unreliable dependent fmul.s path and the inaccurate
   target libm sqrt while retaining better than one-part-per-million accuracy. */
static inline float rms_scale_f(float sum, uint32_t shift) {
  uint32_t sum_bits;
  uint32_t epsilon_bits;
  memcpy(&sum_bits, &sum, sizeof(sum_bits));
  float epsilon = BONSAI_EPS;
  memcpy(&epsilon_bits, &epsilon, sizeof(epsilon_bits));
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
  uint32_t output_bits = ((uint32_t)output_exponent << 23) |
      (output_significand & UINT32_C(0x007fffff));
  float output;
  memcpy(&output, &output_bits, sizeof(output));
  return output;
}

static inline float rcp_f(float x) {
  union { float f; uint32_t u; } v = {x};
  if (x == 0.0f) return 0.0f;
  v.u = 0x7ef311c2u - v.u;
  float y = v.f;
  y = y * (2.0f - x * y);
  y = y * (2.0f - x * y);
  return y;
}

static inline float mul_add_f(float a, float b, float c) {
  return a * b + c;
}

static inline float exp_f(float x) {
  if (x < -20.0f) return 0.0f;
  if (x > 20.0f) x = 20.0f;
  const float inv_ln2 = 1.4426950408889634f;
  int n = (int)(x * inv_ln2 + (x >= 0.0f ? 0.5f : -0.5f));
  float r = x - (float)n * 0.6931471805599453f;
  float r2 = r * r;
  float poly = 1.0f + r + r2 * (0.5f + r * (0.16666667f +
               r * (0.04166667f + r * 0.008333333f)));
  union { float f; uint32_t u; } scale;
  int e = n + 127;
  if (e <= 0) return 0.0f;
  if (e >= 255) e = 254;
  scale.u = (uint32_t)e << 23;
  return poly * scale.f;
}

static inline float round_f(float x) {
  return (float)(int32_t)(x >= 0.0f ? x + 0.5f : x - 0.5f);
}

static inline float exp_f_hardware_model(float x) {
  if (x < -20.0f) return 0.0f;
  if (x > 20.0f) x = 20.0f;
  float scaled = x * 1.4426950408889634f;
  int n = (int)(scaled + (x >= 0.0f ? 0.5f : -0.5f));
  float r = mul_add_f(-(float)n, 0.6931471805599453f, x);
  float r2 = r * r;
  float polynomial = mul_add_f(r, 0.008333333f, 0.04166667f);
  polynomial = mul_add_f(r, polynomial, 0.16666667f);
  polynomial = mul_add_f(r, polynomial, 0.5f);
  polynomial = mul_add_f(r2, polynomial, 1.0f + r);
  union { float f; uint32_t u; } scale;
  int exponent = n + 127;
  if (exponent <= 0) return 0.0f;
  if (exponent >= 255) exponent = 254;
  scale.u = (uint32_t)exponent << 23;
  return polynomial * scale.f;
}

static inline float sigmoid_hardware_model(float x) {
  float denominator = 1.0f + exp_f_hardware_model(-x);
  union { float f; uint32_t u; } reciprocal = {denominator};
  reciprocal.u = 0x7ef311c2u - reciprocal.u;
  float y = reciprocal.f;
  y = y * mul_add_f(-denominator, y, 2.0f);
  y = y * mul_add_f(-denominator, y, 2.0f);
  return y;
}


struct tensor {
  const uint8_t *data;
  uint32_t ne0;
  uint32_t ne1;
  uint32_t type;
};

struct layer {
  struct tensor attn_norm;
  struct tensor q;
  struct tensor k;
  struct tensor v;
  struct tensor q_norm;
  struct tensor k_norm;
  struct tensor attn_out;
  struct tensor ffn_norm;
  struct tensor ffn_up;
  struct tensor ffn_gate;
  struct tensor ffn_down;
};

struct model {
  const uint8_t *base;
  size_t size;
  struct tensor embedding;
  struct tensor output_norm;
  struct layer layers[BONSAI_LAYERS];
};

struct run_state {
  float x[BONSAI_DIM];
  float norm[BONSAI_DIM];
  float op[BONSAI_DIM];
  int32_t q_fixed[BONSAI_DIM];
  float att[BONSAI_DIM];
  float ffn_up[BONSAI_FF_DIM];
  float ffn_gate[BONSAI_FF_DIM];
  int32_t kv_k[BONSAI_LAYERS][BONSAI_MAX_CONTEXT][BONSAI_KV_DIM];
  int32_t kv_v[BONSAI_LAYERS][BONSAI_MAX_CONTEXT][BONSAI_KV_DIM];
};

static struct model g_model;
static int g_model_packed;
static struct run_state g_state;
static float g_logits[BONSAI_VOCAB];
static int32_t g_q_norm_fixed[BONSAI_LAYERS][BONSAI_HEAD_DIM];
static int32_t g_k_norm_fixed[BONSAI_LAYERS][BONSAI_HEAD_DIM];
static uint64_t g_t_attn, g_t_ffn, g_t_norm, g_t_head;
static uint64_t g_forward_count;
static uint64_t g_job_ns[17], g_job_count[17];
static uint64_t g_job_main_ns[17], g_job_wait_ns[17];
static uint64_t g_fused_transform_ns[17], g_fused_quantize_ns[17];
static int g_profile;
static int g_trace;
static int g_accelerator_float_output;
static int g_accelerator_compact_output;
static uint32_t g_parallel_rows = 16384u;
static uint32_t g_layer_count = BONSAI_LAYERS;
static int g_ffn_shadow;
static int g_norm_shadow;
static int g_attn_shadow;
static uint32_t g_hw_attention_skip_mask;
static int g_residual_resident;
static int g_residual_cpu_visibility;
static uint64_t g_ffn_shadow_bytes;
static uint64_t g_ffn_shadow_mismatches;
static uint64_t g_ffn_shadow_exact_mismatches;
static uint64_t g_ffn_shadow_scale_bytes;
static uint64_t g_ffn_shadow_scale_mismatches;
static uint64_t g_ffn_shadow_q_mismatches;
static uint32_t g_ffn_shadow_max_q_delta;
static float g_ffn_shadow_values[BONSAI_FF_DIM];
static uint64_t g_ffn_shadow_packed[(BONSAI_FF_DIM / 128u) * 17u]
    __attribute__((aligned(64)));
static uint64_t g_ffn_shadow_exact_packed[(BONSAI_FF_DIM / 128u) * 17u]
    __attribute__((aligned(64)));
static uint64_t g_norm_shadow_packed[(BONSAI_DIM / 128u) * 17u]
    __attribute__((aligned(64)));
static uint64_t g_norm_shadow_bytes;
static uint64_t g_norm_shadow_mismatches;
static int g_norm_shadow_dumped;
static uint64_t g_attn_shadow_bytes;
static uint64_t g_attn_shadow_mismatches;
static uint64_t g_attn_shadow_packed[(BONSAI_DIM / 128u) * 17u]
    __attribute__((aligned(64)));

static void fail(const char *message) {
  fprintf(stderr, "bonsai-lite: %s\n", message);
  exit(1);
}

static uint64_t now_ns(void) {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return (uint64_t)t.tv_sec * 1000000000ull + (uint64_t)t.tv_nsec;
}

static void report_main_profile(void) {
  if (!g_profile || g_forward_count == 0u) return;
  fprintf(stderr,
          "Bonsai CPU profile: forwards=%llu attention=%.3fms "
          "feed_forward=%.3fms norms=%.3fms heads=%.3fms\n",
          (unsigned long long)g_forward_count, (double)g_t_attn / 1e6,
          (double)g_t_ffn / 1e6, (double)g_t_norm / 1e6,
          (double)g_t_head / 1e6);
  fprintf(stderr,
          "Bonsai worker profile: attn_heads=%.3fms/%llu "
          "accel_argmax=%.3fms/%llu read_rows=%.3fms/%llu "
          "quantize=%.3fms/%llu attn_prep=%.3fms/%llu "
          "ffn_activate=%.3fms/%llu norm_sum=%.3fms/%llu "
          "norm_apply=%.3fms/%llu norm_apply_quantize=%.3fms/%llu "
          "ffn_activate_quantize=%.3fms/%llu "
          "attn_heads_quantize=%.3fms/%llu "
          "read_rows_add_sum=%.3fms/%llu\n",
          (double)g_job_ns[1] / 1e6, (unsigned long long)g_job_count[1],
          (double)g_job_ns[3] / 1e6, (unsigned long long)g_job_count[3],
          (double)g_job_ns[4] / 1e6, (unsigned long long)g_job_count[4],
          (double)g_job_ns[6] / 1e6, (unsigned long long)g_job_count[6],
          (double)g_job_ns[7] / 1e6, (unsigned long long)g_job_count[7],
          (double)g_job_ns[8] / 1e6, (unsigned long long)g_job_count[8],
          (double)g_job_ns[10] / 1e6, (unsigned long long)g_job_count[10],
          (double)g_job_ns[11] / 1e6, (unsigned long long)g_job_count[11],
          (double)g_job_ns[12] / 1e6, (unsigned long long)g_job_count[12],
          (double)g_job_ns[13] / 1e6, (unsigned long long)g_job_count[13],
          (double)g_job_ns[14] / 1e6, (unsigned long long)g_job_count[14],
          (double)g_job_ns[15] / 1e6, (unsigned long long)g_job_count[15]);
  fprintf(stderr,
          "Bonsai worker split: attn_prep main=%.3fms wait=%.3fms "
          "norm_quant main=%.3fms wait=%.3fms "
          "ffn_quant main=%.3fms wait=%.3fms "
          "heads_quant main=%.3fms wait=%.3fms "
          "residual main=%.3fms wait=%.3fms\n",
          (double)g_job_main_ns[7] / 1e6, (double)g_job_wait_ns[7] / 1e6,
          (double)g_job_main_ns[12] / 1e6, (double)g_job_wait_ns[12] / 1e6,
          (double)g_job_main_ns[13] / 1e6, (double)g_job_wait_ns[13] / 1e6,
          (double)g_job_main_ns[14] / 1e6, (double)g_job_wait_ns[14] / 1e6,
          (double)g_job_main_ns[15] / 1e6, (double)g_job_wait_ns[15] / 1e6);
  fprintf(stderr,
          "Bonsai fused split: norm transform=%.3fms quantize=%.3fms "
          "ffn transform=%.3fms quantize=%.3fms "
          "heads transform=%.3fms quantize=%.3fms\n",
          (double)g_fused_transform_ns[12] / 1e6,
          (double)g_fused_quantize_ns[12] / 1e6,
          (double)g_fused_transform_ns[13] / 1e6,
          (double)g_fused_quantize_ns[13] / 1e6,
          (double)g_fused_transform_ns[14] / 1e6,
          (double)g_fused_quantize_ns[14] / 1e6);
}

static void report_ffn_shadow(void) {
  if (!g_ffn_shadow) return;
  fprintf(stderr,
          "Bonsai FFN hardware shadow: mismatched_bytes=%llu/%llu "
          "(%.9f%%) exact_repack_mismatches=%llu "
          "scale_mismatches=%llu/%llu q_mismatches=%llu "
          "max_q_delta=%u\n",
          (unsigned long long)g_ffn_shadow_mismatches,
          (unsigned long long)g_ffn_shadow_bytes,
          g_ffn_shadow_bytes == 0u ? 0.0 :
              100.0 * (double)g_ffn_shadow_mismatches /
                  (double)g_ffn_shadow_bytes,
          (unsigned long long)g_ffn_shadow_exact_mismatches,
          (unsigned long long)g_ffn_shadow_scale_mismatches,
          (unsigned long long)g_ffn_shadow_scale_bytes,
          (unsigned long long)g_ffn_shadow_q_mismatches,
          g_ffn_shadow_max_q_delta);
}

static void report_norm_shadow(void) {
  if (!g_norm_shadow) return;
  fprintf(stderr, "Bonsai norm shadow: bytes=%llu mismatches=%llu\n",
          (unsigned long long)g_norm_shadow_bytes,
          (unsigned long long)g_norm_shadow_mismatches);
}

static void report_attn_shadow(void) {
  if (!g_attn_shadow) return;
  fprintf(stderr,
          "Bonsai attention shadow: bytes=%llu mismatches=%llu (%.9f%%)\n",
          (unsigned long long)g_attn_shadow_bytes,
          (unsigned long long)g_attn_shadow_mismatches,
          g_attn_shadow_bytes == 0u ? 0.0 :
              100.0 * (double)g_attn_shadow_mismatches /
                  (double)g_attn_shadow_bytes);
}

static uint32_t half_to_float_bits(fp16_t h) {
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

static float half_to_float(fp16_t h) {
  uint32_t bits = half_to_float_bits(h);
  float out;
  memcpy(&out, &bits, sizeof(out));
  return out;
}

static fp16_t float_bits_to_half(uint32_t bits) {
  uint32_t sign = (bits >> 16) & 0x8000u;
  int32_t exponent = (int32_t)((bits >> 23) & 0xffu) - 127 + 15;
  uint32_t mantissa = bits & 0x7fffffu;
  if (exponent <= 0) return (fp16_t)sign;
  if (exponent >= 0x1f) return (fp16_t)(sign | 0x7c00u);
  return (fp16_t)(sign | ((uint32_t)exponent << 10) | (mantissa >> 13));
}

static fp16_t float_to_half(float value) {
  uint32_t bits;
  memcpy(&bits, &value, sizeof(bits));
  return float_bits_to_half(bits);
}

static uint32_t truncate_f32_bits_through_half(uint32_t bits) {
  return half_to_float_bits(float_bits_to_half(bits));
}

/* Form Q8_0 scales and samples from IEEE-754 fields using integer arithmetic.
   The custom CVA6 FPU can produce inconsistent results for the dependent
   reciprocal/multiply/convert sequence used by the conventional quantizer.
   Positive finite binary32 magnitudes retain numeric ordering as integers,
   and the ratio x/max is exact as a ratio of 24-bit significands. */
static fp16_t q8_scale_half_from_bits(uint32_t maximum_bits) {
  uint32_t exponent_field = maximum_bits >> 23;
  uint32_t significand = (maximum_bits & UINT32_C(0x007fffff)) |
                         UINT32_C(0x00800000);
  if (exponent_field == 0u) return 0u;
  int exponent = (int)exponent_field - 127;
  uint32_t divided = significand / 127u;
  int leading = 31 - __builtin_clz(divided);
  int half_exponent = exponent - 23 + leading;
  if (half_exponent < -14) {
    int shift = exponent + 1;
    uint64_t numerator = significand;
    uint64_t denominator = 127u;
    if (shift >= 0)
      numerator <<= shift;
    else if (-shift < 56)
      denominator <<= -shift;
    else
      return 0u;
    uint64_t subnormal = numerator / denominator;
    return (fp16_t)(subnormal > 1023u ? 1023u : subnormal);
  }
  if (half_exponent > 15) return UINT16_C(0x7c00);
  uint32_t half_significand =
      significand / (127u << (leading - 10));
  return (fp16_t)(((uint32_t)(half_exponent + 15) << 10) |
                  (half_significand - 1024u));
}

static uint64_t q8_reciprocal_from_bits(uint32_t maximum_bits) {
  uint32_t maximum_significand =
      (maximum_bits & UINT32_C(0x007fffff)) | UINT32_C(0x00800000);
  return ((UINT64_C(127) << 40) + maximum_significand / 2u) /
         maximum_significand;
}

static int8_t q8_sample_from_bits(uint32_t value_bits,
                                  uint32_t maximum_bits,
                                  uint64_t reciprocal) {
  uint32_t magnitude = value_bits & UINT32_C(0x7fffffff);
  if (magnitude == 0u || (maximum_bits >> 23) == 0u) return 0;
  uint32_t value_exponent = magnitude >> 23;
  uint32_t maximum_exponent = maximum_bits >> 23;
  if (value_exponent == 0u) return 0;
  uint32_t value_significand =
      (magnitude & UINT32_C(0x007fffff)) | UINT32_C(0x00800000);
  uint32_t exponent_difference = maximum_exponent - value_exponent;
  if (exponent_difference > 23u) return 0;
  uint32_t shift = 40u + exponent_difference;
  uint64_t scaled = (uint64_t)value_significand * reciprocal;
  uint32_t quantized =
      (uint32_t)((scaled + (UINT64_C(1) << (shift - 1u))) >> shift);
  if (quantized > 127u) quantized = 127u;
  return (value_bits >> 31) != 0u ? -(int8_t)quantized : (int8_t)quantized;
}

static int64_t fp32_bits_to_q16(uint32_t bits) {
  uint32_t magnitude = bits & UINT32_C(0x7fffffff);
  uint32_t exponent_field = magnitude >> 23;
  if (exponent_field == 0u) return 0;
  if (exponent_field == 0xffu)
    return (bits >> 31) != 0u ? -INT64_C(2147418112) :
                                INT64_C(2147418112);
  uint64_t significand =
      (magnitude & UINT32_C(0x007fffff)) | UINT32_C(0x00800000);
  int shift = (int)exponent_field - 127 - 7;
  uint64_t fixed;
  if (shift >= 0)
    fixed = shift > 7 ? UINT64_C(2147418112) : significand << shift;
  else
    fixed = -shift >= 32 ? 0u : significand >> -shift;
  if (fixed > UINT64_C(2147418112)) fixed = UINT64_C(2147418112);
  return (bits >> 31) != 0u ? -(int64_t)fixed : (int64_t)fixed;
}

/* Deterministic sigmoid in Q16, linearly interpolated at quarter-unit
   intervals. The tails are clamped where binary16/Q8 quantization makes the
   remaining error irrelevant. Unlike a simple rational approximation, this
   still suppresses the very large negative FFN gates seen in this model. */
static uint32_t sigmoid_q16(int64_t value) {
  static const uint16_t table[65] = {
    22, 28, 36, 47, 60, 77, 98, 126,
    162, 208, 267, 342, 439, 562, 720, 922,
    1179, 1506, 1921, 2446, 3108, 3938, 4971, 6249,
    7812, 9702, 11955, 14595, 17625, 21025, 24743, 28693,
    32768, 36843, 40793, 44511, 47911, 50941, 53581, 55834,
    57724, 59287, 60565, 61598, 62428, 63090, 63615, 64030,
    64357, 64614, 64816, 64974, 65097, 65194, 65269, 65328,
    65374, 65410, 65438, 65459, 65476, 65489, 65500, 65508,
    65514,
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

static inline float sigmoid_lut_f(float value) {
  uint32_t bits;
  memcpy(&bits, &value, sizeof(bits));
  return (float)(int32_t)sigmoid_q16(fp32_bits_to_q16(bits)) *
         (1.0f / 65536.0f);
}

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
  uint64_t magnitude = value < 0 ? (uint64_t)(-value) : (uint64_t)value;
  /* K-cache compatibility is intentionally bounded to +/-256.  The old
     left-shift conversion underflowed its unsigned shift count for Q16
     magnitudes above 2^24 and happened to produce exactly this clamp on
     CVA6.  Make the coherent, validated model semantics explicit instead of
     relying on target-specific undefined behavior. */
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
              (int64_t)n * INT64_C(744261118); /* ln(2) in Q30 */
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

static int64_t multiply_q16(int64_t a, int64_t b) {
  int64_t product = a * b;
  return product >= 0 ?
      (product + INT64_C(32768)) >> 16 :
      -((-product + INT64_C(32768)) >> 16);
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

static uint32_t divide_f32_bits(uint32_t a, uint32_t b) {
  uint32_t sign = (a ^ b) & UINT32_C(0x80000000);
  uint32_t magnitude_a = a & UINT32_C(0x7fffffff);
  uint32_t magnitude_b = b & UINT32_C(0x7fffffff);
  uint32_t exponent_a = magnitude_a >> 23;
  uint32_t exponent_b = magnitude_b >> 23;
  if (exponent_a == 0u) return sign;
  if (exponent_b == 0u) return sign | UINT32_C(0x7f800000);
  uint32_t significand_a = (magnitude_a & UINT32_C(0x007fffff)) |
                           UINT32_C(0x00800000);
  uint32_t significand_b = (magnitude_b & UINT32_C(0x007fffff)) |
                           UINT32_C(0x00800000);
  uint32_t below_one = significand_a < significand_b;
  uint32_t shift = 23u + below_one;
  uint64_t numerator = (uint64_t)significand_a << shift;
  uint32_t quotient = (uint32_t)(numerator / significand_b);
  uint32_t remainder = (uint32_t)(numerator % significand_b);
  if ((uint64_t)remainder * 2u > significand_b ||
      ((uint64_t)remainder * 2u == significand_b &&
       (quotient & 1u) != 0u))
    ++quotient;
  int exponent = (int)exponent_a - (int)exponent_b + 127 -
                 (int)below_one;
  if (quotient == UINT32_C(0x01000000)) {
    quotient >>= 1;
    ++exponent;
  }
  if (exponent <= 0) return sign;
  if (exponent >= 255) return sign | UINT32_C(0x7f800000);
  return sign | ((uint32_t)exponent << 23) |
         (quotient & UINT32_C(0x007fffff));
}

static uint32_t signed_integer_to_f32_bits(int32_t value) {
  if (value == 0) return 0u;
  uint32_t sign = value < 0 ? UINT32_C(0x80000000) : 0u;
  uint32_t magnitude = value < 0 ? (uint32_t)(-(int64_t)value) :
                                   (uint32_t)value;
  uint32_t leading = 31u - (uint32_t)__builtin_clz(magnitude);
  uint32_t significand = magnitude << (23u - leading);
  return sign | ((leading + 127u) << 23) |
         (significand & UINT32_C(0x007fffff));
}

static inline uint32_t f32_to_bits(float value) {
  uint32_t bits;
  memcpy(&bits, &value, sizeof(bits));
  return bits;
}

static inline float f32_from_bits(uint32_t bits) {
  float value;
  memcpy(&value, &bits, sizeof(value));
  return value;
}

static uint32_t reliable_ffn_activation_bits(uint32_t up_bits,
                                             uint32_t gate_bits) {
  uint32_t sigmoid = sigmoid_q16(fp32_bits_to_q16(gate_bits));
  uint32_t gated_bits = multiply_f32_bits(
      gate_bits, positive_q16_to_f32_bits(sigmoid));
  return multiply_f32_bits(up_bits, gated_bits);
}

static float reliable_ffn_activation(float up, float gate) {
  uint32_t up_bits;
  uint32_t gate_bits;
  memcpy(&up_bits, &up, sizeof(up_bits));
  memcpy(&gate_bits, &gate, sizeof(gate_bits));
  uint32_t result_bits = reliable_ffn_activation_bits(up_bits, gate_bits);
  float result;
  memcpy(&result, &result_bits, sizeof(result));
  return result;
}

static inline float truncate_through_half(float value) {
  return half_to_float(float_to_half(value));
}

#ifdef BONSAI_HOST_STUB
static float stub_reference_row(const uint8_t *base, uint32_t columns,
                                const float *input, uint32_t row) {
  const uint32_t blocks = columns / 128u;
  const uint8_t *r = base + (size_t)row * blocks * 18u;
  float total = 0.0f;
  for (uint32_t b = 0; b < blocks; ++b) {
    const uint8_t *scale;
    const uint8_t *signs;
    if (g_model_packed) {
      const uint8_t *group = r + (b / 4u) * 72u;
      scale = group + (b % 4u) * 2u;
      signs = group + 8u + (b % 4u) * 16u;
    } else {
      scale = r + b * 18u;
      signs = scale + 2u;
    }
    fp16_t d_bits;
    memcpy(&d_bits, scale, 2);
    float d = half_to_float(d_bits);
    float dot = 0.0f;
    for (uint32_t i = 0; i < 128u; ++i) {
      uint32_t bit = (signs[i / 8u] >> (i % 8u)) & 1u;
      float v = input[b * 128u + i];
      dot += bit ? v : -v;
    }
    total += dot * d;
  }
  return total;
}

#ifdef BONSAI_HOST_Q8_STUB
static float stub_quantized_reference_row(const uint8_t *base,
                                          uint32_t columns,
                                          const float *input, uint32_t row) {
  const uint32_t blocks = columns / 128u;
  const uint8_t *r = base + (size_t)row * blocks * 18u;
  float total = 0.0f;
  for (uint32_t block = 0; block < blocks; ++block) {
    const uint8_t *group = r + (block / 4u) * 72u;
    const uint8_t *weight_scale = group + (block % 4u) * 2u;
    const uint8_t *signs = group + 8u + (block % 4u) * 16u;
    fp16_t weight_scale_bits;
    memcpy(&weight_scale_bits, weight_scale, sizeof(weight_scale_bits));
    float block_total = 0.0f;
    for (uint32_t sub = 0; sub < 4u; ++sub) {
      uint32_t input_base = block * 128u + sub * 32u;
      uint32_t maximum_bits = 0u;
      for (uint32_t i = 0; i < 32u; ++i) {
        uint32_t bits;
        memcpy(&bits, input + input_base + i, sizeof(bits));
        bits &= UINT32_C(0x7fffffff);
        if (maximum_bits < bits) maximum_bits = bits;
      }
      fp16_t input_scale_bits = q8_scale_half_from_bits(maximum_bits);
      uint64_t reciprocal = q8_reciprocal_from_bits(maximum_bits);
      int32_t dot = 0;
      for (uint32_t i = 0; i < 32u; ++i) {
        uint32_t bits;
        memcpy(&bits, input + input_base + i, sizeof(bits));
        int32_t sample = q8_sample_from_bits(bits, maximum_bits, reciprocal);
        uint32_t element = sub * 32u + i;
        uint32_t sign = (signs[element / 8u] >> (element % 8u)) & 1u;
        dot += sign ? sample : -sample;
      }
      block_total += (float)dot * half_to_float(input_scale_bits);
    }
    total += block_total * half_to_float(weight_scale_bits);
  }
  return total;
}
#endif

static float stub_matvec_row(const uint8_t *base, uint32_t columns,
                             const float *input, uint32_t row) {
#ifdef BONSAI_HOST_Q8_STUB
  return stub_quantized_reference_row(base, columns, input, row);
#else
  return stub_reference_row(base, columns, input, row);
#endif
}

static const void *stub_weights;
static uint32_t stub_columns;
static const float *stub_input;
static int64_t stub_stored[BONSAI_VOCAB];

static int64_t stub_float_to_q30(float value) {
  return (int64_t)llroundf(value * 1073741824.0f);
}

int bonsai_accelerate_f32_matvec_submit(const void *weights, uint32_t columns,
                                        uint32_t rows, const float *input) {
  stub_weights = weights;
  stub_columns = columns;
  stub_input = input;
  for (uint32_t row = 0; row < rows; ++row)
    stub_stored[row] = stub_float_to_q30(stub_matvec_row(
        (const uint8_t *)weights, columns, input, row));
  return 1;
}

void bonsai_accelerate_read_rows(float *output, uint32_t first,
                                 uint32_t count) {
  for (uint32_t i = 0; i < count; ++i)
    output[first + i] = stub_matvec_row(
        (const uint8_t *)stub_weights, stub_columns, stub_input, first + i);
}

void bonsai_accelerate_release(void) {}

int bonsai_accelerate_f32_matvec_store(const void *weights, uint32_t columns,
                                       uint32_t rows, const float *input,
                                       uint32_t output_first) {
  for (uint32_t row = 0; row < rows; ++row)
    stub_stored[output_first + row] = stub_float_to_q30(stub_matvec_row(
        (const uint8_t *)weights, columns, input, row));
  return 1;
}

void bonsai_accelerate_read_stored_rows(float *output, uint32_t source_first,
                                        uint32_t count) {
  for (uint32_t i = 0; i < count; ++i)
    output[i] = (float)stub_stored[source_first + i] *
                (1.0f / 1073741824.0f);
}

float bonsai_accelerate_read_rows_add_sum(float *residual, uint32_t first,
                                          uint32_t count) {
  float sum = 0.0f;
  for (uint32_t row = first; row < first + count; ++row) {
    float value = residual[row] +
                  (float)stub_stored[row] * (1.0f / 1073741824.0f);
    residual[row] = value;
    sum += value * value;
  }
  return sum;
}

volatile uint64_t *bonsai_accelerator_packed_input(void) { return NULL; }
void bonsai_accelerator_mark_packed_input(const float *input,
                                          uint32_t columns) {
  (void)input;
  (void)columns;
}
void bonsai_accelerator_output_fence(void) {}
int bonsai_accelerator_pack_ffn(uint32_t elements) {
  (void)elements;
  return 0;
}
int bonsai_accelerator_pack_norm(const float *input, const float *weight,
                                 float scale, uint32_t elements) {
  (void)input;
  (void)weight;
  (void)scale;
  (void)elements;
  return 0;
}
int bonsai_accelerator_pack_resident_norm(const float *weight, float scale,
                                          uint32_t elements) {
  (void)weight;
  (void)scale;
  (void)elements;
  return 0;
}
int bonsai_accelerate_f32_matvec_residual(
    const void *weights, uint32_t columns, uint32_t rows, const float *input,
    const float *next_norm_weight, float *sum_out) {
  (void)weights;
  (void)columns;
  (void)rows;
  (void)input;
  (void)next_norm_weight;
  (void)sum_out;
  return 0;
}
int bonsai_accelerator_pack_vector(const float *input, uint32_t elements) {
  (void)input;
  (void)elements;
  return 0;
}
int bonsai_accelerator_attention(uint32_t layer, uint32_t position,
                                 const int32_t *q_weight,
                                 const int32_t *k_weight,
                                 const int32_t *rope_cos,
                                 const int32_t *rope_sin,
                                 float *probe_output) {
  (void)layer;
  (void)position;
  (void)q_weight;
  (void)k_weight;
  (void)rope_cos;
  (void)rope_sin;
  (void)probe_output;
  return 0;
}

const volatile int64_t *bonsai_accelerate_stored_output(void) {
  return stub_stored;
}

void bonsai_accelerate_argmax_rows(uint32_t first, uint32_t count,
                                   int64_t *best_raw, uint32_t *best_token) {
  int64_t score = INT64_MIN;
  uint32_t token = first;
  for (uint32_t row = first; row < first + count; ++row) {
    int64_t value = stub_float_to_q30(stub_matvec_row(
        (const uint8_t *)stub_weights, stub_columns, stub_input, row));
    if (value > score) {
      score = value;
      token = row;
    }
  }
  *best_raw = score;
  *best_token = token;
}

int bonsai_accelerate_f32_matvec_argmax(const void *weights,
                                        uint32_t columns, uint32_t rows,
                                        const float *input,
                                        uint32_t *best_token) {
  int64_t best_raw = INT64_MIN;
  *best_token = 0u;
  for (uint32_t row = 0; row < rows; ++row) {
    int64_t value = stub_float_to_q30(stub_matvec_row(
        (const uint8_t *)weights, columns, input, row));
    if (value > best_raw) {
      best_raw = value;
      *best_token = row;
    }
  }
  return 1;
}

void quantize_row_q8_0_unused_host(void);
#endif

/* ------------------------------------------------------------------ */
/* GGUF parsing: just enough of v3 to find tensor data by name.        */

struct cursor {
  const uint8_t *p;
  const uint8_t *end;
};

static uint64_t take_u64(struct cursor *c) {
  if (c->p + 8 > c->end) fail("gguf truncated");
  uint64_t v;
  memcpy(&v, c->p, 8);
  c->p += 8;
  return v;
}

static uint32_t take_u32(struct cursor *c) {
  if (c->p + 4 > c->end) fail("gguf truncated");
  uint32_t v;
  memcpy(&v, c->p, 4);
  c->p += 4;
  return v;
}

static void skip_value(struct cursor *c, uint32_t type);

static void skip_string(struct cursor *c) {
  uint64_t n = take_u64(c);
  if (c->p + n > c->end) fail("gguf truncated");
  c->p += n;
}

static void skip_value(struct cursor *c, uint32_t type) {
  switch (type) {
    case 0: case 1: c->p += 1; break;
    case 2: case 3: c->p += 2; break;
    case 4: case 5: case 6: c->p += 4; break;
    case 10: case 11: case 12: c->p += 8; break;
    case 7: c->p += 1; break;
    case 8: skip_string(c); break;
    case 9: {
      uint32_t item_type = take_u32(c);
      uint64_t count = take_u64(c);
      for (uint64_t i = 0; i < count; ++i) skip_value(c, item_type);
      break;
    }
    default: fail("gguf metadata type");
  }
}

static void assign_model_tensor(struct model *model, const char *name,
                                struct tensor t) {
  if (strcmp(name, "token_embd.weight") == 0) model->embedding = t;
  else if (strcmp(name, "output_norm.weight") == 0) model->output_norm = t;
  else if (strncmp(name, "blk.", 4) == 0) {
    char *end;
    unsigned long layer_index = strtoul(name + 4, &end, 10);
    if (layer_index >= BONSAI_LAYERS || *end != '.') fail("bad layer");
    struct layer *layer = &model->layers[layer_index];
    const char *suffix = end + 1;
    if (strcmp(suffix, "attn_norm.weight") == 0) layer->attn_norm = t;
    else if (strcmp(suffix, "attn_q.weight") == 0) layer->q = t;
    else if (strcmp(suffix, "attn_k.weight") == 0) layer->k = t;
    else if (strcmp(suffix, "attn_v.weight") == 0) layer->v = t;
    else if (strcmp(suffix, "attn_q_norm.weight") == 0) layer->q_norm = t;
    else if (strcmp(suffix, "attn_k_norm.weight") == 0) layer->k_norm = t;
    else if (strcmp(suffix, "attn_output.weight") == 0) layer->attn_out = t;
    else if (strcmp(suffix, "ffn_norm.weight") == 0) layer->ffn_norm = t;
    else if (strcmp(suffix, "ffn_up.weight") == 0) layer->ffn_up = t;
    else if (strcmp(suffix, "ffn_gate.weight") == 0) layer->ffn_gate = t;
    else if (strcmp(suffix, "ffn_down.weight") == 0) layer->ffn_down = t;
    else fail("unknown layer tensor");
  } else {
    fail("unknown tensor");
  }
}

static void model_init(struct model *model, const uint8_t *base, size_t size) {
  model->base = base;
  model->size = size;
  struct cursor c = { base, base + size };
  uint32_t magic = take_u32(&c);
  if (magic != 0x46554747u && magic != 0x46554742u) fail("not gguf");
  g_model_packed = magic == 0x46554742u;
  const char *disable_manifest = getenv("BONSAI_NO_MANIFEST");
  if (size == (size_t)BONSAI_MODEL_FILE_SIZE &&
      (disable_manifest == NULL || strcmp(disable_manifest, "1") != 0)) {
    for (size_t i = 0u; i < BONSAI_MODEL_ENTRY_COUNT; ++i) {
      const struct bonsai_model_entry *entry = &bonsai_model_entries[i];
      if (entry->offset >= size) fail("manifest tensor offset");
      struct tensor t = {
        .data = base + entry->offset,
        .ne0 = entry->ne0,
        .ne1 = entry->ne1,
        .type = entry->type,
      };
      assign_model_tensor(model, entry->name, t);
    }
    return;
  }
  (void)take_u32(&c);
  uint64_t tensor_count = take_u64(&c);
  uint64_t kv_count = take_u64(&c);
  for (uint64_t i = 0; i < kv_count; ++i) {
    skip_string(&c);
    uint32_t type = take_u32(&c);
    skip_value(&c, type);
  }
  struct pending {
    char name[64];
    uint64_t offset;
    uint32_t ne0;
    uint32_t ne1;
    uint32_t type;
  };
  static struct pending pending[512];
  if (tensor_count > 512) fail("too many tensors");
  for (uint64_t i = 0; i < tensor_count; ++i) {
    uint64_t name_length = take_u64(&c);
    if (name_length >= sizeof(pending[i].name)) fail("tensor name too long");
    if (c.p + name_length > c.end) fail("gguf truncated");
    memcpy(pending[i].name, c.p, name_length);
    pending[i].name[name_length] = 0;
    c.p += name_length;
    uint32_t dims = take_u32(&c);
    uint64_t ne[4] = {1, 1, 1, 1};
    for (uint32_t d = 0; d < dims; ++d) ne[d] = take_u64(&c);
    pending[i].ne0 = (uint32_t)ne[0];
    pending[i].ne1 = (uint32_t)ne[1];
    pending[i].type = take_u32(&c);
    pending[i].offset = take_u64(&c);
  }
  /* Tensor data starts at the next 32-byte alignment boundary. */
  uint64_t data_start = (uint64_t)(c.p - base);
  data_start = (data_start + 31u) & ~31ull;

  for (uint64_t i = 0; i < tensor_count; ++i) {
    struct tensor t;
    t.data = base + data_start + pending[i].offset;
    t.ne0 = pending[i].ne0;
    t.ne1 = pending[i].ne1;
    t.type = pending[i].type;
    assign_model_tensor(model, pending[i].name, t);
  }
}

/* ------------------------------------------------------------------ */
/* YaRN rotary cache, ported from ggml (MIT, Quesnelle & Peng).        */

static void rope_cache_init(void) {}

static void attention_fixed_cache_init(void) {
  for (uint32_t layer_index = 0; layer_index < BONSAI_LAYERS;
       ++layer_index) {
    const struct layer *layer = &g_model.layers[layer_index];
    if (layer->q_norm.type != GGUF_TYPE_F32 ||
        layer->q_norm.ne0 != BONSAI_HEAD_DIM ||
        layer->k_norm.type != GGUF_TYPE_F32 ||
        layer->k_norm.ne0 != BONSAI_HEAD_DIM)
      fail("bad head norm");
    const float *q_weight = (const float *)layer->q_norm.data;
    const float *k_weight = (const float *)layer->k_norm.data;
    for (uint32_t i = 0; i < BONSAI_HEAD_DIM; ++i) {
      g_q_norm_fixed[layer_index][i] =
          (int32_t)fp32_bits_to_q16(f32_to_bits(q_weight[i]));
      g_k_norm_fixed[layer_index][i] =
          (int32_t)fp32_bits_to_q16(f32_to_bits(k_weight[i]));
    }
  }
}

/* ------------------------------------------------------------------ */
/* Worker pool.                                                        */

enum job_kind {
  JOB_NONE = 0,
  JOB_ATTN_HEADS,
  JOB_ARGMAX,
  JOB_ACCEL_ARGMAX,
  JOB_READ_ROWS,
  JOB_READ_STORED_ROWS,
  JOB_QUANTIZE_Q8,
  JOB_ATTN_READ_PREP,
  JOB_FFN_READ_ACTIVATE,
  JOB_VECTOR_ADD,
  JOB_NORM_SUM,
  JOB_NORM_APPLY,
  JOB_NORM_APPLY_QUANTIZE,
  JOB_FFN_READ_ACTIVATE_QUANTIZE,
  JOB_ATTN_HEADS_QUANTIZE,
  JOB_READ_ROWS_ADD_SUM,
  JOB_ATTN_PREP_HEADS_QUANTIZE,
  JOB_ATTN_READ_Q,
  JOB_ATTN_READ_HARDWARE_KV,
};

struct worker_slot {
  volatile uint32_t done;
  float score;
  uint32_t token;
  int64_t raw;
  uint8_t padding[44];
} __attribute__((aligned(64)));

struct parallel_job {
  volatile uint32_t epoch;
  uint32_t kind;
  uint32_t rows;
  uint32_t position;
  uint32_t workers;
  const struct layer *layer;
  float *out;
  const float *input;
  const float *norm_weight;
  const volatile int64_t *raw;
  float norm_scale;
  const float *quantize_input;
  volatile uint64_t *quantize_output;
  struct worker_slot slots[BONSAI_MAX_WORKERS];
};

static struct parallel_job g_job __attribute__((aligned(64)));
static uint32_t g_worker_count = 1;
static uint32_t g_attention_worker_count = 1;
static uint32_t g_residual_worker_count = 1;
static int g_gqa_attention;
static int g_hardware_kv_prep;
static uint32_t g_thread_count = 1;
static int g_spin_workers = 1;
static pthread_t g_threads[BONSAI_MAX_WORKERS - 1];

static inline uint32_t job_load_acquire(volatile uint32_t *address) {
  return __atomic_load_n(address, __ATOMIC_ACQUIRE);
}

static inline void job_store_release(volatile uint32_t *address,
                                     uint32_t value) {
  __atomic_store_n(address, value, __ATOMIC_RELEASE);
}

static uint32_t job_wait_for_change(volatile uint32_t *address,
                                    uint32_t previous) {
  uint32_t polls = 0u;
  for (;;) {
    uint32_t value = __atomic_load_n(address, __ATOMIC_RELAXED);
    if (value != previous) {
      /* Pair with the publishing release store and order the job payload. */
      return job_load_acquire(address);
    }
    /* Refresh periodically in case a peer's invalidation arrives while this
       hart is continuously hitting on the same private-cache line. */
    if ((++polls & 8191u) == 0u) {
      value = job_load_acquire(address);
      if (value != previous) return value;
    }
    __asm__ volatile("nop");
  }
}

static void attention_head_range(uint32_t begin, uint32_t end,
                                 uint32_t position);
static void quantize_q8_group_range(const float *input,
                                    volatile uint64_t *output,
                                    uint32_t begin, uint32_t end);
static void norm_quantize_direct_range(const float *input,
                                       const float *weight, float scale,
                                       volatile uint64_t *output,
                                       uint32_t begin, uint32_t end);
static void ffn_quantize_direct_range(const volatile int64_t *raw,
                                      volatile uint64_t *output,
                                      uint32_t begin, uint32_t end);
static void ffn_shadow_validate_range(uint32_t begin, uint32_t end);
static void norm_shadow_validate(float *out, const float *input,
                                 const float *weight, float scale,
                                 const volatile uint64_t *packed);
static void attention_prep_worker(uint32_t worker, uint32_t workers,
                                  uint32_t position,
                                  const struct layer *layer,
                                  const volatile int64_t *raw);
static void attention_q_worker(uint32_t worker, uint32_t workers,
                               uint32_t position, const struct layer *layer,
                               const volatile int64_t *raw);
static void attention_hardware_kv_worker(uint32_t worker, uint32_t workers,
                                         uint32_t position,
                                         const struct layer *layer);

static inline float q30_to_float(int64_t value) {
  if (g_accelerator_float_output) {
    uint32_t bits = (uint32_t)value;
    float result;
    memcpy(&result, &bits, sizeof(result));
    return result;
  }
  int32_t high = (int32_t)(value >> 32);
  uint32_t low = (uint32_t)value;
  return (float)high * 4.0f + (float)low * (1.0f / 1073741824.0f);
}

static inline uint32_t accelerator_bits_at(const volatile int64_t *raw,
                                           uint32_t index) {
  if (g_accelerator_float_output) {
    return g_accelerator_compact_output ?
        ((const volatile uint32_t *)raw)[index] : (uint32_t)raw[index];
  }
  return f32_to_bits(q30_to_float(raw[index]));
}

static inline float accelerator_float_at(const volatile int64_t *raw,
                                         uint32_t index) {
  uint32_t bits = accelerator_bits_at(raw, index);
  return f32_from_bits(bits);
}

static void execute_job(uint32_t worker) {
  if (worker >= g_job.workers) return;
  /* DONE is observed by the main hart, but accelerator results are consumed
     by every worker hart.  Give each consumer its own device-to-read fence;
     a release/acquire handoff on the cached job epoch does not order a
     different hart's uncached DDR reads against the accelerator's AXI
     writes. */
  if ((g_job.kind == JOB_READ_ROWS ||
      g_job.kind == JOB_READ_ROWS_ADD_SUM ||
      g_job.kind == JOB_READ_STORED_ROWS ||
      g_job.kind == JOB_ATTN_READ_PREP ||
      g_job.kind == JOB_ATTN_PREP_HEADS_QUANTIZE ||
      g_job.kind == JOB_ATTN_READ_Q ||
      g_job.kind == JOB_ATTN_READ_HARDWARE_KV ||
      g_job.kind == JOB_FFN_READ_ACTIVATE ||
      g_job.kind == JOB_FFN_READ_ACTIVATE_QUANTIZE) &&
      !(worker == 0u && g_job.kind == JOB_READ_ROWS_ADD_SUM))
    bonsai_accelerator_output_fence();
  uint32_t begin = (uint32_t)(((uint64_t)g_job.rows * worker) / g_job.workers);
  uint32_t end =
      (uint32_t)(((uint64_t)g_job.rows * (worker + 1u)) / g_job.workers);
  if (g_job.kind == JOB_ATTN_HEADS) {
    if (g_gqa_attention)
      bonsai_attention_gqa_range(
          g_state.att, g_state.q_fixed,
          g_state.kv_k[g_job.layer - g_model.layers][0],
          g_state.kv_v[g_job.layer - g_model.layers][0], begin, end,
          g_job.position);
    else
      attention_head_range(begin, end, g_job.position);
  } else if (g_job.kind == JOB_ARGMAX) {
    uint32_t best = begin;
    float best_score = -3.402823466e38f;
    for (uint32_t row = begin; row < end; ++row) {
      if (g_logits[row] > best_score) {
        best_score = g_logits[row];
        best = row;
      }
    }
    g_job.slots[worker].score = best_score;
    g_job.slots[worker].token = best;
  } else if (g_job.kind == JOB_ACCEL_ARGMAX) {
    bonsai_accelerate_argmax_rows(begin, end - begin,
                                  &g_job.slots[worker].raw,
                                  &g_job.slots[worker].token);
  } else if (g_job.kind == JOB_READ_ROWS) {
    if (begin < end)
      bonsai_accelerate_read_rows(g_job.out, begin, end - begin);
  } else if (g_job.kind == JOB_READ_ROWS_ADD_SUM) {
    g_job.slots[worker].score = begin < end ?
        bonsai_accelerate_read_rows_add_sum(g_job.out, begin, end - begin) :
        0.0f;
  } else if (g_job.kind == JOB_READ_STORED_ROWS) {
    if (begin < end)
      bonsai_accelerate_read_stored_rows(g_job.out + begin, begin,
                                         end - begin);
  } else if (g_job.kind == JOB_QUANTIZE_Q8) {
    if (begin < end)
      quantize_q8_group_range(g_job.quantize_input, g_job.quantize_output,
                              begin, end);
  } else if (g_job.kind == JOB_ATTN_READ_PREP) {
    attention_prep_worker(worker, g_job.workers, g_job.position, g_job.layer,
                          g_job.raw);
  } else if (g_job.kind == JOB_ATTN_READ_Q) {
    attention_q_worker(worker, g_job.workers, g_job.position, g_job.layer,
                       g_job.raw);
  } else if (g_job.kind == JOB_ATTN_READ_HARDWARE_KV) {
    attention_hardware_kv_worker(worker, g_job.workers, g_job.position,
                                 g_job.layer);
  } else if (g_job.kind == JOB_FFN_READ_ACTIVATE) {
    for (uint32_t i = begin; i < end; ++i) {
      float up = accelerator_float_at(g_job.raw, i);
      float gate = accelerator_float_at(g_job.raw, BONSAI_FF_DIM + i);
      g_job.out[i] = reliable_ffn_activation(up, gate);
    }
  } else if (g_job.kind == JOB_FFN_READ_ACTIVATE_QUANTIZE) {
    uint64_t transform_start = g_profile && worker == 0u ? now_ns() : 0u;
    ffn_quantize_direct_range(g_job.raw, g_job.quantize_output, begin, end);
    if (g_profile && worker == 0u) {
      uint64_t quantize_done = now_ns();
      g_fused_quantize_ns[13] += quantize_done - transform_start;
    }
  } else if (g_job.kind == JOB_VECTOR_ADD) {
    for (uint32_t i = begin; i < end; ++i) g_job.out[i] += g_job.input[i];
  } else if (g_job.kind == JOB_NORM_SUM) {
    float sum = 0.0f;
    for (uint32_t i = begin; i < end; ++i) {
      float value = g_job.out[i];
      if (g_job.input != NULL) {
        value += g_job.input[i];
        g_job.out[i] = value;
      }
      sum += value * value;
    }
    g_job.slots[worker].score = sum;
  } else if (g_job.kind == JOB_NORM_APPLY) {
    for (uint32_t i = begin; i < end; ++i)
      g_job.out[i] = g_job.input[i] * g_job.norm_scale * g_job.norm_weight[i];
  } else if (g_job.kind == JOB_NORM_APPLY_QUANTIZE) {
    uint64_t transform_start = g_profile && worker == 0u ? now_ns() : 0u;
    norm_quantize_direct_range(g_job.input, g_job.norm_weight,
                               g_job.norm_scale, g_job.quantize_output,
                               begin, end);
    if (g_profile && worker == 0u) {
      uint64_t quantize_done = now_ns();
      g_fused_quantize_ns[12] += quantize_done - transform_start;
    }
  } else if (g_job.kind == JOB_ATTN_HEADS_QUANTIZE) {
    uint64_t transform_start = g_profile && worker == 0u ? now_ns() : 0u;
    if (g_gqa_attention)
      bonsai_attention_gqa_range(
          g_state.att, g_state.q_fixed,
          g_state.kv_k[g_job.layer - g_model.layers][0],
          g_state.kv_v[g_job.layer - g_model.layers][0], begin, end,
          g_job.position);
    else
      attention_head_range(begin, end, g_job.position);
    uint64_t transform_done = g_profile && worker == 0u ? now_ns() : 0u;
    quantize_q8_group_range(g_state.att, g_job.quantize_output, begin, end);
    if (g_profile && worker == 0u) {
      uint64_t quantize_done = now_ns();
      g_fused_transform_ns[14] += transform_done - transform_start;
      g_fused_quantize_ns[14] += quantize_done - transform_done;
    }
  } else if (g_job.kind == JOB_ATTN_PREP_HEADS_QUANTIZE) {
    attention_prep_worker(worker, g_job.workers, g_job.position, g_job.layer,
                          g_job.raw);
    if (g_gqa_attention)
      bonsai_attention_gqa_range(
          g_state.att, g_state.q_fixed,
          g_state.kv_k[g_job.layer - g_model.layers][0],
          g_state.kv_v[g_job.layer - g_model.layers][0], begin, end,
          g_job.position);
    else
      attention_head_range(begin, end, g_job.position);
    quantize_q8_group_range(g_state.att, g_job.quantize_output, begin, end);
  }
}

static void *worker_start(void *argument) {
  uint32_t worker = (uint32_t)(uintptr_t)argument;
  cpu_set_t affinity;
  CPU_ZERO(&affinity);
  CPU_SET(worker, &affinity);
  (void)pthread_setaffinity_np(pthread_self(), sizeof(affinity), &affinity);
  uint32_t seen = 0;
  for (;;) {
    uint32_t epoch;
    if (!g_spin_workers || worker >= g_worker_count) {
      do {
        epoch = job_load_acquire(&g_job.epoch);
        if (epoch == seen)
          (void)syscall(SYS_futex, &g_job.epoch, FUTEX_WAIT_PRIVATE, seen,
                        NULL, NULL, 0);
      } while (epoch == seen);
    } else {
      epoch = job_wait_for_change(&g_job.epoch, seen);
    }
    execute_job(worker);
    job_store_release(&g_job.slots[worker].done, epoch);
    seen = epoch;
  }
  return NULL;
}

static void parallel_init(uint32_t workers) {
  if (workers < 1u) workers = 1u;
  if (workers > BONSAI_MAX_WORKERS) workers = BONSAI_MAX_WORKERS;
  cpu_set_t affinity;
  CPU_ZERO(&affinity);
  CPU_SET(0, &affinity);
  (void)pthread_setaffinity_np(pthread_self(), sizeof(affinity), &affinity);
  g_worker_count = workers;
  g_attention_worker_count = workers;
  g_residual_worker_count = workers;
  const char *attention_workers = getenv("BONSAI_ATTN_WORKERS");
  if (attention_workers != NULL) {
    unsigned long requested = strtoul(attention_workers, NULL, 10);
    if (requested >= workers && requested <= BONSAI_MAX_WORKERS)
      g_attention_worker_count = (uint32_t)requested;
  }
  const char *residual_workers = getenv("BONSAI_RESIDUAL_WORKERS");
  if (residual_workers != NULL) {
    unsigned long requested = strtoul(residual_workers, NULL, 10);
    if (requested >= workers && requested <= BONSAI_MAX_WORKERS)
      g_residual_worker_count = (uint32_t)requested;
  }
  g_thread_count = g_attention_worker_count > workers ?
      g_attention_worker_count : workers;
  if (g_residual_worker_count > g_thread_count)
    g_thread_count = g_residual_worker_count;
  pthread_attr_t worker_attributes;
  if (pthread_attr_init(&worker_attributes) != 0 ||
      pthread_attr_setstacksize(&worker_attributes, 256u * 1024u) != 0)
    fail("cannot configure worker stack");
  for (uint32_t worker = 1; worker < g_thread_count; ++worker)
    if (pthread_create(&g_threads[worker - 1u], &worker_attributes, worker_start,
                       (void *)(uintptr_t)worker) != 0)
      fail("cannot create worker");
  (void)pthread_attr_destroy(&worker_attributes);
}

static void run_job(uint32_t kind, uint32_t rows, uint32_t position) {
  uint64_t profile_start = g_profile ? now_ns() : 0u;
  g_job.kind = kind;
  g_job.rows = rows;
  g_job.position = position;
  uint32_t workers = g_worker_count;
  if (kind == JOB_ATTN_HEADS || kind == JOB_ATTN_READ_PREP ||
      kind == JOB_ATTN_HEADS_QUANTIZE ||
      kind == JOB_ATTN_PREP_HEADS_QUANTIZE)
    workers = g_attention_worker_count;
  if (kind == JOB_READ_ROWS_ADD_SUM)
    workers = g_residual_worker_count;
  if ((kind == JOB_READ_ROWS) && workers > 2u)
    workers = 2u;
  g_job.workers = workers;
  uint32_t epoch = g_job.epoch + 1u;
  if (epoch == 0u) epoch = 1u;
  job_store_release(&g_job.epoch, epoch);
  if (workers > 1u && (!g_spin_workers || workers > g_worker_count))
    (void)syscall(SYS_futex, &g_job.epoch, FUTEX_WAKE_PRIVATE,
                  (int)(workers - 1u), NULL, NULL, 0);
  uint64_t main_start = g_profile ? now_ns() : 0u;
  execute_job(0);
  uint64_t main_done = g_profile ? now_ns() : 0u;
  job_store_release(&g_job.slots[0].done, epoch);
  for (uint32_t worker = 1; worker < workers; ++worker) {
    uint32_t done = __atomic_load_n(&g_job.slots[worker].done,
                                    __ATOMIC_RELAXED);
    while (done != epoch)
      done = job_wait_for_change(&g_job.slots[worker].done, done);
  }
  if (g_profile && kind < 16u) {
    uint64_t all_done = now_ns();
    g_job_ns[kind] += all_done - profile_start;
    g_job_main_ns[kind] += main_done - main_start;
    g_job_wait_ns[kind] += all_done - main_done;
    ++g_job_count[kind];
  }
}

/* ------------------------------------------------------------------ */
/* Math blocks.                                                        */

static float cpu_reference_row(const struct tensor *weight,
                               const float *input, uint32_t row) {
  /* Float reference of one Q1_0 row: sum over 128-blocks of
     (sign dot) * weight_scale, inputs used directly as f32. */
  const uint32_t blocks = weight->ne0 / 128u;
  const uint8_t *r = weight->data + (size_t)row * blocks * 18u;
  float total = 0.0f;
  for (uint32_t b = 0; b < blocks; ++b) {
    const uint8_t *scale;
    const uint8_t *signs;
    if (g_model_packed) {
      const uint8_t *group = r + (b / 4u) * 72u;
      scale = group + (b % 4u) * 2u;
      signs = group + 8u + (b % 4u) * 16u;
    } else {
      scale = r + b * 18u;
      signs = scale + 2u;
    }
    fp16_t d_bits;
    memcpy(&d_bits, scale, 2);
    float d = half_to_float(d_bits);
    float dot = 0.0f;
    for (uint32_t i = 0; i < 128u; ++i) {
      uint32_t bit = (signs[i / 8u] >> (i % 8u)) & 1u;
      float v = input[b * 128u + i];
      dot += bit ? v : -v;
    }
    total += dot * d;
  }
  return total;
}

static float cpu_reference_q8_row(const struct tensor *weight,
                                  const volatile uint64_t *packed_input,
                                  uint32_t row) {
  const uint32_t blocks = weight->ne0 / 128u;
  const uint8_t *r = weight->data + (size_t)row * blocks * 18u;
  const volatile uint8_t *q8 = (const volatile uint8_t *)packed_input;
  float total = 0.0f;
  for (uint32_t b = 0; b < blocks; ++b) {
    const uint8_t *group = r + (b / 4u) * 72u;
    const uint8_t *scale = group + (b % 4u) * 2u;
    const uint8_t *signs = group + 8u + (b % 4u) * 16u;
    fp16_t weight_scale_bits;
    memcpy(&weight_scale_bits, scale, 2u);
    float block_total = 0.0f;
    for (uint32_t sub = 0; sub < 4u; ++sub) {
      uint32_t q8_block = b * 4u + sub;
      fp16_t input_scale_bits = (fp16_t)q8[q8_block * 34u] |
          (fp16_t)((uint16_t)q8[q8_block * 34u + 1u] << 8);
      int32_t dot = 0;
      for (uint32_t i = 0; i < 32u; ++i) {
        uint32_t element = sub * 32u + i;
        int32_t sample = (int8_t)q8[q8_block * 34u + 2u + i];
        uint32_t sign =
            (signs[element / 8u] >> (element % 8u)) & 1u;
        dot += sign ? sample : -sample;
      }
      block_total += (float)dot * half_to_float(input_scale_bits);
    }
    total += block_total * half_to_float(weight_scale_bits);
  }
  return total;
}

static void matvec_accel(float *out, const struct tensor *weight,
                         const float *input) {
  if (weight->type != GGUF_TYPE_Q1_0) fail("expected Q1_0 tensor");
  if (!bonsai_accelerate_f32_matvec_submit(weight->data, weight->ne0,
                                           weight->ne1, input))
    fail("accelerator declined matvec");
  bonsai_accelerator_output_fence();
  if (weight->ne1 >= g_parallel_rows && g_worker_count > 1u) {
    /* The threshold is tunable on target because the crossover depends on
       uncached DDR latency and worker wakeup cost. */
    g_job.out = out;
    run_job(JOB_READ_ROWS, weight->ne1, 0);
  } else {
    bonsai_accelerate_read_rows(out, 0, weight->ne1);
  }
  bonsai_accelerate_release();
  if (g_trace) {
    static int checked;
    if (!checked) {
      checked = 1;
      float reference = cpu_reference_row(weight, input, 0);
      fprintf(stderr,
              "[trace] matvec check ne0=%u ne1=%u accel[0]=%g cpuref[0]=%g "
              "in0=%g in1=%g\n",
              weight->ne0, weight->ne1, (double)out[0], (double)reference,
              (double)input[0], (double)input[1]);
      fflush(stderr);
    }
  }
}

static void embedding_row(float *out, const struct tensor *embedding,
                          uint32_t token) {
  if (embedding->type != GGUF_TYPE_Q1_0 || token >= embedding->ne1 ||
      embedding->ne0 != BONSAI_DIM)
    fail("bad embedding");
  const uint32_t blocks = BONSAI_DIM / 128u;
  const uint8_t *row = embedding->data + (size_t)token * blocks * 18u;
  for (uint32_t block = 0; block < blocks; ++block) {
    const uint8_t *b;
    const uint8_t *signs;
    if (g_model_packed) {
      const uint8_t *group = row + (block / 4u) * 72u;
      b = group + (block % 4u) * 2u;
      signs = group + 8u + (block % 4u) * 16u;
    } else {
      b = row + block * 18u;
      signs = b + 2u;
    }
    fp16_t d_bits;
    memcpy(&d_bits, b, 2);
    float d = half_to_float(d_bits);
    for (uint32_t i = 0; i < 128u; ++i) {
      uint32_t bit = (signs[i / 8u] >> (i % 8u)) & 1u;
      out[block * 128u + i] = bit ? d : -d;
    }
  }
}

static float reliable_sum_squares(const float *values, uint32_t count) {
  uint32_t sum_bits = 0u;
  for (uint32_t i = 0u; i < count; ++i) {
    uint32_t value_bits = f32_to_bits(values[i]);
    uint32_t square_bits = multiply_f32_bits(value_bits, value_bits);
    sum_bits = add_f32_bits(sum_bits, square_bits);
  }
  return f32_from_bits(sum_bits);
}

float bonsai_reliable_resident_sum_squares(
    const volatile uint32_t *values, uint32_t count) {
  uint32_t sum_first = 0u;
  uint32_t sum_second = 0u;
  uint32_t half = count / 2u;
  for (uint32_t i = 0u; i < count; ++i) {
    uint32_t value_bits = values[i];
    uint32_t square_bits = multiply_f32_bits(value_bits, value_bits);
    if (i < half)
      sum_first = add_f32_bits(sum_first, square_bits);
    else
      sum_second = add_f32_bits(sum_second, square_bits);
  }
  return f32_from_bits(add_f32_bits(sum_first, sum_second));
}

__attribute__((optimize("O3", "no-fast-math", "fp-contract=off")))
float bonsai_fast_resident_sum_squares(
    const volatile uint32_t *values, uint32_t count) {
  float sum_first = 0.0f;
  float sum_second = 0.0f;
  uint32_t half = count / 2u;
  for (uint32_t i = 0u; i < half; ++i) {
    uint32_t bits = values[i];
    float value;
    memcpy(&value, &bits, sizeof(value));
    sum_first += value * value;
  }
  for (uint32_t i = half; i < count; ++i) {
    uint32_t bits = values[i];
    float value;
    memcpy(&value, &bits, sizeof(value));
    sum_second += value * value;
  }
  return sum_first + sum_second;
}

static void rms_norm_add(float *out, float *x, const float *add,
                         const struct tensor *weight, uint32_t n) {
  if (weight->type != GGUF_TYPE_F32 || weight->ne0 != n) fail("bad norm");
  const float *w = (const float *)weight->data;
  float sum;
  if (add == NULL) {
    sum = reliable_sum_squares(x, n);
  } else {
    g_job.out = x;
    g_job.input = add;
    run_job(JOB_NORM_SUM, n, 0u);
    sum = 0.0f;
    for (uint32_t worker = 0; worker < g_job.workers; ++worker)
      sum += g_job.slots[worker].score;
  }
  float scale = rms_scale_f(sum, 11u);
  g_job.out = out;
  g_job.input = x;
  g_job.norm_weight = w;
  g_job.norm_scale = scale;
  volatile uint64_t *packed = bonsai_accelerator_packed_input();
  if (packed != NULL &&
      bonsai_accelerator_pack_norm(x, w, scale, n)) {
    if (g_norm_shadow) norm_shadow_validate(out, x, w, scale, packed);
    bonsai_accelerator_mark_packed_input(out, n);
  } else if (packed != NULL) {
    run_job(JOB_NORM_APPLY, n, 0u);
    g_job.quantize_input = out;
    g_job.quantize_output = packed;
    run_job(JOB_QUANTIZE_Q8, n / 128u, 0u);
    if (g_norm_shadow) norm_shadow_validate(out, x, w, scale, packed);
    bonsai_accelerator_mark_packed_input(out, n);
  } else {
    run_job(JOB_NORM_APPLY, n, 0u);
  }
}

static void rms_norm(float *out, float *x, const struct tensor *weight,
                     uint32_t n) {
  rms_norm_add(out, x, NULL, weight, n);
}

static void rms_norm_from_sum(float *out, float *x,
                              const struct tensor *weight, uint32_t n,
                              float sum) {
  if (weight->type != GGUF_TYPE_F32 || weight->ne0 != n) fail("bad norm");
  g_job.out = out;
  g_job.input = x;
  g_job.norm_weight = (const float *)weight->data;
  g_job.norm_scale = rms_scale_f(sum, 11u);
  volatile uint64_t *packed = bonsai_accelerator_packed_input();
  if (g_residual_resident) {
    if (packed == NULL || !bonsai_accelerator_pack_resident_norm(
                              g_job.norm_weight, g_job.norm_scale, n))
      fail("accelerator declined resident norm packing");
    /* The CPU arrays intentionally remain stale while residual state stays
       in scratch DRAM. The pointer marker tells the following matvec that
       the packed input already contains the current normalized vector. */
    bonsai_accelerator_mark_packed_input(out, n);
  } else if (packed != NULL && bonsai_accelerator_pack_norm(
                            x, g_job.norm_weight, g_job.norm_scale, n)) {
    if (g_norm_shadow)
      norm_shadow_validate(out, x, g_job.norm_weight, g_job.norm_scale,
                           packed);
    bonsai_accelerator_mark_packed_input(out, n);
  } else if (packed != NULL) {
    run_job(JOB_NORM_APPLY, n, 0u);
    g_job.quantize_input = out;
    g_job.quantize_output = packed;
    run_job(JOB_QUANTIZE_Q8, n / 128u, 0u);
    if (g_norm_shadow)
      norm_shadow_validate(out, x, g_job.norm_weight, g_job.norm_scale,
                           packed);
    bonsai_accelerator_mark_packed_input(out, n);
  } else {
    run_job(JOB_NORM_APPLY, n, 0u);
  }
}

static float matvec_accel_residual_sum(float *residual,
                                       const struct tensor *weight,
                                       const float *input,
                                       const struct tensor *next_norm) {
  if (!g_residual_cpu_visibility) {
    float sum;
    if (bonsai_accelerate_f32_matvec_residual(
            weight->data, weight->ne0, weight->ne1, input,
            (const float *)next_norm->data, &sum)) {
      g_residual_resident = 1;
      return sum;
    }
    if (g_residual_resident)
      fail("accelerator lost resident residual state");
  }
  if (!bonsai_accelerate_f32_matvec_submit(weight->data, weight->ne0,
                                           weight->ne1, input))
    fail("accelerator declined residual matvec");
  if (g_trace) {
    static int checked_attention_output;
    static int checked_ffn_down;
    int *checked = weight->ne0 == BONSAI_DIM ?
        &checked_attention_output : &checked_ffn_down;
    if (!*checked && (weight->ne0 == BONSAI_DIM ||
                      weight->ne0 == BONSAI_FF_DIM)) {
      static const uint32_t rows[] = {
          0u, 1u, 511u, 512u, 639u, 640u, 943u, 944u,
          1023u, 1024u, 1028u, 1124u, 1168u, 1432u, 1535u, 2047u,
      };
      const volatile int64_t *raw = bonsai_accelerate_stored_output();
      const volatile uint64_t *packed = bonsai_accelerator_packed_input();
      bonsai_accelerator_output_fence();
      for (size_t i = 0; i < sizeof(rows) / sizeof(rows[0]); ++i) {
        uint32_t row = rows[i];
        float observed = accelerator_float_at(raw, row);
        float reference = cpu_reference_q8_row(weight, packed, row);
        fprintf(stderr,
                "[trace] residual shape=%ux%u row=%u accel=%g q8ref=%g\n",
                weight->ne0, weight->ne1, row, (double)observed,
                (double)reference);
      }
      *checked = 1;
    }
  }
  if (getenv("BONSAI_REPLAY_RESIDUAL") != NULL) {
    static uint32_t reference[BONSAI_DIM];
    const volatile int64_t *raw = bonsai_accelerate_stored_output();
    bonsai_accelerator_output_fence();
    for (uint32_t i = 0; i < weight->ne1; ++i)
      reference[i] = accelerator_bits_at(raw, i);
    bonsai_accelerate_release();
    if (!bonsai_accelerate_f32_matvec_submit(weight->data, weight->ne0,
                                             weight->ne1, input))
      fail("accelerator declined residual replay");
    raw = bonsai_accelerate_stored_output();
    bonsai_accelerator_output_fence();
    uint32_t mismatches = 0u;
    uint32_t first = weight->ne1;
    uint32_t first_observed = 0;
    for (uint32_t i = 0; i < weight->ne1; ++i) {
      uint32_t observed = accelerator_bits_at(raw, i);
      if (observed != reference[i]) {
        if (first == weight->ne1) {
          first = i;
          first_observed = observed;
        }
        ++mismatches;
      }
    }
    (void)usleep(1000u);
    bonsai_accelerator_output_fence();
    uint32_t late_mismatches = 0u;
    for (uint32_t i = 0; i < weight->ne1; ++i)
      late_mismatches += accelerator_bits_at(raw, i) != reference[i];
    fprintf(stderr,
            "Bonsai residual replay: mismatches=%u late=%u first=%u "
            "%08x/%08x\n",
            mismatches, late_mismatches, first,
            first < weight->ne1 ? reference[first] : 0u, first_observed);
  }
  g_job.out = residual;
  /* CVA6's custom CPU/DDR bridge needs the main hart to retire the DMA
     completion before the per-worker fences invalidate their cached output
     aliases. Omitting this apparently duplicate fence causes sequence-state
     divergence after the first token. */
  bonsai_accelerator_output_fence();
  run_job(JOB_READ_ROWS_ADD_SUM, weight->ne1, 0u);
  float sum = 0.0f;
  for (uint32_t worker = 0; worker < g_job.workers; ++worker)
    sum += g_job.slots[worker].score;
  bonsai_accelerate_release();
  return sum;
}

static void head_norm_rope(int32_t *output, const volatile int64_t *raw,
                           uint32_t raw_first, const int32_t *weight,
                           uint32_t position) {
  const int32_t *cs = bonsai_rope_cos_fixed[position];
  const int32_t *sn = bonsai_rope_sin_fixed[position];
  const uint32_t half = BONSAI_HEAD_DIM / 2u;
  int64_t fixed[BONSAI_HEAD_DIM];
  uint64_t sum = 0u;
  for (uint32_t i = 0u; i < BONSAI_HEAD_DIM; ++i) {
    int64_t value = fp32_bits_to_q16(
        accelerator_bits_at(raw, raw_first + i));
    if (value > INT64_C(16777216)) value = INT64_C(16777216);
    if (value < -INT64_C(16777216)) value = -INT64_C(16777216);
    fixed[i] = value;
    sum += (uint64_t)(value * value);
  }
  uint64_t mean = sum / BONSAI_HEAD_DIM + UINT64_C(4295);
  uint64_t root = integer_sqrt_u64(mean);
  int64_t scale = root == 0u ? INT64_C(65536000) :
                  (int64_t)((UINT64_C(1) << 32) / root);
  for (uint32_t i = 0; i < half; ++i) {
    int64_t a = multiply_q16(
        multiply_q16(fixed[i], scale), weight[i]);
    int64_t b = multiply_q16(
        multiply_q16(fixed[i + half], scale), weight[i + half]);
    int64_t cosine = cs[i];
    int64_t sine = sn[i];
    int64_t first = multiply_q16(a, cosine) - multiply_q16(b, sine);
    int64_t second = multiply_q16(a, sine) + multiply_q16(b, cosine);
    output[i] = (int32_t)first;
    output[i + half] = (int32_t)second;
  }
}

static float reliable_attention_score(const float *query, const float *key) {
  uint32_t score = 0u;
  for (uint32_t j = 0u; j < BONSAI_HEAD_DIM; ++j) {
    uint32_t product = multiply_f32_bits(f32_to_bits(query[j]),
                                         f32_to_bits(key[j]));
    score = add_f32_bits(score, product);
  }
  return f32_from_bits(multiply_f32_bits(score, UINT32_C(0x3db504f3)));
}

static float reliable_attention_exp(float x) {
  uint32_t x_bits = f32_to_bits(x);
  int64_t x_q16 = fp32_bits_to_q16(x_bits);
  if (x_q16 < -INT64_C(1310720)) return 0.0f;
  if (x_q16 > INT64_C(1310720)) {
    x_q16 = INT64_C(1310720);
    x_bits = UINT32_C(0x41a00000);
  }
  int64_t scaled = x_q16 * INT64_C(94548);
  int32_t n = scaled >= 0 ?
      (int32_t)((scaled + INT64_C(2147483648)) >> 32) :
      -(int32_t)((-scaled + INT64_C(2147483648)) >> 32);
  uint32_t n_bits = signed_integer_to_f32_bits(n);
  uint32_t r = add_f32_bits(
      x_bits, multiply_f32_bits(n_bits, UINT32_C(0xbf317218)));
  uint32_t r2 = multiply_f32_bits(r, r);
  uint32_t inner = add_f32_bits(
      UINT32_C(0x3d2aaaab),
      multiply_f32_bits(r, UINT32_C(0x3c088889)));
  inner = add_f32_bits(UINT32_C(0x3e2aaaab),
                       multiply_f32_bits(r, inner));
  inner = add_f32_bits(UINT32_C(0x3f000000),
                       multiply_f32_bits(r, inner));
  uint32_t polynomial = add_f32_bits(
      add_f32_bits(UINT32_C(0x3f800000), r),
      multiply_f32_bits(r2, inner));
  int exponent = n + 127;
  if (exponent <= 0) return 0.0f;
  if (exponent >= 255) exponent = 254;
  return f32_from_bits(multiply_f32_bits(
      polynomial, (uint32_t)exponent << 23));
}

static void attention_prep_worker(uint32_t worker, uint32_t workers,
                                  uint32_t position,
                                  const struct layer *layer,
                                  const volatile int64_t *raw) {
  uint32_t layer_index = (uint32_t)(layer - g_model.layers);
  int32_t *kc = g_state.kv_k[layer_index][position];
  int32_t *vc = g_state.kv_v[layer_index][position];
  uint32_t q_begin = (BONSAI_HEADS * worker) / workers;
  uint32_t q_end = (BONSAI_HEADS * (worker + 1u)) / workers;
  for (uint32_t h = q_begin; h < q_end; ++h) {
    head_norm_rope(g_state.q_fixed + h * BONSAI_HEAD_DIM, raw,
                   h * BONSAI_HEAD_DIM, g_q_norm_fixed[layer_index],
                   position);
  }
  uint32_t kv_begin = (BONSAI_KV_HEADS * worker) / workers;
  uint32_t kv_end = (BONSAI_KV_HEADS * (worker + 1u)) / workers;
  for (uint32_t h = kv_begin; h < kv_end; ++h) {
    int32_t k_fixed[BONSAI_HEAD_DIM];
    uint32_t head_first = h * BONSAI_HEAD_DIM;
    head_norm_rope(k_fixed, raw, BONSAI_DIM + head_first,
                   g_k_norm_fixed[layer_index], position);
    for (uint32_t i = 0; i < BONSAI_HEAD_DIM; ++i) {
      kc[h * BONSAI_HEAD_DIM + i] =
          (int32_t)fp32_bits_to_q16(truncate_f32_bits_through_half(
              signed_q16_to_f32_bits(k_fixed[i])));
      vc[h * BONSAI_HEAD_DIM + i] =
          (int32_t)fp32_bits_to_q16(truncate_f32_bits_through_half(
              accelerator_bits_at(raw, BONSAI_DIM + BONSAI_KV_DIM +
                                       head_first + i)));
    }
  }
  if (getenv("BONSAI_LAYER_TRACE") != NULL && layer_index == 0u &&
      worker == 0u && position <= 2u) {
    uint32_t qb[3], kb[3], vb[3];
    memcpy(qb, g_state.q_fixed, sizeof(qb));
    memcpy(kb, kc, sizeof(kb));
    memcpy(vb, vc, sizeof(vb));
    fprintf(stderr,
            "Bonsai prep p=%u q=%08x,%08x,%08x k=%08x,%08x,%08x "
            "v=%08x,%08x,%08x\n",
            position, qb[0], qb[1], qb[2], kb[0], kb[1], kb[2],
            vb[0], vb[1], vb[2]);
  }
}

static void attention_q_worker(uint32_t worker, uint32_t workers,
                               uint32_t position, const struct layer *layer,
                               const volatile int64_t *raw) {
  uint32_t layer_index = (uint32_t)(layer - g_model.layers);
  uint32_t q_begin = (BONSAI_HEADS * worker) / workers;
  uint32_t q_end = (BONSAI_HEADS * (worker + 1u)) / workers;
  for (uint32_t h = q_begin; h < q_end; ++h) {
    head_norm_rope(g_state.q_fixed + h * BONSAI_HEAD_DIM, raw,
                   h * BONSAI_HEAD_DIM, g_q_norm_fixed[layer_index],
                   position);
  }
}

static void attention_hardware_kv_worker(uint32_t worker, uint32_t workers,
                                         uint32_t position,
                                         const struct layer *layer) {
  uint32_t layer_index = (uint32_t)(layer - g_model.layers);
  const volatile uint16_t *hardware_kv =
      bonsai_accelerator_attention_kv(layer_index, position);
  if (hardware_kv == NULL) fail("hardware attention KV is unavailable");
  uint32_t kv_begin = (BONSAI_KV_DIM * worker) / workers;
  uint32_t kv_end = (BONSAI_KV_DIM * (worker + 1u)) / workers;
  int32_t *key = g_state.kv_k[layer_index][position];
  int32_t *value = g_state.kv_v[layer_index][position];
  for (uint32_t i = kv_begin; i < kv_end; ++i) {
    key[i] = (int32_t)fp32_bits_to_q16(
        half_to_float_bits(hardware_kv[i]));
    value[i] = (int32_t)fp32_bits_to_q16(
        half_to_float_bits(hardware_kv[BONSAI_KV_DIM + i]));
  }
}

static const struct layer *g_attn_layer;

static inline uint64_t divide_u64_reciprocal(uint64_t numerator,
                                             uint64_t denominator,
                                             uint64_t reciprocal) {
  uint64_t quotient =
      (uint64_t)(((__uint128_t)numerator * reciprocal) >> 64);
  uint64_t product = quotient * denominator;
  /* reciprocal is ceil(2^64 / denominator), so the estimate is exact or
     one too large for a 64-bit numerator. Correct that possible excess to
     retain the original integer division result bit-for-bit. */
  if (product > numerator) --quotient;
  return quotient;
}

static void attention_head_range(uint32_t begin, uint32_t end,
                                 uint32_t position) {
  const struct layer *layer = g_attn_layer;
  uint32_t layer_index = (uint32_t)(layer - g_model.layers);
  if (position == 0u) {
    for (uint32_t h = begin; h < end; ++h) {
      uint32_t kvh = h / (BONSAI_HEADS / BONSAI_KV_HEADS);
      const int32_t *value =
          g_state.kv_v[layer_index][0] + kvh * BONSAI_HEAD_DIM;
      for (uint32_t j = 0u; j < BONSAI_HEAD_DIM; ++j)
        g_state.att[h * BONSAI_HEAD_DIM + j] =
            f32_from_bits(signed_q16_to_f32_bits(value[j]));
    }
    return;
  }
  int32_t scores[BONSAI_MAX_CONTEXT];
  for (uint32_t h = begin; h < end; ++h) {
    uint32_t kvh = h / (BONSAI_HEADS / BONSAI_KV_HEADS);
    int32_t max_score = INT32_MIN;
    for (uint32_t t = 0; t <= position; ++t) {
      const int32_t *key =
          g_state.kv_k[layer_index][t] + kvh * BONSAI_HEAD_DIM;
      int64_t dot = 0;
      for (uint32_t j = 0u; j < BONSAI_HEAD_DIM; ++j) {
        int64_t query = g_state.q_fixed[h * BONSAI_HEAD_DIM + j];
        dot += query * key[j];
      }
      int64_t scaled = dot * INT64_C(5793); /* 1/sqrt(128) in Q16 */
      int32_t score = scaled >= 0 ?
          (int32_t)((scaled + (INT64_C(1) << 31)) >> 32) :
          -(int32_t)((-scaled + (INT64_C(1) << 31)) >> 32);
      scores[t] = score;
      if (score > max_score) max_score = score;
    }
    uint64_t denominator = 0u;
    for (uint32_t t = 0; t <= position; ++t) {
      scores[t] = (int32_t)exp_q16(scores[t] - max_score);
      denominator += (uint32_t)scores[t];
    }
    int64_t sums[BONSAI_HEAD_DIM] = {0};
    for (uint32_t t = 0; t <= position; ++t) {
      const int32_t *value =
          g_state.kv_v[layer_index][t] + kvh * BONSAI_HEAD_DIM;
      int64_t score = scores[t];
      for (uint32_t j = 0; j < BONSAI_HEAD_DIM; ++j)
        sums[j] += score * value[j];
    }
    uint64_t reciprocal = UINT64_MAX / denominator + 1u;
    uint64_t rounding = denominator / 2u;
    for (uint32_t j = 0; j < BONSAI_HEAD_DIM; ++j) {
      int64_t sum = sums[j];
      uint64_t magnitude = sum >= 0 ? (uint64_t)sum : (uint64_t)-sum;
      int64_t output = (int64_t)divide_u64_reciprocal(
          magnitude + rounding, denominator, reciprocal);
      if (sum < 0) output = -output;
      g_state.att[h * BONSAI_HEAD_DIM + j] =
          f32_from_bits(signed_q16_to_f32_bits(output));
    }
    if (getenv("BONSAI_LAYER_TRACE") != NULL && layer_index == 0u && h == 0u &&
        position <= 2u) {
      uint32_t ab[3];
      memcpy(ab, g_state.att, sizeof(ab));
      fprintf(stderr,
              "Bonsai head p=%u scores=%g,%g,%g denom=%g "
              "att=%08x,%08x,%08x\n",
              position, (double)scores[0] / 65536.0,
              position >= 1u ? (double)scores[1] / 65536.0 : 0.0,
              position >= 2u ? (double)scores[2] / 65536.0 : 0.0,
              (double)denominator / 65536.0,
              ab[0], ab[1], ab[2]);
    }
  }
}

static float attention(struct run_state *state, const struct layer *layer,
                       uint32_t position) {
  if (!bonsai_accelerate_f32_matvec_store(
          layer->q.data, layer->q.ne0, layer->q.ne1, state->norm, 0u) ||
      !bonsai_accelerate_f32_matvec_store(
          layer->k.data, layer->k.ne0, layer->k.ne1, state->norm,
          BONSAI_DIM) ||
      !bonsai_accelerate_f32_matvec_store(
          layer->v.data, layer->v.ne0, layer->v.ne1, state->norm,
          BONSAI_DIM + BONSAI_KV_DIM))
    fail("accelerator declined batched QKV matvec");
  g_accelerator_compact_output = bonsai_accelerator_has_compact_output();
  static int q_reference_checked;
  if (!q_reference_checked && g_trace) {
    const volatile int64_t *raw = bonsai_accelerate_stored_output();
    bonsai_accelerator_output_fence();
    const volatile uint64_t *packed = bonsai_accelerator_packed_input();
    float q_reference = cpu_reference_row(&layer->q, state->norm, 0u);
    float q_quantized = cpu_reference_q8_row(&layer->q, packed, 0u);
    float k_quantized = cpu_reference_q8_row(&layer->k, packed, 0u);
    float v_quantized = cpu_reference_q8_row(&layer->v, packed, 0u);
    fprintf(stderr,
            "[trace] layer0 QKV row0 accel=%g,%g,%g qref=%g "
            "q8ref=%g,%g,%g\n",
            (double)accelerator_float_at(raw, 0u),
            (double)accelerator_float_at(raw, BONSAI_DIM),
            (double)accelerator_float_at(raw, BONSAI_DIM + BONSAI_KV_DIM),
            (double)q_reference, (double)q_quantized, (double)k_quantized,
            (double)v_quantized);
    q_reference_checked = 1;
  }
  static int repeat_checked;
  if (!repeat_checked && getenv("BONSAI_REPEAT_CHECK") != NULL) {
    static uint32_t reference[BONSAI_DIM];
    const volatile int64_t *raw = bonsai_accelerate_stored_output();
    bonsai_accelerator_output_fence();
    for (uint32_t i = 0; i < BONSAI_DIM; ++i)
      reference[i] = accelerator_bits_at(raw, i);
    fprintf(stderr, "Bonsai repeated Q reference: %08x %08x %08x\n",
            reference[0], reference[1], reference[2]);
    for (uint32_t pass = 0; pass < 4u; ++pass) {
      if (!bonsai_accelerate_f32_matvec_store(
              layer->q.data, layer->q.ne0, layer->q.ne1, state->norm, 0u))
        fail("accelerator declined repeated Q matvec");
      raw = bonsai_accelerate_stored_output();
      bonsai_accelerator_output_fence();
      uint32_t mismatches = 0u;
      uint32_t first = BONSAI_DIM;
      for (uint32_t i = 0; i < BONSAI_DIM; ++i) {
        if (accelerator_bits_at(raw, i) != reference[i]) {
          if (first == BONSAI_DIM) first = i;
          ++mismatches;
        }
      }
      fprintf(stderr, "Bonsai repeated Q matvec pass %u: mismatches=%u first=%u\n",
              pass + 1u, mismatches, first);
    }
    repeat_checked = 1;
  }
  g_job.raw = bonsai_accelerate_stored_output();
  g_job.layer = layer;
  g_attn_layer = layer;
  volatile uint64_t *packed = bonsai_accelerator_packed_input();
  uint64_t attention_start = g_profile ? now_ns() : 0u;
  uint32_t layer_index = (uint32_t)(layer - g_model.layers);
  int hardware_attention_layer =
      (g_hw_attention_skip_mask & (UINT32_C(1) << layer_index)) == 0u;
  if (packed != NULL && hardware_attention_layer &&
      bonsai_accelerator_has_attention()) {
    bonsai_accelerator_output_fence();
    if (!bonsai_accelerator_attention(
            layer_index, position, g_q_norm_fixed[layer_index],
            g_k_norm_fixed[layer_index], bonsai_rope_cos_fixed[position],
            bonsai_rope_sin_fixed[position], state->att))
      fail("accelerator declined hardware attention");
    if (getenv("BONSAI_REPEAT_HW_ATTENTION") != NULL) {
      static uint8_t reference[sizeof(g_attn_shadow_packed)];
      const volatile uint8_t *actual = (const volatile uint8_t *)packed;
      bonsai_accelerator_output_fence();
      for (size_t i = 0u; i < sizeof(reference); ++i)
        reference[i] = actual[i];
      if (!bonsai_accelerator_attention(
              layer_index, position, g_q_norm_fixed[layer_index],
              g_k_norm_fixed[layer_index], bonsai_rope_cos_fixed[position],
              bonsai_rope_sin_fixed[position], state->att))
        fail("accelerator declined repeated hardware attention");
      bonsai_accelerator_output_fence();
      uint32_t mismatches = 0u;
      uint32_t first = (uint32_t)sizeof(reference);
      for (uint32_t i = 0u; i < (uint32_t)sizeof(reference); ++i) {
        if (reference[i] != actual[i]) {
          if (first == sizeof(reference)) first = i;
          ++mismatches;
        }
      }
      fprintf(stderr,
              "Bonsai repeated hardware attention layer=%u position=%u "
              "mismatches=%u first=%u\n",
              layer_index, position, mismatches, first);
    }
    if (g_attn_shadow) {
      /* Recompute the same attention result on the CPU without disturbing
         the packed hardware result.  This validates the actual Q8 stream
         consumed by the following matvec, rather than comparing a later
         residual where small errors are harder to localize. */
      run_job(JOB_ATTN_READ_PREP, BONSAI_HEADS, position);
      const volatile uint16_t *hardware_kv =
          bonsai_accelerator_attention_kv(layer_index, position);
      if (hardware_kv != NULL) {
        uint64_t k_mismatches = 0u;
        uint64_t v_mismatches = 0u;
        uint32_t first_k = BONSAI_KV_DIM;
        uint32_t first_v = BONSAI_KV_DIM;
        int32_t first_k_actual = 0;
        int32_t first_v_actual = 0;
        for (uint32_t i = 0u; i < BONSAI_KV_DIM; ++i) {
          int32_t k_actual = (int32_t)fp32_bits_to_q16(
              half_to_float_bits(hardware_kv[i]));
          int32_t v_actual = (int32_t)fp32_bits_to_q16(
              half_to_float_bits(hardware_kv[BONSAI_KV_DIM + i]));
          if (k_actual != g_state.kv_k[layer_index][position][i]) {
            if (first_k == BONSAI_KV_DIM) {
              first_k = i;
              first_k_actual = k_actual;
            }
            ++k_mismatches;
            if (k_mismatches <= 8u) {
              uint32_t head_index = i / BONSAI_HEAD_DIM;
              uint32_t lane = i % BONSAI_HEAD_DIM;
              int32_t transformed[BONSAI_HEAD_DIM];
              head_norm_rope(
                  transformed, g_job.raw,
                  BONSAI_DIM + head_index * BONSAI_HEAD_DIM,
                  g_k_norm_fixed[layer_index], position);
              uint32_t pair_lane = lane < BONSAI_HEAD_DIM / 2u ?
                  lane + BONSAI_HEAD_DIM / 2u :
                  lane - BONSAI_HEAD_DIM / 2u;
              fprintf(stderr,
                      "Bonsai attention K byte layer=%u position=%u "
                      "head=%u lane=%u pair=%u raw=%08x pair_raw=%08x "
                      "transformed=%d expected=%d actual=%d half=%04x\n",
                      layer_index, position, head_index, lane, pair_lane,
                      accelerator_bits_at(
                          g_job.raw, BONSAI_DIM +
                              head_index * BONSAI_HEAD_DIM + lane),
                      accelerator_bits_at(
                          g_job.raw, BONSAI_DIM +
                              head_index * BONSAI_HEAD_DIM + pair_lane),
                      transformed[lane],
                      g_state.kv_k[layer_index][position][i], k_actual,
                      hardware_kv[i]);
            }
          }
          if (v_actual != g_state.kv_v[layer_index][position][i]) {
            if (first_v == BONSAI_KV_DIM) {
              first_v = i;
              first_v_actual = v_actual;
            }
            ++v_mismatches;
          }
        }
        fprintf(stderr,
                "Bonsai attention KV shadow layer=%u position=%u "
                "k=%llu first=%u expected=%d actual=%d "
                "v=%llu first=%u expected=%d actual=%d\n",
                layer_index, position, (unsigned long long)k_mismatches,
                first_k, first_k == BONSAI_KV_DIM ? 0 :
                    g_state.kv_k[layer_index][position][first_k],
                first_k_actual, (unsigned long long)v_mismatches, first_v,
                first_v == BONSAI_KV_DIM ? 0 :
                    g_state.kv_v[layer_index][position][first_v],
                first_v_actual);
      }
      run_job(JOB_ATTN_HEADS, BONSAI_HEADS, position);
      quantize_q8_group_range(state->att, g_attn_shadow_packed, 0u,
                              BONSAI_DIM / 128u);
      const volatile uint8_t *actual = (const volatile uint8_t *)packed;
      const uint8_t *expected = (const uint8_t *)g_attn_shadow_packed;
      size_t bytes = sizeof(g_attn_shadow_packed);
      uint64_t mismatches = 0u;
      size_t first = bytes;
      for (size_t i = 0u; i < bytes; ++i) {
        if (actual[i] != expected[i]) {
          if (first == bytes) first = i;
          ++mismatches;
        }
      }
      g_attn_shadow_bytes += bytes;
      g_attn_shadow_mismatches += mismatches;
      fprintf(stderr,
              "Bonsai attention shadow layer=%u position=%u "
              "mismatches=%llu/%zu first=%zu expected=%02x actual=%02x\n",
              layer_index, position, (unsigned long long)mismatches, bytes,
              first, first == bytes ? 0u : expected[first],
              first == bytes ? 0u : actual[first]);
      uint32_t reported = 0u;
      for (size_t i = 0u; i < bytes && reported < 16u; ++i) {
        if (actual[i] != expected[i]) {
          fprintf(stderr,
                  "Bonsai attention byte layer=%u position=%u offset=%zu "
                  "block=%zu item=%zu expected=%02x actual=%02x\n",
                  layer_index, position, i, i / 34u, i % 34u,
                  expected[i], actual[i]);
          ++reported;
        }
      }
    }
    bonsai_accelerator_mark_packed_input(state->att, BONSAI_DIM);
  } else if (packed != NULL) {
    if (g_hardware_kv_prep && bonsai_accelerator_has_attention()) {
      bonsai_accelerator_output_fence();
      run_job(JOB_ATTN_READ_Q, BONSAI_HEADS, position);
      if (!bonsai_accelerator_attention(
              layer_index, position, g_q_norm_fixed[layer_index],
              g_k_norm_fixed[layer_index], bonsai_rope_cos_fixed[position],
              bonsai_rope_sin_fixed[position], state->att))
        fail("accelerator declined KV preparation");
      run_job(JOB_ATTN_READ_HARDWARE_KV, BONSAI_HEADS, position);
    } else {
      run_job(JOB_ATTN_READ_PREP, BONSAI_HEADS, position);
    }
    run_job(JOB_ATTN_HEADS, BONSAI_HEADS, position);
    g_job.quantize_input = state->att;
    g_job.quantize_output = packed;
    run_job(JOB_QUANTIZE_Q8, BONSAI_DIM / 128u, 0u);
    bonsai_accelerator_mark_packed_input(state->att, BONSAI_DIM);
  } else {
    run_job(JOB_ATTN_READ_PREP, BONSAI_HEADS, position);
    run_job(JOB_ATTN_HEADS, BONSAI_HEADS, position);
  }
  if (g_profile) g_t_head += now_ns() - attention_start;
  return matvec_accel_residual_sum(state->x, &layer->attn_out, state->att,
                                   &layer->ffn_norm);
}

static float feed_forward(struct run_state *state, const struct layer *layer,
                          const struct tensor *next_norm) {
  if (!bonsai_accelerate_f32_matvec_store(
          layer->ffn_up.data, layer->ffn_up.ne0, layer->ffn_up.ne1,
          state->norm, 0u))
    fail("accelerator declined FFN-up matvec");
  if (!bonsai_accelerate_f32_matvec_store(
          layer->ffn_gate.data, layer->ffn_gate.ne0, layer->ffn_gate.ne1,
          state->norm, BONSAI_FF_DIM))
    fail("accelerator declined FFN-gate matvec");
  g_accelerator_compact_output = bonsai_accelerator_has_compact_output();
  g_job.out = state->ffn_up;
  g_job.raw = bonsai_accelerate_stored_output();
  volatile uint64_t *packed = bonsai_accelerator_packed_input();
  if (g_trace) {
    bonsai_accelerator_output_fence();
    static int checked_ffn_inputs;
    if (!checked_ffn_inputs) {
      static const uint32_t rows[] = {
          0u, 1u, 2047u, 2048u, 4095u, 4096u, 6143u,
      };
      for (size_t i = 0; i < sizeof(rows) / sizeof(rows[0]); ++i) {
        uint32_t row = rows[i];
        float up_observed = accelerator_float_at(g_job.raw, row);
        float up_reference = cpu_reference_q8_row(
            &layer->ffn_up, packed, row);
        float up_float_reference = cpu_reference_row(
            &layer->ffn_up, state->norm, row);
        float gate_observed = accelerator_float_at(
            g_job.raw, BONSAI_FF_DIM + row);
        float gate_reference = cpu_reference_q8_row(
            &layer->ffn_gate, packed, row);
        float gate_float_reference = cpu_reference_row(
            &layer->ffn_gate, state->norm, row);
        fprintf(stderr,
                "[trace] ffn row=%u up=%g/%g/%g gate=%g/%g/%g\n", row,
                (double)up_observed, (double)up_reference,
                (double)up_float_reference, (double)gate_observed,
                (double)gate_reference, (double)gate_float_reference);
      }
      checked_ffn_inputs = 1;
    }
  }
  g_job.quantize_output = packed;
  if (packed != NULL) bonsai_accelerator_output_fence();
  if (packed != NULL && bonsai_accelerator_pack_ffn(BONSAI_FF_DIM)) {
    if (g_ffn_shadow)
      ffn_shadow_validate_range(0u, BONSAI_FF_DIM / 128u);
    bonsai_accelerator_mark_packed_input(state->ffn_up, BONSAI_FF_DIM);
  } else if (packed != NULL) {
    run_job(JOB_FFN_READ_ACTIVATE_QUANTIZE, BONSAI_FF_DIM / 128u, 0u);
    if (g_ffn_shadow)
      ffn_shadow_validate_range(0u, BONSAI_FF_DIM / 128u);
    bonsai_accelerator_mark_packed_input(state->ffn_up, BONSAI_FF_DIM);
  } else {
    run_job(JOB_FFN_READ_ACTIVATE, BONSAI_FF_DIM, 0u);
  }
  return matvec_accel_residual_sum(state->x, &layer->ffn_down, state->ffn_up,
                                   next_norm);
}

static uint32_t forward(struct model *model, struct run_state *state,
                        uint32_t token, uint32_t position) {
  if (position >= BONSAI_MAX_CONTEXT) fail("context full");
  embedding_row(state->x, &model->embedding, token);
  g_residual_resident = 0;
  if (position == 0 && g_trace) {
    const float *w = (const float *)model->layers[0].attn_norm.data;
    volatile float fa = 3.0f, fb = 2.0f;
    volatile float fsum = fa + fb;
    volatile float fmul = fa * fb;
    fprintf(stderr, "[trace0] add=%g mul=%g\n", (double)fsum,
            (double)fmul);
    fprintf(stderr,
            "[trace0] emb x[0..2]=%g %g %g attn_norm w[0..2]=%g %g %g "
            "(ne0=%u type=%u data=%p base=%p)\n",
            (double)state->x[0], (double)state->x[1], (double)state->x[2],
            (double)w[0], (double)w[1], (double)w[2],
            model->layers[0].attn_norm.ne0, model->layers[0].attn_norm.type,
            (const void *)model->layers[0].attn_norm.data,
            (const void *)model->base);
  }
  float residual_sum = 0.0f;
  for (uint32_t il = 0; il < g_layer_count; ++il) {
    struct layer *layer = &model->layers[il];
    uint64_t t0 = g_profile ? now_ns() : 0u;
    if (il == 0u)
      rms_norm(state->norm, state->x, &layer->attn_norm, BONSAI_DIM);
    else
      rms_norm_from_sum(state->norm, state->x, &layer->attn_norm,
                        BONSAI_DIM, residual_sum);
    if (il == 0 && position == 0 && g_trace) {
      float sum = 0.0f;
      int bad = 0;
      for (uint32_t i = 0; i < BONSAI_DIM; ++i) {
        sum += state->x[i] * state->x[i];
        float a = state->x[i] < 0 ? -state->x[i] : state->x[i];
        if (state->x[i] != state->x[i] || a > 1e20f) bad++;
      }
      fprintf(stderr, "[trace0] layer0 norm[0..2]=%g %g %g xsumsq=%g badx=%d\n",
              (double)state->norm[0], (double)state->norm[1],
              (double)state->norm[2], (double)sum, bad);
      const volatile uint8_t *norm_q8 =
          (const volatile uint8_t *)bonsai_accelerator_packed_input();
#ifdef BONSAI_HOST_STUB
      uint8_t host_norm_q8[BONSAI_DIM / 32u * 34u];
      quantize_row_q8_0(state->norm, host_norm_q8, BONSAI_DIM);
      norm_q8 = host_norm_q8;
#endif
      fprintf(stderr, "[trace0] layer0 norm q8:");
      for (uint32_t i = 0u; norm_q8 != NULL && i < 34u; ++i)
        fprintf(stderr, " %02x", norm_q8[i]);
      fprintf(stderr, "\n");
      fflush(stderr);
    }
    uint64_t t1 = g_profile ? now_ns() : 0u;
    float attention_sum = attention(state, layer, position);
    uint64_t t2 = g_profile ? now_ns() : 0u;
    if (getenv("BONSAI_LAYER_TRACE") != NULL) {
      uint32_t xbits[3];
      memcpy(xbits, state->x, sizeof(xbits));
      fprintf(stderr, "Bonsai attention %u x=%08x,%08x,%08x sum=%g\n", il,
              xbits[0], xbits[1], xbits[2], (double)attention_sum);
    }
    rms_norm_from_sum(state->norm, state->x, &layer->ffn_norm,
                      BONSAI_DIM, attention_sum);
    if (il == 0u && position == 0u && g_trace) {
      const volatile uint8_t *packed =
          (const volatile uint8_t *)bonsai_accelerator_packed_input();
      fprintf(stderr,
              "[trace0] ffn norm sum=%g scale=%g norm[0..2]=%g,%g,%g "
              "q8scale=%02x%02x q8[0..2]=%d,%d,%d\n",
              (double)attention_sum,
              (double)rms_scale_f(attention_sum, 11u),
              (double)state->norm[0], (double)state->norm[1],
              (double)state->norm[2], packed != NULL ? packed[1] : 0u,
              packed != NULL ? packed[0] : 0u,
              packed != NULL ? (int8_t)packed[2] : 0,
              packed != NULL ? (int8_t)packed[3] : 0,
              packed != NULL ? (int8_t)packed[4] : 0);
    }
    uint64_t t3 = g_profile ? now_ns() : 0u;
    const struct tensor *next_norm = il + 1u < g_layer_count ?
        &model->layers[il + 1u].attn_norm : &model->output_norm;
    residual_sum = feed_forward(state, layer, next_norm);
    uint64_t t4 = g_profile ? now_ns() : 0u;
    if (getenv("BONSAI_LAYER_TRACE") != NULL) {
      uint32_t xbits[3];
      memcpy(&xbits[0], &state->x[0], sizeof(xbits[0]));
      memcpy(&xbits[1], &state->x[1], sizeof(xbits[1]));
      memcpy(&xbits[2], &state->x[2], sizeof(xbits[2]));
      fprintf(stderr, "Bonsai layer %u x=%08x,%08x,%08x sum=%g\n", il,
              xbits[0], xbits[1], xbits[2], (double)residual_sum);
    }
    if (g_trace) {
      static int nan_reported;
      if (!nan_reported) {
        for (uint32_t i = 0; i < BONSAI_DIM; ++i) {
          float ax = state->x[i] < 0 ? -state->x[i] : state->x[i];
          if (state->x[i] != state->x[i] || ax > 1e30f) {
            nan_reported = 1;
            fprintf(stderr,
                    "[trace] first NaN in x at layer %u index %u "
                    "(v=%g att0=%g op0=%g up0=%g gate0=%g)\n",
                    il, i, (double)state->x[i], (double)state->att[0],
                    (double)state->op[0], (double)state->ffn_up[0],
                    (double)state->ffn_gate[0]);
            fflush(stderr);
            break;
          }
        }
      }
    }
    if (g_profile) {
      g_t_norm += (t1 - t0) + (t3 - t2);
      g_t_attn += t2 - t1;
      g_t_ffn += t4 - t3;
    }
  }
  rms_norm_from_sum(state->norm, state->x, &model->output_norm,
                    BONSAI_DIM, residual_sum);
  uint32_t best = 0u;
  int hardware_argmax = 0;
  if (g_trace) {
    matvec_accel(g_logits, &model->embedding, state->norm);
  } else if (bonsai_accelerate_f32_matvec_argmax(
                 model->embedding.data, model->embedding.ne0,
                 model->embedding.ne1, state->norm, &best)) {
    hardware_argmax = 1;
  } else {
    if (!bonsai_accelerate_f32_matvec_submit(
            model->embedding.data, model->embedding.ne0,
            model->embedding.ne1, state->norm))
      fail("accelerator declined output matvec");
    run_job(JOB_ACCEL_ARGMAX, BONSAI_VOCAB, 0);
    bonsai_accelerate_release();
  }
  if (g_trace) {
    float lo = 3.4e38f, hi = -3.4e38f;
    int nan_count = 0;
    for (uint32_t i = 0; i < 1000u; ++i) {
      float v = g_logits[i];
      if (v != v) nan_count++;
      if (v < lo) lo = v;
      if (v > hi) hi = v;
    }
    fprintf(stderr,
            "[trace] pos=%u logits[0..999] lo=%g hi=%g nan=%d x0=%g norm0=%g "
            "attn=%.2fs ffn=%.2fs norms=%.2fs heads=%.2fs\n",
            position, (double)lo, (double)hi, nan_count, (double)state->x[0],
            (double)state->norm[0], (double)g_t_attn / 1e9,
            (double)g_t_ffn / 1e9, (double)g_t_norm / 1e9,
            (double)g_t_head / 1e9);
  }
  if (g_trace) run_job(JOB_ARGMAX, BONSAI_VOCAB, 0);
  if (!g_trace && !hardware_argmax) {
    int64_t best_raw = INT64_MIN;
    for (uint32_t worker = 0; worker < g_worker_count; ++worker) {
      if (g_job.slots[worker].raw > best_raw) {
        best_raw = g_job.slots[worker].raw;
        best = g_job.slots[worker].token;
      }
    }
  } else if (g_trace) {
    float best_score = -3.402823466e38f;
    for (uint32_t worker = 0; worker < g_worker_count; ++worker) {
      if (g_job.slots[worker].score > best_score) {
        best_score = g_job.slots[worker].score;
        best = g_job.slots[worker].token;
      }
    }
  }
  ++g_forward_count;
  return best;
}


/* Support symbols for the linked accelerator driver: the reference Q8_0
   quantizer (llama.cpp semantics) and a stub for the per-row fallback the
   lite runtime never takes. */
struct block_q8_0 {
  fp16_t d;
  int8_t qs[32];
} __attribute__((packed));

void quantize_row_q8_0(const float *x, void *vy, int64_t k) {
  struct block_q8_0 *y = (struct block_q8_0 *)vy;
  const int64_t nb = k / 32;
  for (int64_t i = 0; i < nb; i++) {
    uint32_t maximum_bits = 0u;
    for (int j = 0; j < 32; j++) {
      uint32_t bits;
      memcpy(&bits, x + i * 32 + j, sizeof(bits));
      bits &= UINT32_C(0x7fffffff);
      if (maximum_bits < bits) maximum_bits = bits;
    }
    y[i].d = q8_scale_half_from_bits(maximum_bits);
    uint64_t reciprocal = q8_reciprocal_from_bits(maximum_bits);
    for (int j = 0; j < 32; j++) {
      uint32_t bits;
      memcpy(&bits, x + i * 32 + j, sizeof(bits));
      y[i].qs[j] = q8_sample_from_bits(bits, maximum_bits, reciprocal);
    }
  }
}

/* Lite-runtime fast path: form the Q8_0 stream in registers and issue only
   aligned 64-bit stores to the uncached accelerator scratch mapping. This
   avoids writing and rereading a cacheless intermediate buffer. The caller
   guarantees a multiple of 128 elements, so the 34-byte blocks end on an
   eight-byte boundary in groups of four. */
static void quantize_q8_group_range(const float *x,
                                    volatile uint64_t *output,
                                    uint32_t begin, uint32_t end) {
  for (uint32_t group = begin; group < end; ++group) {
    uint8_t packed[136] __attribute__((aligned(8)));
    for (uint32_t local_block = 0; local_block < 4u; ++local_block) {
      uint32_t base = group * 128u + local_block * 32u;
      uint32_t maximum_bits = 0u;
      for (uint32_t item = 0; item < 32u; ++item) {
        uint32_t bits;
        memcpy(&bits, x + base + item, sizeof(bits));
        bits &= UINT32_C(0x7fffffff);
        if (maximum_bits < bits) maximum_bits = bits;
      }
      fp16_t d16 = q8_scale_half_from_bits(maximum_bits);
      uint64_t reciprocal = q8_reciprocal_from_bits(maximum_bits);
      size_t offset = local_block * 34u;
      memcpy(packed + offset, &d16, sizeof(d16));
      for (uint32_t item = 0; item < 32u; ++item) {
        uint32_t bits;
        memcpy(&bits, x + base + item, sizeof(bits));
        packed[offset + 2u + item] =
            (uint8_t)q8_sample_from_bits(bits, maximum_bits, reciprocal);
      }
    }
    for (uint32_t word_index = 0; word_index < 17u; ++word_index) {
      uint64_t word;
      memcpy(&word, packed + word_index * 8u, sizeof(word));
      output[(size_t)group * 17u + word_index] = word;
    }
  }
}

/* The packed activation is the actual consumer of norm and FFN results.
   Form each 32-value Q8 block in a small stack buffer so we do not write a
   full float vector to D-cache only to reread it immediately for packing. */
static void norm_quantize_direct_range(const float *input,
                                       const float *weight, float scale,
                                       volatile uint64_t *output,
                                       uint32_t begin, uint32_t end) {
  for (uint32_t group = begin; group < end; ++group) {
    uint8_t packed[136] __attribute__((aligned(8)));
    for (uint32_t local_block = 0; local_block < 4u; ++local_block) {
      float values[32];
      float amax = 0.0f;
      uint32_t base = group * 128u + local_block * 32u;
      for (uint32_t item = 0; item < 32u; ++item) {
        float value = input[base + item] * scale * weight[base + item];
        values[item] = value;
        float magnitude = value < 0.0f ? -value : value;
        if (amax < magnitude) amax = magnitude;
      }
      float d = amax * (1.0f / 127.0f);
      float id = rcp_f(d);
      fp16_t d16 = float_to_half(d);
      size_t offset = local_block * 34u;
      memcpy(packed + offset, &d16, sizeof(d16));
      for (uint32_t item = 0; item < 32u; ++item)
        packed[offset + 2u + item] =
            (uint8_t)(int8_t)round_f(values[item] * id);
    }
    for (uint32_t word_index = 0; word_index < 17u; ++word_index) {
      uint64_t word;
      memcpy(&word, packed + word_index * 8u, sizeof(word));
      output[(size_t)group * 17u + word_index] = word;
    }
  }
}

static void ffn_quantize_direct_range(const volatile int64_t *raw,
                                      volatile uint64_t *output,
                                      uint32_t begin, uint32_t end) {
  for (uint32_t group = begin; group < end; ++group) {
    uint8_t packed[136] __attribute__((aligned(8)));
    for (uint32_t local_block = 0; local_block < 4u; ++local_block) {
      uint32_t values[32];
      uint32_t maximum_bits = 0u;
      uint32_t base = group * 128u + local_block * 32u;
      for (uint32_t item = 0; item < 32u; ++item) {
        float up = accelerator_float_at(raw, base + item);
        float gate = accelerator_float_at(
            raw, BONSAI_FF_DIM + base + item);
        uint32_t up_bits;
        uint32_t gate_bits;
        memcpy(&up_bits, &up, sizeof(up_bits));
        memcpy(&gate_bits, &gate, sizeof(gate_bits));
        uint32_t value_bits = reliable_ffn_activation_bits(up_bits,
                                                           gate_bits);
        values[item] = value_bits;
        uint32_t magnitude = value_bits & UINT32_C(0x7fffffff);
        if (maximum_bits < magnitude) maximum_bits = magnitude;
      }
      fp16_t d16 = q8_scale_half_from_bits(maximum_bits);
      uint64_t reciprocal = q8_reciprocal_from_bits(maximum_bits);
      size_t offset = local_block * 34u;
      memcpy(packed + offset, &d16, sizeof(d16));
      for (uint32_t item = 0; item < 32u; ++item) {
        packed[offset + 2u + item] =
            (uint8_t)q8_sample_from_bits(values[item], maximum_bits,
                                         reciprocal);
      }
    }
    for (uint32_t word_index = 0; word_index < 17u; ++word_index) {
      uint64_t word;
      memcpy(&word, packed + word_index * 8u, sizeof(word));
      output[(size_t)group * 17u + word_index] = word;
    }
  }
}

static void ffn_shadow_validate_range(uint32_t begin, uint32_t end) {
  static int dumped;
  uint32_t element_begin = begin * 128u;
  uint32_t element_end = end * 128u;
  for (uint32_t i = element_begin; i < element_end; ++i) {
    float up = accelerator_float_at(g_job.raw, i);
    float gate = accelerator_float_at(g_job.raw, BONSAI_FF_DIM + i);
    g_ffn_shadow_values[i] = up * (gate * sigmoid_hardware_model(gate));
    g_job.out[i] = up * (gate * rcp_f(1.0f + exp_f(-gate)));
  }
  quantize_q8_group_range(g_ffn_shadow_values, g_ffn_shadow_packed, begin,
                          end);
  quantize_q8_group_range(g_job.out, g_ffn_shadow_exact_packed, begin, end);
  const uint8_t *candidate = (const uint8_t *)g_ffn_shadow_packed;
  const uint8_t *exact = (const uint8_t *)g_ffn_shadow_exact_packed;
  const volatile uint8_t *expected =
      (const volatile uint8_t *)g_job.quantize_output;
  if (!dumped) {
    dumped = 1;
    fprintf(stderr, "Bonsai FFN exact packed:");
    for (size_t i = 0; i < 68u; ++i)
      fprintf(stderr, " %02x", ((const uint8_t *)g_ffn_shadow_exact_packed)[i]);
    fprintf(stderr, "\nBonsai FFN direct packed:");
    for (size_t i = 0; i < 68u; ++i) fprintf(stderr, " %02x", expected[i]);
    fprintf(stderr, "\n");
  }
  uint64_t mismatches = 0u;
  uint64_t exact_mismatches = 0u;
  uint64_t scale_bytes = 0u;
  uint64_t scale_mismatches = 0u;
  uint64_t q_mismatches = 0u;
  uint32_t max_q_delta = 0u;
  size_t byte_begin = (size_t)begin * 17u * sizeof(uint64_t);
  size_t byte_end = (size_t)end * 17u * sizeof(uint64_t);
  for (size_t i = byte_begin; i < byte_end; ++i) {
    uint8_t wanted = expected[i];
    exact_mismatches += exact[i] != wanted;
    if (candidate[i] != wanted) {
      ++mismatches;
      if (i % 34u < 2u) {
        ++scale_mismatches;
      } else {
        ++q_mismatches;
        int delta = (int)(int8_t)candidate[i] - (int)(int8_t)wanted;
        uint32_t magnitude = (uint32_t)(delta < 0 ? -delta : delta);
        if (magnitude > max_q_delta) max_q_delta = magnitude;
      }
    }
    scale_bytes += i % 34u < 2u;
  }
  __atomic_fetch_add(&g_ffn_shadow_bytes, byte_end - byte_begin,
                     __ATOMIC_RELAXED);
  __atomic_fetch_add(&g_ffn_shadow_mismatches, mismatches, __ATOMIC_RELAXED);
  __atomic_fetch_add(&g_ffn_shadow_exact_mismatches, exact_mismatches,
                     __ATOMIC_RELAXED);
  __atomic_fetch_add(&g_ffn_shadow_scale_bytes, scale_bytes,
                     __ATOMIC_RELAXED);
  __atomic_fetch_add(&g_ffn_shadow_scale_mismatches, scale_mismatches,
                     __ATOMIC_RELAXED);
  __atomic_fetch_add(&g_ffn_shadow_q_mismatches, q_mismatches,
                     __ATOMIC_RELAXED);
  uint32_t prior = __atomic_load_n(&g_ffn_shadow_max_q_delta,
                                   __ATOMIC_RELAXED);
  while (prior < max_q_delta &&
         !__atomic_compare_exchange_n(&g_ffn_shadow_max_q_delta, &prior,
                                      max_q_delta, 0, __ATOMIC_RELAXED,
                                      __ATOMIC_RELAXED)) {}
}

static void norm_shadow_validate(float *out, const float *input,
                                 const float *weight, float scale,
                                 const volatile uint64_t *packed) {
  for (uint32_t i = 0; i < BONSAI_DIM; ++i)
    out[i] = input[i] * scale * weight[i];
  quantize_q8_group_range(out, g_norm_shadow_packed, 0u,
                          BONSAI_DIM / 128u);
  const uint8_t *expected = (const uint8_t *)g_norm_shadow_packed;
  const volatile uint8_t *actual = (const volatile uint8_t *)packed;
  size_t bytes = sizeof(g_norm_shadow_packed);
  if (!g_norm_shadow_dumped) {
    g_norm_shadow_dumped = 1;
    uint32_t scale_bits;
    memcpy(&scale_bits, &scale, sizeof(scale_bits));
    fprintf(stderr, "Bonsai norm shadow inputs: out=%p input=%p weight=%p scale=%08x",
            (void *)out, (const void *)input, (const void *)weight, scale_bits);
    for (size_t i = 0; i < 4u; ++i)
      fprintf(stderr, " i%zu=%08x w%zu=%08x o%zu=%08x", i,
              ((const uint32_t *)input)[i], i,
              ((const uint32_t *)weight)[i], i, ((const uint32_t *)out)[i]);
    fprintf(stderr, "\nBonsai norm expected:");
    for (size_t i = 0; i < 68u; ++i) fprintf(stderr, " %02x", expected[i]);
    fprintf(stderr, "\nBonsai norm actual:  ");
    for (size_t i = 0; i < 68u; ++i) fprintf(stderr, " %02x", actual[i]);
    fprintf(stderr, "\n");
  }
  for (size_t i = 0; i < bytes; ++i)
    g_norm_shadow_mismatches += expected[i] != actual[i];
  g_norm_shadow_bytes += bytes;
}

void bonsai_quantize_q8_to_scratch(const float *x, volatile uint64_t *output,
                                   int64_t k) {
  uint32_t groups = (uint32_t)k / 128u;
  if (g_worker_count > 1u && groups > 1u) {
    g_job.quantize_input = x;
    g_job.quantize_output = output;
    run_job(JOB_QUANTIZE_Q8, groups, 0);
  } else {
    quantize_q8_group_range(x, output, 0u, groups);
  }
}

void __real_ggml_vec_dot_q1_0_q8_0(int n, float *s, size_t bs, const void *vx,
                                   size_t bx, const void *vy, size_t by,
                                   int nrc) {
  (void)n; (void)s; (void)bs; (void)vx; (void)bx; (void)vy; (void)by;
  (void)nrc;
  fail("per-row fallback is not available in bonsai-lite");
}

/* ------------------------------------------------------------------ */

static const uint8_t *map_model(const char *path, size_t *size_out) {
#ifndef BONSAI_HOST_STUB
  const char *ddr_model = getenv("BONSAI_MODEL_DDR");
  if (ddr_model != NULL && strcmp(ddr_model, "0") != 0) {
    const uint8_t *data = bonsai_accelerator_map_staged_model(size_out);
    if (data == NULL) fail("cannot map staged DDR model");
    return data;
  }
#endif
  int fd = open(path, O_RDONLY);
  if (fd < 0) fail("cannot open model");
  struct stat st;
  if (fstat(fd, &st) != 0) fail("cannot stat model");
  void *data = mmap(NULL, (size_t)st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
  (void)close(fd);
  if (data == MAP_FAILED) fail("cannot map model");
  *size_out = (size_t)st.st_size;
  return (const uint8_t *)data;
}

static int run_server(void) {
  static char output_buffer[1u << 16];
  char line[1u << 20];
  static uint32_t cached_tokens[BONSAI_MAX_CONTEXT];
  static uint32_t requested_tokens[BONSAI_MAX_CONTEXT];
  static uint32_t generated_tokens[BONSAI_MAX_CONTEXT];
  uint32_t cached_count = 0u;
  uint32_t cached_next = 0u;
  const int stream_phases = getenv("BONSAI_STREAM_PHASES") != NULL;
  const int stream_tokens = getenv("BONSAI_STREAM_TOKENS") != NULL;
  /* stdout is a UART tty, so libc otherwise flushes every newline. A normal
     48-token reply contains about 100 protocol lines; buffer each logical
     phase and retain the explicit flushes below to collapse those writes. */
  if (setvbuf(stdout, output_buffer, _IOFBF, sizeof(output_buffer)) != 0)
    fail("cannot buffer server output");
  printf("READY\n");
  fflush(stdout);
  while (fgets(line, sizeof(line), stdin) != NULL) {
    char *p = line;
    if (p[0] == 'Q') return 0;
    if (strncmp(p, "RUN ", 4) != 0) fail("server expected RUN");
    p += 4;
    unsigned long max_tokens = strtoul(p, &p, 10);
    unsigned long count = strtoul(p, &p, 10);
    if (count == 0u || count > BONSAI_MAX_CONTEXT) fail("bad prompt length");
    for (unsigned long i = 0; i < count; ++i) {
      unsigned long id = strtoul(p, &p, 10);
      if (id >= BONSAI_VOCAB) fail("bad prompt token");
      requested_tokens[i] = (uint32_t)id;
    }
    uint32_t common = 0u;
    while (common < (uint32_t)count && common < cached_count &&
           requested_tokens[common] == cached_tokens[common])
      ++common;
    /* If a request truncates the previous one exactly at the common prefix,
       recompute its last token because cached_next belongs to the longer
       sequence. The KV entries before it remain valid. */
    if (common == (uint32_t)count && common < cached_count && common > 0u)
      --common;
    uint32_t next = common == cached_count ? cached_next : 0u;
    uint32_t position = common;
    uint64_t prompt_start = now_ns();
    for (uint32_t i = common; i < (uint32_t)count; ++i)
      next = forward(&g_model, &g_state, requested_tokens[i], position++);
    memcpy(cached_tokens, requested_tokens,
           (size_t)count * sizeof(*cached_tokens));
    cached_count = (uint32_t)count;
    cached_next = next;
    printf("PROMPT %u %u %llu\n", (uint32_t)count - common, common,
           (unsigned long long)(now_ns() - prompt_start));
    printf("DECODE_BEGIN\n");
    if (stream_phases) fflush(stdout);
    uint32_t generated_count = 0u;
    uint32_t evaluated_count = 0u;
    uint64_t decode_start = now_ns();
    for (unsigned long i = 0;
         i < max_tokens && position < BONSAI_MAX_CONTEXT; ++i) {
      uint32_t emitted = next;
      generated_tokens[generated_count] = emitted;
      ++generated_count;
      if (stream_tokens) {
        printf("TOKEN %u\n", emitted);
        fflush(stdout);
      }
      if (emitted == 151643u || emitted == 151645u) break;
      /* The final requested token is already known. Do not evaluate it only
         to compute a next token that this response will never emit. If a
         later chat turn includes it in the history, ordinary prefix replay
         evaluates it then. */
      if (i + 1u == max_tokens) break;
      uint64_t eval_start = stream_tokens ? now_ns() : 0u;
      cached_tokens[position] = emitted;
      next = forward(&g_model, &g_state, emitted, position++);
      cached_count = position;
      cached_next = next;
      ++evaluated_count;
      if (stream_tokens) {
        printf("EVAL %llu\n",
               (unsigned long long)(now_ns() - eval_start));
        fflush(stdout);
      }
    }
    uint64_t decode_elapsed = now_ns() - decode_start;
    printf("DECODE_END\n");
    if (stream_phases) fflush(stdout);
    /* Keep the UART record compact. The host serial bridge expands this back
       into TOKEN/EVAL/DONE records for existing clients. All evaluated tokens
       precede the one final token whose next-token evaluation is skipped. */
    if (!stream_tokens) {
      printf("RESULT %u %u %llu", generated_count, evaluated_count,
             (unsigned long long)decode_elapsed);
      for (uint32_t i = 0u; i < generated_count; ++i)
        printf(" %u", generated_tokens[i]);
      printf("\n");
    }
    printf("DONE\n");
    fflush(stdout);
  }
  return 0;
}

int main(int argc, char **argv) {
#if defined(__riscv_flen)
  /* Linux on this custom CVA6 platform can preserve an arbitrary dynamic
     rounding mode across exec.  The generated scalar FP instructions use
     frm, so make every fresh runtime start in deterministic IEEE RNE mode
     and clear accumulated exception flags. */
  uint32_t clean_fcsr = 0u;
  __asm__ volatile("fscsr %0" : : "r"(clean_fcsr) : "memory");
#endif
  if (getenv("BONSAI_STARTUP_PROFILE") != NULL) {
    fprintf(stderr, "Bonsai main entered\n");
    fflush(stderr);
  }
  const char *path = "/tmp/m.gguf";
  uint32_t workers = BONSAI_MAX_WORKERS;
  int server = 0;
  unsigned int generate = 8;
  static uint32_t prompt[BONSAI_MAX_CONTEXT];
  uint32_t prompt_count = 0;
  for (int i = 1; i < argc; ++i) {
    if (strcmp(argv[i], "--server") == 0) server = 1;
    else if (strcmp(argv[i], "--threads") == 0 && i + 1 < argc)
      workers = (uint32_t)strtoul(argv[++i], NULL, 10);
    else if (strcmp(argv[i], "--model") == 0 && i + 1 < argc)
      path = argv[++i];
    else if (strcmp(argv[i], "-n") == 0 && i + 1 < argc)
      generate = (unsigned int)strtoul(argv[++i], NULL, 10);
    else if (strcmp(argv[i], "--prompt-ids") == 0 && i + 1 < argc) {
      char *p = argv[++i];
      while (*p != 0 && prompt_count < BONSAI_MAX_CONTEXT) {
        prompt[prompt_count++] = (uint32_t)strtoul(p, &p, 10);
        if (*p == ',') p++;
      }
    } else {
      fail("usage: bonsai-lite [--model PATH] [--threads N] [--server] "
           "[-n COUNT] [--prompt-ids a,b,c]");
    }
  }
  uint64_t startup_begin = now_ns();
  size_t model_size;
  g_trace = getenv("BONSAI_TRACE") != NULL;
  g_gqa_attention = getenv("BONSAI_GQA_ATTENTION") != NULL;
  g_hardware_kv_prep = getenv("BONSAI_HARDWARE_KV_PREP") != NULL;
  g_profile = getenv("BONSAI_PROFILE") != NULL || g_trace;
  if (g_profile) (void)atexit(report_main_profile);
  g_ffn_shadow = getenv("BONSAI_FFN_SHADOW") != NULL;
  if (g_ffn_shadow) {
    (void)atexit(report_ffn_shadow);
  }
  g_norm_shadow = getenv("BONSAI_NORM_SHADOW") != NULL;
  if (g_norm_shadow) (void)atexit(report_norm_shadow);
  g_attn_shadow = getenv("BONSAI_ATTN_SHADOW") != NULL;
  if (g_attn_shadow) (void)atexit(report_attn_shadow);
  const char *attention_skip_mask = getenv("BONSAI_HW_ATTENTION_SKIP_MASK");
  if (attention_skip_mask != NULL)
    g_hw_attention_skip_mask =
        (uint32_t)strtoul(attention_skip_mask, NULL, 0);
  g_residual_cpu_visibility = g_trace || g_ffn_shadow || g_norm_shadow ||
      g_attn_shadow ||
      getenv("BONSAI_LAYER_TRACE") != NULL ||
      getenv("BONSAI_REPLAY_RESIDUAL") != NULL;
  const char *parallel_rows = getenv("BONSAI_PARALLEL_ROWS");
  if (parallel_rows != NULL) {
    unsigned long value = strtoul(parallel_rows, NULL, 10);
    if (value > 0u && value <= UINT32_MAX)
      g_parallel_rows = (uint32_t)value;
  }
  const char *layer_count = getenv("BONSAI_LAYERS");
  if (layer_count != NULL) {
    unsigned long value = strtoul(layer_count, NULL, 10);
    if (value > 0u && value <= BONSAI_LAYERS)
      g_layer_count = (uint32_t)value;
  }
  const char *spin_workers = getenv("BONSAI_SPIN_WORKERS");
  if (spin_workers != NULL && strcmp(spin_workers, "0") == 0)
    g_spin_workers = 0;
  const uint8_t *model_data = map_model(path, &model_size);
  model_init(&g_model, model_data, model_size);
  uint64_t model_ready = now_ns();
  rope_cache_init();
  attention_fixed_cache_init();
  uint64_t rope_ready = now_ns();
  /* g_state is static zero-filled storage, and each KV position is written
     before attention reads it.  An eager memset faults and writes almost
     59 MB on every launch, adding several seconds without changing state. */
  setenv("BONSAI_MODEL", path, 0);
  parallel_init(workers);
  uint64_t workers_ready = now_ns();
  if (!bonsai_accelerator_prepare())
    fail("FPGA accelerator or staged model is unavailable");
  g_accelerator_float_output = bonsai_accelerator_has_float_output();
  int hardware_transforms = getenv("BONSAI_HW_TRANSFORMS") != NULL &&
      strcmp(getenv("BONSAI_HW_TRANSFORMS"), "1") == 0;
  int hardware_norm = getenv("BONSAI_HW_NORM") != NULL ?
      strcmp(getenv("BONSAI_HW_NORM"), "1") == 0 : hardware_transforms;
  int hardware_ffn = getenv("BONSAI_HW_FFN") != NULL ?
      strcmp(getenv("BONSAI_HW_FFN"), "1") == 0 : hardware_transforms;
  int hardware_residual = getenv("BONSAI_HW_RESIDUAL") != NULL &&
      strcmp(getenv("BONSAI_HW_RESIDUAL"), "1") == 0;
  fprintf(stderr,
          "Bonsai runtime config: layers=%u/%u workers=%u attn_workers=%u "
          "model_bytes=%zu "
          "packed_model=%d float_output=%d spin_workers=%d "
          "hw_transforms=%d hw_norm=%d hw_ffn=%d hw_residual=%d "
          "hw_attention=%d "
          "hw_argmax=%d\n",
          g_layer_count, BONSAI_LAYERS, workers, g_attention_worker_count,
          model_size,
          g_model_packed, g_accelerator_float_output, g_spin_workers,
          hardware_transforms, hardware_norm, hardware_ffn,
          hardware_residual,
          bonsai_accelerator_has_attention(),
          getenv("BONSAI_HW_ARGMAX") != NULL &&
              strcmp(getenv("BONSAI_HW_ARGMAX"), "1") == 0);
  if (getenv("BONSAI_STARTUP_PROFILE") != NULL) {
    uint64_t accelerator_ready = now_ns();
    fprintf(stderr,
            "Bonsai startup profile: model=%.3fms rope=%.3fms workers=%.3fms "
            "accelerator=%.3fms total=%.3fms\n",
            (double)(model_ready - startup_begin) / 1e6,
            (double)(rope_ready - model_ready) / 1e6,
            (double)(workers_ready - rope_ready) / 1e6,
            (double)(accelerator_ready - workers_ready) / 1e6,
            (double)(accelerator_ready - startup_begin) / 1e6);
  }
  if (server) return run_server();

  if (prompt_count == 0) {
    prompt[0] = 12522u;
    prompt_count = 1;
  }
  fprintf(stderr, "bonsai-lite: %u prompt tokens, %u to generate\n",
          prompt_count, generate);
  uint32_t token = 0;
  uint32_t position = 0;
  uint64_t start = now_ns();
  for (uint32_t i = 0; i < prompt_count; ++i)
    token = forward(&g_model, &g_state, prompt[i], position++);
  uint64_t prompt_ns = now_ns() - start;
  fprintf(stderr, "prompt: %u tokens in %.2fs (%.2fs/token)\n", prompt_count,
          (double)prompt_ns / 1e9,
          (double)prompt_ns / 1e9 / (double)prompt_count);
  printf("%u", token);
  uint64_t decode_start = now_ns();
  uint32_t decoded = 0;
  for (unsigned int i = 0; i + 1u < generate && position < BONSAI_MAX_CONTEXT;
       ++i) {
    token = forward(&g_model, &g_state, token, position++);
    printf(",%u", token);
    fflush(stdout);
    decoded++;
    if (token == 151643u || token == 151645u) break;
  }
  printf("\n");
  uint64_t decode_ns = now_ns() - decode_start;
  if (decoded > 0)
    fprintf(stderr,
            "decode: %u tokens in %.3fs (%.6fs/token, %.6f tok/s)\n",
            decoded, (double)decode_ns / 1e9,
            (double)decode_ns / 1e9 / (double)decoded,
            (double)decoded * 1e9 / (double)decode_ns);
  return 0;
}

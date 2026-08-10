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

/* Polynomial exp and inline round keep every hot float operation on the
   single-precision hardware; libm's generic paths promote through soft
   double on this F-only ABI and cost tens of microseconds per call. */

/* The CVA6 FPU configuration omits fsqrt.s (the LFM decoder carries the
   same workaround); three Newton steps over an exponent-halved seed give
   float-accurate square roots using only mul/add/div. */
static inline float sqrt_f(float x) {
#ifdef BONSAI_HOST_STUB
  return sqrtf(x);
#else
  union { float f; uint32_t u; } v = {x};
  if (x <= 0.0f) return 0.0f;
  v.u = (v.u >> 1) + 0x1fc00000u;
  float y = v.f;
  y = 0.5f * (y + x / y);
  y = 0.5f * (y + x / y);
  y = 0.5f * (y + x / y);
  return y;
#endif
}


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

static inline float rcp_f(float x) {
  union { float f; uint32_t u; } v = {x};
  if (x == 0.0f) return 0.0f;
  v.u = 0x7ef311c2u - v.u;
  float y = v.f;
  y = y * (2.0f - x * y);
  y = y * (2.0f - x * y);
  return y;
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
  float q[BONSAI_DIM];
  float k[BONSAI_KV_DIM];
  float v[BONSAI_KV_DIM];
  float att[BONSAI_DIM];
  float ffn_up[BONSAI_FF_DIM];
  float ffn_gate[BONSAI_FF_DIM];
  float kv_k[BONSAI_LAYERS][BONSAI_MAX_CONTEXT][BONSAI_KV_DIM];
  float kv_v[BONSAI_LAYERS][BONSAI_MAX_CONTEXT][BONSAI_KV_DIM];
};

static struct model g_model;
static struct run_state g_state;
static float g_logits[BONSAI_VOCAB];
static float g_rope_cos[BONSAI_MAX_CONTEXT][BONSAI_HEAD_DIM / 2];
static float g_rope_sin[BONSAI_MAX_CONTEXT][BONSAI_HEAD_DIM / 2];
static uint64_t g_t_attn, g_t_ffn, g_t_norm, g_t_head;
static uint64_t g_forward_count;
static uint64_t g_job_ns[12], g_job_count[12];
static int g_profile;
static int g_trace;
static uint32_t g_parallel_rows = 16384u;

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
          "norm_apply=%.3fms/%llu\n",
          (double)g_job_ns[1] / 1e6, (unsigned long long)g_job_count[1],
          (double)g_job_ns[3] / 1e6, (unsigned long long)g_job_count[3],
          (double)g_job_ns[4] / 1e6, (unsigned long long)g_job_count[4],
          (double)g_job_ns[6] / 1e6, (unsigned long long)g_job_count[6],
          (double)g_job_ns[7] / 1e6, (unsigned long long)g_job_count[7],
          (double)g_job_ns[8] / 1e6, (unsigned long long)g_job_count[8],
          (double)g_job_ns[10] / 1e6, (unsigned long long)g_job_count[10],
          (double)g_job_ns[11] / 1e6, (unsigned long long)g_job_count[11]);
}

static float half_to_float(fp16_t h) {
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
  float out;
  memcpy(&out, &bits, sizeof(out));
  return out;
}

static fp16_t float_to_half(float value) {
  uint32_t bits;
  memcpy(&bits, &value, sizeof(bits));
  uint32_t sign = (bits >> 16) & 0x8000u;
  int32_t exponent = (int32_t)((bits >> 23) & 0xffu) - 127 + 15;
  uint32_t mantissa = bits & 0x7fffffu;
  if (exponent <= 0) return (fp16_t)sign;
  if (exponent >= 0x1f) return (fp16_t)(sign | 0x7c00u);
  return (fp16_t)(sign | ((uint32_t)exponent << 10) | (mantissa >> 13));
}

#ifdef BONSAI_HOST_STUB
static float stub_reference_row(const uint8_t *base, uint32_t columns,
                                const float *input, uint32_t row) {
  const uint32_t blocks = columns / 128u;
  const uint8_t *r = base + (size_t)row * blocks * 18u;
  float total = 0.0f;
  for (uint32_t b = 0; b < blocks; ++b) {
    fp16_t d_bits;
    memcpy(&d_bits, r + b * 18u, 2);
    float d = half_to_float(d_bits);
    float dot = 0.0f;
    for (uint32_t i = 0; i < 128u; ++i) {
      uint32_t bit = (r[b * 18u + 2u + i / 8u] >> (i % 8u)) & 1u;
      float v = input[b * 128u + i];
      dot += bit ? v : -v;
    }
    total += dot * d;
  }
  return total;
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
  (void)rows;
  stub_weights = weights;
  stub_columns = columns;
  stub_input = input;
  return 1;
}

void bonsai_accelerate_read_rows(float *output, uint32_t first,
                                 uint32_t count) {
  for (uint32_t i = 0; i < count; ++i)
    output[first + i] = stub_reference_row(
        (const uint8_t *)stub_weights, stub_columns, stub_input, first + i);
}

void bonsai_accelerate_release(void) {}

int bonsai_accelerate_f32_matvec_store(const void *weights, uint32_t columns,
                                       uint32_t rows, const float *input,
                                       uint32_t output_first) {
  for (uint32_t row = 0; row < rows; ++row)
    stub_stored[output_first + row] = stub_float_to_q30(stub_reference_row(
        (const uint8_t *)weights, columns, input, row));
  return 1;
}

void bonsai_accelerate_read_stored_rows(float *output, uint32_t source_first,
                                        uint32_t count) {
  for (uint32_t i = 0; i < count; ++i)
    output[i] = (float)stub_stored[source_first + i] *
                (1.0f / 1073741824.0f);
}

const volatile int64_t *bonsai_accelerate_stored_output(void) {
  return stub_stored;
}

void bonsai_accelerate_argmax_rows(uint32_t first, uint32_t count,
                                   int64_t *best_raw, uint32_t *best_token) {
  int64_t score = INT64_MIN;
  uint32_t token = first;
  for (uint32_t row = first; row < first + count; ++row) {
    int64_t value = stub_float_to_q30(stub_reference_row(
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
    int64_t value = stub_float_to_q30(stub_reference_row(
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

static void model_init(struct model *model, const uint8_t *base, size_t size) {
  model->base = base;
  model->size = size;
  struct cursor c = { base, base + size };
  if (take_u32(&c) != 0x46554747u) fail("not gguf");
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
    const char *name = pending[i].name;
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
}

/* ------------------------------------------------------------------ */
/* YaRN rotary cache, ported from ggml (MIT, Quesnelle & Peng).        */

static float yarn_corr_dim(float n_rot) {
  /* Double math routes through soft-float, avoiding the broken fdiv.s. */
  double ratio = (double)BONSAI_YARN_ORIG_CTX /
                 ((double)n_rot * 2.0 * M_PI);
  return (float)((double)BONSAI_HEAD_DIM * log(ratio) /
                 (2.0 * log((double)BONSAI_ROPE_BASE)));
}

static void rope_cache_init(void) {
  float corr_low = floorf(yarn_corr_dim(BONSAI_YARN_BETA_FAST));
  float corr_high = ceilf(yarn_corr_dim(BONSAI_YARN_BETA_SLOW));
  if (corr_low < 0.0f) corr_low = 0.0f;
  if (corr_high > (float)(BONSAI_HEAD_DIM - 1u))
    corr_high = (float)(BONSAI_HEAD_DIM - 1u);
  const float freq_scale = 1.0f / BONSAI_YARN_FACTOR;
  const float theta_scale =
      powf(BONSAI_ROPE_BASE, -2.0f / (float)BONSAI_HEAD_DIM);
  const float mscale_base = 1.0f + 0.1f * logf(1.0f / freq_scale);
  for (uint32_t position = 0; position < BONSAI_MAX_CONTEXT; ++position) {
    float theta_extrap = (float)position;
    for (uint32_t i = 0; i < BONSAI_HEAD_DIM / 2u; ++i) {
      float span = corr_high - corr_low > 0.001f ? corr_high - corr_low
                                                  : 0.001f;
      float y = ((float)i - corr_low) * rcp_f(span);
      float ramp = 1.0f - (y < 0.0f ? 0.0f : (y > 1.0f ? 1.0f : y));
      float theta_interp = freq_scale * theta_extrap;
      float theta = theta_interp * (1.0f - ramp) + theta_extrap * ramp;
      g_rope_cos[position][i] = cosf(theta) * mscale_base;
      g_rope_sin[position][i] = sinf(theta) * mscale_base;
      theta_extrap *= theta_scale;
    }
  }
}

static void rope_head(float *v, uint32_t position) {
  const float *cs = g_rope_cos[position];
  const float *sn = g_rope_sin[position];
  const uint32_t half = BONSAI_HEAD_DIM / 2u;
  for (uint32_t i = 0; i < half; ++i) {
    float a = v[i];
    float b = v[i + half];
    v[i] = a * cs[i] - b * sn[i];
    v[i + half] = a * sn[i] + b * cs[i];
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
static int g_spin_workers = 1;
static pthread_t g_threads[BONSAI_MAX_WORKERS - 1];

static void attention_head_range(uint32_t begin, uint32_t end,
                                 uint32_t position);
static void quantize_q8_group_range(const float *input,
                                    volatile uint64_t *output,
                                    uint32_t begin, uint32_t end);
static void attention_prep_head_range(uint32_t begin, uint32_t end,
                                      uint32_t position,
                                      const struct layer *layer);

static inline float q30_to_float(int64_t value) {
  int32_t high = (int32_t)(value >> 32);
  uint32_t low = (uint32_t)value;
  return (float)high * 4.0f + (float)low * (1.0f / 1073741824.0f);
}

static void execute_job(uint32_t worker) {
  uint32_t begin = (uint32_t)(((uint64_t)g_job.rows * worker) / g_worker_count);
  uint32_t end =
      (uint32_t)(((uint64_t)g_job.rows * (worker + 1u)) / g_worker_count);
  if (g_job.kind == JOB_ATTN_HEADS) {
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
  } else if (g_job.kind == JOB_READ_STORED_ROWS) {
    if (begin < end)
      bonsai_accelerate_read_stored_rows(g_job.out + begin, begin,
                                         end - begin);
  } else if (g_job.kind == JOB_QUANTIZE_Q8) {
    if (begin < end)
      quantize_q8_group_range(g_job.quantize_input, g_job.quantize_output,
                              begin, end);
  } else if (g_job.kind == JOB_ATTN_READ_PREP) {
    for (uint32_t h = begin; h < end; ++h) {
      for (uint32_t i = 0; i < BONSAI_HEAD_DIM; ++i)
        g_state.q[h * BONSAI_HEAD_DIM + i] =
            q30_to_float(g_job.raw[h * BONSAI_HEAD_DIM + i]);
      if (h < BONSAI_KV_HEADS) {
        for (uint32_t i = 0; i < BONSAI_HEAD_DIM; ++i) {
          uint32_t index = h * BONSAI_HEAD_DIM + i;
          g_state.k[index] =
              q30_to_float(g_job.raw[BONSAI_DIM + index]);
          g_state.v[index] =
              q30_to_float(g_job.raw[BONSAI_DIM + BONSAI_KV_DIM + index]);
        }
      }
    }
    attention_prep_head_range(begin, end, g_job.position, g_job.layer);
  } else if (g_job.kind == JOB_FFN_READ_ACTIVATE) {
    for (uint32_t i = begin; i < end; ++i) {
      float up = q30_to_float(g_job.raw[i]);
      float gate = q30_to_float(g_job.raw[BONSAI_FF_DIM + i]);
      g_job.out[i] = up * (gate * rcp_f(1.0f + exp_f(-gate)));
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
  }
}

static void *worker_start(void *argument) {
  uint32_t worker = (uint32_t)(uintptr_t)argument;
  uint32_t seen = 0;
  for (;;) {
    uint32_t epoch;
    do {
      epoch = __atomic_load_n(&g_job.epoch, __ATOMIC_ACQUIRE);
      if (epoch == seen && !g_spin_workers)
        (void)syscall(SYS_futex, &g_job.epoch, FUTEX_WAIT_PRIVATE, seen,
                      NULL, NULL, 0);
      else if (epoch == seen)
        __asm__ volatile("nop");
    } while (epoch == seen);
    execute_job(worker);
    __atomic_store_n(&g_job.slots[worker].done, epoch, __ATOMIC_RELEASE);
    seen = epoch;
  }
  return NULL;
}

static void parallel_init(uint32_t workers) {
  if (workers < 1u) workers = 1u;
  if (workers > BONSAI_MAX_WORKERS) workers = BONSAI_MAX_WORKERS;
  g_worker_count = workers;
  for (uint32_t worker = 1; worker < workers; ++worker)
    if (pthread_create(&g_threads[worker - 1u], NULL, worker_start,
                       (void *)(uintptr_t)worker) != 0)
      fail("cannot create worker");
}

static void run_job(uint32_t kind, uint32_t rows, uint32_t position) {
  uint64_t profile_start = g_profile ? now_ns() : 0u;
  g_job.kind = kind;
  g_job.rows = rows;
  g_job.position = position;
  uint32_t epoch = __atomic_load_n(&g_job.epoch, __ATOMIC_RELAXED) + 1u;
  if (epoch == 0u) epoch = 1u;
  __atomic_store_n(&g_job.epoch, epoch, __ATOMIC_RELEASE);
  if (g_worker_count > 1u && !g_spin_workers)
    (void)syscall(SYS_futex, &g_job.epoch, FUTEX_WAKE_PRIVATE,
                  (int)(g_worker_count - 1u), NULL, NULL, 0);
  execute_job(0);
  __atomic_store_n(&g_job.slots[0].done, epoch, __ATOMIC_RELEASE);
  for (uint32_t worker = 1; worker < g_worker_count; ++worker)
    while (__atomic_load_n(&g_job.slots[worker].done, __ATOMIC_ACQUIRE) !=
           epoch)
      ;
  if (g_profile && kind < 12u) {
    g_job_ns[kind] += now_ns() - profile_start;
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
    fp16_t d_bits;
    memcpy(&d_bits, r + b * 18u, 2);
    float d = half_to_float(d_bits);
    float dot = 0.0f;
    for (uint32_t i = 0; i < 128u; ++i) {
      uint32_t bit = (r[b * 18u + 2u + i / 8u] >> (i % 8u)) & 1u;
      float v = input[b * 128u + i];
      dot += bit ? v : -v;
    }
    total += dot * d;
  }
  return total;
}

static void matvec_accel(float *out, const struct tensor *weight,
                         const float *input) {
  if (weight->type != GGUF_TYPE_Q1_0) fail("expected Q1_0 tensor");
  if (!bonsai_accelerate_f32_matvec_submit(weight->data, weight->ne0,
                                           weight->ne1, input))
    fail("accelerator declined matvec");
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
    const uint8_t *b = row + block * 18u;
    fp16_t d_bits;
    memcpy(&d_bits, b, 2);
    float d = half_to_float(d_bits);
    for (uint32_t i = 0; i < 128u; ++i) {
      uint32_t bit = (b[2u + i / 8u] >> (i % 8u)) & 1u;
      out[block * 128u + i] = bit ? d : -d;
    }
  }
}

static void rms_norm_add(float *out, float *x, const float *add,
                         const struct tensor *weight, uint32_t n) {
  if (weight->type != GGUF_TYPE_F32 || weight->ne0 != n) fail("bad norm");
  const float *w = (const float *)weight->data;
  g_job.out = x;
  g_job.input = add;
  run_job(JOB_NORM_SUM, n, 0u);
  float sum = 0.0f;
  for (uint32_t worker = 0; worker < g_worker_count; ++worker)
    sum += g_job.slots[worker].score;
  float scale = rsqrt_f(sum * (1.0f / (float)BONSAI_DIM) + BONSAI_EPS);
  g_job.out = out;
  g_job.input = x;
  g_job.norm_weight = w;
  g_job.norm_scale = scale;
  run_job(JOB_NORM_APPLY, n, 0u);
}

static void rms_norm(float *out, float *x, const struct tensor *weight,
                     uint32_t n) {
  rms_norm_add(out, x, NULL, weight, n);
}

static void head_rms_norm_one(float *v, const struct tensor *weight) {
  const float *w = (const float *)weight->data;
  float sum = 0.0f;
  for (uint32_t i = 0; i < BONSAI_HEAD_DIM; ++i) sum += v[i] * v[i];
  float scale = rsqrt_f(sum * (1.0f / (float)BONSAI_HEAD_DIM) + BONSAI_EPS);
  for (uint32_t i = 0; i < BONSAI_HEAD_DIM; ++i) v[i] *= scale * w[i];
}

static void attention_prep_head_range(uint32_t begin, uint32_t end,
                                      uint32_t position,
                                      const struct layer *layer) {
  uint32_t layer_index = (uint32_t)(layer - g_model.layers);
  float *kc = g_state.kv_k[layer_index][position];
  float *vc = g_state.kv_v[layer_index][position];
  for (uint32_t h = begin; h < end; ++h) {
    float *q = g_state.q + h * BONSAI_HEAD_DIM;
    head_rms_norm_one(q, &layer->q_norm);
    rope_head(q, position);
    if (h < BONSAI_KV_HEADS) {
      float *k = g_state.k + h * BONSAI_HEAD_DIM;
      float *v = g_state.v + h * BONSAI_HEAD_DIM;
      head_rms_norm_one(k, &layer->k_norm);
      rope_head(k, position);
      for (uint32_t i = 0; i < BONSAI_HEAD_DIM; ++i) {
        kc[h * BONSAI_HEAD_DIM + i] = half_to_float(float_to_half(k[i]));
        vc[h * BONSAI_HEAD_DIM + i] = half_to_float(float_to_half(v[i]));
      }
    }
  }
}

static const struct layer *g_attn_layer;

static void attention_head_range(uint32_t begin, uint32_t end,
                                 uint32_t position) {
  const struct layer *layer = g_attn_layer;
  uint32_t layer_index = (uint32_t)(layer - g_model.layers);
  const float inv_sqrt_head = 0.08838834764831845f; /* 1/sqrt(128) */
  float scores[BONSAI_MAX_CONTEXT];
  for (uint32_t h = begin; h < end; ++h) {
    uint32_t kvh = h / (BONSAI_HEADS / BONSAI_KV_HEADS);
    float max_score = -3.402823466e38f;
    for (uint32_t t = 0; t <= position; ++t) {
      const float *key = g_state.kv_k[layer_index][t] + kvh * BONSAI_HEAD_DIM;
      float score = 0.0f;
      for (uint32_t j = 0; j < BONSAI_HEAD_DIM; ++j)
        score += g_state.q[h * BONSAI_HEAD_DIM + j] * key[j];
      score *= inv_sqrt_head;
      scores[t] = score;
      if (score > max_score) max_score = score;
    }
    float denom = 0.0f;
    for (uint32_t t = 0; t <= position; ++t) {
      float e = exp_f(scores[t] - max_score);
      scores[t] = e;
      denom += e;
    }
    float inv_denom = rcp_f(denom);
    for (uint32_t j = 0; j < BONSAI_HEAD_DIM; ++j) {
      float sum = 0.0f;
      for (uint32_t t = 0; t <= position; ++t) {
        const float *value =
            g_state.kv_v[layer_index][t] + kvh * BONSAI_HEAD_DIM;
        sum += scores[t] * value[j];
      }
      g_state.att[h * BONSAI_HEAD_DIM + j] = sum * inv_denom;
    }
  }
}

static void attention(struct run_state *state, const struct layer *layer,
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
  g_job.raw = bonsai_accelerate_stored_output();
  g_job.layer = layer;
  run_job(JOB_ATTN_READ_PREP, BONSAI_HEADS, position);
  g_attn_layer = layer;
  if (g_profile) {
    uint64_t th = now_ns();
    run_job(JOB_ATTN_HEADS, BONSAI_HEADS, position);
    g_t_head += now_ns() - th;
  } else {
    run_job(JOB_ATTN_HEADS, BONSAI_HEADS, position);
  }
  matvec_accel(state->op, &layer->attn_out, state->att);
}

static void feed_forward(struct run_state *state, const struct layer *layer) {
  if (!bonsai_accelerate_f32_matvec_store(
          layer->ffn_up.data, layer->ffn_up.ne0, layer->ffn_up.ne1,
          state->norm, 0u) ||
      !bonsai_accelerate_f32_matvec_store(
          layer->ffn_gate.data, layer->ffn_gate.ne0, layer->ffn_gate.ne1,
          state->norm, BONSAI_FF_DIM))
    fail("accelerator declined batched FFN matvec");
  g_job.out = state->ffn_up;
  g_job.raw = bonsai_accelerate_stored_output();
  run_job(JOB_FFN_READ_ACTIVATE, BONSAI_FF_DIM, 0u);
  matvec_accel(state->op, &layer->ffn_down, state->ffn_up);
}

static uint32_t forward(struct model *model, struct run_state *state,
                        uint32_t token, uint32_t position) {
  if (position >= BONSAI_MAX_CONTEXT) fail("context full");
  embedding_row(state->x, &model->embedding, token);
  if (position == 0 && g_trace) {
    const float *w = (const float *)model->layers[0].attn_norm.data;
    volatile float fa = 3.0f, fb = 2.0f;
    volatile float fdiv_test = fa / fb;
    volatile float fsum = fa + fb;
    volatile float fmul = fa * fb;
    fprintf(stderr, "[trace0] fdiv 3/2=%g add=%g mul=%g sqrt_f(2)=%g\n",
            (double)fdiv_test, (double)fsum, (double)fmul,
            (double)sqrt_f(2.0f));
    fprintf(stderr,
            "[trace0] emb x[0..2]=%g %g %g attn_norm w[0..2]=%g %g %g "
            "(ne0=%u type=%u data=%p base=%p)\n",
            (double)state->x[0], (double)state->x[1], (double)state->x[2],
            (double)w[0], (double)w[1], (double)w[2],
            model->layers[0].attn_norm.ne0, model->layers[0].attn_norm.type,
            (const void *)model->layers[0].attn_norm.data,
            (const void *)model->base);
  }
  for (uint32_t il = 0; il < BONSAI_LAYERS; ++il) {
    struct layer *layer = &model->layers[il];
    uint64_t t0 = g_profile ? now_ns() : 0u;
    if (il == 0u)
      rms_norm(state->norm, state->x, &layer->attn_norm, BONSAI_DIM);
    else
      rms_norm_add(state->norm, state->x, state->op, &layer->attn_norm,
                   BONSAI_DIM);
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
      fflush(stderr);
    }
    uint64_t t1 = g_profile ? now_ns() : 0u;
    attention(state, layer, position);
    uint64_t t2 = g_profile ? now_ns() : 0u;
    rms_norm_add(state->norm, state->x, state->op, &layer->ffn_norm,
                 BONSAI_DIM);
    uint64_t t3 = g_profile ? now_ns() : 0u;
    feed_forward(state, layer);
    uint64_t t4 = g_profile ? now_ns() : 0u;
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
  rms_norm_add(state->norm, state->x, state->op, &model->output_norm,
               BONSAI_DIM);
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
    float amax = 0.0f;
    for (int j = 0; j < 32; j++) {
      const float v = fabsf(x[i * 32 + j]);
      if (amax < v) amax = v;
    }
    const float d = amax * (1.0f / 127.0f);
    const float id = rcp_f(d);
    y[i].d = float_to_half(d);
    for (int j = 0; j < 32; j++)
      y[i].qs[j] = (int8_t)round_f(x[i * 32 + j] * id);
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
  uint64_t word = 0u;
  uint32_t bytes = 0u;
  size_t word_index = (size_t)begin * 17u;
  for (uint32_t block = begin * 4u; block < end * 4u; ++block) {
    float amax = 0.0f;
    for (int j = 0; j < 32; ++j) {
      float value = x[block * 32 + j];
      if (value < 0.0f) value = -value;
      if (amax < value) amax = value;
    }
    const float d = amax * (1.0f / 127.0f);
    const float id = rcp_f(d);
    const fp16_t d16 = float_to_half(d);
    for (int item = -2; item < 32; ++item) {
      uint8_t byte;
      if (item == -2)
        byte = (uint8_t)d16;
      else if (item == -1)
        byte = (uint8_t)(d16 >> 8);
      else
        byte = (uint8_t)(int8_t)round_f(x[block * 32 + item] * id);
      word |= (uint64_t)byte << (bytes * 8u);
      if (++bytes == 8u) {
        output[word_index++] = word;
        word = 0u;
        bytes = 0u;
      }
    }
  }
  if (bytes != 0u) output[word_index] = word;
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
  char line[1u << 20];
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
    memset(&g_state, 0, sizeof(g_state));
    uint32_t next = 0;
    uint32_t position = 0;
    uint64_t prompt_start = now_ns();
    for (unsigned long i = 0; i < count; ++i) {
      unsigned long id = strtoul(p, &p, 10);
      if (id >= BONSAI_VOCAB) fail("bad prompt token");
      next = forward(&g_model, &g_state, (uint32_t)id, position++);
    }
    printf("PROMPT %lu %llu\n", count,
           (unsigned long long)(now_ns() - prompt_start));
    for (unsigned long i = 0;
         i < max_tokens && position < BONSAI_MAX_CONTEXT; ++i) {
      uint32_t emitted = next;
      printf("TOKEN %u\n", emitted);
      fflush(stdout);
      if (emitted == 151643u || emitted == 151645u) break;
      uint64_t start = now_ns();
      next = forward(&g_model, &g_state, emitted, position++);
      printf("EVAL %llu\n", (unsigned long long)(now_ns() - start));
    }
    printf("DONE\n");
    fflush(stdout);
  }
  return 0;
}

int main(int argc, char **argv) {
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
  size_t model_size;
  g_trace = getenv("BONSAI_TRACE") != NULL;
  g_profile = getenv("BONSAI_PROFILE") != NULL || g_trace;
  if (g_profile) (void)atexit(report_main_profile);
  const char *parallel_rows = getenv("BONSAI_PARALLEL_ROWS");
  if (parallel_rows != NULL) {
    unsigned long value = strtoul(parallel_rows, NULL, 10);
    if (value > 0u && value <= UINT32_MAX)
      g_parallel_rows = (uint32_t)value;
  }
  const char *spin_workers = getenv("BONSAI_SPIN_WORKERS");
  if (spin_workers != NULL && strcmp(spin_workers, "0") == 0)
    g_spin_workers = 0;
  const uint8_t *model_data = map_model(path, &model_size);
  model_init(&g_model, model_data, model_size);
  rope_cache_init();
  memset(&g_state, 0, sizeof(g_state));
  setenv("BONSAI_MODEL", path, 0);
  parallel_init(workers);
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
    fprintf(stderr, "decode: %u tokens in %.2fs (%.3fs/token, %.3f tok/s)\n",
            decoded, (double)decode_ns / 1e9,
            (double)decode_ns / 1e9 / (double)decoded,
            (double)decoded * 1e9 / (double)decode_ns);
  return 0;
}

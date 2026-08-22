#define _GNU_SOURCE

#include <math.h>
#include <pthread.h>
#include <sched.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { HARTS = 4, ELEMENTS = 4096, REPEATS = 100 };

struct result {
  uint64_t add;
  uint64_t mul;
  uint64_t madd;
  uint64_t norm;
  uint32_t dot;
  unsigned mismatches;
};

struct worker_arg {
  unsigned hart;
  struct result *result;
};

static float a[ELEMENTS];
static float b[ELEMENTS];
static float c[ELEMENTS];

static void fence_memory(void) {
#ifdef __riscv
  __asm__ volatile("fence rw, rw" ::: "memory");
#else
  __atomic_thread_fence(__ATOMIC_SEQ_CST);
#endif
}

static uint64_t hash_floats(const float *values) {
  uint64_t hash = UINT64_C(1469598103934665603);
  for (unsigned i = 0; i < ELEMENTS; ++i) {
    uint32_t bits;
    memcpy(&bits, &values[i], sizeof(bits));
    hash ^= bits;
    hash *= UINT64_C(1099511628211);
  }
  return hash;
}

static void *worker(void *opaque) {
  struct worker_arg *arg = opaque;
  cpu_set_t cpus;
  CPU_ZERO(&cpus);
  CPU_SET(arg->hart, &cpus);
  if (pthread_setaffinity_np(pthread_self(), sizeof(cpus), &cpus) != 0) {
    arg->result->mismatches = UINT32_MAX;
    return NULL;
  }

  float output[ELEMENTS];
  struct result baseline = {0};
  for (unsigned repeat = 0; repeat < REPEATS; ++repeat) {
    struct result current = {0};
    for (unsigned i = 0; i < ELEMENTS; ++i)
      output[i] = a[i] + b[i];
    fence_memory();
    current.add = hash_floats(output);

    for (unsigned i = 0; i < ELEMENTS; ++i)
      output[i] = a[i] * b[i];
    fence_memory();
    current.mul = hash_floats(output);

    for (unsigned i = 0; i < ELEMENTS; ++i)
      output[i] = a[i] * b[i] + c[i];
    fence_memory();
    current.madd = hash_floats(output);

    float sum = 0.0f;
    for (unsigned i = 0; i < ELEMENTS; ++i)
      sum += output[i] * output[i];
    /* Keep the stress test on the enabled ADDMUL path. DIVSQRT is disabled in
       this FPGA configuration, so executing fsqrt.s would intentionally stall. */
    float scale = 0.125f;
    for (unsigned i = 0; i < ELEMENTS; ++i)
      output[i] *= scale;
    fence_memory();
    current.norm = hash_floats(output);
    memcpy(&current.dot, &sum, sizeof(sum));

    if (repeat == 0) {
      baseline = current;
    } else if (current.add != baseline.add || current.mul != baseline.mul ||
               current.madd != baseline.madd || current.norm != baseline.norm ||
               current.dot != baseline.dot) {
      ++baseline.mismatches;
    }
  }
  *arg->result = baseline;
  return NULL;
}

int main(int argc, char **argv) {
  unsigned active_harts = HARTS;
  if (argc == 2) {
    active_harts = (unsigned)strtoul(argv[1], NULL, 0);
    if (active_harts == 0 || active_harts > HARTS)
      return 2;
  }
  for (unsigned i = 0; i < ELEMENTS; ++i) {
    int32_t x = (int32_t)((i * 1103515245u + 12345u) & 0xffffu) - 32768;
    int32_t y = (int32_t)((i * 2654435761u + 1013904223u) & 0xffffu) - 32768;
    a[i] = (float)x / 8192.0f;
    b[i] = (float)y / 16384.0f;
    c[i] = (float)((int)(i % 257u) - 128) / 256.0f;
  }

  pthread_t threads[HARTS];
  struct worker_arg args[HARTS];
  struct result results[HARTS] = {0};
  for (unsigned hart = 0; hart < active_harts; ++hart) {
    args[hart] = (struct worker_arg){.hart = hart, .result = &results[hart]};
    if (pthread_create(&threads[hart], NULL, worker, &args[hart]) != 0)
      return 2;
  }
  for (unsigned hart = 0; hart < active_harts; ++hart)
    pthread_join(threads[hart], NULL);

  unsigned failed = 0;
  for (unsigned hart = 0; hart < active_harts; ++hart) {
    printf("hart%u add=%016llx mul=%016llx madd=%016llx norm=%016llx "
           "dot=%08x mismatches=%u\n",
           hart, (unsigned long long)results[hart].add,
           (unsigned long long)results[hart].mul,
           (unsigned long long)results[hart].madd,
           (unsigned long long)results[hart].norm, results[hart].dot,
           results[hart].mismatches);
    if (results[hart].mismatches != 0 ||
        memcmp(&results[hart], &results[0],
               offsetof(struct result, mismatches)) != 0)
      failed = 1;
  }
  puts(failed ? "FAIL" : "PASS");
  return failed;
}

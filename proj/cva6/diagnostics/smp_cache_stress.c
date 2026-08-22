#define _GNU_SOURCE

#include <errno.h>
#include <pthread.h>
#include <sched.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

enum { HARTS = 4, SLOTS = 16, ROUNDS = 20000, COUNTS = 250000 };

struct shared_state {
  _Alignas(64) atomic_uint arrived;
  atomic_uint phase;
  atomic_uint failures;
  atomic_uint_fast64_t counter;
  _Alignas(64) uint64_t slot[HARTS][SLOTS];
};

struct worker_arg {
  struct shared_state *shared;
  unsigned hart;
};

static void barrier(struct shared_state *shared) {
  unsigned phase = atomic_load_explicit(&shared->phase, memory_order_acquire);
  if (atomic_fetch_add_explicit(&shared->arrived, 1, memory_order_acq_rel) ==
      HARTS - 1) {
    atomic_store_explicit(&shared->arrived, 0, memory_order_relaxed);
    atomic_fetch_add_explicit(&shared->phase, 1, memory_order_release);
  } else {
    while (atomic_load_explicit(&shared->phase, memory_order_acquire) == phase)
      __asm__ volatile("nop");
  }
}

static uint64_t expected_value(unsigned round, unsigned hart, unsigned slot) {
  return 0xc06e000000000000ULL ^ ((uint64_t)round << 24) ^
         ((uint64_t)hart << 16) ^ slot;
}

static void *worker(void *opaque) {
  struct worker_arg *arg = opaque;
  struct shared_state *shared = arg->shared;
  cpu_set_t cpus;
  CPU_ZERO(&cpus);
  CPU_SET(arg->hart, &cpus);
  if (pthread_setaffinity_np(pthread_self(), sizeof(cpus), &cpus) != 0)
    atomic_fetch_add_explicit(&shared->failures, 1, memory_order_relaxed);

  for (unsigned round = 0; round < ROUNDS; ++round) {
    for (unsigned slot = 0; slot < SLOTS; ++slot)
      shared->slot[arg->hart][slot] =
          expected_value(round, arg->hart, slot);
    barrier(shared);

    for (unsigned hart = 0; hart < HARTS; ++hart) {
      for (unsigned slot = 0; slot < SLOTS; ++slot) {
        if (shared->slot[hart][slot] != expected_value(round, hart, slot))
          atomic_fetch_add_explicit(&shared->failures, 1,
                                    memory_order_relaxed);
      }
    }
    barrier(shared);
  }

  for (unsigned count = 0; count < COUNTS; ++count)
    atomic_fetch_add_explicit(&shared->counter, 1, memory_order_relaxed);
  return NULL;
}

int main(void) {
  struct shared_state *shared = aligned_alloc(64, sizeof(*shared));
  if (shared == NULL) {
    perror("aligned_alloc");
    return 1;
  }
  *shared = (struct shared_state){0};

  pthread_t threads[HARTS];
  struct worker_arg args[HARTS];
  for (unsigned hart = 0; hart < HARTS; ++hart) {
    args[hart] = (struct worker_arg){.shared = shared, .hart = hart};
    int error = pthread_create(&threads[hart], NULL, worker, &args[hart]);
    if (error != 0) {
      errno = error;
      perror("pthread_create");
      return 1;
    }
  }
  for (unsigned hart = 0; hart < HARTS; ++hart)
    pthread_join(threads[hart], NULL);

  uint64_t count = atomic_load_explicit(&shared->counter, memory_order_relaxed);
  unsigned failures =
      atomic_load_explicit(&shared->failures, memory_order_relaxed);
  uint64_t expected_count = (uint64_t)HARTS * COUNTS;
  if (failures != 0 || count != expected_count) {
    fprintf(stderr, "FAIL: failures=%u counter=%llu expected=%llu\n", failures,
            (unsigned long long)count, (unsigned long long)expected_count);
    return 1;
  }
  printf("PASS: %u harts, %u coherence rounds, %llu atomic increments\n",
         HARTS, ROUNDS, (unsigned long long)count);
  free(shared);
  return 0;
}

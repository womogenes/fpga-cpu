#ifndef PROJECT_SYSTEM_H
#define PROJECT_SYSTEM_H

static inline void flush_cpu_dcache(void) {
#ifdef CONFIG_CPU_HAS_DCACHE
  __asm__ volatile(".word 0x500f" ::: "memory");
#endif
}

void flush_l2_cache(void);

#endif

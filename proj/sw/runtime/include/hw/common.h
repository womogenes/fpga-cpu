#ifndef PROJECT_HW_COMMON_H
#define PROJECT_HW_COMMON_H

#include <stdint.h>

#define CSR_ACCESSORS_DEFINED

static inline void csr_write_simple(unsigned long value, unsigned long address) {
  *(volatile uint32_t*)address = (uint32_t)value;
}

static inline unsigned long csr_read_simple(unsigned long address) {
  return *(volatile uint32_t*)address;
}

#endif

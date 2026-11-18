#ifndef SOLO_HW_H
#define SOLO_HW_H

#include <stdint.h>

#define MMIO_BASE 0x10000000u
#define REG(offset) (*(volatile uint32_t *)(MMIO_BASE + (offset)))

#define UART_TX REG(0x00)
#define UART_RX REG(0x04)
#define UART_STATUS REG(0x08)
#define ACCEL_COMMAND REG(0x10)
#define ACCEL_STATUS REG(0x14)
#define CYCLES_LO REG(0x18)
#define CYCLES_HI REG(0x1c)
#define SYSTEM REG(0x20)
#define LEDS REG(0x24)
#define BEAT_WORD(i) REG(0x40 + 4u * (i))
#define BEAT_ADDRESS REG(0x60)
#define BEAT_CONTROL REG(0x64)
#define BEAT_STROBE REG(0x68)

#define BEAT_WRITE 1u
#define BEAT_READ 2u

/* DDR is 0x8000_0000-0xbfff_ffff; the beat port takes 30-bit offsets. */
#define DDR_BASE 0x80000000u

static inline uint64_t cycles(void) {
  uint32_t lo = CYCLES_LO;
  uint32_t hi = CYCLES_HI;
  return ((uint64_t)hi << 32) | lo;
}

static inline void uart_putc(uint8_t c) { UART_TX = c; }

static inline uint8_t uart_getc(void) {
  for (;;) {
    uint32_t value = UART_RX;
    if (value & 0x80000000u) return (uint8_t)value;
  }
}

static inline void uart_put_u32(uint32_t value) {
  for (int i = 0; i < 4; ++i) uart_putc((uint8_t)(value >> (8 * i)));
}

static inline uint32_t uart_get_u32(void) {
  uint32_t value = 0u;
  for (int i = 0; i < 4; ++i) value |= (uint32_t)uart_getc() << (8 * i);
  return value;
}

static inline void beat_write(uint32_t ddr_address, uint32_t strobe) {
  BEAT_ADDRESS = ddr_address & 0x3fffffffu;
  BEAT_STROBE = strobe;
  BEAT_CONTROL = BEAT_WRITE;
}

static inline void beat_read(uint32_t ddr_address) {
  BEAT_ADDRESS = ddr_address & 0x3fffffffu;
  BEAT_CONTROL = BEAT_READ;
}

static inline uint32_t crc32_update(uint32_t crc, uint8_t byte) {
  crc ^= byte;
  for (int bit = 0; bit < 8; ++bit)
    crc = (crc >> 1) ^ ((crc & 1u) ? 0xedb88320u : 0u);
  return crc;
}

#endif

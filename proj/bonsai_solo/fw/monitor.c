/* Boot monitor in BRAM 0x0000-0x0fff. Binary protocol over UART, all
   integers little-endian; every command answers with ACK first.
     'P'                      ping; replies ACK + "SOLO" + calib flag
     'W' addr len bytes...    write; replies ACK + CRC32 of the bytes
     'R' addr len             read; replies ACK + bytes
     'C' addr len             CRC32 of memory; replies ACK + crc
     'T' addr len seed        DDR pattern test; replies ACK + error count
     'J' addr                 call addr(); replies ACK, then the callee's
                              own output, then DONE when it returns
   Addresses below 0x8000_0000 are BRAM/MMIO; DDR accesses that are 32-byte
   aligned in address and length use whole 256-bit beats. */
#include "hw.h"

#define ACK 0x06u
#define DONE 0x04u
#define NAK 0x15u

static uint32_t crc_bytes(uint32_t crc, const volatile uint8_t *p,
                          uint32_t n) {
  for (uint32_t i = 0; i < n; ++i) crc = crc32_update(crc, p[i]);
  return crc;
}

static int beat_aligned(uint32_t address, uint32_t length) {
  return address >= DDR_BASE && ((address | length) & 31u) == 0u;
}

static void command_write(void) {
  uint32_t address = uart_get_u32();
  uint32_t length = uart_get_u32();
  uint32_t crc = 0xffffffffu;
  if (beat_aligned(address, length)) {
    BEAT_STROBE = 0xffffffffu;
    for (uint32_t offset = 0; offset < length; offset += 32u) {
      for (uint32_t word = 0; word < 8u; ++word) {
        uint32_t value = 0u;
        for (uint32_t byte = 0; byte < 4u; ++byte) {
          uint8_t c = uart_getc();
          crc = crc32_update(crc, c);
          value |= (uint32_t)c << (8u * byte);
        }
        BEAT_WORD(word) = value;
      }
      BEAT_ADDRESS = (address + offset) & 0x3fffffffu;
      BEAT_CONTROL = BEAT_WRITE;
    }
  } else {
    volatile uint8_t *p = (volatile uint8_t *)address;
    for (uint32_t i = 0; i < length; ++i) {
      uint8_t c = uart_getc();
      crc = crc32_update(crc, c);
      p[i] = c;
    }
  }
  uart_putc(ACK);
  uart_put_u32(~crc);
}

static void command_read(void) {
  uint32_t address = uart_get_u32();
  uint32_t length = uart_get_u32();
  uart_putc(ACK);
  if (beat_aligned(address, length)) {
    for (uint32_t offset = 0; offset < length; offset += 32u) {
      beat_read(address + offset);
      for (uint32_t word = 0; word < 8u; ++word) uart_put_u32(BEAT_WORD(word));
    }
  } else {
    const volatile uint8_t *p = (const volatile uint8_t *)address;
    for (uint32_t i = 0; i < length; ++i) uart_putc(p[i]);
  }
}

static void command_crc(void) {
  uint32_t address = uart_get_u32();
  uint32_t length = uart_get_u32();
  uint32_t crc = 0xffffffffu;
  if (beat_aligned(address, length)) {
    for (uint32_t offset = 0; offset < length; offset += 32u) {
      beat_read(address + offset);
      for (uint32_t word = 0; word < 8u; ++word) {
        uint32_t value = BEAT_WORD(word);
        for (uint32_t byte = 0; byte < 4u; ++byte)
          crc = crc32_update(crc, (uint8_t)(value >> (8u * byte)));
      }
    }
  } else {
    crc = crc_bytes(crc, (const volatile uint8_t *)address, length);
  }
  uart_putc(ACK);
  uart_put_u32(~crc);
}

/* xorshift32 pattern, one word per 4 bytes; the address is folded into the
   seed so aliasing between address bits is caught. */
static void command_test(void) {
  uint32_t address = uart_get_u32();
  uint32_t length = uart_get_u32();
  uint32_t seed = uart_get_u32();
  uint32_t errors = 0u;
  if (!beat_aligned(address, length)) {
    uart_putc(NAK);
    return;
  }
  BEAT_STROBE = 0xffffffffu;
  uint32_t state = seed | 1u;
  for (uint32_t offset = 0; offset < length; offset += 32u) {
    for (uint32_t word = 0; word < 8u; ++word) {
      state ^= state << 13; state ^= state >> 17; state ^= state << 5;
      BEAT_WORD(word) = state ^ (address + offset + 4u * word);
    }
    BEAT_ADDRESS = (address + offset) & 0x3fffffffu;
    BEAT_CONTROL = BEAT_WRITE;
  }
  state = seed | 1u;
  for (uint32_t offset = 0; offset < length; offset += 32u) {
    beat_read(address + offset);
    for (uint32_t word = 0; word < 8u; ++word) {
      state ^= state << 13; state ^= state >> 17; state ^= state << 5;
      if (BEAT_WORD(word) != (state ^ (address + offset + 4u * word)))
        ++errors;
    }
  }
  uart_putc(ACK);
  uart_put_u32(errors);
}

void monitor_main(void) {
  crc32_init_table();
  LEDS = 0x04u;
  for (;;) {
    uint8_t command = uart_getc();
    switch (command) {
    case 'P':
      uart_putc(ACK);
      uart_putc('S'); uart_putc('O'); uart_putc('L'); uart_putc('O');
      uart_putc((uint8_t)SYSTEM);
      break;
    case 'W': command_write(); break;
    case 'R': command_read(); break;
    case 'C': command_crc(); break;
    case 'T': command_test(); break;
    case 'J': {
      uint32_t address = uart_get_u32();
      uart_putc(ACK);
      ((void (*)(void))address)();
      uart_putc(DONE);
      break;
    }
    default:
      uart_putc(NAK);
      break;
    }
  }
}

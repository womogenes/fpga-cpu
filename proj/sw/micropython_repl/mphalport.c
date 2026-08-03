#include <generated/csr.h>
#include <generated/soc.h>
#include <runtime.h>
#include <stddef.h>
#include <stdint.h>
#include <text_console.h>

#include "py/mphal.h"
#include "py/runtime.h"
#include "py/stream.h"
#include "shared/runtime/interrupt_char.h"

#define UART_INPUT_CAPACITY 256u
#define UART_EVENT_RX (1u << 1)

static uint8_t uart_input[UART_INPUT_CAPACITY];
static uint8_t uart_input_head;
static uint8_t uart_input_tail;
static unsigned int uart_input_count;
static uint32_t random_state;

static uint64_t cycle_count(void) {
  uint32_t high_before;
  uint32_t low;
  uint32_t high_after;

  do {
    __asm__ volatile("csrr %0, mcycleh" : "=r"(high_before));
    __asm__ volatile("csrr %0, mcycle" : "=r"(low));
    __asm__ volatile("csrr %0, mcycleh" : "=r"(high_after));
  } while (high_before != high_after);

  return ((uint64_t)high_before << 32) | low;
}

static void queue_character(uint8_t character) {
  if (uart_input_count == UART_INPUT_CAPACITY)
    return;

  uart_input[uart_input_head++] = character;
  uart_input_count++;
}

static int dequeue_character(void) {
  if (uart_input_count == 0u)
    return -1;

  uint8_t character = uart_input[uart_input_tail++];
  uart_input_count--;
  return character;
}

static void poll_uart(void) {
  while (!uart_rxempty_read()) {
    uint8_t character = (uint8_t)uart_rxtx_read();

    /* This LiteX UART advances its RX FIFO when the RX event is cleared. */
    uart_ev_pending_write(UART_EVENT_RX);

    if (mp_interrupt_char >= 0 && character == (uint8_t)mp_interrupt_char)
      mp_sched_keyboard_interrupt();
    else
      queue_character(character);
  }
}

void mp_hal_event_poll_hook(void) {
  poll_uart();
  text_console_service();
}

uintptr_t mp_hal_stdio_poll(uintptr_t poll_flags) {
  uintptr_t result = 0;
  poll_uart();

  if ((poll_flags & MP_STREAM_POLL_RD) != 0u && uart_input_count != 0u)
    result |= MP_STREAM_POLL_RD;
  if ((poll_flags & MP_STREAM_POLL_WR) != 0u)
    result |= MP_STREAM_POLL_WR;
  return result;
}

int mp_hal_stdin_rx_chr(void) {
  for (;;) {
    poll_uart();
    int character = dequeue_character();
    if (character >= 0)
      return character;
    text_console_service();
    text_console_present();
  }
}

mp_uint_t mp_hal_stdout_tx_strn(const char* data, size_t length) {
  uart_write(data, (unsigned int)length);
  text_console_write(data, length);
  return length;
}

mp_uint_t mp_hal_ticks_ms(void) {
  return (mp_uint_t)(cycle_count() / (CONFIG_CLOCK_FREQUENCY / 1000u));
}

mp_uint_t mp_hal_ticks_us(void) {
  return (mp_uint_t)(cycle_count() / (CONFIG_CLOCK_FREQUENCY / 1000000u));
}

mp_uint_t mp_hal_ticks_cpu(void) {
  return (mp_uint_t)cycle_count();
}

uint64_t mp_hal_time_ns(void) {
  uint64_t cycles = cycle_count();
  uint64_t seconds = cycles / CONFIG_CLOCK_FREQUENCY;
  uint64_t remainder = cycles % CONFIG_CLOCK_FREQUENCY;
  return seconds * 1000000000ull +
         remainder * 1000000000ull / CONFIG_CLOCK_FREQUENCY;
}

void mp_hal_delay_ms(mp_uint_t milliseconds) {
  mp_uint_t start = mp_hal_ticks_ms();
  while ((mp_uint_t)(mp_hal_ticks_ms() - start) < milliseconds)
    mp_hal_event_poll_hook();
}

void mp_hal_delay_us(mp_uint_t microseconds) {
  mp_uint_t start = mp_hal_ticks_us();
  while ((mp_uint_t)(mp_hal_ticks_us() - start) < microseconds)
    mp_hal_event_poll_hook();
}

void mp_hal_get_random(size_t length, uint8_t* buffer) {
  if (random_state == 0u)
    random_state = (uint32_t)cycle_count() ^ 0x9e3779b9u;

  for (size_t i = 0; i < length; i++) {
    random_state = random_state * 1664525u + 1013904223u;
    buffer[i] = (uint8_t)(random_state >> 24);
  }
}

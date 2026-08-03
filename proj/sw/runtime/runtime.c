#include <stdint.h>

#include <generated/csr.h>
#include <generated/mem.h>
#include <generated/soc.h>

#include <runtime.h>

void uart_init(void) {
  /* The BIOS has already configured the UART divisor. */
}

static void uart_write(char value) {
  while (uart_txfull_read());
  uart_rxtx_write((uint32_t)value);
}

int puts(const char* string) {
  while (*string != '\0') uart_write(*string++);
  uart_write('\r');
  uart_write('\n');
  return 0;
}

void flush_l2_cache(void) {
#ifdef CONFIG_L2_SIZE
  volatile const uint32_t* main_ram =
      (volatile const uint32_t*)MAIN_RAM_BASE;

  /* Read twice the cache capacity to evict all current L2 lines. */
  for (unsigned int i = 0; i < 2u * CONFIG_L2_SIZE / 4u; i++)
    (void)main_ram[i];
#endif
}

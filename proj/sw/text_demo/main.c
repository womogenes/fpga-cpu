#include <generated/soc.h>
#include <generated/csr.h>
#include <runtime.h>
#include <stdarg.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <text_console.h>

#ifndef FONT_WIDTH
#define FONT_WIDTH 8
#endif

#ifndef FONT_HEIGHT
#define FONT_HEIGHT 16
#endif

#ifndef FONT_SCALE
#define FONT_SCALE 1
#endif

#ifndef DEMO_LINES_PER_SECOND
#define DEMO_LINES_PER_SECOND 60
#endif

static int console_printf(const char* format, ...) {
  char buffer[192];
  va_list arguments;

  va_start(arguments, format);
  int length = vsnprintf(buffer, sizeof(buffer), format, arguments);
  va_end(arguments);

  if (length < 0)
    return length;

  size_t displayed = (size_t)length;
  if (displayed >= sizeof(buffer))
    displayed = sizeof(buffer) - 1u;
  text_console_write(buffer, displayed);
  return length;
}

static void start_line_period(void) {
  timer0_en_write(0);
  timer0_load_write(CONFIG_CLOCK_FREQUENCY / DEMO_LINES_PER_SECOND);
  timer0_en_write(1);
}

static void wait_for_line_period(void) {
  uint32_t remaining;

  do {
    timer0_update_value_write(1);
    remaining = timer0_value_read();
  } while (remaining != 0u);
}

static int is_prime(int number) {
  if (number < 2)
    return 0;
  for (int divisor = 2; divisor <= number / divisor; divisor++) {
    if (number % divisor == 0)
      return 0;
  }
  return 1;
}

int main(void) {
  struct text_console_config console_config = {
      .framebuffer = (volatile uint16_t*)VIDEO_FRAMEBUFFER_BASE,
      .foreground = text_console_rgb565(224, 238, 255),
      .background = text_console_rgb565(0, 0, 0),
  };

  uart_init();
  puts("Drawing text...");
  text_console_init(&console_config);

  text_console_write("FPGA CPU\nTEXT DEMO\n\n", 20);
  console_printf("FRAMEBUFFER: 1280x720\n");
  console_printf("FONT CELL: %ux%u\n", FONT_WIDTH * FONT_SCALE,
                 FONT_HEIGHT * FONT_SCALE);
  console_printf("UNSIGNED: %u\n", 4294967295u);
  console_printf("SIGNED: %d\n", -12345);
  console_printf("HEX: 0x%08x\n", 0xdeadbeefu);
  text_console_present();

  puts("Text framebuffer ready.");
  timer0_reload_write(0);

  int number = 2;
  int index = 1;
  for (;;) {
    start_line_period();
    console_printf("%dth prime: %d\n", index, number);
    text_console_present();
    do {
      number++;
    } while (!is_prime(number));
    index++;
    wait_for_line_period();
  }
}

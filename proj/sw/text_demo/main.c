#include <generated/csr.h>
#include <generated/mem.h>
#include <runtime.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>

#define FRAMEBUFFER_WIDTH 1280u
#define FRAMEBUFFER_HEIGHT 720u
#ifndef FONT_WIDTH
#define FONT_WIDTH 32
#endif

#ifndef FONT_HEIGHT
#define FONT_HEIGHT 64
#endif

#define FONT_FIRST 32u
#define FONT_LAST 126u
#define FONT_BYTES_PER_ROW ((FONT_WIDTH + 7u) / 8u)
#define FONT_BYTES_PER_GLYPH (FONT_HEIGHT * FONT_BYTES_PER_ROW)
#define TEXT_CELL_WIDTH (FONT_WIDTH * FONT_SCALE)
#define TEXT_CELL_HEIGHT (FONT_HEIGHT * FONT_SCALE)
#define TEXT_COLUMNS (FRAMEBUFFER_WIDTH / TEXT_CELL_WIDTH)
#define TEXT_ROWS (FRAMEBUFFER_HEIGHT / TEXT_CELL_HEIGHT)
#define TEXT_CELL_COUNT (TEXT_COLUMNS * TEXT_ROWS)

#ifndef FONT_SCALE
#define FONT_SCALE 1
#endif

#ifndef DEMO_LINES_PER_SECOND
#define DEMO_LINES_PER_SECOND 60
#endif

#if FONT_SCALE < 1
#error "FONT_SCALE must be at least 1"
#endif

#if FONT_WIDTH * FONT_SCALE > 1280 || FONT_HEIGHT * FONT_SCALE > 720
#error "scaled font cell must fit inside the framebuffer"
#endif

#if DEMO_LINES_PER_SECOND < 1
#error "DEMO_LINES_PER_SECOND must be at least 1"
#endif

extern const uint8_t _binary_spleen_font_bin_start[];

static uint8_t text_cells[TEXT_CELL_COUNT] DDR_BSS;

struct text_screen {
  volatile uint16_t* framebuffer;
  uint8_t* cells;
  unsigned int cursor_x;
  unsigned int cursor_y;
  uint16_t foreground;
  uint16_t background;
};

static uint16_t rgb565(uint8_t red, uint8_t green, uint8_t blue) {
  return ((uint16_t)(red & 0xf8u) << 8) |
         ((uint16_t)(green & 0xfcu) << 3) |
         ((uint16_t)(blue & 0xf8u) >> 3);
}

static void clear_screen(volatile uint16_t* framebuffer, uint16_t color) {
  uint32_t pair = (uint32_t)color | ((uint32_t)color << 16);
  volatile uint32_t* pixels = (volatile uint32_t*)framebuffer;

  for (unsigned int i = 0; i < FRAMEBUFFER_WIDTH * FRAMEBUFFER_HEIGHT / 2u;
       i++)
    pixels[i] = pair;
}

static void draw_character(volatile uint16_t* framebuffer,
                           unsigned int origin_x, unsigned int origin_y,
                           unsigned char character, uint16_t foreground,
                           uint16_t background) {
  if (character < FONT_FIRST || character > FONT_LAST)
    character = '?';

  const uint8_t* glyph = _binary_spleen_font_bin_start +
                         (character - FONT_FIRST) * FONT_BYTES_PER_GLYPH;

  for (unsigned int source_y = 0; source_y < FONT_HEIGHT; source_y++) {
    unsigned int destination_y = origin_y + source_y * FONT_SCALE;
    if (destination_y >= FRAMEBUFFER_HEIGHT)
      break;

    for (unsigned int source_x = 0; source_x < FONT_WIDTH; source_x++) {
      uint8_t bits = glyph[source_y * FONT_BYTES_PER_ROW + source_x / 8u];
      uint16_t color = (bits & (0x80u >> (source_x & 7u)))
                           ? foreground
                           : background;

      unsigned int destination_x = origin_x + source_x * FONT_SCALE;
      for (unsigned int scale_y = 0; scale_y < FONT_SCALE; scale_y++) {
        unsigned int y = destination_y + scale_y;
        if (y >= FRAMEBUFFER_HEIGHT)
          break;
        for (unsigned int scale_x = 0; scale_x < FONT_SCALE; scale_x++) {
          unsigned int x = destination_x + scale_x;
          if (x < FRAMEBUFFER_WIDTH)
            framebuffer[y * FRAMEBUFFER_WIDTH + x] = color;
        }
      }
    }
  }
}

static void text_screen_scroll(struct text_screen* screen) {
  for (unsigned int row = 0; row + 1u < TEXT_ROWS; row++) {
    for (unsigned int column = 0; column < TEXT_COLUMNS; column++) {
      unsigned int destination = row * TEXT_COLUMNS + column;
      unsigned int source = destination + TEXT_COLUMNS;
      uint8_t character = screen->cells[source];

      if (screen->cells[destination] != character) {
        draw_character(screen->framebuffer,
                       column * TEXT_CELL_WIDTH,
                       row * TEXT_CELL_HEIGHT,
                       character,
                       screen->foreground,
                       screen->background);
        screen->cells[destination] = character;
      }
    }
  }

  unsigned int last_row = (TEXT_ROWS - 1u) * TEXT_COLUMNS;
  for (unsigned int column = 0; column < TEXT_COLUMNS; column++) {
    unsigned int cell = last_row + column;
    if (screen->cells[cell] != ' ') {
      draw_character(screen->framebuffer,
                     column * TEXT_CELL_WIDTH,
                     (TEXT_ROWS - 1u) * TEXT_CELL_HEIGHT,
                     ' ',
                     screen->foreground,
                     screen->background);
      screen->cells[cell] = ' ';
    }
  }
}

static void text_screen_newline(struct text_screen* screen) {
  unsigned int cell_height = TEXT_CELL_HEIGHT;

  screen->cursor_x = 0;
  if (screen->cursor_y + 2u * cell_height > FRAMEBUFFER_HEIGHT) {
    text_screen_scroll(screen);
    screen->cursor_y = (TEXT_ROWS - 1u) * cell_height;
  } else {
    screen->cursor_y += cell_height;
  }
}

static void text_screen_putchar(struct text_screen* screen, char character) {
  unsigned int cell_width = TEXT_CELL_WIDTH;
  unsigned int cell_height = TEXT_CELL_HEIGHT;

  if (character == '\n') {
    text_screen_newline(screen);
    return;
  }
  if (character == '\r') {
    screen->cursor_x = 0;
    return;
  }
  if (character == '\t') {
    unsigned int tab_width = 4u * cell_width;
    screen->cursor_x = ((screen->cursor_x / tab_width) + 1u) * tab_width;
    return;
  }
  if (screen->cursor_x + cell_width > FRAMEBUFFER_WIDTH)
    text_screen_newline(screen);
  if (screen->cursor_y + cell_height > FRAMEBUFFER_HEIGHT)
    return;

  unsigned char printable = (unsigned char)character;
  if (printable < FONT_FIRST || printable > FONT_LAST)
    printable = '?';

  draw_character(screen->framebuffer, screen->cursor_x, screen->cursor_y,
                 printable, screen->foreground,
                 screen->background);
  screen->cells[(screen->cursor_y / cell_height) * TEXT_COLUMNS +
                screen->cursor_x / cell_width] = printable;
  screen->cursor_x += cell_width;
}

static void text_screen_puts(struct text_screen* screen, const char* string) {
  while (*string != '\0')
    text_screen_putchar(screen, *string++);
}

static int screen_printf(struct text_screen* screen, const char* format, ...) {
  char buffer[160];
  va_list arguments;

  va_start(arguments, format);
  int length = vsnprintf(buffer, sizeof(buffer), format, arguments);
  va_end(arguments);

  if (length < 0)
    return length;

  text_screen_puts(screen, buffer);
  return length;
}

static void text_screen_present(struct text_screen* screen) {
  (void)screen;
  flush_cpu_dcache();
  flush_l2_cache();
}

static void start_demo_line_period(void) {
  timer0_en_write(0);
  timer0_load_write(CONFIG_CLOCK_FREQUENCY / DEMO_LINES_PER_SECOND);
  timer0_en_write(1);
}

static void wait_for_demo_line_period(void) {
  uint32_t remaining;

  do {
    timer0_update_value_write(1);
    remaining = timer0_value_read();
  } while (remaining != 0u);
}

int is_prime(int n) {
  for (int i = 2; i * i <= n; i++) {
    if (n % i == 0) return 0;
  }
  return 1;
}

int main(void) {
  volatile uint16_t* framebuffer =
      (volatile uint16_t*)VIDEO_FRAMEBUFFER_BASE;
  struct text_screen screen = {
      .framebuffer = framebuffer,
      .cells = text_cells,
      .cursor_x = 0,
      .cursor_y = 0,
      .foreground = rgb565(224, 238, 255),
      .background = rgb565(0, 0, 0),
  };

  uart_init();
  puts("Drawing text...");

  clear_screen(framebuffer, screen.background);
  for (unsigned int i = 0; i < TEXT_CELL_COUNT; i++)
    screen.cells[i] = ' ';

  video_framebuffer_dma_base_write(VIDEO_FRAMEBUFFER_BASE);
  video_framebuffer_dma_length_write(
      FRAMEBUFFER_WIDTH * FRAMEBUFFER_HEIGHT * sizeof(uint16_t));
  video_framebuffer_dma_loop_write(1);
  video_framebuffer_vtg_enable_write(1);

  printf("text_demo: %ux%u framebuffer, %ux%u font\n",
         FRAMEBUFFER_WIDTH, FRAMEBUFFER_HEIGHT,
         FONT_WIDTH * FONT_SCALE, FONT_HEIGHT * FONT_SCALE);

  text_screen_puts(&screen, "FPGA CPU\nTEXT DEMO\n\n");
  screen_printf(&screen, "FRAMEBUFFER: %ux%u\n",
                FRAMEBUFFER_WIDTH, FRAMEBUFFER_HEIGHT);
  screen_printf(&screen, "FONT CELL: %ux%u\n",
                FONT_WIDTH * FONT_SCALE, FONT_HEIGHT * FONT_SCALE);
  screen_printf(&screen, "UNSIGNED: %u\n", 4294967295u);
  screen_printf(&screen, "SIGNED: %d\n", -12345);
  screen_printf(&screen, "HEX: 0x%08x\n", 0xdeadbeefu);

  text_screen_present(&screen);
  video_framebuffer_dma_enable_write(1);

  puts("Text framebuffer ready.");

  timer0_reload_write(0);

  // int count = 0;
  // while (1) {
  // start_demo_line_period();
  // screen_printf(&screen, "Count: %08d\n", count);
  // text_screen_present(&screen);
  // count++;
  // wait_for_demo_line_period();
  // }

  int n = 2;
  int i = 1;
  while (1) {
    start_demo_line_period();
    screen_printf(&screen, "%dth prime: %d\n", i, n);
    n++;
    while (!is_prime(n)) n++;
    i++;
    wait_for_demo_line_period();
  }

  return 0;
}

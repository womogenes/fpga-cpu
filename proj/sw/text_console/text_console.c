#include <generated/csr.h>
#include <generated/soc.h>
#include <runtime.h>
#include <stddef.h>
#include <stdint.h>
#include <system.h>
#include <text_console.h>

#define FRAMEBUFFER_WIDTH 1280u
#define FRAMEBUFFER_HEIGHT 720u

#ifndef FONT_WIDTH
#define FONT_WIDTH 8
#endif

#ifndef FONT_HEIGHT
#define FONT_HEIGHT 16
#endif

#ifndef FONT_SCALE
#define FONT_SCALE 1
#endif

#ifndef TEXT_CONSOLE_MAX_FPS
#define TEXT_CONSOLE_MAX_FPS 60
#endif

#ifndef TEXT_CONSOLE_CURSOR_BLINK_HZ
#define TEXT_CONSOLE_CURSOR_BLINK_HZ 2
#endif

#define FONT_FIRST 32u
#define FONT_LAST 126u
#define FONT_BYTES_PER_ROW ((FONT_WIDTH + 7u) / 8u)
#define FONT_BYTES_PER_GLYPH (FONT_HEIGHT * FONT_BYTES_PER_ROW)
#define CELL_WIDTH (FONT_WIDTH * FONT_SCALE)
#define CELL_HEIGHT (FONT_HEIGHT * FONT_SCALE)
#define CONSOLE_COLUMNS (FRAMEBUFFER_WIDTH / CELL_WIDTH)
#define CONSOLE_ROWS (FRAMEBUFFER_HEIGHT / CELL_HEIGHT)
#define CONSOLE_CELL_COUNT (CONSOLE_COLUMNS * CONSOLE_ROWS)
#define PRESENT_CYCLES (CONFIG_CLOCK_FREQUENCY / TEXT_CONSOLE_MAX_FPS)
#define CURSOR_TOGGLE_CYCLES \
  (CONFIG_CLOCK_FREQUENCY / TEXT_CONSOLE_CURSOR_BLINK_HZ)

#if FONT_SCALE < 1
#error "FONT_SCALE must be at least 1"
#endif

#if CELL_WIDTH > FRAMEBUFFER_WIDTH || CELL_HEIGHT > FRAMEBUFFER_HEIGHT
#error "scaled font cell must fit inside the framebuffer"
#endif

#if TEXT_CONSOLE_MAX_FPS < 1
#error "TEXT_CONSOLE_MAX_FPS must be at least 1"
#endif

extern const uint8_t _binary_spleen_font_bin_start[];

enum escape_state {
  ESCAPE_NONE,
  ESCAPE_SEEN,
  ESCAPE_CSI,
};

struct text_console_state {
  volatile uint16_t* framebuffer;
  uint16_t foreground;
  uint16_t background;
  unsigned int cursor_column;
  unsigned int cursor_row;
  enum escape_state escape;
  unsigned int escape_parameter;
  uint32_t last_present_cycle;
  uint32_t last_cursor_cycle;
  uint8_t escape_has_parameter;
  uint8_t dirty;
  uint8_t initialized;
  uint8_t cursor_enabled;
  uint8_t cursor_drawn;
};

static uint8_t text_cells[CONSOLE_CELL_COUNT] DDR_BSS;
static struct text_console_state console;

static void draw_character(unsigned int column, unsigned int row,
                           unsigned char character);

static uint32_t cycle_count(void) {
  uint32_t cycles;
  __asm__ volatile("csrr %0, mcycle" : "=r"(cycles));
  return cycles;
}

static void cursor_position(unsigned int* column, unsigned int* row) {
  *column = console.cursor_column;
  *row = console.cursor_row;
  if (*column >= CONSOLE_COLUMNS) {
    *column = 0;
    if (*row + 1u < CONSOLE_ROWS)
      (*row)++;
  }
}

static void show_cursor(void) {
  if (!console.cursor_enabled || console.cursor_drawn)
    return;

  unsigned int column;
  unsigned int row;
  cursor_position(&column, &row);
  unsigned int origin_x = column * CELL_WIDTH;
  unsigned int origin_y = row * CELL_HEIGHT;

  /* An underline does not obscure the glyph and can be restored exactly. */
  for (unsigned int y = CELL_HEIGHT - 2u; y < CELL_HEIGHT; y++) {
    volatile uint16_t* pixel =
        console.framebuffer + (origin_y + y) * FRAMEBUFFER_WIDTH + origin_x;
    for (unsigned int x = 0; x < CELL_WIDTH; x++)
      pixel[x] = console.foreground;
  }

  console.cursor_drawn = 1;
  console.dirty = 1;
}

static void hide_cursor(void) {
  if (!console.cursor_drawn)
    return;

  unsigned int column;
  unsigned int row;
  cursor_position(&column, &row);
  draw_character(column, row, text_cells[row * CONSOLE_COLUMNS + column]);
  console.cursor_drawn = 0;
  console.dirty = 1;
}

uint16_t text_console_rgb565(uint8_t red, uint8_t green, uint8_t blue) {
  return ((uint16_t)(red & 0xf8u) << 8) |
         ((uint16_t)(green & 0xfcu) << 3) |
         ((uint16_t)(blue & 0xf8u) >> 3);
}

static void fill_pixels(volatile uint16_t* pixels, unsigned int pixel_count,
                        uint16_t color) {
  uint32_t pair = (uint32_t)color | ((uint32_t)color << 16);
  volatile uint32_t* pairs = (volatile uint32_t*)pixels;

  for (unsigned int i = 0; i < pixel_count / 2u; i++)
    pairs[i] = pair;

  if ((pixel_count & 1u) != 0u)
    pixels[pixel_count - 1u] = color;
}

static void draw_character(unsigned int column, unsigned int row,
                           unsigned char character) {
  if (character < FONT_FIRST || character > FONT_LAST)
    character = '?';

  const uint8_t* glyph = _binary_spleen_font_bin_start +
                         (character - FONT_FIRST) * FONT_BYTES_PER_GLYPH;
  unsigned int origin_x = column * CELL_WIDTH;
  unsigned int origin_y = row * CELL_HEIGHT;

  for (unsigned int source_y = 0; source_y < FONT_HEIGHT; source_y++) {
    for (unsigned int scale_y = 0; scale_y < FONT_SCALE; scale_y++) {
      unsigned int destination_y = origin_y + source_y * FONT_SCALE + scale_y;
      volatile uint16_t* destination =
          console.framebuffer + destination_y * FRAMEBUFFER_WIDTH + origin_x;

      for (unsigned int source_x = 0; source_x < FONT_WIDTH; source_x++) {
        uint8_t bits = glyph[source_y * FONT_BYTES_PER_ROW + source_x / 8u];
        uint16_t color = (bits & (0x80u >> (source_x & 7u)))
                             ? console.foreground
                             : console.background;
        for (unsigned int scale_x = 0; scale_x < FONT_SCALE; scale_x++)
          *destination++ = color;
      }
    }
  }
}

static void scroll_up(void) {
  unsigned int removed_pixels = CELL_HEIGHT * FRAMEBUFFER_WIDTH;
  unsigned int retained_pixels =
      (FRAMEBUFFER_HEIGHT - CELL_HEIGHT) * FRAMEBUFFER_WIDTH;
  volatile uint32_t* destination = (volatile uint32_t*)console.framebuffer;
  volatile uint32_t* source =
      (volatile uint32_t*)(console.framebuffer + removed_pixels);

  for (unsigned int i = 0; i < retained_pixels / 2u; i++)
    destination[i] = source[i];

  fill_pixels(console.framebuffer + retained_pixels, removed_pixels,
              console.background);

  for (unsigned int i = 0; i < CONSOLE_CELL_COUNT - CONSOLE_COLUMNS; i++)
    text_cells[i] = text_cells[i + CONSOLE_COLUMNS];
  for (unsigned int i = CONSOLE_CELL_COUNT - CONSOLE_COLUMNS;
       i < CONSOLE_CELL_COUNT; i++)
    text_cells[i] = ' ';

  console.dirty = 1;
}

static void newline(void) {
  console.cursor_column = 0;
  if (console.cursor_row + 1u < CONSOLE_ROWS) {
    console.cursor_row++;
  } else {
    scroll_up();
    console.cursor_row = CONSOLE_ROWS - 1u;
  }
}

static void cursor_back(unsigned int count) {
  while (count-- != 0u) {
    if (console.cursor_column != 0u) {
      console.cursor_column--;
    } else if (console.cursor_row != 0u) {
      console.cursor_row--;
      console.cursor_column = CONSOLE_COLUMNS - 1u;
    }
  }
}

static void erase_to_end_of_line(void) {
  unsigned int column = console.cursor_column;
  if (column >= CONSOLE_COLUMNS)
    return;

  for (; column < CONSOLE_COLUMNS; column++) {
    unsigned int cell = console.cursor_row * CONSOLE_COLUMNS + column;
    if (text_cells[cell] != ' ') {
      text_cells[cell] = ' ';
      draw_character(column, console.cursor_row, ' ');
      console.dirty = 1;
    }
  }
}

static void write_printable(unsigned char character) {
  if (console.cursor_column >= CONSOLE_COLUMNS)
    newline();

  if (character < FONT_FIRST || character > FONT_LAST)
    character = '?';

  unsigned int cell =
      console.cursor_row * CONSOLE_COLUMNS + console.cursor_column;
  text_cells[cell] = character;
  draw_character(console.cursor_column, console.cursor_row, character);
  console.cursor_column++;
  console.dirty = 1;
}

static void process_character(unsigned char character) {
  if (console.escape == ESCAPE_SEEN) {
    console.escape = character == '[' ? ESCAPE_CSI : ESCAPE_NONE;
    console.escape_parameter = 0;
    console.escape_has_parameter = 0;
    return;
  }

  if (console.escape == ESCAPE_CSI) {
    if (character >= '0' && character <= '9') {
      console.escape_parameter =
          console.escape_parameter * 10u + (unsigned int)(character - '0');
      console.escape_has_parameter = 1;
      return;
    }

    unsigned int parameter =
        console.escape_has_parameter ? console.escape_parameter : 1u;
    if (character == 'D')
      cursor_back(parameter);
    else if (character == 'K')
      erase_to_end_of_line();
    console.escape = ESCAPE_NONE;
    return;
  }

  if (character == 0x1bu) {
    console.escape = ESCAPE_SEEN;
  } else if (character == '\b' || character == 0x7fu) {
    cursor_back(1);
  } else if (character == '\r') {
    console.cursor_column = 0;
  } else if (character == '\n') {
    newline();
  } else if (character == '\t') {
    unsigned int spaces = 4u - (console.cursor_column & 3u);
    while (spaces-- != 0u)
      write_printable(' ');
  } else if (character >= FONT_FIRST) {
    write_printable(character);
  }
}

void text_console_init(const struct text_console_config* config) {
  if (console.initialized)
    return;

  console.framebuffer = config->framebuffer;
  console.foreground = config->foreground;
  console.background = config->background;
  console.cursor_column = 0;
  console.cursor_row = 0;
  console.escape = ESCAPE_NONE;
  console.escape_parameter = 0;
  console.escape_has_parameter = 0;
  console.cursor_enabled = 0;
  console.cursor_drawn = 0;
  console.dirty = 1;
  console.initialized = 1;

  fill_pixels(console.framebuffer, FRAMEBUFFER_WIDTH * FRAMEBUFFER_HEIGHT,
              console.background);
  for (unsigned int i = 0; i < CONSOLE_CELL_COUNT; i++)
    text_cells[i] = ' ';

  video_framebuffer_dma_base_write(VIDEO_FRAMEBUFFER_BASE);
  video_framebuffer_dma_length_write(
      FRAMEBUFFER_WIDTH * FRAMEBUFFER_HEIGHT * sizeof(uint16_t));
  video_framebuffer_dma_loop_write(1);
  video_framebuffer_vtg_enable_write(1);
  text_console_present();
  video_framebuffer_dma_enable_write(1);
}

void text_console_set_cursor_enabled(int enabled) {
  hide_cursor();
  console.cursor_enabled = enabled != 0;
  console.last_cursor_cycle = cycle_count();
  show_cursor();
  text_console_present();
}

void text_console_write(const char* data, size_t length) {
  hide_cursor();
  for (size_t i = 0; i < length; i++)
    process_character((unsigned char)data[i]);
  console.last_cursor_cycle = cycle_count();
  show_cursor();
  text_console_service();
}

void text_console_service(void) {
  uint32_t now = cycle_count();
  if (console.cursor_enabled &&
      (uint32_t)(now - console.last_cursor_cycle) >= CURSOR_TOGGLE_CYCLES) {
    if (console.cursor_drawn)
      hide_cursor();
    else
      show_cursor();
    console.last_cursor_cycle = now;
  }

  if (!console.dirty)
    return;

  if ((uint32_t)(now - console.last_present_cycle) >= PRESENT_CYCLES)
    text_console_present();
}

void text_console_present(void) {
  if (!console.dirty)
    return;

  flush_cpu_dcache();
  flush_l2_cache();
  console.last_present_cycle = cycle_count();
  console.dirty = 0;
}

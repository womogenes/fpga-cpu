#include <generated/csr.h>
#include <generated/mem.h>
#include <runtime.h>
#include <stdint.h>

#define FB_WIDTH 1280u
#define FB_HEIGHT 720u
#define FB_PIXELS (FB_WIDTH * FB_HEIGHT)
#define UPDATES_PER_SECOND 60u

#define COORD_X_BITS 11u
#define COORD_X_MASK ((1u << COORD_X_BITS) - 1u)
#define COORD_Y_STEP (1u << COORD_X_BITS)
#define NEXT_ALIVE_FLAG 0x80000000u
#define COORD_MASK (~NEXT_ALIVE_FLAG)
#define CANDIDATE_MARK 0x80u
#define NEIGHBOR_COUNT_MASK 0x0fu

#define PIXEL_DEAD 0x0000u
#define PIXEL_ALIVE 0xffffu

/* Current board state and combined candidate-mark/neighbor-count bytes. */
static uint8_t grid[FB_PIXELS] DDR_BSS;
static uint8_t candidate_state[FB_PIXELS] DDR_BSS;

/*
 * These are scratch arenas, so startup does not waste time clearing them.
 * Each list can hold every cell on the screen; sparse patterns use only their
 * short prefixes.
 */
static uint32_t live_cells[FB_PIXELS] DDR_NOINIT;
static uint32_t candidate_cells[FB_PIXELS] DDR_NOINIT;

/* Packed by rle_to_bin.py: little-endian width/height, then RLE text. */
extern const uint8_t _binary_initial_pattern_bin_start[];

static inline uint32_t pack_coord(unsigned int x, unsigned int y) {
  return (y << COORD_X_BITS) | x;
}

static inline unsigned int coord_x(uint32_t coord) {
  return coord & COORD_X_MASK;
}

static inline unsigned int coord_y(uint32_t coord) {
  return coord >> COORD_X_BITS;
}

static void clear_framebuffer(volatile uint16_t* framebuffer) {
  volatile uint32_t* words = (volatile uint32_t*)framebuffer;

  for (uint32_t i = 0; i < FB_PIXELS / 2u; i++)
    words[i] = 0;
}

static unsigned int load_rle(volatile uint16_t* framebuffer,
                             const char* rle,
                             unsigned int origin_x,
                             unsigned int origin_y) {
  unsigned int pattern_x = 0u;
  unsigned int pattern_y = 0u;
  unsigned int run = 0u;
  unsigned int live_count = 0u;

  while (*rle != '\0' && *rle != '!') {
    char token = *rle++;

    if (token >= '0' && token <= '9') {
      run = run * 10u + (unsigned int)(token - '0');
      continue;
    }

    unsigned int length = run == 0u ? 1u : run;
    run = 0u;

    if (token == 'b') {
      pattern_x += length;
    } else if (token == '$') {
      pattern_x = 0u;
      pattern_y += length;
    } else if (token == 'o') {
      for (unsigned int i = 0; i < length; i++, pattern_x++) {
        unsigned int x = origin_x + pattern_x;
        unsigned int y = origin_y + pattern_y;

        if (x >= FB_WIDTH || y >= FB_HEIGHT)
          continue;

        uint32_t index = y * FB_WIDTH + x;
        if (!grid[index]) {
          grid[index] = 1u;
          live_cells[live_count++] = pack_coord(x, y);
          framebuffer[index] = PIXEL_ALIVE;
        }
      }
    }
  }

  return live_count;
}

static inline void add_candidate(uint32_t coord,
                                 uint32_t index,
                                 unsigned int neighbor_increment,
                                 unsigned int* candidate_count) {
  uint8_t state = candidate_state[index];

  if (!(state & CANDIDATE_MARK)) {
    state = CANDIDATE_MARK;
    candidate_cells[(*candidate_count)++] = coord;
  }

  candidate_state[index] =
      (uint8_t)(state + neighbor_increment);
}

static unsigned int collect_candidates(unsigned int live_count) {
  unsigned int candidate_count = 0u;

  for (unsigned int i = 0; i < live_count; i++) {
    uint32_t coord = live_cells[i];
    unsigned int x = coord_x(coord);
    unsigned int y = coord_y(coord);
    uint32_t index = y * FB_WIDTH + x;

    if (x > 0u && x + 1u < FB_WIDTH && y > 0u && y + 1u < FB_HEIGHT) {
      add_candidate(coord - COORD_Y_STEP - 1u,
                    index - FB_WIDTH - 1u, 1u, &candidate_count);
      add_candidate(coord - COORD_Y_STEP,
                    index - FB_WIDTH, 1u, &candidate_count);
      add_candidate(coord - COORD_Y_STEP + 1u,
                    index - FB_WIDTH + 1u, 1u, &candidate_count);
      add_candidate(coord - 1u, index - 1u, 1u, &candidate_count);
      add_candidate(coord, index, 0u, &candidate_count);
      add_candidate(coord + 1u, index + 1u, 1u, &candidate_count);
      add_candidate(coord + COORD_Y_STEP - 1u,
                    index + FB_WIDTH - 1u, 1u, &candidate_count);
      add_candidate(coord + COORD_Y_STEP,
                    index + FB_WIDTH, 1u, &candidate_count);
      add_candidate(coord + COORD_Y_STEP + 1u,
                    index + FB_WIDTH + 1u, 1u, &candidate_count);
    } else {
      int min_dx = x == 0u ? 0 : -1;
      int max_dx = x + 1u == FB_WIDTH ? 0 : 1;
      int min_dy = y == 0u ? 0 : -1;
      int max_dy = y + 1u == FB_HEIGHT ? 0 : 1;

      for (int dy = min_dy; dy <= max_dy; dy++) {
        for (int dx = min_dx; dx <= max_dx; dx++) {
          unsigned int candidate_x = (unsigned int)((int)x + dx);
          unsigned int candidate_y = (unsigned int)((int)y + dy);
          uint32_t candidate_index = candidate_y * FB_WIDTH + candidate_x;
          add_candidate(pack_coord(candidate_x, candidate_y),
                        candidate_index,
                        dx != 0 || dy != 0,
                        &candidate_count);
        }
      }
    }
  }

  return candidate_count;
}

static unsigned int step_generation(volatile uint16_t* framebuffer,
                                    unsigned int live_count) {
  unsigned int candidate_count = collect_candidates(live_count);

  /* Evaluate against the unchanged current grid, recording results in bit 31. */
  for (unsigned int i = 0; i < candidate_count; i++) {
    uint32_t coord = candidate_cells[i];
    unsigned int x = coord_x(coord);
    unsigned int y = coord_y(coord);
    uint32_t index = y * FB_WIDTH + x;
    unsigned int neighbors =
        candidate_state[index] & NEIGHBOR_COUNT_MASK;
    unsigned int alive = neighbors == 3u ||
                         (grid[index] && neighbors == 2u);

    candidate_cells[i] = coord | (alive ? NEXT_ALIVE_FLAG : 0u);
  }

  unsigned int next_live_count = 0u;
  unsigned int framebuffer_changed = 0u;

  /* Apply simultaneously, compacting survivors and births into the live list. */
  for (unsigned int i = 0; i < candidate_count; i++) {
    uint32_t result = candidate_cells[i];
    uint32_t coord = result & COORD_MASK;
    unsigned int x = coord_x(coord);
    unsigned int y = coord_y(coord);
    uint32_t index = y * FB_WIDTH + x;
    unsigned int alive = result >> 31;

    candidate_state[index] = 0u;

    if (alive != grid[index]) {
      grid[index] = (uint8_t)alive;
      framebuffer[index] = alive ? PIXEL_ALIVE : PIXEL_DEAD;
      framebuffer_changed = 1u;
    }

    if (alive)
      live_cells[next_live_count++] = coord;
  }

  if (framebuffer_changed) {
    flush_cpu_dcache();
    flush_l2_cache();
  }

  return next_live_count;
}

static void start_update_period(void) {
  timer0_en_write(0);
  timer0_load_write(CONFIG_CLOCK_FREQUENCY / UPDATES_PER_SECOND);
  timer0_en_write(1);
}

static void wait_for_update_period(void) {
  uint32_t remaining;

  do {
    timer0_update_value_write(1);
    remaining = timer0_value_read();
  } while (remaining != 0u);
}

int main(void) {
  volatile uint16_t* framebuffer =
      (volatile uint16_t*)VIDEO_FRAMEBUFFER_BASE;

  uart_init();
  puts("Welcome to game of life!");

  clear_framebuffer(framebuffer);
  const uint8_t* pattern = _binary_initial_pattern_bin_start;
  unsigned int pattern_width =
      (unsigned int)pattern[0] | ((unsigned int)pattern[1] << 8);
  unsigned int pattern_height =
      (unsigned int)pattern[2] | ((unsigned int)pattern[3] << 8);
  unsigned int origin_x = pattern_width < FB_WIDTH
                              ? (FB_WIDTH - pattern_width) / 2u
                              : 0u;
  unsigned int origin_y = pattern_height < FB_HEIGHT
                              ? (FB_HEIGHT - pattern_height) / 2u
                              : 0u;
  unsigned int live_count = load_rle(
      framebuffer,
      (const char*)(pattern + 4u),
      origin_x,
      origin_y);

  flush_cpu_dcache();
  flush_l2_cache();
  video_framebuffer_vtg_enable_write(1);
  video_framebuffer_dma_enable_write(1);
  timer0_reload_write(0);

  while (live_count != 0u) {
    start_update_period();
    live_count = step_generation(framebuffer, live_count);
    wait_for_update_period();
  }

  puts("All cells are dead.");
  while (1);
}

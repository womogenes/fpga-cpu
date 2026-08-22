#include "video_hw.h"

#include <errno.h>
#include <fcntl.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <termios.h>
#include <time.h>
#include <unistd.h>

#define GRID_WIDTH (VIDEO_WIDTH / 2u)
#define GRID_HEIGHT (VIDEO_HEIGHT / 2u)
#define GRID_CELLS (GRID_WIDTH * GRID_HEIGHT)

static volatile sig_atomic_t stop_requested;

static void request_stop(int signal_number) {
  (void)signal_number;
  stop_requested = 1;
}

static unsigned cell_index(unsigned x, unsigned y) {
  return y * GRID_WIDTH + x;
}

static void set_cell(uint8_t *grid, int x, int y) {
  if (x >= 0 && y >= 0 && x < (int)GRID_WIDTH && y < (int)GRID_HEIGHT)
    grid[cell_index((unsigned)x, (unsigned)y)] = 1;
}

static unsigned population(const uint8_t *grid) {
  unsigned total = 0;
  for (unsigned i = 0; i < GRID_CELLS; ++i)
    total += grid[i] != 0;
  return total;
}

static void evolve(const uint8_t *current, uint8_t *next) {
  for (unsigned y = 0; y < GRID_HEIGHT; ++y) {
    for (unsigned x = 0; x < GRID_WIDTH; ++x) {
      unsigned neighbors = 0;
      for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
          if (dx == 0 && dy == 0)
            continue;
          int nx = (int)x + dx;
          int ny = (int)y + dy;
          if (nx >= 0 && ny >= 0 && nx < (int)GRID_WIDTH &&
              ny < (int)GRID_HEIGHT)
            neighbors += current[cell_index((unsigned)nx, (unsigned)ny)] != 0;
        }
      }
      int alive = current[cell_index(x, y)] != 0;
      next[cell_index(x, y)] =
          (uint8_t)(neighbors == 3u || (alive && neighbors == 2u));
    }
  }
}

static const char *load_preset(uint8_t *grid, unsigned preset) {
  static const int glider[][2] = {{1, 0}, {2, 1}, {0, 2}, {1, 2}, {2, 2}};
  static const int acorn[][2] = {
      {1, 0}, {3, 1}, {0, 2}, {1, 2}, {4, 2}, {5, 2}, {6, 2}};
  static const int gun[][2] = {
      {0, 4},  {0, 5},  {1, 4},  {1, 5},  {10, 4}, {10, 5}, {10, 6},
      {11, 3}, {11, 7}, {12, 2}, {12, 8}, {13, 2}, {13, 8}, {14, 5},
      {15, 3}, {15, 7}, {16, 4}, {16, 5}, {16, 6}, {17, 5}, {20, 2},
      {20, 3}, {20, 4}, {21, 2}, {21, 3}, {21, 4}, {22, 1}, {22, 5},
      {24, 0}, {24, 1}, {24, 5}, {24, 6}, {34, 2}, {34, 3}, {35, 2},
      {35, 3}};
  memset(grid, 0, GRID_CELLS);
  int cx = (int)GRID_WIDTH / 2;
  int cy = (int)GRID_HEIGHT / 2;
  if (preset == 0u) {
    for (unsigned i = 0; i < sizeof(glider) / sizeof(glider[0]); ++i)
      set_cell(grid, cx - 1 + glider[i][0], cy - 1 + glider[i][1]);
    return "glider";
  }
  if (preset == 1u) {
    for (unsigned i = 0; i < sizeof(acorn) / sizeof(acorn[0]); ++i)
      set_cell(grid, cx - 3 + acorn[i][0], cy - 1 + acorn[i][1]);
    return "acorn methuselah";
  }
  if (preset == 2u) {
    static const int arms[] = {-4, -3, -2, 2, 3, 4};
    for (unsigned i = 0; i < sizeof(arms) / sizeof(arms[0]); ++i) {
      int a = arms[i];
      set_cell(grid, cx + a, cy - 6);
      set_cell(grid, cx + a, cy - 1);
      set_cell(grid, cx + a, cy + 1);
      set_cell(grid, cx + a, cy + 6);
      set_cell(grid, cx - 6, cy + a);
      set_cell(grid, cx - 1, cy + a);
      set_cell(grid, cx + 1, cy + a);
      set_cell(grid, cx + 6, cy + a);
    }
    return "pulsar";
  }
  for (unsigned i = 0; i < sizeof(gun) / sizeof(gun[0]); ++i)
    set_cell(grid, cx - 18 + gun[i][0], cy - 4 + gun[i][1]);
  return "Gosper glider gun";
}

static void randomize(uint8_t *grid) {
  uint32_t state = (uint32_t)time(NULL) ^ 0x9e3779b9u;
  for (unsigned i = 0; i < GRID_CELLS; ++i) {
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    grid[i] = (uint8_t)((state & 7u) < 2u);
  }
}

static void render(uint8_t *frame, const uint8_t *grid, unsigned cursor_x,
                   unsigned cursor_y, int running) {
  uint8_t alive = video_rgb232(40, 225, 170);
  uint8_t dead = video_rgb232(5, 10, 18);
  uint8_t cursor = video_rgb232(running ? 255 : 255, running ? 90 : 210, 20);
  for (unsigned y = 0; y < GRID_HEIGHT; ++y) {
    for (unsigned x = 0; x < GRID_WIDTH; ++x) {
      uint8_t color = grid[cell_index(x, y)] ? alive : dead;
      if (x == cursor_x && y == cursor_y)
        color = cursor;
      unsigned px = x * 2u;
      unsigned py = y * 2u;
      frame[py * VIDEO_WIDTH + px] = color;
      frame[py * VIDEO_WIDTH + px + 1u] = color;
      frame[(py + 1u) * VIDEO_WIDTH + px] = color;
      frame[(py + 1u) * VIDEO_WIDTH + px + 1u] = color;
    }
  }
}

static void report(const char *preset, int running, unsigned generation,
                   const uint8_t *grid) {
  printf("\rLife: %-18s  %s  generation %-7u population %-5u",
         preset, running ? "running" : "paused ", generation,
         population(grid));
  fflush(stdout);
}

static int self_test(void) {
  uint8_t a[GRID_CELLS] = {0};
  uint8_t b[GRID_CELLS] = {0};
  unsigned cx = GRID_WIDTH / 2u;
  unsigned cy = GRID_HEIGHT / 2u;
  a[cell_index(cx - 1u, cy)] = 1;
  a[cell_index(cx, cy)] = 1;
  a[cell_index(cx + 1u, cy)] = 1;
  evolve(a, b);
  if (population(b) != 3u || !b[cell_index(cx, cy - 1u)] ||
      !b[cell_index(cx, cy)] || !b[cell_index(cx, cy + 1u)])
    return 1;
  evolve(b, a);
  if (population(a) != 3u || !a[cell_index(cx - 1u, cy)] ||
      !a[cell_index(cx, cy)] || !a[cell_index(cx + 1u, cy)])
    return 1;
  puts("Game of Life self-test passed (blinker period 2). ");
  return 0;
}

int main(int argc, char **argv) {
  if (argc == 2 && strcmp(argv[1], "--self-test") == 0)
    return self_test();
  if (argc != 1) {
    fprintf(stderr, "usage: life [--self-test]\n");
    return 2;
  }

  struct video_hw video = {0};
  if (video_hw_open(&video) != 0)
    return 1;
  uint8_t *frame = malloc(VIDEO_PIXELS);
  uint8_t *grid = calloc(GRID_CELLS, 1);
  uint8_t *next = calloc(GRID_CELLS, 1);
  if (frame == NULL || grid == NULL || next == NULL) {
    free(next);
    free(grid);
    free(frame);
    video_hw_close(&video);
    return 1;
  }

  struct termios saved;
  if (tcgetattr(STDIN_FILENO, &saved) != 0) {
    perror("life: tcgetattr");
    return 1;
  }
  struct termios raw = saved;
  cfmakeraw(&raw);
  if (tcsetattr(STDIN_FILENO, TCSANOW, &raw) != 0) {
    perror("life: tcsetattr");
    return 1;
  }
  int saved_flags = fcntl(STDIN_FILENO, F_GETFL, 0);
  (void)fcntl(STDIN_FILENO, F_SETFL, saved_flags | O_NONBLOCK);
  signal(SIGINT, request_stop);
  signal(SIGTERM, request_stop);

  unsigned preset_index = 0;
  const char *preset = load_preset(grid, preset_index);
  unsigned cursor_x = GRID_WIDTH / 2u;
  unsigned cursor_y = GRID_HEIGHT / 2u;
  unsigned generation = 0;
  unsigned delay_ms = 120;
  unsigned escape_state = 0;
  int running = 0;
  int dirty = 1;
  puts("Game of Life: arrows/WASD move, space edits, Enter runs/pauses,");
  puts("              n steps, p changes preset, c clears, r randomizes,");
  puts("              +/- changes speed, q quits.");

  while (!stop_requested) {
    unsigned char key;
    ssize_t count;
    while ((count = read(STDIN_FILENO, &key, 1)) == 1) {
      if (escape_state == 1u) {
        escape_state = key == '[' ? 2u : 0u;
        continue;
      }
      if (escape_state == 2u) {
        if (key == 'A' && cursor_y > 0u)
          --cursor_y;
        else if (key == 'B' && cursor_y + 1u < GRID_HEIGHT)
          ++cursor_y;
        else if (key == 'C' && cursor_x + 1u < GRID_WIDTH)
          ++cursor_x;
        else if (key == 'D' && cursor_x > 0u)
          --cursor_x;
        escape_state = 0u;
        dirty = 1;
        continue;
      }
      if (key == 27u) {
        escape_state = 1u;
      } else if (key == 'q' || key == 'Q') {
        stop_requested = 1;
      } else if ((key == 'w' || key == 'W') && cursor_y > 0u) {
        --cursor_y;
      } else if ((key == 's' || key == 'S') && cursor_y + 1u < GRID_HEIGHT) {
        ++cursor_y;
      } else if ((key == 'a' || key == 'A') && cursor_x > 0u) {
        --cursor_x;
      } else if ((key == 'd' || key == 'D') && cursor_x + 1u < GRID_WIDTH) {
        ++cursor_x;
      } else if (key == ' ') {
        unsigned index = cell_index(cursor_x, cursor_y);
        grid[index] ^= 1u;
        preset = "custom";
      } else if (key == '\r' || key == '\n') {
        running = !running;
      } else if (key == 'n' || key == 'N') {
        evolve(grid, next);
        uint8_t *swap = grid;
        grid = next;
        next = swap;
        ++generation;
        preset = "custom";
      } else if (key == 'p' || key == 'P') {
        preset_index = (preset_index + 1u) % 4u;
        preset = load_preset(grid, preset_index);
        generation = 0;
        running = 0;
      } else if (key == 'c' || key == 'C') {
        memset(grid, 0, GRID_CELLS);
        preset = "blank/custom";
        generation = 0;
        running = 0;
      } else if (key == 'r' || key == 'R') {
        randomize(grid);
        preset = "random";
        generation = 0;
        running = 0;
      } else if (key == '+' && delay_ms > 20u) {
        delay_ms -= 20u;
      } else if (key == '-' && delay_ms < 1000u) {
        delay_ms += 20u;
      }
      dirty = 1;
    }
    if (count < 0 && errno != EAGAIN && errno != EWOULDBLOCK)
      break;

    if (running) {
      evolve(grid, next);
      uint8_t *swap = grid;
      grid = next;
      next = swap;
      ++generation;
      dirty = 1;
    }
    if (dirty) {
      render(frame, grid, cursor_x, cursor_y, running);
      if (video_hw_submit(&video, frame) != 0)
        break;
      report(preset, running, generation, grid);
      dirty = 0;
    }
    struct timespec delay = {.tv_sec = delay_ms / 1000u,
                             .tv_nsec = (long)(delay_ms % 1000u) * 1000000l};
    nanosleep(&delay, NULL);
  }

  (void)fcntl(STDIN_FILENO, F_SETFL, saved_flags);
  (void)tcsetattr(STDIN_FILENO, TCSANOW, &saved);
  video_hw_terminal(&video);
  video_hw_close(&video);
  free(next);
  free(grid);
  free(frame);
  putchar('\n');
  return 0;
}

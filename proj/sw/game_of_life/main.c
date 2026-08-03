#include <generated/csr.h>
#include <generated/mem.h>
#include <runtime.h>
#include <stdint.h>

#define FB_WIDTH 1280u
#define FB_HEIGHT 720u

static uint16_t rgb565(uint8_t red, uint8_t green, uint8_t blue) {
  return ((uint16_t)(red & 0xf8) << 8) |
         ((uint16_t)(green & 0xfc) << 3) |
         ((uint16_t)(blue & 0xf8) >> 3);
}

// Allocate a new framebuffer ig lol
static uint8_t grid_next[FB_HEIGHT][FB_WIDTH] DDR_BSS;

int main(void) {
  // This is given by build/fpga_soc/software/include/generated/mem.h
  // Generated during `make gateware`
  volatile uint16_t* framebuffer = (volatile uint16_t*)VIDEO_FRAMEBUFFER_BASE;

  uart_init();
  puts("Welcome to game of life!");

  grid_next[FB_HEIGHT / 2 + 0][FB_WIDTH / 2 + 0] = 1;
  grid_next[FB_HEIGHT / 2 + 1][FB_WIDTH / 2 + 0] = 1;
  grid_next[FB_HEIGHT / 2 + 0][FB_WIDTH / 2 + 1] = 1;
  grid_next[FB_HEIGHT / 2 + 0][FB_WIDTH / 2 + 2] = 1;
  grid_next[FB_HEIGHT / 2 - 1][FB_WIDTH / 2 + 1] = 1;

  // Enable video drawing
  video_framebuffer_vtg_enable_write(1);
  video_framebuffer_dma_enable_write(1);

  // We need a loop ig
  while (1) {
    for (unsigned int y = 0; y < FB_HEIGHT; y++) {
      for (unsigned int x = 0; x < FB_WIDTH; x++) {
        // Count neighbors
        int nbr_cnt = 0;
        for (int dx = -1; dx <= 1; dx++) {
          for (int dy = -1; dy <= 1; dy++) {
            // Skip own cell
            if (dx == 0 && dy == 0) continue;

            int xx = x + dx;
            int yy = y + dy;

            // Skip if out of bounds
            if (xx < 0 || xx >= FB_WIDTH || yy < 0 || yy > FB_HEIGHT) continue;

            nbr_cnt += (grid_next[yy][xx] > 0);
          }
        }

        int cur = grid_next[y][x];

        int alive_next = (cur && (nbr_cnt == 2 || nbr_cnt == 3)) ||
                         (!cur && (nbr_cnt == 3));

        int color = 0xff;
        uint8_t red = alive_next ? color : 0x00;
        uint8_t green = alive_next ? color : 0x00;
        uint8_t blue = alive_next ? color : 0x00;

        framebuffer[y * FB_WIDTH + x] = rgb565(red, green, blue);
      }
    }

    // Maybe there is a more efficient C way to do this but alas
    for (unsigned int y = 0; y < FB_HEIGHT; y++) {
      for (unsigned int x = 0; x < FB_WIDTH; x++) {
        grid_next[y][x] = framebuffer[y * FB_WIDTH + x] > 0;
      }
    }

    flush_cpu_dcache();
    flush_l2_cache();
  }

  while (1);

  return 0;
}

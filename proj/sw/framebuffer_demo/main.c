#include <generated/csr.h>
#include <generated/mem.h>
#include <runtime.h>
#include <stdint.h>

#define FRAMEBUFFER_WIDTH 1280u
#define FRAMEBUFFER_HEIGHT 720u

static uint16_t rgb565(uint8_t red, uint8_t green, uint8_t blue) {
  return ((uint16_t)(red & 0xf8) << 8) |
         ((uint16_t)(green & 0xfc) << 3) |
         ((uint16_t)(blue & 0xf8) >> 3);
}

int main(void) {
  volatile uint16_t* framebuffer =
      (volatile uint16_t*)VIDEO_FRAMEBUFFER_BASE;

  uart_init();
  puts("Drawing 1280x720 RGB565 color bars...");

  for (unsigned int y = 0; y < FRAMEBUFFER_HEIGHT; y++) {
    for (unsigned int x = 0; x < FRAMEBUFFER_WIDTH; x++) {
      uint8_t red = (x < 640) ? 0xff : 0x00;
      uint8_t green = (x >= 320 && x < 960) ? 0xff : 0x00;
      uint8_t blue = (x >= 640) ? 0xff : 0x00;

      red = (uint8_t)(((uint32_t)red * y) / 719u);
      green = (uint8_t)(((uint32_t)green * y) / 719u);
      blue = (uint8_t)(((uint32_t)blue * y) / 719u);
      framebuffer[y * FRAMEBUFFER_WIDTH + x] =
          rgb565(red, green, blue);
    }
  }

  /* The video DMA bypasses the CPU data cache and LiteDRAM L2 cache. */
  flush_cpu_dcache();
  flush_l2_cache();

  video_framebuffer_vtg_enable_write(1);
  video_framebuffer_dma_enable_write(1);

  if (video_framebuffer_vtg_enable_read() == 1)
    puts("Video timing generator enabled.");
  else
    puts("ERROR: video timing generator did not enable.");

  if (video_framebuffer_dma_enable_read() == 1)
    puts("Framebuffer DMA enabled.");
  else
    puts("ERROR: framebuffer DMA did not enable.");

  uint32_t dma_offset = 0;
  for (unsigned int i = 0; i < 1000000u && dma_offset == 0; i++)
    dma_offset = video_framebuffer_dma_offset_read();

  if (dma_offset != 0)
    puts("Framebuffer DMA is advancing.");
  else
    puts("ERROR: framebuffer DMA is stalled.");

  puts("Framebuffer ready.");

  while (1);

  return 0;
}

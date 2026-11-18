#include <generated/csr.h>
#include <generated/mem.h>
#include <generated/soc.h>
#include <runtime.h>
#include <stdint.h>

#include "uzlib.h"

#define FRAMEBUFFER_WIDTH 1280u
#define FRAMEBUFFER_HEIGHT 720u
#define PIXEL_BLACK 0x0000u
#define PIXEL_WHITE 0xffffu
#define VIDEO_HEADER_SIZE 20u
#define COMPRESSED_HEADER_SIZE 8u
#define VIDEO_STORAGE_CAPACITY (16u * 1024u * 1024u)

extern const uint8_t _binary_video_bin_start[];
extern const uint8_t _binary_video_bin_end[];

static uint8_t frame_state[FRAMEBUFFER_WIDTH * FRAMEBUFFER_HEIGHT / 64u]
    DDR_BSS;
static uint8_t video_storage[VIDEO_STORAGE_CAPACITY] DDR_NOINIT;
static uzlib_uncomp_t video_decompressor DDR_BSS;

static uint16_t read_u16(const uint8_t* data) {
  return (uint16_t)data[0] | ((uint16_t)data[1] << 8);
}

static uint32_t read_u32(const uint8_t* data) {
  return (uint32_t)data[0] | ((uint32_t)data[1] << 8) |
         ((uint32_t)data[2] << 16) | ((uint32_t)data[3] << 24);
}

static int decompress_video(const uint8_t* compressed,
                            uint32_t compressed_size,
                            uint32_t expected_size) {
  if (expected_size > VIDEO_STORAGE_CAPACITY) return 0;

  video_decompressor.source = compressed;
  video_decompressor.source_limit = compressed + compressed_size;
  int window_bits;
  if (uzlib_parse_zlib_gzip_header(&video_decompressor, &window_bits) !=
      UZLIB_HEADER_ZLIB)
    return 0;

  uzlib_uncompress_init(&video_decompressor, 0, 0);
  video_decompressor.dest_start = video_storage;
  video_decompressor.dest = video_storage;
  video_decompressor.dest_limit = video_storage + VIDEO_STORAGE_CAPACITY;
  int result = uzlib_uncompress_chksum(&video_decompressor);
  return result == UZLIB_DONE &&
         (uint32_t)(video_decompressor.dest - video_storage) == expected_size;
}

static void clear_framebuffer(volatile uint16_t* framebuffer) {
  volatile uint32_t* words = (volatile uint32_t*)framebuffer;
  for (uint32_t index = 0;
       index < FRAMEBUFFER_WIDTH * FRAMEBUFFER_HEIGHT / 2u; index++)
    words[index] = 0u;
}

static void clear_frame_state(unsigned int packed_size) {
  for (unsigned int offset = 0; offset < packed_size; offset++)
    frame_state[offset] = 0u;
}

static void draw_source_pixel(volatile uint16_t* framebuffer,
                              unsigned int source_x,
                              unsigned int source_y,
                              unsigned int scale_x,
                              unsigned int scale_y,
                              uint16_t color) {
  uint32_t color_pair = (uint32_t)color | ((uint32_t)color << 16);
  unsigned int destination_x = source_x * scale_x;
  unsigned int destination_y = source_y * scale_y;

  for (unsigned int y = 0; y < scale_y; y++) {
    volatile uint16_t* row = framebuffer +
                             (destination_y + y) * FRAMEBUFFER_WIDTH +
                             destination_x;
    unsigned int x = 0u;
    if ((destination_x & 1u) != 0u && x < scale_x) row[x++] = color;
    for (; x + 1u < scale_x; x += 2u)
      *(volatile uint32_t*)(row + x) = color_pair;
    if (x < scale_x) row[x] = color;
  }
}

static void apply_delta_byte(volatile uint16_t* framebuffer,
                             unsigned int offset,
                             uint8_t delta,
                             unsigned int source_width,
                             unsigned int scale_x,
                             unsigned int scale_y) {
  if (delta == 0u) return;

  uint8_t current = frame_state[offset] ^ delta;
  frame_state[offset] = current;
  unsigned int row_bytes = source_width / 8u;
  unsigned int source_y = offset / row_bytes;
  unsigned int source_x = (offset % row_bytes) * 8u;

  for (unsigned int bit = 0; bit < 8u; bit++) {
    uint8_t mask = (uint8_t)(1u << bit);
    if (delta & mask) {
      draw_source_pixel(framebuffer, source_x + bit, source_y, scale_x,
                        scale_y, current & mask ? PIXEL_WHITE : PIXEL_BLACK);
    }
  }
}

static int decode_frame(volatile uint16_t* framebuffer,
                        const uint8_t* encoded,
                        uint32_t encoded_size,
                        unsigned int packed_size,
                        unsigned int source_width,
                        unsigned int scale_x,
                        unsigned int scale_y) {
  const uint8_t* cursor = encoded;
  const uint8_t* end = encoded + encoded_size;
  unsigned int output_offset = 0u;
  int changed = 0;

  while (cursor < end && output_offset < packed_size) {
    uint8_t command = *cursor++;
    unsigned int length = (command & 0x7fu) + 1u;
    if (length > packed_size - output_offset) return -1;

    if (command & 0x80u) {
      if (cursor == end) return -1;
      uint8_t value = *cursor++;
      if (value != 0u) changed = 1;
      for (unsigned int index = 0; index < length; index++)
        apply_delta_byte(framebuffer, output_offset++, value, source_width,
                         scale_x, scale_y);
    } else {
      if ((unsigned int)(end - cursor) < length) return -1;
      for (unsigned int index = 0; index < length; index++) {
        uint8_t value = *cursor++;
        if (value != 0u) changed = 1;
        apply_delta_byte(framebuffer, output_offset++, value, source_width,
                         scale_x, scale_y);
      }
    }
  }

  return cursor == end && output_offset == packed_size ? changed : -1;
}

static void start_frame_period(unsigned int frames_per_second) {
  timer0_en_write(0);
  timer0_load_write(CONFIG_CLOCK_FREQUENCY / frames_per_second);
  timer0_en_write(1);
}

static void wait_for_frame_period(void) {
  uint32_t remaining;
  do {
    timer0_update_value_write(1);
    remaining = timer0_value_read();
  } while (remaining != 0u);
}

int main(void) {
  volatile uint16_t* framebuffer =
      (volatile uint16_t*)VIDEO_FRAMEBUFFER_BASE;
  const uint8_t* video = _binary_video_bin_start;
  uint32_t video_size =
      (uint32_t)(_binary_video_bin_end - _binary_video_bin_start);

  uart_init();
  puts("Bad Apple video player");

  if (video_size < COMPRESSED_HEADER_SIZE || video[0] != 'M' ||
      video[1] != 'V' || video[2] != 'Z' || video[3] != '1') {
    puts("ERROR: invalid MVZ1 video header.");
    while (1);
  }

  uint32_t uncompressed_size = read_u32(video + 4u);
  if (!decompress_video(video + COMPRESSED_HEADER_SIZE,
                        video_size - COMPRESSED_HEADER_SIZE,
                        uncompressed_size)) {
    puts("ERROR: video decompression failed.");
    while (1);
  }
  video = video_storage;
  video_size = uncompressed_size;

  if (video_size < VIDEO_HEADER_SIZE || video[0] != 'M' ||
      video[1] != 'V' || video[2] != 'F' || video[3] != '1') {
    puts("ERROR: invalid MVF1 video header.");
    while (1);
  }

  unsigned int source_width = read_u16(video + 4u);
  unsigned int source_height = read_u16(video + 6u);
  unsigned int frames_per_second = read_u16(video + 8u);
  uint32_t frame_count = read_u32(video + 12u);
  unsigned int packed_size = read_u32(video + 16u);

  if (source_width == 0u || source_height == 0u ||
      frames_per_second == 0u || frame_count == 0u ||
      source_width % 8u != 0u ||
      FRAMEBUFFER_WIDTH % source_width != 0u ||
      FRAMEBUFFER_HEIGHT % source_height != 0u ||
      packed_size != source_width / 8u * source_height ||
      packed_size > sizeof(frame_state)) {
    puts("ERROR: unsupported MVF1 video geometry.");
    while (1);
  }

  unsigned int scale_x = FRAMEBUFFER_WIDTH / source_width;
  unsigned int scale_y = FRAMEBUFFER_HEIGHT / source_height;
  timer0_reload_write(0);
  clear_framebuffer(framebuffer);
  flush_cpu_dcache();
  flush_l2_cache();
  video_framebuffer_vtg_enable_write(1);
  video_framebuffer_dma_enable_write(1);

  puts("Playing monochrome video.");

  for (;;) {
    const uint8_t* cursor = video + VIDEO_HEADER_SIZE;
    const uint8_t* end = video + video_size;
    clear_frame_state(packed_size);
    clear_framebuffer(framebuffer);
    flush_cpu_dcache();
    flush_l2_cache();

    for (uint32_t frame = 0; frame < frame_count; frame++) {
      if ((unsigned int)(end - cursor) < 4u) {
        puts("ERROR: truncated MVF1 frame table.");
        while (1);
      }

      uint32_t encoded_size = read_u32(cursor);
      cursor += 4u;
      if ((uint32_t)(end - cursor) < encoded_size) {
        puts("ERROR: truncated MVF1 frame.");
        while (1);
      }

      start_frame_period(frames_per_second);
      int changed = decode_frame(framebuffer, cursor, encoded_size,
                                 packed_size, source_width, scale_x, scale_y);
      if (changed < 0) {
        puts("ERROR: corrupt MVF1 frame.");
        while (1);
      }
      if (changed) {
        flush_cpu_dcache();
        flush_l2_cache();
      }
      wait_for_frame_period();
      cursor += encoded_size;
    }
  }
}

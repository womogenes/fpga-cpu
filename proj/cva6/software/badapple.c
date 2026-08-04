#include "video_hw.h"

#include <errno.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <zlib.h>

static volatile sig_atomic_t stop_requested;

static void request_stop(int signal_number) {
  (void)signal_number;
  stop_requested = 1;
}

static uint16_t read_u16(const uint8_t *data) {
  return (uint16_t)data[0] | (uint16_t)data[1] << 8;
}

static uint32_t read_u32(const uint8_t *data) {
  return (uint32_t)data[0] | (uint32_t)data[1] << 8 | (uint32_t)data[2] << 16 | (uint32_t)data[3] << 24;
}

static uint8_t *read_file(const char *path, size_t *size) {
  FILE *input = fopen(path, "rb");
  if (input == NULL) {
    fprintf(stderr, "badapple: cannot open %s: %s\n", path, strerror(errno));
    return NULL;
  }
  if (fseek(input, 0, SEEK_END) != 0) {
    fclose(input);
    return NULL;
  }
  long length = ftell(input);
  rewind(input);
  if (length <= 0) {
    fclose(input);
    return NULL;
  }
  uint8_t *data = malloc((size_t)length);
  if (data == NULL || fread(data, 1, (size_t)length, input) != (size_t)length) {
    free(data);
    fclose(input);
    return NULL;
  }
  fclose(input);
  *size = (size_t)length;
  return data;
}

static int decode_packbits(uint8_t *output, size_t output_size, const uint8_t *input, size_t input_size) {
  size_t output_offset = 0;
  size_t input_offset = 0;
  while (input_offset < input_size && output_offset < output_size) {
    uint8_t control = input[input_offset++];
    size_t length = (control & 0x7f) + 1u;
    if (length > output_size - output_offset)
      return 1;
    if (control & 0x80) {
      if (input_offset >= input_size)
        return 1;
      memset(output + output_offset, input[input_offset++], length);
    } else {
      if (length > input_size - input_offset)
        return 1;
      memcpy(output + output_offset, input + input_offset, length);
      input_offset += length;
    }
    output_offset += length;
  }
  return output_offset != output_size || input_offset != input_size;
}

static void unpack_frame(uint8_t output[VIDEO_PIXELS], const uint8_t *packed, uint16_t width, uint16_t height) {
  for (unsigned y = 0; y < height; y++) {
    for (unsigned x = 0; x < width; x++) {
      unsigned packed_index = y * (width / 8u) + x / 8u;
      output[y * VIDEO_WIDTH + x] = (packed[packed_index] & (1u << (x & 7u))) ? 0x7f : 0;
    }
  }
}

static void sleep_until(const struct timespec *started, uint64_t frame_number, unsigned fps) {
  uint64_t nanoseconds = frame_number * 1000000000ull / fps;
  struct timespec deadline = *started;
  deadline.tv_sec += (time_t)(nanoseconds / 1000000000ull);
  deadline.tv_nsec += (long)(nanoseconds % 1000000000ull);
  if (deadline.tv_nsec >= 1000000000l) {
    deadline.tv_sec++;
    deadline.tv_nsec -= 1000000000l;
  }
  clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &deadline, NULL);
}

static double elapsed_seconds(const struct timespec *start, const struct timespec *end) {
  return (double)(end->tv_sec - start->tv_sec) + (double)(end->tv_nsec - start->tv_nsec) / 1000000000.0;
}

int main(int argc, char **argv) {
  const char *path = argc > 1 ? argv[1] : "/root/media/badapple.mvz";
  size_t compressed_size;
  uint8_t *compressed = read_file(path, &compressed_size);
  if (compressed == NULL)
    return 1;
  if (compressed_size < 8 || memcmp(compressed, "MVZ1", 4) != 0) {
    fprintf(stderr, "badapple: %s is not an MVZ1 video\n", path);
    free(compressed);
    return 1;
  }

  uLongf video_size = read_u32(compressed + 4);
  uint8_t *video_data = malloc(video_size);
  if (video_data == NULL || uncompress(video_data, &video_size, compressed + 8, compressed_size - 8) != Z_OK) {
    fprintf(stderr, "badapple: could not decompress video\n");
    free(video_data);
    free(compressed);
    return 1;
  }
  free(compressed);

  if (video_size < 20 || memcmp(video_data, "MVF1", 4) != 0) {
    fprintf(stderr, "badapple: corrupt MVF1 stream\n");
    free(video_data);
    return 1;
  }
  uint16_t width = read_u16(video_data + 4);
  uint16_t height = read_u16(video_data + 6);
  uint16_t fps = read_u16(video_data + 8);
  uint32_t frame_count = read_u32(video_data + 12);
  uint32_t packed_size = read_u32(video_data + 16);
  if (width != VIDEO_WIDTH || height != VIDEO_HEIGHT || fps == 0 || packed_size != VIDEO_PIXELS / 8u) {
    fprintf(stderr, "badapple: unsupported video geometry\n");
    free(video_data);
    return 1;
  }

  uint8_t *packed = calloc(1, packed_size);
  uint8_t *delta = malloc(packed_size);
  uint8_t *frame = malloc(VIDEO_PIXELS);
  if (packed == NULL || delta == NULL || frame == NULL) {
    free(frame);
    free(delta);
    free(packed);
    free(video_data);
    return 1;
  }

  struct video_hw display = {0};
  if (video_hw_open(&display) != 0) {
    free(frame);
    free(delta);
    free(packed);
    free(video_data);
    return 1;
  }
  signal(SIGINT, request_stop);
  signal(SIGTERM, request_stop);

  const uint8_t *cursor = video_data + 20;
  const uint8_t *end = video_data + video_size;
  struct timespec started;
  clock_gettime(CLOCK_MONOTONIC, &started);
  uint32_t displayed = 0;
  int status = 0;

  while (displayed < frame_count && !stop_requested) {
    if ((size_t)(end - cursor) < 4u) {
      status = 1;
      break;
    }
    uint32_t encoded_size = read_u32(cursor);
    cursor += 4;
    if ((size_t)(end - cursor) < encoded_size || decode_packbits(delta, packed_size, cursor, encoded_size) != 0) {
      status = 1;
      break;
    }
    cursor += encoded_size;
    for (uint32_t index = 0; index < packed_size; index++)
      packed[index] ^= delta[index];
    unpack_frame(frame, packed, width, height);
    if (video_hw_submit(&display, frame) != 0) {
      status = 1;
      break;
    }
    displayed++;
    sleep_until(&started, displayed, fps);
  }

  struct timespec finished;
  clock_gettime(CLOCK_MONOTONIC, &finished);

  video_hw_terminal(&display);
  video_hw_close(&display);
  free(frame);
  free(delta);
  free(packed);
  free(video_data);
  if (status != 0)
    fprintf(stderr, "badapple: corrupt or incomplete video after frame %u\n", displayed);
  else {
    double seconds = elapsed_seconds(&started, &finished);
    printf("Played %u frames in %.2f seconds (%.2f fps, %u fps target).\n", displayed, seconds, seconds > 0 ? displayed / seconds : 0.0, fps);
  }
  return status;
}

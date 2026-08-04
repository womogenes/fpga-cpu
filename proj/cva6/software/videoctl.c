#include <errno.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "video_hw.h"

static volatile sig_atomic_t stop_requested;

static void request_stop(int signal_number) {
  (void)signal_number;
  stop_requested = 1;
}

static void draw_bars(uint8_t *frame, unsigned phase) {
  static const uint8_t colors[] = {
      0x7f, 0x7c, 0x63, 0x60, 0x1f, 0x1c, 0x03, 0x00,
  };
  for (unsigned y = 0; y < VIDEO_HEIGHT; y++) {
    for (unsigned x = 0; x < VIDEO_WIDTH; x++) {
      unsigned shifted_x = (x + phase) % VIDEO_WIDTH;
      uint8_t color = colors[(shifted_x * 8) / VIDEO_WIDTH];
      if (((x + phase) / 8 + y / 8) & 1)
        color ^= video_rgb232(32, 32, 32);
      frame[y * VIDEO_WIDTH + x] = color;
    }
  }
}

static int play_file(struct video_hw *video, const char *path, uint8_t *frame) {
  FILE *input = fopen(path, "rb");
  if (input == NULL) {
    fprintf(stderr, "videoctl: cannot open %s: %s\n", path, strerror(errno));
    return 1;
  }

  while (!stop_requested) {
    size_t received = fread(frame, 1, VIDEO_PIXELS, input);
    if (received == VIDEO_PIXELS) {
      if (video_hw_submit(video, frame) != 0) {
        fclose(input);
        return 1;
      }
      continue;
    }
    if (ferror(input)) {
      fprintf(stderr, "videoctl: cannot read %s: %s\n", path, strerror(errno));
      fclose(input);
      return 1;
    }
    break;
  }

  fclose(input);
  return 0;
}

static void usage(const char *program) {
  fprintf(stderr, "Usage: %s bars | RAW_RGB232 | tty\n", program);
}

int main(int argc, char **argv) {
  if (argc != 2) {
    usage(argv[0]);
    return 2;
  }

  struct video_hw video = {0};
  if (video_hw_open(&video) != 0)
    return 1;

  if (strcmp(argv[1], "tty") == 0) {
    int status = video_hw_terminal(&video);
    video_hw_close(&video);
    return status;
  }

  signal(SIGINT, request_stop);
  signal(SIGTERM, request_stop);

  uint8_t *frame = malloc(VIDEO_PIXELS);
  if (frame == NULL) {
    fprintf(stderr, "videoctl: cannot allocate frame buffer\n");
    video_hw_close(&video);
    return 1;
  }

  int status = 0;
  if (strcmp(argv[1], "bars") == 0) {
    for (unsigned phase = 0; !stop_requested; phase++) {
      draw_bars(frame, phase);
      if (video_hw_submit(&video, frame) != 0) {
        status = 1;
        break;
      }
    }
  } else {
    status = play_file(&video, argv[1], frame);
  }

  video_hw_terminal(&video);
  free(frame);
  video_hw_close(&video);
  return status;
}

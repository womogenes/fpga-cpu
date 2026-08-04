#include "video_hw.h"

#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define MAX_ITERATIONS 36
#define REFINEMENT_PASSES 64u
#define ZOOM_NUMERATOR 995
#define ZOOM_DENOMINATOR 1000

static volatile sig_atomic_t stop_requested;

static void request_stop(int signal_number) {
  (void)signal_number;
  stop_requested = 1;
}

static uint8_t palette(unsigned iteration) {
  if (iteration == MAX_ITERATIONS)
    return 0;
  unsigned red = (iteration * 19) & 255;
  unsigned green = (iteration * 7) & 255;
  unsigned blue = (iteration * 29) & 255;
  return video_rgb232(red, green, blue);
}

static uint8_t sample(float cr, float ci) {
  float zr = 0.0f;
  float zi = 0.0f;
  unsigned iteration = 0;
  while (iteration < MAX_ITERATIONS) {
    float zr2 = zr * zr;
    float zi2 = zi * zi;
    if (zr2 + zi2 > 4.0f)
      break;
    zi = 2.0f * zr * zi + ci;
    zr = zr2 - zi2 + cr;
    iteration++;
  }
  return palette(iteration);
}

static void refine(uint8_t frame[VIDEO_PIXELS], float span, unsigned pass, unsigned passes) {
  const float center_real = -0.7436439f;
  const float center_imag = 0.1318259f;
  const float vertical_span = span * VIDEO_HEIGHT / VIDEO_WIDTH;
  const float real_step = (2.0f * span) / VIDEO_WIDTH;
  const float imag_step = (2.0f * vertical_span) / VIDEO_HEIGHT;
  const float first_real = center_real - span;
  const float first_imag = center_imag - vertical_span;

  for (unsigned index = pass; index < VIDEO_PIXELS; index += passes) {
    unsigned y = index / VIDEO_WIDTH;
    unsigned x = index - y * VIDEO_WIDTH;
    float ci = first_imag + y * imag_step;
    frame[index] = sample(first_real + x * real_step, ci);
  }
}

static void render_preview(uint8_t frame[VIDEO_PIXELS], float span) {
  const float center_real = -0.7436439f;
  const float center_imag = 0.1318259f;
  const float vertical_span = span * VIDEO_HEIGHT / VIDEO_WIDTH;
  const float real_step = (2.0f * span) / VIDEO_WIDTH;
  const float imag_step = (2.0f * vertical_span) / VIDEO_HEIGHT;
  const float first_real = center_real - span;
  const float first_imag = center_imag - vertical_span;
  const unsigned block_size = 8;

  for (unsigned block_y = 0; block_y < VIDEO_HEIGHT; block_y += block_size) {
    for (unsigned block_x = 0; block_x < VIDEO_WIDTH; block_x += block_size) {
      unsigned sample_x = block_x + block_size / 2;
      unsigned sample_y = block_y + block_size / 2;
      if (sample_x >= VIDEO_WIDTH)
        sample_x = VIDEO_WIDTH - 1;
      if (sample_y >= VIDEO_HEIGHT)
        sample_y = VIDEO_HEIGHT - 1;
      uint8_t color = sample(first_real + sample_x * real_step, first_imag + sample_y * imag_step);
      unsigned end_x = block_x + block_size < VIDEO_WIDTH ? block_x + block_size : VIDEO_WIDTH;
      unsigned end_y = block_y + block_size < VIDEO_HEIGHT ? block_y + block_size : VIDEO_HEIGHT;
      for (unsigned y = block_y; y < end_y; y++)
        for (unsigned x = block_x; x < end_x; x++)
          frame[y * VIDEO_WIDTH + x] = color;
    }
  }
}

static void resample_zoom(uint8_t destination[VIDEO_PIXELS], const uint8_t source[VIDEO_PIXELS]) {
  unsigned x_map[VIDEO_WIDTH];
  unsigned y_map[VIDEO_HEIGHT];
  const int center_x = VIDEO_WIDTH / 2;
  const int center_y = VIDEO_HEIGHT / 2;
  for (int x = 0; x < (int)VIDEO_WIDTH; x++)
    x_map[x] = (unsigned)(center_x + (x - center_x) * ZOOM_NUMERATOR / ZOOM_DENOMINATOR);
  for (int y = 0; y < (int)VIDEO_HEIGHT; y++)
    y_map[y] = (unsigned)(center_y + (y - center_y) * ZOOM_NUMERATOR / ZOOM_DENOMINATOR);
  for (unsigned y = 0; y < VIDEO_HEIGHT; y++) {
    const uint8_t *source_row = source + y_map[y] * VIDEO_WIDTH;
    uint8_t *destination_row = destination + y * VIDEO_WIDTH;
    for (unsigned x = 0; x < VIDEO_WIDTH; x++)
      destination_row[x] = source_row[x_map[x]];
  }
}

static double elapsed_seconds(const struct timespec *start, const struct timespec *end) {
  return (double)(end->tv_sec - start->tv_sec) + (double)(end->tv_nsec - start->tv_nsec) / 1000000000.0;
}

int main(void) {
  struct video_hw video = {0};
  if (video_hw_open(&video) != 0)
    return 1;

  uint8_t *frame = malloc(VIDEO_PIXELS);
  uint8_t *scratch = malloc(VIDEO_PIXELS);
  if (frame == NULL || scratch == NULL) {
    free(scratch);
    free(frame);
    video_hw_close(&video);
    return 1;
  }

  signal(SIGINT, request_stop);
  signal(SIGTERM, request_stop);
  struct timespec started;
  clock_gettime(CLOCK_MONOTONIC, &started);
  unsigned frames = 0;
  float span = 1.6f;
  int status = 0;

  render_preview(frame, span);

  while (!stop_requested) {
    if (video_hw_submit(&video, frame) != 0) {
      status = 1;
      break;
    }
    frames++;
    span *= (float)ZOOM_NUMERATOR / ZOOM_DENOMINATOR;
    if (span < 0.000002f) {
      span = 1.6f;
      refine(frame, span, 0, 1);
    } else {
      resample_zoom(scratch, frame);
      refine(scratch, span, frames % REFINEMENT_PASSES, REFINEMENT_PASSES);
      uint8_t *swap = frame;
      frame = scratch;
      scratch = swap;
    }
  }

  struct timespec finished;
  clock_gettime(CLOCK_MONOTONIC, &finished);
  video_hw_terminal(&video);
  video_hw_close(&video);
  free(scratch);
  free(frame);
  double seconds = elapsed_seconds(&started, &finished);
  if (seconds > 0)
    printf("Rendered %u frames in %.2f seconds (%.2f fps).\n", frames, seconds, frames / seconds);
  return status;
}

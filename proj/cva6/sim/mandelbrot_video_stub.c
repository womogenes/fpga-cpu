#include "video_hw.h"

#include <signal.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

static unsigned submitted;
static uint8_t previous[VIDEO_PIXELS];

int video_hw_open(struct video_hw *video) {
  video->gpio = NULL;
  video->toggle = 0u;
  return 0;
}

void video_hw_close(struct video_hw *video) {
  (void)video;
}

int video_hw_submit(struct video_hw *video,
                    const uint8_t frame[VIDEO_PIXELS]) {
  (void)video;
  unsigned histogram[128] = {0};
  unsigned changed = 0u;
  for (unsigned i = 0; i < VIDEO_PIXELS; ++i) {
    if (frame[i] >= 128u)
      return 1;
    ++histogram[frame[i]];
    if (submitted > 0u && frame[i] != previous[i])
      ++changed;
  }
  unsigned colors = 0u;
  for (unsigned i = 0; i < 128u; ++i)
    colors += histogram[i] != 0u;
  if (colors < 8u || (submitted > 0u && changed < 8u))
    return 1;
  memcpy(previous, frame, sizeof(previous));
  if (++submitted == 3u)
    (void)raise(SIGINT);
  return 0;
}

int video_hw_terminal(struct video_hw *video) {
  (void)video;
  if (submitted != 3u)
    return 1;
  puts("mandelbrot simulation passed");
  return 0;
}

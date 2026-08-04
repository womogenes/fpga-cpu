#ifndef CVA6_VIDEO_HW_H
#define CVA6_VIDEO_HW_H

#include <stdint.h>

#define VIDEO_WIDTH 160u
#define VIDEO_HEIGHT 90u
#define VIDEO_PIXELS (VIDEO_WIDTH * VIDEO_HEIGHT)

struct video_hw {
  volatile uint32_t *gpio;
  uint32_t toggle;
};

int video_hw_open(struct video_hw *video);
void video_hw_close(struct video_hw *video);
int video_hw_submit(struct video_hw *video, const uint8_t frame[VIDEO_PIXELS]);
int video_hw_terminal(struct video_hw *video);

static inline uint8_t video_rgb232(unsigned red, unsigned green, unsigned blue) {
  return (uint8_t)(((red >> 6) << 5) | ((green >> 5) << 2) | (blue >> 6));
}

#endif

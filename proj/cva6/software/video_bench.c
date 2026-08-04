#include "video_hw.h"

#include <stdint.h>
#include <stdio.h>
#include <time.h>

#define BENCHMARK_FRAMES 60u

static double elapsed_seconds(const struct timespec *start, const struct timespec *end) {
  return (double)(end->tv_sec - start->tv_sec) + (double)(end->tv_nsec - start->tv_nsec) / 1000000000.0;
}

int main(void) {
  struct video_hw video = {0};
  static uint8_t frame[VIDEO_PIXELS];
  for (unsigned index = 0; index < VIDEO_PIXELS; index++)
    frame[index] = (uint8_t)(index & 0x7fu);
  if (video_hw_open(&video) != 0)
    return 1;

  struct timespec started;
  struct timespec finished;
  clock_gettime(CLOCK_MONOTONIC, &started);
  unsigned completed = 0;
  while (completed < BENCHMARK_FRAMES) {
    if (video_hw_submit(&video, frame) != 0)
      break;
    completed++;
  }
  clock_gettime(CLOCK_MONOTONIC, &finished);
  video_hw_terminal(&video);
  video_hw_close(&video);

  double seconds = elapsed_seconds(&started, &finished);
  if (completed != 0)
    printf("Submitted %u frames in %.3f seconds (%.2f fps, %.3f ms/frame).\n", completed, seconds, completed / seconds, 1000.0 * seconds / completed);
  else
    printf("Submitted no complete frames in %.3f seconds.\n", seconds);
  return completed == BENCHMARK_FRAMES ? 0 : 1;
}

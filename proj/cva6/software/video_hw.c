#include "video_hw.h"

#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

#define GPIO_BASE 0x40000000ul
#define GPIO_MAP_SIZE 4096ul
#define GPIO_DATA_OFFSET 0
#define GPIO_TRI_OFFSET 1
#define GPIO_DATA2_OFFSET 2
#define VIDEO_ACK_MASK 0x20u
#define VIDEO_READY_MASK 0x40u

static int wait_ack(struct video_hw *video) {
  uint32_t expected = video->toggle == 0 ? 0 : VIDEO_ACK_MASK;
  for (unsigned attempt = 0; attempt < 1000000; attempt++) {
    if ((video->gpio[GPIO_DATA2_OFFSET] & VIDEO_ACK_MASK) == expected)
      return 0;
  }
  fprintf(stderr, "video: timed out waiting for command acknowledgement\n");
  return 1;
}

static int write_packet(struct video_hw *video, uint32_t packet) {
  video->toggle ^= 0x80000000u;
  video->gpio[GPIO_DATA_OFFSET] = video->toggle | packet;
  return wait_ack(video);
}

static void fence_io(void) {
  __asm__ volatile("fence iorw, iorw" ::: "memory");
}

static int write_command(struct video_hw *video, uint8_t command) {
  if (write_packet(video, 0x10000000u | command) != 0)
    return 1;
  fence_io();
  return 0;
}

static int wait_ready(struct video_hw *video) {
  for (unsigned attempt = 0; attempt < 10000; attempt++) {
    if (video->gpio[GPIO_DATA2_OFFSET] & VIDEO_READY_MASK)
      return 0;
    usleep(100);
  }
  fprintf(stderr, "video: timed out waiting for display hardware\n");
  return 1;
}

int video_hw_open(struct video_hw *video) {
  int memory = open("/dev/mem", O_RDWR | O_SYNC);
  if (memory < 0) {
    fprintf(stderr, "video: cannot open /dev/mem: %s\n", strerror(errno));
    return 1;
  }

  video->gpio = mmap(NULL, GPIO_MAP_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, memory, GPIO_BASE);
  close(memory);
  if (video->gpio == MAP_FAILED) {
    fprintf(stderr, "video: cannot map GPIO: %s\n", strerror(errno));
    video->gpio = NULL;
    return 1;
  }

  video->gpio[GPIO_TRI_OFFSET] = 0;
  fence_io();
  video->toggle = (video->gpio[GPIO_DATA2_OFFSET] & VIDEO_ACK_MASK) != 0
                      ? 0x80000000u
                      : 0;
  if (write_command(video, 0) != 0) {
    video_hw_close(video);
    return 1;
  }
  return 0;
}

void video_hw_close(struct video_hw *video) {
  if (video->gpio != NULL)
    munmap((void *)video->gpio, GPIO_MAP_SIZE);
  video->gpio = NULL;
}

int video_hw_submit(struct video_hw *video, const uint8_t frame[VIDEO_PIXELS]) {
  if (wait_ready(video) != 0)
    return 1;
  if (write_command(video, 1) != 0)
    return 1;
  for (unsigned index = 0; index < VIDEO_PIXELS; index += 4) {
    uint32_t packet = frame[index] | (uint32_t)frame[index + 1] << 7 | (uint32_t)frame[index + 2] << 14 | (uint32_t)frame[index + 3] << 21;
    if (write_packet(video, packet) != 0)
      return 1;
  }
  fence_io();
  return wait_ready(video);
}

int video_hw_terminal(struct video_hw *video) {
  return write_command(video, 0);
}

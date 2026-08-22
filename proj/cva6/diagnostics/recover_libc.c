#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <string.h>
#include <sys/mount.h>
#include <sys/stat.h>
#include <unistd.h>

static int copy_file(const char *source, const char *destination) {
  char buffer[16384];
  int input = open(source, O_RDONLY);
  if (input < 0) return -1;
  int output = open(destination, O_WRONLY | O_TRUNC);
  if (output < 0) return -1;
  for (;;) {
    ssize_t count = read(input, buffer, sizeof(buffer));
    if (count < 0) return -1;
    if (count == 0) break;
    char *cursor = buffer;
    while (count > 0) {
      ssize_t written = write(output, cursor, (size_t)count);
      if (written <= 0) return -1;
      cursor += written;
      count -= written;
    }
  }
  if (fsync(output) != 0) return -1;
  close(output);
  close(input);
  return 0;
}

int main(void) {
  mkdir("/dev", 0755);
  mkdir("/mnt", 0755);
  if (mount("devtmpfs", "/dev", "devtmpfs", 0, NULL) != 0 && errno != EBUSY) {
    dprintf(2, "RECOVERY_ERROR devtmpfs: %s\n", strerror(errno));
    return 1;
  }
  if (mount("/dev/mmcblk0p3", "/mnt", "ext4", 0, NULL) != 0) {
    dprintf(2, "RECOVERY_ERROR mount: %s\n", strerror(errno));
    return 1;
  }
  if (copy_file("/libc.so.6", "/mnt/lib/libc.so.6") != 0) {
    dprintf(2, "RECOVERY_ERROR copy: %s\n", strerror(errno));
    return 1;
  }
  sync();
  puts("RECOVERY_OK");
  fflush(stdout);
  for (;;) pause();
}

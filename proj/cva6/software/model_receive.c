#define _DEFAULT_SOURCE

#include <errno.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <termios.h>
#include <unistd.h>

enum { BLOCK_SIZE = 4096 };

static struct termios saved_termios;
static int restore_pending;

static void restore_terminal(void) {
  if (restore_pending != 0)
    (void)tcsetattr(STDIN_FILENO, TCSANOW, &saved_termios);
}

static void fail(const char *message) {
  fprintf(stderr, "model-receive: %s: %s\n", message, strerror(errno));
  exit(EXIT_FAILURE);
}

static void write_all(int fd, const void *buffer, size_t size) {
  const uint8_t *bytes = buffer;
  while (size != 0) {
    ssize_t written = write(fd, bytes, size);
    if (written < 0) {
      if (errno == EINTR)
        continue;
      fail("write");
    }
    bytes += (size_t)written;
    size -= (size_t)written;
  }
}

static void read_all(int fd, void *buffer, size_t size) {
  uint8_t *bytes = buffer;
  while (size != 0) {
    ssize_t received = read(fd, bytes, size);
    if (received < 0) {
      if (errno == EINTR)
        continue;
      fail("read");
    }
    if (received == 0) {
      errno = EIO;
      fail("unexpected end of UART input");
    }
    bytes += (size_t)received;
    size -= (size_t)received;
  }
}

int main(int argc, char **argv) {
  if (argc != 3) {
    fprintf(stderr, "usage: model-receive OUTPUT SIZE\n");
    return EXIT_FAILURE;
  }

  char *end = NULL;
  errno = 0;
  unsigned long long parsed_size = strtoull(argv[2], &end, 0);
  if (errno != 0 || end == argv[2] || *end != '\0' || parsed_size > UINT32_MAX) {
    fprintf(stderr, "model-receive: invalid size: %s\n", argv[2]);
    return EXIT_FAILURE;
  }

  int output = open(argv[1], O_WRONLY | O_CREAT | O_TRUNC, 0644);
  if (output < 0)
    fail("open output");

  if (tcgetattr(STDIN_FILENO, &saved_termios) != 0)
    fail("tcgetattr");
  if (atexit(restore_terminal) != 0)
    fail("atexit");
  struct termios raw = saved_termios;
  cfmakeraw(&raw);
  raw.c_cc[VMIN] = 0;
  raw.c_cc[VTIME] = 50;
  if (tcsetattr(STDIN_FILENO, TCSANOW, &raw) != 0)
    fail("tcsetattr raw");
  restore_pending = 1;

  write_all(STDOUT_FILENO, "MODEL_READY\n", 12);
  uint8_t block[BLOCK_SIZE];
  uint32_t checksum = 2166136261u;
  uint32_t remaining = (uint32_t)parsed_size;
  while (remaining != 0) {
    size_t count = remaining < sizeof(block) ? remaining : sizeof(block);
    read_all(STDIN_FILENO, block, count);
    write_all(output, block, count);
    for (size_t i = 0; i < count; i++)
      checksum = (checksum ^ block[i]) * 16777619u;
    remaining -= (uint32_t)count;
    write_all(STDOUT_FILENO, "A", 1);
  }

  if (fsync(output) != 0 || close(output) != 0)
    fail("finish output");
  write_all(STDOUT_FILENO, &checksum, sizeof(checksum));
  if (tcsetattr(STDIN_FILENO, TCSANOW, &saved_termios) != 0)
    fail("restore terminal");
  restore_pending = 0;
  printf("\nMODEL_DONE %08x\n", checksum);
  return EXIT_SUCCESS;
}

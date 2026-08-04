#define _FILE_OFFSET_BITS 64

#include <errno.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

enum {
  MODEL_PHYS = 0xb0000000u,
  MAX_MODEL_SIZE = 224u * 1024u * 1024u,
  COPY_CHUNK = 1024u * 1024u,
};

static void fail(const char *message) {
  fprintf(stderr, "model-export: %s: %s\n", message, strerror(errno));
  exit(EXIT_FAILURE);
}

static void write_all(int fd, const uint8_t *data, size_t size) {
  while (size != 0) {
    ssize_t written = write(fd, data, size);
    if (written < 0) {
      if (errno == EINTR)
        continue;
      fail("write");
    }
    data += (size_t)written;
    size -= (size_t)written;
  }
}

int main(int argc, char **argv) {
  if (argc < 2 || argc > 3) {
    fprintf(stderr, "usage: model-export SIZE [OUTPUT]\n");
    return EXIT_FAILURE;
  }

  char *end = NULL;
  errno = 0;
  unsigned long parsed = strtoul(argv[1], &end, 0);
  if (errno != 0 || end == argv[1] || *end != '\0' || parsed == 0 || parsed > MAX_MODEL_SIZE) {
    fprintf(stderr, "model-export: invalid model size: %s\n", argv[1]);
    return EXIT_FAILURE;
  }
  size_t size = (size_t)parsed;
  const char *path = argc == 3 ? argv[2] : "/root/models/LFM2.5-350M-Q4_0.gguf";

  int memory = open("/dev/mem", O_RDONLY | O_SYNC);
  if (memory < 0)
    fail("open /dev/mem");
  const uint8_t *model = mmap(NULL, size, PROT_READ, MAP_SHARED, memory, MODEL_PHYS);
  if (model == MAP_FAILED)
    fail("map reserved model memory");
  if (size < 4 || memcmp(model, "GGUF", 4) != 0) {
    fprintf(stderr, "model-export: no GGUF model is staged in reserved DDR\n");
    return 2;
  }

  int output = open(path, O_WRONLY | O_CREAT | O_TRUNC, 0644);
  if (output < 0)
    fail("open output");
  size_t copied = 0;
  while (copied < size) {
    size_t count = size - copied;
    if (count > COPY_CHUNK)
      count = COPY_CHUNK;
    write_all(output, model + copied, count);
    copied += count;
  }
  if (fsync(output) != 0 || close(output) != 0)
    fail("finish output");
  if (munmap((void *)model, size) != 0 || close(memory) != 0)
    fail("unmap reserved model memory");
  printf("Exported %zu bytes from reserved DDR to %s\n", size, path);
  return EXIT_SUCCESS;
}

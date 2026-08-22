#define _FILE_OFFSET_BITS 64

#include <errno.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#include "bonsai_model_manifest.h"

static void fail(const char *message) {
  perror(message);
  exit(1);
}

static void write_all(int output, const uint8_t *buffer, size_t size) {
  size_t sent = 0u;
  while (sent < size) {
    ssize_t count = write(output, buffer + sent, size - sent);
    if (count <= 0) fail("write");
    sent += (size_t)count;
  }
}

static void read_all(int input, uint8_t *buffer, size_t size) {
  size_t received = 0u;
  while (received < size) {
    ssize_t count = read(input, buffer + received, size - received);
    if (count < 0) fail("read");
    if (count == 0) {
      errno = EIO;
      fail("unexpected end of model");
    }
    received += (size_t)count;
  }
}

static void repack_tensor(uint8_t *buffer, uint64_t size) {
  enum { GROUP_BYTES = 72 };
  if (size % GROUP_BYTES != 0u) {
    errno = EINVAL;
    fail("unaligned packed tensor");
  }
  for (size_t offset = 0; offset < size; offset += GROUP_BYTES) {
      /* Four ordinary Q1_0 blocks occupy nine aligned 64-bit words. Shuffle
         their scales to the first word and retain the four 16-byte bitmaps.
         Expressing this as nine word operations avoids hundreds of unaligned
         byte loads/stores per group on scalar CVA6. */
      uint64_t *group = (uint64_t *)(buffer + offset);
      uint64_t q0 = group[0];
      uint64_t q1 = group[1];
      uint64_t q2 = group[2];
      uint64_t q3 = group[3];
      uint64_t q4 = group[4];
      uint64_t q5 = group[5];
      uint64_t q6 = group[6];
      uint64_t q7 = group[7];
      uint64_t q8 = group[8];
      group[0] = (q0 & UINT64_C(0x000000000000ffff)) |
                 ((q2 & UINT64_C(0x00000000ffff0000))) |
                 ((q4 & UINT64_C(0x0000ffff00000000))) |
                 ((q6 & UINT64_C(0xffff000000000000)));
      group[1] = (q0 >> 16) | (q1 << 48);
      group[2] = (q1 >> 16) | (q2 << 48);
      group[3] = (q2 >> 32) | (q3 << 32);
      group[4] = (q3 >> 32) | (q4 << 32);
      group[5] = (q4 >> 48) | (q5 << 16);
      group[6] = (q5 >> 48) | (q6 << 16);
      group[7] = q7;
      group[8] = q8;
  }
}

int main(int argc, char **argv) {
  if (argc != 3) {
    fprintf(stderr, "usage: %s INPUT.gguf OUTPUT.gguf\n", argv[0]);
    return 2;
  }
  int input = open(argv[1], O_RDONLY);
  if (input < 0) fail("open input");
  uint8_t *model = aligned_alloc(sizeof(uint64_t), BONSAI_MODEL_FILE_SIZE);
  if (model == NULL) fail("aligned_alloc model");
  fprintf(stderr, "reading source model\n");
  read_all(input, model, BONSAI_MODEL_FILE_SIZE);
  close(input);

  for (size_t tensor_index = 0; tensor_index < BONSAI_TENSOR_COUNT;
       ++tensor_index) {
    const struct bonsai_tensor_spec *tensor = &bonsai_tensors[tensor_index];
    uint64_t row_bytes = (uint64_t)(tensor->ne0 / 128u) * 18u;
    uint64_t tensor_bytes = row_bytes * tensor->ne1;
    if (tensor->offset + tensor_bytes > BONSAI_MODEL_FILE_SIZE) {
      errno = EINVAL;
      fail("tensor outside model");
    }
    repack_tensor(model + tensor->offset, tensor_bytes);
    fprintf(stderr, "repacked %zu/%zu tensors\r", tensor_index + 1u,
            (size_t)BONSAI_TENSOR_COUNT);
  }
  memcpy(model, "BGUF", 4u);

  int output = open(argv[2], O_RDWR | O_CREAT | O_TRUNC, 0644);
  if (output < 0) fail("open output");
  int allocation = posix_fallocate(output, 0, (off_t)BONSAI_MODEL_FILE_SIZE);
  if (allocation != 0) {
    errno = allocation;
    fail("preallocate output");
  }
  fprintf(stderr, "\nwriting packed model\n");
  write_all(output, model, BONSAI_MODEL_FILE_SIZE);
  if (fsync(output) != 0) fail("fsync");
  fprintf(stderr, "\ncreated %s\n", argv[2]);
  close(output);
  free(model);
  return 0;
}

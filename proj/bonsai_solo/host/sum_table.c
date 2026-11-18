/* Precompute the layer-0 RMS sum of squares for every vocabulary token, in
   the CVA6 runtime's exact integer order, so the controller does a table
   lookup instead of 2,048 soft-float adds per token. */
#include <stdio.h>
#include <stdlib.h>

#include "../fw/bonsai_math.h"

#define EMBEDDING_OFFSET 5945120u
#define ROW_BYTES 288u
#define VOCAB 151669u

int main(int argc, char **argv) {
  if (argc != 3) {
    fprintf(stderr, "usage: %s MODEL_COMPACT.bin SUM_TABLE.bin\n", argv[0]);
    return 2;
  }
  FILE *model = fopen(argv[1], "rb");
  if (model == NULL || fseek(model, EMBEDDING_OFFSET, SEEK_SET) != 0) {
    perror("model");
    return 1;
  }
  static uint8_t rows[VOCAB][ROW_BYTES];
  if (fread(rows, ROW_BYTES, VOCAB, model) != VOCAB) {
    perror("read embedding");
    return 1;
  }
  static uint32_t sums[VOCAB];
  for (uint32_t token = 0; token < VOCAB; ++token)
    sums[token] = embedding_sum_squares_bits(rows[token]);
  FILE *out = fopen(argv[2], "wb");
  if (out == NULL || fwrite(sums, 4, VOCAB, out) != VOCAB) {
    perror("write");
    return 1;
  }
  fclose(out);
  return 0;
}

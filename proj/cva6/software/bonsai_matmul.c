#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "ggml.h"
#include "ggml-cpu-impl.h"

int bonsai_accelerate_f32_matvec(const void *weights, uint32_t columns,
                                 uint32_t rows, const float *input,
                                 float *output);
int bonsai_accelerate_f32_matvec_submit(const void *weights, uint32_t columns,
                                        uint32_t rows, const float *input);
void bonsai_accelerate_read_rows(float *output, uint32_t first,
                                 uint32_t count);
void bonsai_accelerate_release(void);

void ggml_compute_forward_mul_mat(
    const struct ggml_compute_params *params, struct ggml_tensor *dst);

/* Every ggml worker enters the wrapped node. Hart zero computes the complete
   matrix through the FPGA while the other workers wait at the existing ggml
   thread-pool barrier. If the hardware path declines the operation, all
   workers leave the barrier together and run the normal partitioned kernel.
   This keeps FPGA matvecs enabled when llama.cpp uses multiple CPU workers for
   attention, normalization, and the remaining scalar graph operations.

   For the single-vector decode path with multiple workers, the raw Q30
   results stay in the uncached scratch region after the submit and every
   worker converts a slice of the output rows. With the private data caches
   disabled each conversion load is a full DDR round trip, so spreading the
   loads across harts overlaps their latency. */
static int whole_matrix_accelerated;

void bonsai_forward_mul_mat_hook(
    const struct ggml_compute_params *params, struct ggml_tensor *dst) {
  static int decline_reported;
  const struct ggml_tensor *weights = dst->src[0];
  const struct ggml_tensor *input = dst->src[1];
  if (weights == NULL || input == NULL || weights->type != GGML_TYPE_Q1_0 ||
      input->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32 ||
      weights->ne[2] != 1 || weights->ne[3] != 1 || input->ne[2] != 1 ||
      input->ne[3] != 1 || dst->ne[2] != 1 || dst->ne[3] != 1 ||
      weights->ne[0] <= 0 || weights->ne[0] > 6144 ||
      (weights->ne[0] % 128) != 0 || weights->ne[1] <= 0 ||
      weights->ne[1] > UINT32_MAX || input->ne[0] != weights->ne[0] ||
      dst->ne[0] != weights->ne[1] || dst->ne[1] != input->ne[1] ||
      weights->nb[1] != (size_t)(weights->ne[0] / 128) * 18u ||
      input->nb[0] != sizeof(float) ||
      input->nb[1] != (size_t)input->ne[0] * sizeof(float) ||
      dst->nb[0] != sizeof(float) ||
      dst->nb[1] != (size_t)dst->ne[0] * sizeof(float)) {
    if (weights != NULL && weights->type == GGML_TYPE_Q1_0 &&
        params->ith == 0 && !decline_reported &&
        getenv("BONSAI_DEBUG") != NULL) {
      decline_reported = 1;
      fprintf(stderr,
              "[bonsai] guard declined Q1 mul_mat: w ne=[%lld,%lld,%lld,%lld]"
              " nb1=%zu (want %zu) in ne0=%lld nb=[%zu,%zu]"
              " dst ne=[%lld,%lld] nb=[%zu,%zu] itype=%d dtype=%d\n",
              (long long)weights->ne[0], (long long)weights->ne[1],
              (long long)weights->ne[2], (long long)weights->ne[3],
              weights->nb[1], (size_t)(weights->ne[0] / 128) * 18u,
              input ? (long long)input->ne[0] : -1,
              input ? input->nb[0] : 0, input ? input->nb[1] : 0,
              (long long)dst->ne[0], (long long)dst->ne[1],
              dst->nb[0], dst->nb[1],
              input ? (int)input->type : -1, (int)dst->type);
      fflush(stderr);
    }
    ggml_compute_forward_mul_mat(params, dst);
    return;
  }

  uint32_t columns = (uint32_t)weights->ne[0];
  uint32_t rows = (uint32_t)weights->ne[1];

  if (input->ne[1] == 1 && params->nth > 1) {
    if (params->ith == 0)
      whole_matrix_accelerated = bonsai_accelerate_f32_matvec_submit(
          weights->data, columns, rows, (const float *)input->data);
    ggml_barrier(params->threadpool);
    if (whole_matrix_accelerated) {
      uint32_t chunk = (rows + (uint32_t)params->nth - 1u) /
                       (uint32_t)params->nth;
      uint32_t first = (uint32_t)params->ith * chunk;
      if (first < rows) {
        uint32_t count = rows - first < chunk ? rows - first : chunk;
        bonsai_accelerate_read_rows((float *)dst->data, first, count);
      }
      ggml_barrier(params->threadpool);
      if (params->ith == 0) bonsai_accelerate_release();
      return;
    }
    ggml_compute_forward_mul_mat(params, dst);
    return;
  }

  if (params->ith == 0) {
    whole_matrix_accelerated = 1;
    for (int64_t vector = 0; vector < input->ne[1]; ++vector) {
      const float *input_vector =
          (const float *)((const uint8_t *)input->data +
                          vector * input->nb[1]);
      float *output_vector =
          (float *)((uint8_t *)dst->data + vector * dst->nb[1]);
      if (!bonsai_accelerate_f32_matvec(weights->data, columns, rows,
                                        input_vector, output_vector)) {
        whole_matrix_accelerated = 0;
        break;
      }
    }
  }

  if (params->nth > 1)
    ggml_barrier(params->threadpool);
  if (!whole_matrix_accelerated)
    ggml_compute_forward_mul_mat(params, dst);
}

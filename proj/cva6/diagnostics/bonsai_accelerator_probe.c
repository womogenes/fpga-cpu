#define _GNU_SOURCE

#include <errno.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>

#define GPIO_PHYS UINT64_C(0x40000000)
#define MODEL_PHYS UINT64_C(0xaf000000)
#define MODEL_TAIL_PHYS UINT64_C(0xbdd00000)
#define MODEL_BYTES UINT32_C(0x0ecccac0)
#define MODEL_TAIL_BYTES (2u * 1024u * 1024u)
#define SCRATCH_PHYS UINT64_C(0xbe000000)
#define SCRATCH_BYTES (4u * 1024u * 1024u)
#define OUTPUT_OFFSET UINT32_C(0x00020000)
#define RESIDUAL_OUTPUT_OFFSET UINT32_C(0x00040000)
#define RESIDUAL_SUM_OFFSET UINT32_C(0x00280000)

#define STATUS_PRESENT UINT32_C(0x80000000)
#define STATUS_ERROR UINT32_C(0x20000000)
#define STATUS_DONE UINT32_C(0x10000000)
#define STATUS_ACK UINT32_C(0x08000000)
#define STATUS_GENERATION UINT32_C(0x04000000)

struct block_q8_0 {
  uint16_t d;
  int8_t qs[32];
} __attribute__((packed));

static volatile uint32_t *gpio;
static uint32_t command_toggle;

static void fence_io(void) {
  __asm__ volatile("fence iorw, iorw" ::: "memory");
}

static double now_seconds(void) {
  struct timespec value;
  if (clock_gettime(CLOCK_MONOTONIC, &value) != 0) return 0.0;
  return (double)value.tv_sec + (double)value.tv_nsec * 1e-9;
}

static void report_status(const char *label, uint32_t status) {
  uint32_t diagnostic = (status >> 8) & UINT32_C(0x0003ffff);
  printf("%s status=%08x outer=%u packer=%u reader_busy=%u "
         "reader_done=%u ffn_busy=%u ffn_done=%u group_drain=%u "
         "block_valid=%u writer_active=%u writes_outstanding=%u\n",
         label, status, (diagnostic >> 12) & 0x3fu,
         (diagnostic >> 8) & 0x0fu, (diagnostic >> 7) & 1u,
         (diagnostic >> 6) & 1u, (diagnostic >> 5) & 1u,
         (diagnostic >> 4) & 1u, (diagnostic >> 3) & 1u,
         (diagnostic >> 2) & 1u, (diagnostic >> 1) & 1u,
         diagnostic & 1u);
  fflush(stdout);
}

static int write_register(uint32_t index, uint32_t value) {
  command_toggle ^= UINT32_C(0x80000000);
  gpio[0] = command_toggle | UINT32_C(0x20000000) |
            ((index & 7u) << 25) | (value & UINT32_C(0x01ffffff));
  fence_io();
  uint32_t expected = command_toggle != 0u ? STATUS_ACK : 0u;
  for (uint32_t timeout = 0; timeout < UINT32_C(100000000); ++timeout) {
    if ((gpio[2] & STATUS_ACK) == expected) return 0;
    __asm__ volatile("nop");
  }
  report_status("register-timeout", gpio[2]);
  return -1;
}

static uint32_t address_config(uint64_t input, uint64_t output,
                               uint32_t scalar_bits) {
  uint32_t input_words = (uint32_t)(input >> 3);
  uint32_t output_words = (uint32_t)(output >> 3);
  return ((input_words >> 25) & 3u) | 4u |
         (((output_words >> 25) & 3u) << 3) | 32u |
         (((scalar_bits >> 25) & 0x7fu) << 7);
}

static int run_command(const char *label, uint32_t mode, uint32_t ne0,
                       uint32_t ne1, uint64_t weight, uint64_t input,
                       uint64_t output, uint32_t scalar_bits) {
  uint32_t old_done = gpio[2] & STATUS_DONE;
  uint32_t input_words = (uint32_t)(input >> 3);
  uint32_t output_words = (uint32_t)(output >> 3);
  if (write_register(0u, (uint32_t)((weight - MODEL_PHYS) >> 3)) != 0 ||
      write_register(1u, input_words) != 0 ||
      write_register(2u, output_words) != 0 ||
      write_register(3u, ne0) != 0 ||
      write_register(4u, ne1) != 0 ||
      write_register(6u, address_config(input, output, scalar_bits)) != 0 ||
      write_register(7u, scalar_bits) != 0)
    return -1;
  double start = now_seconds();
  if (write_register(5u, mode) != 0) return -1;
  uint32_t status;
  do {
    status = gpio[2];
  } while ((status & STATUS_DONE) == old_done);
  double elapsed = now_seconds() - start;
  printf("%s elapsed=%.6f seconds\n", label, elapsed);
  report_status(label, status);
  return (status & STATUS_ERROR) == 0u ? 0 : -1;
}

static int check_q8_output(const char *label, volatile uint8_t *output,
                           uint32_t blocks) {
  uint32_t bad = 0;
  uint32_t reported = 0;
  for (uint32_t block = 0; block < blocks; ++block) {
    volatile struct block_q8_0 *value =
        (volatile struct block_q8_0 *)(output + block * 34u);
    uint32_t block_bad = value->d == 0u ? 1u : 0u;
    for (uint32_t item = 0; item < 32u; ++item)
      if (value->qs[item] != 127) ++block_bad;
    bad += block_bad;
    if (block_bad != 0u && reported < 12u) {
      printf("%s bad block=%u scale=%04x q=%d,%d,%d,%d errors=%u\n",
             label, block, value->d, value->qs[0], value->qs[1],
             value->qs[2], value->qs[3], block_bad);
      ++reported;
    }
  }
  volatile struct block_q8_0 *first =
      (volatile struct block_q8_0 *)output;
  printf("%s output blocks=%u first_scale=%04x first_q=%d bad=%u\n",
         label, blocks, first->d, first->qs[0], bad);
  return bad == 0u ? 0 : -1;
}

static int probe_norm(volatile uint8_t *scratch) {
  volatile uint32_t *source = (volatile uint32_t *)scratch;
  volatile uint8_t *output = scratch + OUTPUT_OFFSET;
  for (uint32_t index = 0; index < 4096u; ++index)
    source[index] = UINT32_C(0x3f800000);
  memset((void *)output, 0, 2048u * 34u / 32u);
  fence_io();
  if (run_command("norm", 7u, 2048u, 2u, MODEL_PHYS,
                  SCRATCH_PHYS, SCRATCH_PHYS + OUTPUT_OFFSET,
                  UINT32_C(0x3f800000)) != 0)
    return -1;
  fence_io();
  return check_q8_output("norm", output, 64u);
}

static int probe_ffn(volatile uint8_t *scratch, uint32_t elements,
                     int mirror_halves, uint32_t output_offset,
                     int clear_output) {
  volatile uint64_t *source = (volatile uint64_t *)scratch;
  volatile uint8_t *output = scratch + output_offset;
  uint64_t source_word = mirror_halves ? UINT64_C(0x3f8000003f800000) :
                                         UINT64_C(0x000000003f800000);
  for (uint32_t index = 0; index < elements * 2u; ++index)
    source[index] = source_word;
  if (clear_output)
    memset((void *)output, 0, elements * 34u / 32u);
  fence_io();
  if (run_command("ffn", 7u, elements, 1u, MODEL_PHYS,
                  SCRATCH_PHYS, SCRATCH_PHYS + output_offset, 0u) != 0)
    return -1;
  fence_io();
  return check_q8_output("ffn", output, elements / 32u);
}

static int probe_residual(volatile uint8_t *scratch) {
  volatile uint32_t *source = (volatile uint32_t *)scratch;
  volatile uint32_t *output =
      (volatile uint32_t *)(scratch + RESIDUAL_OUTPUT_OFFSET);
  volatile uint32_t *sum =
      (volatile uint32_t *)(scratch + RESIDUAL_SUM_OFFSET);
  for (uint32_t index = 0; index < 2048u; ++index) {
    source[index] = UINT32_C(0x3f800000);          /* 1.0 */
    source[2048u + index] = UINT32_C(0x40000000); /* 2.0 */
    output[index] = 0u;
  }
  *sum = 0u;
  fence_io();
  if (run_command("residual", 7u, 2048u, 5u, MODEL_PHYS,
                  SCRATCH_PHYS, SCRATCH_PHYS + RESIDUAL_OUTPUT_OFFSET,
                  0u) != 0)
    return -1;
  fence_io();

  uint32_t bad = 0u;
  for (uint32_t index = 0; index < 2048u; ++index)
    if (output[index] != UINT32_C(0x40400000)) ++bad; /* 3.0 */
  printf("residual output first=%08x last=%08x sum=%08x "
         "expected_sum=46900000 bad=%u\n",
         output[0], output[2047], *sum, bad);
  return bad == 0u && *sum == UINT32_C(0x46900000) ? 0 : -1;
}

static int probe_model_checksum(volatile uint8_t *model) {
  uint32_t checksum = UINT32_C(2166136261);
  double start = now_seconds();
  for (uint32_t index = 0; index < MODEL_BYTES; ++index) {
    checksum ^= model[index];
    checksum *= UINT32_C(16777619);
  }
  printf("model checksum bytes=%u fnv1a=%08x elapsed=%.6f seconds\n",
         MODEL_BYTES, checksum, now_seconds() - start);
  return checksum == UINT32_C(0x43da9b6a) ? 0 : -1;
}

static int probe_matvec(const char *label, uint64_t weight_phys,
                        volatile uint8_t *model,
                        volatile uint8_t *scratch) {
  volatile uint16_t *scales = (volatile uint16_t *)model;
  for (uint32_t index = 0; index < 4u; ++index) scales[index] = 0x3c00u;
  for (uint32_t index = 8u; index < 72u; ++index) model[index] = 0xffu;
  volatile struct block_q8_0 *input =
      (volatile struct block_q8_0 *)scratch;
  for (uint32_t block = 0; block < 16u; ++block) {
    input[block].d = 0x3c00u;
    for (uint32_t item = 0; item < 32u; ++item) input[block].qs[item] = 1;
  }
  volatile uint64_t *output =
      (volatile uint64_t *)(scratch + OUTPUT_OFFSET);
  *output = 0u;
  fence_io();
  if (run_command(label, 5u, 512u, 1u, weight_phys,
                  SCRATCH_PHYS, SCRATCH_PHYS + OUTPUT_OFFSET, 0u) != 0)
    return -1;
  fence_io();
  uint32_t result_bits = (uint32_t)*output;
  float result;
  memcpy(&result, &result_bits, sizeof(result));
  printf("matvec output bits=%08x value=%.6f expected=512.000000\n",
         result_bits, (double)result);
  return result_bits == UINT32_C(0x44000000) ? 0 : -1;
}

static int probe_matvec_large(volatile uint8_t *model,
                              volatile uint8_t *scratch, uint32_t ne0,
                              uint32_t rows, int argmax) {
  static uint32_t generation;
  if (ne0 < 512u || ne0 > 2048u || (ne0 & 511u) != 0u || rows == 0u ||
      (uint64_t)(ne0 / 128u) * 18u * rows > MODEL_TAIL_BYTES)
    return -1;
  uint32_t blocks = ne0 / 128u;
  uint32_t row_bytes = blocks * 18u;
  for (uint32_t row = 0; row < rows; ++row) {
    volatile uint8_t *row_data = model + row * row_bytes;
    for (uint32_t group = 0; group < blocks / 4u; ++group) {
      volatile uint16_t *scales =
          (volatile uint16_t *)(row_data + group * 72u);
      for (uint32_t index = 0; index < 4u; ++index)
        scales[index] = UINT16_C(0x3c00);
      for (uint32_t index = 8u; index < 72u; ++index)
        row_data[group * 72u + index] = UINT8_C(0xff);
    }
  }
  volatile struct block_q8_0 *input =
      (volatile struct block_q8_0 *)scratch;
  ++generation;
  int8_t input_value = (int8_t)(1u + generation % 31u);
  for (uint32_t block = 0; block < ne0 / 32u; ++block) {
    input[block].d = UINT16_C(0x3c00);
    for (uint32_t item = 0; item < 32u; ++item)
      input[block].qs[item] = input_value;
  }
  volatile uint32_t *output =
      (volatile uint32_t *)(scratch + OUTPUT_OFFSET);

  /* This CVA6 platform can acknowledge a fence before the final uncached DDR
     stores have reached the AXI fabric. Read every word written by the probe
     back through the CPU path before launching DMA. The output is deliberately
     not CPU-written: each invocation uses a different input value, so a lost
     FPGA write is observed as the preceding generation's result without a
     CPU cache alias contaminating the measurement. */
  volatile uint64_t drain = 0u;
  const volatile uint64_t *model_words =
      (const volatile uint64_t *)model;
  uint32_t model_word_count = (row_bytes * rows) / sizeof(uint64_t);
  for (uint32_t word = 0; word < model_word_count; ++word)
    drain ^= model_words[word];
  const volatile uint64_t *input_words =
      (const volatile uint64_t *)input;
  uint32_t input_word_count =
      ((ne0 / 32u) * sizeof(*input)) / sizeof(uint64_t);
  for (uint32_t word = 0; word < input_word_count; ++word)
    drain ^= input_words[word];
  (void)drain;
  fence_io();
  const char *label = argmax ? "matvec-argmax" : "matvec-large";
  if (run_command(label, argmax ? 6u : 5u, ne0, rows, MODEL_TAIL_PHYS,
                  SCRATCH_PHYS, SCRATCH_PHYS + OUTPUT_OFFSET, 0u) != 0)
    return -1;
  fence_io();
  if (argmax) {
    puts("matvec-argmax completed without per-row result writes");
    return 0;
  }
  /* Avoid target floating-point state in the oracle. These positive integer
     results fit below 2^24 and therefore have an exact FP32 encoding. */
  uint32_t expected_integer = ne0 * (uint32_t)input_value;
  uint32_t expected_msb = 0u;
  for (uint32_t bit = 0u; bit < 24u; ++bit)
    if ((expected_integer >> bit) != 0u) expected_msb = bit;
  uint32_t expected_bits = ((expected_msb + 127u) << 23) |
      ((expected_integer << (23u - expected_msb)) & UINT32_C(0x007fffff));
  uint32_t bad = 0u;
  uint32_t reported = 0u;
  /* Non-argmax mode 5 uses the ordinary 64-bit output stride.  Float-output
     gateware writes FP32 in the low word and leaves the upper word clear. */
  for (uint32_t row = 0; row < rows; ++row) {
    if (output[row * 2u] != expected_bits) {
      if (reported < 12u) {
        printf("matvec-large bad row=%u got=%08x expected=%08x\n",
               row, output[row * 2u], expected_bits);
        ++reported;
      }
      ++bad;
    }
  }
  printf("matvec-large ne0=%u rows=%u input=%d output first=%08x "
         "last=%08x expected=%08x bad=%u\n", ne0, rows, input_value,
         output[0], output[(rows - 1u) * 2u], expected_bits, bad);
  return bad == 0u ? 0 : -1;
}

int main(int argc, char **argv) {
  int memory = open("/dev/mem", O_RDWR | O_SYNC);
  if (memory < 0) {
    perror("open /dev/mem");
    return 1;
  }
  gpio = mmap(NULL, 4096u, PROT_READ | PROT_WRITE, MAP_SHARED, memory,
              (off_t)GPIO_PHYS);
  volatile uint8_t *scratch = mmap(NULL, SCRATCH_BYTES,
      PROT_READ | PROT_WRITE, MAP_SHARED, memory, (off_t)SCRATCH_PHYS);
  volatile uint8_t *model = mmap(NULL, 4096u, PROT_READ | PROT_WRITE,
      MAP_SHARED, memory, (off_t)MODEL_PHYS);
  volatile uint8_t *model_full = mmap(NULL, MODEL_BYTES, PROT_READ,
      MAP_SHARED, memory, (off_t)MODEL_PHYS);
  volatile uint8_t *model_tail = mmap(NULL, MODEL_TAIL_BYTES,
      PROT_READ | PROT_WRITE, MAP_SHARED, memory, (off_t)MODEL_TAIL_PHYS);
  close(memory);
  if (gpio == MAP_FAILED || scratch == MAP_FAILED || model == MAP_FAILED ||
      model_full == MAP_FAILED || model_tail == MAP_FAILED) {
    perror("mmap accelerator regions");
    return 1;
  }
  uint32_t initial = gpio[2];
  report_status("initial", initial);
  if ((initial & (STATUS_PRESENT | STATUS_GENERATION)) !=
      (STATUS_PRESENT | STATUS_GENERATION)) {
    fprintf(stderr, "transform-capable accelerator is not present\n");
    return 1;
  }

  gpio[0] = UINT32_C(0xae000000);
  fence_io();
  usleep(100u);
  gpio[0] = UINT32_C(0x2e000000);
  fence_io();
  usleep(100u);
  command_toggle = 0u;

  int failures = 0;
  if (argc == 1) {
    failures += probe_matvec("matvec", MODEL_PHYS, model, scratch) != 0;
    failures += probe_norm(scratch) != 0;
    failures += probe_ffn(scratch, 6144u, 0, OUTPUT_OFFSET, 1) != 0;
    failures += probe_residual(scratch) != 0;
  } else {
    uint32_t fresh_output_offset = UINT32_C(0x00040000);
    for (int argument = 1; argument < argc; ++argument) {
      if (strcmp(argv[argument], "matvec") == 0) {
        failures += probe_matvec("matvec", MODEL_PHYS, model, scratch) != 0;
      } else if (strcmp(argv[argument], "matvec-tail") == 0) {
        failures += probe_matvec("matvec-tail", MODEL_TAIL_PHYS,
                                 model_tail, scratch) != 0;
      } else if (strncmp(argv[argument], "matvec-shape:", 13u) == 0 ||
                 strncmp(argv[argument], "matvec-argmax:", 14u) == 0) {
        int argmax = strncmp(argv[argument], "matvec-argmax:", 14u) == 0;
        const char *shape = argv[argument] + (argmax ? 14u : 13u);
        unsigned int ne0 = 0u;
        unsigned int rows = 0u;
        char extra = '\0';
        if (sscanf(shape, "%u:%u%c", &ne0, &rows, &extra) != 2 ||
            ne0 < 512u || ne0 > 2048u ||
            (ne0 & 511u) != 0u || rows == 0u ||
            (uint64_t)(ne0 / 128u) * 18u * rows > MODEL_TAIL_BYTES) {
          fprintf(stderr, "invalid matvec shape: %s\n",
                  argv[argument]);
          ++failures;
        } else {
          failures += probe_matvec_large(model_tail, scratch, ne0, rows,
                                         argmax) != 0;
        }
      } else if (strcmp(argv[argument], "model-checksum") == 0) {
        failures += probe_model_checksum(model_full) != 0;
      } else if (strcmp(argv[argument], "norm") == 0) {
        failures += probe_norm(scratch) != 0;
      } else if (strcmp(argv[argument], "residual") == 0) {
        failures += probe_residual(scratch) != 0;
      } else if (strncmp(argv[argument], "ffn:", 4u) == 0 ||
                 strncmp(argv[argument], "ffn-mirror:", 11u) == 0 ||
                 strncmp(argv[argument], "ffn-fresh:", 10u) == 0) {
        int fresh_output = strncmp(argv[argument], "ffn-fresh:", 10u) == 0;
        int mirror_halves =
            strncmp(argv[argument], "ffn-mirror:", 11u) == 0;
        const char *count_text = argv[argument] +
            (mirror_halves ? 11u : fresh_output ? 10u : 4u);
        char *end = NULL;
        unsigned long elements = strtoul(count_text, &end, 0);
        if (end == count_text || *end != '\0' || elements < 128u ||
            elements > 6144u || (elements & 127u) != 0u) {
          fprintf(stderr, "invalid FFN element count: %s\n", argv[argument]);
          ++failures;
        } else {
          uint32_t output_offset = fresh_output ? fresh_output_offset :
                                                  OUTPUT_OFFSET;
          failures += probe_ffn(scratch, (uint32_t)elements, mirror_halves,
                                output_offset, !fresh_output) != 0;
          if (fresh_output)
            fresh_output_offset += UINT32_C(0x00020000);
        }
      } else {
        fprintf(stderr, "unknown probe: %s (use matvec, matvec-tail, matvec-shape:NE0:ROWS, matvec-argmax:NE0:ROWS, model-checksum, norm, residual, "
                        "ffn:N, ffn-mirror:N, or ffn-fresh:N)\n",
                argv[argument]);
        ++failures;
      }
    }
  }
  printf("Bonsai accelerator probe: %s (%d failures)\n",
         failures == 0 ? "PASS" : "FAIL", failures);
  return failures == 0 ? 0 : 1;
}

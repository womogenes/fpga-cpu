#define _GNU_SOURCE

#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#define MODEL_PHYS UINT64_C(0xaf000000)
#define MODEL_SLOT_BYTES (240u * 1024u * 1024u)
#define WINDOW_BYTES (4u * 1024u * 1024u)
#define WORKER_COUNT 4u

struct worker {
  const uint8_t *source;
  uint8_t *destination;
  size_t size;
};

static double seconds_since(const struct timespec *start) {
  struct timespec now;
  (void)clock_gettime(CLOCK_MONOTONIC, &now);
  return (double)(now.tv_sec - start->tv_sec) +
         (double)(now.tv_nsec - start->tv_nsec) / 1.0e9;
}

static void *copy_worker(void *argument) {
  struct worker *worker = argument;
  (void)memcpy(worker->destination, worker->source, worker->size);
  return NULL;
}

static int read_window(int file, uint8_t *buffer, size_t size) {
  size_t received = 0u;
  while (received < size) {
    ssize_t result = read(file, buffer + received, size - received);
    if (result < 0 && errno == EINTR) continue;
    if (result <= 0) {
      if (result == 0) errno = EIO;
      return -1;
    }
    received += (size_t)result;
  }
  return 0;
}

int main(int argc, char **argv) {
  if (argc != 2) {
    fprintf(stderr, "usage: bonsai-stage-stream PACKED_MODEL\n");
    return EXIT_FAILURE;
  }
  int source_file = open(argv[1], O_RDONLY);
  struct stat status;
  if (source_file < 0 || fstat(source_file, &status) != 0 ||
      !S_ISREG(status.st_mode) || status.st_size < 4 ||
      (uint64_t)status.st_size > MODEL_SLOT_BYTES) {
    perror("bonsai-stage-stream: open/stat model");
    if (source_file >= 0) (void)close(source_file);
    return EXIT_FAILURE;
  }
  size_t model_size = (size_t)status.st_size;
  uint8_t *buffer = NULL;
  int allocation = posix_memalign((void **)&buffer, 4096u, WINDOW_BYTES);
  if (allocation != 0) {
    errno = allocation;
    perror("bonsai-stage-stream: allocate buffer");
    (void)close(source_file);
    return EXIT_FAILURE;
  }
  int memory = open("/dev/mem", O_RDWR | O_SYNC);
  if (memory < 0) {
    perror("bonsai-stage-stream: open /dev/mem");
    free(buffer);
    (void)close(source_file);
    return EXIT_FAILURE;
  }
  uint8_t *model = mmap(NULL, model_size, PROT_READ | PROT_WRITE, MAP_SHARED,
                        memory, (off_t)MODEL_PHYS);
  if (model == MAP_FAILED) {
    perror("bonsai-stage-stream: map reserved DDR");
    (void)close(memory);
    free(buffer);
    (void)close(source_file);
    return EXIT_FAILURE;
  }

  struct timespec start;
  (void)clock_gettime(CLOCK_MONOTONIC, &start);
  printf("Streaming %.1f MiB from SD to reserved DDR with %u harts...\n",
         (double)model_size / (1024.0 * 1024.0), WORKER_COUNT);
  fflush(stdout);
  size_t offset = 0u;
  while (offset < model_size) {
    size_t window = model_size - offset;
    if (window > WINDOW_BYTES) window = WINDOW_BYTES;
    if (read_window(source_file, buffer, window) != 0) {
      perror("bonsai-stage-stream: read model");
      goto failure;
    }
    if (offset == 0u && memcmp(buffer, "BGUF", 4u) != 0) {
      fprintf(stderr, "bonsai-stage-stream: packed model header is not BGUF\n");
      goto failure;
    }

    pthread_t threads[WORKER_COUNT];
    struct worker workers[WORKER_COUNT];
    size_t part = (window + WORKER_COUNT - 1u) / WORKER_COUNT;
    part = (part + 4095u) & ~(size_t)4095u;
    unsigned created = 0u;
    for (unsigned index = 0u; index < WORKER_COUNT; ++index) {
      size_t begin = (size_t)index * part;
      size_t end = begin + part;
      if (begin > window) begin = window;
      if (end > window) end = window;
      workers[index] = (struct worker){
          .source = buffer + begin,
          .destination = model + offset + begin,
          .size = end - begin};
      int error = pthread_create(&threads[index], NULL, copy_worker,
                                 &workers[index]);
      if (error != 0) {
        errno = error;
        perror("bonsai-stage-stream: create worker");
        break;
      }
      ++created;
    }
    int failed = created != WORKER_COUNT;
    for (unsigned index = 0u; index < created; ++index) {
      int error = pthread_join(threads[index], NULL);
      if (error != 0) {
        errno = error;
        perror("bonsai-stage-stream: DDR write");
        failed = 1;
      }
    }
    if (failed) goto failure;
    (void)posix_fadvise(source_file, (off_t)offset, (off_t)window,
                        POSIX_FADV_DONTNEED);
    offset += window;
    double elapsed = seconds_since(&start);
    printf("\rStage: %3zu%% (%.2f MiB/s)", offset * 100u / model_size,
           (double)offset / (1024.0 * 1024.0) / elapsed);
    fflush(stdout);
  }
  __sync_synchronize();
  printf("\nStaged %zu bytes at 0x%llx in %.2f s.\n", model_size,
         (unsigned long long)MODEL_PHYS, seconds_since(&start));
  (void)munmap(model, model_size);
  (void)close(memory);
  (void)close(source_file);
  free(buffer);
  return EXIT_SUCCESS;

failure:
  (void)munmap(model, model_size);
  (void)close(memory);
  (void)close(source_file);
  free(buffer);
  return EXIT_FAILURE;
}

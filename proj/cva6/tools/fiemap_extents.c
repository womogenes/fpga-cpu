#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <linux/fiemap.h>
#include <linux/fs.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/stat.h>
#include <unistd.h>

enum { extent_capacity = 1024 };

int main(int argc, char **argv) {
  if (argc != 4) {
    fprintf(stderr, "usage: fiemap-extents FILE PARTITION_START_LBA DESTINATION\n");
    return EXIT_FAILURE;
  }

  char *end = NULL;
  uint64_t partition_lba = strtoull(argv[2], &end, 0);
  if (*argv[2] == '\0' || *end != '\0') {
    fprintf(stderr, "invalid partition start LBA: %s\n", argv[2]);
    return EXIT_FAILURE;
  }
  uint64_t destination = strtoull(argv[3], &end, 0);
  if (*argv[3] == '\0' || *end != '\0') {
    fprintf(stderr, "invalid destination: %s\n", argv[3]);
    return EXIT_FAILURE;
  }

  int file = open(argv[1], O_RDONLY);
  struct stat status;
  if (file < 0 || fstat(file, &status) != 0) {
    perror("fiemap-extents");
    if (file >= 0) close(file);
    return EXIT_FAILURE;
  }

  size_t bytes = sizeof(struct fiemap) +
                 extent_capacity * sizeof(struct fiemap_extent);
  struct fiemap *map = calloc(1, bytes);
  if (map == NULL) {
    perror("fiemap-extents: calloc");
    close(file);
    return EXIT_FAILURE;
  }
  map->fm_start = 0;
  map->fm_length = FIEMAP_MAX_OFFSET;
  map->fm_extent_count = extent_capacity;
  if (ioctl(file, FS_IOC_FIEMAP, map) != 0) {
    perror("fiemap-extents: FS_IOC_FIEMAP");
    free(map);
    close(file);
    return EXIT_FAILURE;
  }

  uint64_t covered = 0;
  uint64_t file_size = (uint64_t)status.st_size;
  for (uint32_t index = 0; index < map->fm_mapped_extents; ++index) {
    const struct fiemap_extent *extent = &map->fm_extents[index];
    if (extent->fe_logical != covered || (extent->fe_physical & 511) != 0) {
      fprintf(stderr, "non-contiguous or unaligned logical extent %u\n", index);
      free(map);
      close(file);
      return EXIT_FAILURE;
    }
    uint64_t remaining = file_size - covered;
    uint64_t length = extent->fe_length < remaining ? extent->fe_length : remaining;
    uint64_t blocks = (length + 511) / 512;
    uint64_t source_lba = partition_lba + extent->fe_physical / 512;
    printf("--raw-extent 0x%" PRIx64 ":0x%" PRIx64 ":0x%" PRIx64 "\n",
           destination + covered, source_lba, blocks);
    covered += length;
  }

  if (covered != file_size) {
    fprintf(stderr, "extents cover %" PRIu64 " of %" PRIu64 " bytes\n",
            covered, file_size);
    free(map);
    close(file);
    return EXIT_FAILURE;
  }
  fprintf(stderr, "%u extents cover %" PRIu64 " bytes\n",
          map->fm_mapped_extents, file_size);
  free(map);
  close(file);
  return EXIT_SUCCESS;
}

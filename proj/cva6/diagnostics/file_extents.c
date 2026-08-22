#include <errno.h>
#include <fcntl.h>
#include <linux/fiemap.h>
#include <linux/fs.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/stat.h>
#include <unistd.h>

enum { EXTENT_CAPACITY = 128 };

int main(int argc, char **argv) {
  if (argc != 2) {
    fprintf(stderr, "usage: file-extents FILE\n");
    return EXIT_FAILURE;
  }

  int file = open(argv[1], O_RDONLY);
  struct stat status;
  if (file < 0 || fstat(file, &status) != 0) {
    fprintf(stderr, "file-extents: %s: %s\n", argv[1], strerror(errno));
    if (file >= 0) (void)close(file);
    return EXIT_FAILURE;
  }

  size_t bytes = sizeof(struct fiemap) +
                 EXTENT_CAPACITY * sizeof(struct fiemap_extent);
  struct fiemap *map = calloc(1, bytes);
  if (map == NULL) {
    perror("file-extents: calloc");
    (void)close(file);
    return EXIT_FAILURE;
  }
  map->fm_start = 0;
  map->fm_length = FIEMAP_MAX_OFFSET;
  map->fm_flags = FIEMAP_FLAG_SYNC;
  map->fm_extent_count = EXTENT_CAPACITY;
  if (ioctl(file, FS_IOC_FIEMAP, map) != 0) {
    perror("file-extents: FIEMAP");
    free(map);
    (void)close(file);
    return EXIT_FAILURE;
  }

  printf("FILE_SIZE %llu\n", (unsigned long long)status.st_size);
  printf("EXTENT_COUNT %u\n", map->fm_mapped_extents);
  for (uint32_t index = 0; index < map->fm_mapped_extents; ++index) {
    const struct fiemap_extent *extent = &map->fm_extents[index];
    printf("EXTENT %u logical=%llu physical=%llu length=%llu flags=0x%x\n",
           index, (unsigned long long)extent->fe_logical,
           (unsigned long long)extent->fe_physical,
           (unsigned long long)extent->fe_length, extent->fe_flags);
  }

  free(map);
  (void)close(file);
  return EXIT_SUCCESS;
}

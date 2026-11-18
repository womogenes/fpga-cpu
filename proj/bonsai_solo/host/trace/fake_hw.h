/* Force-included into the CVA6 Bonsai runtime when it is built for x86 as a
   trace harness. /dev/mem is replaced by heap-backed physical regions and the
   GPIO command port by an emulator thread, so the unmodified runtime records
   its DDR setup image and accelerator register sequence. */
#ifndef BONSAI_FAKE_HW_H
#define BONSAI_FAKE_HW_H

#include <fcntl.h>
#include <stddef.h>
#include <sys/mman.h>
#include <sys/types.h>

int fake_open(const char *path, int flags, ...);
void *fake_mmap(void *address, size_t length, int protection, int flags,
                int fd, off_t offset);
int fake_munmap(void *address, size_t length);

#define open fake_open
#define mmap fake_mmap
#define munmap fake_munmap

#endif

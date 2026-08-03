#include <errno.h>
#include <stddef.h>
#include <stdint.h>
#include <sys/stat.h>

#include <runtime.h>

extern char _program_data_free;
extern char _program_data_end;

int _write(int file, const void* buffer, size_t length) {
  if (file != 1 && file != 2) {
    errno = EBADF;
    return -1;
  }

  const char* characters = (const char*)buffer;
  for (size_t i = 0; i < length; i++) {
    if (characters[i] == '\n')
      uart_write("\r", 1u);
    uart_write(&characters[i], 1u);
  }
  return (int)length;
}

void* _sbrk(ptrdiff_t increment) {
  static char* heap_end;
  char* previous_end;

  if (heap_end == NULL)
    heap_end = &_program_data_free;
  previous_end = heap_end;

  if (increment > 0 &&
      (uintptr_t)increment > (uintptr_t)&_program_data_end -
                                 (uintptr_t)heap_end) {
    errno = ENOMEM;
    return (void*)-1;
  }
  if (increment < 0 &&
      (uintptr_t)(-increment) > (uintptr_t)heap_end -
                                  (uintptr_t)&_program_data_free) {
    errno = EINVAL;
    return (void*)-1;
  }

  heap_end += increment;
  return previous_end;
}

int _close(int file) {
  if (file >= 0 && file <= 2)
    return 0;
  errno = EBADF;
  return -1;
}

int _fstat(int file, struct stat* status) {
  if (file < 0 || file > 2) {
    errno = EBADF;
    return -1;
  }
  status->st_mode = S_IFCHR;
  return 0;
}

int _isatty(int file) {
  if (file >= 0 && file <= 2)
    return 1;
  errno = EBADF;
  return 0;
}

int _lseek(int file, int offset, int whence) {
  (void)file;
  (void)offset;
  (void)whence;
  return 0;
}

int _read(int file, void* buffer, size_t length) {
  (void)buffer;
  (void)length;
  if (file == 0)
    return 0;
  errno = EBADF;
  return -1;
}

int _getpid(void) {
  return 1;
}

int _kill(int process, int signal) {
  (void)process;
  (void)signal;
  errno = EINVAL;
  return -1;
}

void _exit(int status) {
  (void)status;
  while (1);
}

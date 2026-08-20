/* Minimal initramfs recovery program using only Linux syscalls.
 *
 * This deliberately has no libc dependency: it must still execute when the
 * SD root filesystem's dynamic loader or C runtime has been damaged.
 */

#define AT_FDCWD (-100)
#define O_WRONLY 1
#define O_TRUNC 512

#define SYS_MKDIRAT 34
#define SYS_MOUNT 40
#define SYS_OPENAT 56
#define SYS_CLOSE 57
#define SYS_READ 63
#define SYS_WRITE 64
#define SYS_SYNC 81
#define SYS_FSYNC 82
#define SYS_NANOSLEEP 101
#define SYS_EXIT 93

struct timespec {
  long seconds;
  long nanoseconds;
};

static long syscall1(long number, long arg0) {
  register long a0 __asm__("a0") = arg0;
  register long a7 __asm__("a7") = number;
  __asm__ volatile("ecall" : "+r"(a0) : "r"(a7) : "memory");
  return a0;
}

static long syscall2(long number, long arg0, long arg1) {
  register long a0 __asm__("a0") = arg0;
  register long a1 __asm__("a1") = arg1;
  register long a7 __asm__("a7") = number;
  __asm__ volatile("ecall" : "+r"(a0) : "r"(a1), "r"(a7) : "memory");
  return a0;
}

static long syscall3(long number, long arg0, long arg1, long arg2) {
  register long a0 __asm__("a0") = arg0;
  register long a1 __asm__("a1") = arg1;
  register long a2 __asm__("a2") = arg2;
  register long a7 __asm__("a7") = number;
  __asm__ volatile("ecall" : "+r"(a0) : "r"(a1), "r"(a2), "r"(a7) : "memory");
  return a0;
}

static long syscall5(long number, long arg0, long arg1, long arg2, long arg3,
                     long arg4) {
  register long a0 __asm__("a0") = arg0;
  register long a1 __asm__("a1") = arg1;
  register long a2 __asm__("a2") = arg2;
  register long a3 __asm__("a3") = arg3;
  register long a4 __asm__("a4") = arg4;
  register long a7 __asm__("a7") = number;
  __asm__ volatile("ecall" : "+r"(a0)
                   : "r"(a1), "r"(a2), "r"(a3), "r"(a4), "r"(a7)
                   : "memory");
  return a0;
}

static unsigned long string_length(const char *text) {
  unsigned long length = 0;
  while (text[length]) ++length;
  return length;
}

static void message(const char *text) {
  syscall3(SYS_WRITE, 1, (long)text, (long)string_length(text));
}

static int copy_file(const char *source, const char *destination) {
  static char buffer[16384];
  long input = syscall3(SYS_OPENAT, AT_FDCWD, (long)source, 0);
  if (input < 0) return -1;
  long output = syscall3(SYS_OPENAT, AT_FDCWD, (long)destination,
                         O_WRONLY | O_TRUNC);
  if (output < 0) return -2;

  for (;;) {
    long count = syscall3(SYS_READ, input, (long)buffer, sizeof(buffer));
    if (count < 0) return -3;
    if (count == 0) break;
    long offset = 0;
    while (offset < count) {
      long written = syscall3(SYS_WRITE, output, (long)(buffer + offset),
                              count - offset);
      if (written <= 0) return -4;
      offset += written;
    }
  }

  if (syscall1(SYS_FSYNC, output) < 0) return -5;
  syscall1(SYS_CLOSE, output);
  syscall1(SYS_CLOSE, input);
  return 0;
}

static int recovery_main(void) {
  syscall3(SYS_MKDIRAT, AT_FDCWD, (long)"/dev", 0755);
  syscall3(SYS_MKDIRAT, AT_FDCWD, (long)"/mnt", 0755);

  long result = syscall5(SYS_MOUNT, (long)"devtmpfs", (long)"/dev",
                         (long)"devtmpfs", 0, 0);
  if (result < 0 && result != -16) {
    message("RECOVERY_ERROR devtmpfs\n");
    return 1;
  }
  if (syscall5(SYS_MOUNT, (long)"/dev/mmcblk0p3", (long)"/mnt",
               (long)"ext4", 0, 0) < 0) {
    message("RECOVERY_ERROR rootfs mount\n");
    return 2;
  }

  if (copy_file("/payload/ld-linux-riscv64-lp64.so.1",
                "/mnt/lib/ld-linux-riscv64-lp64.so.1") != 0) {
    message("RECOVERY_ERROR loader copy\n");
    return 3;
  }
  if (copy_file("/payload/libc.so.6", "/mnt/lib/libc.so.6") != 0) {
    message("RECOVERY_ERROR libc copy\n");
    return 4;
  }
  if (copy_file("/payload/libm.so.6", "/mnt/lib/libm.so.6") != 0) {
    message("RECOVERY_ERROR libm copy\n");
    return 5;
  }

  syscall1(SYS_SYNC, 0);
  message("RECOVERY_OK loader libc libm\n");
  return 0;
}

__attribute__((noreturn, section(".text.start"))) void _start(void) {
  int status = recovery_main();
  if (status != 0) syscall1(SYS_EXIT, status);
  const struct timespec delay = {60, 0};
  for (;;) syscall2(SYS_NANOSLEEP, (long)&delay, 0);
}

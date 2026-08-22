/* Syscall-only FIEMAP reporter for the RV64IMAC CVA6 recovery image. */

typedef unsigned int u32;
typedef unsigned long u64;

struct fiemap_extent {
  u64 logical;
  u64 physical;
  u64 length;
  u64 reserved64[2];
  u32 flags;
  u32 reserved[3];
};

struct fiemap {
  u64 start;
  u64 length;
  u32 flags;
  u32 mapped_extents;
  u32 extent_count;
  u32 reserved;
  struct fiemap_extent extents[128];
};

static long syscall3(long number, long a0, long a1, long a2) {
  register long x10 __asm__("a0") = a0;
  register long x11 __asm__("a1") = a1;
  register long x12 __asm__("a2") = a2;
  register long x17 __asm__("a7") = number;
  __asm__ volatile("ecall" : "+r"(x10) : "r"(x11), "r"(x12), "r"(x17)
                   : "memory");
  return x10;
}

static void write_all(int descriptor, const char *data, u64 size) {
  while (size != 0) {
    long written = syscall3(64, descriptor, (long)data, (long)size);
    if (written <= 0) return;
    data += written;
    size -= (u64)written;
  }
}

static char *append_text(char *out, const char *text) {
  while (*text != '\0') *out++ = *text++;
  return out;
}

static char *append_hex(char *out, u64 value) {
  static const char digits[] = "0123456789abcdef";
  int started = 0;
  for (int shift = 60; shift >= 0; shift -= 4) {
    unsigned digit = (unsigned)(value >> shift) & 15u;
    if (digit != 0 || started || shift == 0) {
      *out++ = digits[digit];
      started = 1;
    }
  }
  return out;
}

void _start(void) {
  static const char path[] = "/root/models/Bonsai-1.7B-packed.gguf";
  static struct fiemap map;
  const u64 partition_lba = 0x152800;
  const u64 destination = 0xaf000000;
  const u64 file_size = 248302272;

  long file = syscall3(56, -100, (long)path, 0);
  if (file < 0) {
    write_all(2, "open failed\n", 12);
    syscall3(93, 1, 0, 0);
  }
  map.start = 0;
  map.length = ~0ul;
  map.extent_count = 128;
  if (syscall3(29, file, 0xc020660b, (long)&map) != 0) {
    write_all(2, "fiemap failed\n", 14);
    syscall3(93, 2, 0, 0);
  }

  u64 covered = 0;
  for (u32 index = 0; index < map.mapped_extents; ++index) {
    struct fiemap_extent *extent = &map.extents[index];
    if (extent->logical != covered || (extent->physical & 511) != 0) {
      write_all(2, "invalid extent\n", 15);
      syscall3(93, 3, 0, 0);
    }
    u64 remaining = file_size - covered;
    u64 length = extent->length < remaining ? extent->length : remaining;
    u64 blocks = (length + 511) >> 9;
    u64 source = partition_lba + (extent->physical >> 9);
    char line[96];
    char *out = append_text(line, "--raw-extent 0x");
    out = append_hex(out, destination + covered);
    out = append_text(out, ":0x");
    out = append_hex(out, source);
    out = append_text(out, ":0x");
    out = append_hex(out, blocks);
    *out++ = '\n';
    write_all(1, line, (u64)(out - line));
    covered += length;
  }
  if (covered != file_size) {
    write_all(2, "incomplete extents\n", 19);
    syscall3(93, 4, 0, 0);
  }
  syscall3(93, 0, 0, 0);
  for (;;) {}
}

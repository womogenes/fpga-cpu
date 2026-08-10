# start sequence of the bootloader
#
#
#include <smp.h>
#define DRAM_BASE 0x80000000

  .section .text.init
  .option norvc
  .globl _prog_start
_prog_start:
  // Every hart must reach the SMP-aware C boot path. The stock smp_pause
  // macro parks secondary harts before main(), which is correct only for the
  // original single-core FPGA system. Give each hart a private 4 KiB stack.
  csrr s1, mhartid
  slli s2, s1, 12
  // Keep boot stacks outside the Linux image, model, stage-2 code, and FDT.
  li sp, 0xbfd00000
  sub sp, sp, s2
  call main
1:
  wfi
  j 1b

  .section .dtb
  .globl _dtb
  .align 4, 0
_dtb:
#if __riscv_xlen == 64
  .incbin "cv64a6.dtb"
#else
  .incbin "cv32a6.dtb"
#endif

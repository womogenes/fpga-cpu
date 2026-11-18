MEMORY { ROM (rwx) : ORIGIN = 0x00000000, LENGTH = 0x1000 }
SECTIONS {
  .text : { *(.text.start) *(.text*) *(.rodata*) *(.srodata*) } > ROM
  .data : { *(.data*) *(.sdata*) *(.sbss*) *(.bss*) *(COMMON) } > ROM
}

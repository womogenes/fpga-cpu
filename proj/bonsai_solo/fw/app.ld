MEMORY { APP (rwx) : ORIGIN = 0x00001000, LENGTH = 0xc000 }
SECTIONS {
  .text : { *(.text.start) *(.text*) *(.rodata*) *(.srodata*) } > APP
  .data : { *(.data*) *(.sdata*) *(.sbss*) *(.bss*) *(COMMON) } > APP
}

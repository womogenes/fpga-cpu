INCLUDE generated/output_format.ld
ENTRY(_start)

__DYNAMIC = 0;

MEMORY
{
	sram        (rwx) : ORIGIN = 0x10000000, LENGTH = 0x00002000
	program     (rwx) : ORIGIN = 0x40000000, LENGTH = 0x08000000
	program_data (rw) : ORIGIN = 0x48000000, LENGTH = 0x37e00000
}

SECTIONS
{
	.text :
	{
		_ftext = .;
		*crt0*(.text)
		KEEP(*crt0*(.text))
		KEEP(*(.text.isr))
		*(.text .stub .text.* .gnu.linkonce.t.*)
		_etext = .;
	} > program

	.rodata :
	{
		. = ALIGN(8);
		_frodata = .;
		*(.rodata .rodata.* .gnu.linkonce.r.*)
		*(.rodata1)
		*(.srodata .srodata.*)
		*(.got .got.*)
		*(.toc .toc.*)
		FILL(0);
		. = ALIGN(8);
		_erodata = .;
	} > program

	.preinit_array :
	{
		. = ALIGN(8);
		PROVIDE_HIDDEN(__preinit_array_start = .);
		KEEP(*(SORT(.preinit_array.*)))
		KEEP(*(.preinit_array))
		PROVIDE_HIDDEN(__preinit_array_end = .);
	} > program

	.init_array :
	{
		. = ALIGN(8);
		PROVIDE_HIDDEN(__init_array_start = .);
		KEEP(*(SORT(.init_array.*)))
		KEEP(*(.init_array))
		KEEP(*(SORT(.ctors.*)))
		KEEP(*(.ctors))
		PROVIDE_HIDDEN(__init_array_end = .);
	} > program

	.fini_array :
	{
		. = ALIGN(8);
		PROVIDE_HIDDEN(__fini_array_start = .);
		KEEP(*(SORT(.fini_array.*)))
		KEEP(*(.fini_array))
		KEEP(*(SORT(.dtors.*)))
		KEEP(*(.dtors))
		PROVIDE_HIDDEN(__fini_array_end = .);
	} > program

	.data :
	{
		. = ALIGN(8);
		_fdata = .;
		*(.data .data.* .gnu.linkonce.d.*)
		*(.data1)
		_gp = ALIGN(16);
		*(.sdata .sdata.* .gnu.linkonce.s.*)
		FILL(0);
		. = ALIGN(8);
		_edata = .;
	} > sram AT > program

	.ddr_data :
	{
		. = ALIGN(8);
		_fddr_data = .;
		*(.ddr.data .ddr.data.*)
		FILL(0);
		. = ALIGN(8);
		_eddr_data = .;
	} > program_data AT > program

	.bss :
	{
		. = ALIGN(8);
		_fbss = .;
		*(.sbss .sbss.* .gnu.linkonce.sb.*)
		*(.scommon)
		*(.bss .bss.* .gnu.linkonce.b.*)
		*(COMMON)
		. = ALIGN(8);
		_ebss = .;
		_end = .;
	} > sram

	.ddr_bss (NOLOAD) :
	{
		. = ALIGN(8);
		_fddr_bss = .;
		*(.ddr.bss .ddr.bss.*)
		. = ALIGN(8);
		_eddr_bss = .;
	} > program_data
}

PROVIDE(_fstack = ORIGIN(sram) + LENGTH(sram));
PROVIDE(_fdata_rom = LOADADDR(.data));
PROVIDE(_edata_rom = LOADADDR(.data) + SIZEOF(.data));
PROVIDE(_fddr_data_rom = LOADADDR(.ddr_data));
PROVIDE(_eddr_data_rom = LOADADDR(.ddr_data) + SIZEOF(.ddr_data));
PROVIDE(_program_data_free = ALIGN(_eddr_bss, 16));
PROVIDE(_program_data_end = ORIGIN(program_data) + LENGTH(program_data));

ASSERT(_end <= ORIGIN(sram) + LENGTH(sram),
       "BRAM globals exceed the 8 KiB SRAM region")
ASSERT(_eddr_data_rom <= ORIGIN(program) + LENGTH(program),
       "Program image exceeds the 128 MiB program region")
ASSERT(_eddr_bss <= ORIGIN(program_data) + LENGTH(program_data),
       "DDR data exceeds the 894 MiB program-data region")

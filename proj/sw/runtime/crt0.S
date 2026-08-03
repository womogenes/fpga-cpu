.section .text.start
.global _start

_start:
	la sp, _fstack
	csrci mstatus, 8
	csrw mie, zero
	la t0, trap_entry
	csrw mtvec, t0

	/* Copy initialized data from its load address into SRAM. */
	la t0, _fdata
	la t1, _edata
	la t2, _fdata_rom
1:
	beq t0, t1, 2f
	lw t3, 0(t2)
	sw t3, 0(t0)
	addi t0, t0, 4
	addi t2, t2, 4
	j 1b

	/* Copy explicitly DDR-resident initialized data from the program image. */
2:
	la t0, _fddr_data
	la t1, _eddr_data
	la t2, _fddr_data_rom
3:
	beq t0, t1, 4f
	lw t3, 0(t2)
	sw t3, 0(t0)
	addi t0, t0, 4
	addi t2, t2, 4
	j 3b

	/* Clear BRAM BSS. */
4:
	la t0, _fbss
	la t1, _ebss
5:
	beq t0, t1, 6f
	sw zero, 0(t0)
	addi t0, t0, 4
	j 5b

	/* Clear explicitly DDR-resident BSS. */
6:
	la t0, _fddr_bss
	la t1, _eddr_bss
7:
	beq t0, t1, 8f
	sw zero, 0(t0)
	addi t0, t0, 4
	j 7b

8:
	call main
9:
	j 9b

trap_entry:
	j trap_entry

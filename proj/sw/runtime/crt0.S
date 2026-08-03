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

	/* Clear BSS. */
2:
	la t0, _fbss
	la t1, _ebss
3:
	beq t0, t1, 4f
	sw zero, 0(t0)
	addi t0, t0, 4
	j 3b

4:
	call main
5:
	j 5b

trap_entry:
	j trap_entry

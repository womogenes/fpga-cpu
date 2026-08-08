/* SPDX-License-Identifier: Apache-2.0 */
/* Bare-metal smoke test for boot release, AXI arbitration, and AMOs. */

    .section .text.start, "ax", @progbits
    .globl _start
    .option norvc

    .equ UART_BASE,   0x10000000
    .equ UART_THR,    0
    .equ UART_LSR,    20
    .equ READY_MAGIC, 0x534d5054
    .equ HART_COUNT,  4

_start:
    csrr    s1, mhartid
    li      sp, 0xbfc00000
    slli    t0, s1, 12
    sub     sp, sp, t0

    bnez    s1, secondary_delay
    la      t0, ready
    sw      zero, 0(t0)
    la      t0, uart_lock
    sw      zero, 0(t0)
    la      t0, online_count
    sw      zero, 0(t0)
    fence   rw, rw
    la      t0, ready
    li      t1, READY_MAGIC
    sw      t1, 0(t0)
    fence   rw, rw
    j       report_online

secondary_delay:
    /* Let hart 0 clear any stale magic retained across FPGA reloads. */
    rdcycle t0
    li      t1, 100000
1:
    rdcycle t2
    sub     t2, t2, t0
    bltu    t2, t1, 1b
    la      t0, ready
    li      t1, READY_MAGIC
2:
    lw      t2, 0(t0)
    bne     t2, t1, 2b

report_online:
    call    lock_uart
    la      t0, online_message
    addi    t1, s1, '0'
    sb      t1, 5(t0)
    mv      a0, t0
    call    uart_puts
    call    unlock_uart

    la      t0, online_count
    li      t1, 1
    amoadd.w.aqrl zero, t1, (t0)
3:
    lw      t1, 0(t0)
    li      t2, HART_COUNT
    bne     t1, t2, 3b
    bnez    s1, halt

    call    lock_uart
    la      a0, pass_message
    call    uart_puts
    call    unlock_uart

halt:
    wfi
    j       halt

lock_uart:
    la      t0, uart_lock
    li      t1, 1
4:
    amoswap.w.aq t2, t1, (t0)
    bnez    t2, 4b
    ret

unlock_uart:
    la      t0, uart_lock
    amoswap.w.rl zero, zero, (t0)
    ret

uart_putc:
    li      t0, UART_BASE
5:
    lbu     t1, UART_LSR(t0)
    andi    t1, t1, 0x20
    beqz    t1, 5b
    sb      a0, UART_THR(t0)
    ret

uart_puts:
    addi    sp, sp, -16
    sd      ra, 0(sp)
    sd      s0, 8(sp)
    mv      s0, a0
6:
    lbu     a0, 0(s0)
    beqz    a0, 7f
    call    uart_putc
    addi    s0, s0, 1
    j       6b
7:
    ld      ra, 0(sp)
    ld      s0, 8(sp)
    addi    sp, sp, 16
    ret

    .section .data, "aw", @progbits
online_message:
    .asciz  "hart 0 online\n"
pass_message:
    .asciz  "SMP PASS: 4 harts and AMOs operational\n"

    .section .bss, "aw", @nobits
    .balign 4
ready:
    .word   0
uart_lock:
    .word   0
online_count:
    .word   0

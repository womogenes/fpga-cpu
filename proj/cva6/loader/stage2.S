/* SPDX-License-Identifier: Apache-2.0 */
/* Fast, flow-controlled RAM loader entered by the stock CVA6 boot ROM. */

    .section .text.start, "ax", @progbits
    .globl _start
    .option norvc

    .equ UART_BASE,       0x10000000
    .equ UART_RBR,        0
    .equ UART_THR,        0
    .equ UART_FCR,        8
    .equ UART_MCR,        16
    .equ UART_LSR,        20
    .equ RELOC_BASE,      0xbf000000
    .equ IMAGE_DEST,      0x80000000
    .equ MODEL_DEST,      0xaf000000
    .equ SMP_MAILBOX,     0xbffffe00
    .equ SMP_RELEASE,     0x534d504c
    .equ MAX_IMAGE_SIZE,  0x2f000000
    .equ MAX_MODEL_SIZE,  0x0f000000
    /* Amortize the FTDI USB latency timer over a larger acknowledged burst.
       The receiver still polls the UART byte-by-byte, so this changes only
       ACK frequency and does not increase the FPGA FIFO requirement. */
    .equ BLOCK_SIZE,      16384

_start:
    mv      s11, a1
    la      t0, _start
    la      t1, _end
    li      t2, RELOC_BASE
1:
    beq     t0, t1, 2f
    lbu     t3, 0(t0)
    sb      t3, 0(t2)
    addi    t0, t0, 1
    addi    t2, t2, 1
    j       1b
2:
    fence.i
    la      t0, relocated_entry
    la      t1, _start
    sub     t0, t0, t1
    li      t1, RELOC_BASE
    add     t0, t0, t1
    jr      t0

relocated_entry:
    li      sp, 0xbff00000
    li      s0, UART_BASE

    /* The boot ROM used this word to release harts into stage 2. Clear it
       before a delayed secondary performs its first safe DDR read; launch
       publishes the completed payload with the same release protocol. */
    li      t0, SMP_MAILBOX
    sd      zero, 16(t0)
    fence   rw, rw

    /* Preserve the existing 3 MBaud divisor and enable the 64-byte FIFOs. */
    li      t0, 0xe7
    sb      t0, UART_FCR(s0)
    sb      zero, UART_MCR(s0)
    la      a0, ready_message
    call    uart_puts

handshake:
    la      t0, handshake_magic
    li      t1, 4
1:
    call    uart_getc
    lbu     t2, 0(t0)
    bne     a0, t2, handshake
    addi    t0, t0, 1
    addi    t1, t1, -1
    bnez    t1, 1b
    la      a0, ack_message
    call    uart_puts

    /* The host sends little-endian Linux and optional model lengths. */
    call    uart_get_u32
    mv      s4, a0
    call    uart_get_u32
    mv      s6, a0
    beqz    s4, loader_error
    li      t0, MAX_IMAGE_SIZE
    bgtu    s4, t0, loader_error
    li      t0, MAX_MODEL_SIZE
    bgtu    s6, t0, loader_error
    bnez    s6, model_header_ready
    li      t0, MODEL_DEST
    sw      zero, 0(t0)
model_header_ready:

    li      s3, 2166136261
    li      s5, 16777619
    mv      s1, s4
    li      s2, IMAGE_DEST
    call    receive_segment
    mv      s1, s6
    li      s2, MODEL_DEST
    call    receive_segment

    /* Block ACKs plus this target-side receive checksum validate transport.
       Avoid rereading hundreds of MiB here: uncached DDR verification took
       several minutes and left the host unable to capture early boot logs. */
    fence   rw, rw
    la      a0, done_message
    call    uart_puts
    call    uart_getc
    li      t0, 'C'
    bne     a0, t0, loader_error
    mv      t0, s3
    li      t1, 4
4:
    andi    a0, t0, 0xff
    call    uart_putc
    srli    t0, t0, 8
    addi    t1, t1, -1
    bnez    t1, 4b
    call    uart_getc
    li      t0, 'G'
    bne     a0, t0, loader_error

launch:
    /* Publish the payload to boot-ROM parked secondary harts. */
    li      t0, SMP_MAILBOX
    li      t1, IMAGE_DEST
    sd      t1, 0(t0)
    sd      s11, 8(t0)
    fence   rw, rw
    li      t1, SMP_RELEASE
    sd      t1, 16(t0)
    fence   rw, rw
    fence.i
    li      a0, 0
    mv      a1, s11
    li      t0, IMAGE_DEST
    jr      t0

loader_error:
    li      a0, 'E'
    call    uart_putc
7:
    wfi
    j       7b

uart_get_u32:
    addi    sp, sp, -16
    sd      ra, 0(sp)
    li      t0, 0
    li      t1, 0
3:
    call    uart_getc
    slli    t2, t1, 3
    sll     a0, a0, t2
    or      t0, t0, a0
    addi    t1, t1, 1
    li      t2, 4
    bne     t1, t2, 3b
    mv      a0, t0
    ld      ra, 0(sp)
    addi    sp, sp, 16
    ret

receive_segment:
    addi    sp, sp, -16
    sd      ra, 0(sp)
block_loop:
    beqz    s1, segment_done
    li      t0, BLOCK_SIZE
    bgeu    s1, t0, block_size_ready
    mv      t0, s1
block_size_ready:
    mv      t1, t0
byte_loop:
    call    uart_getc
    sb      a0, 0(s2)
    addi    s2, s2, 1
    xor     s3, s3, a0
    mul     s3, s3, s5
    addi    t1, t1, -1
    bnez    t1, byte_loop
    sub     s1, s1, t0
    li      a0, 'A'
    call    uart_putc
    j       block_loop
segment_done:
    ld      ra, 0(sp)
    addi    sp, sp, 16
    ret

uart_getc:
8:
    lbu     t6, UART_LSR(s0)
    andi    t6, t6, 1
    beqz    t6, 8b
    lbu     a0, UART_RBR(s0)
    ret

uart_putc:
9:
    lbu     t6, UART_LSR(s0)
    andi    t6, t6, 0x20
    beqz    t6, 9b
    sb      a0, UART_THR(s0)
    ret

uart_puts:
    addi    sp, sp, -16
    sd      ra, 0(sp)
    mv      t5, a0
10:
    lbu     a0, 0(t5)
    beqz    a0, 11f
    call    uart_putc
    addi    t5, t5, 1
    j       10b
11:
    ld      ra, 0(sp)
    addi    sp, sp, 16
    ret

ready_message:
    .asciz  "S2READY\n"
handshake_magic:
    .ascii  "S2UP"
ack_message:
    .asciz  "S2ACK\n"
done_message:
    .asciz  "\nS2DONE\n"

    .globl _end
_end:

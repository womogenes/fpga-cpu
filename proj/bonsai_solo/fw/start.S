    .section .text.start
    .globl _start
_start:
    li sp, 0x10000
    call monitor_main
1:  j 1b

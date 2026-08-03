#ifndef PROJECT_RUNTIME_H
#define PROJECT_RUNTIME_H

#include <system.h>

#define DDR_DATA __attribute__((section(".ddr.data")))
#define DDR_BSS __attribute__((section(".ddr.bss")))
#define DDR_NOINIT __attribute__((section(".ddr.noinit")))

void uart_init(void);
int uart_write(const char* data, unsigned int length);
int puts(const char* string);

#endif

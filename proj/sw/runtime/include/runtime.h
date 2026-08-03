#ifndef PROJECT_RUNTIME_H
#define PROJECT_RUNTIME_H

#include <system.h>

#define DDR_DATA __attribute__((section(".ddr.data")))
#define DDR_BSS __attribute__((section(".ddr.bss")))

void uart_init(void);
int puts(const char* string);

#endif

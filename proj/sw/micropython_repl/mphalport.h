#ifndef MICROPY_INCLUDED_FPGA_CPU_MPHALPORT_H
#define MICROPY_INCLUDED_FPGA_CPU_MPHALPORT_H

#include <stddef.h>
#include <stdint.h>

#include "shared/runtime/interrupt_char.h"

void mp_hal_event_poll_hook(void);
void mp_hal_get_random(size_t length, uint8_t* buffer);

#endif

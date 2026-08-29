// Copyright OpenHW Group contributors.
// Licensed under the Apache License, Version 2.0.

#include "sd.h"
#include "uart.h"
#include <stdint.h>

#define SDC_BASE 0x20000000UL
#define SDC_CMD_CC   0x01u
#define SDC_CMD_ERR  0x02u
#define SDC_DAT_DONE 0x01u
#define SDC_DAT_ERR  0x02u

struct sdc_regs {
    volatile uint32_t argument;
    volatile uint32_t command;
    volatile uint32_t response[4];
    volatile uint32_t data_timeout;
    volatile uint32_t control;
    volatile uint32_t cmd_timeout;
    volatile uint32_t clock_divider;
    volatile uint32_t software_reset;
    volatile uint32_t power_control;
    volatile uint32_t capability;
    volatile uint32_t cmd_int_status;
    volatile uint32_t cmd_int_enable;
    volatile uint32_t dat_int_status;
    volatile uint32_t dat_int_enable;
    volatile uint32_t block_size;
    volatile uint32_t block_count;
    volatile uint32_t card_detect;
    volatile uint32_t reserved[4];
    volatile uint64_t dma_address;
};

static struct sdc_regs *const sdc = (struct sdc_regs *)SDC_BASE;

struct cmd_trace {
    uint32_t opcode;
    uint32_t status;
    uint32_t response;
};

static inline uint64_t read_cycle(void)
{
    uint64_t cycle;
    __asm__ volatile ("csrr %0, cycle" : "=r" (cycle));
    return cycle;
}

static void delay_cycles(uint64_t cycles)
{
    uint64_t start = read_cycle();
    while (read_cycle() - start < cycles)
        __asm__ volatile ("nop");
}

static int init_error(int error, const struct cmd_trace *trace)
{
    print_uart("native SD init error: code/status/opcode/response ");
    print_uart_int((uint32_t)error);
    print_uart("/");
    print_uart_int(trace->status);
    print_uart("/");
    print_uart_int(trace->opcode);
    print_uart("/");
    print_uart_int(trace->response);
    print_uart("\r\n");
    return error;
}

static inline void dma_barrier(void)
{
    __asm__ volatile ("fence rw, rw" ::: "memory");
}

static int wait_not_reset(void)
{
    for (uint32_t timeout = 0; timeout < 10000000; timeout++)
        if (sdc->software_reset == 0)
            return 0;
    return -1;
}

static int send_cmd(struct cmd_trace *trace, uint32_t opcode,
                    uint32_t argument, uint32_t flags, uint32_t *response)
{
    trace->opcode = opcode;
    trace->status = 0;
    trace->response = 0;
    sdc->cmd_int_status = 0;
    sdc->command = (opcode << 8) | flags;
    sdc->cmd_timeout = 0x00ffffff;
    sdc->argument = argument;

    for (uint32_t timeout = 0; timeout < 50000000; timeout++) {
        uint32_t status = sdc->cmd_int_status;
        if (!status)
            continue;
        trace->status = status;
        sdc->cmd_int_status = 0;
        if (wait_not_reset() || status != SDC_CMD_CC)
            return -1;
        trace->response = sdc->response[0];
        if (response)
            *response = trace->response;
        return 0;
    }
    return -1;
}

static int app_cmd(struct cmd_trace *trace, uint32_t rca, uint32_t opcode,
                   uint32_t argument, uint32_t flags, uint32_t *response)
{
    if (send_cmd(trace, 55, rca << 16, 0x19, 0))
        return -1;
    return send_cmd(trace, opcode, argument, flags, response);
}

int init_sd(void)
{
    struct cmd_trace trace = {0};
    uint32_t card_rca;
    uint32_t response;

    print_uart("initializing native SD...\r\n");

    /*
     * An SD card remains in SPI mode until its power is removed. FPGA
     * reconfiguration alone therefore cannot switch a card previously used by
     * the SPI controller back to native mode. Bit 1 controls the Genesys 2 SD
     * slot power switch: one cuts power and zero enables it. The one-second
     * intervals duplicate the sequence validated on the board and leave ample
     * discharge and power-ramp margin.
     */
    sdc->control = 2;
    delay_cycles(CLOCK_FREQUENCY);
    sdc->control = 0;
    delay_cycles(CLOCK_FREQUENCY);

    sdc->clock_divider = 116; /* 94 MHz / (2 * 117) = 402 kHz. */
    sdc->software_reset = 1;
    for (uint32_t timeout = 0; timeout < 1000000; timeout++)
        if (sdc->software_reset & 1)
            break;
    sdc->software_reset = 0;
    if (wait_not_reset())
        return init_error(SD_INIT_ERROR_CMD0, &trace);
    sdc->control = 0;
    /* Supply well over the required 74 idle clocks before CMD0. */
    delay_cycles(CLOCK_FREQUENCY / 100);
    sdc->cmd_int_enable = 0;
    sdc->dat_int_enable = 0;
    sdc->cmd_int_status = 0;
    sdc->dat_int_status = 0;

    if (send_cmd(&trace, 0, 0, 0, 0))
        return init_error(SD_INIT_ERROR_CMD0, &trace);
    if (send_cmd(&trace, 8, 0x1aa, 0x19, &response) ||
        (response & 0xfff) != 0x1aa)
        return init_error(SD_INIT_ERROR_CMD8, &trace);

    card_rca = 0;
    for (uint32_t retry = 0; retry < 10000; retry++) {
        if (app_cmd(&trace, card_rca, 41, 0x40300000, 0x01, &response))
            return init_error(SD_INIT_ERROR_ACMD41, &trace);
        if (response & 0x80000000)
            break;
        if (retry == 9999)
            return init_error(SD_INIT_ERROR_ACMD41, &trace);
    }
    /* The native boot path intentionally supports SDHC/SDXC block addressing. */
    if (!(response & 0x40000000))
        return init_error(-8, &trace);

    if (send_cmd(&trace, 2, 0, 0x0a, 0))
        return init_error(-4, &trace);
    if (send_cmd(&trace, 3, 0, 0x19, &response))
        return init_error(-5, &trace);
    card_rca = response >> 16;
    if (send_cmd(&trace, 7, card_rca << 16, 0x1d, 0))
        return init_error(-6, &trace);
    if (app_cmd(&trace, card_rca, 6, 2, 0x19, 0))
        return init_error(-7, &trace);

    sdc->control = 1;       /* Four-bit bus. */
    sdc->clock_divider = 1; /* 23.5 MHz default-speed SD clock. */
    return 0;
}

int sd_copy(void *dst, uint32_t src_lba, uint32_t blocks)
{
    struct cmd_trace trace = {0};
    uint32_t opcode = blocks == 1 ? 17 : 18;
    uint32_t status;

    if (!blocks || blocks > 0x10000)
        return -1;

    dma_barrier();
    sdc->dma_address = (uintptr_t)dst;
    sdc->block_size = 511;
    sdc->block_count = blocks - 1;
    sdc->data_timeout = 0;
    sdc->dat_int_status = 0;
    if (send_cmd(&trace, opcode, src_lba, 0x39, 0))
        return SD_COPY_ERROR_CMD18;

    for (uint32_t timeout = 0; timeout < 0xffffffffu; timeout++) {
        status = sdc->dat_int_status;
        if (status)
            break;
    }
    sdc->dat_int_status = 0;
    if (wait_not_reset() || status != SDC_DAT_DONE)
        return SD_COPY_ERROR_CMD18;
    if (blocks > 1 &&
        send_cmd(&trace, SD_CMD_STOP_TRANSMISSION, 0, 0x1d, 0))
        return SD_COPY_ERROR_CMD18;
    dma_barrier();
    return 0;
}

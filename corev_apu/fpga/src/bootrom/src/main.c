// Copyright OpenHW Group contributors.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#include "uart.h"
#include "spi.h"
#include "sd.h"
#include "gpt.h"

#define SECOND_CYCLES   CLOCK_FREQUENCY
#define WAIT_SECONDS    (10)
#define SMP_MAILBOX_ADDRESS 0xbffffe00UL
#define SMP_RELEASE_MAGIC   0x534d504cUL

struct smp_mailbox {
    volatile uintptr_t entry;
    volatile uintptr_t dtb;
    volatile uintptr_t release;
};

static inline uintptr_t get_hart_id() {
    uintptr_t hart_id;
    __asm__ volatile ("csrr %0, mhartid" : "=r" (hart_id));
    return hart_id;
}

static inline uintptr_t get_cycle_count() {
    uintptr_t cycle;
    __asm__ volatile ("csrr %0, cycle" : "=r" (cycle));
    return cycle;
}

__attribute__((noreturn)) static void jump_to_payload(uintptr_t hart_id, uintptr_t entry, uintptr_t dtb) {
    __asm__ volatile(
        "fence rw, rw\n"
        "fence.i\n"
        "mv a0, %0\n"
        "mv a1, %1\n"
        "jr %2"
        :
        : "r" (hart_id), "r" (dtb), "r" (entry)
        : "a0", "a1", "memory");
    __builtin_unreachable();
}

__attribute__((noreturn)) static void wait_for_boot_hart(uintptr_t hart_id) {
    volatile struct smp_mailbox *mailbox = (volatile struct smp_mailbox *)SMP_MAILBOX_ADDRESS;
    // Do not issue a DDR request while the MIG is still calibrating: an AXI
    // read accepted during that window can leave a secondary hart stalled
    // forever. Hart 0 and the stage-2 loader are ready well within one second.
    uintptr_t start = get_cycle_count();
    while (get_cycle_count() - start < CLOCK_FREQUENCY);
    while (mailbox->release != SMP_RELEASE_MAGIC) {
        // Avoid letting parked harts saturate the uncached DDR fabric while
        // hart 0 is receiving the payload. This is about a 10 us poll period.
        for (volatile unsigned int delay = 0; delay < 256; delay++)
            __asm__ volatile ("nop");
    }
    jump_to_payload(hart_id, mailbox->entry, mailbox->dtb);
}

static void release_secondary_harts(uintptr_t entry, uintptr_t dtb) {
    volatile struct smp_mailbox *mailbox = (volatile struct smp_mailbox *)SMP_MAILBOX_ADDRESS;
    mailbox->entry = entry;
    mailbox->dtb = dtb;
    __asm__ volatile ("fence rw, rw" ::: "memory");
    mailbox->release = SMP_RELEASE_MAGIC;
    __asm__ volatile ("fence rw, rw" ::: "memory");
}

int update(uint8_t *dest)
{
    int i;
    uint32_t size = 0;

    print_uart("receiving boot image\r\nsize: ");
    for(i = 0; i < sizeof(uint32_t); i++) {
        while(!read_serial(&((uint8_t *) &size)[i]));
    }

    print_uart_int(size);
    print_uart("\r\nreceiving ");

    for(i = 0; i < size; i++) {
        while(!read_serial(&dest[i]));

        if(i % (size >> 4) == 0) {
            print_uart(".");
        }
    }

    print_uart(" done!\r\n");
    return 0;
}

int main()
{
    int i, ret = 0;
    uint8_t uart_res = 0;
    uintptr_t start;


    uintptr_t hart_id = get_hart_id();
    volatile struct smp_mailbox *mailbox = (volatile struct smp_mailbox *)SMP_MAILBOX_ADDRESS;
    if (hart_id != 0)
        wait_for_boot_hart(hart_id);
    mailbox->release = 0;
    __asm__ volatile ("fence rw, rw" ::: "memory");

    #ifndef PLAT_AGILEX
    init_uart(UART_CLOCK_FREQUENCY, UART_BITRATE); //not needed in intel setup as UART IP is already configured via HW
    #endif 
    print_uart("Hello World!\r\n");

    // See if we should enter update mode
    print_uart("Hit any key to enter update mode ");
    for(i = 0; i < WAIT_SECONDS && !ret; i++) {
        print_uart(".");
        start = get_cycle_count();
        while(get_cycle_count() - start < SECOND_CYCLES) {
            ret = read_serial(&uart_res);
            if(ret) {
                break;
            }
        }
    }

    int res;
    if(ret) {
        print_uart(" updating!\r\n");
        res = update((uint8_t *)0x80000000UL);
    } else {
        print_uart(" booting!\r\n");
        #ifndef PLAT_AGILEX
        res = gpt_find_boot_partition((uint8_t *)0x80000000UL, 2 * 16384); 
        #else 
            int start_block_fw_payload  = 0x32800; //payload at 100MB
            print_uart("I am Agilex 7! \r\n");

            print_uart("Loading fw_payload into memory address 0x80000000 \n");
            for (uint64_t i = 0; i < 15000; i++){
                res = sd_copy_mmc((uint8_t *)0x80000000UL + (i * 0x200), start_block_fw_payload + i, 1); // for now hardcoded, need to develop the code to find the file in the SD card

                if (res)
                {
                    print_uart("TRANSFER ERROR\n");
                    return res;
                }
		    }
        #endif 
    }

    if (res == 0)
    {
        uintptr_t entry = 0x80000000UL;
        uintptr_t dtb;
        __asm__ volatile ("la %0, _dtb" : "=r" (dtb));
        release_secondary_harts(entry, dtb);
        jump_to_payload(0, entry, dtb);
    }

    while (1)
    {
        // do nothing
    }
}

void handle_trap(void)
{
    // print_uart("trap\r\n");
}

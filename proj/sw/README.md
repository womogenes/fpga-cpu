# Software interface

Applications are freestanding C programs. They do not use a hosted C library or operating system, so their APIs come from two places:

| Source         | Included header   | What it provides                                       |
| -------------- | ----------------- | ------------------------------------------------------ |
| `sw/runtime`   | `runtime.h`       | Project-owned UART and cache helpers                   |
| Gateware build | `generated/csr.h` | Inline accessors for hardware control/status registers |
| Gateware build | `generated/mem.h` | Memory-region addresses and sizes                      |
| Gateware build | `generated/soc.h` | SoC configuration constants such as cache sizes        |

The generated headers live under `build/fpga_soc/software/include/generated/`. `make gateware` recreates them from the hardware defined in `hdl/soc.py`; applications should include them but must not edit them.

## Application memory layout

The linker divides DDR into non-overlapping instruction, application-data, and framebuffer regions while retaining the small, fast BRAM SRAM:

| Region                  | Address range             | Capacity | Default contents                                        |
| ----------------------- | ------------------------- | -------- | ------------------------------------------------------- |
| BRAM SRAM               | `0x10000000`–`0x10001fff` | 8 KiB    | Ordinary `.data`, `.bss`, and stack                     |
| Program                 | `0x40000000`–`0x47ffffff` | 128 MiB  | Startup, instructions, constants, and initial values    |
| DDR program data        | `0x48000000`–`0x7fdfffff` | 894 MiB  | Explicit large initialized and zero-initialized objects |
| Framebuffer reservation | `0x7fe00000`–`0x7fffffff` | 2 MiB    | 1280x720 RGB565 image plus alignment padding            |

Normal globals continue to use BRAM, so small frequently accessed state and the stack remain fast. They must collectively fit in 8 KiB. Put large objects in DDR with the attributes from `runtime.h`:

```c
static uint8_t grid[1280 * 720] DDR_BSS;
static uint32_t palette[] DDR_DATA = {0x00000000, 0x0000ffff};
static uint32_t scratch[1280 * 720] DDR_NOINIT;
```

`DDR_BSS` occupies no space in the uploaded binary and is cleared by `crt0.S`. `DDR_DATA` stores its initial value compactly in the program image; `crt0.S` copies it into the DDR data region before calling `main()`. The linker exports `_program_data_free` and `_program_data_end` for a future heap or manual arena after the explicitly allocated DDR objects.

`DDR_NOINIT` also occupies no space in the uploaded binary, but startup does not
clear it. Use it for large scratch buffers whose contents are always overwritten
before being read.

## Shared runtime helpers

These functions are declared below `sw/runtime/include/` and implemented in `sw/runtime/runtime.c` or directly in the header.

### `uart_init()`

Provides an initialization hook for application code. It is currently empty because the LiteX BIOS configures the UART divisor before loading and starting the application.

### `puts(const char *string)`

Writes a null-terminated string to the UART, followed by carriage return and newline. It polls `uart_txfull_read()` before writing each character with `uart_rxtx_write()`, so it blocks when the transmit FIFO is full. This is the project's small freestanding implementation, not the hosted libc version; it does not provide formatting.

### `flush_cpu_dcache()`

Issues VexRiscv's data-cache flush instruction. This writes dirty CPU D-cache lines toward the next memory level and prevents the compiler from moving memory operations across the instruction. It compiles to a no-op when the generated SoC configuration does not define `CONFIG_CPU_HAS_DCACHE`.

### `flush_l2_cache()`

Reads twice the configured L2 capacity from main RAM. Those reads evict the current LiteDRAM L2 lines, forcing dirty data onward to DDR. It compiles to a no-op when `CONFIG_L2_SIZE` is absent.

The video DMA reads DDR through a dedicated port and is not coherent with the CPU D-cache or LiteDRAM L2 cache. After drawing through the CPU, flush both levels in this order:

```c
flush_cpu_dcache();
flush_l2_cache();
```

Without both operations, the CPU can see its new pixels while HDMI continues to read older data from DDR.

## Generated CSR accessors

LiteX converts each exposed control/status register into a `static inline` function in `generated/csr.h`. The function name is assembled from the module, register, and operation. For example:

```c
video_framebuffer_vtg_enable_write(1);
```

writes `1` to the `enable` register on the `video_framebuffer_vtg` module. These functions use the project's `csr_read_simple()` and `csr_write_simple()` helpers to perform volatile 32-bit MMIO accesses in the CSR region. They are header functions, so no extra library is linked.

### Framebuffer DMA registers

The framebuffer DMA continuously reads RGB565 pixels from DDR and feeds the HDMI pixel pipeline.

| Accessor stem                  | Meaning                                                                                                    |
| ------------------------------ | ---------------------------------------------------------------------------------------------------------- |
| `video_framebuffer_dma_base`   | DDR byte address at which the image begins; defaults to `VIDEO_FRAMEBUFFER_BASE` (`0x7fe00000`)            |
| `video_framebuffer_dma_length` | Number of bytes read per pass; defaults to 1,843,200 bytes for 1280 x 720 RGB565                           |
| `video_framebuffer_dma_enable` | Writing `1` starts reads; writing `0` stops and resets DMA progress                                        |
| `video_framebuffer_dma_done`   | Reads as `1` after a non-looping transfer reaches its end                                                  |
| `video_framebuffer_dma_loop`   | Writing `1` makes the DMA wrap to offset zero after each frame; the reset default is `1`                   |
| `video_framebuffer_dma_offset` | Current progress in native DRAM words; useful for checking that reads are advancing, but not a byte offset |

Each writable register has `_write(value)` and `_read()` accessors. Status-only registers such as `done` and `offset` only have `_read()`.

### Video timing generator registers

The video timing generator (VTG) creates data-enable, horizontal-sync, and vertical-sync timing. The gateware initializes these registers for 1280x720 at 60 Hz.

| Accessor stem                       | Meaning                                                                       |
| ----------------------------------- | ----------------------------------------------------------------------------- |
| `video_framebuffer_vtg_enable`      | Writing `1` starts timing generation; writing `0` holds the generator reset   |
| `video_framebuffer_vtg_hres`        | Number of visible pixels per line                                             |
| `video_framebuffer_vtg_hsync_start` | Horizontal counter at which the sync pulse starts                             |
| `video_framebuffer_vtg_hsync_end`   | Horizontal counter at which the sync pulse ends                               |
| `video_framebuffer_vtg_hscan`       | Final horizontal counter value; the complete line is `hscan + 1` pixel clocks |
| `video_framebuffer_vtg_vres`        | Number of visible lines per frame                                             |
| `video_framebuffer_vtg_vsync_start` | Vertical counter at which the sync pulse starts                               |
| `video_framebuffer_vtg_vsync_end`   | Vertical counter at which the sync pulse ends                                 |
| `video_framebuffer_vtg_vscan`       | Final vertical counter value; the complete frame is `vscan + 1` lines         |

Disable the VTG before changing several timing registers, then re-enable it after the complete mode has been written. Partially updated timing can produce an invalid HDMI signal.

## Generated memory and configuration symbols

`generated/mem.h` describes the synthesized address map. Common symbols are:

| Symbol                   | Current value | Meaning                                   |
| ------------------------ | ------------- | ----------------------------------------- |
| `MAIN_RAM_BASE`          | `0x40000000`  | Start of the 1 GiB DDR main-memory region |
| `VIDEO_FRAMEBUFFER_BASE` | `0x7fe00000`  | Start of the reserved framebuffer region  |
| `VIDEO_FRAMEBUFFER_SIZE` | `0x001c2000`  | 1,843,200-byte visible RGB565 image       |

`generated/soc.h` provides hardware-dependent feature constants including `CONFIG_CPU_HAS_DCACHE`, `CONFIG_L2_SIZE`, framebuffer resolution, and pixel depth. These values can change when the gateware configuration changes.

## Typical framebuffer sequence

```c
#include <generated/csr.h>
#include <generated/mem.h>
#include <runtime.h>

volatile uint16_t *framebuffer =
    (volatile uint16_t *)VIDEO_FRAMEBUFFER_BASE;

/* Write 1280 x 720 RGB565 pixels through framebuffer here. */

flush_cpu_dcache();
flush_l2_cache();
video_framebuffer_vtg_enable_write(1);
video_framebuffer_dma_enable_write(1);
```

The `_read()` forms can confirm that control writes took effect. `video_framebuffer_dma_offset_read()` changing from zero confirms that the DMA is consuming the framebuffer. It does not prove that the display selected the framebuffer input; physical SW0 must also be in the framebuffer position.

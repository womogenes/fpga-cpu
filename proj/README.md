# Genesys 2 LiteX CPU

(this is all codex btw if it wasn't already obvious)

## Project structure

```text
proj/
├── hdl/                    LiteX/Migen SoC definitions
│   └── diagnostics/        Standalone hardware checks
├── sw/                     RISC-V software and minimal local runtime
├── tools/                  Build and programming entry points
├── requirements.txt        Pinned LiteX dependencies
├── .venv/                  Downloaded dependencies (ignored)
└── build/                  Generated output (ignored)
```

The project does not depend on a sibling LiteX checkout.

## Setup

Create the local Python environment once:

```sh
make setup
```

## Hardware

- VexRiscv at 100 MHz
- 1 GiB DDR3 at `0x40000000`
- 8 KiB LiteDRAM L2 cache
- Cached VexRiscv instruction and data buses
- 1280x720 at 60 Hz HDMI output
- RGB565 framebuffer at `0x7fe00000`
- LiteX BIOS and UART serial boot at 115200 baud

Physical SW0 selects the HDMI source: down shows hardware color bars and up
shows the DDR framebuffer. The Genesys 2 platform's switch resource numbering
is reversed, so physical SW0 is `user_sw[7]` in the gateware.

The framebuffer uses 1,843,200 bytes inside a 2 MiB reserved region at the top
of DDR3. The video reader has a dedicated native LiteDRAM DMA port.

## Build

From this directory:

```sh
make gateware
```

The bitstream is written to:

```text
build/genesys2_mvp/gateware/digilent_genesys2.bit
```

The small CPU/DDR-independent HDMI fallback can be rebuilt with:

```sh
make diagnostic
```

Its stable convenience path is `build/known_good_hdmi.bit`.

## Build the framebuffer demo

The hardware build generates the SoC-specific headers needed by software:

```sh
make framebuffer-demo
```

The demo is freestanding and uses the local `sw/runtime` startup, UART, CSR,
and cache helpers. It does not depend on the sibling LiteX checkout when
compiling; it only needs the generated headers in `build/genesys2_mvp` and the
system `riscv64-elf` toolchain.

Upload it after loading the bitstream:

```sh
make upload
```

The UART defaults to `/dev/ttyUSB0` at 115200 baud. Override it with, for
example, `LITEX_UART=/dev/ttyUSB1 make upload`.

Framebuffer writers must call `flush_cpu_dcache()` and `flush_l2_cache()` after
drawing so the independent video DMA sees the new pixels.

## Load the FPGA

With the Genesys 2 attached locally:

```sh
make load
```

The loader uses `openFPGALoader -b genesys2 --freq 1000000`. Loading is
volatile and does not write the board's configuration flash.

## Remote workflow

Mirror this project, including generated gateware and C binaries, to thinkmax:

```sh
make sync
```

`.venv`, editor state, and Python caches are not copied. On the remote host,
run `make setup` once, then use `make flash` and `make upload`. Set
`REMOTE_PROJECT=user@host:path/ make sync` to use another destination.

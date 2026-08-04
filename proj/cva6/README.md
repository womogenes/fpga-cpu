# CVA6 Genesys 2

This directory contains the OpenHW Group CVA6 Genesys 2 platform running one 64-bit hart at 100 MHz, a 1280×720 HDMI terminal and software-video output, an FPGA matrix-vector accelerator, and a Buildroot Linux workstation that can boot from a writable RAM filesystem today and a persistent SD-card filesystem later. The upstream repositories are pinned as Git submodules; project-owned patches inside CORE-V-APU capture the CPU timing configuration, pipelined floating-point unit, exact 3 MBaud UART reference, and reliable divisor-2 receiver sampling.

## Layout

- `hardware/` contains the CVA6 core, CORE-V-APU, Genesys 2 DDR3 controller, SD-card controller, UART, JTAG, boot ROM, and FPGA build flow.
- `sdk/` contains Buildroot, OpenSBI, U-Boot, Linux, the initramfs, and SD-card image generation.
- `rtl/` contains the HDMI terminal, UART mirror, TMDS pipeline, and software-video plane maintained by this project.
- `loader/` contains the relocatable stage-2 RAM loader used for fast, checked Linux uploads.
- `software/` contains programs compiled into the Linux filesystem by the post-build hook.
- `build/` contains parent-owned generated output and is ignored by Git.

## Initialize

After cloning the parent repository, initialize CVA6 and every pinned dependency with:

```sh
make -C proj/cva6 init
```

## Build gateware

Build the 64-bit `cv64a6_imafdc_sv39` Genesys 2 FPGA design with:

```sh
make -C proj/cva6 gateware
```

The wrapped flow copies the bitstream and reports into `build/gateware/` and reuses the bitstream when all hashed inputs are unchanged. Never start this target while any other Vivado process is running on the machine.

## Build Linux image

Build the 64-bit Genesys 2 toolchain, OpenSBI, U-Boot, Linux kernel, persistent ext4 filesystem, and complete SD-card image with:

```sh
make -C proj/cva6 image
```

The resulting SD-card image is written beneath `build/sdk/`. The first build downloads and compiles a complete RISC-V Linux toolchain and can take a long time. The image includes Bash, Python, NumPy/OpenBLAS, Vim, tmux, TinyCC, games, the 160×45 terminal configuration, and the software-video demos. IPython is intentionally omitted because its size and startup cost do not justify carrying it on the current single-hart system.

## Boot Linux from RAM

An SD card is not required for the current development flow. Build an OpenSBI payload containing Linux and the complete initramfs with:

```sh
make -C proj/cva6 ram-image
```

The result is `build/sdk/ram-linux.bin`. Program the FPGA, reset it, and start the boot-ROM upload during its ten-second prompt with:

```sh
make -C proj/cva6 upload-ram SERIAL=/dev/ttyUSB0
```

The upload target first sends a 770-byte relocatable stage-2 loader through the stock boot ROM using drained 32-byte writes. Stage 2 relocates near the top of DDR, enables the UART's 64-byte FIFO, receives Linux directly at `0x80000000` with an acknowledgement after each 4096-byte block, and re-reads every stored byte for an end-to-end FNV-1a memory checksum before launch. The UART remains at 3 MBaud throughout and sustains about 232 KiB/s. The current 250.7 MiB uncompressed workstation takes about 18½ minutes to transfer; keeping the initramfs uncompressed avoids a CVA6 failure observed while unpacking both gzip and Zstandard streams. The command enters the serial console after transfer. The root filesystem is writable and behaves like a normal Linux filesystem, but changes disappear at power-off until the SD-card image is used.

For the normal remote-board workflow, build gateware and Linux locally and then deploy, program, upload, and attach in one command with `make -C proj/cva6 remote-boot`. It copies only the board-ready bitstream, Linux image, stage-2 loader, and uploader to `thinkmax:~/fpga/fpga-cpu/build/cva6/`; override `REMOTE_HOST`, `REMOTE_CVA6_DIR`, or `REMOTE_SERIAL` when needed. The RAM-boot device tree exposes 768 MiB to Linux and reserves the following 224 MiB for optional model staging; the persistent SD-card image exposes the full 1 GiB because its model lives in the filesystem instead.

## Serial console

The default system runs the UART at exactly 3,000,000 baud, the maximum supported by the Genesys 2 FT232R bridge. The UART baud generator uses a 96 MHz effective reference derived by suppressing one clock enable out of every 25 cycles of the 100 MHz system clock; divisor 2 therefore produces exactly 3 MBaud without adding another clock domain. After loading the FPGA and inserting the generated SD card, connect with:

```sh
make -C proj/cva6 monitor SERIAL=/dev/ttyUSB0
```

The monitor target uses Python pyserial when available and falls back to picocom. The exact `/dev/ttyUSB*` device can vary. The generated Linux system logs in as `root` with no password by default. If the USB-UART adapter cannot sustain the maximum rate, the gateware, boot ROM, U-Boot, Linux command line, device tree, and monitor rate must be changed together.

## Linux workstation

The image boots to Bash in `/root` and provides both `python` and `python3`, NumPy, Vim, tmux, make, target binutils, and TinyCC as `cc`. NumPy is linked to single-threaded OpenBLAS built for `rv64imafdc`, so array and matrix operations use the CVA6 floating-point unit without thread overhead on the single-hart system. Run `numpy-bench` to print the NumPy/OpenBLAS configuration and measure representative float32 kernels. IPython and its dependency stack are not installed.

Run `workstation-check` to verify the core tools, NumPy, the standalone LFM tokenizer, and a native C smoke test directly on CVA6.

Games are available as `snake`, `tetris`, and `ascii-invaders`. Editable C examples live in `/root/examples`; `make -C /root/examples` rebuilds them on the FPGA without rebuilding Linux or gateware.

## Standalone language model

The Linux image contains a project-local C inference runtime under `lfm` and a standalone tokenizer for LiquidAI LFM2.5-350M; no llama.cpp code or executable is built into the image. The 209 MiB Q4_0 GGUF weights are deliberately kept outside the initramfs to avoid holding a second copy while Linux unpacks it. For a RAM boot, stage Linux and the model through the fast bare-metal loader with:

```sh
make -C proj/cva6 remote-boot-model
```

The stage-2 loader receives Linux at `0x80000000` and the model in the reserved DDR region at `0xb0000000`, sustains about 232 KiB/s for both, and verifies one checksum over the bytes re-read from both destinations. Linux mounts a dedicated 224 MiB tmpfs at `/root/models` and automatically exports a staged GGUF to `/root/models/LFM2.5-350M-Q4_0.gguf`, preserving the normal filesystem interface without exhausting the initramfs root mount's independent 252 MiB cap. Set `MODEL=/path/to/model.gguf` to override the source. The staged and exported weights disappear at reboot until they can live directly on the persistent SD-card filesystem.

`make -C proj/cva6 remote-model` remains available as a recovery path after Linux is already running, but interrupt-driven receive is reliable only near 10 KiB/s and is therefore inappropriate for the full model. Its receiver uses block acknowledgements, an end-to-end FNV-1a checksum, and an inactivity timeout that restores Bash after an interrupted upload.

The runtime maps the model without copying it, expands the tied Q6 vocabulary to a 64 MiB int8 workspace once at startup, and offloads its Q4_0×Q8_0 and Q6_K×Q8_0 matrix-vector kernels to the project-owned 200 MHz FPGA accelerator. Scalar model logic remains ordinary C software and the accelerator accesses weights and activations through reserved DDR, so model and chat behavior can still be changed without resynthesizing gateware.

Run `lfm-chat` for an interactive prompt, `/clear` to reset its 256-token context, and `/quit` to exit. `lfm-chat --self-test` checks the standalone GPT-2/Llama-3-style BPE tokenizer against the established token-ID regression without running the model. The deterministic `lfm MODEL 2` test must produce greedy token ID `521`, matching the host oracle; measured prompt and decode rates are printed by the interactive client.

## Software video

The HDMI pipeline always keeps the 160×45 UART terminal current, even while software video is visible. A double-buffered 160×90 RGB232 plane in BRAM scales each source pixel 8× to fill 1280×720, and completed frames become visible only at vertical blank.

Blinking cursor styles default to a 1 Hz full cycle, synchronized to the 60 Hz video frame boundary. Software can change the half-period without rebuilding gateware by writing the project-private control sequence `ESC [ ? n z`, where `n` is the number of video frames per on or off phase; for example, `printf '\033[?30z'` selects the 1 Hz default, `printf '\033[?15z'` selects 2 Hz, and `printf '\033[?0z'` restores the default. Standard DECSCUSR sequences still select blinking versus steady block, underline, or bar cursor shapes.

Linux writes the plane through the existing 32-bit AXI GPIO output. Each transaction carries four packed RGB232 pixels plus a toggle strobe, reducing the memory-mapped I/O traffic by four compared with the original byte-wide protocol. GPIO input bit 5 acknowledges every command after the pixel domain has accepted it, and bit 6 reports when a completed frame has swapped at vertical blank; software waits for both handshakes instead of relying on clock-domain timing. Physical switch 7 remains connected as the emergency terminal override, while switches 5 and 6 are reserved for the acknowledgement and ready flags as seen by software. This leaves UART exclusively available for the login shell and avoids modifying the CVA6 core or DDR controller. Because the GPIO is reserved as an internal command bus, the local device tree does not register it as heartbeat LEDs.

Run animated color bars from Linux with:

```sh
videoctl bars
```

Press Ctrl-C to finish the current frame and return to the terminal. `videoctl tty` also restores the terminal, and physical switch 7 forces the terminal immediately if a program crashes. A raw file passed to `videoctl` is interpreted as consecutive 14,400-byte RGB232 frames, with red in bits 6:5, green in bits 4:2, and blue in bits 1:0.

`mandelbrot` renders a floating-point color zoom and reports its measured frame rate. `badapple` streams the bundled delta-compressed animation from `/root/media/badapple.mvz`. `video-bench` measures the packed frame-submission path independently of rendering. All three are ordinary Linux programs, so their behavior can be changed without resynthesizing the FPGA.

## Validated build

The August 4, 2026 Genesys 2 implementation met 100 MHz timing after post-route physical optimization with 0.000 ns WNS and 0.000 ns TNS. It uses 73,594 LUTs, 46,911 flip-flops, 62 RAMB36 blocks, 2 RAMB18 blocks, and 37 DSPs. The bitstream is `build/gateware/cva6_genesys2_hdmi.bit` with SHA-256 `47bc18e5b25a78b2a94e78536e2e310085a1bd211289f3aa1ed4a40a4e73e3df`.

The hardware-tested 262,885,896-byte RAM payload is `build/sdk/ram-linux.bin` with SHA-256 `f711161718ead07d7d727016bbc77c896f0924c8884c88d902f599146e217e3f`. Its 770-byte stage-2 loader has SHA-256 `f6151208e2e7a6f9b947f9863221517e5999e1a9820308b31d2dea17e39995cf`, and the 219,309,792-byte LFM2.5-350M Q4_0 model has SHA-256 `85e32858daafad55b7bcd6b97a1343ee0661188e8036f9862d14d6b563142f50`. The FPGA re-read the complete Linux and model payloads from DDR and passed the combined checksum as `23856c9d`; OpenSBI then handed Linux the non-overlapping device tree at `0xbfe00000`, Linux reported 768 MiB of RAM, and the following 224 MiB remained reserved for model staging.

Physical serial tests on the final bitstream passed for Linux 6.19.6, Bash 5.2.37, both `python` and `python3` at Python 3.14.3 with the extended headless standard-library test set, NumPy 2.4.0 linked to OpenBLAS 0.3.30, Vim 9.1 alternate-screen editing and saving, tmux 3.6a at 160×45 with independent windows, native C compilation, Snake, color Tetris, and ASCII Invaders. NumPy measured 2.06 million float32 elementwise operations per second and OpenBLAS measured 6.15 million float32 matrix-vector operations per second. Ten independent accelerator runs produced greedy token ID `521` ten times, exactly matching the host oracle; `lfm-chat` answered “Two plus three = 5” at 0.579 prompt tokens/s and 0.551 decode tokens/s. Ten consecutive 60-frame software-video tests sustained 54.93–57.55 fps without a stall, the optimized color Mandelbrot zoom sustained 39.42 fps for 1,888 frames, and Bad Apple sustained 28.81 fps for 477 frames against its 30 fps media clock before a clean Ctrl-C return to the terminal.

The local build machine exposes no camera device, and both integrated cameras on `thinkmax` returned dark or obstructed frames even while bright hardware color bars were active. Consequently the final run proves the pixel-domain request/acknowledgement, vertical-blank swap, animation rate, UART terminal behavior, and clean graphics-to-terminal restoration on the FPGA, but it does not include usable camera evidence of the projected image. Repeat the camera check after exposing and aiming one of the remote cameras.

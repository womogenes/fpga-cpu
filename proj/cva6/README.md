# CVA6 Genesys 2

This directory contains the OpenHW Group CVA6 Genesys 2 platform configured for four 64-bit harts at 100 MHz, a 1280×720 HDMI terminal and software-video output, an FPGA matrix-vector accelerator, and a Buildroot Linux workstation that boots from either a writable RAM filesystem or a persistent SD-card filesystem. Each hart has its own single-precision floating-point unit, instruction cache, and private 8 KiB write-through data cache. The post-crossbar DDR path broadcasts completed writes to peer data caches and excludes the writing hart by its prepended AXI source ID, providing coherent Linux SMP shared-memory semantics without a directory or snoop responses. The upstream repositories are pinned as Git submodules; one consolidated project-owned platform patch captures the SMP topology and coherence path, boot protocol, CPU timing configuration, pipelined floating-point units, exact 3 MBaud UART reference, and reliable divisor-2 receiver sampling.

## Layout

- `hardware/` contains the CVA6 core, CORE-V-APU, Genesys 2 DDR3 controller, SD-card controller, UART, JTAG, boot ROM, and FPGA build flow.
- `sdk/` contains Buildroot, OpenSBI, U-Boot, Linux, the initramfs, and SD-card image generation.
- `rtl/` contains the HDMI terminal, UART mirror, TMDS pipeline, and software-video plane maintained by this project.
- `loader/` contains the relocatable stage-2 RAM loader used for fast, checked Linux uploads.
- `diagnostics/` contains the tiny bare-metal four-hart and AMO smoke test.
- `software/` contains programs compiled into the Linux filesystem by the post-build hook.
- `build/` contains parent-owned generated output and is ignored by Git.

## Initialize

After cloning the parent repository, initialize CVA6 and every pinned dependency with:

```sh
make -C proj/cva6 init
```

## Build gateware

Build the four-hart 64-bit RV64IMAFC Genesys 2 FPGA design with:

```sh
make -C proj/cva6 gateware
```

The wrapped flow copies the bitstream and reports into `build/gateware/` and reuses the bitstream when all hashed inputs are unchanged. Never start this target while any other Vivado process is running on the machine.

## Build Linux image

Build the 64-bit Genesys 2 toolchain, OpenSBI, U-Boot, Linux kernel, persistent ext4 filesystem, and complete SD-card image with:

```sh
make -C proj/cva6 image
```

The resulting SD-card image is written beneath `build/sdk/`. The first build downloads and compiles a complete RISC-V Linux toolchain and can take a long time. The image includes Bash, Python, NumPy/OpenBLAS, Vim, tmux, TinyCC, games, the 160×45 terminal configuration, and the software-video demos. IPython is intentionally omitted because its size and startup cost do not justify carrying it in the current RAM image.

## Boot Linux from RAM

An SD card is not required for the current development flow. Build an OpenSBI payload containing Linux and the complete initramfs with:

```sh
make -C proj/cva6 ram-image
```

The result is `build/sdk/ram-linux.bin`. Program the FPGA, reset it, and start the boot-ROM upload during its ten-second prompt with:

```sh
make -C proj/cva6 upload-ram SERIAL=/dev/ttyUSB0
```

The upload target first sends a 722-byte relocatable stage-2 loader through the stock boot ROM using drained 32-byte writes. Stage 2 relocates near the top of DDR, enables the UART's 64-byte FIFO, receives Linux directly at `0x80000000` with an acknowledgement after each 4096-byte block, and computes a target-side FNV-1a checksum over the received bytes before launch. It then publishes the payload entry and device-tree address through a reserved mailbox so all four harts enter OpenSBI with distinct hart IDs. The UART remains at 3 MBaud throughout and sustains about 232 KiB/s. Keeping the initramfs uncompressed avoids a CVA6 failure observed while unpacking both gzip and Zstandard streams. The command enters the serial console after transfer. The root filesystem is writable and behaves like a normal Linux filesystem, but changes disappear at power-off until the SD-card image is used.

For the normal remote-board workflow, build gateware and Linux locally and then deploy, program, upload, and attach in one command with `make -C proj/cva6 remote-boot`. It copies only the board-ready bitstream, Linux image, stage-2 loader, and uploader to `thinkmax:~/fpga/fpga-cpu/build/cva6/`; override `REMOTE_HOST`, `REMOTE_CVA6_DIR`, or `REMOTE_SERIAL` when needed. The remote launcher opens UART before starting `openFPGALoader`, so it catches the boot-ROM update prompt as soon as JTAG releases reset and does not require pressing the CPU-reset button. HDMI has no signal while the volatile bitstream is still shifting; the launcher requires `openFPGALoader` to reach 100% successfully before continuing. The RAM-boot device tree exposes 752 MiB to Linux and reserves the following 272 MiB for optional model staging; the persistent SD-card image uses the same reservation so either model can be copied once from its filesystem into the accelerator-visible uncached region.

## Install the persistent SD card

`make -C proj/cva6 image` produces `build/sdk/sdcard.img` for a conventional host card reader. It contains a raw OpenSBI/U-Boot partition, a FAT boot partition, a writable 640 MiB ext4 root filesystem, and a writable ext4 models partition containing both GGUF files.

The board can also install its own onboard card without a reader or a 1.7 GiB serial transfer. `make -C proj/cva6 ram-image` appends only the current OpenSBI/U-Boot and FIT payloads to the RAM initramfs. Boot that image with LFM staged, then run:

```sh
sd-install prepare
sd-install model lfm
```

`sd-install prepare` deliberately destroys the contents of `/dev/mmcblk0`, creates the four-partition GPT expected by the boot ROM and U-Boot, formats the persistent filesystems, installs the boot payloads, and copies the live Linux tree. It refuses to repartition the device containing the running root filesystem. To stage Bonsai without uploading the large RAM image again, set `MODEL=/path/to/Bonsai-1.7B.gguf` and run `make -C proj/cva6 remote-boot-sd-model`; stage 2 transfers only the 2.5 MiB OpenSBI/U-Boot payload and the model, then U-Boot starts the existing SD Linux. Run `sd-install model bonsai`; each model copy is committed under `/root/models` only after its SHA-256 hash matches the source artifact. `sd-install status` reports the resulting layout and persistent model files.

After both one-time model installations, FPGA programming still loads the volatile bitstream, but the boot ROM loads OpenSBI/U-Boot and Linux from SD after its update timeout. Linux mounts the SD root read-write and mounts the models partition at `/root/models`, so home-directory changes and both weight files survive resets and power cycles.

## Serial console

The default system runs the UART at exactly 3,000,000 baud, the maximum supported by the Genesys 2 FT232R bridge. The UART baud generator uses a 96 MHz effective reference derived by suppressing one clock enable out of every 25 cycles of the 100 MHz system clock; divisor 2 therefore produces exactly 3 MBaud without adding another clock domain. After loading the FPGA and inserting the generated SD card, connect with:

```sh
make -C proj/cva6 monitor SERIAL=/dev/ttyUSB0
```

The monitor target uses Python pyserial when available and falls back to picocom. The exact `/dev/ttyUSB*` device can vary. The generated Linux system logs in as `root` with no password by default. If the USB-UART adapter cannot sustain the maximum rate, the gateware, boot ROM, U-Boot, Linux command line, device tree, and monitor rate must be changed together.

## Multicore architecture

The SoC contains four complete 100 MHz RV64IMAFC CVA6 harts and four private FP32-capable FPUs. OpenSBI HSM starts all harts, Linux is built with `CONFIG_SMP=y` and `CONFIG_NR_CPUS=4`, the CLINT exposes four software-interrupt and timer contexts, and the PLIC exposes machine and supervisor contexts for every hart. The boot ROM gives each hart a private stack in reserved high DDR and parks secondary harts on a low-rate mailbox until the payload is complete.

Private instruction and write-through data caches start enabled. Every CPU store passes through the consolidated post-crossbar DDR channel, which records the address and prepended source-port ID when AW is accepted and broadcasts the address to peer caches only after the matching DDR write response. The cache adapter invalidates all ways at that index; the writer keeps its locally updated line, while debug-system writes invalidate every hart. Accelerator scratch, activation, and result mappings remain uncached, so ordinary fences order their accesses without flushing hot model state from every data cache.

Four is the practical maximum while retaining the HDMI terminal and inference accelerator. Removing per-hart FP64, crypto, CV-X-IF, trace, and debug logic creates room for the fourth core; a fifth core would exceed the XC7K325T LUT budget. Linux userspace uses glibc's standard `LP64` ABI because glibc does not support `LP64F`. Compilers can still emit FP32 instructions, while C `double` operations use software helpers instead of requiring FP64 hardware.

Build the bare-metal validation payload with `make -C proj/cva6 smp-smoke`. It releases all four harts, serializes their UART messages with an AMO lock, performs a shared atomic barrier, and prints `SMP PASS: 4 harts and AMOs operational` only if every hart participates.

## Linux workstation

The image boots to Bash in `/root` and provides both `python` and `python3`, NumPy, Vim, tmux, make, target binutils, and TinyCC as `cc`. The glibc toolchain uses the portable RV64IMAC/LP64 baseline because upstream glibc rejects F-without-D targets; project-owned performance programs and both inference runtimes are compiled explicitly for RV64IMAFC/LP64 and use the four FP32 units. NumPy is linked to threaded OpenBLAS and can use all four harts. Run `numpy-bench` to print the NumPy/OpenBLAS configuration and measure representative float32 kernels. IPython and its dependency stack are not installed.

Run `workstation-check` to verify the core tools, NumPy, the standalone LFM tokenizer, and a native C smoke test directly on CVA6.

Games are available as `snake`, `tetris`, and `ascii-invaders`. Editable C examples live in `/root/examples`; `make -C /root/examples` rebuilds them on the FPGA without rebuilding Linux or gateware.

## Standalone language model

The Linux image contains a project-local C inference runtime under `lfm` and a standalone tokenizer for LiquidAI LFM2.5-350M. Bonsai uses a separate stripped and accelerator-enabled llama.cpp runtime exposed through `bonsai-chat`; LFM does not depend on it. The 209 MiB Q4_0 GGUF weights are deliberately kept outside the initramfs to avoid holding a second copy while Linux unpacks it. For a RAM boot, stage Linux and the model through the fast bare-metal loader with:

```sh
make -C proj/cva6 remote-boot-model
```

The stage-2 loader receives Linux at `0x80000000` and the model in the reserved DDR region at `0xaf000000`, sustains about 232 KiB/s for both, acknowledges every block, and compares its receive checksum with the host checksum. The LFM runtime maps staged weights directly and uncached through `/dev/mem`, eliminating the previous 209 MiB DDR-to-tmpfs copy and its multi-minute startup delay. Set `MODEL=/path/to/model.gguf` to override the host source. On an SD boot, the runtime copies the persistent model into this accelerator-visible reserved region once during startup.

`make -C proj/cva6 remote-model` remains available as a recovery path after Linux is already running, but interrupt-driven receive is reliable only near 10 KiB/s and is therefore inappropriate for the full model. Its receiver writes directly into reserved DDR and uses block acknowledgements, an end-to-end FNV-1a checksum, and an inactivity timeout that restores Bash after an interrupted upload.

The runtime offloads its Q4_0×Q8_0 and Q6_K×Q8_0 matrix-vector kernels to the project-owned 200 MHz FPGA accelerator without expanding or copying the tied Q6 vocabulary. Eight activation BRAM banks feed 16 Q4 products or 8 Q6 products per accelerator cycle. Bonsai uses a Q1_0×Q8_0 whole-matrix hook: each activation is quantized once, one hardware command computes every output row, and one contiguous loop converts the Q30 result, avoiding roughly 725,000 per-row callbacks per generated token. Scalar model logic remains ordinary C software and the accelerator accesses weights and activations through reserved DDR, so model and chat behavior can still be changed without resynthesizing gateware. Bonsai defaults to four workers and LFM retains its cache-local single-worker path.

Run `lfm-chat` for an interactive prompt, `/clear` to reset its 256-token context, and `/quit` to exit. Startup immediately displays an in-place bar while Python starts, tokenizer indices are built, and the decoder maps the model; pass `--no-progress` to suppress it. A shell fast path handles `lfm-chat --help` without starting Python or accessing the model. `lfm-chat --self-test` checks the standalone GPT-2/Llama-3-style BPE tokenizer against the established token-ID regression without running the model. The deterministic `lfm MODEL 2` test must produce greedy token ID `521`, matching the host oracle; measured prompt and decode rates are printed by the interactive client.

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

The preserved August 5, 2026 three-hart FP64 accelerator baseline runs the CPUs at 100 MHz with −0.964 ns post-route WNS after physical optimization. This small timing violation is accepted for development bring-up. It uses 164,801 LUTs, 79,270 flip-flops, 114 RAMB36 blocks, 2 RAMB18 blocks, and 107 DSPs. Its bitstream is `build/gateware/baselines/three-hart-fp64-accelerated/cva6_genesys2_hdmi.bit` with SHA-256 `39b97784a974bc078f4a9045301cbcf816c7e9873589a91a987e302aafbb7662`.

The preserved three-hart LP64D RAM payload is 283,854,856 bytes with SHA-256 `66409d5093f7cd8ef0548f68d2e07d1634e8de2bb02d8f200ad59b9d9d752209`. Its 722-byte stage-2 loader has SHA-256 `e7a99b6e906cee72056ccabda4ee23a9fbf881ab780827f74e92891fbd9e23cf`, and the 219,309,792-byte LFM2.5-350M Q4_0 model has SHA-256 `85e32858daafad55b7bcd6b97a1343ee0661188e8036f9862d14d6b563142f50`. The last physically validated SMP image reported harts 0–2, Linux 6.19.6 brought up all three CPUs, `nproc` returned `3`, `/sys/devices/system/cpu/online` reported `0-2`, and Python multiprocessing reported three CPUs. Linux reported 768 MiB of RAM, with the remaining DDR reserved for model staging.

Physical serial tests on the workstation stack before the SMP expansion passed for Linux 6.19.6, Bash 5.2.37, both `python` and `python3` at Python 3.14.3, NumPy 2.4.0 linked to OpenBLAS 0.3.30, Vim 9.1 alternate-screen editing and saving, tmux 3.6a at 160×45 with independent windows, native C compilation, Snake, color Tetris, and ASCII Invaders. Python reports 297 standard-library module names and resolves 282 of them; the 15 absent names are Windows/macOS-only modules, Tk/IDLE on this headless system, optional DBM and native UUID accelerators with pure-Python fallbacks, and `pydoc_data`, while SSL, SQLite, curses, ctypes, multiprocessing, venv, ensurepip, compression, XML, HTTP, and asyncio import successfully. Rebuilding Mandelbrot, Bad Apple, Snake, and Hello directly under `/root/examples` with TinyCC succeeded; the optimized image binaries remain preferable for animation because the native TinyCC Bad Apple build measured 17.96 fps while the cross-compiled binary measured 28.81 fps. NumPy measured 2.06 million float32 elementwise operations per second and OpenBLAS measured 6.15 million float32 matrix-vector operations per second. Ten independent accelerator runs produced greedy token ID `521` ten times, exactly matching the host oracle; `lfm-chat` answered “Two plus three = 5” at 0.579 prompt tokens/s and 0.551 decode tokens/s. Ten consecutive 60-frame software-video tests sustained 54.93–57.55 fps without a stall, the optimized color Mandelbrot zoom sustained 39.42 fps for 1,888 frames, and optimized Bad Apple sustained 28.81 fps for 477 frames against its 30 fps media clock before a clean Ctrl-C return to the terminal.

## Four-hart coherent accelerated build (August 10, 2026)

The current four-hart RV64IMAFC build carries a redesigned streaming Q1_0 accelerator: four replicated 64-bit activation banks deliver 64 bytes per cycle, the exact Q30 scale arithmetic runs as a free-running pipeline overlapped with weight streaming, row results retire through a non-blocking write FIFO, and an `axi_cut` register slice isolates the accelerator from the MIG AXI shim. Exact-output simulation covers 256/1024/2048-column and 1-to-96-row shapes plus arbiter integration under random mid-burst stalls; the sustained simulated rate is about 3 cycles per 128-weight block versus about 20 before. A Q1 argmax command compares signed Q30 row results in hardware and returns an 18-bit token without writing the 1.2 MiB vocabulary result vector to DDR; simulation verifies exact results through row 65,537, beyond the 16-bit boundary. The undefined-state trap drains through the flush path so a corrupted FSM can never release the DDR bus with live AXI transactions. Status bit 27 advertises hardware argmax, bits 25:8 return its token, and the same field exposes the live FSM state for other modes.

Linux 6.19.6 carries the project F-without-D patch: OpenSBI reports `rv64imafcb`, Linux advertises `rv64imafc` with no "F but not D" masking, and all four harts come online. The candidate coherent bitstream is `build/gateware/cva6_genesys2_hdmi.bit`, 11,443,743 bytes, SHA-256 `626552e7000776dbc608f4ff7a84f98cc2811b8bb6863b5ac9c0bb5c864dad96`, with global post-route WNS −0.313 ns and no failed or unrouted nets. The physically validated predecessor is preserved separately for automatic recovery. The physical `smp_cache_stress` diagnostic passed twice with `PASS: 4 harts, 20000 coherence rounds, 1000000 atomic increments`; its producer/consumer rounds deliberately move ownership of shared cache lines among all harts while independent AMO contention checks atomic progress. The refill ordering fix gives an external invalidation priority over a new load request and waits for the read queue to drain before invalidating, preventing a stale in-flight refill from repopulating a line after a peer write completes.

The clean RAM Linux payload is 296,438,792 bytes with SHA-256 `675a4676a1eee4d409edd869823d7fac052b64c34677b2c252ffa56536f88d0c`; its 722-byte stage-2 loader has SHA-256 `e7a99b6e906cee72056ccabda4ee23a9fbf881ab780827f74e92891fbd9e23cf`. The image bundles Python 3.14, Bash, Vim, tmux, the current optimized `bonsai-lite`, `bonsai-chat`, and its 151,669-token tokenizer. The checked stage-2 flow uploads that image and the 248,302,272-byte Bonsai model into disjoint DDR regions before Linux starts.

Physical Bonsai-1.7B generation through the accelerator is validated end to end. The slim `bonsai-lite` runtime (`software/bonsai_main.c`) implements the full Qwen3 forward pass with every Q1_0 matrix on the accelerator and preserves the established greedy sequence `25,220,16,24,24,15,11,279,1156` on hardware. Private data caches reduced the original 8.336 seconds per token to 2.269 seconds per token before software parallelization. A four-hart busy-spin worker pool, float KV storage with exact FP16 insertion rounding, parallel fused residual/RMS normalization, fused Q30 result conversion with attention and feed-forward preparation, RV64 bit-manipulation tuning, and raw Q30 argmax reduced the final physically tested eight-token decode to 8.02 seconds, or 1.003 seconds per token and 0.997 tokens per second. Repeated one-token acceptance runs measured 1.001–1.007 tokens per second. The new hardware argmax path is fully simulated and remains backward-compatible with the physically validated predecessor through capability detection and the raw-Q30 software fallback.

The output-head transfer and conversion that formerly consumed a substantial share of each token are eliminated by hardware argmax on the candidate bitstream. Attention QKV conversion is fused with head preparation, and feed-forward up/gate conversion is fused with SiLU activation. The four-thread production path sets `BONSAI_PARALLEL_ROWS=1` so all nontrivial rows use the persistent worker pool; set `BONSAI_SPIN_WORKERS=0` to restore futex sleeping for workloads where idle power matters more than token latency. Three FPU defects remain worked around in software: `fdiv.s` returns NaN, `fsqrt.s` is absent from the configured FPU, and `fcvt.s.l` corrupts 64-bit conversions. Newton reciprocal/rsqrt and signed-high/unsigned-low Q30 conversion avoid those instructions without changing the validated tokens.

Interactive Bonsai chat ships as `bonsai-chat` (`software/bonsai_chat.py`), a Python client over the `bonsai-lite --server` TOKEN/EVAL line protocol. It carries a pure-Python qwen2-style byte-level BPE tokenizer fed from `/usr/share/bonsai/tokenizer.pickle`, validated byte-exact against `llama-tokenize` across contractions, digit splitting, Unicode, and code fragments; `bonsai-chat --self-test` re-checks the reference vectors. Prompts render in Qwen3 ChatML with no BOS and an empty `<think></think>` prefill so decoding starts on the reply immediately. Set `BONSAI_MODEL_PRESTAGED=1` after a stage-2 model upload to validate and use the accelerator-visible reserved-DDR weights without copying the SD file into that region again; `bonsai-lite` still maps the persistent file for GGUF metadata and CPU-side tensors. The previous llama.cpp-based wrapper remains available as `bonsai-runtime-chat`.

LFM2.5-350M is validated on the same four-hart F-only build: `lfm-chat` with the model exported to tmpfs runs FPGA-accelerated generation at 0.572 prompt and 0.534 decode tokens per second, matching the previously established three-hart baseline within noise and meeting the 0.5 tokens-per-second acceptance threshold. The deterministic `lfm MODEL 2` probe now repeatably produces greedy token ID `43987` instead of the historical `521`: the old oracle was captured on the soft-float userspace that the F-masking kernel forced, while the F-hardware build evaluates scalar float expressions with single-precision hardware rounding. The host oracle must be regenerated with matching float32 semantics before token-identity comparisons are meaningful again.

Model staging without the SD card: boot with `--model`, then set `BONSAI_MODEL_PRESTAGED=1` for Bonsai or use the LFM runtime's direct reserved-DDR path. SD bulk I/O is slow by construction rather than degraded: the slot is SPI-attached (`CONFIG_MMC_SPI` through the Xilinx SPI core in PIO mode), so `dd conv=fsync` measures about 20 KiB/s with system time approximately equal to wall time (PIO polling), while sequential installer writes with batched writeback sustain roughly 90 KiB/s. Before bulk SD writes, shrink `vm.dirty_bytes` to 8 MiB and `vm.dirty_background_bytes` to 2 MiB so writeback throttles smoothly instead of stalling in `balance_dirty_pages`. The card can be power-cycled without touching the board: Genesys 2 pin AE24 (`sd_reset`) drives the slot power switch, so loading the one-line `sd_power_off.bit` bitstream (source in `tools/sd_power_off/`) for about 20 seconds and then reloading the CVA6 bitstream re-enumerates the card cleanly. After such a power cycle on August 9, 2026, `sd-install model bonsai` completed end to end with the writeback limits above: the 248,302,272-byte model exported from reserved DDR to the SD models partition in about three hours, and the installer's full SHA-256 read-back of the on-card copy matched the source artifact (`3d7c6c90dd98717a203adb22d5eacd2581850e40aa5327e144b97766cae5f7e3`), so both Bonsai and LFM now persist on the card and survive power loss.

The local build machine exposes no camera device, and both integrated cameras on `thinkmax` returned dark or obstructed frames even while bright hardware color bars were active. Maximum manual exposure produced featureless gray or white frames with optical vignetting rather than a visible scene, confirming that both lenses are physically covered or facing a nearby surface; automatic exposure and focus were restored afterward. Consequently the final run proves the pixel-domain request/acknowledgement, vertical-blank swap, animation rate, UART terminal behavior, and clean graphics-to-terminal restoration on the FPGA, but it does not include usable camera evidence of the projected image. Repeat the camera check after exposing and aiming one of the remote cameras.

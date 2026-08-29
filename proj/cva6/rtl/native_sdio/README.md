# Native 4-bit SD integration

This directory stages the shortest known route from the current 25 MHz SPI-mode SD interface to a native 4-bit SD High Speed interface on Genesys 2. The upstream controller is pinned to commit `3264579e8286b0946c49aa44680738cc98cdd6f3` from `https://github.com/eugene-tarassov/vivado-risc-v` and is kept unmodified under `upstream/`.

The imported implementation is a particularly good fit for this project:

- it is already used on Genesys 2 and uses the exact board pins present in `hardware/corev_apu/fpga/constraints/genesys-2.xdc`;
- it implements native 1-bit and 4-bit SD, including SD High Speed at 50 MHz;
- its control side is AXI4-Lite and its integrated DMA is an AXI4 master;
- matching U-Boot DM-MMC and Linux MMC host drivers are staged in `software/native_sdio/upstream/`;
- upstream reports 25 MB/s and supports the same partition/filesystem operations used by this image.

The default build remains on the verified SPI path. Set `NATIVE_SD=1` (or use `make native-sd-gateware` and `make native-sd-image`) to select this controller, its native-SD boot ROM, and the matching alternate device tree. Native gateware uses separate filenames and input hashes, while the native software image uses `build/buildroot-native-sd` and `build/sdk-native-sd`; this option cannot silently replace or reuse the default SPI artifacts.

## Genesys 2 pin map

| SD signal | FPGA pin | Current SPI alias |
|-----------|----------|-------------------|
| CLK       | R28      | `spi_clk_o`       |
| CMD       | R29      | `spi_mosi`        |
| DAT0      | R26      | `spi_miso`        |
| DAT1      | R30      | unused            |
| DAT2      | P29      | unused            |
| DAT3      | T30      | `spi_ss`          |
| RESET     | AE24     | unused            |
| CD        | P28      | unused            |

All signals are 3.3 V. `DAT[3:0]` and `CMD` are bidirectional and must use IOBUFs. The imported controller already instantiates those IOBUFs.

## Integration architecture

Keep the controller register aperture at `0x20000000` and PLIC source 2. This allows the native controller to replace the AXI Quad SPI peripheral without changing the SoC address map.

The controller has two AXI connections:

1. A compact lane adapter maps 32-bit controller registers into the existing 64-bit SPI register aperture.
2. A fourth main-crossbar initiator carries the controller's 32-bit narrow AXI bursts directly over the 64-bit fabric. Each beat is placed in the low or high 32-bit lane according to address bit 2.

For the current two-core configuration, changing `ariane_soc::NrSlaves` from `NumCores + 1` to `NumCores + 2` adds DMA port 3 without increasing the downstream ID width: both three and four initiators require two source-ID bits. That avoids perturbing the MIG interface width.

Configure the raw DMA AXI attributes as follows because the upstream controller only exposes the signals it needs:

- `AWID/ARID = 0`
- `AWSIZE/ARSIZE = 2` (four-byte source beats)
- `AWBURST/ARBURST = INCR`
- `WSTRB = 4'b1111`
- lock, cache, protection, QoS, region, user, and ATOP fields = 0

The imported controller defaults to 32 address bits. This covers the complete Genesys 2 DRAM window (`0x80000000` through `0xbfffffff`) and the existing model preload destination at `0xaf000000`.

The active CVA6/peripheral clock is 94 MHz, so the controller's divide-by-two fast setting produces a 47 MHz SD clock. Describe that exact input clock in the device tree and set `max-frequency = <47000000>`. A later clock-domain wrapper can raise it to 50 MHz, but the extra 6% is not worth adding CDC risk during first bring-up.

## Device tree replacement

Replace the `xps-spi@20000000` node with:

```dts
mmc@20000000 {
  compatible = "riscv,axi-sd-card-1.0";
  reg = <0x0 0x20000000 0x0 0x10000>;
  interrupt-parent = <&PLIC0>;
  interrupts = <2>;
  clock = <94000000>;
  bus-width = <4>;
  fifo-depth = <128>;
  max-frequency = <47000000>;
  cap-sd-highspeed;
  no-sdio;
  disable-wp;
};
```

The exact PLIC interrupt-cell encoding must match the other working devices in the final DT. The current SPI node uses two cells while the PLIC binding elsewhere is inconsistent, so validate this with `dtc` and boot logs rather than copying blindly.

## Software work

The U-Boot driver is integrated through `sdk/patches/genesys2/uboot/`, is compatible with the DM-MMC API, and supports multi-block DMA. The native image uses a separate U-Boot defconfig whose `CONFIG_DEFAULT_DEVICE_TREE` selects `cv64a6_imafdc_sv39_native_sdio`; merely copying an alternate DTS is not enough for U-Boot's `OF_SEPARATE` build. The existing `mmc read` preload command and its raw block/count values are unchanged.

The Linux driver is integrated through `sdk/patches/genesys2/linux/`, uses the generic MMC core and DMA mapping APIs, and advertises a single DMA segment so the MMC core does not submit unsupported scatter-gather requests. Both drivers are enabled in their defconfigs but remain inactive in the default SPI device tree.

The first full native synthesis passed RTL elaboration with zero errors, then the kernel OOM killer terminated Vivado during technology mapping while another large workload occupied RAM and swap. This is not an RTL failure. Retry only with Vivado otherwise idle. The current hook limits Vivado to two threads and the native build closes the preliminary elaborated design before launching optimized synthesis, avoiding two large netlists resident at once.

The U-Boot and Linux drivers must be present in the same SD image as the native-controller bitstream. A bitstream-only test cannot boot because native SD and SPI mode are mutually exclusive after card initialization.

## Throughput target

The current measured path is about 388.7 KiB/s, taking roughly 10 minutes to read the 248,302,272-byte model. Native 4-bit at 47 MHz has a raw ceiling of 23.5 MB/s. Expected first-pass sustained throughput is 15-22 MB/s, corresponding to roughly 11-17 seconds for the model. Even a conservative 10 MB/s result loads it in about 25 seconds.

## Bring-up gates

Before flashing a native-SD candidate:

1. Build the native controller and both drivers without touching the verified bitstream artifact.
2. Run RTL elaboration and verify the new DMA crossbar port has no width or ID warnings.
3. Require positive setup and hold slack at the current 94 MHz system clock.
4. Fix the staged Linux driver's lock initialization and cleanup ordering, and provide working non-coherent DMA cache maintenance in U-Boot.
5. Write the matching kernel/U-Boot image to a spare or recoverable SD card.
6. First test U-Boot `mmc info`, partition listing, a small `mmc read`, and a known checksum. Repeat the read into a cache-line-aligned destination that the CPU filled with a different pattern first.
7. Time the existing 484,966-block model preload and verify its B8UF marker.
8. Boot Linux, mount every existing partition, run filesystem reads and writes, and verify model inference.

Keep `cva6_bonsai_bank8_q4prefetch_4aece477.bit` as the rollback bitstream. Do not overwrite it.

## Main risks

- The controller and its drivers are custom rather than SDHCI-standard, so all three pieces must stay in sync.
- DMA is non-coherent. U-Boot and Linux must invalidate/flush buffers correctly; checksum tests should include buffers previously touched by both CPUs.
- Adding a crossbar initiator may affect timing even though the ID width is unchanged.
- Card detect and reset polarity must be checked on real hardware.
- The imported RTL is LGPL-2.1-or-later and the two software drivers are GPL-licensed. Preserve their notices and source availability.

# Native SD software staging

The files in `upstream/` are pinned reference drivers imported from `eugene-tarassov/vivado-risc-v` commit `3264579e8286b0946c49aa44680738cc98cdd6f3`, with the small CVA6 correctness adaptations listed below.

- `linux-fpga-axi-sdc.c` is a Linux MMC host driver and declares GPL v2 licensing.
- `uboot-fpga-axi-sdc.c` is a U-Boot DM-MMC driver derived under LGPL-2.1-or-later notices.

The adapted copies are mirrored into Buildroot board patches under `sdk/patches/genesys2/linux/` and `sdk/patches/genesys2/uboot/`. They compile against the pinned Linux 6.19.6 and U-Boot 2026.01 trees. The native device tree activates them only in the `NATIVE_SD=1` image.

The review found and fixed these issues in the staged copies:

- The Linux lock is initialized before reset and IRQ registration, IRQs are freed on subsequent probe failures, DMA is unmapped before completing a request, and request completion runs outside the driver's spinlock. Device removal quiesces the MMC host before freeing its IRQ.
- U-Boot now places full memory barriers around DMA. The native-SD hardware option extends the existing CVA6 SMP invalidation gate to invalidate every L1 line in an AXI write burst before returning the device's write response. Because these L1s are write-through, this supplies the missing device-DMA coherence without relying on unavailable Zicbom instructions or an uncached bounce buffer.

Linux retains `dma_map_sg()` and `dma_unmap_sg()` for ownership ordering. Test both DMA directions with CPU-dirtied buffers before allowing filesystem writes; the hardware invalidation range logic is the correctness boundary.

See `rtl/native_sdio/README.md` for the hardware architecture, device-tree node, throughput target, and bring-up gates.

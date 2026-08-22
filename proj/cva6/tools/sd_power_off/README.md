# sd_power_off — remote SD card power cycle

The Genesys 2 SD slot is powered through a switch controlled by FPGA pin AE24
(`sd_reset`); driving it high cuts slot power. This one-line design does only
that, so the card can be power-cycled without physical access:

```sh
vivado -mode batch -source build.tcl        # produces sd_power_off.bit
openFPGALoader -b genesys2 sd_power_off.bit # SD power off
sleep 20
# reload the CVA6 bitstream (normal program_and_upload.py boot) — SD power returns
```

Used 2026-08-08 to recover the card after writeback stalls; it re-enumerated
cleanly (mmc0 29.1 GiB, partitions p1–p4).

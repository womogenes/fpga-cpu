# Additional Genesys 2 pins used only by the NATIVE_SD build. CLK, CMD, DAT0,
# and DAT3 retain the existing SPI-named top-level ports and constraints.
set_property -dict {PACKAGE_PIN R30 IOSTANDARD LVCMOS33} [get_ports sd_dat1]
set_property -dict {PACKAGE_PIN P29 IOSTANDARD LVCMOS33} [get_ports sd_dat2]
set_property -dict {PACKAGE_PIN AE24 IOSTANDARD LVCMOS33} [get_ports sd_reset]
set_property -dict {PACKAGE_PIN P28 IOSTANDARD LVCMOS33} [get_ports sd_cd]

set_property PULLUP true [get_ports {spi_mosi spi_miso spi_ss sd_dat1 sd_dat2}]
set_property SLEW FAST [get_ports {spi_clk_o spi_mosi spi_miso spi_ss sd_dat1 sd_dat2}]
set_property IOB TRUE [get_ports {spi_clk_o spi_mosi spi_miso spi_ss sd_dat1 sd_dat2}]

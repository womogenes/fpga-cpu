set_property -dict {PACKAGE_PIN AA20 IOSTANDARD TMDS_33} [get_ports hdmi_out_clk_p]
set_property -dict {PACKAGE_PIN AB20 IOSTANDARD TMDS_33} [get_ports hdmi_out_clk_n]
set_property -dict {PACKAGE_PIN AC20 IOSTANDARD TMDS_33} [get_ports {hdmi_out_data_p[0]}]
set_property -dict {PACKAGE_PIN AC21 IOSTANDARD TMDS_33} [get_ports {hdmi_out_data_n[0]}]
set_property -dict {PACKAGE_PIN AA22 IOSTANDARD TMDS_33} [get_ports {hdmi_out_data_p[1]}]
set_property -dict {PACKAGE_PIN AA23 IOSTANDARD TMDS_33} [get_ports {hdmi_out_data_n[1]}]
set_property -dict {PACKAGE_PIN AB24 IOSTANDARD TMDS_33} [get_ports {hdmi_out_data_p[2]}]
set_property -dict {PACKAGE_PIN AC25 IOSTANDARD TMDS_33} [get_ports {hdmi_out_data_n[2]}]

# The UART TX wire is intentionally sampled as asynchronous serial data. Only the
# first synchronizer stage accepts the asynchronous transition.
set_false_path -to [get_pins i_terminal/i_uart_rx/serial_meta_reg/D]

# Software-video commands cross from the 50 MHz AXI GPIO clock into the pixel
# clock through command_meta_reg and command_sync_reg. Only the first stage is
# asynchronous; the second stage remains timed normally.
set_false_path -to [get_pins {i_terminal/i_software_video/command_meta_reg[*]/D}]

# The upstream Genesys 2 constraint targets this MIG-generated reset before the
# HDMI wrapper adds the i_soc hierarchy level. Preserve the intended exception.
set_false_path -from [get_pins i_soc/i_ddr/u_xlnx_mig_7_ddr3_mig/u_ddr3_infrastructure/rstdiv0_sync_r1_reg_rep/C]

# CVA6's optional DPTI PMOD clock is supplied externally and is unrelated to the
# SoC clock. Its CDC logic must not be timed as a synchronous clock crossing.
set_clock_groups -asynchronous \
    -group [get_clocks clk_pll_i] \
    -group [get_clocks -include_generated_clocks prog_clko_pin]

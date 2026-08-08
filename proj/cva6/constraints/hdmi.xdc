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

# Software-video commands cross from the 100 MHz AXI GPIO clock into the pixel
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

# The accelerator command word crosses from the 100 MHz GPIO clock into the
# DDR UI clock through command_meta_reg. The toggle-plus-settle protocol
# tolerates one ambiguous capture, but the data bits must settle well inside
# the driver's settle window; cap their delay explicitly so placement cannot
# stretch this crossing.
set_max_delay -datapath_only 5.0 -to [get_pins {i_soc/i_lfm_matvec_accelerator/command_meta_reg[*]/D}]

# The accelerator and its AXI register slice deassert their asynchronous reset
# through the main reset synchronizer while the DDR domain is quiescent, so
# recovery/removal timing on the released edge is not functional.
set_false_path -from [get_pins {i_soc/i_rstgen_main/i_rstgen_bypass/synch_regs_q_reg[*]/C}] -to [get_cells -hier -filter {NAME =~ *i_lfm_matvec_accelerator* || NAME =~ *i_lfm_axi_cut*}]

# The Q30 block scale and 32-bit dot product are held for two clocks before the
# matrix-block product is captured. Target sequential cells only so synthesized
# helper LUT names cannot accidentally become timing endpoints.
set lfm_block_dot_registers [get_cells -hier -filter {NAME =~ *i_lfm_matvec_accelerator*block_dot_reg* && REF_NAME =~ FD*}]
set lfm_scale_alignment_registers [get_cells -hier -filter {(NAME =~ *i_lfm_matvec_accelerator*scale_mantissa_product* || NAME =~ *i_lfm_matvec_accelerator*scale_shift_reg* || NAME =~ *i_lfm_matvec_accelerator*scale_zero_reg*) && (REF_NAME =~ FD* || REF_NAME == DSP48E1)}]
set lfm_scale_product_registers [get_cells -hier -filter {NAME =~ *i_lfm_matvec_accelerator*scale_product_reg* && (REF_NAME =~ FD* || REF_NAME == DSP48E1)}]
set lfm_scaled_term_cells [get_cells -hier -filter {NAME =~ *i_lfm_matvec_accelerator*scaled_term* && (REF_NAME =~ FD* || REF_NAME == DSP48E1)}]
set_multicycle_path -setup 2 -from $lfm_block_dot_registers -to $lfm_scaled_term_cells
set_multicycle_path -hold 1 -from $lfm_block_dot_registers -to $lfm_scaled_term_cells
set_multicycle_path -setup 2 -from $lfm_scale_alignment_registers -to $lfm_scale_product_registers
set_multicycle_path -hold 1 -from $lfm_scale_alignment_registers -to $lfm_scale_product_registers
set_multicycle_path -setup 2 -from $lfm_scale_product_registers -to $lfm_scaled_term_cells
set_multicycle_path -hold 1 -from $lfm_scale_product_registers -to $lfm_scaled_term_cells

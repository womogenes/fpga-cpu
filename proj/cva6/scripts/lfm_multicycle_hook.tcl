# Applied as STEPS.OPT_DESIGN.TCL.PRE on impl_1 so the linked netlist exists
# when the cells are looked up (XDC files are evaluated too early for the
# REF_NAME filters to match). The Q4/Q6 scale tail of the LFM/Bonsai
# accelerator holds block_dot, the scale alignment operands, and the scale
# product stable for two cycles by FSM construction (Q4_SCALE_WAIT /
# Q6_SCALE_* wait states). The Q1 streaming pipeline (s0-s6 registers) is
# genuinely single-cycle and is deliberately not matched by these patterns.
set lfm_block_dot [get_cells -hier -filter {NAME =~ *i_lfm_matvec_accelerator*block_dot_reg* && REF_NAME =~ FD*}]
set lfm_scale_align [get_cells -hier -filter {(NAME =~ *i_lfm_matvec_accelerator*scale_mantissa_product* || NAME =~ *i_lfm_matvec_accelerator*scale_shift_reg* || NAME =~ *i_lfm_matvec_accelerator*scale_zero_reg*) && (REF_NAME =~ FD* || REF_NAME == DSP48E1)}]
set lfm_scale_product [get_cells -hier -filter {NAME =~ *i_lfm_matvec_accelerator*scale_product_reg* && (REF_NAME =~ FD* || REF_NAME == DSP48E1)}]
set lfm_scaled_term [get_cells -hier -filter {NAME =~ *i_lfm_matvec_accelerator*scaled_term* && (REF_NAME =~ FD* || REF_NAME == DSP48E1)}]
if {[llength $lfm_block_dot] == 0} {
    set lfm_block_dot [get_cells -hier -filter {NAME =~ block_dot_reg* && REF_NAME =~ FD*}]
    set lfm_scale_align [get_cells -hier -filter {(NAME =~ scale_mantissa_product* || NAME =~ scale_shift_reg* || NAME =~ scale_zero_reg*) && (REF_NAME =~ FD* || REF_NAME == DSP48E1)}]
    set lfm_scale_product [get_cells -hier -filter {NAME =~ scale_product_reg* && (REF_NAME =~ FD* || REF_NAME == DSP48E1)}]
    set lfm_scaled_term [get_cells -hier -filter {NAME =~ scaled_term* && (REF_NAME =~ FD* || REF_NAME == DSP48E1)}]
}
set lfm_command_meta [get_cells -hier -filter {NAME =~ *i_lfm_matvec_accelerator*command_meta_reg* && REF_NAME =~ FD*}]
set lfm_status_meta [get_cells -hier -filter {NAME =~ *lfm_status_meta_reg* && REF_NAME =~ FD*}]
set lfm_row_block [get_cells -hier -filter {NAME =~ *i_lfm_matvec_accelerator*row_block_reg* && REF_NAME =~ FD*}]
puts "LFM_MULTICYCLE block_dot=[llength $lfm_block_dot] scale_align=[llength $lfm_scale_align] scale_product=[llength $lfm_scale_product] scaled_term=[llength $lfm_scaled_term]"
# The upstream DMI constraints bound only half of the symmetric two-phase CDC.
# Apply a symmetric 50 ns datapath limit to each payload/toggle/ack crossing.
# These buses remain stable until the opposite-domain acknowledgement, and the
# 50 ns bound is still half of the 100 ns JTAG clock period.
set dmi_resp_data_src [get_pins {i_soc/i_dmi_jtag/i_dmi_cdc/i_cdc_resp/i_src/data_src_q_reg*/C}]
set dmi_resp_data_dst [get_pins {i_soc/i_dmi_jtag/i_dmi_cdc/i_cdc_resp/i_dst/data_dst_q_reg*/D}]
set dmi_resp_toggle_src [get_pins {i_soc/i_dmi_jtag/i_dmi_cdc/i_cdc_resp/i_src/req_src_q_reg/C}]
set dmi_resp_toggle_dst [get_pins {i_soc/i_dmi_jtag/i_dmi_cdc/i_cdc_resp/i_dst/req_dst_q_reg/D}]
set dmi_req_data_src [get_pins {i_soc/i_dmi_jtag/i_dmi_cdc/i_cdc_req/i_src/data_src_q_reg*/C}]
set dmi_req_data_dst [get_pins {i_soc/i_dmi_jtag/i_dmi_cdc/i_cdc_req/i_dst/data_dst_q_reg*/D}]
set dmi_req_toggle_src [get_pins {i_soc/i_dmi_jtag/i_dmi_cdc/i_cdc_req/i_src/req_src_q_reg/C}]
set dmi_req_toggle_dst [get_pins {i_soc/i_dmi_jtag/i_dmi_cdc/i_cdc_req/i_dst/req_dst_q_reg/D}]
set dmi_resp_ack_src [get_pins {i_soc/i_dmi_jtag/i_dmi_cdc/i_cdc_resp/i_dst/ack_dst_q_reg/C}]
set dmi_resp_ack_dst [get_pins {i_soc/i_dmi_jtag/i_dmi_cdc/i_cdc_resp/i_src/ack_src_q_reg/D}]
set dmi_req_ack_src [get_pins {i_soc/i_dmi_jtag/i_dmi_cdc/i_cdc_req/i_dst/ack_dst_q_reg/C}]
set dmi_req_ack_dst [get_pins {i_soc/i_dmi_jtag/i_dmi_cdc/i_cdc_req/i_src/ack_src_q_reg/D}]
puts "DMI_CDC resp_data=[llength $dmi_resp_data_src]/[llength $dmi_resp_data_dst] resp_toggle=[llength $dmi_resp_toggle_src]/[llength $dmi_resp_toggle_dst] resp_ack=[llength $dmi_resp_ack_src]/[llength $dmi_resp_ack_dst] req_data=[llength $dmi_req_data_src]/[llength $dmi_req_data_dst] req_toggle=[llength $dmi_req_toggle_src]/[llength $dmi_req_toggle_dst] req_ack=[llength $dmi_req_ack_src]/[llength $dmi_req_ack_dst]"
if {[llength $dmi_resp_data_src] > 0 && [llength $dmi_resp_data_dst] > 0} {
    set_max_delay -datapath_only 50.000 -from $dmi_resp_data_src -to $dmi_resp_data_dst
}
if {[llength $dmi_resp_toggle_src] > 0 && [llength $dmi_resp_toggle_dst] > 0} {
    set_max_delay -datapath_only 50.000 -from $dmi_resp_toggle_src -to $dmi_resp_toggle_dst
}
if {[llength $dmi_req_data_src] > 0 && [llength $dmi_req_data_dst] > 0} {
    set_max_delay -datapath_only 50.000 -from $dmi_req_data_src -to $dmi_req_data_dst
}
if {[llength $dmi_req_toggle_src] > 0 && [llength $dmi_req_toggle_dst] > 0} {
    set_max_delay -datapath_only 50.000 -from $dmi_req_toggle_src -to $dmi_req_toggle_dst
}
if {[llength $dmi_resp_ack_src] > 0 && [llength $dmi_resp_ack_dst] > 0} {
    set_max_delay -datapath_only 50.000 -from $dmi_resp_ack_src -to $dmi_resp_ack_dst
}
if {[llength $dmi_req_ack_src] > 0 && [llength $dmi_req_ack_dst] > 0} {
    set_max_delay -datapath_only 50.000 -from $dmi_req_ack_src -to $dmi_req_ack_dst
}
# command_meta and lfm_status_meta are the first registers of explicit
# two-flop clock-domain synchronizers. Only their second stages are timed.
if {[llength $lfm_command_meta] > 0} {
    set_false_path -to $lfm_command_meta
}
if {[llength $lfm_status_meta] > 0} {
    set_false_path -to $lfm_status_meta
}
# The FFN arithmetic unit uses explicit request/response toggle synchronizers.
# Its operand/result holding registers remain stable throughout each crossing.
set ffn_request_meta [get_cells -hier -filter {NAME =~ *i_ffn_packer*i_fp32_cdc*request_meta_reg* && REF_NAME =~ FD*}]
set ffn_response_meta [get_cells -hier -filter {NAME =~ *i_ffn_packer*i_fp32_cdc*response_meta_reg* && REF_NAME =~ FD*}]
set ffn_fp_operands [get_cells -hier -filter {(NAME =~ *i_ffn_packer*i_fp32_cdc/fp_operation_reg* || NAME =~ *i_ffn_packer*i_fp32_cdc/fp_a_reg* || NAME =~ *i_ffn_packer*i_fp32_cdc/fp_b_reg* || NAME =~ *i_ffn_packer*i_fp32_cdc/fp_c_reg*) && REF_NAME =~ FD*}]
set ffn_source_result [get_cells -hier -filter {NAME =~ *i_ffn_packer*i_fp32_cdc*result_o_reg* && REF_NAME =~ FD*}]
# The transform controller now lives with the 85 MHz FP pipe. Its stream
# mailboxes use synchronized toggles while their multi-bit payload registers
# remain stable until the acknowledgement returns. Exclude only those
# intentional mailbox/config publication paths; both clock domains retain
# normal timing internally.
set ffn_mailbox_meta [get_cells -hier -filter {
    NAME =~ *i_ffn_packer* &&
    (NAME =~ *source_request_meta_reg* || NAME =~ *source_ack_meta_reg* ||
     NAME =~ *start_meta_reg* || NAME =~ *block_meta_reg* ||
     NAME =~ *block_ack_meta_reg* || NAME =~ *busy_meta_reg* ||
     NAME =~ *done_meta_reg*) && REF_NAME =~ FD*}]
set ffn_source_payload [get_cells -hier -filter {
    NAME =~ *i_ffn_packer*source_hold_reg* && REF_NAME =~ FD*}]
set ffn_block_payload [get_cells -hier -filter {
    NAME =~ *i_ffn_packer*block_hold_reg* && REF_NAME =~ FD*}]
# The fused residual unit uses the same toggle-plus-stable-payload mailbox
# protocol as the FFN controller.  The *_meta registers are the first stages
# of two-flop synchronizers.  source_hold, result_hold, and sum_hold remain
# unchanged until well after the corresponding synchronized toggle is seen.
set residual_mailbox_meta [get_cells -hier -filter {
     NAME =~ *i_residual* &&
    (NAME =~ *source_request_meta_reg* || NAME =~ *source_ack_meta_reg* ||
     NAME =~ *start_meta_reg* || NAME =~ *result_meta_reg* ||
     NAME =~ *result_ack_meta_reg* || NAME =~ *busy_meta_reg* ||
     NAME =~ *done_meta_reg* || NAME =~ *sum_meta_reg*) && REF_NAME =~ FD*}]
set residual_source_payload [get_cells -hier -filter {
    NAME =~ *i_residual*source_hold_reg* && REF_NAME =~ FD*}]
set residual_start_tag_payload [get_cells -hier -filter {
    NAME =~ *i_residual*start_tag_hold_reg* && REF_NAME =~ FD*}]
set residual_result_payload [get_cells -hier -filter {
    NAME =~ *i_residual*result_hold_reg* && REF_NAME =~ FD*}]
set residual_sum_payload [get_cells -hier -filter {
    NAME =~ *i_residual*sum_hold_reg* && REF_NAME =~ FD*}]
set residual_sum_tag_payload [get_cells -hier -filter {
    NAME =~ *i_residual*sum_tag_hold_reg* && REF_NAME =~ FD*}]
# The deterministic residual FP32 ALU is deliberately a nine-cycle path.
# Its operands are latched and held until fp_latency_count expires, so only
# the paths from those holding registers to fp_result receive this exception.
set residual_fp_operands [get_cells -hier -filter {
    NAME =~ *i_residual* &&
    (NAME =~ *fp_operation_hold_reg* || NAME =~ *fp_a_hold_reg* ||
     NAME =~ *fp_b_hold_reg*) && REF_NAME =~ FD*}]
set residual_fp_result [get_cells -hier -filter {
    NAME =~ *i_residual*fp_result_reg* && REF_NAME =~ FD*}]
set ffn_config_sources [get_cells -hier -filter {
    NAME =~ *i_lfm_matvec_accelerator* &&
    (NAME =~ *ne0_reg* || NAME =~ *ne1_reg* ||
     NAME =~ *scalar_low_reg* || NAME =~ *scalar_high_reg* ||
     NAME =~ *output_compact_config_reg* ||
     NAME =~ *attention_mode_reg*) &&
    REF_NAME =~ FD*}]
# ndmreset_n asynchronously asserts the DDR and FP-domain reset synchronizers.
# Their deassertion is intentionally synchronous, so recovery/removal timing
# applies to the generated resets, not to the synchronizer clear pins.
set lfm_reset_sync_clear [get_pins -hier -filter {NAME =~ *lfm_ddr_reset_sync_reg*/CLR || NAME =~ *lfm_fp_reset_sync_reg*/CLR}]
if {[llength $lfm_reset_sync_clear] > 0} {
    set_false_path -to $lfm_reset_sync_clear
}
# The platform reset generator is clocked by the CPU MMCM output, while the
# MIG AXI reset replicas are the first synchronization stage in clk_pll_i.
# Only this reset publication crossing is asynchronous; downstream MIG reset
# stages remain timed normally.
set mig_axi_reset_first_stage [get_cells -hier -filter {(NAME =~ *i_ddr*u_axi_mc/areset_d1_reg* || NAME =~ *i_ddr*u_axi_mc*register_slice_inst/reset_reg) && REF_NAME =~ FD*}]
if {[llength $mig_axi_reset_first_stage] > 0} {
    set_false_path -to $mig_axi_reset_first_stage
}
if {[llength $ffn_request_meta] > 0} { set_false_path -to $ffn_request_meta }
if {[llength $ffn_response_meta] > 0} { set_false_path -to $ffn_response_meta }
if {[llength $ffn_fp_operands] > 0} { set_false_path -to $ffn_fp_operands }
if {[llength $ffn_source_result] > 0} { set_false_path -to $ffn_source_result }
if {[llength $ffn_mailbox_meta] > 0} { set_false_path -to $ffn_mailbox_meta }
if {[llength $ffn_source_payload] > 0} {
    set_false_path -from $ffn_source_payload
}
if {[llength $ffn_block_payload] > 0} {
    set_false_path -from $ffn_block_payload
}
puts "LFM_RESIDUAL_CDC meta=[llength $residual_mailbox_meta] source=[llength $residual_source_payload] start_tag=[llength $residual_start_tag_payload] result=[llength $residual_result_payload] sum=[llength $residual_sum_payload] sum_tag=[llength $residual_sum_tag_payload]"
puts "LFM_RESIDUAL_FP_MULTICYCLE operands=[llength $residual_fp_operands] result=[llength $residual_fp_result]"
if {[llength $residual_mailbox_meta] > 0} {
    set_false_path -to $residual_mailbox_meta
}
if {[llength $residual_source_payload] > 0} {
    set_false_path -from $residual_source_payload
}
if {[llength $residual_start_tag_payload] > 0} {
    set_false_path -from $residual_start_tag_payload
}
if {[llength $residual_result_payload] > 0} {
    set_false_path -from $residual_result_payload
}
if {[llength $residual_sum_payload] > 0} {
    set_false_path -from $residual_sum_payload
}
if {[llength $residual_sum_tag_payload] > 0} {
    set_false_path -from $residual_sum_tag_payload
}
if {[llength $residual_fp_operands] > 0 && [llength $residual_fp_result] > 0} {
    set_multicycle_path -setup 9 -from $residual_fp_operands -to $residual_fp_result
    set_multicycle_path -hold 8 -from $residual_fp_operands -to $residual_fp_result
}
if {[llength $ffn_config_sources] > 0} {
    set_false_path -from $ffn_config_sources -to [get_clocks clk_out1_xlnx_clk_gen]
}
if {[llength $lfm_block_dot] > 0 && [llength $lfm_scaled_term] > 0} {
    set_multicycle_path -setup 2 -from $lfm_block_dot -to $lfm_scaled_term
    set_multicycle_path -hold 1 -from $lfm_block_dot -to $lfm_scaled_term
}
if {[llength $lfm_scale_align] > 0 && [llength $lfm_scale_product] > 0} {
    set_multicycle_path -setup 2 -from $lfm_scale_align -to $lfm_scale_product
    set_multicycle_path -hold 1 -from $lfm_scale_align -to $lfm_scale_product
}
if {[llength $lfm_row_block] > 0 && [llength $lfm_scale_align] > 0} {
    set_multicycle_path -setup 2 -from $lfm_row_block -to $lfm_scale_align
    set_multicycle_path -hold 1 -from $lfm_row_block -to $lfm_scale_align
}
if {[llength $lfm_scale_product] > 0 && [llength $lfm_scaled_term] > 0} {
    set_multicycle_path -setup 2 -from $lfm_scale_product -to $lfm_scaled_term
    set_multicycle_path -hold 1 -from $lfm_scale_product -to $lfm_scaled_term
}

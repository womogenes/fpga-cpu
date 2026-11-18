# Accelerator timing exceptions, ported from the CVA6 build
# (proj/cva6/scripts/lfm_multicycle_hook.tcl): multicycle arithmetic paths,
# quasi-static configuration, and the FFN/residual FP32 clock crossings.
# Sourced after synth_design so the cell queries resolve.
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
set lfm_reader_state [get_cells -hier -filter {
    NAME =~ *i_lfm_matvec_accelerator/state_reg* && REF_NAME =~ FD*}]
set lfm_reader_bookkeeping [get_cells -hier -filter {
    NAME =~ *i_lfm_matvec_accelerator*i_reader* &&
    (NAME =~ *beat_offset_reg* || NAME =~ *beat_bytes0_reg* ||
     NAME =~ *beat_bytes1_reg* || NAME =~ *beat_data0_reg* ||
     NAME =~ *beat_data1_reg*) && REF_NAME =~ FD*}]
set lfm_activation_rams [get_cells -hier -filter {
    NAME =~ *i_lfm_matvec_accelerator/wide_q* && REF_NAME =~ RAMB*}]
set lfm_mode_static_targets [get_cells -hier -filter {
    NAME =~ *i_lfm_matvec_accelerator* &&
    ((NAME =~ *input_q* && REF_NAME =~ RAMB*) ||
     (NAME =~ *pend_weight_scale_reg* && REF_NAME =~ FD*))}]
puts "LFM_MULTICYCLE block_dot=[llength $lfm_block_dot] scale_align=[llength $lfm_scale_align] scale_product=[llength $lfm_scale_product] scaled_term=[llength $lfm_scaled_term]"
puts "LFM_READER_MULTICYCLE states=[llength $lfm_reader_state] bookkeeping=[llength $lfm_reader_bookkeeping] mode_static=[llength $lfm_mode_static_targets]"
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
# The deterministic residual FP32 ALU is deliberately a ten-cycle path.
# Its operands are latched and held until fp_latency_count expires, so only
# the paths from those holding registers to fp_result receive this exception.
set residual_fp_operands [get_cells -hier -filter {
    NAME =~ *i_residual* &&
    (NAME =~ *fp_operation_hold_reg* || NAME =~ *fp_a_hold_reg* ||
     NAME =~ *fp_b_hold_reg*) && REF_NAME =~ FD*}]
set residual_fp_result [get_cells -hier -filter {
    NAME =~ *i_residual*fp_result_reg* && REF_NAME =~ FD*}]
set residual_fp_retimed [get_cells -hier -filter {
    NAME =~ *i_residual* &&
    (NAME =~ *fp_a_hold_reg*_fret* || NAME =~ *fp_b_hold_reg*_fret* ||
     NAME =~ *fp_result_reg*_fret*) && REF_NAME =~ FD*}]
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
set lfm_reset_sync_clear [get_pins -hier -filter {NAME =~ *rst_sync_reg*/CLR || NAME =~ *fp_rst_sync_reg*/CLR}]
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
puts "LFM_RESIDUAL_FP_MULTICYCLE operands=[llength $residual_fp_operands] result=[llength $residual_fp_result] retimed=[llength $residual_fp_retimed]"
if {[llength $residual_fp_retimed] > 0} {
    error "Residual FP register boundary was retimed; refusing implementation"
}
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
    set_multicycle_path -setup 10 -from $residual_fp_operands -to $residual_fp_result
    set_multicycle_path -hold 9 -from $residual_fp_operands -to $residual_fp_result
}
if {[llength $ffn_config_sources] > 0} {
    set_false_path -from $ffn_config_sources -to [get_clocks -of_objects [get_pins -hier -filter {NAME =~ *i_fp_mmcm/CLKOUT0}]]
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
if {[llength $lfm_reader_state] > 0 && [llength $lfm_reader_bookkeeping] > 0} {
    set_multicycle_path -setup 2 -from $lfm_reader_state -to $lfm_reader_bookkeeping
    set_multicycle_path -hold 1 -from $lfm_reader_state -to $lfm_reader_bookkeeping
}
if {[llength $lfm_reader_state] > 0 && [llength $lfm_activation_rams] > 0} {
    set_multicycle_path -setup 2 -from $lfm_reader_state -to $lfm_activation_rams
    set_multicycle_path -hold 1 -from $lfm_reader_state -to $lfm_activation_rams
}
# The outer stream state is established before an AXI response can arrive and
# remains unchanged until the reader has drained.  Its mode selects therefore
# have two cycles to reach narrow activation addresses and Q1 pending-scale
# controls; data-dependent reader handshakes remain normally single-cycle.
if {[llength $lfm_reader_state] > 0 && [llength $lfm_mode_static_targets] > 0} {
    set_multicycle_path -setup 2 -from $lfm_reader_state -to $lfm_mode_static_targets
    set_multicycle_path -hold 1 -from $lfm_reader_state -to $lfm_mode_static_targets
}

# Solo build: the accelerator/arbiter reset is a single synchronizer flop with
# ~60k loads. Its domain is idle until the controller (which leaves reset six
# cycles later, ctrl_rst_chain) issues a command, so allow four cycles.
set solo_reset [get_cells -hier -filter {NAME =~ rst_sync_reg[2]}]
if {[llength $solo_reset] > 0} {
    set_multicycle_path -setup 4 -from $solo_reset
    set_multicycle_path -hold 3 -from $solo_reset
}
set solo_fp_reset [get_cells -hier -filter {NAME =~ fp_rst_sync_reg[2]}]
if {[llength $solo_fp_reset] > 0} {
    set_multicycle_path -setup 4 -from $solo_fp_reset
    set_multicycle_path -hold 3 -from $solo_fp_reset
}
# Debug-state synchronizers from the 125 MHz FP domain are status-only.
set solo_debug_meta [get_cells -hier -filter {NAME =~ *debug*_meta_reg* && REF_NAME =~ FD*}]
puts "SOLO_HOOK reset=[llength $solo_reset] fp_reset=[llength $solo_fp_reset] debug_meta=[llength $solo_debug_meta]"
if {[llength $solo_debug_meta] > 0} { set_false_path -to $solo_debug_meta }

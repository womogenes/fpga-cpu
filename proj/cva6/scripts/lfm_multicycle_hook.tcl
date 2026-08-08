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
puts "LFM_MULTICYCLE block_dot=[llength $lfm_block_dot] scale_align=[llength $lfm_scale_align] scale_product=[llength $lfm_scale_product] scaled_term=[llength $lfm_scaled_term]"
if {[llength $lfm_block_dot] > 0 && [llength $lfm_scaled_term] > 0} {
    set_multicycle_path -setup 2 -from $lfm_block_dot -to $lfm_scaled_term
    set_multicycle_path -hold 1 -from $lfm_block_dot -to $lfm_scaled_term
}
if {[llength $lfm_scale_align] > 0 && [llength $lfm_scale_product] > 0} {
    set_multicycle_path -setup 2 -from $lfm_scale_align -to $lfm_scale_product
    set_multicycle_path -hold 1 -from $lfm_scale_align -to $lfm_scale_product
}
if {[llength $lfm_scale_product] > 0 && [llength $lfm_scaled_term] > 0} {
    set_multicycle_path -setup 2 -from $lfm_scale_product -to $lfm_scaled_term
    set_multicycle_path -hold 1 -from $lfm_scale_product -to $lfm_scaled_term
}

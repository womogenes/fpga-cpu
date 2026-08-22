open_checkpoint hardware/corev_apu/fpga/ariane.runs/impl_1/cva6_genesys2_hdmi_postroute_margin.dcp

set residual_fp_operands [get_cells -hierarchical -filter {
    NAME =~ *i_residual* &&
    (NAME =~ *fp_operation_hold_reg* || NAME =~ *fp_a_hold_reg* ||
     NAME =~ *fp_b_hold_reg*) && REF_NAME =~ FD*
}]
set residual_fp_result [get_cells -hierarchical -filter {
    NAME =~ *i_residual*fp_result_reg* && REF_NAME =~ FD*
}]
puts "Residual FP cells: operands=[llength $residual_fp_operands] result=[llength $residual_fp_result]"

report_timing -delay_type max -max_paths 100 -nworst 4 \
    -from $residual_fp_operands -to $residual_fp_result \
    -file hardware/corev_apu/fpga/ariane.runs/impl_1/cva6_residual_fp_setup.rpt
report_timing -delay_type min -max_paths 100 -nworst 4 \
    -from $residual_fp_operands -to $residual_fp_result \
    -file hardware/corev_apu/fpga/ariane.runs/impl_1/cva6_residual_fp_hold.rpt
close_design

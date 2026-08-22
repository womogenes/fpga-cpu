open_checkpoint hardware/corev_apu/fpga/ariane.runs/impl_1/cva6_genesys2_hdmi_postroute_margin.dcp

set argmax_cells [get_cells -hierarchical -filter {
    IS_SEQUENTIAL &&
    (NAME =~ *i_lfm_matvec_accelerator/q1_argmax* ||
     NAME =~ *i_lfm_matvec_accelerator/best_score* ||
     NAME =~ *i_lfm_matvec_accelerator/best_token*)
}]
puts "Argmax sequential cells: [llength $argmax_cells]"

report_timing -delay_type max -max_paths 100 -nworst 2 \
    -from $argmax_cells -to $argmax_cells \
    -file hardware/corev_apu/fpga/ariane.runs/impl_1/cva6_argmax_setup.rpt
report_timing -delay_type min -max_paths 100 -nworst 2 \
    -from $argmax_cells -to $argmax_cells \
    -file hardware/corev_apu/fpga/ariane.runs/impl_1/cva6_argmax_hold.rpt
close_design

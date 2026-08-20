open_checkpoint hardware/corev_apu/fpga/ariane.runs/impl_1/cva6_genesys2_hdmi_postroute_physopt.dcp
set attention_cells [get_cells -hierarchical -filter {
    NAME =~ *i_lfm_matvec_accelerator/i_attention* && IS_SEQUENTIAL
}]
report_timing -delay_type max -max_paths 50 -nworst 1 \
    -from $attention_cells -to $attention_cells \
    -file hardware/corev_apu/fpga/ariane.runs/impl_1/cva6_attention_setup.rpt
close_design

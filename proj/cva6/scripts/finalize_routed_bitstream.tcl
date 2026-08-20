set root [file normalize [file join [file dirname [info script]] ..]]
set run_dir [file join $root hardware corev_apu fpga ariane.runs impl_1]
open_checkpoint [file join $run_dir cva6_genesys2_hdmi_routed.dcp]
source [file join $root scripts lfm_multicycle_hook.tcl]
phys_opt_design -hold_fix
report_timing_summary -delay_type min_max -max_paths 20 \
    -file [file join $run_dir cva6_genesys2_hdmi_timing_summary_final.rpt]
write_checkpoint -force [file join $run_dir cva6_genesys2_hdmi_final.dcp]
file mkdir [file join $root build gateware]
write_bitstream -force [file join $root build gateware cva6_genesys2_hdmi.bit]

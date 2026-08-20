set run_dir "hardware/corev_apu/fpga/ariane.runs/impl_1"
set checkpoint "$run_dir/cva6_genesys2_hdmi_postroute_margin.dcp"
set repaired_checkpoint "$run_dir/cva6_genesys2_hdmi_postroute_replication.dcp"
set timing_report "$run_dir/cva6_genesys2_hdmi_timing_summary_postroute_replication.rpt"
set route_report "$run_dir/cva6_genesys2_hdmi_route_status_postroute_replication.rpt"
set bitstream "build/gateware/cva6_genesys2_hdmi_postroute_replication.bit"

open_checkpoint $checkpoint
source scripts/lfm_multicycle_hook.tcl
phys_opt_design -directive AlternateReplication
phys_opt_design -directive AggressiveExplore
report_timing_summary -max_paths 10 -report_unconstrained -warn_on_violation -file $timing_report
report_route_status -file $route_report
write_checkpoint -force $repaired_checkpoint
set worst_path [get_timing_paths -delay_type max -max_paths 1]
set worst_slack [get_property SLACK $worst_path]
set worst_hold_path [get_timing_paths -delay_type min -max_paths 1]
set worst_hold_slack [get_property SLACK $worst_hold_path]
puts "REPLICATION_REPAIR_WNS=$worst_slack"
puts "REPLICATION_REPAIR_WHS=$worst_hold_slack"
if {$worst_slack >= 0.0 && $worst_hold_slack >= 0.0} {
    write_bitstream -force $bitstream
}
close_design

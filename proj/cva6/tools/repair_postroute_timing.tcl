set run_dir "hardware/corev_apu/fpga/ariane.runs/impl_1"
set checkpoint "$run_dir/cva6_genesys2_hdmi_postroute_physopt.dcp"
set repaired_checkpoint "$run_dir/cva6_genesys2_hdmi_postroute_repaired.dcp"
set timing_report "$run_dir/cva6_genesys2_hdmi_timing_summary_postroute_repaired.rpt"
set route_report "$run_dir/cva6_genesys2_hdmi_route_status_postroute_repaired.rpt"
set bitstream "build/gateware/cva6_genesys2_hdmi_postroute_repaired.bit"

open_checkpoint $checkpoint
phys_opt_design -directive AggressiveExplore
report_timing_summary -max_paths 10 -report_unconstrained -warn_on_violation -file $timing_report
report_route_status -file $route_report
write_checkpoint -force $repaired_checkpoint
write_bitstream -force $bitstream
close_design

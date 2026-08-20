set run_dir "hardware/corev_apu/fpga/ariane.runs/impl_1"
set checkpoint "$run_dir/cva6_genesys2_hdmi_postroute_physopt.dcp"
set repaired_checkpoint "$run_dir/cva6_genesys2_hdmi_postroute_margin.dcp"
set timing_report "$run_dir/cva6_genesys2_hdmi_timing_summary_postroute_margin.rpt"
set route_report "$run_dir/cva6_genesys2_hdmi_route_status_postroute_margin.rpt"
set bitstream "build/gateware/cva6_genesys2_hdmi_postroute_margin.bit"

open_checkpoint $checkpoint
source scripts/lfm_multicycle_hook.tcl
set accelerator_clock [get_clocks clk_pll_i]
# Make paths with less than 200 ps of real cushion appear failing so physical
# optimization does not stop at the formally met but unusable 5 ps result.
set_clock_uncertainty -setup 0.200 $accelerator_clock
phys_opt_design -directive AggressiveExplore
# Remove the temporary margin request before producing sign-off reports.
set_clock_uncertainty -setup 0.000 $accelerator_clock
report_timing_summary -max_paths 20 -report_unconstrained \
    -warn_on_violation -file $timing_report
report_route_status -file $route_report
write_checkpoint -force $repaired_checkpoint
set worst_path [get_timing_paths -delay_type max -max_paths 1]
set worst_slack [get_property SLACK $worst_path]
set worst_hold_path [get_timing_paths -delay_type min -max_paths 1]
set worst_hold_slack [get_property SLACK $worst_hold_path]
puts "MARGIN_REPAIR_WNS=$worst_slack"
puts "MARGIN_REPAIR_WHS=$worst_hold_slack"
if {$worst_slack >= 0.0 && $worst_hold_slack >= 0.0} {
    write_bitstream -force $bitstream
}
close_design

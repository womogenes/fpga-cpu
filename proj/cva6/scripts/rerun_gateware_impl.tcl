set project_root [file normalize [file join [file dirname [info script]] ..]]
set fpga_root [file join $project_root hardware corev_apu fpga]

open_project [file join $fpga_root ariane.xpr]
set_property strategy Performance_ExplorePostRoutePhysOpt [get_runs impl_1]
set_property STEPS.OPT_DESIGN.TCL.PRE [file join $project_root scripts lfm_multicycle_hook.tcl] [get_runs impl_1]

reset_run impl_1
launch_runs impl_1 -to_step write_bitstream
wait_on_run impl_1

open_run impl_1
file mkdir [file join $project_root build gateware]
report_timing_summary -delay_type min_max -max_paths 20 -file [file join $project_root build gateware timing_summary.rpt]

set setup_path [get_timing_paths -delay_type max -max_paths 1]
set hold_path [get_timing_paths -delay_type min -max_paths 1]
set setup_slack [get_property SLACK $setup_path]
set hold_slack [get_property SLACK $hold_path]
puts "FINAL_SETUP_SLACK=$setup_slack"
puts "FINAL_HOLD_SLACK=$hold_slack"

if {$setup_slack < 0.0 || $hold_slack < 0.0} {
    error "Implementation failed timing: setup=$setup_slack hold=$hold_slack"
}

file copy -force [file join $fpga_root ariane.runs impl_1 cva6_genesys2_hdmi.bit] [file join $project_root build gateware cva6_genesys2_hdmi.bit]
close_project

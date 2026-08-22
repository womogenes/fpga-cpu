set run_dir "hardware/corev_apu/fpga/ariane.runs/impl_1"
set checkpoint "$run_dir/cva6_genesys2_hdmi_postroute_physopt.dcp"
set repaired_checkpoint "$run_dir/cva6_genesys2_hdmi_postroute_targeted.dcp"
set timing_report "$run_dir/cva6_genesys2_hdmi_timing_summary_postroute_targeted.rpt"
set route_report "$run_dir/cva6_genesys2_hdmi_route_status_postroute_targeted.rpt"
set bitstream "build/gateware/cva6_genesys2_hdmi_postroute_targeted.bit"

open_checkpoint $checkpoint

# The only failing setup path in the post-route checkpoint ends at the q3
# input RAM address. Reroute its longest point-to-point segment without
# perturbing the already timing-clean attention block or the rest of the chip.
set target_nets [get_nets -hierarchical -filter {
    NAME =~ *i_lfm_matvec_accelerator/input_read_b_address*478*11* &&
    NAME !~ *_repN
}]
if {[llength $target_nets] != 1} {
    error "Expected one targeted address net, found [llength $target_nets]: $target_nets"
}

route_design -unroute -nets $target_nets
route_design -nets $target_nets

report_timing_summary -max_paths 20 -report_unconstrained \
    -warn_on_violation -file $timing_report
report_route_status -file $route_report
write_checkpoint -force $repaired_checkpoint

set worst_path [get_timing_paths -delay_type max -max_paths 1]
set worst_slack [get_property SLACK $worst_path]
puts "TARGETED_REPAIR_WNS=$worst_slack"
if {$worst_slack >= 0.0} {
    write_bitstream -force $bitstream
}
close_design

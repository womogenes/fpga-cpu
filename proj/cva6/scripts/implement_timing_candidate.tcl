if {$argc != 3} {
    puts stderr "usage: implement_timing_candidate.tcl INPUT.dcp OUTPUT.bit REPORT.rpt"
    exit 2
}

lassign $argv input_checkpoint output_bit timing_report
set_param general.maxThreads 8

open_checkpoint $input_checkpoint
source [file join [file dirname [info script]] lfm_multicycle_hook.tcl]

opt_design -directive Explore
place_design -directive ExtraTimingOpt
phys_opt_design -directive AggressiveExplore
route_design -directive AggressiveExplore -tns_cleanup
phys_opt_design -directive AggressiveExplore

report_timing_summary -max_paths 20 -report_unconstrained \
    -warn_on_violation -file $timing_report
write_checkpoint -force "${output_bit}.dcp"

set setup_paths [get_timing_paths -delay_type max -max_paths 1]
set hold_paths [get_timing_paths -delay_type min -max_paths 1]
set setup_slack [get_property SLACK [lindex $setup_paths 0]]
set hold_slack [get_property SLACK [lindex $hold_paths 0]]
puts "CANDIDATE_TIMING WNS=$setup_slack WHS=$hold_slack"

if {$setup_slack < 0.0 || $hold_slack < 0.0} {
    puts stderr "CANDIDATE_REJECTED timing is not closed"
    exit 3
}

write_bitstream -force $output_bit
puts "CANDIDATE_ACCEPTED $output_bit"

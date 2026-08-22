if {$argc != 2} {
    puts stderr "usage: report_timing_candidate.tcl INPUT.dcp REPORT.rpt"
    exit 2
}

lassign $argv input_checkpoint timing_report
open_checkpoint $input_checkpoint
source [file join [file dirname [info script]] lfm_multicycle_hook.tcl]
report_timing_summary -delay_type min_max -max_paths 20 \
    -report_unconstrained -warn_on_violation -file $timing_report

set setup_paths [get_timing_paths -delay_type max -max_paths 1]
set hold_paths [get_timing_paths -delay_type min -max_paths 1]
set setup_slack [get_property SLACK [lindex $setup_paths 0]]
set hold_slack [get_property SLACK [lindex $hold_paths 0]]
puts "CANDIDATE_TIMING WNS=$setup_slack WHS=$hold_slack"

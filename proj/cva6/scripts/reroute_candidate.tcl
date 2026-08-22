if {$argc != 3} {
    puts stderr "usage: reroute_candidate.tcl INPUT.dcp OUTPUT.bit REPORT.rpt"
    exit 2
}

lassign $argv input_checkpoint output_bit timing_report
open_checkpoint $input_checkpoint
route_design -directive AggressiveExplore
phys_opt_design -directive AggressiveExplore
report_timing_summary -max_paths 20 -report_unconstrained \
    -warn_on_violation -file $timing_report
write_checkpoint -force "${output_bit}.dcp"
write_bitstream -force $output_bit

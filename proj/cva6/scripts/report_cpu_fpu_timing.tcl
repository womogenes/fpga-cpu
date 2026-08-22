if {$argc != 2} {
    puts stderr "usage: report_cpu_fpu_timing.tcl INPUT.dcp OUTPUT.rpt"
    exit 2
}

lassign $argv input_checkpoint output_report
open_checkpoint $input_checkpoint
set fpu_cells [get_cells -hier -quiet -filter {
    IS_SEQUENTIAL &&
    (NAME =~ "*fpu*" || NAME =~ "*fpr*" ||
     NAME =~ "*ariane_regfile_fpga*")
}]
puts "FPU/register-file sequential cells: [llength $fpu_cells]"
set endpoints [get_pins -quiet -of_objects $fpu_cells -filter {
    REF_PIN_NAME =~ "D*" || REF_PIN_NAME =~ "I*"
}]
puts "FPU/register-file endpoints: [llength $endpoints]"
report_timing -delay_type max -max_paths 200 -nworst 10 \
    -to $endpoints -file $output_report

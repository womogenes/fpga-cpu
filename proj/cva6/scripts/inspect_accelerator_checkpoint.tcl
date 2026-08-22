if {$argc != 2} {
    puts stderr "usage: inspect_accelerator_checkpoint.tcl CHECKPOINT REPORT"
    exit 2
}

set checkpoint [lindex $argv 0]
set report_path [lindex $argv 1]

open_checkpoint $checkpoint

set output [open $report_path w]
puts $output "checkpoint=$checkpoint"

foreach pattern {
    i_soc/i_lfm_matvec_accelerator/*
    i_soc/i_lfm_matvec_accelerator/i_ffn_packer/*
    i_soc/i_lfm_matvec_accelerator/watchdog_fired_reg
    i_soc/i_lfm_matvec_accelerator/watchdog_diagnostic_reg[*]
    i_soc/i_lfm_matvec_accelerator/ffn_debug_state_meta_reg[*]
    i_soc/gen_harts[*]/i_ariane/i_cva6/issue_stage_i/issue_read_operands_i/gen_asic_regfile/i_ariane_regfile/regfile_ram_block[*]/*
    *regfile_ram_block[1]*
} {
    set cells [get_cells -quiet -hierarchical -filter "NAME =~ $pattern"]
    puts $output "cells $pattern [llength $cells]"
}

foreach primitive {LUT1 LUT2 LUT3 LUT4 LUT5 LUT6 FDRE FDCE DSP48E1 RAMB18E1 RAMB36E1} {
    set cells [get_cells -quiet -hierarchical -filter \
        "NAME =~ i_soc/i_lfm_matvec_accelerator/* && REF_NAME == $primitive"]
    puts $output "primitive $primitive [llength $cells]"
}

set request_cell [get_cells -quiet \
    i_soc/i_lfm_matvec_accelerator/bus_request_o_reg]
if {[llength $request_cell]} {
    puts $output "bus_request_init=[get_property INIT $request_cell]"
}

close $output
close_design
exit 0

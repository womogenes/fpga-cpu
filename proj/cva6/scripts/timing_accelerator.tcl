set root $::env(CVA6_HDMI_ROOT)
set output_dir "$root/build/timing-accelerator"
set_param general.maxThreads 4

file mkdir $output_dir
create_project -force lfm_accelerator $output_dir -part xc7k325tffg900-2
set_property target_language Verilog [current_project]
set_property include_dirs [list \
    "$root/hardware/vendor/pulp-platform/axi/include" \
    "$root/hardware/vendor/pulp-platform/common_cells/include" \
    "$root/hardware/core/cvfpu/src"] [current_fileset]

read_verilog -sv [list \
    "$root/hardware/vendor/pulp-platform/axi/src/axi_pkg.sv" \
    "$root/hardware/vendor/pulp-platform/axi/src/axi_intf.sv" \
    "$root/hardware/core/cvfpu/src/fpnew_pkg.sv" \
    "$root/hardware/vendor/pulp-platform/common_cells/src/cf_math_pkg.sv" \
    "$root/hardware/vendor/pulp-platform/common_cells/src/lzc.sv" \
    "$root/hardware/core/cvfpu/src/fpnew_classifier.sv" \
    "$root/hardware/core/cvfpu/src/fpnew_rounding.sv" \
    "$root/hardware/core/cvfpu/src/fpnew_fma.sv" \
    "$root/rtl/bonsai_fp32_pipe.sv" \
    "$root/rtl/bonsai_fp32_alu.sv" \
    "$root/rtl/bonsai_fp32_cdc.sv" \
    "$root/rtl/bonsai_ffn_packer.sv" \
    "$root/rtl/bonsai_residual.sv" \
    "$root/rtl/bonsai_attention.sv" \
    "$root/rtl/lfm_axi_byte_reader.sv" \
    "$root/rtl/lfm_matvec_accelerator.sv" \
    "$root/sim/lfm_accelerator_timing_top.sv"]

synth_design -top lfm_accelerator_timing_top -part xc7k325tffg900-2 -flatten_hierarchy rebuilt
create_clock -name accelerator_clk -period 5.000 [get_ports clk_i]
create_clock -name fp_clk -period 11.111 [get_ports fp_clk_i]
set_clock_groups -asynchronous -group [get_clocks accelerator_clk] -group [get_clocks fp_clk]
# This wrapper exists only to route the accelerator in isolation.  Its AXI and
# command ports have no package timing model, so exclude wrapper I/O paths and
# measure the internal 200 MHz register-to-register datapath.
set nonclock_inputs [get_ports -filter {DIRECTION == IN && NAME != clk_i && NAME != fp_clk_i}]
set nonclock_outputs [get_ports -filter {DIRECTION == OUT}]
set_false_path -from $nonclock_inputs
set_false_path -to $nonclock_outputs
set block_dot_registers [get_cells -hier -filter {NAME =~ *block_dot_reg* && REF_NAME =~ FD*}]
set scale_alignment_registers [get_cells -hier -filter {(NAME =~ *scale_mantissa_product* || NAME =~ *scale_shift_reg* || NAME =~ *scale_zero_reg*) && (REF_NAME =~ FD* || REF_NAME == DSP48E1)}]
set scale_product_registers [get_cells -hier -filter {NAME =~ *scale_product_reg* && (REF_NAME =~ FD* || REF_NAME == DSP48E1)}]
set scaled_term_cells [get_cells -hier -filter {NAME =~ *scaled_term* && (REF_NAME =~ FD* || REF_NAME == DSP48E1)}]
set residual_fp_operands [get_cells -hier -filter {
    NAME =~ *i_residual* &&
    (NAME =~ *fp_operation_hold_reg* || NAME =~ *fp_a_hold_reg* ||
     NAME =~ *fp_b_hold_reg*) && REF_NAME =~ FD*}]
set residual_fp_result [get_cells -hier -filter {
    NAME =~ *i_residual*fp_result_reg* && REF_NAME =~ FD*}]
puts "ACCELERATOR_TIMING multicycle block_dot=[llength $block_dot_registers] scale_alignment=[llength $scale_alignment_registers] scale_product=[llength $scale_product_registers] destinations=[llength $scaled_term_cells]"
puts "ACCELERATOR_TIMING residual_fp_multicycle operands=[llength $residual_fp_operands] result=[llength $residual_fp_result]"
set_multicycle_path -setup 2 -from $block_dot_registers -to $scaled_term_cells
set_multicycle_path -hold 1 -from $block_dot_registers -to $scaled_term_cells
set_multicycle_path -setup 2 -from $scale_alignment_registers -to $scale_product_registers
set_multicycle_path -hold 1 -from $scale_alignment_registers -to $scale_product_registers
set_multicycle_path -setup 2 -from $scale_product_registers -to $scaled_term_cells
set_multicycle_path -hold 1 -from $scale_product_registers -to $scaled_term_cells
set_multicycle_path -setup 7 -from $residual_fp_operands -to $residual_fp_result
set_multicycle_path -hold 6 -from $residual_fp_operands -to $residual_fp_result
opt_design
place_design -directive ExtraTimingOpt
phys_opt_design -directive AggressiveExplore
route_design -directive Explore

report_timing_summary -delay_type max -report_unconstrained -check_timing_verbose -max_paths 20 -input_pins -file "$output_dir/timing.rpt"
report_utilization -hierarchical -file "$output_dir/utilization.rpt"
write_checkpoint -force "$output_dir/lfm_accelerator_routed.dcp"

set timing_paths [get_timing_paths -delay_type max -max_paths 1]
if {[llength $timing_paths] == 0} {
    puts "ACCELERATOR_TIMING no constrained paths"
    exit 1
}
set worst_slack [get_property SLACK [lindex $timing_paths 0]]
puts "ACCELERATOR_TIMING WNS=$worst_slack ns"
if {$worst_slack < 0.0} {
    exit 2
}

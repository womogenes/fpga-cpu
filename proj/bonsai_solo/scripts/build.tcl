# Non-project Vivado build of the Bonsai solo image. Run on vivado-v0:
#   SOLO_DDR=1600 vivado -mode batch -source scripts/build.tcl
# SOLO_DDR selects the MIG configuration: 1600 (validated board setting) or
# 1866 (overclock experiment: 1071 ps tCK, CL13/CWL9, 233 MHz UI clock).
set root [file normalize [file dirname [info script]]/..]
set ddr [expr {[info exists ::env(SOLO_DDR)] ? $::env(SOLO_DDR) : 1600}]
set out [expr {[info exists ::env(SOLO_OUT)] ? $::env(SOLO_OUT) : "$root/build/ddr$ddr"}]
set part xc7k325tffg900-2
file mkdir $out
set_param general.maxThreads 16

# ------------------------------------------------------------------ MIG IP
create_project -in_memory -part $part
set prj [read [open $root/vendor/mig/mig_genesys2.prj]]
if {$ddr == 1866} {
    regsub {<TimePeriod>1250</TimePeriod>} $prj {<TimePeriod>1071</TimePeriod>} prj
    regsub {<mrCasLatency name="CAS Latency" >11</mrCasLatency>} $prj \
        {<mrCasLatency name="CAS Latency" >13</mrCasLatency>} prj
    regsub {<mr2CasWriteLatency name="CAS write latency" >8</mr2CasWriteLatency>} $prj \
        {<mr2CasWriteLatency name="CAS write latency" >9</mr2CasWriteLatency>} prj
} elseif {$ddr != 1600} {
    error "SOLO_DDR must be 1600 or 1866"
}
set ipdir $out/ip
file delete -force $ipdir
file mkdir $ipdir
create_ip -name mig_7series -vendor xilinx.com -library ip \
    -module_name xlnx_mig_7_ddr3 -dir $ipdir
set fh [open $ipdir/xlnx_mig_7_ddr3/mig_a.prj w]
puts -nonewline $fh $prj
close $fh
set_property -dict [list CONFIG.XML_INPUT_FILE {mig_a.prj} \
    CONFIG.RESET_BOARD_INTERFACE {Custom} CONFIG.MIG_DONT_TOUCH_PARAM {Custom} \
    CONFIG.BOARD_MIG_PARAM {Custom}] [get_ips xlnx_mig_7_ddr3]
generate_target all [get_ips xlnx_mig_7_ddr3]
synth_ip [get_ips xlnx_mig_7_ddr3]

# ----------------------------------------------------------------- sources
set rtl [list \
    vendor/pulp_axi/axi_pkg.sv vendor/pulp_axi/axi_intf.sv \
    vendor/cvfpu/fpnew_pkg.sv vendor/common_cells/cf_math_pkg.sv \
    vendor/common_cells/lzc.sv vendor/common_cells/spill_register_flushable.sv \
    vendor/common_cells/spill_register.sv vendor/pulp_axi/axi_cut.sv \
    vendor/cvfpu/fpnew_classifier.sv vendor/cvfpu/fpnew_rounding.sv \
    vendor/cvfpu/fpnew_fma.sv \
    rtl/bonsai_fp32_pipe.sv rtl/bonsai_fp32_alu.sv rtl/bonsai_fp32_cdc.sv \
    rtl/bonsai_ffn_packer.sv rtl/bonsai_residual.sv rtl/bonsai_rms_scale.sv \
    rtl/bonsai_attention.sv rtl/bonsai_attention_quad.sv \
    rtl/lfm_axi_byte_reader.sv rtl/lfm_axi_arbiter.sv \
    rtl/lfm_matvec_accelerator.sv rtl/uart_rx.sv rtl/uart_tx.sv \
    rtl/solo_ctrl.sv rtl/solo_core.sv rtl/solo_top.sv]
foreach f $rtl { read_verilog -sv $root/$f }
read_verilog $root/vendor/picorv32/picorv32.v
read_xdc $root/scripts/solo.xdc

synth_design -top solo_top -part $part \
    -include_dirs [list $root/vendor/pulp_axi/include \
                        $root/vendor/common_cells/include $root/vendor/cvfpu] \
    -generic BOOT_HEX=$root/fw/build/boot.hex \
    -flatten_hierarchy rebuilt -keep_equivalent_registers -fanout_limit 400 \
    -resource_sharing off -no_lc -shreg_min_size 5 -retiming
# Same settings as the CVA6 build (Flow_PerfOptimized_high + retiming): the
# accelerator relies on its replicated registers surviving synthesis.
source $root/scripts/timing_hook.tcl
report_utilization -file $out/utilization_synth.rpt
write_checkpoint -force $out/post_synth.dcp

# SOLO_IMPL=explore mirrors the CVA6 build's Performance_ExplorePostRoutePhysOpt.
set impl [expr {[info exists ::env(SOLO_IMPL)] ? $::env(SOLO_IMPL) : "aggressive"}]
opt_design -directive Explore
if {$impl eq "explore"} {
    place_design -directive Explore
    phys_opt_design -directive Explore
    route_design -directive Explore
    phys_opt_design -directive Explore
} else {
    place_design -directive ExtraTimingOpt
    phys_opt_design -directive AggressiveExplore
    route_design -directive AggressiveExplore
    phys_opt_design -directive AggressiveExplore
}
write_checkpoint -force $out/post_route.dcp

report_timing_summary -max_paths 20 -file $out/timing.rpt
report_utilization -hierarchical -hierarchical_depth 4 -file $out/utilization.rpt
report_route_status -file $out/route_status.rpt
report_drc -file $out/drc.rpt
write_bitstream -force $out/bonsai_solo_ddr$ddr.bit
set wns [get_property SLACK [get_timing_paths -max_paths 1 -nworst 1 -setup]]
set whs [get_property SLACK [get_timing_paths -max_paths 1 -nworst 1 -hold]]
puts "SOLO_RESULT ddr=$ddr wns=$wns whs=$whs"

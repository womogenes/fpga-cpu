set cva6_hdmi_root [file normalize $::env(CVA6_HDMI_ROOT)]
set cva6_hdmi_prepared 0

# Keep synthesis below the host's memory ceiling. The accelerator's replicated
# activation BRAMs are inexpensive, but unrestricted parallel synthesis can
# still exhaust RAM while Vivado performs technology mapping.
set_param general.maxThreads 2

proc cva6_hdmi_prepare {command operation} {
    global cva6_hdmi_root
    global cva6_hdmi_prepared
    if {$cva6_hdmi_prepared} {
        return
    }
    set cva6_hdmi_prepared 1
    set axi_typedef "$cva6_hdmi_root/hardware/vendor/pulp-platform/axi/include/axi/typedef.svh"
    add_files -fileset sources_1 -norecurse $axi_typedef
    set axi_typedef_file [get_files -of_objects [get_filesets sources_1] $axi_typedef]
    set_property -dict {file_type {Verilog Header} is_global_include 1} -objects $axi_typedef_file
    read_verilog -sv [glob -directory "$cva6_hdmi_root/rtl" *.sv]
    if {[info exists ::env(CVA6_NATIVE_SD)] && $::env(CVA6_NATIVE_SD) eq "1"} {
        set_property verilog_define NATIVE_SD [current_fileset]
        set_property include_dirs [concat [get_property include_dirs [current_fileset]] "$cva6_hdmi_root/rtl/native_sdio/upstream"] [current_fileset]
        read_verilog [glob -directory "$cva6_hdmi_root/rtl/native_sdio/upstream" *.v]
        read_verilog -sv "$cva6_hdmi_root/rtl/native_sdio/bonsai_native_sdio.sv"
        add_files -fileset constrs_1 -norecurse "$cva6_hdmi_root/constraints/native_sdio.xdc"
    }
    add_files -fileset constrs_1 -norecurse "$cva6_hdmi_root/constraints/hdmi.xdc"
    set_property top cva6_genesys2_hdmi [current_fileset]
    if {[info exists ::env(CVA6_UART_BAUD)]} {
        set_property generic "UART_BAUD=$::env(CVA6_UART_BAUD)" [current_fileset]
    }
    update_compile_order -fileset sources_1
}

trace add execution synth_design enter cva6_hdmi_prepare

# The upstream project performs a complete RTL elaboration before launching a
# second Vivado process for optimized synthesis. Close that preliminary design
# in the guarded low-memory flow so the launcher does not retain its multi-GB
# netlist while the synthesis child reaches technology mapping.
proc cva6_hdmi_release_preliminary_design {command code result operation} {
    if {![info exists ::env(CVA6_LOW_MEMORY)] || $::env(CVA6_LOW_MEMORY) ne "1"} {
        return
    }
    if {[lsearch -exact $command "-rtl"] < 0} {
        return
    }
    if {[catch {close_design} close_error]} {
        puts "WARNING: could not release preliminary RTL design: $close_error"
    } else {
        puts "INFO: released preliminary RTL design before optimized synthesis"
    }
}

trace add execution synth_design leave cva6_hdmi_release_preliminary_design

proc cva6_hdmi_preserve_bitstream {command code result operation} {
    set generated "ariane.runs/impl_1/cva6_genesys2_hdmi.bit"
    set expected "ariane.runs/impl_1/ariane_xilinx.bit"
    if {[file exists $generated]} {
        file copy -force $generated $expected
    }
}

trace add execution write_sdf leave cva6_hdmi_preserve_bitstream

set cva6_hdmi_root [file normalize $::env(CVA6_HDMI_ROOT)]
set cva6_hdmi_prepared 0

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
    add_files -fileset constrs_1 -norecurse "$cva6_hdmi_root/constraints/hdmi.xdc"
    set_property top cva6_genesys2_hdmi [current_fileset]
    if {[info exists ::env(CVA6_UART_BAUD)]} {
        set_property generic "UART_BAUD=$::env(CVA6_UART_BAUD)" [current_fileset]
    }
    update_compile_order -fileset sources_1
}

trace add execution synth_design enter cva6_hdmi_prepare

proc cva6_hdmi_preserve_bitstream {command code result operation} {
    set generated "ariane.runs/impl_1/cva6_genesys2_hdmi.bit"
    set expected "ariane.runs/impl_1/ariane_xilinx.bit"
    if {[file exists $generated]} {
        file copy -force $generated $expected
    }
}

trace add execution write_sdf leave cva6_hdmi_preserve_bitstream

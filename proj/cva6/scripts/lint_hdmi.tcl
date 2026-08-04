set root [file normalize $::env(CVA6_HDMI_ROOT)]
set output [file normalize "$root/build/lint"]
file mkdir $output
create_project hdmi_lint $output -force -part xc7k325tffg900-2
read_verilog -sv [glob -directory "$root/rtl" *.sv]
synth_design -top uart_hdmi_terminal -part xc7k325tffg900-2
check_syntax
report_utilization -hierarchical -hierarchical_depth 3 -file "$output/utilization.rpt"
close_project

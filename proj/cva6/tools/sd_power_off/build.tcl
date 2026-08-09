read_verilog top.v
read_xdc top.xdc
synth_design -top sd_power_off -part xc7k325tffg900-2
opt_design
place_design
route_design
write_bitstream -force sd_power_off.bit

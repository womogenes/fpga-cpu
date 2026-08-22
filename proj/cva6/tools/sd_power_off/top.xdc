set_property -dict { PACKAGE_PIN AE24 IOSTANDARD LVCMOS33 } [get_ports { sd_reset }]
set_property CFGBVS VCCO [current_design]
set_property CONFIG_VOLTAGE 3.3 [current_design]

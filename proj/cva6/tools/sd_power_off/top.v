// Minimal design: hold the Genesys2 SD slot power switch off.
module sd_power_off (output wire sd_reset);
  assign sd_reset = 1'b1;
endmodule

module tmds_serializer (
    input  logic       pixel_clk,
    input  logic       serial_clk,
    input  logic       reset,
    input  logic [9:0] data,
    output wire        serial
);

    wire [1:0] shift;
    wire unused_slave_output;
    wire unused_master_shift_1;
    wire unused_master_shift_2;

    OSERDESE2 #(
        .DATA_RATE_OQ("DDR"),
        .DATA_RATE_TQ("DDR"),
        .DATA_WIDTH(10),
        .SERDES_MODE("MASTER"),
        .TRISTATE_WIDTH(1)
    ) i_master (
        .CLK(serial_clk),
        .CLKDIV(pixel_clk),
        .D1(data[0]),
        .D2(data[1]),
        .D3(data[2]),
        .D4(data[3]),
        .D5(data[4]),
        .D6(data[5]),
        .D7(data[6]),
        .D8(data[7]),
        .OCE(1'b1),
        .RST(reset),
        .SHIFTIN1(shift[0]),
        .SHIFTIN2(shift[1]),
        .T1(1'b0),
        .T2(1'b0),
        .T3(1'b0),
        .T4(1'b0),
        .TBYTEIN(1'b0),
        .TCE(1'b0),
        .OQ(serial),
        .SHIFTOUT1(unused_master_shift_1),
        .SHIFTOUT2(unused_master_shift_2)
    );

    OSERDESE2 #(
        .DATA_RATE_OQ("DDR"),
        .DATA_RATE_TQ("DDR"),
        .DATA_WIDTH(10),
        .SERDES_MODE("SLAVE"),
        .TRISTATE_WIDTH(1)
    ) i_slave (
        .CLK(serial_clk),
        .CLKDIV(pixel_clk),
        .D1(1'b0),
        .D2(1'b0),
        .D3(data[8]),
        .D4(data[9]),
        .D5(1'b0),
        .D6(1'b0),
        .D7(1'b0),
        .D8(1'b0),
        .OCE(1'b1),
        .RST(reset),
        .SHIFTIN1(1'b0),
        .SHIFTIN2(1'b0),
        .T1(1'b0),
        .T2(1'b0),
        .T3(1'b0),
        .T4(1'b0),
        .TBYTEIN(1'b0),
        .TCE(1'b0),
        .OQ(unused_slave_output),
        .SHIFTOUT1(shift[0]),
        .SHIFTOUT2(shift[1])
    );

endmodule

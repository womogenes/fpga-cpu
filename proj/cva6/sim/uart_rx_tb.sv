`timescale 1ns/1ps

module uart_rx_tb;
    localparam integer BAUD = 3000000;
    localparam realtime BIT_PERIOD_NS = 1.0e9 / BAUD;

    logic clk = 0;
    logic reset = 1;
    logic serial = 1;
    logic [7:0] data;
    logic valid;

    always #6.734 clk = ~clk;

    uart_rx #(
        .CLOCK_HZ(74250000),
        .BAUD(BAUD)
    ) dut (
        .clk,
        .reset,
        .serial,
        .data,
        .valid
    );

    task automatic send_byte(input logic [7:0] value);
        serial = 0;
        #(BIT_PERIOD_NS);
        for (int bit_index = 0; bit_index < 8; bit_index++) begin
            serial = value[bit_index];
            #(BIT_PERIOD_NS);
        end
        serial = 1;
        #(BIT_PERIOD_NS);
    endtask

    task automatic expect_byte(input logic [7:0] expected);
        @(posedge valid);
        if (data !== expected)
            $fatal(1, "expected 0x%02x, received 0x%02x", expected, data);
    endtask

    initial begin
        repeat (8) @(posedge clk);
        reset = 0;
        fork
            begin
                send_byte(8'h55);
                send_byte(8'ha6);
                send_byte(8'h0a);
            end
            begin
                expect_byte(8'h55);
                expect_byte(8'ha6);
                expect_byte(8'h0a);
            end
        join
        $display("uart_rx_tb passed at %0d baud", BAUD);
        $finish;
    end
endmodule

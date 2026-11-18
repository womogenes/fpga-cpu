module uart_tx #(
    parameter integer CLOCK_HZ = 200000000,
    parameter integer BAUD = 3000000
) (
    input  logic       clk,
    input  logic       reset,
    input  logic [7:0] data,
    input  logic       valid,
    output logic       ready,
    output logic       serial
);

    localparam integer CLKS_PER_BIT = (CLOCK_HZ + BAUD / 2) / BAUD;
    localparam integer COUNTER_WIDTH = $clog2(CLKS_PER_BIT + 1);

    // bits_left counts the remaining bit periods: start, eight data, stop.
    logic [COUNTER_WIDTH-1:0] counter;
    logic [3:0] bits_left;
    logic [8:0] shift;

    assign ready = bits_left == 4'd0;

    always_ff @(posedge clk) begin
        if (reset) begin
            serial    <= 1'b1;
            bits_left <= 4'd0;
            counter   <= '0;
            shift     <= '1;
        end else if (bits_left == 4'd0) begin
            serial <= 1'b1;
            if (valid) begin
                serial    <= 1'b0;
                shift     <= {1'b1, data};
                bits_left <= 4'd10;
                counter   <= '0;
            end
        end else if (counter == COUNTER_WIDTH'(CLKS_PER_BIT - 1)) begin
            counter   <= '0;
            bits_left <= bits_left - 4'd1;
            if (bits_left != 4'd1) begin
                serial <= shift[0];
                shift  <= {1'b1, shift[8:1]};
            end
        end else begin
            counter <= counter + 1'b1;
        end
    end

endmodule

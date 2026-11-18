module uart_rx #(
    parameter integer CLOCK_HZ = 74250000,
    parameter integer BAUD = 3000000
) (
    input  logic       clk,
    input  logic       reset,
    input  logic       serial,
    output logic [7:0] data,
    output logic       valid
);

    localparam integer CLKS_PER_BIT = (CLOCK_HZ + BAUD / 2) / BAUD;
    localparam integer COUNTER_WIDTH = $clog2(CLKS_PER_BIT + 1);

    (* ASYNC_REG = "TRUE" *) logic serial_meta;
    (* ASYNC_REG = "TRUE" *) logic serial_sync;
    logic [COUNTER_WIDTH-1:0] counter;
    logic [2:0] bit_index;
    logic [7:0] shift;

    typedef enum logic [1:0] {IDLE, START, DATA_BITS, STOP} state_t;
    state_t state;

    always_ff @(posedge clk) begin
        serial_meta <= serial;
        serial_sync <= serial_meta;
        valid <= 1'b0;

        if (reset) begin
            state <= IDLE;
            counter <= '0;
            bit_index <= '0;
            shift <= '0;
            data <= '0;
        end else begin
            case (state)
                IDLE: begin
                    if (!serial_sync) begin
                        counter <= CLKS_PER_BIT / 2;
                        state <= START;
                    end
                end
                START: begin
                    if (counter != 0) begin
                        counter <= counter - 1'b1;
                    end else if (!serial_sync) begin
                        counter <= CLKS_PER_BIT - 1;
                        bit_index <= 0;
                        state <= DATA_BITS;
                    end else begin
                        state <= IDLE;
                    end
                end
                DATA_BITS: begin
                    if (counter != 0) begin
                        counter <= counter - 1'b1;
                    end else begin
                        shift[bit_index] <= serial_sync;
                        counter <= CLKS_PER_BIT - 1;
                        if (bit_index == 7) begin
                            state <= STOP;
                        end else begin
                            bit_index <= bit_index + 1'b1;
                        end
                    end
                end
                STOP: begin
                    if (counter != 0) begin
                        counter <= counter - 1'b1;
                    end else begin
                        if (serial_sync) begin
                            data <= shift;
                            valid <= 1'b1;
                        end
                        state <= IDLE;
                    end
                end
                default: state <= IDLE;
            endcase
        end
    end

endmodule

module byte_fifo #(
    parameter integer DEPTH = 64
) (
    input  logic       clk,
    input  logic       reset,
    input  logic [7:0] push_data,
    input  logic       push,
    input  logic       pop,
    output logic [7:0] data,
    output logic       valid,
    output logic       overflow
);

    localparam integer ADDRESS_WIDTH = $clog2(DEPTH);
    logic [7:0] memory [0:DEPTH-1];
    logic [ADDRESS_WIDTH-1:0] write_pointer;
    logic [ADDRESS_WIDTH-1:0] read_pointer;
    logic [ADDRESS_WIDTH:0] count;
    wire full = count == DEPTH;
    wire empty = count == 0;
    wire do_pop = pop && !empty;
    wire do_push = push && (!full || do_pop);

    assign data = memory[read_pointer];
    assign valid = !empty;

    always_ff @(posedge clk) begin
        if (reset) begin
            write_pointer <= 0;
            read_pointer <= 0;
            count <= 0;
            overflow <= 1'b0;
        end else begin
            if (do_push) begin
                memory[write_pointer] <= push_data;
                write_pointer <= write_pointer + 1'b1;
            end
            if (do_pop)
                read_pointer <= read_pointer + 1'b1;
            case ({do_push, do_pop})
                2'b10: count <= count + 1'b1;
                2'b01: count <= count - 1'b1;
                default: begin end
            endcase
            if (push && full && !do_pop)
                overflow <= 1'b1;
        end
    end

endmodule

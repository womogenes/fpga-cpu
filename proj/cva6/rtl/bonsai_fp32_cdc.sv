// One-operation clock-domain bridge for the sequential FFN arithmetic path.
// Operands remain stable from request publication through response receipt;
// two-flop toggle synchronizers provide control ordering in both directions.
module bonsai_fp32_cdc (
    input  logic        source_clk_i,
    input  logic        fp_clk_i,
    input  logic        source_rst_ni,
    input  logic        fp_rst_ni,
    input  logic [1:0]  operation_i,
    input  logic [31:0] a_i,
    input  logic [31:0] b_i,
    input  logic [31:0] c_i,
    input  logic        in_valid_i,
    output logic        in_ready_o,
    output logic [31:0] result_o,
    output logic        out_valid_o,
    input  logic        out_ready_i
);
    logic [1:0] source_operation;
    logic [31:0] source_a;
    logic [31:0] source_b;
    logic [31:0] source_c;
    logic request_toggle;
    logic source_busy;
    logic response_toggle;
    logic [31:0] fp_result_hold;

    (* async_reg = "true" *) logic response_meta;
    (* async_reg = "true" *) logic response_sync;
    (* async_reg = "true" *) logic request_meta;
    (* async_reg = "true" *) logic request_sync;
    logic request_seen;
    logic [1:0] fp_operation;
    logic [31:0] fp_a;
    logic [31:0] fp_b;
    logic [31:0] fp_c;
    logic fp_in_valid;
    logic fp_in_ready;
    logic [31:0] fp_result;
    logic fp_out_valid;

    assign in_ready_o = !source_busy && !out_valid_o;

    always_ff @(posedge source_clk_i or negedge source_rst_ni) begin
        if (!source_rst_ni) begin
            source_operation <= 0;
            source_a <= 0;
            source_b <= 0;
            source_c <= 0;
            request_toggle <= 0;
            source_busy <= 0;
            response_meta <= 0;
            response_sync <= 0;
            result_o <= 0;
            out_valid_o <= 0;
        end else begin
            response_meta <= response_toggle;
            response_sync <= response_meta;
            if (in_valid_i && in_ready_o) begin
                source_operation <= operation_i;
                source_a <= a_i;
                source_b <= b_i;
                source_c <= c_i;
                request_toggle <= ~request_toggle;
                source_busy <= 1'b1;
            end
            if (source_busy && response_sync == request_toggle &&
                !out_valid_o) begin
                result_o <= fp_result_hold;
                out_valid_o <= 1'b1;
                source_busy <= 1'b0;
            end
            if (out_valid_o && out_ready_i)
                out_valid_o <= 1'b0;
        end
    end

    always_ff @(posedge fp_clk_i or negedge fp_rst_ni) begin
        if (!fp_rst_ni) begin
            request_meta <= 0;
            request_sync <= 0;
            request_seen <= 0;
            fp_operation <= 0;
            fp_a <= 0;
            fp_b <= 0;
            fp_c <= 0;
            fp_in_valid <= 0;
            fp_result_hold <= 0;
            response_toggle <= 0;
        end else begin
            request_meta <= request_toggle;
            request_sync <= request_meta;
            fp_in_valid <= 1'b0;
            if (request_sync != request_seen && fp_in_ready) begin
                fp_operation <= source_operation;
                fp_a <= source_a;
                fp_b <= source_b;
                fp_c <= source_c;
                fp_in_valid <= 1'b1;
                request_seen <= request_sync;
            end
            if (fp_out_valid) begin
                fp_result_hold <= fp_result;
                response_toggle <= request_seen;
            end
        end
    end

    bonsai_fp32_pipe i_fp32 (
        .clk_i(fp_clk_i),
        .rst_ni(fp_rst_ni),
        .flush_i(1'b0),
        .operation_i(fp_operation),
        .a_i(fp_a),
        .b_i(fp_b),
        .c_i(fp_c),
        .in_valid_i(fp_in_valid),
        .in_ready_o(fp_in_ready),
        .result_o(fp_result),
        .out_valid_o(fp_out_valid),
        .out_ready_i(1'b1)
    );
endmodule

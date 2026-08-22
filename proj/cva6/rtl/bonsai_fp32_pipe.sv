module bonsai_fp32_pipe (
    input  logic        clk_i,
    input  logic        rst_ni,
    input  logic        flush_i,
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
    logic [2:0][31:0] operands;
    fpnew_pkg::operation_e operation;
    fpnew_pkg::status_t status;
    logic extension_bit;
    logic tag;
    logic mask;
    logic aux;
    logic busy;
    logic early_valid;

    always_comb begin
        operands = {c_i, b_i, a_i};
        case (operation_i)
            2'd0: begin
                operation = fpnew_pkg::ADD;
                // fpnew implements ADD as operand[1] + operand[2].
                operands = {b_i, a_i, 32'h3f80_0000};
            end
            2'd1: operation = fpnew_pkg::MUL;
            default: operation = fpnew_pkg::FMADD;
        endcase
    end

    fpnew_fma #(
        .FpFormat(fpnew_pkg::FP32),
        .NumPipeRegs(3),
        .PipeConfig(fpnew_pkg::DISTRIBUTED),
        .TagType(logic),
        .AuxType(logic)
    ) i_fma (
        .clk_i,
        .rst_ni,
        .operands_i(operands),
        .is_boxed_i(3'b111),
        .rnd_mode_i(fpnew_pkg::RNE),
        .op_i(operation),
        .op_mod_i(1'b0),
        .tag_i(1'b0),
        .mask_i(1'b1),
        .aux_i(1'b0),
        .in_valid_i,
        .in_ready_o,
        .flush_i,
        .result_o,
        .status_o(status),
        .extension_bit_o(extension_bit),
        .tag_o(tag),
        .mask_o(mask),
        .aux_o(aux),
        .out_valid_o,
        .out_ready_i,
        .busy_o(busy),
        .reg_ena_i(3'b000),
        .early_out_valid_o(early_valid)
    );
endmodule

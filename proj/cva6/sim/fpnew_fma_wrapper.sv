module fpnew_fma_wrapper (
    input  logic        clk_i,
    input  logic        rst_ni,
    input  logic [31:0] operand_a_i,
    input  logic [31:0] operand_b_i,
    input  logic [31:0] operand_c_i,
    input  logic [1:0]  operation_i,
    input  logic [7:0]  tag_i,
    input  logic        in_valid_i,
    output logic        in_ready_o,
    input  logic        out_ready_i,
    output logic [31:0] result_o,
    output logic [7:0]  tag_o,
    output logic        out_valid_o
);
  fpnew_pkg::operation_e operation;
  always_comb begin
    unique case (operation_i)
      2'd0: operation = fpnew_pkg::ADD;
      2'd1: operation = fpnew_pkg::MUL;
      default: operation = fpnew_pkg::FMADD;
    endcase
  end

  fpnew_pkg::status_t status;
  logic extension_bit;
  logic mask;
  logic aux;
  logic busy;
  logic early_valid;
  logic [2:0][31:0] operands;
  assign operands[0] = operand_a_i;
  assign operands[1] = operand_b_i;
  assign operands[2] = operand_c_i;

  fpnew_fma #(
      .FpFormat    (fpnew_pkg::FP32),
      .NumPipeRegs (3),
      .PipeConfig  (fpnew_pkg::DISTRIBUTED),
      .TagType     (logic [7:0]),
      .AuxType     (logic)
  ) dut (
      .clk_i,
      .rst_ni,
      .operands_i       (operands),
      .is_boxed_i       ('1),
      .rnd_mode_i       (fpnew_pkg::RNE),
      .op_i             (operation),
      .op_mod_i         (1'b0),
      .tag_i,
      .mask_i           (1'b1),
      .aux_i            (1'b0),
      .in_valid_i,
      .in_ready_o,
      .flush_i          (1'b0),
      .result_o,
      .status_o         (status),
      .extension_bit_o  (extension_bit),
      .tag_o,
      .mask_o           (mask),
      .aux_o            (aux),
      .out_valid_o,
      .out_ready_i,
      .busy_o           (busy),
      .early_out_valid_o(early_valid),
      .reg_ena_i        ('0)
  );
endmodule

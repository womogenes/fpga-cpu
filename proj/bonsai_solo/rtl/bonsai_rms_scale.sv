// Bit-exact hardware translation of software/bonsai_main.c:rms_scale_f.
// The input is the positive binary32 residual sum and shift is log2(elements).
// No floating-point operator is used: every rounding and truncation point is
// the same explicit integer operation as the production CVA6 implementation.
module bonsai_rms_scale (
    input  logic        clk_i,
    input  logic        rst_ni,
    input  logic        start_i,
    input  logic [31:0] sum_i,
    input  logic [4:0]  shift_i,
    output logic [31:0] scale_o,
    output logic        busy_o,
    output logic        done_o,
    output logic [2:0]  debug_state_o
);
    localparam logic [31:0] EPSILON_BITS = 32'h3586_37bd;

    typedef enum logic [2:0] {
        S_IDLE,
        S_INPUT_ADD,
        S_INPUT_ROUND,
        S_DIVIDE_SETUP,
        S_DIVIDE,
        S_SQRT_ALIGN,
        S_SQRT_RUN,
        S_FINISH
    } state_t;

    state_t state;
    logic [31:0] divisor;
    logic [63:0] dividend;
    logic [63:0] quotient;
    logic [63:0] divide_remainder;
    logic [5:0] divide_bit;
    logic signed [9:0] half_exponent;
    logic [63:0] radicand;
    logic [63:0] sqrt_remainder;
    logic [63:0] sqrt_result;
    logic [63:0] sqrt_bit;
    logic [31:0] root;
    logic [31:0] mean_bits;
    logic [31:0] x_bits;
    logic [7:0] add_exponent;
    logic [27:0] add_aligned_a;
    logic [27:0] add_aligned_b;

    logic [64:0] divide_trial;
    logic divide_take;
    logic [63:0] divide_quotient_next;
    logic [63:0] sqrt_trial;
    logic sqrt_take;
    logic [63:0] sqrt_result_next;

    function automatic logic [31:0] mean_input_bits(
        input logic [31:0] sum,
        input logic [4:0] shift
    );
        logic [7:0] exponent;
        logic [23:0] significand;
        logic [5:0] down;
        logic [31:0] mean_bits;
        begin
            exponent = sum[30:23];
            if (exponent > shift) begin
                mean_bits = sum - ({27'b0, shift} << 23);
            end else begin
                significand = {1'b1, sum[22:0]};
                down = {1'b0, shift} + 1'b1 - exponent;
                mean_bits = down < 24 ? significand >> down : 0;
            end
            mean_input_bits = mean_bits;
        end
    endfunction

    function automatic logic [4:0] leading_index(input logic [31:0] value);
        logic [4:0] found;
        begin
            found = 0;
            for (integer bit_index = 0; bit_index < 32;
                 bit_index = bit_index + 1)
                if (value[bit_index]) found = bit_index[4:0];
            leading_index = found;
        end
    endfunction

    // Consume the dividend from the MSB with fixed shifts.  This is the same
    // restoring division as the original variable-index implementation, but
    // avoids synthesizing a 64:1 bit-select mux and a 64-bit barrel shifter.
    assign divide_trial = {divide_remainder, dividend[63]};
    assign divide_take = divide_trial >= {33'b0, divisor};
    assign divide_quotient_next = {quotient[62:0], divide_take};
    assign sqrt_trial = sqrt_result + sqrt_bit;
    assign sqrt_take = sqrt_remainder >= sqrt_trial;
    // sqrt_bit advances by two positions per iteration. Every accepted digit
    // is therefore above the next iteration's one-hot bit, so the shifted
    // partial root and sqrt_bit are disjoint. OR is exactly equivalent to the
    // addition here and avoids a 33-CARRY4 chain on the 200 MHz update path.
    assign sqrt_result_next = sqrt_take ?
        (sqrt_result >> 1) | sqrt_bit : sqrt_result >> 1;

    assign busy_o = state != S_IDLE;
    assign debug_state_o = state;

    always_ff @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
            state <= S_IDLE;
            scale_o <= 0;
            done_o <= 0;
            divisor <= 0;
            dividend <= 0;
            quotient <= 0;
            divide_remainder <= 0;
            divide_bit <= 0;
            half_exponent <= 0;
            radicand <= 0;
            sqrt_remainder <= 0;
            sqrt_result <= 0;
            sqrt_bit <= 0;
            root <= 0;
            mean_bits <= 0;
            x_bits <= 0;
            add_exponent <= 0;
            add_aligned_a <= 0;
            add_aligned_b <= 0;
        end else begin
            done_o <= 1'b0;
            case (state)
                S_IDLE: if (start_i) begin
                    mean_bits <= mean_input_bits(sum_i, shift_i);
                    state <= S_INPUT_ADD;
                end

                S_INPUT_ADD: begin
                    logic [7:0] exponent_a;
                    logic [7:0] exponent_b;
                    logic [7:0] exponent_swap;
                    logic [24:0] significand_a;
                    logic [24:0] significand_b;
                    logic [24:0] significand_swap;
                    logic [7:0] difference;
                    logic [27:0] aligned_a;
                    logic [27:0] aligned_b;
                    logic [27:0] discarded_mask;
                    if (mean_bits == 0) begin
                        x_bits <= EPSILON_BITS;
                        state <= S_DIVIDE_SETUP;
                    end else begin
                        exponent_a = mean_bits[30:23];
                        exponent_b = EPSILON_BITS[30:23];
                        significand_a = {1'b0, 1'b1, mean_bits[22:0]};
                        significand_b =
                            {1'b0, 1'b1, EPSILON_BITS[22:0]};
                        if (exponent_a < exponent_b) begin
                            exponent_swap = exponent_a;
                            exponent_a = exponent_b;
                            exponent_b = exponent_swap;
                            significand_swap = significand_a;
                            significand_a = significand_b;
                            significand_b = significand_swap;
                        end
                        difference = exponent_a - exponent_b;
                        aligned_a = significand_a << 3;
                        aligned_b = significand_b << 3;
                        if (difference >= 28) begin
                            aligned_b = aligned_b != 0;
                        end else if (difference != 0) begin
                            discarded_mask =
                                (28'd1 << difference) - 1'b1;
                            aligned_b = (aligned_b >> difference) |
                                ((aligned_b & discarded_mask) != 0);
                        end
                        add_exponent <= exponent_a;
                        add_aligned_a <= aligned_a;
                        add_aligned_b <= aligned_b;
                        state <= S_INPUT_ROUND;
                    end
                end

                S_INPUT_ROUND: begin
                    logic [7:0] exponent;
                    logic [28:0] total;
                    logic [24:0] significand;
                    logic [2:0] remainder;
                    exponent = add_exponent;
                    total = {1'b0, add_aligned_a} +
                            {1'b0, add_aligned_b};
                    if (total[27]) begin
                        total = (total >> 1) | total[0];
                        exponent = exponent + 1'b1;
                    end
                    significand = total >> 3;
                    remainder = total[2:0];
                    if (remainder > 3'd4 ||
                        (remainder == 3'd4 && significand[0]))
                        significand = significand + 1'b1;
                    if (significand == 25'h1_000000) begin
                        significand = significand >> 1;
                        exponent = exponent + 1'b1;
                    end
                    x_bits <= {1'b0, exponent, significand[22:0]};
                    state <= S_DIVIDE_SETUP;
                end

                S_DIVIDE_SETUP: begin
                    logic signed [9:0] unbiased;
                    logic signed [9:0] half_value;
                    logic odd;
                    unbiased = $signed({2'b0, x_bits[30:23]}) - 10'sd127;
                    half_value = unbiased >= 0 ? unbiased / 2 :
                        -((-unbiased + 1) / 2);
                    odd = unbiased - 2 * half_value;
                    divisor <= {7'b0, 1'b1, x_bits[22:0]} << odd;
                    dividend <= 64'h8000_0000_0000_0000;
                    quotient <= 0;
                    divide_remainder <= 0;
                    divide_bit <= 6'd63;
                    half_exponent <= half_value;
                    state <= S_DIVIDE;
                end

                S_DIVIDE: begin
                    divide_remainder <= divide_take ?
                        divide_trial[63:0] - divisor : divide_trial[63:0];
                    dividend <= dividend << 1;
                    quotient <= divide_quotient_next;
                    if (divide_bit == 0) begin
                        radicand <= divide_quotient_next;
                        sqrt_remainder <= divide_quotient_next;
                        sqrt_result <= 0;
                        sqrt_bit <= 64'h4000_0000_0000_0000;
                        state <= S_SQRT_ALIGN;
                    end else begin
                        divide_bit <= divide_bit - 1'b1;
                    end
                end

                S_SQRT_ALIGN: begin
                    if (sqrt_bit > radicand) begin
                        sqrt_bit <= sqrt_bit >> 2;
                    end else begin
                        state <= S_SQRT_RUN;
                    end
                end

                S_SQRT_RUN: begin
                    sqrt_remainder <= sqrt_take ?
                        sqrt_remainder - sqrt_trial : sqrt_remainder;
                    sqrt_result <= sqrt_result_next;
                    sqrt_bit <= sqrt_bit >> 2;
                    if (sqrt_bit == 1) begin
                        root <= sqrt_result_next[31:0];
                        state <= S_FINISH;
                    end
                end

                S_FINISH: begin
                    logic [4:0] leading;
                    logic [31:0] output_significand;
                    logic signed [10:0] output_exponent;
                    leading = leading_index(root);
                    output_significand = root << (23 - leading);
                    output_exponent = $signed({6'b0, leading}) - 11'sd20 -
                                      half_exponent + 11'sd127;
                    scale_o <= {1'b0, output_exponent[7:0],
                                output_significand[22:0]};
                    done_o <= 1'b1;
                    state <= S_IDLE;
                end

                default: state <= S_IDLE;
            endcase
        end
    end
endmodule

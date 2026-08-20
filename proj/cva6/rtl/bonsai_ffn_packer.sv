// Convert a streamed pair of FFN up/gate vectors into the exact Q8_0 stream
// consumed by the matrix engine. The up vector is retained in block RAM;
// gate values are accepted one at a time while the shared pipelined FP32 FMA
// unit evaluates SiLU and the resulting 32-value block is quantized.
module bonsai_ffn_packer #(
    parameter integer ELEMENTS = 6144
) (
    input  logic         clk_i,
    input  logic         fp_clk_i,
    input  logic         rst_ni,
    input  logic         fp_rst_ni,
    input  logic         start_i,
    input  logic         norm_mode_i,
    input  logic         vector_mode_i,
    input  logic         compact_source_i,
    input  logic [12:0]  element_count_i,
    input  logic [31:0]  norm_scale_i,
    input  logic [63:0]  source_data_i,
    input  logic         source_valid_i,
    output logic         source_ready_o,
    output logic [271:0] block_data_o,
    output logic         block_valid_o,
    input  logic         block_ready_i,
    output logic         busy_o,
    output logic         done_o,
    output logic [3:0]   debug_state_o
);
    typedef enum logic [3:0] {
        FFN_IDLE,
        FFN_LOAD_UP,
        FFN_ACCEPT_GATE,
        FFN_WAIT_UP,
        FFN_WAIT_SIGMOID,
        FFN_ACTIVATE_FIRST,
        FFN_ACTIVATE_SECOND,
        FFN_SCALE,
        FFN_QUANTIZE_FIRST,
        FFN_QUANTIZE_SECOND,
        FFN_EMIT,
        FFN_DONE
    } ffn_state_t;

    ffn_state_t state;
    (* ram_style = "block" *) logic [31:0] up_memory [0:ELEMENTS-1];
    logic [31:0] activation [0:31];
    logic [31:0] up_block [0:31];
    logic [31:0] gate_block [0:31];
    logic [31:0] sigmoid_block [0:31];
    logic [31:0] intermediate [0:31];
    logic [31:0] scaled_block [0:31];
    logic [12:0] element_index;
    logic [5:0] block_position;
    logic [31:0] up_value;
    logic [31:0] up_read_data;
    logic [31:0] gate_value;
    logic [31:0] max_absolute;
    logic [271:0] packed_block;

    logic [5:0] fp_step;
    logic [5:0] fp_sent;
    logic [5:0] fp_received;
    logic fp_waiting;
    logic [31:0] fp_a;
    logic [31:0] fp_b;
    logic [31:0] fp_c;
    logic [1:0] fp_operation;
    logic fp_input_valid;
    logic fp_input_ready;
    logic [31:0] fp_result;
    logic fp_result_valid;

    logic sig0_valid;
    logic [4:0] sig0_slot;
    logic signed [31:0] sig0_fixed;
    logic sig1_valid;
    logic [4:0] sig1_slot;
    logic [16:0] sig1_lower;
    logic signed [16:0] sig1_difference;
    logic [13:0] sig1_fraction;
    logic sig2_valid;
    logic [4:0] sig2_slot;
    logic [16:0] sig2_lower;
    logic signed [31:0] sig2_interpolation;
    logic sig3_valid;
    logic [4:0] sig3_slot;
    logic [16:0] sig3_q16;

    logic [31:0] x_value;
    logic [31:0] scaled_value;
    logic signed [31:0] exponent_integer;
    logic [31:0] exponent_float;
    logic [31:0] remainder;
    logic [31:0] remainder_squared;
    logic [31:0] polynomial;
    logic [31:0] exponential;
    logic [31:0] denominator;
    logic [31:0] reciprocal;
    logic [31:0] activation_value;
    logic [31:0] scale_value;
    logic [31:0] quant_reciprocal;
    logic [31:0] sigmoid_value;
    logic signed [31:0] sigmoid_fixed;
    logic [31:0] sigmoid_magnitude;
    logic sigmoid_negative;
    logic [15:0] sigmoid_lower;
    logic [15:0] sigmoid_upper;
    logic [13:0] sigmoid_fraction;
    logic signed [16:0] sigmoid_difference;
    logic [13:0] sigmoid_fraction_pipe;
    logic signed [31:0] sigmoid_interpolation;
    logic [16:0] sigmoid_q16;
    logic exp_underflow;
    logic fp_local_wait;
    logic norm_high_pending;
    logic [31:0] norm_high_word;

    logic [63:0] source_hold;
    logic source_request_toggle;
    logic source_busy;
    (* async_reg = "true" *) logic source_request_meta;
    (* async_reg = "true" *) logic source_request_sync;
    logic source_request_seen;
    (* async_reg = "true" *) logic source_ack_meta;
    (* async_reg = "true" *) logic source_ack_sync;
    logic start_toggle;
    (* async_reg = "true" *) logic start_meta;
    (* async_reg = "true" *) logic start_sync;
    logic start_seen;
    logic [271:0] block_hold;
    logic block_toggle;
    logic block_pending;
    (* async_reg = "true" *) logic block_meta;
    (* async_reg = "true" *) logic block_sync;
    logic block_seen;
    logic block_ack_toggle;
    (* async_reg = "true" *) logic block_ack_meta;
    (* async_reg = "true" *) logic block_ack_sync;
    (* async_reg = "true" *) logic busy_meta;
    (* async_reg = "true" *) logic busy_sync;
    (* async_reg = "true" *) logic done_meta;
    (* async_reg = "true" *) logic done_sync;

    wire core_source_valid = source_request_sync != source_request_seen;
    wire [63:0] core_source_data = source_hold;
    wire core_start = start_sync != start_seen;
    wire paired_source = norm_mode_i || vector_mode_i || compact_source_i;
    wire core_source_ready = (state == FFN_LOAD_UP ||
                              (state == FFN_ACCEPT_GATE && !vector_mode_i)) &&
                             !(paired_source && norm_high_pending);
    wire core_block_valid = state == FFN_EMIT;
    wire core_block_ready = !block_pending;
    wire core_busy = state != FFN_IDLE && state != FFN_DONE;
    wire core_done = state == FFN_DONE && !block_pending;

    wire source_word_available = core_source_valid ||
        (paired_source && norm_high_pending);
    wire [31:0] source_word = paired_source && norm_high_pending ?
        norm_high_word : core_source_data[31:0];

    wire [20:0] sigmoid_position = sigmoid_fixed + 32'sd524288;
    wire [6:0] sigmoid_index = sigmoid_position[20:14];

    function automatic logic signed [31:0] fp32_to_int_rtz(
        input logic [31:0] value);
        logic [7:0] exponent;
        logic [23:0] mantissa;
        logic [31:0] magnitude;
        integer shift;
        begin
            exponent = value[30:23];
            mantissa = {1'b1, value[22:0]};
            if (exponent < 8'd127) begin
                magnitude = 0;
            end else begin
                shift = $signed({24'b0, exponent}) - 150;
                magnitude = shift >= 0 ? {8'b0, mantissa} << shift :
                                         {8'b0, mantissa} >> (-shift);
            end
            fp32_to_int_rtz = value[31] ? -$signed(magnitude) :
                                          $signed(magnitude);
        end
    endfunction

    function automatic logic [31:0] small_int_to_fp32(
        input logic signed [31:0] value);
        logic sign;
        logic [31:0] magnitude;
        logic [31:0] normalized;
        logic [5:0] msb;
        integer bit_index;
        begin
            sign = value < 0;
            magnitude = sign ? -value : value;
            msb = 0;
            for (bit_index = 0; bit_index < 32; bit_index = bit_index + 1)
                if (magnitude[bit_index]) msb = bit_index[5:0];
            if (magnitude == 0)
                small_int_to_fp32 = 0;
            else begin
                normalized = magnitude << (23 - msb);
                small_int_to_fp32 = {sign, 8'(127 + msb), normalized[22:0]};
            end
        end
    endfunction

    function automatic logic [15:0] fp32_to_fp16_truncate(
        input logic [31:0] value);
        logic [7:0] source_exponent;
        logic signed [9:0] exponent;
        begin
            source_exponent = value[30:23];
            exponent = $signed({2'b00, source_exponent}) - 10'sd127 + 10'sd15;
            if (exponent <= 0)
                fp32_to_fp16_truncate = {value[31], 15'b0};
            else if (exponent >= 31)
                fp32_to_fp16_truncate = {value[31], 5'h1f, 10'b0};
            else
                fp32_to_fp16_truncate =
                    {value[31], exponent[4:0], value[22:13]};
        end
    endfunction

    function automatic logic signed [7:0] rounded_fp32_to_i8(
        input logic [31:0] value);
        logic [31:0] adjusted;
        logic signed [31:0] int_value;
        begin
            // The preceding FP operation adds +/-0.5 exactly as round_f.
            adjusted = value;
            int_value = fp32_to_int_rtz(adjusted);
            rounded_fp32_to_i8 = int_value[7:0];
        end
    endfunction

    function automatic logic [31:0] absolute_fp32(input logic [31:0] value);
        absolute_fp32 = {1'b0, value[30:0]};
    endfunction

    function automatic logic [15:0] sigmoid_table_q16(
        input logic [6:0] index);
        begin
            case (index)
                0: sigmoid_table_q16 = 16'd22;
                1: sigmoid_table_q16 = 16'd28;
                2: sigmoid_table_q16 = 16'd36;
                3: sigmoid_table_q16 = 16'd47;
                4: sigmoid_table_q16 = 16'd60;
                5: sigmoid_table_q16 = 16'd77;
                6: sigmoid_table_q16 = 16'd98;
                7: sigmoid_table_q16 = 16'd126;
                8: sigmoid_table_q16 = 16'd162;
                9: sigmoid_table_q16 = 16'd208;
                10: sigmoid_table_q16 = 16'd267;
                11: sigmoid_table_q16 = 16'd342;
                12: sigmoid_table_q16 = 16'd439;
                13: sigmoid_table_q16 = 16'd562;
                14: sigmoid_table_q16 = 16'd720;
                15: sigmoid_table_q16 = 16'd922;
                16: sigmoid_table_q16 = 16'd1179;
                17: sigmoid_table_q16 = 16'd1506;
                18: sigmoid_table_q16 = 16'd1921;
                19: sigmoid_table_q16 = 16'd2446;
                20: sigmoid_table_q16 = 16'd3108;
                21: sigmoid_table_q16 = 16'd3938;
                22: sigmoid_table_q16 = 16'd4971;
                23: sigmoid_table_q16 = 16'd6249;
                24: sigmoid_table_q16 = 16'd7812;
                25: sigmoid_table_q16 = 16'd9702;
                26: sigmoid_table_q16 = 16'd11955;
                27: sigmoid_table_q16 = 16'd14595;
                28: sigmoid_table_q16 = 16'd17625;
                29: sigmoid_table_q16 = 16'd21025;
                30: sigmoid_table_q16 = 16'd24743;
                31: sigmoid_table_q16 = 16'd28693;
                32: sigmoid_table_q16 = 16'd32768;
                33: sigmoid_table_q16 = 16'd36843;
                34: sigmoid_table_q16 = 16'd40793;
                35: sigmoid_table_q16 = 16'd44511;
                36: sigmoid_table_q16 = 16'd47911;
                37: sigmoid_table_q16 = 16'd50941;
                38: sigmoid_table_q16 = 16'd53581;
                39: sigmoid_table_q16 = 16'd55834;
                40: sigmoid_table_q16 = 16'd57724;
                41: sigmoid_table_q16 = 16'd59287;
                42: sigmoid_table_q16 = 16'd60565;
                43: sigmoid_table_q16 = 16'd61598;
                44: sigmoid_table_q16 = 16'd62428;
                45: sigmoid_table_q16 = 16'd63090;
                46: sigmoid_table_q16 = 16'd63615;
                47: sigmoid_table_q16 = 16'd64030;
                48: sigmoid_table_q16 = 16'd64357;
                49: sigmoid_table_q16 = 16'd64614;
                50: sigmoid_table_q16 = 16'd64816;
                51: sigmoid_table_q16 = 16'd64974;
                52: sigmoid_table_q16 = 16'd65097;
                53: sigmoid_table_q16 = 16'd65194;
                54: sigmoid_table_q16 = 16'd65269;
                55: sigmoid_table_q16 = 16'd65328;
                56: sigmoid_table_q16 = 16'd65374;
                57: sigmoid_table_q16 = 16'd65410;
                58: sigmoid_table_q16 = 16'd65438;
                59: sigmoid_table_q16 = 16'd65459;
                60: sigmoid_table_q16 = 16'd65476;
                61: sigmoid_table_q16 = 16'd65489;
                62: sigmoid_table_q16 = 16'd65500;
                63: sigmoid_table_q16 = 16'd65508;
                default: sigmoid_table_q16 = 16'd65514;
            endcase
        end
    endfunction

    function automatic logic [31:0] fp32_to_q16_magnitude(
        input logic [31:0] value);
        logic [30:0] magnitude;
        logic [7:0] exponent;
        logic [23:0] significand;
        logic [31:0] fixed_magnitude;
        integer shift;
        begin
            magnitude = value[30:0];
            exponent = magnitude[30:23];
            significand = {1'b1, magnitude[22:0]};
            if (exponent == 0) begin
                fixed_magnitude = 0;
            end else if (exponent == 8'hff) begin
                fixed_magnitude = 32'h7fff_0000;
            end else begin
                shift = $signed({1'b0, exponent}) - 9'sd134;
                if (shift >= 0)
                    fixed_magnitude = shift > 7 ? 32'h7fff_0000 :
                                      {8'b0, significand} << shift;
                else
                    fixed_magnitude = -shift >= 32 ? 0 :
                                      {8'b0, significand} >> (-shift);
                if (fixed_magnitude > 32'h7fff_0000)
                    fixed_magnitude = 32'h7fff_0000;
            end
            fp32_to_q16_magnitude = fixed_magnitude;
        end
    endfunction

    function automatic logic [31:0] positive_q16_to_fp32(
        input logic [16:0] value);
        logic [4:0] msb;
        logic [23:0] normalized;
        integer bit_index;
        begin
            msb = 0;
            for (bit_index = 0; bit_index < 17; bit_index = bit_index + 1)
                if (value[bit_index]) msb = bit_index[4:0];
            if (value == 0)
                positive_q16_to_fp32 = 0;
            else begin
                normalized = {7'b0, value} << (23 - msb);
                positive_q16_to_fp32 =
                    {1'b0, 8'(127 + msb - 16), normalized[22:0]};
            end
        end
    endfunction

    function automatic logic [31:0] sigmoid_lut_fp32(
        input logic [31:0] value);
        logic [31:0] magnitude;
        logic signed [31:0] fixed_value;
        logic [20:0] position;
        logic [6:0] index;
        logic [13:0] fraction;
        logic [15:0] lower;
        logic [15:0] upper;
        logic signed [16:0] difference;
        logic signed [31:0] interpolation;
        logic [16:0] q16;
        begin
            magnitude = fp32_to_q16_magnitude(value);
            fixed_value = value[31] ? -$signed(magnitude) :
                                      $signed(magnitude);
            if (fixed_value <= -32'sd524288) begin
                q16 = 17'd0;
            end else if (fixed_value >= 32'sd524288) begin
                q16 = 17'd65536;
            end else begin
                position = fixed_value + 32'sd524288;
                index = position[20:14];
                fraction = position[13:0];
                lower = sigmoid_table_q16(index);
                upper = sigmoid_table_q16(index + 1'b1);
                difference = $signed({1'b0, upper}) -
                             $signed({1'b0, lower});
                interpolation = difference * $signed({1'b0, fraction});
                q16 = {1'b0, lower} + (interpolation >>> 14);
            end
            sigmoid_lut_fp32 = positive_q16_to_fp32(q16);
        end
    endfunction

    function automatic logic [31:0] power_of_two_from_n(
        input logic signed [31:0] value,
        input logic force_zero);
        logic signed [31:0] biased;
        begin
            biased = value + 32'sd127;
            if (force_zero || biased <= 0)
                power_of_two_from_n = 0;
            else if (biased >= 255)
                power_of_two_from_n = 32'h7f00_0000;
            else
                power_of_two_from_n = {1'b0, $unsigned(biased[7:0]), 23'b0};
        end
    endfunction

    assign busy_o = busy_sync;
    assign done_o = done_sync;
    assign debug_state_o = state;
    assign source_ready_o = !source_busy;
    assign fp_input_valid =
        ((state == FFN_ACTIVATE_FIRST ||
          state == FFN_ACTIVATE_SECOND ||
          state == FFN_QUANTIZE_FIRST ||
          state == FFN_QUANTIZE_SECOND) && fp_sent < 6'd32) ||
        (state == FFN_SCALE && !fp_waiting && !fp_local_wait);

    always_comb begin
        fp_operation = 2'd2;
        fp_a = 32'b0;
        fp_b = 32'b0;
        fp_c = 32'b0;
        if (state == FFN_ACTIVATE_FIRST) begin
            fp_operation = 2'd1;
            fp_a = (norm_mode_i || vector_mode_i) ?
                up_block[fp_sent] : gate_block[fp_sent];
            fp_b = (norm_mode_i || vector_mode_i) ?
                gate_block[fp_sent] : sigmoid_block[fp_sent];
        end else if (state == FFN_ACTIVATE_SECOND) begin
            fp_operation = 2'd1;
            fp_a = intermediate[fp_sent];
            fp_b = (norm_mode_i || vector_mode_i) ?
                norm_scale_i : up_block[fp_sent];
        end else if (state == FFN_SCALE) begin
            case (fp_step)
                0: begin fp_operation = 2'd1; fp_a = max_absolute;
                         fp_b = 32'h3c01_0204; end // 1/127
                1, 3: begin fp_a = {~scale_value[31], scale_value[30:0]};
                            fp_b = quant_reciprocal; fp_c = 32'h4000_0000; end
                default: begin fp_operation = 2'd1; fp_a = quant_reciprocal;
                               fp_b = polynomial; end
            endcase
        end else if (state == FFN_QUANTIZE_FIRST) begin
                fp_operation = 2'd1;
                fp_a = activation[fp_sent];
                fp_b = quant_reciprocal;
        end else if (state == FFN_QUANTIZE_SECOND) begin
            fp_operation = 2'd0;
            fp_a = scaled_block[fp_sent];
            fp_b = scaled_block[fp_sent][31] ? 32'hbf00_0000 :
                                               32'h3f00_0000;
        end
    end

    // Run the complete transform controller with its FP pipeline. Only the
    // coarse source/block streams cross the clock boundary, instead of paying
    // a request/response CDC round trip for every arithmetic operation.
    bonsai_fp32_pipe i_fp32 (
        .clk_i(fp_clk_i),
        .rst_ni(fp_rst_ni),
        .flush_i(1'b0),
        .operation_i(fp_operation),
        .a_i(fp_a),
        .b_i(fp_b),
        .c_i(fp_c),
        .in_valid_i(fp_input_valid),
        .in_ready_o(fp_input_ready),
        .result_o(fp_result),
        .out_valid_o(fp_result_valid),
        .out_ready_i(1'b1)
    );

    // DDR-domain sides of the one-entry source and result mailboxes.
    always_ff @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
            source_hold <= 0;
            source_request_toggle <= 0;
            source_busy <= 0;
            source_ack_meta <= 0;
            source_ack_sync <= 0;
            start_toggle <= 0;
            block_meta <= 0;
            block_sync <= 0;
            block_seen <= 0;
            block_ack_toggle <= 0;
            block_data_o <= 0;
            block_valid_o <= 0;
            busy_meta <= 0;
            busy_sync <= 0;
            done_meta <= 0;
            done_sync <= 0;
        end else begin
            source_ack_meta <= source_request_seen;
            source_ack_sync <= source_ack_meta;
            block_meta <= block_toggle;
            block_sync <= block_meta;
            busy_meta <= core_busy;
            busy_sync <= busy_meta;
            done_meta <= core_done;
            done_sync <= done_meta;
            if (start_i)
                start_toggle <= ~start_toggle;
            if (source_valid_i && source_ready_o) begin
                source_hold <= source_data_i;
                source_request_toggle <= ~source_request_toggle;
                source_busy <= 1'b1;
            end
            if (source_busy && source_ack_sync == source_request_toggle)
                source_busy <= 1'b0;
            if (!block_valid_o && block_sync != block_seen) begin
                block_data_o <= block_hold;
                block_valid_o <= 1'b1;
                block_seen <= block_sync;
            end
            if (block_valid_o && block_ready_i) begin
                block_valid_o <= 1'b0;
                block_ack_toggle <= block_seen;
            end
        end
    end

    // FP-domain sides. Payloads remain stable until the acknowledgement has
    // crossed back, so only the toggle bits require synchronizer flops.
    always_ff @(posedge fp_clk_i or negedge fp_rst_ni) begin
        if (!fp_rst_ni) begin
            source_request_meta <= 0;
            source_request_sync <= 0;
            source_request_seen <= 0;
            start_meta <= 0;
            start_sync <= 0;
            start_seen <= 0;
            block_hold <= 0;
            block_toggle <= 0;
            block_pending <= 0;
            block_ack_meta <= 0;
            block_ack_sync <= 0;
        end else begin
            source_request_meta <= source_request_toggle;
            source_request_sync <= source_request_meta;
            start_meta <= start_toggle;
            start_sync <= start_meta;
            block_ack_meta <= block_ack_toggle;
            block_ack_sync <= block_ack_meta;
            if (core_start)
                start_seen <= start_sync;
            if (core_source_valid && core_source_ready)
                source_request_seen <= source_request_sync;
            if (block_pending && block_ack_sync == block_toggle)
                block_pending <= 1'b0;
            if (core_block_valid && core_block_ready) begin
                block_hold <= packed_block;
                block_toggle <= ~block_toggle;
                block_pending <= 1'b1;
            end
        end
    end

    // Keep the RAM itself out of the asynchronously-reset control process;
    // this is the canonical synchronous-read block-RAM inference pattern.
    always_ff @(posedge fp_clk_i) begin
        if (state == FFN_LOAD_UP && source_word_available)
            up_memory[element_index] <= source_word;
        up_read_data <= up_memory[element_index];
    end

    // Stream the exact sigmoid lookup through short registered stages. Gate
    // words arrive only every other FP clock (ACCEPT_GATE/WAIT_UP), so this
    // pipeline finishes a block shortly after the final up-memory read without
    // reducing source throughput.
    always_ff @(posedge fp_clk_i or negedge fp_rst_ni) begin
        if (!fp_rst_ni) begin
            sig0_valid <= 1'b0;
            sig1_valid <= 1'b0;
            sig2_valid <= 1'b0;
            sig3_valid <= 1'b0;
        end else begin
            sig0_valid <= state == FFN_WAIT_UP;
            if (state == FFN_WAIT_UP) begin
                sig0_slot <= block_position[4:0];
                sig0_fixed <= gate_block[block_position[4:0]][31] ?
                    -$signed(fp32_to_q16_magnitude(
                        gate_block[block_position[4:0]])) :
                    $signed(fp32_to_q16_magnitude(
                        gate_block[block_position[4:0]]));
            end

            sig1_valid <= sig0_valid;
            if (sig0_valid) begin
                sig1_slot <= sig0_slot;
                if (sig0_fixed <= -32'sd524288) begin
                    sig1_lower <= 17'd0;
                    sig1_difference <= 17'sd0;
                    sig1_fraction <= 14'd0;
                end else if (sig0_fixed >= 32'sd524288) begin
                    sig1_lower <= 17'd65536;
                    sig1_difference <= 17'sd0;
                    sig1_fraction <= 14'd0;
                end else begin
                    sig1_lower <= {1'b0, sigmoid_table_q16(
                        (sig0_fixed + 32'sd524288) >> 14)};
                    sig1_difference <=
                        $signed({1'b0, sigmoid_table_q16(
                            ((sig0_fixed + 32'sd524288) >> 14) + 1'b1)}) -
                        $signed({1'b0, sigmoid_table_q16(
                            (sig0_fixed + 32'sd524288) >> 14)});
                    sig1_fraction <= sig0_fixed + 32'sd524288;
                end
            end

            sig2_valid <= sig1_valid;
            if (sig1_valid) begin
                sig2_slot <= sig1_slot;
                sig2_lower <= sig1_lower;
                sig2_interpolation <= sig1_difference *
                    $signed({1'b0, sig1_fraction});
            end

            sig3_valid <= sig2_valid;
            if (sig2_valid) begin
                sig3_slot <= sig2_slot;
                sig3_q16 <= sig2_lower + (sig2_interpolation >>> 14);
            end

            if (sig3_valid)
                sigmoid_block[sig3_slot] <=
                    positive_q16_to_fp32(sig3_q16);
        end
    end

    // This controller writes the block memories.  A synchronous reset keeps
    // those memories inferable and also avoids manufacturing asynchronous
    // recovery paths from the 200 MHz configuration domain into their reset
    // pins.
    always_ff @(posedge fp_clk_i) begin
        if (!fp_rst_ni) begin
            state <= FFN_IDLE;
            element_index <= 0;
            block_position <= 0;
            fp_step <= 0;
            fp_sent <= 0;
            fp_received <= 0;
            fp_waiting <= 0;
            fp_local_wait <= 0;
            packed_block <= 0;
            max_absolute <= 0;
            norm_high_pending <= 1'b0;
            norm_high_word <= 0;
        end else begin
            if (core_start) begin
                state <= FFN_LOAD_UP;
                element_index <= 0;
                block_position <= 0;
                fp_step <= 0;
                fp_sent <= 0;
                fp_received <= 0;
                fp_waiting <= 0;
                fp_local_wait <= 0;
                packed_block <= 0;
                max_absolute <= 0;
                norm_high_pending <= 1'b0;
            end else begin
                if (fp_input_valid && fp_input_ready && state == FFN_SCALE)
                    fp_waiting <= 1'b1;
                if (fp_result_valid && state == FFN_SCALE)
                    fp_waiting <= 1'b0;
                if (fp_local_wait) begin
                    exponent_float <= small_int_to_fp32(exponent_integer);
                    exponential <= power_of_two_from_n(
                        exponent_integer, exp_underflow);
                    fp_local_wait <= 1'b0;
                end

                case (state)
                    FFN_IDLE: state <= FFN_IDLE;
                    FFN_LOAD_UP: if (source_word_available) begin
                        if (paired_source) begin
                            if (norm_high_pending)
                                norm_high_pending <= 1'b0;
                            else begin
                                norm_high_word <= core_source_data[63:32];
                                norm_high_pending <= 1'b1;
                            end
                        end
                        if (element_index == element_count_i - 1'b1) begin
                            element_index <= 0;
                            state <= FFN_ACCEPT_GATE;
                        end else begin
                            element_index <= element_index + 1'b1;
                        end
                    end
                    FFN_ACCEPT_GATE: begin
                        if (vector_mode_i) begin
                            gate_block[block_position[4:0]] <= 32'h3f80_0000;
                            state <= FFN_WAIT_UP;
                        end else if (source_word_available) begin
                            if (paired_source) begin
                                if (norm_high_pending)
                                    norm_high_pending <= 1'b0;
                                else begin
                                    norm_high_word <= core_source_data[63:32];
                                    norm_high_pending <= 1'b1;
                                end
                            end
                            gate_block[block_position[4:0]] <= source_word;
                            state <= FFN_WAIT_UP;
                        end
                    end
                    FFN_WAIT_UP: begin
                        up_block[block_position[4:0]] <= up_read_data;
                        if (block_position == 31) begin
                            state <= FFN_WAIT_SIGMOID;
                        end else begin
                            block_position <= block_position + 1'b1;
                            element_index <= element_index + 1'b1;
                            state <= FFN_ACCEPT_GATE;
                        end
                    end
                    FFN_WAIT_SIGMOID: begin
                        if (sig3_valid && sig3_slot == 5'd31) begin
                            fp_sent <= 0;
                            fp_received <= 0;
                            state <= FFN_ACTIVATE_FIRST;
                        end
                    end
                    FFN_ACTIVATE_FIRST: begin
                        if (fp_input_valid && fp_input_ready)
                            fp_sent <= fp_sent + 1'b1;
                        if (fp_result_valid) begin
                            intermediate[fp_received] <= fp_result;
                            if (fp_received == 6'd31) begin
                                fp_sent <= 0;
                                fp_received <= 0;
                                state <= FFN_ACTIVATE_SECOND;
                            end else begin
                                fp_received <= fp_received + 1'b1;
                            end
                        end
                    end
                    FFN_ACTIVATE_SECOND: begin
                        if (fp_input_valid && fp_input_ready)
                            fp_sent <= fp_sent + 1'b1;
                        if (fp_result_valid) begin
                            activation[fp_received] <= fp_result;
                            if (absolute_fp32(fp_result) > max_absolute)
                                max_absolute <= absolute_fp32(fp_result);
                            if (fp_received == 6'd31) begin
                                fp_sent <= 0;
                                fp_received <= 0;
                                fp_step <= 0;
                                state <= FFN_SCALE;
                            end else begin
                                fp_received <= fp_received + 1'b1;
                            end
                        end
                    end
                    FFN_SCALE: if (fp_result_valid) begin
                        case (fp_step)
                            0: begin
                                scale_value <= fp_result;
                                quant_reciprocal <= 32'h7ef3_11c2 - fp_result;
                            end
                            1, 3: polynomial <= fp_result;
                            default: quant_reciprocal <= fp_result;
                        endcase
                        if (fp_step == 4) begin
                            packed_block <= 0;
                            packed_block[15:0] <= fp32_to_fp16_truncate(scale_value);
                            fp_sent <= 0;
                            fp_received <= 0;
                            state <= FFN_QUANTIZE_FIRST;
                        end else begin
                            fp_step <= fp_step + 1'b1;
                        end
                    end
                    FFN_QUANTIZE_FIRST: begin
                        if (fp_input_valid && fp_input_ready)
                            fp_sent <= fp_sent + 1'b1;
                        if (fp_result_valid) begin
                            scaled_block[fp_received] <= fp_result;
                            if (fp_received == 6'd31) begin
                                fp_sent <= 0;
                                fp_received <= 0;
                                state <= FFN_QUANTIZE_SECOND;
                            end else begin
                                fp_received <= fp_received + 1'b1;
                            end
                        end
                    end
                    FFN_QUANTIZE_SECOND: begin
                        if (fp_input_valid && fp_input_ready)
                            fp_sent <= fp_sent + 1'b1;
                        if (fp_result_valid) begin
                            packed_block[16 + fp_received * 8 +: 8] <=
                                rounded_fp32_to_i8(fp_result);
                            if (fp_received == 6'd31) begin
                                state <= FFN_EMIT;
                            end else begin
                                fp_received <= fp_received + 1'b1;
                            end
                        end
                    end
                    FFN_EMIT: if (core_block_ready) begin
                        max_absolute <= 0;
                        block_position <= 0;
                        if (element_index == element_count_i - 1'b1)
                            state <= FFN_DONE;
                        else begin
                            element_index <= element_index + 1'b1;
                            state <= FFN_ACCEPT_GATE;
                        end
                    end
                    // Keep completion asserted until the final packed block
                    // has crossed into the DDR domain and been consumed by
                    // the group writer. Otherwise the outer FSM can observe
                    // DONE while the last four-block group is still empty.
                    FFN_DONE: if (!block_pending) state <= FFN_IDLE;
                    default: state <= FFN_IDLE;
                endcase
            end
        end
    end
endmodule

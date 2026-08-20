module bonsai_attention #(
    parameter integer ID_WIDTH = 5,
    parameter logic [29:0] KV_BASE = 30'h3e24_0000
) (
    input  logic                clk_i,
    input  logic                rst_ni,
    input  logic                start_i,
    input  logic [29:0]         raw_base_i,
    input  logic [29:0]         output_base_i,
    input  logic [29:0]         table_base_i,
    input  logic [4:0]          layer_i,
    input  logic [7:0]          position_i,
    output logic                busy_o,
    output logic                done_o,
    output logic                error_o,
    output logic [ID_WIDTH-1:0] arid_o,
    output logic [29:0]         araddr_o,
    output logic [7:0]          arlen_o,
    output logic [2:0]          arsize_o,
    output logic [1:0]          arburst_o,
    output logic                arvalid_o,
    input  logic                arready_i,
    input  logic [ID_WIDTH-1:0] rid_i,
    input  logic [63:0]         rdata_i,
    input  logic [1:0]          rresp_i,
    input  logic                rlast_i,
    input  logic                rvalid_i,
    output logic                rready_o,
    output logic [ID_WIDTH-1:0] awid_o,
    output logic [29:0]         awaddr_o,
    output logic [7:0]          awlen_o,
    output logic                awvalid_o,
    input  logic                awready_i,
    output logic [63:0]         wdata_o,
    output logic [7:0]          wstrb_o,
    output logic                wlast_o,
    output logic                wvalid_o,
    input  logic                wready_i,
    input  logic [ID_WIDTH-1:0] bid_i,
    input  logic [1:0]          bresp_i,
    input  logic                bvalid_i,
    output logic                bready_o,
    output logic                packer_start_o,
    output logic [63:0]         packer_source_data_o,
    output logic                packer_source_valid_o,
    input  logic                packer_source_ready_i,
    input  logic [271:0]        packer_block_data_i,
    input  logic                packer_block_valid_i,
    output logic                packer_block_ready_o
);

    typedef enum logic [6:0] {
        S_IDLE, S_TABLE_REQ, S_TABLE_WAIT, S_TABLE_STORE,
        S_K_LOAD_REQ, S_K_LOAD_WAIT, S_VALUES_CONVERT_WAIT, S_VALUES_STORE,
        S_SUM, S_SUM_CLAMP, S_SUM_START, S_SUM_WAIT, S_SQRT,
        S_SQRT_DECIDE, S_SCALE_DIV,
        S_TRANSFORM, S_TRANSFORM_CLAMP, S_TRANSFORM_SCALE_START,
        S_TRANSFORM_SCALE_WAIT, S_TRANSFORM_WEIGHT_WAIT,
        S_TRANSFORM_ROPE_WAIT, S_TRANSFORM_CAPTURE,
        S_TRANSFORM_CONVERT_WAIT, S_TRANSFORM_PACK, S_TRANSFORM_ADVANCE,
        S_K_WRITE_REQ, S_K_WRITE_WAIT,
        S_V_LOAD_REQ, S_V_LOAD_WAIT, S_V_WRITE_REQ, S_V_WRITE_WAIT,
        S_NEXT_KV, S_Q_LOAD_REQ, S_Q_LOAD_WAIT, S_Q_SUM,
        S_Q_SQRT, S_Q_SCALE_DIV,
        S_SCORE_REQ, S_SCORE_WAIT, S_HALF_CAPTURE, S_HALF_CONVERT, S_HALF_START,
        S_SCORE_MUL_WAIT, S_SCORE_FINISH,
        S_SCORE_SCALE_FINISH, S_SCORE_STORE,
        S_EXP_START, S_EXP_RUN, S_WEIGHT_INIT, S_WEIGHT_REQ,
        S_WEIGHT_WAIT, S_WEIGHT_MUL_WAIT, S_WEIGHT_ACCUM, S_WEIGHT_WRITE,
        S_OUTPUT_READ, S_OUTPUT_START, S_OUTPUT_ROUND, S_OUTPUT_DIV,
        S_OUTPUT_CAPTURE, S_OUTPUT_CONVERT_WAIT, S_OUTPUT_PACK,
        S_PACK_WAIT_BLOCK, S_PACK_WRITE_REQ, S_PACK_WRITE_WAIT,
        S_NEXT_HEAD, S_DONE
    } state_t;

    state_t state;
    logic [29:0] raw_base, output_base, table_base;
    logic [4:0] layer;
    logic [7:0] position;
    // Bank the scratch memories around their access patterns.  The former
    // monolithic arrays synthesized as registers followed by 128/256-way
    // muxes, which dominated both area and timing.
    (* ram_style = "distributed" *) logic signed [31:0] table_q0_even [0:31];
    (* ram_style = "distributed" *) logic signed [31:0] table_q0_odd [0:31];
    (* ram_style = "distributed" *) logic signed [31:0] table_q1_even [0:31];
    (* ram_style = "distributed" *) logic signed [31:0] table_q1_odd [0:31];
    (* ram_style = "distributed" *) logic signed [31:0] table_k0_even [0:31];
    (* ram_style = "distributed" *) logic signed [31:0] table_k0_odd [0:31];
    (* ram_style = "distributed" *) logic signed [31:0] table_k1_even [0:31];
    (* ram_style = "distributed" *) logic signed [31:0] table_k1_odd [0:31];
    (* ram_style = "distributed" *) logic signed [31:0] table_cos_even [0:31];
    (* ram_style = "distributed" *) logic signed [31:0] table_cos_odd [0:31];
    (* ram_style = "distributed" *) logic signed [31:0] table_sin_even [0:31];
    (* ram_style = "distributed" *) logic signed [31:0] table_sin_odd [0:31];

    (* ram_style = "distributed" *) logic signed [31:0] values_lo_even [0:31];
    (* ram_style = "distributed" *) logic signed [31:0] values_lo_odd [0:31];
    (* ram_style = "distributed" *) logic signed [31:0] values_hi_even [0:31];
    (* ram_style = "distributed" *) logic signed [31:0] values_hi_odd [0:31];
    (* ram_style = "distributed" *) logic [63:0] half_words_lo [0:15];
    (* ram_style = "distributed" *) logic [63:0] half_words_hi [0:15];
    (* ram_style = "distributed" *) logic [127:0] query_words_lo [0:15];
    (* ram_style = "distributed" *) logic [127:0] query_words_hi [0:15];
    (* ram_style = "distributed" *) logic signed [31:0] scores0 [0:63];
    (* ram_style = "distributed" *) logic signed [31:0] scores1 [0:63];
    (* ram_style = "distributed" *) logic signed [31:0] scores2 [0:63];
    (* ram_style = "distributed" *) logic signed [31:0] scores3 [0:63];
    (* ram_style = "block" *) logic signed [63:0] sums0 [0:31];
    (* ram_style = "block" *) logic signed [63:0] sums1 [0:31];
    (* ram_style = "block" *) logic signed [63:0] sums2 [0:31];
    (* ram_style = "block" *) logic signed [63:0] sums3 [0:31];

    logic [63:0] table_q0_word, table_q1_word;
    logic [63:0] table_k0_word, table_k1_word;
    logic [63:0] table_cos_word, table_sin_word;

    logic [47:0] half_pack_low, half_pack_high;
    logic [31:0] value_pack;
    logic [95:0] query_pack_low, query_pack_high;
    logic signed [63:0] transform_first, transform_second;
    logic [31:0] transform_low_bits, transform_high_bits;
    logic signed [31:0] half_q0, half_q1, half_q2, half_q3;
    logic [15:0] half_raw0, half_raw1, half_raw2, half_raw3;
    logic half_for_weight;
    logic signed [31:0] loaded_value0, loaded_value1;
    logic values_for_query;
    logic signed [31:0] value_at_index, value_at_upper;
    logic signed [31:0] transform_weight0, transform_weight1;
    logic signed [31:0] transform_cos, transform_sin;
    logic [63:0] half_word_at_index;
    logic [127:0] query_word_at_index;
    logic signed [31:0] score_at_token;
    logic signed [63:0] sum_at_index;

    logic [8:0] index;
    logic [7:0] token_index;
    logic [4:0] head;
    logic [2:0] kv_head;
    logic [63:0] square_sum;
    logic signed [63:0] dot_sum;
    logic signed [31:0] max_score;
    logic signed [31:0] score_value_reg;
    logic signed [31:0] score_query0, score_query1;
    logic signed [31:0] score_query2, score_query3;
    logic signed [31:0] score_weight_reg;
    logic signed [31:0] sum_raw_reg;
    logic signed [31:0] sum_value_reg;
    logic [63:0] sqrt_output_reg;
    logic sqrt_zero_reg;
    logic signed [63:0] sum_accum0, sum_accum1;
    logic signed [63:0] sum_accum2, sum_accum3;
    logic signed [63:0] sum_next0, sum_next1;
    logic signed [63:0] sum_next2, sum_next3;
    logic [4:0] sum_read_address, sum_write_address;
    logic sum_write_enable;
    logic signed [63:0] sum_write_data0, sum_write_data1;
    logic signed [63:0] sum_write_data2, sum_write_data3;
    logic signed [63:0] mul_result0, mul_result1;
    logic signed [63:0] mul_result2, mul_result3;
    logic [2:0] mul_stage;
    logic [31:0] mul_p00 [0:3];
    logic signed [32:0] mul_p10 [0:3];
    logic signed [32:0] mul_p01 [0:3];
    logic signed [31:0] mul_p11 [0:3];
    logic signed [33:0] mul_cross [0:3];
    logic signed [63:0] mul_pair_low [0:3];
    logic signed [63:0] mul_pair_high [0:3];
    logic signed [31:0] transform_raw0, transform_raw1;
    logic signed [31:0] transform_value0, transform_value1;
    logic [63:0] denominator;
    logic [63:0] output_magnitude;
    logic output_negative;
    logic [3:0] output_negative4;
    logic signed [63:0] output_value2, output_value3;
    logic output_subpair;
    logic signed [31:0] scale_q16;
    logic [31:0] output_low;
    logic signed [63:0] output_value;
    logic [63:0] read_data;
    logic transform_query;
    logic sum_query;

    logic [1087:0] pack_group;
    logic [1:0] pack_group_blocks;
    logic [4:0] pack_group_word;
    logic [4:0] pack_group_index;

    always_comb begin
        if (!index[6]) begin
            value_at_index = index[0] ? values_lo_odd[index[5:1]] :
                                        values_lo_even[index[5:1]];
        end else begin
            value_at_index = index[0] ? values_hi_odd[index[5:1]] :
                                        values_hi_even[index[5:1]];
        end
        value_at_upper = index[0] ? values_hi_odd[index[5:1]] :
                                    values_hi_even[index[5:1]];

        if (transform_query) begin
            transform_weight0 = index[0] ? table_q0_odd[index[5:1]] :
                                           table_q0_even[index[5:1]];
            transform_weight1 = index[0] ? table_q1_odd[index[5:1]] :
                                           table_q1_even[index[5:1]];
        end else begin
            transform_weight0 = index[0] ? table_k0_odd[index[5:1]] :
                                           table_k0_even[index[5:1]];
            transform_weight1 = index[0] ? table_k1_odd[index[5:1]] :
                                           table_k1_even[index[5:1]];
        end
        transform_cos = index[0] ? table_cos_odd[index[5:1]] :
                                   table_cos_even[index[5:1]];
        transform_sin = index[0] ? table_sin_odd[index[5:1]] :
                                   table_sin_even[index[5:1]];

        half_word_at_index = index[4] ? half_words_hi[index[3:0]] :
                                        half_words_lo[index[3:0]];
        query_word_at_index = index[4] ? query_words_hi[index[3:0]] :
                                         query_words_lo[index[3:0]];
        case (token_index[7:6])
            2'd0: score_at_token = scores0[token_index[5:0]];
            2'd1: score_at_token = scores1[token_index[5:0]];
            2'd2: score_at_token = scores2[token_index[5:0]];
            default: score_at_token = scores3[token_index[5:0]];
        endcase
        case (index[1:0])
            2'd0: sum_at_index = sum_accum0;
            2'd1: sum_at_index = sum_accum1;
            2'd2: sum_at_index = sum_accum2;
            default: sum_at_index = sum_accum3;
        endcase
    end

    always_comb begin
        case (state)
            S_OUTPUT_READ, S_OUTPUT_START, S_OUTPUT_ROUND, S_OUTPUT_DIV,
            S_OUTPUT_CAPTURE, S_OUTPUT_CONVERT_WAIT, S_OUTPUT_PACK:
                sum_read_address = index[6:2];
            default: sum_read_address = index[4:0];
        endcase
        sum_write_enable = state == S_WEIGHT_INIT ||
                           state == S_WEIGHT_WRITE;
        sum_write_address = index[4:0];
        if (state == S_WEIGHT_INIT) begin
            sum_write_data0 = 0;
            sum_write_data1 = 0;
            sum_write_data2 = 0;
            sum_write_data3 = 0;
        end else begin
            sum_write_data0 = sum_next0;
            sum_write_data1 = sum_next1;
            sum_write_data2 = sum_next2;
            sum_write_data3 = sum_next3;
        end
    end

    // Four independent synchronous banks provide the four value lanes needed
    // by the weighted sum.  Keeping each read and write in this canonical
    // form makes Vivado infer block RAM instead of thousands of scattered
    // flip-flops and their long selection routes.
    always_ff @(posedge clk_i) begin
        sum_accum0 <= sums0[sum_read_address];
        sum_accum1 <= sums1[sum_read_address];
        sum_accum2 <= sums2[sum_read_address];
        sum_accum3 <= sums3[sum_read_address];
        if (sum_write_enable) begin
            sums0[sum_write_address] <= sum_write_data0;
            sums1[sum_write_address] <= sum_write_data1;
            sums2[sum_write_address] <= sum_write_data2;
            sums3[sum_write_address] <= sum_write_data3;
        end
    end

    logic mul_kick, mul_active, mul_done;
    logic [5:0] mul_bit;
    logic signed [31:0] mul_input_a0, mul_input_a1;
    logic signed [31:0] mul_input_a2, mul_input_a3;
    logic signed [31:0] mul_input_b0, mul_input_b1;
    logic signed [31:0] mul_input_b2, mul_input_b3;
    logic [63:0] mul_multiplicand0, mul_multiplicand1;
    logic [63:0] mul_multiplicand2, mul_multiplicand3;
    logic [31:0] mul_multiplier0, mul_multiplier1;
    logic [31:0] mul_multiplier2, mul_multiplier3;
    logic [63:0] mul_accumulator0, mul_accumulator1;
    logic [63:0] mul_accumulator2, mul_accumulator3;
    logic mul_negative0, mul_negative1, mul_negative2, mul_negative3;
    logic exp_mul_kick;
    logic signed [31:0] exp_mul_input_a, exp_mul_input_b;
    logic signed [31:0] mul_selected_a0, mul_selected_b0;
    logic signed [31:0] mul_selected_a1, mul_selected_b1;
    logic signed [31:0] mul_selected_a2, mul_selected_b2;
    logic signed [31:0] mul_selected_a3, mul_selected_b3;
    logic signed [31:0] mul_operand_a [0:3];
    logic signed [31:0] mul_operand_b [0:3];
    logic signed [63:0] score_scale_high, score_scale_low;

    logic read_active, read_address_sent;
    logic read_kick, memory_error;
    logic read_consumer_ready;
    logic [29:0] read_address;
    logic [7:0] read_length, read_beat;
    logic read_beat_accept;
    logic read_done;
    logic write_active, write_aw_sent, write_w_sent;
    logic write_kick;
    logic [29:0] write_address;
    logic [63:0] write_data;
    logic [7:0] write_length, write_beat;
    logic [1:0] write_stream;
    logic write_beat_accept;
    logic write_done;

    logic sqrt_active;
    logic sqrt_kick;
    logic [63:0] sqrt_input;
    logic [63:0] sqrt_value, sqrt_result, sqrt_bit, sqrt_trial;
    logic sqrt_evaluate;
    logic [63:0] sqrt_output;
    logic sqrt_done;

    logic div_active;
    logic div_kick;
    logic [63:0] div_input_numerator, div_input_denominator;
    logic [63:0] div_numerator, div_denominator, div_quotient;
    logic [64:0] div_remainder;
    logic [6:0] div_bit;
    logic [63:0] div_output;
    logic div_done;

    logic output_div_active, output_div_kick, output_div_done;
    logic [63:0] output_div_input0, output_div_input1;
    logic [63:0] output_div_input2, output_div_input3;
    logic [63:0] output_div_numerator0, output_div_numerator1;
    logic [63:0] output_div_numerator2, output_div_numerator3;
    logic [63:0] output_div_quotient0, output_div_quotient1;
    logic [63:0] output_div_quotient2, output_div_quotient3;
    logic [64:0] output_div_remainder0, output_div_remainder1;
    logic [64:0] output_div_remainder2, output_div_remainder3;
    logic [6:0] output_div_bit;

    logic [3:0] exp_step;
    logic exp_kick;
    logic exp_div_kick;
    logic [63:0] exp_div_numerator, exp_div_denominator;
    logic exp_div_negative;
    logic signed [31:0] exp_input;
    logic signed [31:0] exp_n;
    logic signed [63:0] exp_r, exp_power, exp_poly;
    logic signed [63:0] exp_shift_value;
    logic [6:0] exp_shift_count;
    logic exp_shift_left;
    logic [31:0] exp_output;
    logic exp_done;

    logic qfp_kick, qfp_active, qfp_done;
    logic signed [63:0] qfp_input0, qfp_input1;
    logic [31:0] qfp_magnitude0, qfp_magnitude1;
    logic [7:0] qfp_exponent0, qfp_exponent1;
    logic qfp_sign0, qfp_sign1, qfp_lane_done0, qfp_lane_done1;
    logic [31:0] qfp_output0, qfp_output1;

    logic fpq_kick, fpq_active, fpq_done;
    logic [31:0] fpq_input0, fpq_input1;
    logic [63:0] fpq_value0, fpq_value1;
    logic [5:0] fpq_shift0, fpq_shift1;
    logic fpq_left0, fpq_left1, fpq_sign0, fpq_sign1;
    logic fpq_lane_done0, fpq_lane_done1;
    logic signed [31:0] fpq_output0, fpq_output1;

    function automatic [15:0] fp32_to_half_trunc(input logic [31:0] bits);
        integer exponent;
        begin
            exponent = bits[30:23] - 127 + 15;
            if (exponent <= 0)
                fp32_to_half_trunc = {bits[31], 15'b0};
            else if (exponent >= 31)
                fp32_to_half_trunc = {bits[31], 5'h1f, 10'b0};
            else
                fp32_to_half_trunc = {bits[31], exponent[4:0], bits[22:13]};
        end
    endfunction

    function automatic signed [31:0] half_to_q16(input logic [15:0] half);
        logic [10:0] significand;
        logic [63:0] fixed;
        integer shift;
        begin
            significand = {1'b1, half[9:0]};
            if (half[14:10] == 0)
                fixed = 0;
            else if (half[14:10] >= 30)
                fixed = 64'd2147418112;
            else if (half[14:10] >= 9) begin
                shift = half[14:10] - 9;
                fixed = significand << shift;
            end else begin
                shift = 9 - half[14:10];
                fixed = significand >> shift;
            end
            half_to_q16 = half[15] ? -$signed(fixed) : $signed(fixed);
        end
    endfunction

    function automatic signed [63:0] round_q16_product(
        input signed [63:0] product);
        begin
            round_q16_product = product >= 0 ?
                (product + 64'sd32768) >>> 16 :
                -((-product + 64'sd32768) >>> 16);
        end
    endfunction

    function automatic [29:0] kv_address(
        input [4:0] layer_number, input [7:0] token,
        input logic value_select, input logic [2:0] kv_select,
        input [4:0] item_group);
        logic [22:0] slot;
        begin
            slot = layer_number * 256 + token;
            kv_address = KV_BASE + (slot << 12) +
                         (value_select ? 30'd2048 : 30'd0) +
                         kv_select * 30'd256 +
                         ({25'b0, item_group} << 3);
        end
    endfunction

    assign busy_o = state != S_IDLE && state != S_DONE;
    assign done_o = state == S_DONE;
    assign arid_o = {ID_WIDTH{1'b1}};
    assign araddr_o = read_address;
    assign arlen_o = read_length;
    assign arsize_o = 3'd3;
    assign arburst_o = 2'b01;
    assign arvalid_o = read_active && !read_address_sent;
    always_comb begin
        case (state)
            S_TABLE_WAIT, S_K_LOAD_WAIT, S_V_LOAD_WAIT, S_Q_LOAD_WAIT,
            S_SCORE_WAIT, S_WEIGHT_WAIT: read_consumer_ready = 1'b1;
            default: read_consumer_ready = 1'b0;
        endcase
    end
    // K/Q conversion consumes one beat over several cycles.  Backpressure the
    // burst between beats instead of paying a fresh AXI address round trip for
    // every pair of values.
    assign rready_o = read_active && read_address_sent && read_consumer_ready;
    assign read_beat_accept = read_active && rvalid_i && rready_o;
    assign awid_o = {ID_WIDTH{1'b1}};
    assign awaddr_o = write_address;
    assign awlen_o = write_length;
    assign awvalid_o = write_active && !write_aw_sent;
    assign wdata_o = write_stream == 2'd1 ? half_word_at_index :
                     write_stream == 2'd2 ? pack_group[63:0] : write_data;
    assign wstrb_o = 8'hff;
    assign wlast_o = write_beat == write_length;
    assign wvalid_o = write_active && write_aw_sent && !write_w_sent;
    assign write_beat_accept = wvalid_o && wready_i;
    assign bready_o = write_active && write_aw_sent && write_w_sent;
    assign mul_selected_a0 = exp_mul_kick ? exp_mul_input_a : mul_input_a0;
    assign mul_selected_b0 = exp_mul_kick ? exp_mul_input_b : mul_input_b0;
    assign mul_selected_a1 = exp_mul_kick ? 0 : mul_input_a1;
    assign mul_selected_b1 = exp_mul_kick ? 0 : mul_input_b1;
    assign mul_selected_a2 = exp_mul_kick ? 0 : mul_input_a2;
    assign mul_selected_b2 = exp_mul_kick ? 0 : mul_input_b2;
    assign mul_selected_a3 = exp_mul_kick ? 0 : mul_input_a3;
    assign mul_selected_b3 = exp_mul_kick ? 0 : mul_input_b3;

    // Share the accelerator's existing transform packer. Commands are
    // mutually exclusive, so duplicating its FP datapath only hurts placement.
    assign packer_block_ready_o = state == S_PACK_WAIT_BLOCK;

    // Four pipelined 32x32 products replace the former 32-cycle shift/add
    // unit. Splitting each operand into 16-bit halves maps cleanly onto DSPs;
    // registers between partial products and the two 64-bit additions retain
    // 200 MHz timing without changing the request/completion handshake.
    always_ff @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
            mul_active <= 0;
            mul_done <= 0;
            mul_stage <= 0;
        end else begin
            mul_done <= 0;
            if (mul_kick || exp_mul_kick) begin
                mul_operand_a[0] <= mul_selected_a0;
                mul_operand_a[1] <= mul_selected_a1;
                mul_operand_a[2] <= mul_selected_a2;
                mul_operand_a[3] <= mul_selected_a3;
                mul_operand_b[0] <= mul_selected_b0;
                mul_operand_b[1] <= mul_selected_b1;
                mul_operand_b[2] <= mul_selected_b2;
                mul_operand_b[3] <= mul_selected_b3;
                mul_active <= 1;
                mul_stage <= 0;
            end else if (mul_active) begin
                case (mul_stage)
                    0: begin
                        for (integer lane = 0; lane < 4; lane = lane + 1) begin
                            mul_p00[lane] <= mul_operand_a[lane][15:0] *
                                             mul_operand_b[lane][15:0];
                            mul_p10[lane] <=
                                $signed(mul_operand_a[lane][31:16]) *
                                $signed({1'b0, mul_operand_b[lane][15:0]});
                            mul_p01[lane] <=
                                $signed({1'b0, mul_operand_a[lane][15:0]}) *
                                $signed(mul_operand_b[lane][31:16]);
                            mul_p11[lane] <=
                                $signed(mul_operand_a[lane][31:16]) *
                                $signed(mul_operand_b[lane][31:16]);
                        end
                        mul_stage <= 1;
                    end
                    1: begin
                        mul_cross[0] <= mul_p10[0] + mul_p01[0];
                        mul_cross[1] <= mul_p10[1] + mul_p01[1];
                        mul_cross[2] <= mul_p10[2] + mul_p01[2];
                        mul_cross[3] <= mul_p10[3] + mul_p01[3];
                        mul_stage <= 2;
                    end
                    2: begin
                        mul_pair_low[0] <= $signed({32'b0, mul_p00[0]}) +
                            ($signed({{30{mul_cross[0][33]}},
                                      mul_cross[0]}) <<< 16);
                        mul_pair_low[1] <= $signed({32'b0, mul_p00[1]}) +
                            ($signed({{30{mul_cross[1][33]}},
                                      mul_cross[1]}) <<< 16);
                        mul_pair_low[2] <= $signed({32'b0, mul_p00[2]}) +
                            ($signed({{30{mul_cross[2][33]}},
                                      mul_cross[2]}) <<< 16);
                        mul_pair_low[3] <= $signed({32'b0, mul_p00[3]}) +
                            ($signed({{30{mul_cross[3][33]}},
                                      mul_cross[3]}) <<< 16);
                        mul_pair_high[0] <=
                            $signed({{32{mul_p11[0][31]}}, mul_p11[0]}) <<< 32;
                        mul_pair_high[1] <=
                            $signed({{32{mul_p11[1][31]}}, mul_p11[1]}) <<< 32;
                        mul_pair_high[2] <=
                            $signed({{32{mul_p11[2][31]}}, mul_p11[2]}) <<< 32;
                        mul_pair_high[3] <=
                            $signed({{32{mul_p11[3][31]}}, mul_p11[3]}) <<< 32;
                        mul_stage <= 3;
                    end
                    3: begin
                        mul_result0 <= mul_pair_low[0] + mul_pair_high[0];
                        mul_result1 <= mul_pair_low[1] + mul_pair_high[1];
                        mul_result2 <= mul_pair_low[2] + mul_pair_high[2];
                        mul_result3 <= mul_pair_low[3] + mul_pair_high[3];
                        mul_stage <= 4;
                    end
                    default: begin
                        mul_active <= 0;
                        mul_done <= 1;
                        mul_stage <= 0;
                    end
                endcase
            end
        end
    end

    always_ff @(posedge clk_i or negedge rst_ni) begin
        logic [63:0] magnitude0, magnitude1;
        if (!rst_ni) begin
            qfp_active <= 0;
            qfp_done <= 0;
            qfp_lane_done0 <= 0;
            qfp_lane_done1 <= 0;
        end else begin
            qfp_done <= 0;
            if (qfp_kick) begin
                magnitude0 = qfp_input0 < 0 ? -qfp_input0 : qfp_input0;
                magnitude1 = qfp_input1 < 0 ? -qfp_input1 : qfp_input1;
                // Match signed_q16_to_f32_bits() in the validated decoder.
                // Its compatibility semantics saturate attention values at
                // +/-256 before conversion. Without this clamp, rare large
                // V/output lanes differ by one Q8 step after packing.
                qfp_magnitude0 <= magnitude0 > 64'd16777216 ?
                                  32'd16777216 : magnitude0[31:0];
                qfp_magnitude1 <= magnitude1 > 64'd16777216 ?
                                  32'd16777216 : magnitude1[31:0];
                qfp_exponent0 <= 8'd134;
                qfp_exponent1 <= 8'd134;
                qfp_sign0 <= qfp_input0 < 0;
                qfp_sign1 <= qfp_input1 < 0;
                qfp_lane_done0 <= 0;
                qfp_lane_done1 <= 0;
                qfp_active <= 1;
            end else if (qfp_active) begin
                if (!qfp_lane_done0) begin
                    if (qfp_magnitude0 == 0) begin
                        qfp_output0 <= 0;
                        qfp_lane_done0 <= 1;
                    end else if (|qfp_magnitude0[31:24]) begin
                        qfp_magnitude0 <= qfp_magnitude0 >> 1;
                        qfp_exponent0 <= qfp_exponent0 + 1'b1;
                    end else if (!qfp_magnitude0[23]) begin
                        qfp_magnitude0 <= qfp_magnitude0 << 1;
                        qfp_exponent0 <= qfp_exponent0 - 1'b1;
                    end else begin
                        qfp_output0 <= {qfp_sign0, qfp_exponent0,
                                        qfp_magnitude0[22:0]};
                        qfp_lane_done0 <= 1;
                    end
                end
                if (!qfp_lane_done1) begin
                    if (qfp_magnitude1 == 0) begin
                        qfp_output1 <= 0;
                        qfp_lane_done1 <= 1;
                    end else if (|qfp_magnitude1[31:24]) begin
                        qfp_magnitude1 <= qfp_magnitude1 >> 1;
                        qfp_exponent1 <= qfp_exponent1 + 1'b1;
                    end else if (!qfp_magnitude1[23]) begin
                        qfp_magnitude1 <= qfp_magnitude1 << 1;
                        qfp_exponent1 <= qfp_exponent1 - 1'b1;
                    end else begin
                        qfp_output1 <= {qfp_sign1, qfp_exponent1,
                                        qfp_magnitude1[22:0]};
                        qfp_lane_done1 <= 1;
                    end
                end
                if (qfp_lane_done0 && qfp_lane_done1) begin
                    qfp_active <= 0;
                    qfp_done <= 1;
                end
            end
        end
    end

    always_ff @(posedge clk_i or negedge rst_ni) begin
        logic [63:0] limited0, limited1;
        if (!rst_ni) begin
            fpq_active <= 0;
            fpq_done <= 0;
            fpq_lane_done0 <= 0;
            fpq_lane_done1 <= 0;
        end else begin
            fpq_done <= 0;
            if (fpq_kick) begin
                fpq_sign0 <= fpq_input0[31];
                fpq_sign1 <= fpq_input1[31];
                fpq_left0 <= fpq_input0[30:23] >= 134;
                fpq_left1 <= fpq_input1[30:23] >= 134;
                fpq_lane_done0 <= 0;
                fpq_lane_done1 <= 0;
                fpq_active <= 1;
                if (fpq_input0[30:23] == 0 ||
                    fpq_input0[30:23] <= 102) begin
                    fpq_value0 <= 0;
                    fpq_shift0 <= 0;
                end else if (fpq_input0[30:23] == 8'hff ||
                             fpq_input0[30:23] > 141) begin
                    fpq_value0 <= 64'd2147418112;
                    fpq_shift0 <= 0;
                end else begin
                    fpq_value0 <= {40'b0, 1'b1, fpq_input0[22:0]};
                    fpq_shift0 <= fpq_input0[30:23] >= 134 ?
                                  fpq_input0[30:23] - 134 :
                                  134 - fpq_input0[30:23];
                end
                if (fpq_input1[30:23] == 0 ||
                    fpq_input1[30:23] <= 102) begin
                    fpq_value1 <= 0;
                    fpq_shift1 <= 0;
                end else if (fpq_input1[30:23] == 8'hff ||
                             fpq_input1[30:23] > 141) begin
                    fpq_value1 <= 64'd2147418112;
                    fpq_shift1 <= 0;
                end else begin
                    fpq_value1 <= {40'b0, 1'b1, fpq_input1[22:0]};
                    fpq_shift1 <= fpq_input1[30:23] >= 134 ?
                                  fpq_input1[30:23] - 134 :
                                  134 - fpq_input1[30:23];
                end
            end else if (fpq_active) begin
                if (!fpq_lane_done0) begin
                    if (fpq_shift0 != 0) begin
                        fpq_value0 <= fpq_left0 ? fpq_value0 << 1 :
                                                 fpq_value0 >> 1;
                        fpq_shift0 <= fpq_shift0 - 1'b1;
                    end else begin
                        limited0 = fpq_value0 > 64'd2147418112 ?
                                   64'd2147418112 : fpq_value0;
                        fpq_output0 <= fpq_sign0 ? -$signed(limited0) :
                                                  $signed(limited0);
                        fpq_lane_done0 <= 1;
                    end
                end
                if (!fpq_lane_done1) begin
                    if (fpq_shift1 != 0) begin
                        fpq_value1 <= fpq_left1 ? fpq_value1 << 1 :
                                                 fpq_value1 >> 1;
                        fpq_shift1 <= fpq_shift1 - 1'b1;
                    end else begin
                        limited1 = fpq_value1 > 64'd2147418112 ?
                                   64'd2147418112 : fpq_value1;
                        fpq_output1 <= fpq_sign1 ? -$signed(limited1) :
                                                  $signed(limited1);
                        fpq_lane_done1 <= 1;
                    end
                end
                if (fpq_lane_done0 && fpq_lane_done1) begin
                    fpq_active <= 0;
                    fpq_done <= 1;
                end
            end
        end
    end

    always_ff @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
            read_active <= 0;
            read_address_sent <= 0;
            read_beat <= 0;
            read_done <= 0;
            write_active <= 0;
            write_aw_sent <= 0;
            write_w_sent <= 0;
            write_beat <= 0;
            write_done <= 0;
            memory_error <= 0;
        end else begin
            read_done <= 0;
            write_done <= 0;
            memory_error <= 0;
            if (read_kick) begin
                read_active <= 1;
                read_address_sent <= 0;
                read_beat <= 0;
            end
            if (write_kick) begin
                write_active <= 1;
                write_aw_sent <= 0;
                write_w_sent <= 0;
                write_beat <= 0;
            end
            if (read_active && arvalid_o && arready_i)
                read_address_sent <= 1;
            if (read_active && rvalid_i && rready_o) begin
                read_data <= rdata_i;
                if (rresp_i != 0 || rid_i != {ID_WIDTH{1'b1}} ||
                    rlast_i != (read_beat == read_length))
                    memory_error <= 1;
                if (rlast_i || read_beat == read_length) begin
                    read_active <= 0;
                    read_address_sent <= 0;
                    read_done <= 1;
                end else begin
                    read_beat <= read_beat + 1'b1;
                end
            end
            if (write_active && awvalid_o && awready_i) write_aw_sent <= 1;
            if (write_beat_accept) begin
                if (write_beat == write_length)
                    write_w_sent <= 1;
                else
                    write_beat <= write_beat + 1'b1;
            end
            if (write_active && bvalid_i && bready_o) begin
                write_active <= 0;
                write_aw_sent <= 0;
                write_w_sent <= 0;
                write_done <= 1;
                if (bresp_i != 0 || bid_i != {ID_WIDTH{1'b1}})
                    memory_error <= 1;
            end
        end
    end

    always_ff @(posedge clk_i or negedge rst_ni) begin
        logic [64:0] next_remainder;
        if (!rst_ni) begin
            sqrt_active <= 0;
            sqrt_done <= 0;
            sqrt_evaluate <= 0;
            div_active <= 0;
            div_done <= 0;
        end else begin
            sqrt_done <= 0;
            div_done <= 0;
            if (sqrt_kick) begin
                sqrt_value <= sqrt_input;
                sqrt_result <= 0;
                sqrt_bit <= 64'h4000_0000_0000_0000;
                sqrt_evaluate <= 0;
                sqrt_active <= 1;
            end else if (sqrt_active) begin
                if (!sqrt_evaluate && sqrt_bit == 0) begin
                    sqrt_output <= sqrt_result;
                    sqrt_active <= 0;
                    sqrt_done <= 1;
                end else if (!sqrt_evaluate) begin
                    /* Register the trial sum separately from the comparison
                       and update.  The former single-cycle expression formed
                       two 64-bit carry chains plus selection logic and left
                       essentially no physical timing margin at 200 MHz. */
                    sqrt_trial <= sqrt_result + sqrt_bit;
                    sqrt_evaluate <= 1;
                end else begin
                    if (sqrt_value >= sqrt_trial) begin
                        sqrt_value <= sqrt_value - sqrt_trial;
                        sqrt_result <= (sqrt_result >> 1) + sqrt_bit;
                    end else begin
                        sqrt_result <= sqrt_result >> 1;
                    end
                    sqrt_bit <= sqrt_bit >> 2;
                    sqrt_evaluate <= 0;
                end
            end
            if (div_kick || exp_div_kick) begin
                div_numerator <= exp_div_kick ? exp_div_numerator :
                                                div_input_numerator;
                div_denominator <= exp_div_kick ? exp_div_denominator :
                                                  div_input_denominator;
                div_quotient <= 0;
                div_remainder <= 0;
                div_bit <= 63;
                div_active <= 1;
            end else if (div_active) begin
                next_remainder = {div_remainder[63:0],
                                  div_numerator[63]};
                div_numerator <= div_numerator << 1;
                if (next_remainder >= {1'b0, div_denominator}) begin
                    div_remainder <= next_remainder - div_denominator;
                    div_quotient <= {div_quotient[62:0], 1'b1};
                end else begin
                    div_remainder <= next_remainder;
                    div_quotient <= {div_quotient[62:0], 1'b0};
                end
                if (div_bit == 0) begin
                    div_output <= next_remainder >= {1'b0, div_denominator} ?
                        {div_quotient[62:0], 1'b1} :
                        {div_quotient[62:0], 1'b0};
                    div_active <= 0;
                    div_done <= 1;
                end else begin
                    div_bit <= div_bit - 1'b1;
                end
            end
        end
    end

    // The weighted output is banked four ways. Divide all four lanes in
    // parallel so normalization costs 32 long divisions per head rather than
    // 128; each lane retains the exact rounded integer quotient.
    always_ff @(posedge clk_i or negedge rst_ni) begin
        logic [64:0] next0, next1, next2, next3;
        if (!rst_ni) begin
            output_div_active <= 0;
            output_div_done <= 0;
        end else begin
            output_div_done <= 0;
            if (output_div_kick) begin
                output_div_numerator0 <= output_div_input0;
                output_div_numerator1 <= output_div_input1;
                output_div_numerator2 <= output_div_input2;
                output_div_numerator3 <= output_div_input3;
                output_div_quotient0 <= 0;
                output_div_quotient1 <= 0;
                output_div_quotient2 <= 0;
                output_div_quotient3 <= 0;
                output_div_remainder0 <= 0;
                output_div_remainder1 <= 0;
                output_div_remainder2 <= 0;
                output_div_remainder3 <= 0;
                output_div_bit <= 63;
                output_div_active <= 1;
            end else if (output_div_active) begin
                next0 = {output_div_remainder0[63:0],
                         output_div_numerator0[63]};
                next1 = {output_div_remainder1[63:0],
                         output_div_numerator1[63]};
                next2 = {output_div_remainder2[63:0],
                         output_div_numerator2[63]};
                next3 = {output_div_remainder3[63:0],
                         output_div_numerator3[63]};
                output_div_numerator0 <= output_div_numerator0 << 1;
                output_div_numerator1 <= output_div_numerator1 << 1;
                output_div_numerator2 <= output_div_numerator2 << 1;
                output_div_numerator3 <= output_div_numerator3 << 1;
                if (next0 >= {1'b0, denominator}) begin
                    output_div_remainder0 <= next0 - denominator;
                    output_div_quotient0 <=
                        {output_div_quotient0[62:0], 1'b1};
                end else begin
                    output_div_remainder0 <= next0;
                    output_div_quotient0 <=
                        {output_div_quotient0[62:0], 1'b0};
                end
                if (next1 >= {1'b0, denominator}) begin
                    output_div_remainder1 <= next1 - denominator;
                    output_div_quotient1 <=
                        {output_div_quotient1[62:0], 1'b1};
                end else begin
                    output_div_remainder1 <= next1;
                    output_div_quotient1 <=
                        {output_div_quotient1[62:0], 1'b0};
                end
                if (next2 >= {1'b0, denominator}) begin
                    output_div_remainder2 <= next2 - denominator;
                    output_div_quotient2 <=
                        {output_div_quotient2[62:0], 1'b1};
                end else begin
                    output_div_remainder2 <= next2;
                    output_div_quotient2 <=
                        {output_div_quotient2[62:0], 1'b0};
                end
                if (next3 >= {1'b0, denominator}) begin
                    output_div_remainder3 <= next3 - denominator;
                    output_div_quotient3 <=
                        {output_div_quotient3[62:0], 1'b1};
                end else begin
                    output_div_remainder3 <= next3;
                    output_div_quotient3 <=
                        {output_div_quotient3[62:0], 1'b0};
                end
                if (output_div_bit == 0) begin
                    output_div_active <= 0;
                    output_div_done <= 1;
                end else output_div_bit <= output_div_bit - 1'b1;
            end
        end
    end

    always_ff @(posedge clk_i or negedge rst_ni) begin
        logic signed [63:0] scaled;
        logic signed [63:0] next_power;
        logic signed [63:0] rounded;
        logic signed [31:0] next_n;
        if (!rst_ni) begin
            exp_step <= 0;
            exp_done <= 0;
            exp_div_kick <= 0;
            exp_mul_kick <= 0;
        end else begin
            exp_done <= 0;
            exp_div_kick <= 0;
            exp_mul_kick <= 0;
            if (exp_kick) begin
                exp_step <= 1;
            end else if (exp_step != 0) begin
                case (exp_step)
                    1: begin
                        if (exp_input <= -32'sd1310720) begin
                            exp_output <= 0;
                            exp_done <= 1;
                            exp_step <= 0;
                        end else if (exp_input >= 0) begin
                            exp_output <= 32'd65536;
                            exp_done <= 1;
                            exp_step <= 0;
                        end else begin
                            exp_mul_input_a <= exp_input;
                            exp_mul_input_b <= 32'sd94548;
                            exp_mul_kick <= 1;
                            exp_step <= 2;
                        end
                    end
                    2: if (mul_done) begin
                        next_n = -(($signed(-mul_result0) +
                                   64'sd2147483648) >>> 32);
                        exp_n <= next_n;
                        exp_mul_input_a <= next_n;
                        exp_mul_input_b <= 32'sd744261118;
                        exp_mul_kick <= 1;
                        exp_step <= 3;
                    end
                    3: if (mul_done) begin
                        scaled = $signed(exp_input);
                        scaled = (scaled <<< 14) - mul_result0;
                        exp_r <= scaled;
                        exp_mul_input_a <= scaled[31:0];
                        exp_mul_input_b <= scaled[31:0];
                        exp_mul_kick <= 1;
                        exp_step <= 4;
                    end
                    4: if (mul_done) begin
                        next_power = mul_result0 >>> 30;
                        exp_power <= next_power;
                        exp_poly <= 64'sd1073741824 + exp_r;
                        exp_step <= 5;
                    end
                    5: begin
                        exp_poly <= exp_poly + (exp_power >>> 1);
                        exp_mul_input_a <= exp_power[31:0];
                        exp_mul_input_b <= exp_r[31:0];
                        exp_mul_kick <= 1;
                        exp_step <= 6;
                    end
                    6, 8, 10: if (mul_done) begin
                        next_power = mul_result0 >>> 30;
                        exp_power <= next_power;
                        exp_div_numerator <= next_power < 0 ?
                            -next_power : next_power;
                        exp_div_denominator <= exp_step == 6 ? 6 :
                                               exp_step == 8 ? 24 : 120;
                        exp_div_negative <= next_power < 0;
                        exp_div_kick <= 1;
                        exp_step <= exp_step + 1'b1;
                    end
                    7, 9: if (div_done) begin
                        exp_poly <= exp_div_negative ?
                            exp_poly - $signed(div_output) :
                            exp_poly + $signed(div_output);
                        exp_mul_input_a <= exp_power[31:0];
                        exp_mul_input_b <= exp_r[31:0];
                        exp_mul_kick <= 1;
                        exp_step <= exp_step + 1'b1;
                    end
                    11: if (div_done) begin
                        exp_poly <= exp_div_negative ?
                            exp_poly - $signed(div_output) :
                            exp_poly + $signed(div_output);
                        exp_step <= 12;
                    end
                    12: begin
                        exp_shift_left <= exp_n >= 0;
                        if ((exp_n <= -32'sd64) || (exp_n >= 32'sd64)) begin
                            exp_shift_value <= exp_n < 0 && exp_poly < 0 ?
                                               -64'sd1 : 64'sd0;
                            exp_shift_count <= 0;
                        end else begin
                            exp_shift_value <= exp_poly;
                            exp_shift_count <= exp_n < 0 ? -exp_n : exp_n;
                        end
                        exp_step <= 13;
                    end
                    13: begin
                        if (exp_shift_count != 0) begin
                            exp_shift_value <= exp_shift_left ?
                                exp_shift_value <<< 1 : exp_shift_value >>> 1;
                            exp_shift_count <= exp_shift_count - 1'b1;
                        end else begin
                            exp_step <= 14;
                        end
                    end
                    default: begin
                        rounded = (exp_shift_value + 64'sd8192) >>> 14;
                        exp_output <= rounded <= 0 ? 0 :
                                      rounded > 65536 ? 65536 : rounded[31:0];
                        exp_done <= 1;
                        exp_step <= 0;
                    end
                endcase
            end
        end
    end

    // Keep the main controller reset synchronous.  This process also owns all
    // attention scratch-memory writes; an asynchronous reset in its
    // sensitivity list makes Vivado dissolve every otherwise inferable RAM
    // into flip-flops, even though the memories themselves are not reset.
    always_ff @(posedge clk_i) begin
        logic signed [63:0] a, b, first, second;
        logic signed [63:0] scaled_dot;
        logic signed [31:0] score_value;
        if (!rst_ni) begin
            state <= S_IDLE;
            error_o <= 0;
            read_length <= 0;
            read_kick <= 0;
            write_kick <= 0;
            sqrt_kick <= 0;
            div_kick <= 0;
            output_div_kick <= 0;
            exp_kick <= 0;
            mul_kick <= 0;
            qfp_kick <= 0;
            fpq_kick <= 0;
            packer_start_o <= 0;
            packer_source_valid_o <= 0;
            packer_source_data_o <= 0;
            pack_group <= 0;
            pack_group_blocks <= 0;
            pack_group_word <= 0;
            pack_group_index <= 0;
            write_length <= 0;
            write_stream <= 0;
        end else begin
            read_kick <= 0;
            write_kick <= 0;
            sqrt_kick <= 0;
            div_kick <= 0;
            output_div_kick <= 0;
            exp_kick <= 0;
            mul_kick <= 0;
            qfp_kick <= 0;
            fpq_kick <= 0;
            packer_start_o <= 0;
            packer_source_valid_o <= 0;
            if (memory_error) error_o <= 1;
            case (state)
                S_IDLE: if (start_i) begin
                    raw_base <= raw_base_i;
                    output_base <= output_base_i;
                    table_base <= table_base_i;
                    layer <= layer_i;
                    position <= position_i;
                    error_o <= 0;
                    index <= 0;
                    packer_start_o <= 1;
                    pack_group <= 0;
                    pack_group_blocks <= 0;
                    pack_group_word <= 0;
                    pack_group_index <= 0;
                    state <= S_TABLE_REQ;
                end
                S_TABLE_REQ: if (!read_active) begin
                    read_address <= table_base;
                    read_length <= 191;
                    read_kick <= 1;
                    state <= S_TABLE_WAIT;
                end
                S_TABLE_WAIT: if (read_beat_accept) begin
                    case (index[7:5])
                        3'd0: table_q0_word <= rdata_i;
                        3'd1: table_q1_word <= rdata_i;
                        3'd2: table_k0_word <= rdata_i;
                        3'd3: table_k1_word <= rdata_i;
                        3'd4: table_cos_word <= rdata_i;
                        default: table_sin_word <= rdata_i;
                    endcase
                    state <= S_TABLE_STORE;
                end
                S_TABLE_STORE: begin
                    case (index[7:5])
                        3'd0: begin
                            table_q0_even[index[4:0]] <= table_q0_word[31:0];
                            table_q0_odd[index[4:0]] <= table_q0_word[63:32];
                        end
                        3'd1: begin
                            table_q1_even[index[4:0]] <= table_q1_word[31:0];
                            table_q1_odd[index[4:0]] <= table_q1_word[63:32];
                        end
                        3'd2: begin
                            table_k0_even[index[4:0]] <= table_k0_word[31:0];
                            table_k0_odd[index[4:0]] <= table_k0_word[63:32];
                        end
                        3'd3: begin
                            table_k1_even[index[4:0]] <= table_k1_word[31:0];
                            table_k1_odd[index[4:0]] <= table_k1_word[63:32];
                        end
                        3'd4: begin
                            table_cos_even[index[4:0]] <= table_cos_word[31:0];
                            table_cos_odd[index[4:0]] <= table_cos_word[63:32];
                        end
                        default: begin
                            table_sin_even[index[4:0]] <= table_sin_word[31:0];
                            table_sin_odd[index[4:0]] <= table_sin_word[63:32];
                        end
                    endcase
                    if (index == 191) begin
                        kv_head <= 0;
                        index <= 0;
                        square_sum <= 0;
                        state <= S_K_LOAD_REQ;
                    end else begin
                        index <= index + 1'b1;
                        state <= S_TABLE_WAIT;
                    end
                end
                S_K_LOAD_REQ: if (!read_active) begin
                    read_address <= raw_base + 30'd8192 +
                                    kv_head * 30'd512;
                    read_length <= 63;
                    read_kick <= 1;
                    state <= S_K_LOAD_WAIT;
                end
                S_K_LOAD_WAIT: if (read_beat_accept) begin
                    fpq_input0 <= rdata_i[31:0];
                    fpq_input1 <= rdata_i[63:32];
                    fpq_kick <= 1;
                    values_for_query <= 0;
                    state <= S_VALUES_CONVERT_WAIT;
                end
                S_VALUES_CONVERT_WAIT: if (fpq_done) begin
                    loaded_value0 <= fpq_output0;
                    loaded_value1 <= fpq_output1;
                    state <= S_VALUES_STORE;
                end
                S_VALUES_STORE: begin
                    if (!index[5]) begin
                        values_lo_even[index[4:0]] <=
                            loaded_value0;
                        values_lo_odd[index[4:0]] <=
                            loaded_value1;
                    end else begin
                        values_hi_even[index[4:0]] <=
                            loaded_value0;
                        values_hi_odd[index[4:0]] <=
                            loaded_value1;
                    end
                    if (index == 63) begin
                        index <= 0;
                        square_sum <= 0;
                        if (values_for_query) state <= S_Q_SUM;
                        else state <= S_SUM;
                    end else begin
                        index <= index + 1'b1;
                        if (values_for_query) state <= S_Q_LOAD_WAIT;
                        else state <= S_K_LOAD_WAIT;
                    end
                end
                S_SUM, S_Q_SUM: begin
                    sum_raw_reg <= value_at_index;
                    sum_query <= state == S_Q_SUM;
                    state <= S_SUM_CLAMP;
                end
                S_SUM_CLAMP: begin
                    a = sum_raw_reg;
                    if (a > 64'sd16777216) a = 64'sd16777216;
                    if (a < -64'sd16777216) a = -64'sd16777216;
                    sum_value_reg <= a[31:0];
                    state <= S_SUM_START;
                end
                S_SUM_START: begin
                    mul_input_a0 <= sum_value_reg;
                    mul_input_b0 <= sum_value_reg;
                    mul_input_a1 <= 0;
                    mul_input_b1 <= 0;
                    mul_input_a2 <= 0;
                    mul_input_b2 <= 0;
                    mul_input_a3 <= 0;
                    mul_input_b3 <= 0;
                    mul_kick <= 1;
                    state <= S_SUM_WAIT;
                end
                S_SUM_WAIT: if (mul_done) begin
                    square_sum <= square_sum + mul_result0;
                    if (index == 127) begin
                        sqrt_input <= ((square_sum + mul_result0) >> 7) +
                                      64'd4295;
                        sqrt_kick <= 1;
                        if (sum_query) state <= S_Q_SQRT;
                        else state <= S_SQRT;
                    end else begin
                        index <= index + 1'b1;
                        if (sum_query) state <= S_Q_SUM;
                        else state <= S_SUM;
                    end
                end
                S_SQRT, S_Q_SQRT: if (sqrt_done) begin
                    sqrt_output_reg <= sqrt_output;
                    sqrt_zero_reg <= sqrt_output == 0;
                    transform_query <= state == S_Q_SQRT;
                    state <= S_SQRT_DECIDE;
                end
                S_SQRT_DECIDE: begin
                    if (sqrt_zero_reg) begin
                        scale_q16 <= 32'sd65536000;
                        index <= 0;
                        state <= S_TRANSFORM;
                    end else begin
                        div_input_numerator <= 64'h1_0000_0000;
                        div_input_denominator <= sqrt_output_reg;
                        div_kick <= 1;
                        if (transform_query) state <= S_Q_SCALE_DIV;
                        else state <= S_SCALE_DIV;
                    end
                end
                S_SCALE_DIV, S_Q_SCALE_DIV: if (div_done) begin
                    scale_q16 <= div_output[31:0];
                    index <= 0;
                    transform_query <= state == S_Q_SCALE_DIV;
                    state <= S_TRANSFORM;
                end
                S_TRANSFORM: begin
                    transform_raw0 <= value_at_index;
                    transform_raw1 <= value_at_upper;
                    state <= S_TRANSFORM_CLAMP;
                end
                S_TRANSFORM_CLAMP: begin
                    a = transform_raw0;
                    b = transform_raw1;
                    if (a > 64'sd16777216) a = 64'sd16777216;
                    if (a < -64'sd16777216) a = -64'sd16777216;
                    if (b > 64'sd16777216) b = 64'sd16777216;
                    if (b < -64'sd16777216) b = -64'sd16777216;
                    transform_value0 <= a[31:0];
                    transform_value1 <= b[31:0];
                    state <= S_TRANSFORM_SCALE_START;
                end
                S_TRANSFORM_SCALE_START: begin
                    mul_input_a0 <= transform_value0;
                    mul_input_b0 <= scale_q16;
                    mul_input_a1 <= transform_value1;
                    mul_input_b1 <= scale_q16;
                    mul_input_a2 <= 0;
                    mul_input_b2 <= 0;
                    mul_input_a3 <= 0;
                    mul_input_b3 <= 0;
                    mul_kick <= 1;
                    state <= S_TRANSFORM_SCALE_WAIT;
                end
                S_TRANSFORM_SCALE_WAIT: if (mul_done) begin
                    a = round_q16_product(mul_result0);
                    b = round_q16_product(mul_result1);
                    mul_input_a0 <= a[31:0];
                    mul_input_b0 <= transform_weight0;
                    mul_input_a1 <= b[31:0];
                    mul_input_b1 <= transform_weight1;
                    mul_kick <= 1;
                    state <= S_TRANSFORM_WEIGHT_WAIT;
                end
                S_TRANSFORM_WEIGHT_WAIT: if (mul_done) begin
                    a = round_q16_product(mul_result0);
                    b = round_q16_product(mul_result1);
                    mul_input_a0 <= a[31:0];
                    mul_input_b0 <= transform_cos;
                    mul_input_a1 <= b[31:0];
                    mul_input_b1 <= transform_sin;
                    mul_input_a2 <= a[31:0];
                    mul_input_b2 <= transform_sin;
                    mul_input_a3 <= b[31:0];
                    mul_input_b3 <= transform_cos;
                    mul_kick <= 1;
                    state <= S_TRANSFORM_ROPE_WAIT;
                end
                S_TRANSFORM_ROPE_WAIT: if (mul_done) begin
                    transform_first <= round_q16_product(mul_result0) -
                                       round_q16_product(mul_result1);
                    transform_second <= round_q16_product(mul_result2) +
                                        round_q16_product(mul_result3);
                    state <= S_TRANSFORM_CAPTURE;
                end
                S_TRANSFORM_CAPTURE: begin
                    if (!transform_query) begin
                        // Match the decoder's explicitly bounded K-cache
                        // conversion. Values beyond +/-256 Q16 otherwise
                        // survive hardware half conversion but collapse in
                        // the validated CPU compatibility path.
                        a = transform_first;
                        b = transform_second;
                        if (a > 64'sd16777216) a = 64'sd16777216;
                        if (a < -64'sd16777216) a = -64'sd16777216;
                        if (b > 64'sd16777216) b = 64'sd16777216;
                        if (b < -64'sd16777216) b = -64'sd16777216;
                        qfp_input0 <= a;
                        qfp_input1 <= b;
                        qfp_kick <= 1;
                        state <= S_TRANSFORM_CONVERT_WAIT;
                    end else begin
                        case (index[1:0])
                            2'd0: begin
                                query_pack_low[31:0] <= transform_first[31:0];
                                query_pack_high[31:0] <= transform_second[31:0];
                            end
                            2'd1: begin
                                query_pack_low[63:32] <= transform_first[31:0];
                                query_pack_high[63:32] <= transform_second[31:0];
                            end
                            2'd2: begin
                                query_pack_low[95:64] <= transform_first[31:0];
                                query_pack_high[95:64] <= transform_second[31:0];
                            end
                            default: begin
                                query_words_lo[index[5:2]] <=
                                    {transform_first[31:0], query_pack_low};
                                query_words_hi[index[5:2]] <=
                                    {transform_second[31:0], query_pack_high};
                            end
                        endcase
                        state <= S_TRANSFORM_ADVANCE;
                    end
                end
                S_TRANSFORM_CONVERT_WAIT: if (qfp_done) begin
                    transform_low_bits <= qfp_output0;
                    transform_high_bits <= qfp_output1;
                    state <= S_TRANSFORM_PACK;
                end
                S_TRANSFORM_PACK: begin
                    case (index[1:0])
                        2'd0: begin
                            half_pack_low[15:0] <=
                                fp32_to_half_trunc(transform_low_bits);
                            half_pack_high[15:0] <=
                                fp32_to_half_trunc(transform_high_bits);
                        end
                        2'd1: begin
                            half_pack_low[31:16] <=
                                fp32_to_half_trunc(transform_low_bits);
                            half_pack_high[31:16] <=
                                fp32_to_half_trunc(transform_high_bits);
                        end
                        2'd2: begin
                            half_pack_low[47:32] <=
                                fp32_to_half_trunc(transform_low_bits);
                            half_pack_high[47:32] <=
                                fp32_to_half_trunc(transform_high_bits);
                        end
                        default: begin
                            half_words_lo[index[5:2]] <=
                                {fp32_to_half_trunc(transform_low_bits),
                                 half_pack_low};
                            half_words_hi[index[5:2]] <=
                                {fp32_to_half_trunc(transform_high_bits),
                                 half_pack_high};
                        end
                    endcase
                    state <= S_TRANSFORM_ADVANCE;
                end
                S_TRANSFORM_ADVANCE: begin
                    if (index == 63) begin
                        index <= 0;
                        if (!transform_query) state <= S_K_WRITE_REQ;
                        else state <= S_SCORE_REQ;
                        if (transform_query) begin
                            token_index <= 0;
                            dot_sum <= 0;
                            max_score <= 32'sh8000_0000;
                        end
                    end else begin
                        index <= index + 1'b1;
                        state <= S_TRANSFORM;
                    end
                end
                S_K_WRITE_REQ: if (!write_active) begin
                    write_address <= kv_address(layer, position, 0, kv_head,
                                                0);
                    write_length <= 31;
                    write_stream <= 1;
                    write_kick <= 1;
                    state <= S_K_WRITE_WAIT;
                end
                S_K_WRITE_WAIT: begin
                    if (write_beat_accept && index != 31)
                        index <= index + 1'b1;
                    if (write_done) begin
                        index <= 0;
                        state <= S_V_LOAD_REQ;
                    end
                end
                S_V_LOAD_REQ: if (!read_active) begin
                    read_address <= raw_base + 30'd12288 +
                                    kv_head * 30'd512;
                    read_length <= 63;
                    read_kick <= 1;
                    state <= S_V_LOAD_WAIT;
                end
                S_V_LOAD_WAIT: if (read_beat_accept) begin
                    if (!index[0]) begin
                        value_pack <= {
                            fp32_to_half_trunc(rdata_i[63:32]),
                            fp32_to_half_trunc(rdata_i[31:0])};
                    end else if (!index[5]) begin
                        half_words_lo[index[4:1]] <= {
                            fp32_to_half_trunc(rdata_i[63:32]),
                            fp32_to_half_trunc(rdata_i[31:0]), value_pack};
                    end else begin
                        half_words_hi[index[4:1]] <= {
                            fp32_to_half_trunc(rdata_i[63:32]),
                            fp32_to_half_trunc(rdata_i[31:0]), value_pack};
                    end
                    if (index == 63) begin index <= 0; state <= S_V_WRITE_REQ; end
                    else begin index <= index + 1'b1; state <= S_V_LOAD_WAIT; end
                end
                S_V_WRITE_REQ: if (!write_active) begin
                    write_address <= kv_address(layer, position, 1, kv_head,
                                                0);
                    write_length <= 31;
                    write_stream <= 1;
                    write_kick <= 1;
                    state <= S_V_WRITE_WAIT;
                end
                S_V_WRITE_WAIT: begin
                    if (write_beat_accept && index != 31)
                        index <= index + 1'b1;
                    if (write_done) state <= S_NEXT_KV;
                end
                S_NEXT_KV: begin
                    if (kv_head != 7) begin
                        kv_head <= kv_head + 1'b1;
                        index <= 0;
                        square_sum <= 0;
                        state <= S_K_LOAD_REQ;
                    end else begin
                        head <= 0;
                        index <= 0;
                        square_sum <= 0;
                        state <= S_Q_LOAD_REQ;
                    end
                end
                S_Q_LOAD_REQ: if (!read_active) begin
                    read_address <= raw_base + (head * 30'd512);
                    read_length <= 63;
                    read_kick <= 1;
                    state <= S_Q_LOAD_WAIT;
                end
                S_Q_LOAD_WAIT: if (read_beat_accept) begin
                    fpq_input0 <= rdata_i[31:0];
                    fpq_input1 <= rdata_i[63:32];
                    fpq_kick <= 1;
                    values_for_query <= 1;
                    state <= S_VALUES_CONVERT_WAIT;
                end
                S_SCORE_REQ: if (!read_active) begin
                    read_address <= kv_address(layer, token_index, 0,
                                                head[3:1], 0);
                    read_length <= 31;
                    read_kick <= 1;
                    state <= S_SCORE_WAIT;
                end
                S_SCORE_WAIT: begin
                    if (read_beat_accept) begin
                        if (read_beat[4])
                            half_words_hi[read_beat[3:0]] <= rdata_i;
                        else
                            half_words_lo[read_beat[3:0]] <= rdata_i;
                    end
                    if (read_done) begin
                        index <= 0;
                        half_for_weight <= 0;
                        state <= S_HALF_CAPTURE;
                    end
                end
                S_HALF_CAPTURE: begin
                    half_raw0 <= half_word_at_index[15:0];
                    half_raw1 <= half_word_at_index[31:16];
                    half_raw2 <= half_word_at_index[47:32];
                    half_raw3 <= half_word_at_index[63:48];
                    state <= S_HALF_CONVERT;
                end
                S_HALF_CONVERT: begin
                    half_q0 <= half_to_q16(half_raw0);
                    half_q1 <= half_to_q16(half_raw1);
                    half_q2 <= half_to_q16(half_raw2);
                    half_q3 <= half_to_q16(half_raw3);
                    score_query0 <= query_word_at_index[31:0];
                    score_query1 <= query_word_at_index[63:32];
                    score_query2 <= query_word_at_index[95:64];
                    score_query3 <= query_word_at_index[127:96];
                    score_weight_reg <= score_at_token;
                    state <= S_HALF_START;
                end
                S_HALF_START: begin
                    if (!half_for_weight) begin
                        mul_input_a0 <= score_query0;
                        mul_input_b0 <= half_q0;
                        mul_input_a1 <= score_query1;
                        mul_input_b1 <= half_q1;
                        mul_input_a2 <= score_query2;
                        mul_input_b2 <= half_q2;
                        mul_input_a3 <= score_query3;
                        mul_input_b3 <= half_q3;
                        state <= S_SCORE_MUL_WAIT;
                    end else begin
                        mul_input_a0 <= half_q0;
                        mul_input_b0 <= score_weight_reg;
                        mul_input_a1 <= half_q1;
                        mul_input_b1 <= score_weight_reg;
                        mul_input_a2 <= half_q2;
                        mul_input_b2 <= score_weight_reg;
                        mul_input_a3 <= half_q3;
                        mul_input_b3 <= score_weight_reg;
                        state <= S_WEIGHT_MUL_WAIT;
                    end
                    mul_kick <= 1;
                end
                S_SCORE_MUL_WAIT: if (mul_done) begin
                    dot_sum <= dot_sum + mul_result0 + mul_result1 +
                               mul_result2 + mul_result3;
                    if (index == 31) state <= S_SCORE_FINISH;
                    else begin
                        index <= index + 1'b1;
                        state <= S_HALF_CAPTURE;
                    end
                end
                S_SCORE_FINISH: begin
                    score_scale_high <= (dot_sum <<< 12) +
                                        (dot_sum <<< 10) +
                                        (dot_sum <<< 9);
                    score_scale_low <= (dot_sum <<< 7) +
                                       (dot_sum <<< 5) + dot_sum;
                    state <= S_SCORE_SCALE_FINISH;
                end
                S_SCORE_SCALE_FINISH: begin
                    scaled_dot = score_scale_high + score_scale_low;
                    score_value = scaled_dot >= 0 ?
                        (scaled_dot + (64'sd1 <<< 31)) >>> 32 :
                        -((-scaled_dot + (64'sd1 <<< 31)) >>> 32);
                    score_value_reg <= score_value;
                    state <= S_SCORE_STORE;
                end
                S_SCORE_STORE: begin
                    case (token_index[7:6])
                        2'd0: scores0[token_index[5:0]] <= score_value_reg;
                        2'd1: scores1[token_index[5:0]] <= score_value_reg;
                        2'd2: scores2[token_index[5:0]] <= score_value_reg;
                        default: scores3[token_index[5:0]] <= score_value_reg;
                    endcase
                    if (score_value_reg > max_score) max_score <= score_value_reg;
                    if (token_index == position) begin
                        token_index <= 0;
                        denominator <= 0;
                        state <= S_EXP_START;
                    end else begin
                        token_index <= token_index + 1'b1;
                        index <= 0;
                        dot_sum <= 0;
                        state <= S_SCORE_REQ;
                    end
                end
                S_EXP_START: begin
                    exp_input <= score_at_token - max_score;
                    exp_kick <= 1;
                    state <= S_EXP_RUN;
                end
                S_EXP_RUN: if (exp_done) begin
                    case (token_index[7:6])
                        2'd0: scores0[token_index[5:0]] <= exp_output;
                        2'd1: scores1[token_index[5:0]] <= exp_output;
                        2'd2: scores2[token_index[5:0]] <= exp_output;
                        default: scores3[token_index[5:0]] <= exp_output;
                    endcase
                    denominator <= denominator + exp_output;
                    if (token_index == position) begin
                        index <= 0;
                        state <= S_WEIGHT_INIT;
                    end else begin
                        token_index <= token_index + 1'b1;
                        state <= S_EXP_START;
                    end
                end
                S_WEIGHT_INIT: begin
                    if (index == 31) begin
                        token_index <= 0; index <= 0; state <= S_WEIGHT_REQ;
                    end else index <= index + 1'b1;
                end
                S_WEIGHT_REQ: if (!read_active) begin
                    read_address <= kv_address(layer, token_index, 1,
                                                head[3:1], 0);
                    read_length <= 31;
                    read_kick <= 1;
                    state <= S_WEIGHT_WAIT;
                end
                S_WEIGHT_WAIT: begin
                    if (read_beat_accept) begin
                        if (read_beat[4])
                            half_words_hi[read_beat[3:0]] <= rdata_i;
                        else
                            half_words_lo[read_beat[3:0]] <= rdata_i;
                    end
                    if (read_done) begin
                        index <= 0;
                        half_for_weight <= 1;
                        state <= S_HALF_CAPTURE;
                    end
                end
                S_WEIGHT_MUL_WAIT: if (mul_done) begin
                    state <= S_WEIGHT_ACCUM;
                end
                S_WEIGHT_ACCUM: begin
                    sum_next0 <= sum_accum0 + mul_result0;
                    sum_next1 <= sum_accum1 + mul_result1;
                    sum_next2 <= sum_accum2 + mul_result2;
                    sum_next3 <= sum_accum3 + mul_result3;
                    state <= S_WEIGHT_WRITE;
                end
                S_WEIGHT_WRITE: begin
                    if (index == 31) begin
                        if (token_index == position) begin
                            index <= 0; state <= S_OUTPUT_READ;
                        end else begin
                            token_index <= token_index + 1'b1;
                            index <= 0; state <= S_WEIGHT_REQ;
                        end
                    end else begin
                        index <= index + 1'b1;
                        state <= S_HALF_CAPTURE;
                    end
                end
                S_OUTPUT_READ: state <= S_OUTPUT_START;
                S_OUTPUT_START: begin
                    if (position == 0) begin
                        // With one token, score and denominator are both
                        // exactly 65536.  Match the CPU's direct-V fast path
                        // instead of introducing reciprocal-divider error.
                        qfp_input0 <= sum_accum0 >>> 16;
                        qfp_input1 <= sum_accum1 >>> 16;
                        output_value2 <= sum_accum2 >>> 16;
                        output_value3 <= sum_accum3 >>> 16;
                        output_subpair <= 0;
                        qfp_kick <= 1;
                        state <= S_OUTPUT_CONVERT_WAIT;
                    end else begin
                        output_negative4 <= {sum_accum3 < 0, sum_accum2 < 0,
                                             sum_accum1 < 0, sum_accum0 < 0};
                        // Capture the BRAM outputs before the signed magnitude
                        // and rounding add.  Keeping that arithmetic off the
                        // RAM-to-register path provides deterministic timing
                        // margin at 200 MHz without changing the quotient.
                        output_div_input0 <= sum_accum0;
                        output_div_input1 <= sum_accum1;
                        output_div_input2 <= sum_accum2;
                        output_div_input3 <= sum_accum3;
                        state <= S_OUTPUT_ROUND;
                    end
                end
                S_OUTPUT_ROUND: begin
                        output_div_input0 <= output_negative4[0] ?
                            -output_div_input0 + (denominator >> 1) :
                             output_div_input0 + (denominator >> 1);
                        output_div_input1 <= output_negative4[1] ?
                            -output_div_input1 + (denominator >> 1) :
                             output_div_input1 + (denominator >> 1);
                        output_div_input2 <= output_negative4[2] ?
                            -output_div_input2 + (denominator >> 1) :
                             output_div_input2 + (denominator >> 1);
                        output_div_input3 <= output_negative4[3] ?
                            -output_div_input3 + (denominator >> 1) :
                             output_div_input3 + (denominator >> 1);
                        output_div_kick <= 1;
                        state <= S_OUTPUT_DIV;
                end
                S_OUTPUT_DIV: if (output_div_done) begin
                    qfp_input0 <= output_negative4[0] ?
                        -$signed(output_div_quotient0) :
                         $signed(output_div_quotient0);
                    qfp_input1 <= output_negative4[1] ?
                        -$signed(output_div_quotient1) :
                         $signed(output_div_quotient1);
                    output_value2 <= output_negative4[2] ?
                        -$signed(output_div_quotient2) :
                         $signed(output_div_quotient2);
                    output_value3 <= output_negative4[3] ?
                        -$signed(output_div_quotient3) :
                         $signed(output_div_quotient3);
                    output_subpair <= 0;
                    qfp_kick <= 1;
                    state <= S_OUTPUT_CONVERT_WAIT;
                end
                S_OUTPUT_CONVERT_WAIT: if (qfp_done) begin
                    packer_source_data_o <= {qfp_output1, qfp_output0};
                    state <= S_OUTPUT_PACK;
                end
                S_OUTPUT_PACK: if (packer_source_ready_i) begin
                    packer_source_valid_o <= 1;
                    if (!output_subpair) begin
                        qfp_input0 <= output_value2;
                        qfp_input1 <= output_value3;
                        qfp_kick <= 1;
                        output_subpair <= 1;
                        state <= S_OUTPUT_CONVERT_WAIT;
                    end else if (index == 124) begin
                        if (head == 15) begin
                            pack_group_blocks <= 0;
                            state <= S_PACK_WAIT_BLOCK;
                        end else begin
                            head <= head + 1'b1;
                            index <= 0;
                            square_sum <= 0;
                            state <= S_Q_LOAD_REQ;
                        end
                    end else begin
                        index <= index + 3'd4;
                        state <= S_OUTPUT_READ;
                    end
                end
                S_PACK_WAIT_BLOCK: if (packer_block_valid_i) begin
                    pack_group[pack_group_blocks * 272 +: 272] <=
                        packer_block_data_i;
                    if (pack_group_blocks == 3) begin
                        pack_group_blocks <= 0;
                        pack_group_word <= 0;
                        state <= S_PACK_WRITE_REQ;
                    end else begin
                        pack_group_blocks <= pack_group_blocks + 1'b1;
                    end
                end
                S_PACK_WRITE_REQ: if (!write_active) begin
                    write_address <= output_base +
                        (pack_group_index * 30'd136);
                    write_length <= 16;
                    write_stream <= 2;
                    write_kick <= 1;
                    state <= S_PACK_WRITE_WAIT;
                end
                S_PACK_WRITE_WAIT: begin
                    if (write_beat_accept && pack_group_word != 16) begin
                        pack_group <= pack_group >> 64;
                        pack_group_word <= pack_group_word + 1'b1;
                    end
                    if (write_done) begin
                        pack_group_word <= 0;
                        if (pack_group_index == 15) begin
                            state <= S_DONE;
                        end else begin
                            pack_group_index <= pack_group_index + 1'b1;
                            state <= S_PACK_WAIT_BLOCK;
                        end
                    end
                end
                S_DONE: state <= S_IDLE;
                default: begin error_o <= 1; state <= S_DONE; end
            endcase
        end
    end
endmodule

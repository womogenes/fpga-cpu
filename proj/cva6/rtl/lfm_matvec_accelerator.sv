// DDR-streaming integer accelerator for the Q4_0 matrices and Q6_K output
// embedding used by LFM2.5-350M. Floating-point scales are represented as
// signed Q30 integers, so the datapath does not require an FPGA FPU.
module lfm_matvec_accelerator #(
    parameter integer ID_WIDTH = 5
) (
    input  logic        clk_i,
    input  logic        fp_clk_i,
    input  logic        rst_ni,
    input  logic        fp_rst_ni,
    input  logic [31:0] command_i,
    output logic [31:0] status_o,
    output logic        bus_request_o,
    input  logic        bus_grant_i,
    AXI_BUS.Master      axi
);

    localparam logic [29:0] MODEL_BASE = 30'h2f00_0000;
    localparam logic [29:0] RESIDUAL_SUM_ADDRESS = 30'h3e28_0000;
    localparam integer MAX_INPUT = 6144;
    localparam integer MAX_BLOCKS = MAX_INPUT / 32;

    typedef enum logic [5:0] {
        IDLE,
        WAIT_GRANT,
        START_INPUT,
        LOAD_INPUT,
        PREPARE_WEIGHTS,
        START_WEIGHTS,
        LAUNCH_WEIGHTS,
        Q4_STREAM,
        Q4_DRAIN_PRODUCT,
        Q4_DRAIN,
        Q4_DRAIN_SUM,
        Q4_SCALE,
        Q4_SCALE_MANTISSA,
        Q4_SCALE_ALIGN_WAIT,
        Q4_SCALE_ALIGN,
        Q4_SCALE_WAIT,
        Q4_SCALE_MULTIPLY,
        Q4_ACCUMULATE,
        Q4_WRITE,
        Q4_WRITE_RESPONSE,
        Q6_STREAM,
        Q6_COMPUTE,
        Q6_DRAIN_PAIR,
        Q6_DRAIN_SCALE,
        Q6_DRAIN_ACCUMULATE,
        Q6_SCALE,
        Q6_SCALE_MANTISSA,
        Q6_SCALE_ALIGN_WAIT,
        Q6_SCALE_ALIGN,
        Q6_SCALE_WAIT,
        Q6_SCALE_MULTIPLY,
        Q6_ACCUMULATE,
        Q6_COMPARE,
        Q1_STREAM,
        Q1_FLUSH,
        RESIDUAL_FENCE_START,
        RESIDUAL_FENCE_WAIT,
        FFN_START,
        FFN_STREAM,
        ATTN_START,
        ATTN_WAIT,
        ABORT_DRAIN,
        FINISH,
        RELEASE_BUS,
        SIGNAL_DONE
    } state_t;

    state_t state;
    logic [31:0] command_meta;
    logic [31:0] command_sync;
    logic        command_toggle_seen;
    logic        command_pending;

    logic [24:0] weight_offset_words;
    logic [24:0] input_offset_words;
    logic [1:0]  input_address_high;
    logic        input_address_absolute;
    logic [24:0] output_offset_words;
    logic [1:0]  output_address_high;
    logic        output_address_absolute;
    logic [29:0] output_base_address;
    logic [15:0] ne0;
    logic [17:0] ne1;
    logic        q6_mode;
    logic        q1_mode;
    logic        q1_argmax_mode;
    logic        q1_packed_mode;
    logic        output_compact_config;
    logic        q1_compact_output;
    logic        compact_hold_valid;
    logic [31:0] compact_hold_data;
    logic [17:0] compact_hold_row;
    logic        ffn_mode;
    logic        attention_mode;
    logic        residual_mode;
    logic [24:0] scalar_low;
    logic [6:0]  scalar_high;

    // Bank the Q8 scales by their index modulo four. A flat 192-entry
    // register array inferred a high-fanout write-enable decoder whose worst
    // physical path ended at the upper FFN-only entries with essentially no
    // 200 MHz margin. Q1 also consumes four consecutive scales together, so
    // these banks remove that decoder and match the read pattern directly.
    logic [15:0] input_scale0 [0:MAX_BLOCKS/4-1];
    logic [15:0] input_scale1 [0:MAX_BLOCKS/4-1];
    logic [15:0] input_scale2 [0:MAX_BLOCKS/4-1];
    logic [15:0] input_scale3 [0:MAX_BLOCKS/4-1];
    // Q1 consumes four adjacent Q8 blocks together.  Keep a synchronous
    // replica of the scale banks for that path instead of feeding its 200 MHz
    // pipeline from the asynchronous scale mux used by the slower Q4/Q6
    // state machines.  The dedicated replicas infer four small RAMs and give
    // the high FFN-only scale entries a full registered read cycle.
    (* ram_style = "block" *) logic [15:0] q1_input_scale0 [0:MAX_BLOCKS/4-1];
    (* ram_style = "block" *) logic [15:0] q1_input_scale1 [0:MAX_BLOCKS/4-1];
    (* ram_style = "block" *) logic [15:0] q1_input_scale2 [0:MAX_BLOCKS/4-1];
    (* ram_style = "block" *) logic [15:0] q1_input_scale3 [0:MAX_BLOCKS/4-1];
    logic [15:0] q1_input_scale_read [0:3];
    logic [7:0]  input_scale_low;
    logic [3:0]  input_scale_write_enable;
    logic [13:0] input_scale_write_address;
    logic [15:0] input_scale_write_data;
    // Eight replicated true-dual-port banks provide sixteen synchronous reads
    // without expanding the activation vector into registers. Q4_0 uses all
    // sixteen ports; Q6_K uses eight of them.
    // Loading and matrix evaluation are mutually exclusive, so port A becomes
    // the write port only while the compact input vector is being received.
    (* ram_style = "block" *) logic signed [7:0] input_q0 [0:MAX_INPUT-1];
    (* ram_style = "block" *) logic signed [7:0] input_q1 [0:MAX_INPUT-1];
    (* ram_style = "block" *) logic signed [7:0] input_q2 [0:MAX_INPUT-1];
    (* ram_style = "block" *) logic signed [7:0] input_q3 [0:MAX_INPUT-1];
    (* ram_style = "block" *) logic signed [7:0] input_q4 [0:MAX_INPUT-1];
    (* ram_style = "block" *) logic signed [7:0] input_q5 [0:MAX_INPUT-1];
    (* ram_style = "block" *) logic signed [7:0] input_q6 [0:MAX_INPUT-1];
    (* ram_style = "block" *) logic signed [7:0] input_q7 [0:MAX_INPUT-1];
    logic [12:0] input_write_address;
    logic [12:0] input_read_a_address [0:7];
    logic [12:0] input_read_b_address [0:7];
    logic signed [7:0] input_read_a [0:7];
    logic signed [7:0] input_read_b [0:7];
    logic input_write_enable;
    logic [7:0] q6_ql [0:127];
    logic [7:0] q6_qh [0:63];
    logic signed [7:0] q6_scales [0:15];

    // Q1_0 streams one 128-weight block as a 2-byte scale beat plus two
    // 8-byte sign-bit beats. Four replicated 64-bit-wide banks provide the
    // 64 activation bytes that pair with each sign-bit beat, so the block
    // datapath keeps pace with the AXI stream and the arithmetic tail runs
    // as a free-running pipeline instead of FSM states.
    (* ram_style = "block" *) logic [63:0] wide_q0 [0:MAX_INPUT/8-1];
    (* ram_style = "block" *) logic [63:0] wide_q1 [0:MAX_INPUT/8-1];
    (* ram_style = "block" *) logic [63:0] wide_q2 [0:MAX_INPUT/8-1];
    (* ram_style = "block" *) logic [63:0] wide_q3 [0:MAX_INPUT/8-1];
    logic [55:0] wide_assemble;
    logic        wide_write_enable;
    logic [9:0]  wide_write_address;
    logic [63:0] wide_write_data;
    logic [9:0]  wide_read_a_address [0:3];
    logic [9:0]  wide_read_b_address [0:3];
    logic [63:0] wide_read_a [0:3];
    logic [63:0] wide_read_b [0:3];
    logic [9:0]  q1_wide_base;

    logic [31:0] input_byte_index;
    logic [15:0] input_block;
    logic [8:0]  input_position;
    logic [31:0] weight_byte_index;
    logic [32:0] weight_block_count;
    logic [8:0]  weight_position;
    logic [15:0] row_block;
    logic [17:0] row_index;
    logic [15:0] row_blocks;
    logic [15:0] weight_scale;
    logic signed [31:0] block_dot;
    logic signed [63:0] row_sum;
    logic signed [63:0] scaled_term;
    logic signed [63:0] best_score;
    logic signed [63:0] q6_row_result;
    logic [17:0] best_token;
    logic [4:0] q6_step;

    // Q1 stream-side capture of the pending block's scales and row bookkeeping.
    logic [15:0] pend_input_scale [0:3];
    logic [15:0] packed_weight_scale [0:3];
    logic        pend_last;
    logic [17:0] pend_row;

    // Q1 free-running compute pipeline. V1 sign-selects one 64-weight half
    // against the wide activation banks and registers eight 4:1 partial sums
    // per 32-weight group; V2 reduces them to group dots and, on the second
    // half, injects the completed block into the exact Q30 scale pipeline
    // (stages s0 through s6 mirror the former Q1_SCALE_* states).
    logic [63:0] v_bits;
    logic        v_half;
    logic        v_valid;
    // R stage: register the wide BRAM outputs (and the matching sign bits
    // and block context) before any arithmetic, keeping the BRAM output
    // routing off the critical path.
    logic [63:0] r_data [0:7];
    logic [63:0] r_bits;
    logic        r_half;
    logic        r_valid;
    logic [15:0] r_wscale;
    logic [15:0] r_iscale [0:3];
    logic        r_last;
    logic [17:0] r_row;
    logic [7:0]  q1w_raw [0:63];
    logic signed [8:0]  q1w_signed [0:63];
    logic signed [10:0] q1w_partial_c [0:1][0:7];
    logic signed [10:0] p_partial [0:1][0:7];
    logic        p_half;
    logic        p_valid;
    logic [15:0] p_wscale;
    logic [15:0] p_iscale [0:3];
    logic        p_last;
    logic [17:0] p_row;
    logic signed [15:0] q1w_dot [0:1];
    logic signed [15:0] h_dot0;
    logic signed [15:0] h_dot1;
    logic signed [15:0] s0_dot [0:3];
    logic [15:0] s0_wscale;
    logic [15:0] s0_iscale [0:3];
    logic        s0_last;
    logic [17:0] s0_row;
    logic        s0_valid;
    logic [4:0]  s0_wexp;
    logic [10:0] s0_wmant;
    logic signed [7:0] s0_wpow;
    logic [4:0]  s0_iexp [0:3];
    logic [10:0] s0_imant [0:3];
    logic signed [7:0] s0_ipow [0:3];
    logic [10:0] s1_wmant [0:3];
    logic [10:0] s1_imant [0:3];
    logic        s1_neg [0:3];
    logic signed [8:0] s1_shift [0:3];
    logic        s1_zero [0:3];
    logic signed [15:0] s1_dot [0:3];
    logic        s1_last;
    logic [17:0] s1_row;
    logic        s1_valid;
    logic [21:0] s2_mprod [0:3];
    logic        s2_neg [0:3];
    logic signed [8:0] s2_shift [0:3];
    logic        s2_zero [0:3];
    logic signed [15:0] s2_dot [0:3];
    logic        s2_last;
    logic [17:0] s2_row;
    logic        s2_valid;
    logic [31:0] s2_aligned [0:3];
    logic signed [31:0] s3_sprod [0:3];
    logic signed [15:0] s3_dot [0:3];
    logic        s3_last;
    logic [17:0] s3_row;
    logic        s3_valid;
    logic signed [31:0] s4_low_product [0:3];
    logic signed [31:0] s4_high_product [0:3];
    logic signed [15:0] s4_correction [0:3];
    logic        s4_last;
    logic [17:0] s4_row;
    logic        s4_valid;
    logic signed [63:0] s5_scaled [0:3];
    logic        s5_last;
    logic [17:0] s5_row;
    logic        s5_valid;
    logic signed [63:0] s6_pair [0:1];
    logic        s6_last;
    logic [17:0] s6_row;
    logic        s6_valid;
    logic signed [63:0] s7_term;
    logic        s7_last;
    logic [17:0] s7_row;
    logic        s7_valid;
    logic        q1_argmax_valid;
    logic [17:0] q1_argmax_row;
    logic signed [63:0] q1_argmax_score;
    logic        q1_argmax_compare_valid;
    logic [17:0] q1_argmax_compare_row;
    logic signed [63:0] q1_argmax_compare_score;
    logic        q1_argmax_compare_better;
    logic signed [63:0] q1_row_accum;
    logic        q1_pipe_busy;
    logic        q1_convert_value_valid;
    logic [17:0] q1_convert_value_row;
    logic signed [63:0] q1_convert_value;
    logic        q1_convert_parts_valid;
    logic [17:0] q1_convert_parts_row;
    logic        q1_convert_high_sign;
    logic [31:0] q1_convert_high_magnitude;
    logic [5:0]  q1_convert_high_msb;
    logic [31:0] q1_convert_low_magnitude;
    logic [5:0]  q1_convert_low_msb;
    logic        q1_convert_trunc_valid;
    logic [17:0] q1_convert_trunc_row;
    logic        q1_convert_trunc_high_sign;
    logic [31:0] q1_convert_trunc_high_base;
    logic [31:0] q1_convert_trunc_high_increment;
    logic [31:0] q1_convert_trunc_low_base;
    logic [31:0] q1_convert_trunc_low_increment;
    logic        q1_convert_round_valid;
    logic [17:0] q1_convert_round_row;
    logic signed [32:0] q1_convert_rounded_high;
    logic [32:0] q1_convert_rounded_low;
    logic        q1_convert_split_valid;
    logic [17:0] q1_convert_split_row;
    logic signed [64:0] q1_convert_split_value;
    logic        q1_convert_mag_valid;
    logic [17:0] q1_convert_mag_row;
    logic        q1_convert_sign;
    logic [64:0] q1_convert_magnitude;
    logic        q1_convert_msb_valid;
    logic [17:0] q1_convert_msb_row;
    logic        q1_convert_msb_sign;
    logic [64:0] q1_convert_msb_magnitude;
    logic [6:0]  q1_convert_msb;
    logic        q1_float_prep_valid;
    logic [17:0] q1_float_prep_row;
    logic        q1_float_prep_sign;
    logic        q1_float_prep_zero;
    logic [8:0]  q1_float_prep_exponent;
    logic [24:0] q1_float_prep_significand;
    logic        q1_float_prep_round_up;
    logic        q1_float_valid;
    logic [17:0] q1_float_row;
    logic [31:0] q1_float_data;
    logic [3:0]  q1_rows_inflight;
    logic        q1_row_launch;

    // Q1 non-blocking row-result writer: results queue in an eight-entry
    // FIFO; the stream backpressures at four entries so in-flight pipeline
    // blocks always fit. B responses are tracked out of band.
    logic [63:0] wfifo_data [0:7];
    logic [29:0] wfifo_addr [0:7];
    logic [3:0]  wfifo_count;
    logic [2:0]  wfifo_rd;
    logic [2:0]  wfifo_wr;
    logic        wr_active;
    logic        wr_aw_done;
    logic        wr_w_done;
    // The DDR interconnect can accept far more than sixteen writes before
    // returning B responses.  A four-bit counter wrapped to zero on physical
    // hardware and let Q1_FLUSH expose DONE while a few result rows were still
    // in flight.  Cover the largest supported matrix row count instead.
    logic [18:0] wr_outstanding;
    logic [29:0] eng_addr;
    logic [63:0] eng_data;
    logic        wfifo_push;
    logic        wfifo_pop;
    logic        wr_issue_done;

    logic        ffn_start;
    logic        ffn_source_ready;
    logic [63:0] ffn_source_buffer_data;
    logic        ffn_source_buffer_valid;
    logic [271:0] ffn_block_data;
    logic        ffn_block_valid;
    logic        ffn_block_ready;
    logic        ffn_busy;
    logic        ffn_done;
    logic [3:0]  ffn_debug_state;
    (* async_reg = "true" *) logic [3:0] ffn_debug_state_meta;
    (* async_reg = "true" *) logic [3:0] ffn_debug_state_sync;
    logic [1087:0] ffn_group;
    logic [2:0]  ffn_group_blocks;
    logic [8:0]  ffn_blocks_accepted;
    logic        ffn_group_drain;
    logic [4:0]  ffn_group_word;
    logic [29:0] ffn_group_address;
    logic        ffn_fifo_push;
    logic        engine_write_mode;
    logic        ffn_reader_finished;
    logic        ffn_compute_finished;

    logic        residual_start;
    logic        residual_source_ready;
    logic [63:0] residual_result_data;
    logic        residual_result_valid;
    logic        residual_result_ready;
    logic [31:0] residual_sum;
    logic        residual_sum_valid;
    logic [7:0]  residual_sum_tag;
    logic        residual_sum_pending;
    logic [7:0]  residual_command_tag;
    logic        residual_busy;
    logic        residual_done;
    logic        residual_done_delayed;
    logic        residual_idle_seen;
    logic        residual_busy_seen;
    logic        residual_done_armed;
    logic [3:0]  residual_debug_state;
    logic [10:0] residual_result_word;
    logic        residual_sum_enqueued;
    logic        residual_fifo_push;
    logic        residual_sum_push;
    logic [31:0] residual_fence_expected;
    logic        residual_fence_match;
    logic        residual_sum_high_select;

    logic        attention_start;
    logic        attention_busy;
    logic        attention_done;
    logic        attention_error;
    logic [ID_WIDTH-1:0] attention_arid;
    logic [29:0] attention_araddr;
    logic [7:0] attention_arlen;
    logic [2:0] attention_arsize;
    logic [1:0] attention_arburst;
    logic attention_arvalid;
    logic attention_rready;
    logic [ID_WIDTH-1:0] attention_awid;
    logic [29:0] attention_awaddr;
    logic [7:0] attention_awlen;
    logic attention_awvalid;
    logic [63:0] attention_wdata;
    logic [7:0] attention_wstrb;
    logic attention_wlast;
    logic attention_wvalid;
    logic attention_bready;
    logic attention_packer_start;
    logic [63:0] attention_packer_source_data;
    logic attention_packer_source_valid;
    logic attention_packer_source_ready;
    logic attention_packer_block_ready;
    logic shared_packer_source_ready;
    logic shared_packer_busy;
    logic shared_packer_done;

    // Convert the exact signed Q30 accumulator to IEEE-754 binary32 with
    // round-to-nearest, ties-to-even. Standard Q1 commands return this value
    // in the low word of their existing eight-byte result slots; argmax keeps
    // comparing the original integer accumulator.
    function automatic logic [31:0] q30_magnitude_to_float_bits(
        input logic sign,
        input logic [64:0] magnitude,
        input logic [6:0] msb_value);
        logic [65:0] significand;
        logic [64:0] low_mask;
        logic guard_bit;
        logic sticky_bit;
        integer shift;
        integer exponent;
        begin
            if (magnitude == 0) begin
                q30_magnitude_to_float_bits = 32'b0;
            end else begin
                exponent = msb_value + 97;
                significand = '0;
                if (msb_value <= 23) begin
                    significand = {1'b0, magnitude} << (23 - msb_value);
                end else begin
                    shift = msb_value - 23;
                    significand = {1'b0, magnitude >> shift};
                    guard_bit = magnitude[shift - 1];
                    low_mask = shift > 1 ?
                        ((65'd1 << (shift - 1)) - 65'd1) : 65'd0;
                    sticky_bit = |(magnitude & low_mask);
                    if (guard_bit && (sticky_bit || significand[0]))
                        significand = significand + 66'd1;
                    if (significand[24]) begin
                        significand = significand >> 1;
                        exponent = exponent + 1;
                    end
                end

                if (exponent >= 255)
                    q30_magnitude_to_float_bits = {sign, 8'hff, 23'b0};
                else if (exponent <= 0)
                    q30_magnitude_to_float_bits = {sign, 31'b0};
                else
                    q30_magnitude_to_float_bits =
                        {sign, exponent[7:0], significand[22:0]};
            end
        end
    endfunction

    // Return the exactly representable integer selected by an IEEE-754
    // integer-to-binary32 conversion (round-to-nearest, ties-to-even). The
    // CPU compatibility path converts the signed high and unsigned low Q30
    // halves separately before adding them, so reproducing that intermediate
    // rounding is required for token-exact results.
    function automatic logic [32:0] round_integer_to_fp32_precision(
        input logic [31:0] magnitude,
        input logic [5:0] msb_value);
        logic [32:0] significand;
        logic [31:0] low_mask;
        logic guard_bit;
        logic sticky_bit;
        integer shift;
        begin
            if (magnitude == 0 || msb_value <= 23) begin
                round_integer_to_fp32_precision = {1'b0, magnitude};
            end else begin
                shift = msb_value - 23;
                significand = {1'b0, magnitude >> shift};
                guard_bit = magnitude[shift - 1];
                low_mask = shift > 1 ?
                    ((32'd1 << (shift - 1)) - 32'd1) : 32'd0;
                sticky_bit = |(magnitude & low_mask);
                if (guard_bit && (sticky_bit || significand[0]))
                    significand = significand + 33'd1;
                round_integer_to_fp32_precision = significand << shift;
            end
        end
    endfunction

    function automatic logic [5:0] integer_msb32(
        input logic [31:0] magnitude);
        integer bit_index;
        begin
            integer_msb32 = '0;
            for (bit_index = 0; bit_index < 32; bit_index = bit_index + 1)
                if (magnitude[bit_index]) integer_msb32 = bit_index[5:0];
        end
    endfunction

    function automatic logic [31:0] fp32_integer_round_base(
        input logic [31:0] magnitude,
        input logic [5:0] msb_value);
        logic [31:0] low_mask;
        integer shift;
        begin
            if (magnitude == 0 || msb_value <= 23) begin
                fp32_integer_round_base = magnitude;
            end else begin
                shift = msb_value - 23;
                low_mask = (32'd1 << shift) - 32'd1;
                fp32_integer_round_base = magnitude & ~low_mask;
            end
        end
    endfunction

    function automatic logic [31:0] fp32_integer_round_increment(
        input logic [31:0] magnitude,
        input logic [5:0] msb_value);
        logic [31:0] low_mask;
        logic [31:0] remainder;
        logic [31:0] halfway;
        integer shift;
        begin
            fp32_integer_round_increment = 0;
            if (magnitude != 0 && msb_value > 23) begin
                shift = msb_value - 23;
                low_mask = (32'd1 << shift) - 32'd1;
                remainder = magnitude & low_mask;
                halfway = 32'd1 << (shift - 1);
                if (remainder > halfway ||
                    (remainder == halfway && magnitude[shift]))
                    fp32_integer_round_increment = 32'd1 << shift;
            end
        end
    endfunction

    function automatic logic [24:0] q30_float_significand_base(
        input logic [64:0] magnitude,
        input logic [6:0] msb_value);
        integer shift;
        begin
            if (magnitude == 0)
                q30_float_significand_base = 0;
            else if (msb_value <= 23)
                q30_float_significand_base = magnitude << (23 - msb_value);
            else begin
                shift = msb_value - 23;
                q30_float_significand_base = magnitude >> shift;
            end
        end
    endfunction

    function automatic logic q30_float_round_up(
        input logic [64:0] magnitude,
        input logic [6:0] msb_value);
        logic [64:0] low_mask;
        logic guard_bit;
        logic sticky_bit;
        integer shift;
        begin
            q30_float_round_up = 0;
            if (magnitude != 0 && msb_value > 23) begin
                shift = msb_value - 23;
                guard_bit = magnitude[shift - 1];
                low_mask = shift > 1 ?
                    ((65'd1 << (shift - 1)) - 65'd1) : 65'd0;
                sticky_bit = |(magnitude & low_mask);
                q30_float_round_up = guard_bit &&
                    (sticky_bit || magnitude[shift]);
            end
        end
    endfunction

    function automatic logic [31:0] q30_float_pack(
        input logic sign,
        input logic zero,
        input logic [8:0] exponent_in,
        input logic [24:0] significand_in,
        input logic round_up);
        logic [25:0] significand;
        logic [8:0] exponent;
        begin
            significand = {1'b0, significand_in} + round_up;
            exponent = exponent_in;
            if (significand[24]) begin
                significand = significand >> 1;
                exponent = exponent + 1'b1;
            end
            if (zero || exponent == 0)
                q30_float_pack = {sign, 31'b0};
            else if (exponent >= 255)
                q30_float_pack = {sign, 8'hff, 23'b0};
            else
                q30_float_pack = {sign, exponent[7:0], significand[22:0]};
        end
    endfunction

    logic done_toggle;
    logic error_flag;
    logic [63:0] cycle_count;
    logic        watchdog_fired;
    state_t      watchdog_state;
    logic [17:0] watchdog_diagnostic;

    logic reader_start;
    logic [29:0] reader_address;
    logic [31:0] reader_byte_count;
    logic reader_busy;
    logic reader_done;
    logic reader_error;
    logic [63:0] reader_data;
    logic reader_byte_valid;
    logic reader_byte_ready;
    logic [3:0] reader_unit_bytes;

    logic [ID_WIDTH-1:0] reader_arid;
    logic [29:0] reader_araddr;
    logic [7:0] reader_arlen;
    logic [2:0] reader_arsize;
    logic [1:0] reader_arburst;
    logic reader_arvalid;
    logic reader_rready;

    logic write_aw_sent;
    logic write_w_sent;
    logic [29:0] write_address;
    logic [63:0] write_data;

    logic signed [8:0]  q4_value [0:15];
    logic signed [8:0]  q4_value_pipe [0:15];
    logic signed [7:0]  q4_input_pipe [0:15];
    logic signed [16:0] q4_product [0:15];
    logic signed [21:0] q4_product_sum;
    logic signed [21:0] q4_product_reduction;
    logic signed [17:0] q4_sum_pair [0:7];
    logic signed [18:0] q4_sum_group [0:3];
    logic signed [18:0] q4_sum_group_pipe [0:3];
    logic signed [19:0] q4_sum_half [0:1];
    logic signed [31:0] q6_contribution;
    logic [21:0] scale_mantissa_product;
    logic signed [8:0]  scale_shift;
    logic                scale_zero;
    logic                scale_negative;
    logic signed [31:0] scale_product;
    logic signed [63:0] wide_scaled_product;

    logic [4:0] scale_ea;
    logic [4:0] scale_eb;
    logic [10:0] scale_ma;
    logic [10:0] scale_mb;
    logic [10:0] scale_mantissa_a;
    logic [10:0] scale_mantissa_b;
    logic signed [7:0] scale_pa;
    logic signed [7:0] scale_pb;
    logic [31:0] aligned_scale_product;
    logic [15:0] active_input_scale;

    logic [15:0] active_weight_scale;
    logic [12:0] q4_prefetch_base;
    logic [12:0] q4_prefetch_next_base;
    logic        q4_prefetch_eight;
    logic   q4_accept;

    logic [4:0] q6_l0;
    logic [4:0] q6_l1;
    logic q6_half;
    integer q6_ql_base;
    integer q6_qh_base;
    integer q6_prefetch_step;
    logic [4:0] q6_prefetch_l0;
    logic [4:0] q6_prefetch_l1;
    logic q6_prefetch_half;
    logic [12:0] q6_prefetch_base;
    integer q6_scale_base;
    logic signed [6:0]  q6_value [0:7];
    logic signed [6:0]  q6_value_pipe [0:7];
    logic signed [7:0]  q6_input_pipe [0:7];
    logic signed [7:0]  q6_scale_value [0:7];
    logic signed [7:0]  q6_scale_decode [0:7];
    logic signed [14:0] q6_pair_product [0:7];
    logic signed [7:0]  q6_scale_pipe [0:7];
    logic signed [22:0] q6_scaled_product [0:7];
    logic signed [25:0] q6_sum_group0;
    logic signed [25:0] q6_sum_group1;
    integer lane;
    integer state_lane;

    function automatic logic [31:0] align_q30_magnitude(
        input logic [21:0] mantissa,
        input logic signed [8:0] shift,
        input logic zero
    );
        logic [31:0] magnitude;
        logic [31:0] rounding;
        logic [8:0] right_shift;
        begin
            magnitude = {{10{1'b0}}, mantissa};
            right_shift = $unsigned(-shift);
            rounding = 32'd1 << (right_shift - 1'b1);
            if (zero)
                align_q30_magnitude = 32'd0;
            else if (shift >= 0)
                align_q30_magnitude = magnitude << shift;
            else
                align_q30_magnitude = (magnitude + rounding) >> right_shift;
        end
    endfunction

    always_comb begin
        for (lane = 0; lane < 16; lane = lane + 1)
            q4_value[lane] = $signed({1'b0, reader_data[lane*4 +: 4]}) - 9'sd8;
        for (lane = 0; lane < 8; lane = lane + 1)
            q4_sum_pair[lane] =
                {q4_product[lane*2][16], q4_product[lane*2]} +
                {q4_product[lane*2 + 1][16], q4_product[lane*2 + 1]};
        for (lane = 0; lane < 4; lane = lane + 1)
            q4_sum_group[lane] =
                {q4_sum_pair[lane*2][17], q4_sum_pair[lane*2]} +
                {q4_sum_pair[lane*2 + 1][17], q4_sum_pair[lane*2 + 1]};
        q4_sum_half[0] =
            {q4_sum_group_pipe[0][18], q4_sum_group_pipe[0]} +
            {q4_sum_group_pipe[1][18], q4_sum_group_pipe[1]};
        q4_sum_half[1] =
            {q4_sum_group_pipe[2][18], q4_sum_group_pipe[2]} +
            {q4_sum_group_pipe[3][18], q4_sum_group_pipe[3]};
        q4_product_reduction =
            {{2{q4_sum_half[0][19]}}, q4_sum_half[0]} +
            {{2{q4_sum_half[1][19]}}, q4_sum_half[1]};

        // Each Q4 cycle consumes eight packed bytes (sixteen weights).
        // Request the following activation slice only when the current unit is
        // accepted so synchronous BRAM data remains aligned across AXI gaps.
        // The two candidate bases are pure functions of registers; the
        // late-arriving reader handshake only picks between them in the final
        // per-lane mux, keeping the BRAM address arithmetic off that path.
        // Q4 has only two activation slices per block.  Spell out the three
        // stream positions instead of using generic integer subtract/multiply
        // arithmetic, which otherwise creates a long counter-to-BRAM-address
        // carry chain at 200 MHz.
        // Q4 only visits positions 0, 1, and 2.  Express the next activation
        // half from those low state bits so the unrelated upper bits of the
        // shared stream counter cannot enter the BRAM address cone.
        q4_prefetch_eight = weight_position[1] ||
                            (q4_accept && weight_position[0]);
        q4_prefetch_base = {row_block[7:0], 5'b0};
        q4_prefetch_next_base = q4_prefetch_base +
                                (q4_prefetch_eight ? 13'd8 : 13'd0);
        // Q4_STREAM is an unconditional reader consumer.  Keep its capture
        // enable independent of the shared ready mux so the mutually
        // exclusive FFN mailbox backpressure cannot become a false timing
        // path into every Q4/Q6 storage register after synthesis.
        q4_accept = state == Q4_STREAM && reader_byte_valid;

        // Each Q6 cycle decodes two positions from each of four 64-value
        // quarters, for eight exact products per cycle.
        q6_l0 = {q6_step[3:0], 1'b0};
        q6_l1 = q6_l0 + 1'b1;
        q6_half = q6_step[4];
        q6_ql_base = q6_half ? 64 : 0;
        q6_qh_base = q6_half ? 32 : 0;
        q6_scale_base = (q6_half ? 8 : 0) + (q6_l0[4] ? 1 : 0);
        q6_value[0] = $signed({1'b0, q6_ql[q6_ql_base + q6_l0][3:0]}) +
                        $signed({1'b0, q6_qh[q6_qh_base + q6_l0][1:0], 4'b0}) - 7'sd32;
        q6_value[1] = $signed({1'b0, q6_ql[q6_ql_base + 32 + q6_l0][3:0]}) +
                        $signed({1'b0, q6_qh[q6_qh_base + q6_l0][3:2], 4'b0}) - 7'sd32;
        q6_value[2] = $signed({1'b0, q6_ql[q6_ql_base + q6_l0][7:4]}) +
                        $signed({1'b0, q6_qh[q6_qh_base + q6_l0][5:4], 4'b0}) - 7'sd32;
        q6_value[3] = $signed({1'b0, q6_ql[q6_ql_base + 32 + q6_l0][7:4]}) +
                        $signed({1'b0, q6_qh[q6_qh_base + q6_l0][7:6], 4'b0}) - 7'sd32;
        q6_value[4] = $signed({1'b0, q6_ql[q6_ql_base + q6_l1][3:0]}) +
                        $signed({1'b0, q6_qh[q6_qh_base + q6_l1][1:0], 4'b0}) - 7'sd32;
        q6_value[5] = $signed({1'b0, q6_ql[q6_ql_base + 32 + q6_l1][3:0]}) +
                        $signed({1'b0, q6_qh[q6_qh_base + q6_l1][3:2], 4'b0}) - 7'sd32;
        q6_value[6] = $signed({1'b0, q6_ql[q6_ql_base + q6_l1][7:4]}) +
                        $signed({1'b0, q6_qh[q6_qh_base + q6_l1][5:4], 4'b0}) - 7'sd32;
        q6_value[7] = $signed({1'b0, q6_ql[q6_ql_base + 32 + q6_l1][7:4]}) +
                        $signed({1'b0, q6_qh[q6_qh_base + q6_l1][7:6], 4'b0}) - 7'sd32;
        q6_scale_value[0] = q6_scales[q6_scale_base];
        q6_scale_value[1] = q6_scales[q6_scale_base + 2];
        q6_scale_value[2] = q6_scales[q6_scale_base + 4];
        q6_scale_value[3] = q6_scales[q6_scale_base + 6];
        q6_scale_value[4] = q6_scales[q6_scale_base];
        q6_scale_value[5] = q6_scales[q6_scale_base + 2];
        q6_scale_value[6] = q6_scales[q6_scale_base + 4];
        q6_scale_value[7] = q6_scales[q6_scale_base + 6];
        q6_sum_group0 =
            {{3{q6_scaled_product[0][22]}}, q6_scaled_product[0]} +
            {{3{q6_scaled_product[1][22]}}, q6_scaled_product[1]} +
            {{3{q6_scaled_product[2][22]}}, q6_scaled_product[2]} +
            {{3{q6_scaled_product[3][22]}}, q6_scaled_product[3]};
        q6_sum_group1 =
            {{3{q6_scaled_product[4][22]}}, q6_scaled_product[4]} +
            {{3{q6_scaled_product[5][22]}}, q6_scaled_product[5]} +
            {{3{q6_scaled_product[6][22]}}, q6_scaled_product[6]} +
            {{3{q6_scaled_product[7][22]}}, q6_scaled_product[7]};
        q6_contribution = {{6{q6_sum_group0[25]}}, q6_sum_group0} +
                          {{6{q6_sum_group1[25]}}, q6_sum_group1};

        // Q1 V1: sign-select one 64-weight half from the registered wide-bank
        // data and form eight 4:1 partial sums for each 32-weight group.
        for (lane = 0; lane < 64; lane = lane + 1)
            q1w_raw[lane] = r_data[lane >> 3][(lane & 7)*8 +: 8];
        for (lane = 0; lane < 64; lane = lane + 1)
            q1w_signed[lane] = r_bits[lane] ?
                 $signed({q1w_raw[lane][7], q1w_raw[lane]}) :
                -$signed({q1w_raw[lane][7], q1w_raw[lane]});
        for (lane = 0; lane < 8; lane = lane + 1) begin
            q1w_partial_c[0][lane] =
                {{2{q1w_signed[lane*4][8]}},     q1w_signed[lane*4]} +
                {{2{q1w_signed[lane*4 + 1][8]}}, q1w_signed[lane*4 + 1]} +
                {{2{q1w_signed[lane*4 + 2][8]}}, q1w_signed[lane*4 + 2]} +
                {{2{q1w_signed[lane*4 + 3][8]}}, q1w_signed[lane*4 + 3]};
            q1w_partial_c[1][lane] =
                {{2{q1w_signed[32 + lane*4][8]}},     q1w_signed[32 + lane*4]} +
                {{2{q1w_signed[32 + lane*4 + 1][8]}}, q1w_signed[32 + lane*4 + 1]} +
                {{2{q1w_signed[32 + lane*4 + 2][8]}}, q1w_signed[32 + lane*4 + 2]} +
                {{2{q1w_signed[32 + lane*4 + 3][8]}}, q1w_signed[32 + lane*4 + 3]};
        end

        // Q1 V2: reduce the registered partial sums into two group dots.
        q1w_dot[0] =
            {{5{p_partial[0][0][10]}}, p_partial[0][0]} +
            {{5{p_partial[0][1][10]}}, p_partial[0][1]} +
            {{5{p_partial[0][2][10]}}, p_partial[0][2]} +
            {{5{p_partial[0][3][10]}}, p_partial[0][3]} +
            {{5{p_partial[0][4][10]}}, p_partial[0][4]} +
            {{5{p_partial[0][5][10]}}, p_partial[0][5]} +
            {{5{p_partial[0][6][10]}}, p_partial[0][6]} +
            {{5{p_partial[0][7][10]}}, p_partial[0][7]};
        q1w_dot[1] =
            {{5{p_partial[1][0][10]}}, p_partial[1][0]} +
            {{5{p_partial[1][1][10]}}, p_partial[1][1]} +
            {{5{p_partial[1][2][10]}}, p_partial[1][2]} +
            {{5{p_partial[1][3][10]}}, p_partial[1][3]} +
            {{5{p_partial[1][4][10]}}, p_partial[1][4]} +
            {{5{p_partial[1][5][10]}}, p_partial[1][5]} +
            {{5{p_partial[1][6][10]}}, p_partial[1][6]} +
            {{5{p_partial[1][7][10]}}, p_partial[1][7]};

        // Q1 T0: decompose the captured fp16 scales exactly as the former
        // Q1_SCALE_PREP state did.
        s0_wexp = s0_wscale[14:10];
        s0_wmant = s0_wexp == 0 ? {1'b0, s0_wscale[9:0]} :
                                  {1'b1, s0_wscale[9:0]};
        s0_wpow = s0_wexp == 0 ? -8'sd24 :
                  $signed({3'b000, s0_wexp}) - 8'sd25;
        for (lane = 0; lane < 4; lane = lane + 1) begin
            s0_iexp[lane] = s0_iscale[lane][14:10];
            s0_imant[lane] = s0_iexp[lane] == 0 ?
                             {1'b0, s0_iscale[lane][9:0]} :
                             {1'b1, s0_iscale[lane][9:0]};
            s0_ipow[lane] = s0_iexp[lane] == 0 ? -8'sd24 :
                            $signed({3'b000, s0_iexp[lane]}) - 8'sd25;
            s2_aligned[lane] = align_q30_magnitude(
                s2_mprod[lane], s2_shift[lane], s2_zero[lane]);
        end
        q1_pipe_busy = v_valid || r_valid || p_valid || s0_valid ||
                       s1_valid || s2_valid || s3_valid || s4_valid ||
                       s5_valid || s6_valid || s7_valid ||
                       q1_argmax_valid ||
                       q1_argmax_compare_valid ||
                       q1_convert_value_valid ||
                       q1_convert_parts_valid || q1_convert_trunc_valid ||
                       q1_convert_round_valid ||
                       q1_convert_split_valid || q1_convert_mag_valid ||
                       q1_convert_msb_valid || q1_float_prep_valid ||
                       q1_float_valid;
        q1_row_launch = state == Q1_STREAM && !q1_argmax_mode &&
                        reader_byte_valid && reader_byte_ready &&
                        ((q1_packed_mode && weight_position == 64 &&
                          row_block + 1 == row_blocks) ||
                         (!q1_packed_mode && weight_position == 10 &&
                          row_block + 1 == row_blocks));
        ffn_block_ready = ffn_mode && !residual_mode && !watchdog_fired &&
                          !ffn_group_drain;
        ffn_fifo_push = ffn_mode && !residual_mode && !watchdog_fired &&
                        ffn_group_drain &&
                        wfifo_count < 4'd4;
        residual_result_ready = residual_mode && !watchdog_fired &&
                                wfifo_count < 4'd6;
        residual_fifo_push = residual_result_valid && residual_result_ready;
        // Enqueue only a sum whose bundled-data toggle was observed for this
        // transaction. DONE crosses independently and can otherwise expose
        // the preceding transaction's scalar during back-to-back commands.
        // The residual scalar is the final item in the acknowledged result
        // stream.  Do not use the old independent sum mailbox, whose event
        // and payload could lag a back-to-back command in silicon.
        residual_sum_push = 1'b0;
        wfifo_push = (q1_float_valid &&
                      (!q1_compact_output || compact_hold_valid)) ||
                     ffn_fifo_push || residual_fifo_push ||
                     residual_sum_push;
        wfifo_pop = !wr_active && wfifo_count != 4'd0;
        wr_issue_done = wr_active &&
            (wr_aw_done || (axi.aw_valid && axi.aw_ready)) &&
            (wr_w_done || (axi.w_valid && axi.w_ready));

        input_write_enable = state == LOAD_INPUT && reader_byte_valid &&
                             input_position >= 2;
        input_write_address = (q6_mode ? input_block * 256 : input_block * 32) +
                              (input_position >= 2 ? input_position - 2 : 0);

        // Activation bytes arrive at strictly sequential addresses, so the
        // wide banks are filled by assembling eight bytes into one word.
        wide_write_enable = input_write_enable &&
                            input_write_address[2:0] == 3'd7;
        wide_write_address = input_write_address[12:3];
        wide_write_data = {reader_data[7:0], wide_assemble};

        q6_prefetch_step = state == Q6_COMPUTE && q6_step < 31 ? q6_step + 1 : 0;
        q6_prefetch_l0 = {q6_prefetch_step[3:0], 1'b0};
        q6_prefetch_l1 = q6_prefetch_l0 + 1'b1;
        q6_prefetch_half = q6_prefetch_step[4];
        // A Q6 activation block is exactly 256 bytes: two 128-byte halves,
        // each containing four 32-byte quarters. Build those fields directly
        // instead of cascading generic additions into every BRAM address.
        q6_prefetch_base = {row_block[4:0], q6_prefetch_half, 7'b0};
        // Wide-bank addresses track the stream: the eight words of the half
        // that the next accepted bits beat will pair with. The BRAM output
        // registers refresh every cycle from a held address, so data is
        // stable across AXI stalls.
        q1_wide_base = {row_block[5:0], 4'b0000} +
                       ((state == Q1_STREAM &&
                         ((!q1_packed_mode && weight_position == 10) ||
                          (q1_packed_mode && weight_position != 0 &&
                           weight_position[3:0] == 0))) ?
                        10'd8 : 10'd0);
        for (lane = 0; lane < 4; lane = lane + 1) begin
            wide_read_a_address[lane] = q1_wide_base + lane[9:0];
            wide_read_b_address[lane] = q1_wide_base + 10'd4 + lane[9:0];
        end
        if (q6_mode) begin
            input_read_a_address[0] = {q6_prefetch_base[12:7], 2'b00, q6_prefetch_l0};
            input_read_b_address[0] = {q6_prefetch_base[12:7], 2'b01, q6_prefetch_l0};
            input_read_a_address[1] = {q6_prefetch_base[12:7], 2'b10, q6_prefetch_l0};
            input_read_b_address[1] = {q6_prefetch_base[12:7], 2'b11, q6_prefetch_l0};
            input_read_a_address[2] = {q6_prefetch_base[12:7], 2'b00, q6_prefetch_l1};
            input_read_b_address[2] = {q6_prefetch_base[12:7], 2'b01, q6_prefetch_l1};
            input_read_a_address[3] = {q6_prefetch_base[12:7], 2'b10, q6_prefetch_l1};
            input_read_b_address[3] = {q6_prefetch_base[12:7], 2'b11, q6_prefetch_l1};
            for (lane = 4; lane < 8; lane = lane + 1) begin
                input_read_a_address[lane] = q6_prefetch_base;
                input_read_b_address[lane] = q6_prefetch_base;
            end
        end else if (state == Q4_STREAM) begin
            for (lane = 0; lane < 8; lane = lane + 1) begin
                input_read_a_address[lane] = q4_prefetch_next_base + lane;
                input_read_b_address[lane] = q4_prefetch_next_base + lane + 16;
            end
        end else begin
            // Q1 uses the dedicated wide activation banks and all remaining
            // states ignore the narrow-bank outputs. Holding these addresses
            // at zero keeps the unrelated nine-bit weight stream position
            // out of the BRAM address timing cone.
            for (lane = 0; lane < 8; lane = lane + 1) begin
                input_read_a_address[lane] = '0;
                input_read_b_address[lane] = '0;
            end
        end

        reader_unit_bytes = 4'd1;
        if (state == Q4_STREAM)
            reader_unit_bytes = weight_position == 0 ? 4'd2 : 4'd8;
        else if (state == Q6_STREAM)
            reader_unit_bytes = weight_position == 208 ? 4'd2 : 4'd8;
        else if (state == Q1_STREAM || state == FFN_STREAM ||
                 state == RESIDUAL_FENCE_WAIT)
            reader_unit_bytes = q1_packed_mode ? 4'd8 :
                                (state == FFN_STREAM ||
                                 state == RESIDUAL_FENCE_WAIT ? 4'd8 :
                                 (weight_position == 0 ? 4'd2 : 4'd8));

        active_weight_scale =
            state == Q4_STREAM &&
            weight_position == 0 && reader_byte_valid ?
            reader_data[15:0] : weight_scale;
        case (row_block[1:0])
            2'd0: active_input_scale = input_scale0[row_block >> 2];
            2'd1: active_input_scale = input_scale1[row_block >> 2];
            2'd2: active_input_scale = input_scale2[row_block >> 2];
            default: active_input_scale = input_scale3[row_block >> 2];
        endcase
        scale_ea = active_weight_scale[14:10];
        scale_eb = active_input_scale[14:10];
        scale_ma = scale_ea == 0 ? {1'b0, active_weight_scale[9:0]} :
                                  {1'b1, active_weight_scale[9:0]};
        scale_mb = scale_eb == 0 ? {1'b0, active_input_scale[9:0]} :
                                  {1'b1, active_input_scale[9:0]};
        scale_pa = scale_ea == 0 ? -8'sd24 : $signed({3'b000, scale_ea}) - 8'sd25;
        scale_pb = scale_eb == 0 ? -8'sd24 : $signed({3'b000, scale_eb}) - 8'sd25;
        aligned_scale_product = align_q30_magnitude(
            scale_mantissa_product, scale_shift, scale_zero);
        wide_scaled_product = block_dot * scale_product;
    end

    always_ff @(posedge clk_i) begin
        if (input_scale_write_enable[0]) begin
            input_scale0[input_scale_write_address] <= input_scale_write_data;
            q1_input_scale0[input_scale_write_address] <= input_scale_write_data;
        end
        if (input_scale_write_enable[1]) begin
            input_scale1[input_scale_write_address] <= input_scale_write_data;
            q1_input_scale1[input_scale_write_address] <= input_scale_write_data;
        end
        if (input_scale_write_enable[2]) begin
            input_scale2[input_scale_write_address] <= input_scale_write_data;
            q1_input_scale2[input_scale_write_address] <= input_scale_write_data;
        end
        if (input_scale_write_enable[3]) begin
            input_scale3[input_scale_write_address] <= input_scale_write_data;
            q1_input_scale3[input_scale_write_address] <= input_scale_write_data;
        end
    end

    always_ff @(posedge clk_i) begin
        if (input_write_enable) begin
            input_q0[input_write_address] <= reader_data[7:0];
            input_q1[input_write_address] <= reader_data[7:0];
            input_q2[input_write_address] <= reader_data[7:0];
            input_q3[input_write_address] <= reader_data[7:0];
            input_q4[input_write_address] <= reader_data[7:0];
            input_q5[input_write_address] <= reader_data[7:0];
            input_q6[input_write_address] <= reader_data[7:0];
            input_q7[input_write_address] <= reader_data[7:0];
        end else begin
            input_read_a[0] <= input_q0[input_read_a_address[0]];
            input_read_a[1] <= input_q1[input_read_a_address[1]];
            input_read_a[2] <= input_q2[input_read_a_address[2]];
            input_read_a[3] <= input_q3[input_read_a_address[3]];
            input_read_a[4] <= input_q4[input_read_a_address[4]];
            input_read_a[5] <= input_q5[input_read_a_address[5]];
            input_read_a[6] <= input_q6[input_read_a_address[6]];
            input_read_a[7] <= input_q7[input_read_a_address[7]];
        end
        input_read_b[0] <= input_q0[input_read_b_address[0]];
        input_read_b[1] <= input_q1[input_read_b_address[1]];
        input_read_b[2] <= input_q2[input_read_b_address[2]];
        input_read_b[3] <= input_q3[input_read_b_address[3]];
        input_read_b[4] <= input_q4[input_read_b_address[4]];
        input_read_b[5] <= input_q5[input_read_b_address[5]];
        input_read_b[6] <= input_q6[input_read_b_address[6]];
        input_read_b[7] <= input_q7[input_read_b_address[7]];
    end

    // The Q1 stream holds row_block for at least the scale/header beat before
    // the second 64-weight half consumes these values.  A synchronous read is
    // therefore both naturally aligned and substantially easier to route than
    // the former asynchronous high-entry decoder.
    always_ff @(posedge clk_i) begin
        q1_input_scale_read[0] <= q1_input_scale0[row_block];
        q1_input_scale_read[1] <= q1_input_scale1[row_block];
        q1_input_scale_read[2] <= q1_input_scale2[row_block];
        q1_input_scale_read[3] <= q1_input_scale3[row_block];
    end

    always_ff @(posedge clk_i) begin
        if (input_write_enable && input_write_address[2:0] != 3'd7)
            wide_assemble[input_write_address[2:0]*8 +: 8] <= reader_data[7:0];
        if (wide_write_enable) begin
            wide_q0[wide_write_address] <= wide_write_data;
            wide_q1[wide_write_address] <= wide_write_data;
            wide_q2[wide_write_address] <= wide_write_data;
            wide_q3[wide_write_address] <= wide_write_data;
        end else begin
            wide_read_a[0] <= wide_q0[wide_read_a_address[0]];
            wide_read_a[1] <= wide_q1[wide_read_a_address[1]];
            wide_read_a[2] <= wide_q2[wide_read_a_address[2]];
            wide_read_a[3] <= wide_q3[wide_read_a_address[3]];
        end
        wide_read_b[0] <= wide_q0[wide_read_b_address[0]];
        wide_read_b[1] <= wide_q1[wide_read_b_address[1]];
        wide_read_b[2] <= wide_q2[wide_read_b_address[2]];
        wide_read_b[3] <= wide_q3[wide_read_b_address[3]];
    end

    // Argmax modes report their winning token. On an error, the same field is
    // a latched watchdog diagnostic: outer state [17:12], FFN packer state
    // [11:8], then reader busy/done, FFN busy/done, group drain/block valid,
    // writer active, and nonzero outstanding writes in [7:0].
    // Residual sums are consumed immediately by the CPU. Returning them in
    // two status payload reads avoids relying on cross-master DDR visibility:
    // the accelerator's own AXI read can observe a just-written scalar before
    // the CPU-facing port does. Register 6 bit 24 selects the upper fragment;
    // DONE, error, command acknowledgement, and feature bits remain stable.
    assign status_o = {1'b1, bus_request_o, error_flag, done_toggle,
                       command_toggle_seen, 1'b1,
                       error_flag ? watchdog_diagnostic :
                       (q6_mode || q1_argmax_mode) ? best_token :
                       residual_mode ?
                           (residual_sum_high_select ?
                               {4'b0, residual_fence_expected[31:18]} :
                               residual_fence_expected[17:0]) :
                           {3'b111, 8'b0, state},
                       3'b0, 1'b1, 1'b1, 1'b1, 1'b1, 1'b1};

    assign output_base_address = output_address_absolute ?
        {output_address_high, output_offset_words, 3'b000} :
        MODEL_BASE + {output_offset_words, 3'b000};

    assign axi.aw_id     = attention_mode ? attention_awid :
                           {ID_WIDTH{1'b1}};
    assign engine_write_mode = q1_mode || ffn_mode;
    assign axi.aw_addr   = {{34{1'b0}}, attention_mode ? attention_awaddr :
                           engine_write_mode ? eng_addr : write_address};
    assign axi.aw_len    = attention_mode ? attention_awlen : 8'd0;
    assign axi.aw_size   = 3'd3;
    assign axi.aw_burst  = 2'b01;
    assign axi.aw_lock   = 1'b0;
    assign axi.aw_cache  = 4'b0010;
    assign axi.aw_prot   = 3'b000;
    assign axi.aw_qos    = 4'b0000;
    assign axi.aw_region = 4'b0000;
    assign axi.aw_atop   = 6'b000000;
    assign axi.aw_user   = '0;
    assign axi.aw_valid  = attention_mode ? attention_awvalid :
                           engine_write_mode ? (wr_active && !wr_aw_done) :
                           (state == Q4_WRITE && !write_aw_sent);
    assign axi.w_data    = attention_mode ? attention_wdata :
                           engine_write_mode ? eng_data : write_data;
    assign axi.w_strb    = attention_mode ? attention_wstrb : 8'hff;
    assign axi.w_last    = attention_mode ? attention_wlast : 1'b1;
    assign axi.w_user    = '0;
    assign axi.w_valid   = attention_mode ? attention_wvalid :
                           engine_write_mode ? (wr_active && !wr_w_done) :
                           (state == Q4_WRITE && !write_w_sent);
    assign axi.b_ready   = attention_mode ? attention_bready :
                           engine_write_mode ? 1'b1 :
                           (state == Q4_WRITE_RESPONSE);

    assign axi.ar_id     = attention_mode ? attention_arid : reader_arid;
    assign axi.ar_addr   = {{34{1'b0}}, attention_mode ? attention_araddr :
                           reader_araddr};
    assign axi.ar_len    = attention_mode ? attention_arlen : reader_arlen;
    assign axi.ar_size   = attention_mode ? attention_arsize : reader_arsize;
    assign axi.ar_burst  = attention_mode ? attention_arburst : reader_arburst;
    assign axi.ar_lock   = 1'b0;
    assign axi.ar_cache  = 4'b0010;
    assign axi.ar_prot   = 3'b000;
    assign axi.ar_qos    = 4'b0000;
    assign axi.ar_region = 4'b0000;
    assign axi.ar_user   = '0;
    assign axi.ar_valid  = attention_mode ? attention_arvalid : reader_arvalid;
    assign axi.r_ready   = attention_mode ? attention_rready : reader_rready;

    // Q1_FLUSH also consumes (and discards) any residual stream so an
    // error/trap drain can always retire the reader before the bus releases.
    assign reader_byte_ready = (state == LOAD_INPUT) ||
                               (state == Q4_STREAM) ||
                               (state == Q6_STREAM) ||
                               // FIFO occupancy does not include rows already
                               // in the deep arithmetic/FP32 pipeline, so cap
                               // launches with the separate in-flight count.
                               // Four non-argmax rows can produce at most four
                               // queued writes (two in compact mode), safely
                               // below the eight-entry FIFO while eliminating
                               // the former full pipeline drain between rows.
                               (state == Q1_STREAM && wfifo_count < 4'd6 &&
                                (q1_argmax_mode || row_block != 0 ||
                                 weight_position != 0 ||
                                 q1_rows_inflight < 4'd4)) ||
                               (state == FFN_STREAM && !ffn_group_drain &&
                                (!ffn_source_buffer_valid ||
                                 ffn_source_ready)) ||
                               (state == ABORT_DRAIN) ||
                               (state == RESIDUAL_FENCE_WAIT) ||
                               (state == Q1_FLUSH);

    bonsai_attention #(
        .ID_WIDTH(ID_WIDTH)
    ) i_attention (
        .clk_i,
        .rst_ni,
        .start_i(attention_start),
        .raw_base_i(input_address_absolute ?
            {input_address_high, input_offset_words, 3'b000} :
            MODEL_BASE + {input_offset_words, 3'b000}),
        .output_base_i(output_base_address),
        .table_base_i(30'h3e20_0000),
        .layer_i(ne0[12:8]),
        .position_i(ne0[7:0]),
        .busy_o(attention_busy),
        .done_o(attention_done),
        .error_o(attention_error),
        .arid_o(attention_arid),
        .araddr_o(attention_araddr),
        .arlen_o(attention_arlen),
        .arsize_o(attention_arsize),
        .arburst_o(attention_arburst),
        .arvalid_o(attention_arvalid),
        .arready_i(axi.ar_ready),
        .rid_i(axi.r_id),
        .rdata_i(axi.r_data),
        .rresp_i(axi.r_resp),
        .rlast_i(axi.r_last),
        .rvalid_i(axi.r_valid),
        .rready_o(attention_rready),
        .awid_o(attention_awid),
        .awaddr_o(attention_awaddr),
        .awlen_o(attention_awlen),
        .awvalid_o(attention_awvalid),
        .awready_i(axi.aw_ready),
        .wdata_o(attention_wdata),
        .wstrb_o(attention_wstrb),
        .wlast_o(attention_wlast),
        .wvalid_o(attention_wvalid),
        .wready_i(axi.w_ready),
        .bid_i(axi.b_id),
        .bresp_i(axi.b_resp),
        .bvalid_i(axi.b_valid),
        .bready_o(attention_bready),
        .packer_start_o(attention_packer_start),
        .packer_source_data_o(attention_packer_source_data),
        .packer_source_valid_o(attention_packer_source_valid),
        .packer_source_ready_i(attention_packer_source_ready),
        .packer_block_data_i(ffn_block_data),
        .packer_block_valid_i(ffn_block_valid),
        .packer_block_ready_o(attention_packer_block_ready)
    );

    lfm_axi_byte_reader #(
        .ID_WIDTH(ID_WIDTH)
    ) i_reader (
        .clk_i,
        .rst_ni,
        .start_i(reader_start),
        .address_i(reader_address),
        .byte_count_i(reader_byte_count),
        .unit_bytes_i(reader_unit_bytes),
        .busy_o(reader_busy),
        .done_o(reader_done),
        .error_o(reader_error),
        .data_o(reader_data),
        .byte_valid_o(reader_byte_valid),
        .byte_ready_i(reader_byte_ready),
        .arid_o(reader_arid),
        .araddr_o(reader_araddr),
        .arlen_o(reader_arlen),
        .arsize_o(reader_arsize),
        .arburst_o(reader_arburst),
        .arvalid_o(reader_arvalid),
        .arready_i(axi.ar_ready),
        .rid_i(axi.r_id),
        .rdata_i(axi.r_data),
        .rresp_i(axi.r_resp),
        .rlast_i(axi.r_last),
        .rvalid_i(axi.r_valid),
        .rready_o(reader_rready)
    );

    // Break the reader's byte-valid decode before it reaches the FFN packer's
    // row-control enables. This one-entry elastic stage can consume and refill
    // in the same cycle, so sustained stream throughput remains one word per
    // clock while the cross-module control path ends at a local register.
    always_ff @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
            ffn_source_buffer_data <= '0;
            ffn_source_buffer_valid <= 1'b0;
        end else if (state != FFN_STREAM) begin
            ffn_source_buffer_valid <= 1'b0;
        end else begin
            case ({reader_byte_valid && reader_byte_ready,
                   ffn_source_buffer_valid && ffn_source_ready &&
                   !ffn_group_drain})
                2'b10, 2'b11: begin
                    ffn_source_buffer_data <= reader_data;
                    ffn_source_buffer_valid <= 1'b1;
                end
                2'b01: ffn_source_buffer_valid <= 1'b0;
                default: begin end
            endcase
        end
    end

    bonsai_ffn_packer i_ffn_packer (
        .clk_i,
        .fp_clk_i,
        .rst_ni,
        .fp_rst_ni,
        .start_i(attention_mode ? attention_packer_start : ffn_start),
        .norm_mode_i(!attention_mode && ne1 == 18'd2),
        .vector_mode_i(attention_mode || ne1 == 18'd3),
        .compact_source_i(!attention_mode && output_compact_config),
        .element_count_i(attention_mode ? 13'd2048 : ne0[12:0]),
        .norm_scale_i(attention_mode ? 32'h3f80_0000 :
                                      {scalar_high, scalar_low}),
        .source_data_i(attention_mode ? attention_packer_source_data :
                                        ffn_source_buffer_data),
        .source_valid_i(attention_mode ? attention_packer_source_valid :
            (ffn_source_buffer_valid && !ffn_group_drain && !residual_mode)),
        .source_ready_o(shared_packer_source_ready),
        .block_data_o(ffn_block_data),
        .block_valid_o(ffn_block_valid),
        .block_ready_i(attention_mode ? attention_packer_block_ready :
                                       ffn_block_ready),
        .busy_o(shared_packer_busy),
        .done_o(shared_packer_done),
        .debug_state_o(ffn_debug_state)
    );

    bonsai_residual i_residual (
        .clk_i,
        .fp_clk_i,
        .rst_ni,
        .fp_rst_ni,
        .start_i(residual_start),
        .source_data_i(ffn_source_buffer_data),
        .source_valid_i(ffn_source_buffer_valid && residual_mode),
        .source_ready_o(residual_source_ready),
        .result_data_o(residual_result_data),
        .result_valid_o(residual_result_valid),
        .result_ready_i(residual_result_ready),
        .sum_o(residual_sum),
        .sum_valid_o(residual_sum_valid),
        .sum_tag_o(residual_sum_tag),
        .busy_o(residual_busy),
        .done_o(residual_done),
        .debug_state_o(residual_debug_state)
    );

    assign attention_packer_source_ready = attention_mode &&
                                           shared_packer_source_ready;
    assign ffn_source_ready = !attention_mode &&
                              (residual_mode ? residual_source_ready :
                                               shared_packer_source_ready);
    assign ffn_busy = !attention_mode &&
                      (residual_mode ? residual_busy : shared_packer_busy);
    assign ffn_done = !attention_mode &&
                      (residual_mode ?
                          (residual_done_armed && residual_done) :
                          shared_packer_done);

    always_ff @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
            command_meta        <= '0;
            command_sync        <= '0;
            command_toggle_seen <= 1'b0;
            command_pending     <= 1'b0;
        end else begin
            command_meta <= command_i;
            command_sync <= command_meta;
            if (!command_pending &&
                command_sync[31] != command_toggle_seen) begin
                // The toggle and payload cross the clock boundary together.
                // Hold the source unacknowledged for one additional DDR
                // clock so every payload bit has settled before consuming
                // the command.  Software keeps command_i stable until the
                // acknowledgement phase changes.
                command_pending <= 1'b1;
            end else if (command_pending) begin
                command_toggle_seen <= command_sync[31];
                command_pending <= 1'b0;
            end
        end
    end

    // The detailed transform controller runs on fp_clk_i. Its state remains
    // stable during a deadlock, and this ordinary two-flop sampling chain
    // gives the DDR-domain watchdog a safe diagnostic snapshot.
    always_ff @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
            ffn_debug_state_meta <= '0;
            ffn_debug_state_sync <= '0;
        end else begin
            ffn_debug_state_meta <= residual_mode ? residual_debug_state :
                                                     ffn_debug_state;
            ffn_debug_state_sync <= ffn_debug_state_meta;
        end
    end

    always_ff @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
            state               <= IDLE;
            bus_request_o       <= 1'b0;
            reader_start        <= 1'b0;
            reader_address      <= '0;
            reader_byte_count   <= '0;
            weight_offset_words <= '0;
            input_offset_words  <= '0;
            input_address_high  <= '0;
            input_address_absolute <= 1'b0;
            output_offset_words <= '0;
            output_address_high <= '0;
            output_address_absolute <= 1'b0;
            ne0                 <= '0;
            ne1                 <= '0;
            q6_mode             <= 1'b0;
            q1_mode             <= 1'b0;
            q1_argmax_mode      <= 1'b0;
            q1_packed_mode      <= 1'b0;
            output_compact_config <= 1'b0;
            q1_compact_output   <= 1'b0;
            compact_hold_valid  <= 1'b0;
            compact_hold_data   <= '0;
            compact_hold_row    <= '0;
            ffn_mode            <= 1'b0;
            attention_mode      <= 1'b0;
            residual_mode       <= 1'b0;
            scalar_low          <= '0;
            scalar_high         <= '0;
            input_byte_index    <= '0;
            input_block         <= '0;
            input_position      <= '0;
            input_scale_low     <= '0;
            input_scale_write_enable <= '0;
            input_scale_write_address <= '0;
            input_scale_write_data <= '0;
            weight_byte_index   <= '0;
            weight_block_count  <= '0;
            weight_position     <= '0;
            row_block           <= '0;
            row_index           <= '0;
            row_blocks          <= '0;
            weight_scale        <= '0;
            block_dot           <= '0;
            for (state_lane = 0; state_lane < 16; state_lane = state_lane + 1) begin
                q4_product[state_lane]    <= '0;
                q4_value_pipe[state_lane] <= '0;
                q4_input_pipe[state_lane] <= '0;
            end
            for (state_lane = 0; state_lane < 4; state_lane = state_lane + 1)
                q4_sum_group_pipe[state_lane] <= '0;
            q4_product_sum      <= '0;
            row_sum             <= '0;
            scaled_term         <= '0;
            scale_mantissa_product <= '0;
            scale_mantissa_a    <= '0;
            scale_mantissa_b    <= '0;
            scale_shift         <= '0;
            scale_zero          <= 1'b0;
            scale_negative      <= 1'b0;
            scale_product       <= '0;
            best_score          <= 64'sh8000_0000_0000_0000;
            q6_row_result       <= '0;
            best_token          <= '0;
            q6_step             <= '0;
            v_bits              <= '0;
            v_half              <= 1'b0;
            v_valid             <= 1'b0;
            r_bits              <= '0;
            r_half              <= 1'b0;
            r_valid             <= 1'b0;
            p_half              <= 1'b0;
            p_valid             <= 1'b0;
            s0_valid            <= 1'b0;
            s1_valid            <= 1'b0;
            s2_valid            <= 1'b0;
            s3_valid            <= 1'b0;
            s4_valid            <= 1'b0;
            s5_valid            <= 1'b0;
            s6_valid            <= 1'b0;
            s7_valid            <= 1'b0;
            q1_argmax_valid     <= 1'b0;
            q1_argmax_row       <= '0;
            q1_argmax_score     <= '0;
            q1_argmax_compare_valid <= 1'b0;
            q1_argmax_compare_row <= '0;
            q1_argmax_compare_score <= '0;
            q1_argmax_compare_better <= 1'b0;
            q1_float_valid      <= 1'b0;
            q1_float_row        <= '0;
            q1_float_data       <= '0;
            q1_rows_inflight    <= '0;
            h_dot0              <= '0;
            h_dot1              <= '0;
            q1_row_accum        <= '0;
            q1_convert_value_valid <= 1'b0;
            q1_convert_value_row <= '0;
            q1_convert_value    <= '0;
            q1_convert_parts_valid <= 1'b0;
            q1_convert_parts_row <= '0;
            q1_convert_high_sign <= 1'b0;
            q1_convert_high_magnitude <= '0;
            q1_convert_high_msb <= '0;
            q1_convert_low_magnitude <= '0;
            q1_convert_low_msb <= '0;
            q1_convert_trunc_valid <= 1'b0;
            q1_convert_trunc_row <= '0;
            q1_convert_trunc_high_sign <= 1'b0;
            q1_convert_trunc_high_base <= '0;
            q1_convert_trunc_high_increment <= '0;
            q1_convert_trunc_low_base <= '0;
            q1_convert_trunc_low_increment <= '0;
            q1_convert_round_valid <= 1'b0;
            q1_convert_round_row <= '0;
            q1_convert_rounded_high <= '0;
            q1_convert_rounded_low <= '0;
            q1_convert_split_valid <= 1'b0;
            q1_convert_split_row <= '0;
            q1_convert_split_value <= '0;
            q1_convert_mag_valid <= 1'b0;
            q1_convert_mag_row  <= '0;
            q1_convert_sign     <= 1'b0;
            q1_convert_magnitude <= '0;
            q1_convert_msb_valid <= 1'b0;
            q1_convert_msb_row  <= '0;
            q1_convert_msb_sign <= 1'b0;
            q1_convert_msb_magnitude <= '0;
            q1_convert_msb      <= '0;
            q1_float_prep_valid <= 1'b0;
            q1_float_prep_row <= '0;
            q1_float_prep_sign <= 1'b0;
            q1_float_prep_zero <= 1'b0;
            q1_float_prep_exponent <= '0;
            q1_float_prep_significand <= '0;
            q1_float_prep_round_up <= 1'b0;
            pend_last           <= 1'b0;
            pend_row            <= '0;
            wfifo_count         <= '0;
            wfifo_rd            <= '0;
            wfifo_wr            <= '0;
            wr_active           <= 1'b0;
            wr_aw_done          <= 1'b0;
            wr_w_done           <= 1'b0;
            wr_outstanding      <= '0;
            eng_addr            <= '0;
            eng_data            <= '0;
            ffn_start           <= 1'b0;
            residual_start      <= 1'b0;
            attention_start     <= 1'b0;
            ffn_group           <= '0;
            ffn_group_blocks    <= '0;
            ffn_blocks_accepted <= '0;
            ffn_group_drain     <= 1'b0;
            ffn_group_word      <= '0;
            ffn_group_address   <= '0;
            ffn_reader_finished <= 1'b0;
            ffn_compute_finished <= 1'b0;
            residual_result_word <= '0;
            residual_sum_enqueued <= 1'b0;
            residual_sum_pending <= 1'b0;
            residual_command_tag <= 1'b0;
            residual_fence_expected <= '0;
            residual_fence_match <= 1'b0;
            residual_sum_high_select <= 1'b0;
            residual_done_delayed <= 1'b0;
            residual_idle_seen <= 1'b0;
            residual_busy_seen <= 1'b0;
            residual_done_armed <= 1'b0;
            for (state_lane = 0; state_lane < 8; state_lane = state_lane + 1) begin
                q6_value_pipe[state_lane]     <= '0;
                q6_input_pipe[state_lane]     <= '0;
                q6_scale_decode[state_lane]   <= '0;
                q6_pair_product[state_lane]   <= '0;
                q6_scale_pipe[state_lane]     <= '0;
                q6_scaled_product[state_lane] <= '0;
            end
            done_toggle         <= 1'b0;
            error_flag          <= 1'b0;
            cycle_count         <= '0;
            watchdog_fired      <= 1'b0;
            watchdog_state      <= IDLE;
            watchdog_diagnostic <= '0;
            write_aw_sent       <= 1'b0;
            write_w_sent        <= 1'b0;
            write_address       <= '0;
            write_data          <= '0;
        end else begin
            residual_done_delayed <= residual_done;
            if (residual_sum_valid &&
                residual_sum_tag == residual_command_tag)
                residual_sum_pending <= 1'b1;
            if (residual_sum_push)
                residual_sum_pending <= 1'b0;
            if (state == RESIDUAL_FENCE_WAIT && reader_byte_valid &&
                reader_byte_ready &&
                reader_data[31:0] == residual_fence_expected)
                residual_fence_match <= 1'b1;
            // A delayed BUSY/DONE tail from the preceding FP-domain operation
            // can overlap the next command.  Establish a fresh idle sample
            // first, then accept only a subsequent BUSY rise and fall.  This
            // binds the scalar sum and completion to the current command.
            if (residual_mode && !residual_busy && !residual_busy_seen)
                residual_idle_seen <= 1'b1;
            if (residual_mode && residual_idle_seen && residual_busy)
                residual_busy_seen <= 1'b1;
            if (residual_mode && residual_busy_seen && !residual_busy &&
                !residual_done)
                residual_done_armed <= 1'b1;
            reader_start <= 1'b0;
            ffn_start <= 1'b0;
            residual_start <= 1'b0;
            attention_start <= 1'b0;
            input_scale_write_enable <= '0;
            if (bus_request_o)
                cycle_count <= cycle_count + 1'b1;
            if (reader_error)
                error_flag <= 1'b1;

            case ({q1_row_launch, q1_float_valid && !q1_argmax_mode})
                2'b10: q1_rows_inflight <= q1_rows_inflight + 1'b1;
                2'b01: q1_rows_inflight <= q1_rows_inflight - 1'b1;
                default: q1_rows_inflight <= q1_rows_inflight;
            endcase

            if (command_pending && command_sync[30:28] == 3'b010) begin
                case (command_sync[27:25])
                    3'd0: weight_offset_words <= command_sync[24:0];
                    3'd1: input_offset_words  <= command_sync[24:0];
                    3'd2: output_offset_words <= command_sync[24:0];
                    3'd3: ne0                 <= command_sync[15:0];
                    3'd4: ne1                 <= command_sync[17:0] == 0 ? 18'd65536 : command_sync[17:0];
                    3'd5: begin
                        if (state != IDLE) begin
                            error_flag <= 1'b1;
                        end else if (command_sync[2:0] != 3'd0) begin
                            q6_mode       <= command_sync[2:0] == 3'd2;
                            q1_mode       <= command_sync[2:0] == 3'd3 ||
                                             command_sync[2:0] == 3'd4 ||
                                             command_sync[2:0] == 3'd5 ||
                                             command_sync[2:0] == 3'd6;
                            q1_argmax_mode <= command_sync[2:0] == 3'd4 ||
                                              command_sync[2:0] == 3'd6;
                            q1_packed_mode <= command_sync[2:0] == 3'd5 ||
                                             command_sync[2:0] == 3'd6;
                            q1_compact_output <= output_compact_config &&
                                (command_sync[2:0] == 3'd3 ||
                                 command_sync[2:0] == 3'd5);
                            compact_hold_valid <= 1'b0;
                            ffn_mode       <= command_sync[2:0] == 3'd7 &&
                                              ne1 != 18'd4;
                            attention_mode <= command_sync[2:0] == 3'd7 &&
                                              ne1 == 18'd4;
                            residual_mode  <= command_sync[2:0] == 3'd7 &&
                                              ne1 == 18'd5;
                            error_flag    <= 1'b0;
                            best_score    <= 64'sh8000_0000_0000_0000;
                            best_token    <= '0;
                            cycle_count   <= '0;
                            watchdog_fired <= 1'b0;
                            watchdog_state <= IDLE;
                            watchdog_diagnostic <= '0;
                            bus_request_o <= 1'b1;
                            state         <= WAIT_GRANT;
                        end
                    end
                    3'd6: begin
                        input_address_high <= command_sync[1:0];
                        input_address_absolute <= command_sync[2];
                        output_address_high <= command_sync[4:3];
                        output_address_absolute <= command_sync[5];
                        output_compact_config <= command_sync[6];
                        scalar_high <= command_sync[13:7];
                        residual_sum_high_select <= command_sync[24];
                    end
                    3'd7: scalar_low <= command_sync[24:0];
                    default: error_flag <= error_flag;
                endcase
            end

            // Q1 free-running compute pipeline. Runs regardless of FSM state;
            // Q1_STREAM injects one 64-weight half per accepted bits beat and
            // LAUNCH_WEIGHTS resets the valid chain (textually later, so its
            // assignments win over these defaults).
            v_valid <= 1'b0;
            for (state_lane = 0; state_lane < 4; state_lane = state_lane + 1) begin
                r_data[state_lane]     <= wide_read_a[state_lane];
                r_data[state_lane + 4] <= wide_read_b[state_lane];
            end
            r_bits   <= v_bits;
            r_half   <= v_half;
            r_valid  <= v_valid;
            r_wscale <= weight_scale;
            for (state_lane = 0; state_lane < 4; state_lane = state_lane + 1)
                r_iscale[state_lane] <= pend_input_scale[state_lane];
            r_last <= pend_last;
            r_row  <= pend_row;

            for (state_lane = 0; state_lane < 8; state_lane = state_lane + 1) begin
                p_partial[0][state_lane] <= q1w_partial_c[0][state_lane];
                p_partial[1][state_lane] <= q1w_partial_c[1][state_lane];
            end
            p_half   <= r_half;
            p_valid  <= r_valid;
            p_wscale <= r_wscale;
            for (state_lane = 0; state_lane < 4; state_lane = state_lane + 1)
                p_iscale[state_lane] <= r_iscale[state_lane];
            p_last <= r_last;
            p_row  <= r_row;

            if (p_valid) begin
                if (!p_half) begin
                    h_dot0 <= q1w_dot[0];
                    h_dot1 <= q1w_dot[1];
                end else begin
                    s0_dot[0]  <= h_dot0;
                    s0_dot[1]  <= h_dot1;
                    s0_dot[2]  <= q1w_dot[0];
                    s0_dot[3]  <= q1w_dot[1];
                    s0_wscale  <= p_wscale;
                    for (state_lane = 0; state_lane < 4; state_lane = state_lane + 1)
                        s0_iscale[state_lane] <= p_iscale[state_lane];
                    s0_last <= p_last;
                    s0_row  <= p_row;
                end
            end
            s0_valid <= p_valid && p_half;

            for (state_lane = 0; state_lane < 4; state_lane = state_lane + 1) begin
                s1_wmant[state_lane] <= s0_wmant;
                s1_imant[state_lane] <= s0_imant[state_lane];
                s1_neg[state_lane]   <= s0_wscale[15] ^ s0_iscale[state_lane][15];
                s1_shift[state_lane] <= s0_wpow + s0_ipow[state_lane] + 9'sd30;
                s1_zero[state_lane]  <=
                    (s0_wexp == 0 && s0_wscale[9:0] == 0) ||
                    (s0_iexp[state_lane] == 0 && s0_iscale[state_lane][9:0] == 0);
                s1_dot[state_lane]   <= s0_dot[state_lane];
            end
            s1_last  <= s0_last;
            s1_row   <= s0_row;
            s1_valid <= s0_valid;

            for (state_lane = 0; state_lane < 4; state_lane = state_lane + 1) begin
                s2_mprod[state_lane] <= s1_wmant[state_lane] * s1_imant[state_lane];
                s2_neg[state_lane]   <= s1_neg[state_lane];
                s2_shift[state_lane] <= s1_shift[state_lane];
                s2_zero[state_lane]  <= s1_zero[state_lane];
                s2_dot[state_lane]   <= s1_dot[state_lane];
            end
            s2_last  <= s1_last;
            s2_row   <= s1_row;
            s2_valid <= s1_valid;

            // Apply the product sign to the 16-bit dot instead of the 32-bit
            // aligned magnitude: (-d)*a == d*(-a) exactly, and the short
            // carry chain keeps the barrel shifter off the critical path.
            for (state_lane = 0; state_lane < 4; state_lane = state_lane + 1) begin
                s3_sprod[state_lane] <= $signed(s2_aligned[state_lane]);
                s3_dot[state_lane]   <= s2_neg[state_lane] ?
                    -s2_dot[state_lane] : s2_dot[state_lane];
            end
            s3_last  <= s2_last;
            s3_row   <= s2_row;
            s3_valid <= s2_valid;

            // A direct signed 16x32 multiply maps to two cascaded DSP48s and
            // left only 90 ps of negative margin at 200 MHz.  Split the
            // 32-bit operand into two signed halves, register both 16x16
            // products, then restore the unsigned interpretation of the low
            // half in the following stage.  This identity is exact:
            // B = signed(B[31:16])*2^16 + signed(B[15:0]) + B[15]*2^16.
            for (state_lane = 0; state_lane < 4; state_lane = state_lane + 1) begin
                s4_low_product[state_lane] <=
                    s3_dot[state_lane] * $signed(s3_sprod[state_lane][15:0]);
                s4_high_product[state_lane] <=
                    s3_dot[state_lane] * $signed(s3_sprod[state_lane][31:16]);
                s4_correction[state_lane] <= s3_sprod[state_lane][15] ?
                    s3_dot[state_lane] : 16'sd0;
            end
            s4_last  <= s3_last;
            s4_row   <= s3_row;
            s4_valid <= s3_valid;

            for (state_lane = 0; state_lane < 4; state_lane = state_lane + 1)
                s5_scaled[state_lane] <=
                    $signed({{32{s4_low_product[state_lane][31]}},
                              s4_low_product[state_lane]}) +
                    ($signed({{32{s4_high_product[state_lane][31]}},
                               s4_high_product[state_lane]}) <<< 16) +
                    ($signed({{48{s4_correction[state_lane][15]}},
                               s4_correction[state_lane]}) <<< 16);
            s5_last  <= s4_last;
            s5_row   <= s4_row;
            s5_valid <= s4_valid;

            s6_pair[0] <= s5_scaled[0] + s5_scaled[1];
            s6_pair[1] <= s5_scaled[2] + s5_scaled[3];
            s6_last  <= s5_last;
            s6_row   <= s5_row;
            s6_valid <= s5_valid;

            s7_term  <= s6_pair[0] + s6_pair[1];
            s7_last  <= s6_last;
            s7_row   <= s6_row;
            s7_valid <= s6_valid;

            // Register the final row sum before the argmax comparison. A
            // same-cycle 64-bit add feeding a 64-bit signed comparison made
            // the best-token enable a 17-level carry path with effectively
            // no physical margin at 200 MHz.
            q1_argmax_valid <= s7_valid && s7_last && q1_argmax_mode;
            if (s7_valid && s7_last && q1_argmax_mode) begin
                q1_argmax_row <= s7_row;
                q1_argmax_score <= q1_row_accum + s7_term;
            end
            // Register the comparison result separately from both candidate
            // capture and best-token commit.  This makes the valid/score/token
            // relationship explicit and keeps the signed compare away from
            // the externally visible completion update.
            q1_argmax_compare_valid <= q1_argmax_valid;
            if (q1_argmax_valid) begin
                q1_argmax_compare_row <= q1_argmax_row;
                q1_argmax_compare_score <= q1_argmax_score;
                q1_argmax_compare_better <=
                    $signed(q1_argmax_score) > $signed(best_score);
            end
            if (q1_argmax_compare_valid && q1_argmax_compare_better) begin
                best_score <= q1_argmax_compare_score;
                best_token <= q1_argmax_compare_row;
            end

            q1_convert_value_valid <=
                s7_valid && s7_last && !q1_argmax_mode;
            if (s7_valid && s7_last && !q1_argmax_mode) begin
                q1_convert_value_row <= s7_row;
                q1_convert_value <= q1_row_accum + s7_term;
            end

            q1_convert_parts_valid <= q1_convert_value_valid;
            q1_convert_parts_row <= q1_convert_value_row;
            q1_convert_high_sign <= q1_convert_value[63];
            q1_convert_high_magnitude <= q1_convert_value[63] ?
                (~q1_convert_value[63:32] + 32'd1) :
                 q1_convert_value[63:32];
            q1_convert_high_msb <= integer_msb32(q1_convert_value[63] ?
                (~q1_convert_value[63:32] + 32'd1) :
                 q1_convert_value[63:32]);
            q1_convert_low_magnitude <= q1_convert_value[31:0];
            q1_convert_low_msb <= integer_msb32(q1_convert_value[31:0]);

            q1_convert_trunc_valid <= q1_convert_parts_valid;
            q1_convert_trunc_row <= q1_convert_parts_row;
            q1_convert_trunc_high_sign <= q1_convert_high_sign;
            q1_convert_trunc_high_base <= fp32_integer_round_base(
                q1_convert_high_magnitude, q1_convert_high_msb);
            q1_convert_trunc_high_increment <= fp32_integer_round_increment(
                q1_convert_high_magnitude, q1_convert_high_msb);
            q1_convert_trunc_low_base <= fp32_integer_round_base(
                q1_convert_low_magnitude, q1_convert_low_msb);
            q1_convert_trunc_low_increment <= fp32_integer_round_increment(
                q1_convert_low_magnitude, q1_convert_low_msb);

            q1_convert_round_valid <= q1_convert_trunc_valid;
            q1_convert_round_row <= q1_convert_trunc_row;
            q1_convert_rounded_high <= q1_convert_trunc_high_sign ?
                -$signed({1'b0, q1_convert_trunc_high_base} +
                         {1'b0, q1_convert_trunc_high_increment}) :
                 $signed({1'b0, q1_convert_trunc_high_base} +
                         {1'b0, q1_convert_trunc_high_increment});
            q1_convert_rounded_low <=
                {1'b0, q1_convert_trunc_low_base} +
                {1'b0, q1_convert_trunc_low_increment};

            q1_convert_split_valid <= q1_convert_round_valid;
            q1_convert_split_row <= q1_convert_round_row;
            q1_convert_split_value <=
                ($signed({{32{q1_convert_rounded_high[32]}},
                           q1_convert_rounded_high}) <<< 32) +
                $signed({32'b0, q1_convert_rounded_low});

            q1_convert_mag_valid <= q1_convert_split_valid;
            q1_convert_mag_row <= q1_convert_split_row;
            q1_convert_sign <= q1_convert_split_value[64];
            q1_convert_magnitude <= q1_convert_split_value[64] ?
                (~$unsigned(q1_convert_split_value) + 65'd1) :
                 $unsigned(q1_convert_split_value);

            q1_convert_msb_valid <= q1_convert_mag_valid;
            q1_convert_msb_row <= q1_convert_mag_row;
            q1_convert_msb_sign <= q1_convert_sign;
            q1_convert_msb_magnitude <= q1_convert_magnitude;
            q1_convert_msb <= '0;
            for (state_lane = 0; state_lane < 65;
                 state_lane = state_lane + 1)
                if (q1_convert_magnitude[state_lane])
                    q1_convert_msb <= state_lane[6:0];

            q1_float_prep_valid <= q1_convert_msb_valid;
            q1_float_prep_row <= q1_convert_msb_row;
            q1_float_prep_sign <= q1_convert_msb_sign;
            q1_float_prep_zero <= q1_convert_msb_magnitude == 0;
            q1_float_prep_exponent <= q1_convert_msb + 9'd97;
            q1_float_prep_significand <= q30_float_significand_base(
                q1_convert_msb_magnitude, q1_convert_msb);
            q1_float_prep_round_up <= q30_float_round_up(
                q1_convert_msb_magnitude, q1_convert_msb);

            q1_float_valid <= q1_float_prep_valid;
            q1_float_row <= q1_float_prep_row;
            q1_float_data <= q30_float_pack(
                q1_float_prep_sign, q1_float_prep_zero,
                q1_float_prep_exponent, q1_float_prep_significand,
                q1_float_prep_round_up);

            if (s7_valid) begin
                if (s7_last) begin
                    q1_row_accum <= '0;
                end else begin
                    q1_row_accum <= q1_row_accum + s7_term;
                end
            end

            if (q1_float_valid) begin
                if (q1_compact_output) begin
                    if (!compact_hold_valid) begin
                        compact_hold_valid <= 1'b1;
                        compact_hold_data <= q1_float_data;
                        compact_hold_row <= q1_float_row;
                    end else begin
                        compact_hold_valid <= 1'b0;
                        wfifo_addr[wfifo_wr] <= output_base_address +
                            ({12'b0, compact_hold_row} << 2);
                        wfifo_data[wfifo_wr] <=
                            {q1_float_data, compact_hold_data};
                        wfifo_wr <= wfifo_wr + 1'b1;
                    end
                end else begin
                    wfifo_addr[wfifo_wr] <= output_base_address +
                        ({12'b0, q1_float_row} << 3);
                    wfifo_data[wfifo_wr] <= {32'b0, q1_float_data};
                    wfifo_wr <= wfifo_wr + 1'b1;
                end
            end else if (residual_fifo_push) begin
                wfifo_addr[wfifo_wr] <=
                    residual_result_word == (ne0 >> 1) ?
                    RESIDUAL_SUM_ADDRESS : output_base_address +
                    ({19'b0, residual_result_word} << 3);
                wfifo_data[wfifo_wr] <= residual_result_data;
                wfifo_wr <= wfifo_wr + 1'b1;
                residual_result_word <= residual_result_word + 1'b1;
                if (residual_result_word == (ne0 >> 1))
                    residual_fence_expected <= residual_result_data[31:0];
            end else if (residual_sum_push) begin
                wfifo_addr[wfifo_wr] <= RESIDUAL_SUM_ADDRESS;
                wfifo_data[wfifo_wr] <= {32'b0, residual_sum};
                wfifo_wr <= wfifo_wr + 1'b1;
                residual_sum_enqueued <= 1'b1;
            end else if (ffn_fifo_push) begin
                wfifo_addr[wfifo_wr] <= ffn_group_address +
                    ({25'b0, ffn_group_word} << 3);
                wfifo_data[wfifo_wr] <= ffn_group[63:0];
                wfifo_wr <= wfifo_wr + 1'b1;
            end

            if (ffn_block_valid && ffn_block_ready) begin
                ffn_group[ffn_group_blocks * 272 +: 272] <= ffn_block_data;
                ffn_blocks_accepted <= ffn_blocks_accepted + 1'b1;
                if (ffn_group_blocks == 3) begin
                    ffn_group_blocks <= 0;
                    ffn_group_word <= 0;
                    ffn_group_drain <= 1'b1;
                end else begin
                    ffn_group_blocks <= ffn_group_blocks + 1'b1;
                end
            end
            if (ffn_fifo_push) begin
                ffn_group <= ffn_group >> 64;
                if (ffn_group_word == 16) begin
                    ffn_group_word <= 0;
                    ffn_group_drain <= 1'b0;
                    ffn_group_address <= ffn_group_address + 30'd136;
                end else begin
                    ffn_group_word <= ffn_group_word + 1'b1;
                end
            end

            if (wfifo_push && !wfifo_pop)
                wfifo_count <= wfifo_count + 1'b1;
            else if (!wfifo_push && wfifo_pop)
                wfifo_count <= wfifo_count - 1'b1;

            // A full FIFO indicates that Q1 reader backpressure did not
            // reserve enough room for results already in the compute pipe.
            if (q1_float_valid && wfifo_count == 4'd8 && !wfifo_pop)
                error_flag <= 1'b1;

            if (wfifo_pop) begin
                wr_active  <= 1'b1;
                wr_aw_done <= 1'b0;
                wr_w_done  <= 1'b0;
                eng_addr   <= wfifo_addr[wfifo_rd];
                eng_data   <= wfifo_data[wfifo_rd];
                wfifo_rd   <= wfifo_rd + 1'b1;
            end
            if (wr_active) begin
                if (axi.aw_valid && axi.aw_ready)
                    wr_aw_done <= 1'b1;
                if (axi.w_valid && axi.w_ready)
                    wr_w_done <= 1'b1;
                if (wr_issue_done)
                    wr_active <= 1'b0;
            end
            wr_outstanding <= wr_outstanding +
                (wr_issue_done ? 19'd1 : 19'd0) -
                ((engine_write_mode && axi.b_valid) ? 19'd1 : 19'd0);
            if (engine_write_mode && axi.b_valid &&
                (axi.b_resp != 2'b00 || axi.b_id != {ID_WIDTH{1'b1}}))
                error_flag <= 1'b1;

            case (state)
                IDLE: begin
                end

                WAIT_GRANT: begin
                    if (bus_grant_i) begin
                        if ((!attention_mode &&
                             (ne0 == 0 || ne1 == 0 || ne0 > MAX_INPUT)) ||
                            (!q6_mode && !q1_mode && !ffn_mode &&
                             !attention_mode &&
                             (ne0[4:0] != 0 || ((ne0 >> 5) & 3) != 0)) ||
                            (q6_mode &&
                             (ne0[7:0] != 0 || ((ne0 >> 8) & 3) != 0)) ||
                            (q1_mode && !q1_packed_mode && ne0[6:0] != 0) ||
                            (q1_packed_mode && ne0[8:0] != 0) ||
                            (ffn_mode && ne0[6:0] != 0)) begin
                            error_flag <= 1'b1;
                            state <= FINISH;
                        end else begin
                            row_blocks <= q6_mode ? (ne0 >> 8) :
                                          q1_mode ? (ne0 >> 7) :
                                                    (ne0 >> 5);
                            if (attention_mode) state <= ATTN_START;
                            else state <= ffn_mode ? FFN_START : START_INPUT;
                        end
                    end
                end

                FFN_START: begin
                    reader_address <= input_address_absolute ?
                        {input_address_high, input_offset_words, 3'b000} :
                        MODEL_BASE + {input_offset_words, 3'b000};
                    reader_byte_count <= ne1 == 18'd3 ?
                        {12'b0, ne0, 2'b00} :
                        (ne1 == 18'd2 || ne1 == 18'd5 ||
                         output_compact_config ?
                         {11'b0, ne0, 3'b000} :
                         {10'b0, ne0, 4'b0000});
                    reader_start <= 1'b1;
                    ffn_start <= !residual_mode;
                    residual_start <= residual_mode;
                    if (residual_mode)
                        residual_command_tag <= residual_command_tag + 1'b1;
                    ffn_group <= '0;
                    ffn_group_blocks <= 0;
                    ffn_blocks_accepted <= 0;
                    ffn_group_drain <= 1'b0;
                    ffn_group_word <= 0;
                    ffn_group_address <= output_base_address;
                    ffn_reader_finished <= 1'b0;
                    ffn_compute_finished <= 1'b0;
                    residual_result_word <= '0;
                    residual_sum_enqueued <= 1'b0;
                    residual_sum_pending <= 1'b0;
                    residual_done_delayed <= 1'b0;
                    residual_idle_seen <= 1'b0;
                    residual_busy_seen <= 1'b0;
                    residual_done_armed <= 1'b0;
                    wfifo_count <= '0;
                    wfifo_rd <= '0;
                    wfifo_wr <= '0;
                    wr_active <= 1'b0;
                    wr_outstanding <= '0;
                    state <= FFN_STREAM;
                end

                FFN_STREAM: begin
                    if (reader_done)
                        ffn_reader_finished <= 1'b1;
                    if (ffn_done)
                        ffn_compute_finished <= 1'b1;
                    if (residual_mode) begin
                        if ((ffn_reader_finished || reader_done) &&
                            (ffn_compute_finished || ffn_done) &&
                            residual_result_word == (ne0 >> 1) + 1'b1)
                            state <= Q1_FLUSH;
                    end else if ((ffn_reader_finished || reader_done) &&
                                 (ffn_compute_finished || ffn_done) &&
                                 ffn_blocks_accepted == (ne0 >> 5) &&
                                 !ffn_group_drain) begin
                        state <= Q1_FLUSH;
                    end
                end

                ATTN_START: begin
                    attention_start <= 1'b1;
                    state <= ATTN_WAIT;
                end

                ATTN_WAIT: begin
                    if (attention_error) begin
                        error_flag <= 1'b1;
                        state <= FINISH;
                    end else if (attention_done) begin
                        state <= FINISH;
                    end
                end

                START_INPUT: begin
                    reader_address <= input_address_absolute ?
                        {input_address_high, input_offset_words, 3'b000} :
                        MODEL_BASE + {input_offset_words, 3'b000};
                    // Compact Q6_K blocks contain 256 values plus a 2-byte
                    // scale; compact Q4_0 blocks contain 32 values plus a
                    // 2-byte scale. Express the fixed products as shifts and
                    // adds so command setup does not infer a DSP cascade.
                    reader_byte_count <= q6_mode ?
                        ({16'b0, ne0} + ({16'b0, ne0} >> 7)) :
                        ({16'b0, ne0} + ({16'b0, ne0} >> 4));
                    reader_start     <= 1'b1;
                    input_byte_index <= '0;
                    input_block      <= '0;
                    input_position   <= '0;
                    state            <= LOAD_INPUT;
                end

                LOAD_INPUT: begin
                    if (reader_byte_valid) begin
                        input_byte_index <= input_byte_index + 1'b1;
                        if (input_position == 0) begin
                            input_scale_low <= reader_data[7:0];
                        end else if (input_position == 1) begin
                            input_scale_write_enable[input_block[1:0]] <= 1'b1;
                            input_scale_write_address <= input_block >> 2;
                            input_scale_write_data <=
                                {reader_data[7:0], input_scale_low};
                        end

                        if ((!q6_mode && input_position == 33) ||
                            (q6_mode && input_position == 257)) begin
                            input_position <= '0;
                            input_block <= input_block + 1'b1;
                        end else begin
                            input_position <= input_position + 1'b1;
                        end
                    end
                    if (reader_done) begin
                        if (reader_error) begin
                            state <= FINISH;
                        end else begin
                            state <= PREPARE_WEIGHTS;
                        end
                    end
                end

                PREPARE_WEIGHTS: begin
                    // This is the only variable multiply in the stream-size
                    // setup. Register it before applying the per-block byte
                    // count in the following cycle.
                    weight_block_count <= row_blocks * ne1;
                    state <= START_WEIGHTS;
                end

                START_WEIGHTS: begin
                    reader_address <= MODEL_BASE + {weight_offset_words, 3'b000};
                    reader_byte_count <= q6_mode ?
                        ((weight_block_count << 7) +
                         (weight_block_count << 6) +
                         (weight_block_count << 4) +
                         (weight_block_count << 1)) :
                        ((weight_block_count << 4) +
                         (weight_block_count << 1));
                    state <= LAUNCH_WEIGHTS;
                end

                LAUNCH_WEIGHTS: begin
                    reader_start      <= 1'b1;
                    weight_byte_index <= '0;
                    weight_position   <= '0;
                    row_block         <= '0;
                    row_index         <= '0;
                    row_sum           <= '0;
                    block_dot         <= '0;
                    v_valid           <= 1'b0;
                    r_valid           <= 1'b0;
                    p_valid           <= 1'b0;
                    s0_valid          <= 1'b0;
                    s1_valid          <= 1'b0;
                    s2_valid          <= 1'b0;
                    s3_valid          <= 1'b0;
                    s4_valid          <= 1'b0;
                    s5_valid          <= 1'b0;
                    s6_valid          <= 1'b0;
                    s7_valid          <= 1'b0;
                    q1_argmax_valid   <= 1'b0;
                    q1_argmax_compare_valid <= 1'b0;
                    q1_convert_value_valid <= 1'b0;
                    q1_convert_parts_valid <= 1'b0;
                    q1_convert_trunc_valid <= 1'b0;
                    q1_convert_round_valid <= 1'b0;
                    q1_convert_split_valid <= 1'b0;
                    q1_convert_mag_valid <= 1'b0;
                    q1_convert_msb_valid <= 1'b0;
                    q1_float_prep_valid <= 1'b0;
                    q1_float_valid    <= 1'b0;
                    q1_rows_inflight  <= '0;
                    q1_row_accum      <= '0;
                    wfifo_count       <= '0;
                    wfifo_rd          <= '0;
                    wfifo_wr          <= '0;
                    wr_active         <= 1'b0;
                    wr_outstanding    <= '0;
                    state             <= q6_mode ? Q6_STREAM :
                                         q1_mode ? Q1_STREAM : Q4_STREAM;
                end

                Q4_STREAM: begin
                    if (reader_byte_valid) begin
                        if (weight_position == 0) begin
                            weight_byte_index <= weight_byte_index + 2;
                            weight_scale <= reader_data[15:0];
                            block_dot <= '0;
                            for (state_lane = 0; state_lane < 16; state_lane = state_lane + 1) begin
                                q4_value_pipe[state_lane] <= '0;
                                q4_input_pipe[state_lane] <= '0;
                                q4_product[state_lane] <= '0;
                            end
                            for (state_lane = 0; state_lane < 4; state_lane = state_lane + 1)
                                q4_sum_group_pipe[state_lane] <= '0;
                            q4_product_sum <= '0;
                            scale_mantissa_a <= scale_ma;
                            scale_mantissa_b <= scale_mb;
                            scale_negative <=
                                active_weight_scale[15] ^ active_input_scale[15];
                            scale_shift <= scale_pa + scale_pb + 9'sd30;
                            scale_zero <=
                                (scale_ea == 0 && active_weight_scale[9:0] == 0) ||
                                (scale_eb == 0 && active_input_scale[9:0] == 0);
                            weight_position <= 1;
                        end else begin
                            weight_byte_index <= weight_byte_index + 8;
                            for (state_lane = 0; state_lane < 16; state_lane = state_lane + 1) begin
                                q4_value_pipe[state_lane] <= q4_value[state_lane];
                                q4_product[state_lane] <= q4_value_pipe[state_lane] *
                                                          q4_input_pipe[state_lane];
                            end
                            for (state_lane = 0; state_lane < 4; state_lane = state_lane + 1)
                                q4_sum_group_pipe[state_lane] <= q4_sum_group[state_lane];
                            for (state_lane = 0; state_lane < 8; state_lane = state_lane + 1) begin
                                q4_input_pipe[state_lane*2] <= input_read_a[state_lane];
                                q4_input_pipe[state_lane*2 + 1] <= input_read_b[state_lane];
                            end
                            if (weight_position == 1)
                                scale_mantissa_product <=
                                    scale_mantissa_a * scale_mantissa_b;
                            if (weight_position == 2)
                                scale_product <= scale_negative ?
                                    -$signed(aligned_scale_product) :
                                     $signed(aligned_scale_product);
                            if (weight_position == 2) begin
                                weight_position <= '0;
                                state <= Q4_DRAIN_PRODUCT;
                            end else begin
                                weight_position <= weight_position + 1'b1;
                            end
                        end
                    end
                end

                Q4_DRAIN_PRODUCT: begin
                    for (state_lane = 0; state_lane < 16; state_lane = state_lane + 1)
                        q4_product[state_lane] <= q4_value_pipe[state_lane] *
                                                  q4_input_pipe[state_lane];
                    for (state_lane = 0; state_lane < 4; state_lane = state_lane + 1)
                        q4_sum_group_pipe[state_lane] <= q4_sum_group[state_lane];
                    q4_product_sum <= q4_product_reduction;
                    state <= Q4_DRAIN;
                end

                Q4_DRAIN: begin
                    for (state_lane = 0; state_lane < 4; state_lane = state_lane + 1)
                        q4_sum_group_pipe[state_lane] <= q4_sum_group[state_lane];
                    q4_product_sum <= q4_product_reduction;
                    state <= Q4_DRAIN_SUM;
                end

                Q4_DRAIN_SUM: begin
                    q4_product_sum <= q4_product_reduction;
                    block_dot <= block_dot +
                        {{10{q4_product_sum[21]}}, q4_product_sum};
                    state <= Q4_SCALE;
                end

                Q4_SCALE: begin
                    block_dot <= block_dot +
                        {{10{q4_product_sum[21]}}, q4_product_sum};
                    state <= Q4_SCALE_WAIT;
                end

                Q4_SCALE_WAIT: begin
                    state <= Q4_SCALE_MULTIPLY;
                end

                Q4_SCALE_MULTIPLY: begin
                    scaled_term <= wide_scaled_product[63:0];
                    state <= Q4_ACCUMULATE;
                end

                Q4_ACCUMULATE: begin
                    if (row_block + 1 == row_blocks) begin
                        write_data    <= row_sum + scaled_term;
                        write_address <= output_base_address + (row_index << 3);
                        write_aw_sent <= 1'b0;
                        write_w_sent  <= 1'b0;
                        state         <= Q4_WRITE;
                    end else begin
                        row_sum   <= row_sum + scaled_term;
                        row_block <= row_block + 1'b1;
                        state     <= Q4_STREAM;
                    end
                end

                Q4_WRITE: begin
                    if (axi.aw_valid && axi.aw_ready)
                        write_aw_sent <= 1'b1;
                    if (axi.w_valid && axi.w_ready)
                        write_w_sent <= 1'b1;
                    if ((write_aw_sent || (axi.aw_valid && axi.aw_ready)) &&
                        (write_w_sent || (axi.w_valid && axi.w_ready)))
                        state <= Q4_WRITE_RESPONSE;
                end

                Q4_WRITE_RESPONSE: begin
                    if (axi.b_valid && axi.b_ready) begin
                        if (axi.b_resp != 2'b00 || axi.b_id != {ID_WIDTH{1'b1}})
                            error_flag <= 1'b1;
                        if (row_index + 1 == ne1) begin
                            state <= FINISH;
                        end else begin
                            row_index <= row_index + 1'b1;
                            row_block <= '0;
                            row_sum   <= '0;
                            state     <= Q4_STREAM;
                        end
                    end
                end

                Q6_STREAM: begin
                    if (reader_byte_valid) begin
                        weight_byte_index <= weight_byte_index + reader_unit_bytes;
                        if (weight_position < 128) begin
                            for (state_lane = 0; state_lane < 8; state_lane = state_lane + 1)
                                q6_ql[weight_position + state_lane] <=
                                    reader_data[state_lane*8 +: 8];
                        end else if (weight_position < 192) begin
                            for (state_lane = 0; state_lane < 8; state_lane = state_lane + 1)
                                q6_qh[weight_position - 128 + state_lane] <=
                                    reader_data[state_lane*8 +: 8];
                        end else if (weight_position < 208) begin
                            for (state_lane = 0; state_lane < 8; state_lane = state_lane + 1)
                                q6_scales[weight_position - 192 + state_lane] <=
                                    reader_data[state_lane*8 +: 8];
                        end else begin
                            weight_scale <= reader_data[15:0];
                        end

                        if (weight_position == 208) begin
                            weight_position <= '0;
                            block_dot <= '0;
                            q6_step <= '0;
                            for (state_lane = 0; state_lane < 8; state_lane = state_lane + 1) begin
                                q6_value_pipe[state_lane] <= '0;
                                q6_input_pipe[state_lane] <= '0;
                                q6_scale_decode[state_lane] <= '0;
                                q6_pair_product[state_lane] <= '0;
                                q6_scale_pipe[state_lane] <= '0;
                                q6_scaled_product[state_lane] <= '0;
                            end
                            state <= Q6_COMPUTE;
                        end else begin
                            weight_position <= weight_position + 8;
                        end
                    end
                end

                Q6_COMPUTE: begin
                    // Register the variable Q6_K unpacking and BRAM outputs
                    // before multiplication. This removes the unpack muxes
                    // and offset arithmetic from the 200 MHz product path.
                    for (state_lane = 0; state_lane < 8; state_lane = state_lane + 1) begin
                        q6_value_pipe[state_lane] <= q6_value[state_lane];
                        q6_scale_decode[state_lane] <= q6_scale_value[state_lane];
                        q6_pair_product[state_lane] <= q6_value_pipe[state_lane] *
                                                       q6_input_pipe[state_lane];
                        q6_scale_pipe[state_lane] <= q6_scale_decode[state_lane];
                        if (q6_step >= 2)
                            q6_scaled_product[state_lane] <=
                                q6_pair_product[state_lane] * q6_scale_pipe[state_lane];
                    end
                    q6_input_pipe[0] <= input_read_a[0];
                    q6_input_pipe[1] <= input_read_b[0];
                    q6_input_pipe[2] <= input_read_a[1];
                    q6_input_pipe[3] <= input_read_b[1];
                    q6_input_pipe[4] <= input_read_a[2];
                    q6_input_pipe[5] <= input_read_b[2];
                    q6_input_pipe[6] <= input_read_a[3];
                    q6_input_pipe[7] <= input_read_b[3];
                    if (q6_step >= 3)
                        block_dot <= block_dot + q6_contribution;
                    if (q6_step == 31) begin
                        state <= Q6_DRAIN_PAIR;
                    end else begin
                        q6_step <= q6_step + 1'b1;
                    end
                end

                Q6_DRAIN_PAIR: begin
                    for (state_lane = 0; state_lane < 8; state_lane = state_lane + 1) begin
                        q6_pair_product[state_lane] <= q6_value_pipe[state_lane] *
                                                       q6_input_pipe[state_lane];
                        q6_scale_pipe[state_lane] <= q6_scale_decode[state_lane];
                        q6_scaled_product[state_lane] <=
                            q6_pair_product[state_lane] * q6_scale_pipe[state_lane];
                    end
                    block_dot <= block_dot + q6_contribution;
                    state <= Q6_DRAIN_SCALE;
                end

                Q6_DRAIN_SCALE: begin
                    for (state_lane = 0; state_lane < 8; state_lane = state_lane + 1)
                        q6_scaled_product[state_lane] <=
                            q6_pair_product[state_lane] * q6_scale_pipe[state_lane];
                    block_dot <= block_dot + q6_contribution;
                    state <= Q6_DRAIN_ACCUMULATE;
                end

                Q6_DRAIN_ACCUMULATE: begin
                    block_dot <= block_dot + q6_contribution;
                    state <= Q6_SCALE;
                end

                Q6_SCALE: begin
                    scale_mantissa_a <= scale_ma;
                    scale_mantissa_b <= scale_mb;
                    scale_negative <=
                        weight_scale[15] ^ active_input_scale[15];
                    scale_shift <= scale_pa + scale_pb + 9'sd30;
                    scale_zero <= (scale_ea == 0 && weight_scale[9:0] == 0) ||
                                  (scale_eb == 0 && active_input_scale[9:0] == 0);
                    state <= Q6_SCALE_MANTISSA;
                end

                Q6_SCALE_MANTISSA: begin
                    scale_mantissa_product <= scale_mantissa_a * scale_mantissa_b;
                    state <= Q6_SCALE_ALIGN_WAIT;
                end

                Q6_SCALE_ALIGN_WAIT: begin
                    state <= Q6_SCALE_ALIGN;
                end

                Q6_SCALE_ALIGN: begin
                    scale_product <= scale_negative ?
                        -$signed(aligned_scale_product) :
                         $signed(aligned_scale_product);
                    state <= Q6_SCALE_WAIT;
                end

                Q6_SCALE_WAIT: begin
                    state <= Q6_SCALE_MULTIPLY;
                end

                Q6_SCALE_MULTIPLY: begin
                    scaled_term <= wide_scaled_product[63:0];
                    state <= Q6_ACCUMULATE;
                end

                Q6_ACCUMULATE: begin
                    if (row_block + 1 == row_blocks) begin
                        q6_row_result <= row_sum + scaled_term;
                        state <= Q6_COMPARE;
                    end else begin
                        row_sum   <= row_sum + scaled_term;
                        row_block <= row_block + 1'b1;
                        state     <= Q6_STREAM;
                    end
                end

                Q6_COMPARE: begin
                    if ($signed(q6_row_result) > $signed(best_score)) begin
                        best_score <= q6_row_result;
                        best_token <= {2'b0, row_index[15:0]};
                    end
                    begin
                        if (row_index + 1 == ne1) begin
                            state <= FINISH;
                        end else begin
                            row_index <= row_index + 1'b1;
                            row_block <= '0;
                            row_sum   <= '0;
                            state     <= Q6_STREAM;
                        end
                    end
                end

                Q1_STREAM: begin
                    if (reader_byte_valid && reader_byte_ready) begin
                        weight_byte_index <= weight_byte_index + reader_unit_bytes;
                        if (q1_packed_mode && weight_position == 0) begin
                            for (state_lane = 0; state_lane < 4;
                                 state_lane = state_lane + 1)
                                packed_weight_scale[state_lane] <=
                                    reader_data[state_lane * 16 +: 16];
                            weight_position <= 8;
                        end else if (q1_packed_mode) begin
                            v_bits <= reader_data;
                            v_half <= weight_position[3:0] == 0;
                            v_valid <= 1'b1;
                            weight_scale <= packed_weight_scale[row_block[1:0]];
                            pend_input_scale[0] <= q1_input_scale_read[0];
                            pend_input_scale[1] <= q1_input_scale_read[1];
                            pend_input_scale[2] <= q1_input_scale_read[2];
                            pend_input_scale[3] <= q1_input_scale_read[3];
                            if (weight_position[3:0] == 0) begin
                                pend_last <= row_block + 1 == row_blocks;
                                pend_row <= row_index;
                                if (weight_position == 64) begin
                                    weight_position <= '0;
                                    if (row_block + 1 == row_blocks) begin
                                        row_block <= '0;
                                        if (row_index + 1 == ne1)
                                            state <= Q1_FLUSH;
                                        else
                                            row_index <= row_index + 1'b1;
                                    end else begin
                                        row_block <= row_block + 1'b1;
                                    end
                                end else begin
                                    weight_position <= weight_position + 8;
                                    row_block <= row_block + 1'b1;
                                end
                            end else begin
                                weight_position <= weight_position + 8;
                            end
                        end else if (weight_position == 0) begin
                            weight_scale <= reader_data[15:0];
                            weight_position <= 2;
                        end else if (weight_position == 2) begin
                            v_bits <= reader_data;
                            v_half <= 1'b0;
                            v_valid <= 1'b1;
                            pend_input_scale[0] <= q1_input_scale_read[0];
                            pend_input_scale[1] <= q1_input_scale_read[1];
                            pend_input_scale[2] <= q1_input_scale_read[2];
                            pend_input_scale[3] <= q1_input_scale_read[3];
                            weight_position <= 10;
                        end else begin
                            v_bits <= reader_data;
                            v_half <= 1'b1;
                            v_valid <= 1'b1;
                            pend_last <= row_block + 1 == row_blocks;
                            pend_row <= row_index;
                            weight_position <= '0;
                            if (row_block + 1 == row_blocks) begin
                                row_block <= '0;
                                if (row_index + 1 == ne1)
                                    state <= Q1_FLUSH;
                                else
                                    row_index <= row_index + 1'b1;
                            end else begin
                                row_block <= row_block + 1'b1;
                            end
                        end
                    end
                end

                Q1_FLUSH: begin
                    // Wait for the compute pipeline, the row-result FIFO, and
                    // every outstanding AXI write before signaling completion.
                    if (!q1_pipe_busy && !ffn_busy && !ffn_group_drain &&
                        wfifo_count == 4'd0 && !wr_active &&
                        wr_outstanding == 19'd0)
                        state <= residual_mode ? RESIDUAL_FENCE_START : FINISH;
                end

                RESIDUAL_FENCE_START: begin
                    // A DDR write response only guarantees that the
                    // controller accepted the scalar. On Genesys2 the CPU can
                    // regain the shared port and briefly read the old value
                    // while that write is still draining. Read the scalar
                    // back through this AXI master before releasing the bus;
                    // read-after-write ordering makes DONE a visibility fence.
                    reader_address <= RESIDUAL_SUM_ADDRESS;
                    reader_byte_count <= 8;
                    reader_start <= 1'b1;
                    residual_fence_match <= 1'b0;
                    state <= RESIDUAL_FENCE_WAIT;
                end

                RESIDUAL_FENCE_WAIT: begin
                    if (reader_done)
                        state <= residual_fence_match ?
                            FINISH : RESIDUAL_FENCE_START;
                end

                ABORT_DRAIN: begin
                    // A transform/reader bug must never retain the exclusive
                    // DDR grant forever. Drain accepted AXI traffic, then
                    // return an error completion with the timed-out state
                    // preserved in status_o for software diagnostics.
                    if (!reader_busy && wfifo_count == 4'd0 && !wr_active &&
                        wr_outstanding == 19'd0) begin
                        bus_request_o <= 1'b0;
                        state <= RELEASE_BUS;
                    end
                end

                FINISH: begin
                    // Do not expose completion until the streaming reader has
                    // retired its final AXI beat. This is normally already
                    // true after the arithmetic drain, but making it explicit
                    // prevents a subsequent job from reusing a live reader.
                    if (!reader_busy) begin
                        bus_request_o <= 1'b0;
                        state <= RELEASE_BUS;
                    end
                end

                RELEASE_BUS: begin
                    // The exclusive arbiter drops its registered grant one
                    // cycle after request. Keep DONE unchanged until the CPU
                    // once again owns DDR, so software cannot race the handoff.
                    if (!bus_grant_i)
                        state <= SIGNAL_DONE;
                end

                SIGNAL_DONE: begin
                    done_toggle <= ~done_toggle;
                    state <= IDLE;
                end

                default: begin
                    // Trap undefined states through the ordinary completion
                    // path so the bus is never released while the reader or
                    // writer still owns outstanding AXI transactions.
                    error_flag <= 1'b1;
                    state      <= Q1_FLUSH;
                end
            endcase


            // At 200 MHz this is approximately 2.68 seconds. Every command
            // in a genuine 2 tok/s decoder must complete well inside that
            // bound; exceeding it is a deadlock or already misses the target.
            if (bus_request_o && !watchdog_fired && cycle_count[29]) begin
                watchdog_fired <= 1'b1;
                watchdog_state <= state;
                watchdog_diagnostic <= {
                    state,
                    ffn_debug_state_sync,
                    reader_busy,
                    reader_done,
                    ffn_busy,
                    ffn_done,
                    ffn_group_drain,
                    ffn_block_valid,
                    wr_active,
                    |wr_outstanding
                };
                error_flag <= 1'b1;
                state <= ABORT_DRAIN;
            end
        end
    end

endmodule

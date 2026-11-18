// Add a streamed FP32 matvec result to a resident residual vector, retain the
// result, and compute its sum of squares. Six bit-exact integer FP32 ALUs form
// a three-stage pipeline: pair add, pair square, and 16-lane accumulation.
// Striped accumulators remove the serial dependency while retaining a fully
// deterministic, separately-rounded FP32 result.
module bonsai_residual #(
    parameter integer ELEMENTS = 2048
) (
    input  logic        clk_i,
    input  logic        fp_clk_i,
    input  logic        rst_ni,
    input  logic        fp_rst_ni,
    input  logic        start_i,
    input  logic [63:0] source_data_i,
    input  logic        source_valid_i,
    output logic        source_ready_o,
    output logic [63:0] result_data_o,
    output logic        result_valid_o,
    input  logic        result_ready_i,
    output logic [31:0] sum_o,
    output logic        sum_valid_o,
    output logic [7:0]  sum_tag_o,
    output logic        busy_o,
    output logic        done_o,
    output logic [3:0]  debug_state_o
);
    localparam integer WORDS = ELEMENTS / 2;
    localparam integer WORD_INDEX_WIDTH = $clog2(WORDS);

    typedef enum logic [3:0] {
        RESIDUAL_IDLE,
        RESIDUAL_LOAD_OUTPUT,
        RESIDUAL_LOAD_RESIDUAL,
        RESIDUAL_PREFETCH_0,
        RESIDUAL_PREFETCH_1,
        RESIDUAL_COMPUTE,
        RESIDUAL_REDUCE_GAP,
        RESIDUAL_REDUCE_LAUNCH,
        RESIDUAL_REDUCE_WAIT,
        RESIDUAL_REDUCE_CAPTURE,
        RESIDUAL_OUTPUT_READ,
        RESIDUAL_OUTPUT_WAIT,
        RESIDUAL_OUTPUT_EMIT,
        RESIDUAL_SUM_EMIT,
        RESIDUAL_DONE
    } residual_state_t;

    residual_state_t state;
    (* ram_style = "block" *) logic [63:0] vector_memory [0:WORDS-1];
    (* ram_style = "block" *) logic [63:0] residual_memory [0:WORDS-1];
    logic [WORD_INDEX_WIDTH-1:0] word_index;
    logic [WORD_INDEX_WIDTH-1:0] memory_read_address;
    logic [63:0] vector_read_data;
    logic [63:0] residual_read_data;
    logic vector_write_enable;
    logic [WORD_INDEX_WIDTH-1:0] vector_write_address;
    logic [63:0] vector_write_data;

    logic [31:0] lane_sum [0:15];
    logic [4:0] reduce_stride;
    logic [4:0] reduce_index;
    logic       reduce_second_valid;

    // All six operand banks remain stable for ten clocks before capture.
    // The extra clock provides full-SoC routing margin at 125 MHz; it costs
    // one clock per two streamed elements and does not reduce stream width.
    // The implementation constraint scopes these named registers to the
    // corresponding fp_result registers.
    localparam logic [3:0] FP_LATENCY_COUNT = 4'd9;
    // These registers are the architectural boundaries of the deliberate
    // ten-cycle FP path.  Full-SoC synthesis otherwise retimes the adder
    // backward through fp_a_hold and creates an impossible one-cycle path
    // from the vector BRAM into the FP carry chain.
    (* dont_touch = "yes" *) logic [31:0] fp_a_hold [0:5];
    (* dont_touch = "yes" *) logic [31:0] fp_b_hold [0:5];
    logic [31:0] fp_result_comb [0:5];
    (* dont_touch = "yes" *) logic [31:0] fp_result [0:5];
    logic [3:0]  fp_latency_count;
    logic        fp_capture_pending;

    logic add_active;
    logic square_active;
    logic accumulate_active;
    logic [WORD_INDEX_WIDTH-1:0] add_word;
    logic [WORD_INDEX_WIDTH-1:0] square_word;
    logic [WORD_INDEX_WIDTH-1:0] accumulate_word;
    logic [WORD_INDEX_WIDTH:0] launch_word;

    for (genvar lane = 0; lane < 6; ++lane) begin : gen_fp_alu
        bonsai_fp32_alu i_fp32 (
            .fused_i(1'b0),
            .multiply_i(lane == 2 || lane == 3),
            .a_i(fp_a_hold[lane]),
            .b_i(fp_b_hold[lane]),
            .c_i(32'b0),
            .result_o(fp_result_comb[lane])
        );
    end

    // Toggle-plus-stable-payload mailboxes between AXI and FP domains.
    logic [63:0] source_hold;
    logic source_request_toggle;
    logic source_busy;
    (* async_reg = "true" *) logic source_request_meta;
    (* async_reg = "true" *) logic source_request_sync;
    logic source_request_seen;
    (* async_reg = "true" *) logic source_ack_meta;
    (* async_reg = "true" *) logic source_ack_sync;
    logic start_toggle;
    logic [7:0] start_tag_hold;
    logic [7:0] active_tag;
    (* async_reg = "true" *) logic start_meta;
    (* async_reg = "true" *) logic start_sync;
    logic start_seen;
    logic [63:0] result_hold;
    logic result_toggle;
    logic result_pending;
    (* async_reg = "true" *) logic result_meta;
    (* async_reg = "true" *) logic result_sync;
    logic result_seen;
    logic result_ack_toggle;
    (* async_reg = "true" *) logic result_ack_meta;
    (* async_reg = "true" *) logic result_ack_sync;
    logic [31:0] sum_hold;
    logic [7:0] sum_tag_hold;
    logic sum_toggle;
    (* async_reg = "true" *) logic sum_meta;
    (* async_reg = "true" *) logic sum_sync;
    logic sum_seen;
    (* async_reg = "true" *) logic busy_meta;
    (* async_reg = "true" *) logic busy_sync;
    (* async_reg = "true" *) logic done_meta;
    (* async_reg = "true" *) logic done_sync;

    wire core_start = start_sync != start_seen;
    wire core_source_valid = source_request_sync != source_request_seen;
    wire [63:0] core_source_data = source_hold;
    wire core_source_ready = state == RESIDUAL_LOAD_OUTPUT ||
                             state == RESIDUAL_LOAD_RESIDUAL;
    wire core_result_valid = state == RESIDUAL_OUTPUT_EMIT ||
                             state == RESIDUAL_SUM_EMIT;
    wire core_result_ready = !result_pending;
    wire core_busy = state != RESIDUAL_IDLE && state != RESIDUAL_DONE;
    wire core_done = state == RESIDUAL_DONE && !result_pending;

    assign source_ready_o = !source_busy;
    assign busy_o = busy_sync;
    assign done_o = done_sync;
    assign debug_state_o = state;
    assign vector_write_enable =
        (state == RESIDUAL_LOAD_OUTPUT && core_source_valid) ||
        (state == RESIDUAL_COMPUTE && fp_capture_pending && add_active);
    assign vector_write_address = state == RESIDUAL_LOAD_OUTPUT ?
        word_index : add_word;
    assign vector_write_data = state == RESIDUAL_LOAD_OUTPUT ?
        core_source_data : {fp_result[1], fp_result[0]};

    always_ff @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
            source_hold <= 0;
            source_request_toggle <= 0;
            source_busy <= 0;
            source_ack_meta <= 0;
            source_ack_sync <= 0;
            start_toggle <= 0;
            start_tag_hold <= 0;
            result_meta <= 0;
            result_sync <= 0;
            result_seen <= 0;
            result_ack_toggle <= 0;
            result_data_o <= 0;
            result_valid_o <= 0;
            busy_meta <= 0;
            busy_sync <= 0;
            done_meta <= 0;
            done_sync <= 0;
            sum_meta <= 0;
            sum_sync <= 0;
            sum_seen <= 0;
            sum_o <= 0;
            sum_valid_o <= 0;
            sum_tag_o <= 0;
        end else begin
            sum_valid_o <= 0;
            source_ack_meta <= source_request_seen;
            source_ack_sync <= source_ack_meta;
            result_meta <= result_toggle;
            result_sync <= result_meta;
            busy_meta <= core_busy;
            busy_sync <= busy_meta;
            done_meta <= core_done;
            done_sync <= done_meta;
            sum_meta <= sum_toggle;
            sum_sync <= sum_meta;
            if (sum_sync != sum_seen) begin
                sum_o <= sum_hold;
                sum_tag_o <= sum_tag_hold;
                sum_seen <= sum_sync;
                sum_valid_o <= 1'b1;
            end
            if (start_i) begin
                start_toggle <= ~start_toggle;
                start_tag_hold <= start_tag_hold + 1'b1;
            end
            if (source_valid_i && source_ready_o) begin
                source_hold <= source_data_i;
                source_request_toggle <= ~source_request_toggle;
                source_busy <= 1'b1;
            end
            if (source_busy && source_ack_sync == source_request_toggle)
                source_busy <= 1'b0;
            if (!result_valid_o && result_sync != result_seen) begin
                result_data_o <= result_hold;
                result_valid_o <= 1'b1;
                result_seen <= result_sync;
            end
            if (result_valid_o && result_ready_i) begin
                result_valid_o <= 1'b0;
                result_ack_toggle <= result_seen;
            end
        end
    end

    always_ff @(posedge fp_clk_i or negedge fp_rst_ni) begin
        if (!fp_rst_ni) begin
            source_request_meta <= 0;
            source_request_sync <= 0;
            source_request_seen <= 0;
            start_meta <= 0;
            start_sync <= 0;
            start_seen <= 0;
            result_hold <= 0;
            result_toggle <= 0;
            result_pending <= 0;
            result_ack_meta <= 0;
            result_ack_sync <= 0;
        end else begin
            source_request_meta <= source_request_toggle;
            source_request_sync <= source_request_meta;
            start_meta <= start_toggle;
            start_sync <= start_meta;
            result_ack_meta <= result_ack_toggle;
            result_ack_sync <= result_ack_meta;
            if (core_start) start_seen <= start_sync;
            if (core_source_valid && core_source_ready)
                source_request_seen <= source_request_sync;
            if (result_pending && result_ack_sync == result_toggle)
                result_pending <= 1'b0;
            if (core_result_valid && core_result_ready) begin
                result_hold <= state == RESIDUAL_SUM_EMIT ?
                    {32'b0, sum_hold} : vector_read_data;
                result_toggle <= ~result_toggle;
                result_pending <= 1'b1;
            end
        end
    end

    // Both source vectors and the updated result infer true block RAMs.
    always_ff @(posedge fp_clk_i) begin
        vector_read_data <= vector_memory[memory_read_address];
        residual_read_data <= residual_memory[memory_read_address];
        if (vector_write_enable)
            vector_memory[vector_write_address] <= vector_write_data;
        if (state == RESIDUAL_LOAD_RESIDUAL && core_source_valid)
            residual_memory[word_index] <= core_source_data;
    end

    integer lane;
    always_ff @(posedge fp_clk_i or negedge fp_rst_ni) begin
        if (!fp_rst_ni) begin
            state <= RESIDUAL_IDLE;
            word_index <= 0;
            memory_read_address <= 0;
            active_tag <= 0;
            sum_hold <= 0;
            sum_tag_hold <= 0;
            sum_toggle <= 0;
            fp_latency_count <= 0;
            fp_capture_pending <= 0;
            add_active <= 0;
            square_active <= 0;
            accumulate_active <= 0;
            add_word <= 0;
            square_word <= 0;
            accumulate_word <= 0;
            launch_word <= 0;
            reduce_stride <= 0;
            reduce_index <= 0;
            reduce_second_valid <= 0;
            for (lane = 0; lane < 16; lane = lane + 1)
                lane_sum[lane] <= 0;
            for (lane = 0; lane < 6; lane = lane + 1) begin
                fp_a_hold[lane] <= 0;
                fp_b_hold[lane] <= 0;
                fp_result[lane] <= 0;
            end
        end else if (core_start) begin
            active_tag <= start_tag_hold;
            state <= RESIDUAL_LOAD_OUTPUT;
            word_index <= 0;
            memory_read_address <= 0;
            fp_latency_count <= 0;
            fp_capture_pending <= 0;
            add_active <= 0;
            square_active <= 0;
            accumulate_active <= 0;
            launch_word <= 0;
            for (lane = 0; lane < 16; lane = lane + 1)
                lane_sum[lane] <= 0;
        end else begin
            case (state)
                RESIDUAL_IDLE: state <= RESIDUAL_IDLE;
                RESIDUAL_LOAD_OUTPUT: if (core_source_valid) begin
                    if (word_index == WORDS - 1) begin
                        word_index <= 0;
                        state <= RESIDUAL_LOAD_RESIDUAL;
                    end else word_index <= word_index + 1'b1;
                end
                RESIDUAL_LOAD_RESIDUAL: if (core_source_valid) begin
                    if (word_index == WORDS - 1) begin
                        word_index <= 0;
                        memory_read_address <= 0;
                        state <= RESIDUAL_PREFETCH_0;
                    end else word_index <= word_index + 1'b1;
                end
                RESIDUAL_PREFETCH_0: state <= RESIDUAL_PREFETCH_1;
                RESIDUAL_PREFETCH_1: begin
                    fp_a_hold[0] <= vector_read_data[31:0];
                    fp_b_hold[0] <= residual_read_data[31:0];
                    fp_a_hold[1] <= vector_read_data[63:32];
                    fp_b_hold[1] <= residual_read_data[63:32];
                    add_active <= 1'b1;
                    add_word <= 0;
                    launch_word <= 1;
                    memory_read_address <= 1;
                    fp_latency_count <= FP_LATENCY_COUNT;
                    state <= RESIDUAL_COMPUTE;
                end
                RESIDUAL_COMPUTE: begin
                    if (!fp_capture_pending && fp_latency_count != 0)
                        fp_latency_count <= fp_latency_count - 1'b1;
                    else if (!fp_capture_pending) begin
                        for (lane = 0; lane < 6; lane = lane + 1)
                            fp_result[lane] <= fp_result_comb[lane];
                        fp_capture_pending <= 1'b1;
                    end else begin
                        fp_capture_pending <= 1'b0;
                        if (accumulate_active) begin
                            lane_sum[{accumulate_word[2:0], 1'b0}] <= fp_result[4];
                            lane_sum[{accumulate_word[2:0], 1'b1}] <= fp_result[5];
                        end
                        accumulate_active <= square_active;
                        accumulate_word <= square_word;
                        if (square_active) begin
                            fp_a_hold[4] <= lane_sum[{square_word[2:0], 1'b0}];
                            fp_b_hold[4] <= fp_result[2];
                            fp_a_hold[5] <= lane_sum[{square_word[2:0], 1'b1}];
                            fp_b_hold[5] <= fp_result[3];
                        end
                        square_active <= add_active;
                        square_word <= add_word;
                        if (add_active) begin
                            fp_a_hold[2] <= fp_result[0];
                            fp_b_hold[2] <= fp_result[0];
                            fp_a_hold[3] <= fp_result[1];
                            fp_b_hold[3] <= fp_result[1];
                        end
                        if (launch_word < WORDS) begin
                            fp_a_hold[0] <= vector_read_data[31:0];
                            fp_b_hold[0] <= residual_read_data[31:0];
                            fp_a_hold[1] <= vector_read_data[63:32];
                            fp_b_hold[1] <= residual_read_data[63:32];
                            add_active <= 1'b1;
                            add_word <= launch_word[WORD_INDEX_WIDTH-1:0];
                            launch_word <= launch_word + 1'b1;
                            if (launch_word + 1'b1 < WORDS)
                                memory_read_address <= launch_word + 1'b1;
                        end else add_active <= 1'b0;
                        if (!add_active && !square_active &&
                            launch_word >= WORDS) begin
                            accumulate_active <= 1'b0;
                            if (accumulate_active) begin
                                reduce_stride <= 8;
                                reduce_index <= 0;
                                state <= RESIDUAL_REDUCE_GAP;
                            end else begin
                                fp_latency_count <= FP_LATENCY_COUNT;
                            end
                        end else fp_latency_count <= FP_LATENCY_COUNT;
                    end
                end
                RESIDUAL_REDUCE_GAP: begin
                    state <= RESIDUAL_REDUCE_LAUNCH;
                end
                RESIDUAL_REDUCE_LAUNCH: begin
                    fp_a_hold[0] <= lane_sum[reduce_index];
                    fp_b_hold[0] <= lane_sum[reduce_index + reduce_stride];
                    reduce_second_valid <= reduce_index + 1'b1 < reduce_stride;
                    if (reduce_index + 1'b1 < reduce_stride) begin
                        fp_a_hold[1] <= lane_sum[reduce_index + 1'b1];
                        fp_b_hold[1] <= lane_sum[reduce_index + reduce_stride + 1'b1];
                    end
                    fp_latency_count <= FP_LATENCY_COUNT;
                    state <= RESIDUAL_REDUCE_WAIT;
                end
                RESIDUAL_REDUCE_WAIT: if (fp_latency_count != 0) begin
                    fp_latency_count <= fp_latency_count - 1'b1;
                end else begin
                    fp_result[0] <= fp_result_comb[0];
                    fp_result[1] <= fp_result_comb[1];
                    state <= RESIDUAL_REDUCE_CAPTURE;
                end
                RESIDUAL_REDUCE_CAPTURE: begin
                    lane_sum[reduce_index] <= fp_result[0];
                    if (reduce_second_valid)
                        lane_sum[reduce_index + 1'b1] <= fp_result[1];
                    if (reduce_stride == 1) begin
                        sum_hold <= fp_result[0];
                        sum_tag_hold <= active_tag;
                        sum_toggle <= ~sum_toggle;
                        word_index <= 0;
                        memory_read_address <= 0;
                        state <= RESIDUAL_OUTPUT_READ;
                    end else if (reduce_index + 2 >= reduce_stride) begin
                        reduce_stride <= reduce_stride >> 1;
                        reduce_index <= 0;
                        state <= RESIDUAL_REDUCE_GAP;
                    end else begin
                        reduce_index <= reduce_index + 2;
                        state <= RESIDUAL_REDUCE_LAUNCH;
                    end
                end
                RESIDUAL_OUTPUT_READ: begin
                    memory_read_address <= word_index;
                    state <= RESIDUAL_OUTPUT_WAIT;
                end
                RESIDUAL_OUTPUT_WAIT: state <= RESIDUAL_OUTPUT_EMIT;
                RESIDUAL_OUTPUT_EMIT: if (core_result_ready) begin
                    if (word_index == WORDS - 1)
                        state <= RESIDUAL_SUM_EMIT;
                    else begin
                        word_index <= word_index + 1'b1;
                        state <= RESIDUAL_OUTPUT_READ;
                    end
                end
                RESIDUAL_SUM_EMIT: if (core_result_ready)
                    state <= RESIDUAL_DONE;
                RESIDUAL_DONE: if (!result_pending)
                    state <= RESIDUAL_IDLE;
                default: state <= RESIDUAL_IDLE;
            endcase
        end
    end
endmodule

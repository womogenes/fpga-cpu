// Add a streamed FP32 matvec result to a resident residual vector, retain the
// updated vector in block RAM, and compute the two-half FP32 sum of squares
// used by the software RMS-normalization path.  Coarse 64-bit mailboxes cross
// between the 200 MHz AXI domain and the slower FPNew clock domain.
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
    localparam integer ELEMENT_INDEX_WIDTH = $clog2(ELEMENTS);

    typedef enum logic [3:0] {
        RESIDUAL_IDLE,
        RESIDUAL_LOAD_OUTPUT,
        RESIDUAL_LOAD_RESIDUAL,
        RESIDUAL_WAIT_VECTOR,
        RESIDUAL_ADD_LOW,
        RESIDUAL_ADD_HIGH,
        RESIDUAL_SUM_READ,
        RESIDUAL_SUM_WAIT,
        RESIDUAL_SQUARE,
        RESIDUAL_ACCUMULATE,
        RESIDUAL_FINAL_SUM,
        RESIDUAL_OUTPUT_READ,
        RESIDUAL_OUTPUT_WAIT,
        RESIDUAL_OUTPUT_EMIT,
        RESIDUAL_SUM_EMIT,
        RESIDUAL_DONE
    } residual_state_t;

    residual_state_t state;
    (* ram_style = "block" *) logic [63:0] vector_memory [0:WORDS-1];
    logic [WORD_INDEX_WIDTH-1:0] word_index;
    logic [ELEMENT_INDEX_WIDTH-1:0] sum_index;
    logic [WORD_INDEX_WIDTH-1:0] memory_read_address;
    logic [63:0] memory_read_data;
    logic [63:0] residual_pair;
    logic [63:0] result_pair;
    logic [31:0] square_value;
    logic [31:0] sum_first;
    logic [31:0] sum_second;
    logic [31:0] total_sum;

    logic [1:0] fp_operation;
    logic [31:0] fp_a;
    logic [31:0] fp_b;
    logic fp_input_valid;
    logic fp_input_ready;
    logic [31:0] fp_result;
    logic fp_result_valid;
    logic fp_waiting;
    logic [1:0] fp_operation_hold;
    logic [31:0] fp_a_hold;
    logic [31:0] fp_b_hold;
    logic [3:0] fp_latency_count;
    logic fp_active;

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

    always_comb begin
        fp_operation = 2'd0;
        fp_a = 0;
        fp_b = 0;
        if (state == RESIDUAL_ADD_LOW) begin
            fp_a = memory_read_data[31:0];
            fp_b = residual_pair[31:0];
        end else if (state == RESIDUAL_ADD_HIGH) begin
            fp_a = memory_read_data[63:32];
            fp_b = residual_pair[63:32];
        end else if (state == RESIDUAL_SQUARE) begin
            fp_operation = 2'd1;
            fp_a = sum_index[0] ? memory_read_data[63:32] :
                                  memory_read_data[31:0];
            fp_b = fp_a;
        end else if (state == RESIDUAL_ACCUMULATE) begin
            fp_a = sum_index < ELEMENTS / 2 ? sum_first : sum_second;
            fp_b = square_value;
        end else if (state == RESIDUAL_FINAL_SUM) begin
            fp_a = sum_first;
            fp_b = sum_second;
        end
    end

    assign fp_input_valid = (state == RESIDUAL_ADD_LOW ||
                             state == RESIDUAL_ADD_HIGH ||
                             state == RESIDUAL_SQUARE ||
                             state == RESIDUAL_ACCUMULATE ||
                             state == RESIDUAL_FINAL_SUM) && !fp_waiting;

    logic [31:0] fp_comb_result;

    bonsai_fp32_alu i_fp32 (
        .fused_i(1'b0),
        .multiply_i(fp_operation_hold == 2'd1),
        .a_i(fp_a_hold),
        .b_i(fp_b_hold),
        .c_i(32'b0),
        .result_o(fp_comb_result)
    );

    assign fp_input_ready = !fp_active;

    // The shared FPNew pipe is not repeatable on this CVA6/Genesys2 clock
    // crossing.  Hold operands stable while the bit-exact integer FP32 ALU
    // settles, then capture after nine clocks.  The matching multicycle
    // constraint is scoped from these holding registers to fp_result.
    always_ff @(posedge fp_clk_i or negedge fp_rst_ni) begin
        if (!fp_rst_ni) begin
            fp_result <= 0;
            fp_result_valid <= 0;
            fp_operation_hold <= 0;
            fp_a_hold <= 0;
            fp_b_hold <= 0;
            fp_latency_count <= 0;
            fp_active <= 0;
        end else begin
            fp_result_valid <= 0;
            if (fp_input_valid && fp_input_ready) begin
                fp_operation_hold <= fp_operation;
                fp_a_hold <= fp_a;
                fp_b_hold <= fp_b;
                // Capture on the ninth following edge.  The routed ALU takes
                // roughly 64 ns at the characterized corner; seven cycles at
                // 90 MHz proved intermittently marginal on the hot board.
                fp_latency_count <= 4'd8;
                fp_active <= 1;
            end else if (fp_active && fp_latency_count != 0) begin
                fp_latency_count <= fp_latency_count - 1'b1;
            end else if (fp_active) begin
                fp_result <= fp_comb_result;
                fp_result_valid <= 1;
                fp_active <= 0;
            end
        end
    end

    // AXI-domain halves of the source/result mailboxes.
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
            // Bundled-data mailbox: FINAL_SUM changes sum_toggle only after
            // writing sum_hold.  Two synchronizer clocks give the 32-bit
            // payload time to settle before this domain captures it, and the
            // toggle identifies the exact residual transaction independently
            // of BUSY/DONE pulse overlap between back-to-back commands.
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

    // FP-domain control synchronizers and output mailbox.
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
            if (core_start)
                start_seen <= start_sync;
            if (core_source_valid && core_source_ready)
                source_request_seen <= source_request_sync;
            if (result_pending && result_ack_sync == result_toggle)
                result_pending <= 1'b0;
            if (core_result_valid && core_result_ready) begin
                // Return the scalar through the same acknowledged mailbox as
                // the vector.  The former independent sum toggle could be
                // observed one transaction late on hardware; this mailbox
                // cannot advance until the AXI domain acknowledges each word.
                result_hold <= state == RESIDUAL_SUM_EMIT ?
                    {32'b0, total_sum} : memory_read_data;
                result_toggle <= ~result_toggle;
                result_pending <= 1'b1;
            end
        end
    end

    // True dual-port-style vector storage: the first stream is overwritten by
    // the exact FP32 residual result once its matching second-stream word has
    // been consumed.
    always_ff @(posedge fp_clk_i) begin
        memory_read_data <= vector_memory[memory_read_address];
        if (state == RESIDUAL_LOAD_OUTPUT && core_source_valid)
            vector_memory[word_index] <= core_source_data;
        if (state == RESIDUAL_ADD_HIGH && fp_result_valid)
            vector_memory[word_index] <= {fp_result, result_pair[31:0]};
    end

    always_ff @(posedge fp_clk_i) begin
        if (!fp_rst_ni) begin
            state <= RESIDUAL_IDLE;
            word_index <= 0;
            sum_index <= 0;
            memory_read_address <= 0;
            residual_pair <= 0;
            result_pair <= 0;
            square_value <= 0;
            sum_first <= 0;
            sum_second <= 0;
            total_sum <= 0;
            sum_hold <= 0;
            sum_tag_hold <= 0;
            active_tag <= 0;
            sum_toggle <= 0;
            fp_waiting <= 0;
        end else begin
            if (fp_input_valid && fp_input_ready)
                fp_waiting <= 1'b1;
            if (fp_result_valid)
                fp_waiting <= 1'b0;

            if (core_start) begin
                active_tag <= start_tag_hold;
                state <= RESIDUAL_LOAD_OUTPUT;
                word_index <= 0;
                sum_index <= 0;
                memory_read_address <= 0;
                residual_pair <= 0;
                result_pair <= 0;
                square_value <= 0;
                sum_first <= 0;
                sum_second <= 0;
                total_sum <= 0;
                fp_waiting <= 0;
            end else begin
                case (state)
                    RESIDUAL_IDLE: state <= RESIDUAL_IDLE;
                    RESIDUAL_LOAD_OUTPUT: if (core_source_valid) begin
                        if (word_index == WORDS - 1) begin
                            word_index <= 0;
                            state <= RESIDUAL_LOAD_RESIDUAL;
                        end else begin
                            word_index <= word_index + 1'b1;
                        end
                    end
                    RESIDUAL_LOAD_RESIDUAL: if (core_source_valid) begin
                        residual_pair <= core_source_data;
                        memory_read_address <= word_index;
                        state <= RESIDUAL_WAIT_VECTOR;
                    end
                    RESIDUAL_WAIT_VECTOR: state <= RESIDUAL_ADD_LOW;
                    RESIDUAL_ADD_LOW: if (fp_result_valid) begin
                        result_pair[31:0] <= fp_result;
                        state <= RESIDUAL_ADD_HIGH;
                    end
                    RESIDUAL_ADD_HIGH: if (fp_result_valid) begin
                        result_pair[63:32] <= fp_result;
                        if (word_index == WORDS - 1) begin
                            word_index <= 0;
                            sum_index <= 0;
                            memory_read_address <= 0;
                            sum_first <= 0;
                            sum_second <= 0;
                            state <= RESIDUAL_SUM_READ;
                        end else begin
                            word_index <= word_index + 1'b1;
                            state <= RESIDUAL_LOAD_RESIDUAL;
                        end
                    end
                    RESIDUAL_SUM_READ: begin
                        memory_read_address <=
                            sum_index[ELEMENT_INDEX_WIDTH-1:1];
                        state <= RESIDUAL_SUM_WAIT;
                    end
                    RESIDUAL_SUM_WAIT: state <= RESIDUAL_SQUARE;
                    RESIDUAL_SQUARE: if (fp_result_valid) begin
                        square_value <= fp_result;
                        state <= RESIDUAL_ACCUMULATE;
                    end
                    RESIDUAL_ACCUMULATE: if (fp_result_valid) begin
                        if (sum_index < ELEMENTS / 2)
                            sum_first <= fp_result;
                        else
                            sum_second <= fp_result;
                        if (sum_index == ELEMENTS - 1) begin
                            // FINAL_SUM must see the just-computed second-half
                            // value rather than the old nonblocking register.
                            sum_second <= fp_result;
                            square_value <= fp_result;
                            state <= RESIDUAL_FINAL_SUM;
                        end else begin
                            sum_index <= sum_index + 1'b1;
                            memory_read_address <=
                                (sum_index + 1'b1) >> 1;
                            state <= RESIDUAL_SUM_WAIT;
                        end
                    end
                    RESIDUAL_FINAL_SUM: if (fp_result_valid) begin
                        total_sum <= fp_result;
                        sum_hold <= fp_result;
                        // Carry the full source-domain transaction sequence
                        // with the bundled sum payload. A single toggle bit can
                        // alias if an old mailbox event arrives two commands
                        // late, making a stale sum appear current again.
                        sum_tag_hold <= active_tag;
                        sum_toggle <= ~sum_toggle;
                        word_index <= 0;
                        memory_read_address <= 0;
                        state <= RESIDUAL_OUTPUT_READ;
                    end
                    RESIDUAL_OUTPUT_READ: begin
                        memory_read_address <= word_index;
                        state <= RESIDUAL_OUTPUT_WAIT;
                    end
                    RESIDUAL_OUTPUT_WAIT: state <= RESIDUAL_OUTPUT_EMIT;
                    RESIDUAL_OUTPUT_EMIT: if (core_result_ready) begin
                        if (word_index == WORDS - 1) begin
                            state <= RESIDUAL_SUM_EMIT;
                        end else begin
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
    end
endmodule

// DDR-streaming integer accelerator for the Q4_0 matrices and Q6_K output
// embedding used by LFM2.5-350M. Floating-point scales are represented as
// signed Q30 integers, so the datapath does not require an FPGA FPU.
module lfm_matvec_accelerator #(
    parameter integer ID_WIDTH = 5
) (
    input  logic        clk_i,
    input  logic        rst_ni,
    input  logic [31:0] command_i,
    output logic [31:0] status_o,
    output logic        bus_request_o,
    input  logic        bus_grant_i,
    AXI_BUS.Master      axi
);

    localparam logic [29:0] MODEL_BASE = 30'h3000_0000;
    localparam integer MAX_INPUT = 4608;
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
        FINISH,
        RELEASE_BUS,
        SIGNAL_DONE
    } state_t;

    state_t state;
    logic [31:0] command_meta;
    logic [31:0] command_sync;
    logic        command_toggle_seen;

    logic [24:0] weight_offset_words;
    logic [24:0] input_offset_words;
    logic [24:0] output_offset_words;
    logic [15:0] ne0;
    logic [16:0] ne1;
    logic        q6_mode;

    logic [15:0] input_scale [0:MAX_BLOCKS-1];
    // Two replicated true-dual-port banks provide the four synchronous reads
    // needed by Q6_K without expanding the activation vector into registers.
    // Loading and matrix evaluation are mutually exclusive, so port A becomes
    // the write port only while the compact input vector is being received.
    (* ram_style = "block" *) logic signed [7:0] input_q0 [0:MAX_INPUT-1];
    (* ram_style = "block" *) logic signed [7:0] input_q1 [0:MAX_INPUT-1];
    logic [12:0] input_write_address;
    logic [12:0] input_read0a_address;
    logic [12:0] input_read0b_address;
    logic [12:0] input_read1a_address;
    logic [12:0] input_read1b_address;
    logic signed [7:0] input_read0a;
    logic signed [7:0] input_read0b;
    logic signed [7:0] input_read1a;
    logic signed [7:0] input_read1b;
    logic input_write_enable;
    logic [7:0] q6_ql [0:127];
    logic [7:0] q6_qh [0:63];
    logic signed [7:0] q6_scales [0:15];

    logic [31:0] input_byte_index;
    logic [15:0] input_block;
    logic [8:0]  input_position;
    logic [31:0] weight_byte_index;
    logic [32:0] weight_block_count;
    logic [8:0]  weight_position;
    logic [15:0] row_block;
    logic [16:0] row_index;
    logic [15:0] row_blocks;
    logic [15:0] weight_scale;
    logic signed [31:0] block_dot;
    logic signed [63:0] row_sum;
    logic signed [63:0] scaled_term;
    logic signed [63:0] best_score;
    logic signed [63:0] q6_row_result;
    logic [15:0] best_token;
    logic [5:0] q6_step;
    logic done_toggle;
    logic error_flag;
    logic [63:0] cycle_count;

    logic reader_start;
    logic [29:0] reader_address;
    logic [31:0] reader_byte_count;
    logic reader_busy;
    logic reader_done;
    logic reader_error;
    logic [15:0] reader_data;
    logic reader_byte_valid;
    logic reader_byte_ready;
    logic reader_word_mode;

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

    logic signed [16:0] q4_product0;
    logic signed [16:0] q4_product1;
    logic signed [16:0] q4_product2;
    logic signed [16:0] q4_product3;
    logic signed [8:0] q4_value_pipe0;
    logic signed [8:0] q4_value_pipe1;
    logic signed [8:0] q4_value_pipe2;
    logic signed [8:0] q4_value_pipe3;
    logic signed [7:0] q4_input_pipe0;
    logic signed [7:0] q4_input_pipe1;
    logic signed [7:0] q4_input_pipe2;
    logic signed [7:0] q4_input_pipe3;
    logic signed [18:0] q4_pair_sum;
    logic signed [31:0] q6_contribution;
    logic signed [63:0] scale_mantissa_product;
    logic signed [8:0]  scale_shift;
    logic                scale_zero;
    logic signed [31:0] scale_product;
    logic signed [63:0] wide_scaled_product;

    logic [4:0] scale_ea;
    logic [4:0] scale_eb;
    logic [10:0] scale_ma;
    logic [10:0] scale_mb;
    logic signed [11:0] scale_signed_ma;
    logic signed [11:0] scale_signed_mb;
    logic signed [7:0] scale_pa;
    logic signed [7:0] scale_pb;
    logic signed [63:0] aligned_scale_product;
    integer scale_right_shift;

    logic signed [8:0] q4_lo0;
    logic signed [8:0] q4_hi0;
    logic signed [8:0] q4_lo1;
    logic signed [8:0] q4_hi1;
    logic [15:0] active_weight_scale;
    integer q4_prefetch_position;

    logic [5:0] q6_l;
    logic q6_half;
    integer q6_ql_base;
    integer q6_qh_base;
    integer q6_prefetch_step;
    logic [5:0] q6_prefetch_l;
    logic q6_prefetch_half;
    integer q6_prefetch_base;
    integer q6_scale_base;
    logic signed [6:0] q6_v0;
    logic signed [6:0] q6_v1;
    logic signed [6:0] q6_v2;
    logic signed [6:0] q6_v3;
    logic signed [6:0] q6_value_pipe0;
    logic signed [6:0] q6_value_pipe1;
    logic signed [6:0] q6_value_pipe2;
    logic signed [6:0] q6_value_pipe3;
    logic signed [7:0] q6_input_pipe0;
    logic signed [7:0] q6_input_pipe1;
    logic signed [7:0] q6_input_pipe2;
    logic signed [7:0] q6_input_pipe3;
    logic signed [7:0] q6_scale_decode0;
    logic signed [7:0] q6_scale_decode1;
    logic signed [7:0] q6_scale_decode2;
    logic signed [7:0] q6_scale_decode3;
    logic signed [14:0] q6_pair_product0;
    logic signed [14:0] q6_pair_product1;
    logic signed [14:0] q6_pair_product2;
    logic signed [14:0] q6_pair_product3;
    logic signed [7:0] q6_scale_pipe0;
    logic signed [7:0] q6_scale_pipe1;
    logic signed [7:0] q6_scale_pipe2;
    logic signed [7:0] q6_scale_pipe3;
    logic signed [22:0] q6_scaled_product0;
    logic signed [22:0] q6_scaled_product1;
    logic signed [22:0] q6_scaled_product2;
    logic signed [22:0] q6_scaled_product3;

    always_comb begin
        q4_lo0 = $signed({1'b0, reader_data[3:0]}) - 9'sd8;
        q4_hi0 = $signed({1'b0, reader_data[7:4]}) - 9'sd8;
        q4_lo1 = $signed({1'b0, reader_data[11:8]}) - 9'sd8;
        q4_hi1 = $signed({1'b0, reader_data[15:12]}) - 9'sd8;
        // Q4 weights are consumed two packed bytes at a time. Request the next
        // activation pair only when the current word is accepted so the
        // synchronous BRAM outputs remain aligned across AXI burst gaps.
        q4_prefetch_position = weight_position == 0 ? 1 :
                               ((reader_byte_valid && reader_byte_ready &&
                                 weight_position < 8) ? weight_position + 1 :
                                weight_position);

        q6_l = q6_step[4:0];
        q6_half = q6_step[5];
        q6_ql_base = q6_half ? 64 : 0;
        q6_qh_base = q6_half ? 32 : 0;
        q6_scale_base = (q6_half ? 8 : 0) + (q6_l[4] ? 1 : 0);
        q6_v0 = $signed({1'b0, q6_ql[q6_ql_base + q6_l][3:0]}) +
                $signed({1'b0, q6_qh[q6_qh_base + q6_l][1:0], 4'b0}) - 7'sd32;
        q6_v1 = $signed({1'b0, q6_ql[q6_ql_base + 32 + q6_l][3:0]}) +
                $signed({1'b0, q6_qh[q6_qh_base + q6_l][3:2], 4'b0}) - 7'sd32;
        q6_v2 = $signed({1'b0, q6_ql[q6_ql_base + q6_l][7:4]}) +
                $signed({1'b0, q6_qh[q6_qh_base + q6_l][5:4], 4'b0}) - 7'sd32;
        q6_v3 = $signed({1'b0, q6_ql[q6_ql_base + 32 + q6_l][7:4]}) +
                $signed({1'b0, q6_qh[q6_qh_base + q6_l][7:6], 4'b0}) - 7'sd32;
        q6_contribution =
            {{9{q6_scaled_product0[22]}}, q6_scaled_product0} +
            {{9{q6_scaled_product1[22]}}, q6_scaled_product1} +
            {{9{q6_scaled_product2[22]}}, q6_scaled_product2} +
            {{9{q6_scaled_product3[22]}}, q6_scaled_product3};

        input_write_enable = state == LOAD_INPUT && reader_byte_valid &&
                             reader_byte_ready && input_position >= 2;
        input_write_address = (q6_mode ? input_block * 256 : input_block * 32) +
                              (input_position >= 2 ? input_position - 2 : 0);

        q6_prefetch_step = state == Q6_COMPUTE && q6_step < 63 ? q6_step + 1 : 0;
        q6_prefetch_l = q6_prefetch_step[4:0];
        q6_prefetch_half = q6_prefetch_step[5];
        q6_prefetch_base = row_block * 256 + (q6_prefetch_half ? 128 : 0);
        if (q6_mode) begin
            input_read0a_address = q6_prefetch_base + q6_prefetch_l;
            input_read0b_address = q6_prefetch_base + q6_prefetch_l + 32;
            input_read1a_address = q6_prefetch_base + q6_prefetch_l + 64;
            input_read1b_address = q6_prefetch_base + q6_prefetch_l + 96;
        end else begin
            input_read0a_address = row_block * 32 +
                                   ((q4_prefetch_position - 1) << 1);
            input_read0b_address = row_block * 32 +
                                   ((q4_prefetch_position - 1) << 1) + 16;
            input_read1a_address = row_block * 32 +
                                   ((q4_prefetch_position - 1) << 1) + 1;
            input_read1b_address = row_block * 32 +
                                   ((q4_prefetch_position - 1) << 1) + 17;
        end

        active_weight_scale =
            state == Q4_STREAM && weight_position == 0 && reader_byte_valid ?
            reader_data : weight_scale;
        scale_ea = active_weight_scale[14:10];
        scale_eb = input_scale[row_block][14:10];
        scale_ma = scale_ea == 0 ? {1'b0, active_weight_scale[9:0]} :
                                  {1'b1, active_weight_scale[9:0]};
        scale_mb = scale_eb == 0 ? {1'b0, input_scale[row_block][9:0]} :
                                  {1'b1, input_scale[row_block][9:0]};
        scale_pa = scale_ea == 0 ? -8'sd24 : $signed({3'b000, scale_ea}) - 8'sd25;
        scale_pb = scale_eb == 0 ? -8'sd24 : $signed({3'b000, scale_eb}) - 8'sd25;
        scale_right_shift = -scale_shift;
        if (scale_zero) begin
            aligned_scale_product = 64'sd0;
        end else if (scale_shift >= 0) begin
            aligned_scale_product = scale_mantissa_product <<< scale_shift;
        end else if (scale_mantissa_product >= 0) begin
            aligned_scale_product =
                (scale_mantissa_product + (64'sd1 <<< (scale_right_shift - 1))) >>>
                scale_right_shift;
        end else begin
            aligned_scale_product =
                -((-scale_mantissa_product + (64'sd1 <<< (scale_right_shift - 1))) >>>
                  scale_right_shift);
        end
        wide_scaled_product = block_dot * scale_product;
    end

    always_ff @(posedge clk_i) begin
        if (input_write_enable) begin
            input_q0[input_write_address] <= reader_data[7:0];
            input_q1[input_write_address] <= reader_data[7:0];
        end else begin
            input_read0a <= input_q0[input_read0a_address];
            input_read1a <= input_q1[input_read1a_address];
        end
        input_read0b <= input_q0[input_read0b_address];
        input_read1b <= input_q1[input_read1b_address];
    end

    assign status_o = {1'b1, bus_request_o, error_flag, done_toggle,
                       4'b0, best_token, 8'b0};

    assign axi.aw_id     = {ID_WIDTH{1'b1}};
    assign axi.aw_addr   = {{34{1'b0}}, write_address};
    assign axi.aw_len    = 8'd0;
    assign axi.aw_size   = 3'd3;
    assign axi.aw_burst  = 2'b01;
    assign axi.aw_lock   = 1'b0;
    assign axi.aw_cache  = 4'b0010;
    assign axi.aw_prot   = 3'b000;
    assign axi.aw_qos    = 4'b0000;
    assign axi.aw_region = 4'b0000;
    assign axi.aw_atop   = 6'b000000;
    assign axi.aw_user   = '0;
    assign axi.aw_valid  = state == Q4_WRITE && !write_aw_sent;
    assign axi.w_data    = write_data;
    assign axi.w_strb    = 8'hff;
    assign axi.w_last    = 1'b1;
    assign axi.w_user    = '0;
    assign axi.w_valid   = state == Q4_WRITE && !write_w_sent;
    assign axi.b_ready   = state == Q4_WRITE_RESPONSE;

    assign axi.ar_id     = reader_arid;
    assign axi.ar_addr   = {{34{1'b0}}, reader_araddr};
    assign axi.ar_len    = reader_arlen;
    assign axi.ar_size   = reader_arsize;
    assign axi.ar_burst  = reader_arburst;
    assign axi.ar_lock   = 1'b0;
    assign axi.ar_cache  = 4'b0010;
    assign axi.ar_prot   = 3'b000;
    assign axi.ar_qos    = 4'b0000;
    assign axi.ar_region = 4'b0000;
    assign axi.ar_user   = '0;
    assign axi.ar_valid  = reader_arvalid;
    assign axi.r_ready   = reader_rready;

    assign reader_byte_ready = (state == LOAD_INPUT) ||
                               (state == Q4_STREAM) ||
                               (state == Q6_STREAM);

    lfm_axi_byte_reader #(
        .ID_WIDTH(ID_WIDTH)
    ) i_reader (
        .clk_i,
        .rst_ni,
        .start_i(reader_start),
        .address_i(reader_address),
        .byte_count_i(reader_byte_count),
        .word_mode_i(reader_word_mode),
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

    always_ff @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
            command_meta        <= '0;
            command_sync        <= '0;
            command_toggle_seen <= 1'b0;
        end else begin
            command_meta <= command_i;
            command_sync <= command_meta;
            if (command_sync[31] != command_toggle_seen)
                command_toggle_seen <= command_sync[31];
        end
    end

    always_ff @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
            state               <= IDLE;
            bus_request_o       <= 1'b0;
            reader_start        <= 1'b0;
            reader_word_mode    <= 1'b0;
            reader_address      <= '0;
            reader_byte_count   <= '0;
            weight_offset_words <= '0;
            input_offset_words  <= '0;
            output_offset_words <= '0;
            ne0                 <= '0;
            ne1                 <= '0;
            q6_mode             <= 1'b0;
            input_byte_index    <= '0;
            input_block         <= '0;
            input_position      <= '0;
            weight_byte_index   <= '0;
            weight_block_count  <= '0;
            weight_position     <= '0;
            row_block           <= '0;
            row_index           <= '0;
            row_blocks          <= '0;
            weight_scale        <= '0;
            block_dot           <= '0;
            q4_product0         <= '0;
            q4_product1         <= '0;
            q4_product2         <= '0;
            q4_product3         <= '0;
            q4_value_pipe0      <= '0;
            q4_value_pipe1      <= '0;
            q4_value_pipe2      <= '0;
            q4_value_pipe3      <= '0;
            q4_input_pipe0      <= '0;
            q4_input_pipe1      <= '0;
            q4_input_pipe2      <= '0;
            q4_input_pipe3      <= '0;
            q4_pair_sum         <= '0;
            row_sum             <= '0;
            scaled_term         <= '0;
            scale_mantissa_product <= '0;
            scale_signed_ma     <= '0;
            scale_signed_mb     <= '0;
            scale_shift         <= '0;
            scale_zero          <= 1'b0;
            scale_product       <= '0;
            best_score          <= 64'sh8000_0000_0000_0000;
            q6_row_result       <= '0;
            best_token          <= '0;
            q6_step             <= '0;
            q6_value_pipe0      <= '0;
            q6_value_pipe1      <= '0;
            q6_value_pipe2      <= '0;
            q6_value_pipe3      <= '0;
            q6_input_pipe0      <= '0;
            q6_input_pipe1      <= '0;
            q6_input_pipe2      <= '0;
            q6_input_pipe3      <= '0;
            q6_scale_decode0    <= '0;
            q6_scale_decode1    <= '0;
            q6_scale_decode2    <= '0;
            q6_scale_decode3    <= '0;
            q6_pair_product0    <= '0;
            q6_pair_product1    <= '0;
            q6_pair_product2    <= '0;
            q6_pair_product3    <= '0;
            q6_scale_pipe0      <= '0;
            q6_scale_pipe1      <= '0;
            q6_scale_pipe2      <= '0;
            q6_scale_pipe3      <= '0;
            q6_scaled_product0  <= '0;
            q6_scaled_product1  <= '0;
            q6_scaled_product2  <= '0;
            q6_scaled_product3  <= '0;
            done_toggle         <= 1'b0;
            error_flag          <= 1'b0;
            cycle_count         <= '0;
            write_aw_sent       <= 1'b0;
            write_w_sent        <= 1'b0;
            write_address       <= '0;
            write_data          <= '0;
        end else begin
            reader_start <= 1'b0;
            if (bus_request_o)
                cycle_count <= cycle_count + 1'b1;
            if (reader_error)
                error_flag <= 1'b1;

            if (command_sync[31] != command_toggle_seen && command_sync[30:28] == 3'b010) begin
                case (command_sync[27:25])
                    3'd0: weight_offset_words <= command_sync[24:0];
                    3'd1: input_offset_words  <= command_sync[24:0];
                    3'd2: output_offset_words <= command_sync[24:0];
                    3'd3: ne0                 <= command_sync[15:0];
                    3'd4: ne1                 <= command_sync[15:0] == 0 ? 17'd65536 : {1'b0, command_sync[15:0]};
                    3'd5: begin
                        if (state != IDLE) begin
                            error_flag <= 1'b1;
                        end else if (command_sync[1:0] == 2'd1 || command_sync[1:0] == 2'd2) begin
                            q6_mode       <= command_sync[1];
                            error_flag    <= 1'b0;
                            best_score    <= 64'sh8000_0000_0000_0000;
                            best_token    <= '0;
                            cycle_count   <= '0;
                            bus_request_o <= 1'b1;
                            state         <= WAIT_GRANT;
                        end
                    end
                    default: error_flag <= error_flag;
                endcase
            end

            case (state)
                IDLE: begin
                end

                WAIT_GRANT: begin
                    if (bus_grant_i) begin
                        if (ne0 == 0 || ne1 == 0 || ne0 > MAX_INPUT ||
                            (!q6_mode && (ne0[4:0] != 0 || ((ne0 >> 5) & 3) != 0)) ||
                            (q6_mode && (ne0[7:0] != 0 || ((ne0 >> 8) & 3) != 0))) begin
                            error_flag <= 1'b1;
                            state <= FINISH;
                        end else begin
                            row_blocks <= q6_mode ? (ne0 >> 8) : (ne0 >> 5);
                            state <= START_INPUT;
                        end
                    end
                end

                START_INPUT: begin
                    reader_word_mode <= 1'b0;
                    reader_address <= MODEL_BASE + {input_offset_words, 3'b000};
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
                    if (reader_byte_valid && reader_byte_ready) begin
                        input_byte_index <= input_byte_index + 1'b1;
                        if (input_position == 0)
                            input_scale[input_block][7:0] <= reader_data[7:0];
                        else if (input_position == 1)
                            input_scale[input_block][15:8] <= reader_data[7:0];

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
                    reader_word_mode <= !q6_mode;
                    reader_start      <= 1'b1;
                    weight_byte_index <= '0;
                    weight_position   <= '0;
                    row_block         <= '0;
                    row_index         <= '0;
                    row_sum           <= '0;
                    block_dot         <= '0;
                    state             <= q6_mode ? Q6_STREAM : Q4_STREAM;
                end

                Q4_STREAM: begin
                    if (reader_byte_valid && reader_byte_ready) begin
                        weight_byte_index <= weight_byte_index + 2'd2;
                        if (weight_position == 0) begin
                            weight_scale <= reader_data;
                            block_dot   <= '0;
                            q4_value_pipe0 <= '0;
                            q4_value_pipe1 <= '0;
                            q4_value_pipe2 <= '0;
                            q4_value_pipe3 <= '0;
                            q4_input_pipe0 <= '0;
                            q4_input_pipe1 <= '0;
                            q4_input_pipe2 <= '0;
                            q4_input_pipe3 <= '0;
                            q4_product0 <= '0;
                            q4_product1 <= '0;
                            q4_product2 <= '0;
                            q4_product3 <= '0;
                            q4_pair_sum <= '0;
                            scale_signed_ma <=
                                active_weight_scale[15] ^ input_scale[row_block][15] ?
                                -$signed({1'b0, scale_ma}) :
                                 $signed({1'b0, scale_ma});
                            scale_signed_mb <= $signed({1'b0, scale_mb});
                            scale_shift <= scale_pa + scale_pb + 9'sd30;
                            scale_zero <=
                                (scale_ea == 0 && active_weight_scale[9:0] == 0) ||
                                (scale_eb == 0 && input_scale[row_block][9:0] == 0);
                            weight_position <= 1;
                        end else begin
                            // Register both multiplier operands so the 200 MHz
                            // path starts at local flip-flops instead of the
                            // synchronous activation BRAM outputs.
                            q4_value_pipe0 <= q4_lo0;
                            q4_value_pipe1 <= q4_hi0;
                            q4_value_pipe2 <= q4_lo1;
                            q4_value_pipe3 <= q4_hi1;
                            q4_input_pipe0 <= input_read0a;
                            q4_input_pipe1 <= input_read0b;
                            q4_input_pipe2 <= input_read1a;
                            q4_input_pipe3 <= input_read1b;
                            q4_product0 <= q4_value_pipe0 * q4_input_pipe0;
                            q4_product1 <= q4_value_pipe1 * q4_input_pipe1;
                            q4_product2 <= q4_value_pipe2 * q4_input_pipe2;
                            q4_product3 <= q4_value_pipe3 * q4_input_pipe3;
                            q4_pair_sum <=
                                {{2{q4_product0[16]}}, q4_product0} +
                                {{2{q4_product1[16]}}, q4_product1} +
                                {{2{q4_product2[16]}}, q4_product2} +
                                {{2{q4_product3[16]}}, q4_product3};
                            if (weight_position > 3)
                                block_dot <= block_dot +
                                    {{13{q4_pair_sum[18]}}, q4_pair_sum};
                            if (weight_position == 1)
                                scale_mantissa_product <=
                                    scale_signed_ma * scale_signed_mb;
                            if (weight_position == 3)
                                scale_product <= aligned_scale_product[31:0];
                            if (weight_position == 8) begin
                                weight_position <= '0;
                                state <= Q4_DRAIN_PRODUCT;
                            end else begin
                                weight_position <= weight_position + 1'b1;
                            end
                        end
                    end
                end

                Q4_DRAIN_PRODUCT: begin
                    q4_product0 <= q4_value_pipe0 * q4_input_pipe0;
                    q4_product1 <= q4_value_pipe1 * q4_input_pipe1;
                    q4_product2 <= q4_value_pipe2 * q4_input_pipe2;
                    q4_product3 <= q4_value_pipe3 * q4_input_pipe3;
                    q4_pair_sum <=
                        {{2{q4_product0[16]}}, q4_product0} +
                        {{2{q4_product1[16]}}, q4_product1} +
                        {{2{q4_product2[16]}}, q4_product2} +
                        {{2{q4_product3[16]}}, q4_product3};
                    block_dot <= block_dot +
                        {{13{q4_pair_sum[18]}}, q4_pair_sum};
                    state <= Q4_DRAIN;
                end

                Q4_DRAIN: begin
                    q4_pair_sum <=
                        {{2{q4_product0[16]}}, q4_product0} +
                        {{2{q4_product1[16]}}, q4_product1} +
                        {{2{q4_product2[16]}}, q4_product2} +
                        {{2{q4_product3[16]}}, q4_product3};
                    block_dot <= block_dot +
                        {{13{q4_pair_sum[18]}}, q4_pair_sum};
                    state <= Q4_SCALE;
                end

                Q4_SCALE: begin
                    block_dot <= block_dot +
                        {{13{q4_pair_sum[18]}}, q4_pair_sum};
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
                        write_address <= MODEL_BASE + {output_offset_words, 3'b000} + (row_index << 3);
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
                    if (reader_byte_valid && reader_byte_ready) begin
                        weight_byte_index <= weight_byte_index + 1'b1;
                        if (weight_position < 128)
                            q6_ql[weight_position] <= reader_data[7:0];
                        else if (weight_position < 192)
                            q6_qh[weight_position - 128] <= reader_data[7:0];
                        else if (weight_position < 208)
                            q6_scales[weight_position - 192] <= reader_data[7:0];
                        else if (weight_position == 208)
                            weight_scale[7:0] <= reader_data[7:0];
                        else
                            weight_scale[15:8] <= reader_data[7:0];

                        if (weight_position == 209) begin
                            weight_position <= '0;
                            block_dot <= '0;
                            q6_step <= '0;
                            q6_value_pipe0 <= '0;
                            q6_value_pipe1 <= '0;
                            q6_value_pipe2 <= '0;
                            q6_value_pipe3 <= '0;
                            q6_input_pipe0 <= '0;
                            q6_input_pipe1 <= '0;
                            q6_input_pipe2 <= '0;
                            q6_input_pipe3 <= '0;
                            q6_scale_decode0 <= '0;
                            q6_scale_decode1 <= '0;
                            q6_scale_decode2 <= '0;
                            q6_scale_decode3 <= '0;
                            q6_pair_product0 <= '0;
                            q6_pair_product1 <= '0;
                            q6_pair_product2 <= '0;
                            q6_pair_product3 <= '0;
                            q6_scaled_product0 <= '0;
                            q6_scaled_product1 <= '0;
                            q6_scaled_product2 <= '0;
                            q6_scaled_product3 <= '0;
                            state <= Q6_COMPUTE;
                        end else begin
                            weight_position <= weight_position + 1'b1;
                        end
                    end
                end

                Q6_COMPUTE: begin
                    // Register the variable Q6_K unpacking and BRAM outputs
                    // before multiplication. This removes the unpack muxes
                    // and offset arithmetic from the 200 MHz product path.
                    q6_value_pipe0 <= q6_v0;
                    q6_value_pipe1 <= q6_v1;
                    q6_value_pipe2 <= q6_v2;
                    q6_value_pipe3 <= q6_v3;
                    q6_input_pipe0 <= input_read0a;
                    q6_input_pipe1 <= input_read0b;
                    q6_input_pipe2 <= input_read1a;
                    q6_input_pipe3 <= input_read1b;
                    q6_scale_decode0 <= q6_scales[q6_scale_base];
                    q6_scale_decode1 <= q6_scales[q6_scale_base + 2];
                    q6_scale_decode2 <= q6_scales[q6_scale_base + 4];
                    q6_scale_decode3 <= q6_scales[q6_scale_base + 6];
                    q6_pair_product0 <= q6_value_pipe0 * q6_input_pipe0;
                    q6_pair_product1 <= q6_value_pipe1 * q6_input_pipe1;
                    q6_pair_product2 <= q6_value_pipe2 * q6_input_pipe2;
                    q6_pair_product3 <= q6_value_pipe3 * q6_input_pipe3;
                    q6_scale_pipe0 <= q6_scale_decode0;
                    q6_scale_pipe1 <= q6_scale_decode1;
                    q6_scale_pipe2 <= q6_scale_decode2;
                    q6_scale_pipe3 <= q6_scale_decode3;
                    if (q6_step >= 2) begin
                        q6_scaled_product0 <= q6_pair_product0 * q6_scale_pipe0;
                        q6_scaled_product1 <= q6_pair_product1 * q6_scale_pipe1;
                        q6_scaled_product2 <= q6_pair_product2 * q6_scale_pipe2;
                        q6_scaled_product3 <= q6_pair_product3 * q6_scale_pipe3;
                    end
                    if (q6_step >= 3)
                        block_dot <= block_dot + q6_contribution;
                    if (q6_step == 63) begin
                        state <= Q6_DRAIN_PAIR;
                    end else begin
                        q6_step <= q6_step + 1'b1;
                    end
                end

                Q6_DRAIN_PAIR: begin
                    q6_pair_product0 <= q6_value_pipe0 * q6_input_pipe0;
                    q6_pair_product1 <= q6_value_pipe1 * q6_input_pipe1;
                    q6_pair_product2 <= q6_value_pipe2 * q6_input_pipe2;
                    q6_pair_product3 <= q6_value_pipe3 * q6_input_pipe3;
                    q6_scale_pipe0 <= q6_scale_decode0;
                    q6_scale_pipe1 <= q6_scale_decode1;
                    q6_scale_pipe2 <= q6_scale_decode2;
                    q6_scale_pipe3 <= q6_scale_decode3;
                    q6_scaled_product0 <= q6_pair_product0 * q6_scale_pipe0;
                    q6_scaled_product1 <= q6_pair_product1 * q6_scale_pipe1;
                    q6_scaled_product2 <= q6_pair_product2 * q6_scale_pipe2;
                    q6_scaled_product3 <= q6_pair_product3 * q6_scale_pipe3;
                    block_dot <= block_dot + q6_contribution;
                    state <= Q6_DRAIN_SCALE;
                end

                Q6_DRAIN_SCALE: begin
                    q6_scaled_product0 <= q6_pair_product0 * q6_scale_pipe0;
                    q6_scaled_product1 <= q6_pair_product1 * q6_scale_pipe1;
                    q6_scaled_product2 <= q6_pair_product2 * q6_scale_pipe2;
                    q6_scaled_product3 <= q6_pair_product3 * q6_scale_pipe3;
                    block_dot <= block_dot + q6_contribution;
                    state <= Q6_DRAIN_ACCUMULATE;
                end

                Q6_DRAIN_ACCUMULATE: begin
                    block_dot <= block_dot + q6_contribution;
                    state <= Q6_SCALE;
                end

                Q6_SCALE: begin
                    scale_signed_ma <=
                        weight_scale[15] ^ input_scale[row_block][15] ?
                        -$signed({1'b0, scale_ma}) :
                         $signed({1'b0, scale_ma});
                    scale_signed_mb <= $signed({1'b0, scale_mb});
                    scale_shift <= scale_pa + scale_pb + 9'sd30;
                    scale_zero <= (scale_ea == 0 && weight_scale[9:0] == 0) ||
                                  (scale_eb == 0 && input_scale[row_block][9:0] == 0);
                    state <= Q6_SCALE_MANTISSA;
                end

                Q6_SCALE_MANTISSA: begin
                    scale_mantissa_product <= scale_signed_ma * scale_signed_mb;
                    state <= Q6_SCALE_ALIGN_WAIT;
                end

                Q6_SCALE_ALIGN_WAIT: begin
                    state <= Q6_SCALE_ALIGN;
                end

                Q6_SCALE_ALIGN: begin
                    scale_product <= aligned_scale_product[31:0];
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
                        best_token <= row_index[15:0];
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
                    error_flag    <= 1'b1;
                    bus_request_o <= 1'b0;
                    state         <= IDLE;
                end
            endcase
        end
    end

endmodule

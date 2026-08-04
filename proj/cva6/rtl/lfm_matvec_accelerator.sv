// DDR-streaming integer accelerator for the Q4_0 matrices and Q6_K output
// embedding used by LFM2.5-350M. Floating-point scales are represented as
// signed Q40 integers, so the datapath does not require an FPGA FPU.
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

    typedef enum logic [4:0] {
        IDLE,
        WAIT_GRANT,
        START_INPUT,
        LOAD_INPUT,
        START_WEIGHTS,
        Q4_STREAM,
        Q4_SCALE,
        Q4_ACCUMULATE,
        Q4_WRITE,
        Q4_WRITE_RESPONSE,
        Q6_STREAM,
        Q6_COMPUTE,
        Q6_SCALE,
        Q6_ACCUMULATE,
        FINISH
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
    logic signed [7:0] input_q [0:MAX_INPUT-1];
    logic [7:0] q6_ql [0:127];
    logic [7:0] q6_qh [0:63];
    logic signed [7:0] q6_scales [0:15];

    logic [31:0] input_byte_index;
    logic [15:0] input_block;
    logic [8:0]  input_position;
    logic [31:0] weight_byte_index;
    logic [8:0]  weight_position;
    logic [15:0] row_block;
    logic [16:0] row_index;
    logic [15:0] row_blocks;
    logic [15:0] weight_scale;
    logic signed [31:0] block_dot;
    logic signed [63:0] row_sum;
    logic signed [63:0] scaled_term;
    logic signed [63:0] best_score;
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
    logic [7:0] reader_byte;
    logic reader_byte_valid;
    logic reader_byte_ready;

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

    logic signed [31:0] q4_contribution;
    logic signed [31:0] q6_contribution;
    logic signed [63:0] scale_product;
    logic signed [95:0] wide_scaled_product;

    function automatic signed [63:0] fp16_product_q40(
        input logic [15:0] a,
        input logic [15:0] b
    );
        logic [4:0] ea;
        logic [4:0] eb;
        logic [9:0] fa;
        logic [9:0] fb;
        logic signed [63:0] product;
        logic signed [63:0] magnitude;
        logic signed [7:0] pa;
        logic signed [7:0] pb;
        logic signed [8:0] shift;
        integer right_shift;
        begin
            ea = a[14:10];
            eb = b[14:10];
            fa = a[9:0];
            fb = b[9:0];
            if ((ea == 0 && fa == 0) || (eb == 0 && fb == 0)) begin
                fp16_product_q40 = 64'sd0;
            end else begin
                magnitude = (ea == 0 ? {54'd0, fa} : {53'd0, 1'b1, fa}) *
                            (eb == 0 ? {54'd0, fb} : {53'd0, 1'b1, fb});
                product = (a[15] ^ b[15]) ? -magnitude : magnitude;
                pa = ea == 0 ? -8'sd24 : $signed({3'b000, ea}) - 8'sd25;
                pb = eb == 0 ? -8'sd24 : $signed({3'b000, eb}) - 8'sd25;
                shift = pa + pb + 9'sd40;
                if (shift >= 0) begin
                    fp16_product_q40 = product <<< shift;
                end else begin
                    right_shift = -shift;
                    if (product >= 0)
                        fp16_product_q40 = (product + (64'sd1 <<< (right_shift - 1))) >>> right_shift;
                    else
                        fp16_product_q40 = -((-product + (64'sd1 <<< (right_shift - 1))) >>> right_shift);
                end
            end
        end
    endfunction

    logic signed [8:0] q4_lo;
    logic signed [8:0] q4_hi;
    logic signed [7:0] q4_input_lo;
    logic signed [7:0] q4_input_hi;
    integer q4_input_base;

    logic [5:0] q6_l;
    logic q6_half;
    integer q6_ql_base;
    integer q6_qh_base;
    integer q6_input_base;
    integer q6_scale_base;
    logic signed [6:0] q6_v0;
    logic signed [6:0] q6_v1;
    logic signed [6:0] q6_v2;
    logic signed [6:0] q6_v3;

    always_comb begin
        q4_lo = $signed({1'b0, reader_byte[3:0]}) - 9'sd8;
        q4_hi = $signed({1'b0, reader_byte[7:4]}) - 9'sd8;
        q4_input_base = row_block * 32;
        if (weight_position >= 2) begin
            q4_input_lo = input_q[q4_input_base + weight_position - 2];
            q4_input_hi = input_q[q4_input_base + weight_position + 14];
        end else begin
            q4_input_lo = '0;
            q4_input_hi = '0;
        end
        q4_contribution = q4_lo * q4_input_lo + q4_hi * q4_input_hi;

        q6_l = q6_step[4:0];
        q6_half = q6_step[5];
        q6_ql_base = q6_half ? 64 : 0;
        q6_qh_base = q6_half ? 32 : 0;
        q6_input_base = row_block * 256 + (q6_half ? 128 : 0);
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
            q6_v0 * input_q[q6_input_base + q6_l] * q6_scales[q6_scale_base] +
            q6_v1 * input_q[q6_input_base + q6_l + 32] * q6_scales[q6_scale_base + 2] +
            q6_v2 * input_q[q6_input_base + q6_l + 64] * q6_scales[q6_scale_base + 4] +
            q6_v3 * input_q[q6_input_base + q6_l + 96] * q6_scales[q6_scale_base + 6];

        scale_product = fp16_product_q40(weight_scale, input_scale[row_block]);
        wide_scaled_product = block_dot * scale_product;
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
        .busy_o(reader_busy),
        .done_o(reader_done),
        .error_o(reader_error),
        .byte_o(reader_byte),
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
            weight_position     <= '0;
            row_block           <= '0;
            row_index           <= '0;
            row_blocks          <= '0;
            weight_scale        <= '0;
            block_dot           <= '0;
            row_sum             <= '0;
            scaled_term         <= '0;
            best_score          <= 64'sh8000_0000_0000_0000;
            best_token          <= '0;
            q6_step             <= '0;
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
                    reader_address <= MODEL_BASE + {input_offset_words, 3'b000};
                    reader_byte_count <= q6_mode ? ((ne0 >> 8) * 258) : ((ne0 >> 5) * 34);
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
                            input_scale[input_block][7:0] <= reader_byte;
                        else if (input_position == 1)
                            input_scale[input_block][15:8] <= reader_byte;
                        else
                            input_q[(q6_mode ? input_block * 256 : input_block * 32) + input_position - 2] <= reader_byte;

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
                            state <= START_WEIGHTS;
                        end
                    end
                end

                START_WEIGHTS: begin
                    reader_address <= MODEL_BASE + {weight_offset_words, 3'b000};
                    reader_byte_count <= q6_mode ?
                        ((ne0 >> 8) * 210 * ne1) :
                        ((ne0 >> 5) * 18 * ne1);
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
                        weight_byte_index <= weight_byte_index + 1'b1;
                        if (weight_position == 0) begin
                            weight_scale[7:0] <= reader_byte;
                            block_dot <= '0;
                            weight_position <= 1;
                        end else if (weight_position == 1) begin
                            weight_scale[15:8] <= reader_byte;
                            weight_position <= 2;
                        end else begin
                            if (weight_position == 17) begin
                                block_dot <= block_dot + q4_contribution;
                                weight_position <= '0;
                                state <= Q4_SCALE;
                            end else begin
                                block_dot <= block_dot + q4_contribution;
                                weight_position <= weight_position + 1'b1;
                            end
                        end
                    end
                end

                Q4_SCALE: begin
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
                            q6_ql[weight_position] <= reader_byte;
                        else if (weight_position < 192)
                            q6_qh[weight_position - 128] <= reader_byte;
                        else if (weight_position < 208)
                            q6_scales[weight_position - 192] <= reader_byte;
                        else if (weight_position == 208)
                            weight_scale[7:0] <= reader_byte;
                        else
                            weight_scale[15:8] <= reader_byte;

                        if (weight_position == 209) begin
                            weight_position <= '0;
                            block_dot <= '0;
                            q6_step <= '0;
                            state <= Q6_COMPUTE;
                        end else begin
                            weight_position <= weight_position + 1'b1;
                        end
                    end
                end

                Q6_COMPUTE: begin
                    if (q6_step == 63) begin
                        block_dot <= block_dot + q6_contribution;
                        state <= Q6_SCALE;
                    end else begin
                        block_dot <= block_dot + q6_contribution;
                        q6_step <= q6_step + 1'b1;
                    end
                end

                Q6_SCALE: begin
                    scaled_term <= wide_scaled_product[63:0];
                    state <= Q6_ACCUMULATE;
                end

                Q6_ACCUMULATE: begin
                    if (row_block + 1 == row_blocks) begin
                        if ($signed(row_sum + scaled_term) > $signed(best_score)) begin
                            best_score <= row_sum + scaled_term;
                            best_token <= row_index[15:0];
                        end
                        if (row_index + 1 == ne1) begin
                            state <= FINISH;
                        end else begin
                            row_index <= row_index + 1'b1;
                            row_block <= '0;
                            row_sum   <= '0;
                            state     <= Q6_STREAM;
                        end
                    end else begin
                        row_sum   <= row_sum + scaled_term;
                        row_block <= row_block + 1'b1;
                        state     <= Q6_STREAM;
                    end
                end

                FINISH: begin
                    bus_request_o <= 1'b0;
                    done_toggle   <= ~done_toggle;
                    state         <= IDLE;
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

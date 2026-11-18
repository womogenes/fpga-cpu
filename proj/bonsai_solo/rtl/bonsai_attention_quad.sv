// Parallel Bonsai attention wrapper. CORE_COUNT=2 is the placement-friendly
// production fallback (eight query/four GQA heads per core); CORE_COUNT=4
// gives the requested four-way arithmetic experiment (four query/two GQA
// heads per core). A transaction arbiter preserves the 256-bit AXI port.
module bonsai_attention_quad #(
    parameter integer ID_WIDTH = 5,
    parameter integer DATA_WIDTH = 128,
    parameter logic [29:0] KV_BASE = 30'h2000_0000,
    parameter integer CORE_COUNT = 2
) (
    input  logic                 clk_i,
    input  logic                 rst_ni,
    input  logic                 start_i,
    input  logic [29:0]          raw_base_i,
    input  logic [29:0]          output_base_i,
    input  logic [29:0]          table_base_i,
    input  logic [4:0]           layer_i,
    input  logic [10:0]          position_i,
    input  logic                 direct_raw_i,
    output logic [10:0]          direct_raw_word_o,
    input  logic [63:0]          direct_raw_data_i,
    output logic                 busy_o,
    output logic                 done_o,
    output logic                 error_o,
    output logic [ID_WIDTH-1:0]  arid_o,
    output logic [29:0]          araddr_o,
    output logic [7:0]           arlen_o,
    output logic [2:0]           arsize_o,
    output logic [1:0]           arburst_o,
    output logic                 arvalid_o,
    input  logic                 arready_i,
    input  logic [ID_WIDTH-1:0]  rid_i,
    input  logic [DATA_WIDTH-1:0] rdata_i,
    input  logic [1:0]           rresp_i,
    input  logic                 rlast_i,
    input  logic                 rvalid_i,
    output logic                 rready_o,
    output logic [ID_WIDTH-1:0]  awid_o,
    output logic [29:0]          awaddr_o,
    output logic [7:0]           awlen_o,
    output logic                 awvalid_o,
    input  logic                 awready_i,
    output logic [DATA_WIDTH-1:0] wdata_o,
    output logic [DATA_WIDTH/8-1:0] wstrb_o,
    output logic                 wlast_o,
    output logic                 wvalid_o,
    input  logic                 wready_i,
    input  logic [ID_WIDTH-1:0]  bid_i,
    input  logic [1:0]           bresp_i,
    input  logic                 bvalid_i,
    output logic                 bready_o,
    output logic                 packer_start_o,
    output logic [63:0]          packer_source_data_o,
    output logic                 packer_source_valid_o,
    input  logic                 packer_source_ready_i,
    input  logic [271:0]         packer_block_data_i,
    input  logic                 packer_block_valid_i,
    output logic                 packer_block_ready_o
);
    localparam integer CORES = CORE_COUNT;
    localparam integer HEADS_PER_CORE = 16 / CORE_COUNT;
    localparam integer KV_HEADS_PER_CORE = 8 / CORE_COUNT;
    localparam integer SOURCE_PAIRS_PER_CORE = HEADS_PER_CORE * 64;
    localparam integer BLOCKS_PER_CORE = HEADS_PER_CORE * 4;

    logic [CORES-1:0] core_busy, core_done, core_error;
    logic [CORES-1:0] done_seen;

    logic [CORES-1:0] direct_req, direct_ready;
    logic [10:0] direct_word [0:CORES-1];
    logic [63:0] direct_data [0:CORES-1];
    logic [1:0] direct_select, direct_owner;
    logic direct_select_valid, direct_owner_valid;

    logic [ID_WIDTH-1:0] core_arid [0:CORES-1];
    logic [29:0] core_araddr [0:CORES-1];
    logic [7:0] core_arlen [0:CORES-1];
    logic [2:0] core_arsize [0:CORES-1];
    logic [1:0] core_arburst [0:CORES-1];
    logic [CORES-1:0] core_arvalid, core_arready;
    logic [CORES-1:0] core_rvalid, core_rready;

    logic [ID_WIDTH-1:0] core_awid [0:CORES-1];
    logic [29:0] core_awaddr [0:CORES-1];
    logic [7:0] core_awlen [0:CORES-1];
    logic [CORES-1:0] core_awvalid, core_awready;
    logic [DATA_WIDTH-1:0] core_wdata [0:CORES-1];
    logic [DATA_WIDTH/8-1:0] core_wstrb [0:CORES-1];
    logic [CORES-1:0] core_wlast, core_wvalid, core_wready;
    logic [CORES-1:0] core_bvalid, core_bready;

    logic [1:0] read_select, read_owner;
    logic read_select_valid, read_owner_valid;
    logic [1:0] write_select, write_owner;
    logic write_select_valid, write_owner_valid;

    logic [CORES-1:0] packer_start;
    logic [63:0] packer_source_data [0:CORES-1];
    logic [CORES-1:0] packer_source_valid, packer_source_ready;
    logic [271:0] packer_block_data [0:CORES-1];
    logic [CORES-1:0] packer_block_valid, packer_block_ready;
    logic [1:0] packer_source_owner, packer_block_owner;
    logic [8:0] packer_source_count;
    logic [5:0] packer_block_count;

    genvar g;
    generate
        for (g = 0; g < CORES; g = g + 1) begin : gen_attention
            bonsai_attention #(
                .ID_WIDTH(ID_WIDTH),
                .DATA_WIDTH(DATA_WIDTH),
                .KV_BASE(KV_BASE),
                .HEAD_FIRST(g * HEADS_PER_CORE),
                .HEAD_LAST(g * HEADS_PER_CORE + HEADS_PER_CORE - 1),
                .KV_HEAD_FIRST(g * KV_HEADS_PER_CORE),
                .KV_HEAD_LAST(g * KV_HEADS_PER_CORE + KV_HEADS_PER_CORE - 1),
                .PACK_GROUP_FIRST(g * HEADS_PER_CORE),
                .PACK_GROUP_LAST(g * HEADS_PER_CORE + HEADS_PER_CORE - 1),
                // Each partition owns complete adjacent GQA query pairs, so
                // retain the proven two-query/shared-KV traversal inside
                // every parallel core.  Disabling it here erased most of the
                // gain from splitting heads across cores.
                .DUAL_QUERY(1'b1)
            ) i_core (
                .clk_i, .rst_ni, .start_i,
                .raw_base_i, .output_base_i, .table_base_i,
                .layer_i, .position_i, .direct_raw_i,
                .direct_raw_req_o(direct_req[g]),
                .direct_raw_word_o(direct_word[g]),
                .direct_raw_ready_i(direct_ready[g]),
                .direct_raw_data_i(direct_data[g]),
                .busy_o(core_busy[g]), .done_o(core_done[g]),
                .error_o(core_error[g]),
                .arid_o(core_arid[g]), .araddr_o(core_araddr[g]),
                .arlen_o(core_arlen[g]), .arsize_o(core_arsize[g]),
                .arburst_o(core_arburst[g]), .arvalid_o(core_arvalid[g]),
                .arready_i(core_arready[g]), .rid_i,
                .rdata_i, .rresp_i, .rlast_i,
                .rvalid_i(core_rvalid[g]), .rready_o(core_rready[g]),
                .awid_o(core_awid[g]), .awaddr_o(core_awaddr[g]),
                .awlen_o(core_awlen[g]), .awvalid_o(core_awvalid[g]),
                .awready_i(core_awready[g]), .wdata_o(core_wdata[g]),
                .wstrb_o(core_wstrb[g]), .wlast_o(core_wlast[g]),
                .wvalid_o(core_wvalid[g]), .wready_i(core_wready[g]),
                .bid_i, .bresp_i, .bvalid_i(core_bvalid[g]),
                .bready_o(core_bready[g]),
                .packer_start_o(packer_start[g]),
                .packer_source_data_o(packer_source_data[g]),
                .packer_source_valid_o(packer_source_valid[g]),
                .packer_source_ready_i(packer_source_ready[g]),
                .packer_block_data_i(packer_block_data[g]),
                .packer_block_valid_i(packer_block_valid[g]),
                .packer_block_ready_o(packer_block_ready[g])
            );

        end
    endgenerate

    // The vector packer stores all 2,048 FP32 values before quantizing, so
    // feed it each core's 512-value range in global head order. Its 64 Q8
    // blocks return in the same order and are routed sixteen at a time to the
    // core that owns the corresponding four output groups.
    always_comb begin
        packer_start_o = |packer_start;
        packer_source_data_o = packer_source_data[packer_source_owner];
        packer_source_valid_o = packer_source_valid[packer_source_owner];
        packer_source_ready = '0;
        packer_source_ready[packer_source_owner] = packer_source_ready_i;
        for (integer lane = 0; lane < CORES; lane = lane + 1) begin
            packer_block_data[lane] = packer_block_data_i;
            packer_block_valid[lane] = 1'b0;
        end
        packer_block_valid[packer_block_owner] = packer_block_valid_i;
        packer_block_ready_o = packer_block_ready[packer_block_owner];
    end

    always_comb begin
        direct_select = 0;
        direct_select_valid = 0;
        direct_ready = '0;
        direct_raw_word_o = 0;
        for (integer lane = CORES - 1; lane >= 0; lane = lane - 1) begin
            if (direct_req[lane]) begin
                direct_select = lane[1:0];
                direct_select_valid = 1;
            end
        end
        if (direct_select_valid) begin
            direct_ready[direct_select] = 1'b1;
            direct_raw_word_o = direct_word[direct_select];
        end
        for (integer lane = 0; lane < CORES; lane = lane + 1)
            direct_data[lane] = direct_owner_valid &&
                                direct_owner == lane[1:0] ?
                                direct_raw_data_i : 64'b0;
    end

    always_comb begin
        read_select = 0;
        read_select_valid = 0;
        core_arready = '0;
        arid_o = '0;
        araddr_o = '0;
        arlen_o = '0;
        arsize_o = '0;
        arburst_o = '0;
        arvalid_o = 0;
        if (!read_owner_valid) begin
            for (integer lane = CORES - 1; lane >= 0; lane = lane - 1) begin
                if (core_arvalid[lane]) begin
                    read_select = lane[1:0];
                    read_select_valid = 1;
                end
            end
            if (read_select_valid) begin
                arid_o = core_arid[read_select];
                araddr_o = core_araddr[read_select];
                arlen_o = core_arlen[read_select];
                arsize_o = core_arsize[read_select];
                arburst_o = core_arburst[read_select];
                arvalid_o = core_arvalid[read_select];
                core_arready[read_select] = arready_i;
            end
        end
        core_rvalid = '0;
        rready_o = 0;
        if (read_owner_valid) begin
            core_rvalid[read_owner] = rvalid_i;
            rready_o = core_rready[read_owner];
        end
    end

    always_comb begin
        write_select = 0;
        write_select_valid = 0;
        core_awready = '0;
        core_wready = '0;
        core_bvalid = '0;
        awid_o = '0;
        awaddr_o = '0;
        awlen_o = '0;
        awvalid_o = 0;
        wdata_o = '0;
        wstrb_o = '0;
        wlast_o = 0;
        wvalid_o = 0;
        bready_o = 0;
        if (!write_owner_valid) begin
            for (integer lane = CORES - 1; lane >= 0; lane = lane - 1) begin
                if (core_awvalid[lane]) begin
                    write_select = lane[1:0];
                    write_select_valid = 1;
                end
            end
            if (write_select_valid) begin
                awid_o = core_awid[write_select];
                awaddr_o = core_awaddr[write_select];
                awlen_o = core_awlen[write_select];
                awvalid_o = core_awvalid[write_select];
                core_awready[write_select] = awready_i;
            end
        end else begin
            wdata_o = core_wdata[write_owner];
            wstrb_o = core_wstrb[write_owner];
            wlast_o = core_wlast[write_owner];
            wvalid_o = core_wvalid[write_owner];
            core_wready[write_owner] = wready_i;
            core_bvalid[write_owner] = bvalid_i;
            bready_o = core_bready[write_owner];
        end
    end

    always_ff @(posedge clk_i or negedge rst_ni) begin
        logic [CORES-1:0] next_done;
        if (!rst_ni) begin
            read_owner_valid <= 0;
            write_owner_valid <= 0;
            direct_owner_valid <= 0;
            packer_source_owner <= 0;
            packer_source_count <= 0;
            packer_block_owner <= 0;
            packer_block_count <= 0;
            busy_o <= 0;
            done_o <= 0;
            error_o <= 0;
            done_seen <= '0;
        end else begin
            done_o <= 0;
            direct_owner_valid <= direct_select_valid;
            if (direct_select_valid) direct_owner <= direct_select;

            if (!read_owner_valid && arvalid_o && arready_i) begin
                read_owner <= read_select;
                read_owner_valid <= 1;
            end else if (read_owner_valid && rvalid_i && rready_o && rlast_i) begin
                read_owner_valid <= 0;
            end
            if (!write_owner_valid && awvalid_o && awready_i) begin
                write_owner <= write_select;
                write_owner_valid <= 1;
            end else if (write_owner_valid && bvalid_i && bready_o) begin
                write_owner_valid <= 0;
            end

            if (start_i && !busy_o) begin
                busy_o <= 1;
                error_o <= 0;
                done_seen <= '0;
                packer_source_owner <= 0;
                packer_source_count <= 0;
                packer_block_owner <= 0;
                packer_block_count <= 0;
            end else if (busy_o) begin
                next_done = done_seen | core_done;
                done_seen <= next_done;
                error_o <= error_o | (|core_error);
                if (&next_done) begin
                    busy_o <= 0;
                    done_o <= 1;
                end
            end
            if (packer_source_valid_o && packer_source_ready_i) begin
                if (packer_source_count == SOURCE_PAIRS_PER_CORE - 1) begin
                    packer_source_count <= 0;
                    if (packer_source_owner != CORE_COUNT - 1)
                        packer_source_owner <= packer_source_owner + 1'b1;
                end else begin
                    packer_source_count <= packer_source_count + 1'b1;
                end
            end
            if (packer_block_valid_i && packer_block_ready_o) begin
                if (packer_block_count == BLOCKS_PER_CORE - 1) begin
                    packer_block_count <= 0;
                    if (packer_block_owner != CORE_COUNT - 1)
                        packer_block_owner <= packer_block_owner + 1'b1;
                end else begin
                    packer_block_count <= packer_block_count + 1'b1;
                end
            end
        end
    end
endmodule

`timescale 1ns/1ps

module bonsai_attention_tb;
    localparam logic [29:0] RAW_BASE = 30'h0001_0000;
    localparam logic [29:0] TABLE_BASE = 30'h0002_0000;
    localparam logic [29:0] OUTPUT_BASE = 30'h0003_0000;
    localparam logic [29:0] KV_BASE = 30'h3e24_0000;

    logic clk = 0;
    logic rst_n = 0;
    logic start = 0;
    logic [7:0] position_cfg = 0;
    logic busy, done, error;
    logic [4:0] arid, rid, awid, bid;
    logic [29:0] araddr, awaddr;
    logic [7:0] arlen, awlen;
    logic [2:0] arsize;
    logic [1:0] arburst, rresp, bresp;
    logic arvalid, arready = 1;
    logic [63:0] rdata;
    logic rlast, rvalid, rready;
    logic awvalid, awready = 1;
    logic [63:0] wdata;
    logic [7:0] wstrb;
    logic wlast, wvalid, wready = 1;
    logic bvalid, bready;
    logic packer_start;
    logic [63:0] packer_source_data;
    logic packer_source_valid, packer_source_ready;
    logic [271:0] packer_block_data;
    logic packer_block_valid, packer_block_ready;
    logic packer_busy, packer_done;
    logic [3:0] packer_debug_state;

    logic [63:0] raw_mem [0:2047];
    logic [63:0] table_mem [0:191];
    logic [63:0] kv_mem [0:2047];
    logic [63:0] output_mem [0:1023];
    logic [63:0] replay_reference [0:271];
    logic read_pending;
    logic [29:0] read_address;
    logic [7:0] read_remaining;
    logic response_pending, write_pending;
    logic [29:0] pending_write_address;
    logic [7:0] pending_write_remaining;
    integer cycle_count;
    integer i;
    integer block_index;
    integer byte_index;

    always #2.5 clk = ~clk;

    bonsai_ffn_packer #(.ELEMENTS(2048)) packer (
        .clk_i(clk), .fp_clk_i(clk), .rst_ni(rst_n), .fp_rst_ni(rst_n),
        .start_i(packer_start), .norm_mode_i(1'b0),
        .vector_mode_i(1'b1), .compact_source_i(1'b0),
        .element_count_i(13'd2048), .norm_scale_i(32'h3f80_0000),
        .source_data_i(packer_source_data),
        .source_valid_i(packer_source_valid),
        .source_ready_o(packer_source_ready),
        .block_data_o(packer_block_data),
        .block_valid_o(packer_block_valid),
        .block_ready_i(packer_block_ready),
        .busy_o(packer_busy), .done_o(packer_done),
        .debug_state_o(packer_debug_state)
    );

    function automatic [63:0] memory_read(input logic [29:0] address);
        begin
            if (address >= RAW_BASE && address < RAW_BASE + 30'd16384)
                memory_read = raw_mem[(address - RAW_BASE) >> 3];
            else if (address >= TABLE_BASE && address < TABLE_BASE + 30'd1536)
                memory_read = table_mem[(address - TABLE_BASE) >> 3];
            else if (address >= KV_BASE && address < KV_BASE + 30'd16384)
                memory_read = kv_mem[(address - KV_BASE) >> 3];
            else
                memory_read = 64'hxxxx_xxxx_xxxx_xxxx;
        end
    endfunction

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            read_pending <= 0;
            read_remaining <= 0;
            rvalid <= 0;
            rdata <= 0;
            rlast <= 0;
            rid <= 0;
            rresp <= 0;
            bvalid <= 0;
            bid <= 0;
            bresp <= 0;
            response_pending <= 0;
            write_pending <= 0;
        end else begin
            if (arvalid && arready) begin
                read_pending <= 1;
                read_address <= araddr;
                read_remaining <= arlen;
            end
            if (read_pending && !rvalid) begin
                read_pending <= 0;
                rvalid <= 1;
                rdata <= memory_read(read_address);
                rlast <= read_remaining == 0;
                rid <= {5{1'b1}};
                rresp <= 0;
            end
            if (rvalid && rready) begin
                rvalid <= 0;
                rlast <= 0;
                if (read_remaining != 0) begin
                    read_pending <= 1;
                    read_address <= read_address + 8;
                    read_remaining <= read_remaining - 1'b1;
                end
            end

            if (awvalid && awready) begin
                pending_write_address <= awaddr;
                pending_write_remaining <= awlen;
                write_pending <= 1;
            end
            if (wvalid && wready) begin
                if (!write_pending)
                    $fatal(1, "write data without address");
                if (wlast != (pending_write_remaining == 0))
                    $fatal(1, "write last mismatch");
                if (pending_write_address >= KV_BASE &&
                    pending_write_address < KV_BASE + 30'd16384)
                    kv_mem[(pending_write_address - KV_BASE) >> 3] <= wdata;
                else if (pending_write_address >= OUTPUT_BASE &&
                         pending_write_address < OUTPUT_BASE + 30'd8192)
                    output_mem[(pending_write_address - OUTPUT_BASE) >> 3] <=
                        wdata;
                else
                    $fatal(1, "unexpected write address %08x",
                           pending_write_address);
                if (pending_write_remaining == 0) begin
                    write_pending <= 0;
                    response_pending <= 1;
                end else begin
                    pending_write_address <= pending_write_address + 8;
                    pending_write_remaining <= pending_write_remaining - 1'b1;
                end
            end
            if (response_pending && !bvalid) begin
                response_pending <= 0;
                bvalid <= 1;
                bid <= {5{1'b1}};
                bresp <= 0;
            end
            if (bvalid && bready) bvalid <= 0;
        end
    end

    bonsai_attention #(.KV_BASE(KV_BASE)) dut (
        .clk_i(clk), .rst_ni(rst_n),
        .start_i(start),
        .raw_base_i(RAW_BASE), .output_base_i(OUTPUT_BASE),
        .table_base_i(TABLE_BASE), .layer_i(5'd0),
        .position_i(position_cfg),
        .busy_o(busy), .done_o(done), .error_o(error),
        .arid_o(arid), .araddr_o(araddr), .arlen_o(arlen),
        .arsize_o(arsize), .arburst_o(arburst), .arvalid_o(arvalid),
        .arready_i(arready), .rid_i(rid), .rdata_i(rdata),
        .rresp_i(rresp), .rlast_i(rlast), .rvalid_i(rvalid),
        .rready_o(rready), .awid_o(awid), .awaddr_o(awaddr),
        .awlen_o(awlen),
        .awvalid_o(awvalid), .awready_i(awready), .wdata_o(wdata),
        .wstrb_o(wstrb), .wlast_o(wlast), .wvalid_o(wvalid),
        .wready_i(wready), .bid_i(bid), .bresp_i(bresp),
        .bvalid_i(bvalid), .bready_o(bready),
        .packer_start_o(packer_start),
        .packer_source_data_o(packer_source_data),
        .packer_source_valid_o(packer_source_valid),
        .packer_source_ready_i(packer_source_ready),
        .packer_block_data_i(packer_block_data),
        .packer_block_valid_i(packer_block_valid),
        .packer_block_ready_o(packer_block_ready)
    );

    initial begin
        for (i = 0; i < 2048; i = i + 1) raw_mem[i] = 0;
        for (i = 0; i < 192; i = i + 1) table_mem[i] = 0;
        for (i = 0; i < 2048; i = i + 1) kv_mem[i] = 0;
        for (i = 0; i < 1024; i = i + 1) output_mem[i] = 0;
        // Q/K norm weights and RoPE cosines are 1.0 in Q16. Sines are zero.
        for (i = 0; i < 128; i = i + 1) begin
            table_mem[i >> 1][(i & 1) * 32 +: 32] = 32'd65536;
            table_mem[(128 + i) >> 1][((128 + i) & 1) * 32 +: 32] =
                32'd65536;
        end
        for (i = 0; i < 64; i = i + 1)
            table_mem[(256 + i) >> 1][((256 + i) & 1) * 32 +: 32] =
                32'd65536;
        // Every Q and K lane starts at +1.0. This exercises RMS normalization
        // and produces identical cached keys for position zero.
        for (i = 0; i < 1536; i = i + 1)
            raw_mem[i] = 64'h3f80_0000_3f80_0000;
        // Position zero V alternates exactly representable +0.25 and -0.125.
        for (i = 1536; i < 2048; i = i + 1)
            raw_mem[i] = 64'hbe00_0000_3e80_0000;

        repeat (8) @(posedge clk);
        rst_n <= 1;
        repeat (4) @(posedge clk);
        start <= 1;
        @(posedge clk);
        start <= 0;
        cycle_count = 0;
        while (!done && cycle_count < 1000000) begin
            @(posedge clk);
            cycle_count = cycle_count + 1;
        end
        if (!done) $fatal(1, "attention timeout state=%0d", dut.state);
        if (error) $fatal(1, "attention reported AXI error");
        for (block_index = 0; block_index < 64;
             block_index = block_index + 1) begin
            byte_index = block_index * 34;
            if (output_mem[byte_index >> 3][(byte_index & 7) * 8 +: 8]
                    !== 8'h08 ||
                output_mem[(byte_index + 1) >> 3]
                          [((byte_index + 1) & 7) * 8 +: 8] !== 8'h18)
                $fatal(1, "packed scale mismatch block=%0d", block_index);
            for (i = 0; i < 32; i = i + 1)
                if (output_mem[(byte_index + 2 + i) >> 3]
                              [((byte_index + 2 + i) & 7) * 8 +: 8]
                        !== (i[0] ? 8'hc1 : 8'h7f))
                    $fatal(1, "packed value mismatch block=%0d item=%0d",
                           block_index, i);
        end
        $display("PASS bonsai attention position-zero identity cycles=%0d",
                 cycle_count);

        // Flip four lanes in each new K head. Against the all-positive Q this
        // gives scores 741504 and 695160 in Q16, so the second softmax weight
        // is exactly 32312. The independent integer reference therefore
        // expects +21794/-10897 Q16 after mixing the two cached V vectors.
        repeat (3) @(posedge clk);
        for (i = 0; i < 8; i = i + 1) begin
            raw_mem[1024 + i * 64] = 64'hbf80_0000_bf80_0000;
            raw_mem[1025 + i * 64] = 64'hbf80_0000_bf80_0000;
        end
        for (i = 1536; i < 2048; i = i + 1)
            raw_mem[i] = 64'hbe80_0000_3f00_0000;
        for (i = 0; i < 1024; i = i + 1) output_mem[i] = 0;
        position_cfg = 1;
        start = 1;
        @(posedge clk);
        start = 0;
        cycle_count = 0;
        while (!done && cycle_count < 1000000) begin
            @(posedge clk);
            cycle_count = cycle_count + 1;
        end
        if (!done) $fatal(1, "position-one timeout state=%0d", dut.state);
        if (error) $fatal(1, "position-one AXI error");
        for (block_index = 0; block_index < 64;
             block_index = block_index + 1) begin
            byte_index = block_index * 34;
            if (output_mem[byte_index >> 3][(byte_index & 7) * 8 +: 8]
                    !== 8'h5c ||
                output_mem[(byte_index + 1) >> 3]
                          [((byte_index + 1) & 7) * 8 +: 8] !== 8'h19)
                $fatal(1, "position-one packed scale mismatch block=%0d",
                       block_index);
            for (i = 0; i < 32; i = i + 1)
                if (output_mem[(byte_index + 2 + i) >> 3]
                              [((byte_index + 2 + i) & 7) * 8 +: 8]
                        !== (i[0] ? 8'hc1 : 8'h7f))
                    $fatal(1,
                           "position-one packed value mismatch block=%0d item=%0d",
                           block_index, i);
        end
        $display("PASS bonsai attention nonzero softmax cycles=%0d",
                 cycle_count);

        // Replaying an identical command must overwrite every packed output
        // word identically. This catches state which is initialized only by
        // global reset rather than at each attention start.
        for (i = 0; i < 272; i = i + 1)
            replay_reference[i] = output_mem[i];
        repeat (3) @(posedge clk);
        start = 1;
        @(posedge clk);
        start = 0;
        cycle_count = 0;
        while (!done && cycle_count < 1000000) begin
            @(posedge clk);
            cycle_count = cycle_count + 1;
        end
        if (!done) $fatal(1, "attention replay timeout state=%0d", dut.state);
        if (error) $fatal(1, "attention replay AXI error");
        for (i = 0; i < 272; i = i + 1)
            if (output_mem[i] !== replay_reference[i])
                $fatal(1,
                       "attention replay mismatch word=%0d first=%016x second=%016x",
                       i, replay_reference[i], output_mem[i]);
        $display("PASS bonsai attention consecutive replay cycles=%0d",
                 cycle_count);

        // Add a third, distinct key/value pair.  Two-token coverage is not
        // sufficient to catch score-bank or token-index pipeline errors: the
        // expected weights are 65536, 32312, and 15931, respectively.
        repeat (3) @(posedge clk);
        for (i = 0; i < 8; i = i + 1) begin
            raw_mem[1026 + i * 64] = 64'hbf80_0000_bf80_0000;
            raw_mem[1027 + i * 64] = 64'hbf80_0000_bf80_0000;
        end
        for (i = 1536; i < 2048; i = i + 1)
            raw_mem[i] = 64'hbf00_0000_3f80_0000;
        for (i = 0; i < 1024; i = i + 1) output_mem[i] = 0;
        position_cfg = 2;
        start = 1;
        @(posedge clk);
        start = 0;
        cycle_count = 0;
        while (!done && cycle_count < 1000000) begin
            @(posedge clk);
            cycle_count = cycle_count + 1;
        end
        if (!done) $fatal(1, "position-two timeout state=%0d", dut.state);
        if (error) $fatal(1, "position-two AXI error");
        if (dut.denominator !== 64'd113779)
            $fatal(1, "position-two denominator mismatch got=%0d expected=113779",
                   dut.denominator);
        if (dut.sums0[0] !== 64'sd3176595456 ||
            dut.sums1[0] !== -64'sd1588297728 ||
            dut.sums2[0] !== 64'sd3176595456 ||
            dut.sums3[0] !== -64'sd1588297728)
            $fatal(1,
                   "position-two weighted sum mismatch got=%0d,%0d,%0d,%0d",
                   dut.sums0[0], dut.sums1[0], dut.sums2[0], dut.sums3[0]);
        $display("PASS bonsai attention three-token softmax cycles=%0d",
                 cycle_count);

        // The validated CPU cache semantics bound transformed K values to
        // +/-256 before binary16 storage. Exercise that edge explicitly so a
        // later cleanup cannot reintroduce token drift on large norm weights.
        repeat (3) @(posedge clk);
        table_mem[(128 + 50) >> 1][((128 + 50) & 1) * 32 +: 32] =
            32'sd33554432;
        table_mem[(128 + 114) >> 1][((128 + 114) & 1) * 32 +: 32] =
            32'sd33554432;
        position_cfg = 3;
        start = 1;
        @(posedge clk);
        start = 0;
        cycle_count = 0;
        while (!done && cycle_count < 1000000) begin
            @(posedge clk);
            cycle_count = cycle_count + 1;
        end
        if (!done) $fatal(1, "K-clamp timeout state=%0d", dut.state);
        if (error) $fatal(1, "K-clamp AXI error");
        for (i = 0; i < 8; i = i + 1) begin
            if (kv_mem[(3 * 4096 + i * 256 + 50 * 2) >> 3]
                      [((50 * 2) & 7) * 8 +: 16] !== 16'h5c00 ||
                kv_mem[(3 * 4096 + i * 256 + 114 * 2) >> 3]
                      [((114 * 2) & 7) * 8 +: 16] !== 16'h5c00)
                $fatal(1, "K-clamp mismatch head=%0d", i);
        end
        $display("PASS bonsai attention K-cache compatibility clamp cycles=%0d",
                 cycle_count);
        $finish;
    end
endmodule

`timescale 1ns/1ps

module bonsai_ffn_packer_long_tb;
    logic clk = 0, fp_clk = 0, rst_n = 0, start = 0;
    logic [63:0] source_data = 64'h00000000_3f800000;
    logic source_valid = 0, source_ready;
    logic [271:0] block_data;
    logic block_valid, done;
    integer sent = 0, blocks = 0;

    bonsai_ffn_packer dut (
        .clk_i(clk), .fp_clk_i(fp_clk), .rst_ni(rst_n),
        .fp_rst_ni(rst_n), .start_i(start), .norm_mode_i(1'b0),
        .vector_mode_i(1'b0),
        .compact_source_i(1'b0),
        .element_count_i(13'd6144), .norm_scale_i(32'h3f800000),
        .source_data_i(source_data), .source_valid_i(source_valid),
        .source_ready_o(source_ready), .block_data_o(block_data),
        .block_valid_o(block_valid), .block_ready_i(1'b1), .busy_o(),
        .done_o(done));

    always #2.5 clk = ~clk;
    always #5.555 fp_clk = ~fp_clk;

    initial begin
        repeat (5) @(posedge clk);
        rst_n = 1;
        @(posedge clk); start = 1;
        @(posedge clk); start = 0; source_valid = 1;
    end

    always @(posedge clk) begin
        if (source_valid && source_ready) begin
            sent <= sent + 1;
            if (sent == 12287) source_valid <= 0;
        end
        if (block_valid) begin
            if (block_data[15:0] !== 16'h1de5)
                $fatal(1, "scale block %0d got %04x", blocks,
                       block_data[15:0]);
            for (integer byte_index = 0; byte_index < 32; ++byte_index)
                if (block_data[16 + byte_index*8 +: 8] !== 8'h7f)
                    $fatal(1, "q block %0d byte %0d got %02x", blocks,
                           byte_index,
                           block_data[16 + byte_index*8 +: 8]);
            blocks <= blocks + 1;
        end
        if (done) begin
            if (blocks != 192) $fatal(1, "only %0d blocks", blocks);
            $display("PASS bonsai long FFN CDC stream");
            $finish;
        end
    end

    initial begin
        repeat (10000000) @(posedge clk);
        $fatal(1, "timeout sent=%0d blocks=%0d", sent, blocks);
    end
endmodule

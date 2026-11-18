`timescale 1ns/1ps

module bonsai_ffn_packer_tb;
    logic clk;
    logic fp_clk;
    logic rst_n;
    logic start;
    logic [63:0] source_data;
    logic source_valid;
    logic source_ready;
    logic [271:0] block_data;
    logic block_valid;
    logic done;
    logic [31:0] up [0:31];
    logic [31:0] gate [0:31];
    logic [7:0] expected [0:33];

    bonsai_ffn_packer #(.ELEMENTS(32)) dut (
        .clk_i(clk),
        .fp_clk_i(fp_clk),
        .rst_ni(rst_n),
        .fp_rst_ni(rst_n),
        .start_i(start),
        .norm_mode_i(1'b0),
        .vector_mode_i(1'b0),
        .compact_source_i(1'b1),
        .wide_source_i(1'b0),
        .element_count_i(13'd32),
        .norm_scale_i(32'h3f80_0000),
        .source_data_i(source_data),
        .source_valid_i(source_valid),
        .source_ready_o(source_ready),
        .block_data_o(block_data),
        .block_valid_o(block_valid),
        .block_ready_i(1'b1),
        .busy_o(),
        .done_o(done)
    );

    always #2.5 clk = ~clk;
    always #4 fp_clk = ~fp_clk;

    task automatic send_pair(input logic [31:0] low,
                             input logic [31:0] high);
        begin
            @(negedge clk);
            source_data = {high, low};
            source_valid = 1;
            while (!source_ready) @(negedge clk);
            @(posedge clk);
        end
    endtask

    initial begin
        clk = 0;
        fp_clk = 0;
        rst_n = 0;
        start = 0;
        source_valid = 0;
        source_data = 0;
        up = '{32'hc0000000, 32'hbff00000, 32'hbfe00000, 32'hbfd00000,
               32'hbfc00000, 32'hbfb00000, 32'hbfa00000, 32'hbf900000,
               32'hbf800000, 32'hbf600000, 32'hbf400000, 32'hbf200000,
               32'hbf000000, 32'hbec00000, 32'hbe800000, 32'hbe000000,
               32'h00000000, 32'h3e000000, 32'h3e800000, 32'h3ec00000,
               32'h3f000000, 32'h3f200000, 32'h3f400000, 32'h3f600000,
               32'h3f800000, 32'h3f900000, 32'h3fa00000, 32'h3fb00000,
               32'h3fc00000, 32'h3fd00000, 32'h3fe00000, 32'h3ff00000};
        gate = '{32'hc06aaaab, 32'hbfaaaaab, 32'h3f800000, 32'h40555555,
                 32'hc0000000, 32'h3eaaaaab, 32'h402aaaab, 32'hc02aaaab,
                 32'hbeaaaaab, 32'h40000000, 32'hc0555555, 32'hbf800000,
                 32'h3faaaaab, 32'h406aaaab, 32'hbfd55555, 32'h3f2aaaab,
                 32'h40400000, 32'hc0155555, 32'h00000000, 32'h40155555,
                 32'hc0400000, 32'hbf2aaaab, 32'h3fd55555, 32'hc06aaaab,
                 32'hbfaaaaab, 32'h3f800000, 32'h40555555, 32'hc0000000,
                 32'h3eaaaaab, 32'h402aaaab, 32'hc02aaaab, 32'hbeaaaaab};
        expected = '{8'h45, 8'h29, 8'h04, 8'h0d, 8'he1, 8'h81,
                     8'h09, 8'hfa, 8'hb4, 8'h05, 8'h03, 8'hdb, 8'h02,
                     8'h04, 8'hf3, 8'hdf, 8'h02, 8'hff, 8'h00, 8'hff,
                     8'h00, 8'h13, 8'hfe, 8'hfd, 8'h1a, 8'hfe, 8'hf9,
                     8'h14, 8'h62, 8'hf8, 8'h07, 8'h62, 8'hf9, 8'hfa};
        repeat (5) @(posedge clk);
        rst_n <= 1;
        @(posedge clk);
        start <= 1;
        @(posedge clk);
        start <= 0;
        for (integer i = 0; i < 32; i += 2) send_pair(up[i], up[i + 1]);
        for (integer i = 0; i < 32; i += 2)
            send_pair(gate[i], gate[i + 1]);
        @(negedge clk);
        source_valid = 0;
    end

    always @(posedge clk) begin
        if (block_valid) begin
        for (integer byte_index = 0; byte_index < 34; byte_index++) begin
            if (block_data[byte_index*8 +: 8] !== expected[byte_index])
                $fatal(1, "byte[%0d] mismatch: got=%02x expected=%02x",
                       byte_index, block_data[byte_index*8 +: 8],
                       expected[byte_index]);
        end
        $display("PASS bonsai_ffn_packer exact diverse block");
        $finish;
        end
    end

    initial begin
        repeat (25000) @(posedge clk);
        $fatal(1, "timeout state=%0d element=%0d block=%0d fp_step=%0d waiting=%0d in_ready=%0d out_valid=%0d",
               dut.state, dut.element_index, dut.block_position, dut.fp_step,
               dut.fp_waiting, dut.fp_input_ready, dut.fp_result_valid);
    end
endmodule

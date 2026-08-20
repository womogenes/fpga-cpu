`timescale 1ns/1ps

module bonsai_norm_packer_tb;
    logic clk = 0;
    logic fp_clk = 0;
    logic rst_n = 0;
    logic start = 0;
    logic [63:0] source_data = 0;
    logic source_valid = 0;
    logic source_ready;
    logic [271:0] block_data;
    logic block_valid;
    logic done;
    logic [31:0] input_values [0:31];

    bonsai_ffn_packer #(.ELEMENTS(32)) dut (
        .clk_i(clk),
        .fp_clk_i(fp_clk),
        .rst_ni(rst_n),
        .fp_rst_ni(rst_n),
        .start_i(start),
        .norm_mode_i(1'b1),
        .vector_mode_i(1'b0),
        .compact_source_i(1'b0),
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
    always #5.555 fp_clk = ~fp_clk;

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
        input_values = '{32'hc0000000, 32'hbff00000, 32'hbfe00000,
                         32'hbfd00000, 32'hbfc00000, 32'hbfb00000,
                         32'hbfa00000, 32'hbf900000, 32'hbf800000,
                         32'hbf600000, 32'hbf400000, 32'hbf200000,
                         32'hbf000000, 32'hbec00000, 32'hbe800000,
                         32'hbe000000, 32'h00000000, 32'h3e000000,
                         32'h3e800000, 32'h3ec00000, 32'h3f000000,
                         32'h3f200000, 32'h3f400000, 32'h3f600000,
                         32'h3f800000, 32'h3f900000, 32'h3fa00000,
                         32'h3fb00000, 32'h3fc00000, 32'h3fd00000,
                         32'h3fe00000, 32'h3ff00000};
        repeat (5) @(posedge clk);
        rst_n = 1;
        @(posedge clk);
        start = 1;
        @(posedge clk);
        start = 0;
        for (integer i = 0; i < 32; i += 2)
            send_pair(input_values[i], input_values[i + 1]);
        for (integer i = 0; i < 32; i += 2)
            send_pair(32'h3f80_0000, 32'h3f80_0000);
        @(negedge clk);
        source_valid = 0;
    end

    always @(posedge clk) begin
        if (block_valid) begin
            byte signed expected [0:31];
            expected = '{-127, -119, -111, -103, -95, -87, -79, -71,
                         -63, -56, -48, -40, -32, -24, -16, -8,
                         0, 8, 16, 24, 32, 40, 48, 56,
                         63, 71, 79, 87, 95, 103, 111, 119};
            if (block_data[15:0] !== 16'h2408)
                $fatal(1, "scale mismatch: got=%04x expected=2408",
                       block_data[15:0]);
            for (integer i = 0; i < 32; i++)
                if ($signed(block_data[16 + i*8 +: 8]) !== expected[i])
                    $fatal(1, "q[%0d] mismatch: got=%0d expected=%0d", i,
                           $signed(block_data[16 + i*8 +: 8]), expected[i]);
            $display("PASS bonsai norm apply and exact Q8_0 packing");
            $finish;
        end
    end

    initial begin
        repeat (5000) @(posedge clk);
        $fatal(1, "timeout state=%0d element=%0d step=%0d",
               dut.state, dut.element_index, dut.fp_step);
    end
endmodule

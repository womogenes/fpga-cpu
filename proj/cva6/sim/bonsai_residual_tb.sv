`timescale 1ns/1ps

module bonsai_residual_tb;
    logic clk = 0;
    logic fp_clk = 0;
    logic rst_n = 0;
    logic start = 0;
    logic [63:0] source_data = 0;
    logic source_valid = 0;
    logic source_ready;
    logic [63:0] result_data;
    logic result_valid;
    logic result_ready = 0;
    logic [15:0] ready_lfsr = 16'h1;
    logic [31:0] sum;
    logic sum_valid;
    logic [7:0] sum_tag;
    logic busy;
    logic done;
    logic [3:0] debug_state;
    logic [31:0] matvec_values [0:15];
    logic [31:0] residual_values [0:15];
    logic [31:0] expected_values [0:15];
    integer output_words = 0;
    integer timeout;

    always #2.5 clk = ~clk;
    always #5.555 fp_clk = ~fp_clk;

    // Exercise the result mailbox with deterministic bursts of AXI-writer
    // backpressure instead of accepting every word immediately.
    always_ff @(posedge clk) begin
        if (!rst_n) begin
            ready_lfsr <= 16'h1;
            result_ready <= 1'b0;
        end else begin
            ready_lfsr <= {ready_lfsr[14:0],
                           ready_lfsr[15] ^ ready_lfsr[13] ^
                           ready_lfsr[12] ^ ready_lfsr[10]};
            result_ready <= ready_lfsr[2:0] != 3'b000;
        end
    end

    bonsai_residual #(.ELEMENTS(2048)) dut (
        .clk_i(clk),
        .fp_clk_i(fp_clk),
        .rst_ni(rst_n),
        .fp_rst_ni(rst_n),
        .start_i(start),
        .source_data_i(source_data),
        .source_valid_i(source_valid),
        .source_ready_o(source_ready),
        .result_data_o(result_data),
        .result_valid_o(result_valid),
        .result_ready_i(result_ready),
        .sum_o(sum),
        .sum_valid_o(sum_valid),
        .sum_tag_o(sum_tag),
        .busy_o(busy),
        .done_o(done),
        .debug_state_o(debug_state)
    );

    task automatic send_word(input logic [63:0] value);
        begin
            @(negedge clk);
            source_data = value;
            source_valid = 1'b1;
            do @(posedge clk); while (!source_ready);
            @(negedge clk);
            source_valid = 1'b0;
        end
    endtask

    always @(posedge clk) begin
        if (result_valid && result_ready) begin
            if (output_words < 1024) begin
                if (result_data[31:0] !==
                        expected_values[(output_words * 2) % 16] ||
                    result_data[63:32] !==
                        expected_values[(output_words * 2 + 1) % 16])
                    $fatal(1, "residual output word %0d mismatch: %016x",
                           output_words, result_data);
            end else if (output_words == 1024) begin
                if (result_data !== 64'h0000_0000_4930_b60d)
                    $fatal(1, "residual streamed sum mismatch: %016x",
                           result_data);
            end else begin
                $fatal(1, "unexpected residual output word %0d: %016x",
                       output_words, result_data);
            end
            output_words <= output_words + 1;
        end
    end

    initial begin
        // These nontrivial binary32 values cover signs, cancellation, large
        // exponent gaps, and round-to-nearest boundaries. The expected words
        // and two-half sum were generated with separately rounded FP32 add,
        // multiply, and accumulation operations, matching the CPU path.
        matvec_values = '{32'h3f000001, 32'hbf7fffff, 32'h41234567,
                          32'hc0200000, 32'h3a83126f, 32'hba03126f,
                          32'h3eaaaaab, 32'hbe4ccccd, 32'h42c80001,
                          32'hc2480001, 32'h358637bd, 32'hb58637bd,
                          32'h3f7fffff, 32'h40000001, 32'hc0400001,
                          32'h3dcccccd};
        residual_values = '{32'h3e800001, 32'h3f000001, 32'hc0a00001,
                            32'h3fa00001, 32'h3b03126f, 32'h3a83126f,
                            32'hbe2aaaab, 32'h3e99999a, 32'hc1c80001,
                            32'h42480000, 32'h360637bd, 32'h350637bd,
                            32'hbf000001, 32'hc0000000, 32'h40400000,
                            32'hbd4ccccd};
        expected_values = '{32'h3f400002, 32'hbefffffc, 32'h40a68acd,
                            32'hbf9fffff, 32'h3b449ba6, 32'h3a03126f,
                            32'h3e2aaaab, 32'h3dccccce, 32'h42960001,
                            32'hb6800000, 32'h3649539c, 32'hb50637bd,
                            32'h3efffffc, 32'h34800000, 32'hb4800000,
                            32'h3d4ccccd};
        repeat (8) @(posedge clk);
        rst_n <= 1'b1;
        repeat (8) @(posedge clk);
        start <= 1'b1;
        @(posedge clk);
        start <= 1'b0;

        for (integer word = 0; word < 1024; word = word + 1)
            send_word({matvec_values[(word * 2 + 1) % 16],
                       matvec_values[(word * 2) % 16]});
        for (integer word = 0; word < 1024; word = word + 1)
            send_word({residual_values[(word * 2 + 1) % 16],
                       residual_values[(word * 2) % 16]});

        timeout = 0;
        while (!done && timeout < 2000000) begin
            @(posedge clk);
            timeout = timeout + 1;
        end
        if (!done)
            $fatal(1, "residual engine timed out in state %0d word %0d",
                   debug_state, dut.word_index);
        if (output_words != 1025)
            $fatal(1, "residual output count mismatch: %0d", output_words);
        if (sum !== 32'h4930b60d)
            $fatal(1, "residual sum mismatch: %08x", sum);
        $display("PASS: mixed FP32 fused residual add/sum (%0d cycles)",
                 timeout);
        $finish;
    end
endmodule

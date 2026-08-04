`timescale 1ns/1ps

module lfm_matvec_accelerator_tb;

    localparam logic [29:0] MODEL_BASE = 30'h3000_0000;
    localparam integer MEMORY_BYTES = 4 * 1024 * 1024;
    localparam integer INPUT_OFFSET = 32'h0000_0000;
    // Match the 0xae0 page offset of blk.1.ffn_down.weight, the first
    // production tensor whose hardware result diverged.
    localparam integer WEIGHT_OFFSET = 32'h0000_2ae0;
    localparam integer OUTPUT_OFFSET = 32'h0030_0000;

    logic clk = 1'b0;
    logic rst_n = 1'b0;
    logic [31:0] command = '0;
    logic [31:0] status;
    logic request;
    logic grant;
    logic toggle = 1'b0;
    byte unsigned memory [0:MEMORY_BYTES-1];

    AXI_BUS #(
        .AXI_ADDR_WIDTH(64),
        .AXI_DATA_WIDTH(64),
        .AXI_ID_WIDTH(5),
        .AXI_USER_WIDTH(1)
    ) axi();

    always #2.5 clk = ~clk;
    // Match the real arbiter's registered grant/release behavior.
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            grant <= 1'b0;
        else
            grant <= request;
    end

    lfm_matvec_accelerator #(
        .ID_WIDTH(5)
    ) dut (
        .clk_i(clk),
        .rst_ni(rst_n),
        .command_i(command),
        .status_o(status),
        .bus_request_o(request),
        .bus_grant_i(grant),
        .axi(axi)
    );

    logic read_active;
    logic [29:0] read_address;
    logic [8:0] read_beats;
    logic aw_have;
    logic [29:0] aw_address;
    logic w_have;
    logic [63:0] w_data;

    assign axi.ar_ready = !read_active;
    assign axi.aw_ready = !aw_have && !axi.b_valid;
    assign axi.w_ready = !w_have && !axi.b_valid;
    assign axi.b_id = 5'h1f;
    assign axi.b_resp = 2'b00;
    assign axi.b_user = '0;
    assign axi.r_id = 5'h1f;
    assign axi.r_resp = 2'b00;
    assign axi.r_user = '0;

    integer i;
    integer memory_index;
    always_ff @(posedge clk) begin
        if (!rst_n) begin
            read_active <= 1'b0;
            read_address <= '0;
            read_beats <= '0;
            axi.r_data <= '0;
            axi.r_last <= 1'b0;
            axi.r_valid <= 1'b0;
            aw_have <= 1'b0;
            aw_address <= '0;
            w_have <= 1'b0;
            w_data <= '0;
            axi.b_valid <= 1'b0;
        end else begin
            if (axi.ar_valid && axi.ar_ready) begin
                read_active <= 1'b1;
                read_address <= axi.ar_addr[29:0];
                read_beats <= {1'b0, axi.ar_len} + 1'b1;
            end
            if (read_active && !axi.r_valid) begin
                memory_index = read_address - MODEL_BASE;
                for (i = 0; i < 8; i = i + 1)
                    axi.r_data[i*8 +: 8] <= memory[memory_index + i];
                axi.r_last <= read_beats == 1;
                axi.r_valid <= 1'b1;
            end
            if (axi.r_valid && axi.r_ready) begin
                if (axi.r_last) begin
                    axi.r_valid <= 1'b0;
                    read_active <= 1'b0;
                end else begin
                    read_address <= read_address + 8;
                    read_beats <= read_beats - 1'b1;
                    memory_index = read_address + 8 - MODEL_BASE;
                    for (i = 0; i < 8; i = i + 1)
                        axi.r_data[i*8 +: 8] <= memory[memory_index + i];
                    axi.r_last <= read_beats == 2;
                    axi.r_valid <= 1'b1;
                end
            end

            if (axi.aw_valid && axi.aw_ready) begin
                aw_have <= 1'b1;
                aw_address <= axi.aw_addr[29:0];
            end
            if (axi.w_valid && axi.w_ready) begin
                w_have <= 1'b1;
                w_data <= axi.w_data;
            end
            if (aw_have && w_have && !axi.b_valid) begin
                memory_index = aw_address - MODEL_BASE;
                for (i = 0; i < 8; i = i + 1)
                    memory[memory_index + i] <= w_data[i*8 +: 8];
                aw_have <= 1'b0;
                w_have <= 1'b0;
                axi.b_valid <= 1'b1;
            end
            if (axi.b_valid && axi.b_ready)
                axi.b_valid <= 1'b0;
        end
    end

    function automatic signed [63:0] fp16_product_q30(
        input logic [15:0] a,
        input logic [15:0] b
    );
        integer ea;
        integer eb;
        integer fa;
        integer fb;
        integer pa;
        integer pb;
        integer shift;
        longint signed product;
        longint signed bias;
        begin
            ea = (a >> 10) & 31;
            eb = (b >> 10) & 31;
            fa = a & 1023;
            fb = b & 1023;
            if ((ea == 0 && fa == 0) || (eb == 0 && fb == 0)) begin
                fp16_product_q30 = 0;
            end else begin
                product = (ea == 0 ? fa : 1024 + fa) *
                          (eb == 0 ? fb : 1024 + fb);
                if ((a ^ b) & 16'h8000)
                    product = -product;
                pa = ea == 0 ? -24 : ea - 25;
                pb = eb == 0 ? -24 : eb - 25;
                shift = pa + pb + 30;
                if (shift >= 0) begin
                    fp16_product_q30 = product <<< shift;
                end else begin
                    bias = 64'sd1 <<< (-shift - 1);
                    fp16_product_q30 = product >= 0 ?
                        (product + bias) >>> -shift :
                        -((-product + bias) >>> -shift);
                end
            end
        end
    endfunction

    task automatic write_u16(input integer address, input logic [15:0] value);
        begin
            memory[address] = value[7:0];
            memory[address + 1] = value[15:8];
        end
    endtask

    task automatic send_register(input logic [2:0] index, input logic [24:0] value);
        begin
            toggle = ~toggle;
            command = {toggle, 3'b010, index, value};
            repeat (8) @(posedge clk);
        end
    endtask

    task automatic wait_done(input logic previous_toggle);
        integer timeout;
        begin
            timeout = 0;
            while (status[28] == previous_toggle && timeout < 20000000) begin
                @(posedge clk);
                timeout = timeout + 1;
            end
            if (timeout == 20000000)
                $fatal(1, "accelerator timeout state=%0d status=%08x", dut.state, status);
            if (request || grant)
                $fatal(1, "DONE became visible before DDR was released");
            if (status[29])
                $fatal(1, "accelerator reported an error");
        end
    endtask

    task automatic run_q4_test;
        integer block;
        integer row;
        integer j;
        integer address;
        integer packed_q;
        integer qlo;
        integer qhi;
        integer ilo;
        integer ihi;
        integer dot;
        longint signed expected;
        longint signed actual;
        logic [15:0] input_d;
        logic [15:0] weight_d;
        logic old_done;
        begin
            for (block = 0; block < 4; block = block + 1) begin
                address = INPUT_OFFSET + block * 34;
                input_d = 16'h3000 + block * 16'h0040;
                write_u16(address, input_d);
                for (j = 0; j < 32; j = j + 1)
                    memory[address + 2 + j] = byte'(((block * 37 + j * 11) % 255) - 127);
            end
            for (row = 0; row < 2; row = row + 1) begin
                for (block = 0; block < 4; block = block + 1) begin
                    address = WEIGHT_OFFSET + (row * 4 + block) * 18;
                    weight_d = 16'h2c00 + row * 16'h0100 + block * 16'h0020;
                    write_u16(address, weight_d);
                    for (j = 0; j < 16; j = j + 1) begin
                        qlo = (row * 3 + block * 5 + j * 7) & 15;
                        qhi = (row * 11 + block * 2 + j * 3) & 15;
                        memory[address + 2 + j] = (qhi << 4) | qlo;
                    end
                end
            end

            old_done = status[28];
            send_register(0, WEIGHT_OFFSET >> 3);
            send_register(1, INPUT_OFFSET >> 3);
            send_register(2, OUTPUT_OFFSET >> 3);
            send_register(3, 128);
            send_register(4, 2);
            send_register(5, 1);
            wait_done(old_done);

            for (row = 0; row < 2; row = row + 1) begin
                expected = 0;
                for (block = 0; block < 4; block = block + 1) begin
                    dot = 0;
                    address = WEIGHT_OFFSET + (row * 4 + block) * 18;
                    for (j = 0; j < 16; j = j + 1) begin
                        packed_q = memory[address + 2 + j];
                        qlo = (packed_q & 15) - 8;
                        qhi = (packed_q >> 4) - 8;
                        ilo = $signed(memory[INPUT_OFFSET + block * 34 + 2 + j]);
                        ihi = $signed(memory[INPUT_OFFSET + block * 34 + 18 + j]);
                        dot = dot + qlo * ilo + qhi * ihi;
                    end
                    weight_d = {memory[address + 1], memory[address]};
                    input_d = {memory[INPUT_OFFSET + block * 34 + 1],
                               memory[INPUT_OFFSET + block * 34]};
                    expected = expected + dot * fp16_product_q30(weight_d, input_d);
                end
                actual = 0;
                for (j = 0; j < 8; j = j + 1)
                    actual[j*8 +: 8] = memory[OUTPUT_OFFSET + row * 8 + j];
                if (actual !== expected)
                    $fatal(1, "Q4 row %0d mismatch: got %0d expected %0d", row, actual, expected);
            end
            $display("PASS: Q4_0 exact Q30 matrix-vector output");
        end
    endtask

    task automatic run_q4_large_input_test;
        integer block;
        integer row;
        integer j;
        integer address;
        integer packed_q;
        integer qlo;
        integer qhi;
        integer ilo;
        integer ihi;
        integer dot;
        longint signed expected;
        longint signed actual;
        logic [15:0] input_d;
        logic [15:0] weight_d;
        logic old_done;
        begin
            for (block = 0; block < 144; block = block + 1) begin
                address = INPUT_OFFSET + block * 34;
                input_d = 16'h2800 + (block % 32) * 16'h0020;
                write_u16(address, input_d);
                for (j = 0; j < 32; j = j + 1)
                    memory[address + 2 + j] = byte'(((block * 37 + j * 11) % 255) - 127);
            end
            for (row = 0; row < 1024; row = row + 1) begin
                for (block = 0; block < 144; block = block + 1) begin
                    address = WEIGHT_OFFSET + (row * 144 + block) * 18;
                    // Keep the synthetic scale product within signed Q30's
                    // [-2, 2) range. Row-dependent packed weights still vary
                    // every dot product and exercise the production-sized
                    // 1024x4608 stream that exposed the hardware failure.
                    weight_d = 16'h2400 + (row % 8) * 16'h0040 +
                               (block % 32) * 16'h0010;
                    write_u16(address, weight_d);
                    for (j = 0; j < 16; j = j + 1) begin
                        qlo = (row * 3 + block * 5 + j * 7) & 15;
                        qhi = (row * 11 + block * 2 + j * 3) & 15;
                        memory[address + 2 + j] = (qhi << 4) | qlo;
                    end
                end
            end

            old_done = status[28];
            send_register(0, WEIGHT_OFFSET >> 3);
            send_register(1, INPUT_OFFSET >> 3);
            send_register(2, OUTPUT_OFFSET >> 3);
            send_register(3, 4608);
            send_register(4, 1024);
            send_register(5, 1);
            wait_done(old_done);

            for (row = 0; row < 1024; row = row + 1) begin
                expected = 0;
                for (block = 0; block < 144; block = block + 1) begin
                    dot = 0;
                    address = WEIGHT_OFFSET + (row * 144 + block) * 18;
                    for (j = 0; j < 16; j = j + 1) begin
                        packed_q = memory[address + 2 + j];
                        qlo = (packed_q & 15) - 8;
                        qhi = (packed_q >> 4) - 8;
                        ilo = $signed(memory[INPUT_OFFSET + block * 34 + 2 + j]);
                        ihi = $signed(memory[INPUT_OFFSET + block * 34 + 18 + j]);
                        dot = dot + qlo * ilo + qhi * ihi;
                    end
                    weight_d = {memory[address + 1], memory[address]};
                    input_d = {memory[INPUT_OFFSET + block * 34 + 1],
                               memory[INPUT_OFFSET + block * 34]};
                    expected = expected + dot * fp16_product_q30(weight_d, input_d);
                end
                actual = 0;
                for (j = 0; j < 8; j = j + 1)
                    actual[j*8 +: 8] = memory[OUTPUT_OFFSET + row * 8 + j];
                if (actual !== expected)
                    $fatal(1, "large Q4 row %0d mismatch: got %0d expected %0d", row, actual, expected);
            end
            $display("PASS: Q4_0 exact 4608-element output");
        end
    endtask

    task automatic run_q6_test;
        integer block;
        integer row;
        integer j;
        integer address;
        integer group;
        integer k;
        integer half;
        integer l;
        integer ql_base;
        integer qh_base;
        integer base;
        integer sg;
        integer q0;
        integer q1;
        integer q2;
        integer q3;
        integer part;
        integer isum;
        integer qvalue;
        longint signed score;
        longint signed expected_best_score;
        integer expected_token;
        logic [15:0] input_d;
        logic [15:0] weight_d;
        logic old_done;
        begin
            for (block = 0; block < 4; block = block + 1) begin
                address = INPUT_OFFSET + block * 258;
                input_d = 16'h2e00 + block * 16'h0060;
                write_u16(address, input_d);
                for (j = 0; j < 256; j = j + 1)
                    memory[address + 2 + j] = byte'(((block * 41 + j * 13) % 255) - 127);
            end
            for (row = 0; row < 3; row = row + 1) begin
                for (block = 0; block < 4; block = block + 1) begin
                    address = WEIGHT_OFFSET + (row * 4 + block) * 210;
                    for (j = 0; j < 128; j = j + 1)
                        memory[address + j] = (row * 19 + block * 23 + j * 7) & 8'hff;
                    for (j = 0; j < 64; j = j + 1)
                        memory[address + 128 + j] = (row * 31 + block * 11 + j * 5) & 8'hff;
                    for (j = 0; j < 16; j = j + 1)
                        memory[address + 192 + j] = byte'(((row * 17 + block * 7 + j * 9) % 127) - 63);
                    weight_d = 16'h2a00 + row * 16'h0080 + block * 16'h0010;
                    write_u16(address + 208, weight_d);
                end
            end

            expected_best_score = 64'sh8000_0000_0000_0000;
            expected_token = 0;
            for (row = 0; row < 3; row = row + 1) begin
                score = 0;
                for (block = 0; block < 4; block = block + 1) begin
                    address = WEIGHT_OFFSET + (row * 4 + block) * 210;
                    isum = 0;
                    for (half = 0; half < 2; half = half + 1) begin
                        ql_base = half * 64;
                        qh_base = half * 32;
                        base = half * 128;
                        for (l = 0; l < 32; l = l + 1) begin
                            sg = half * 8 + l / 16;
                            q0 = ((memory[address + ql_base + l] & 15) |
                                  ((memory[address + 128 + qh_base + l] & 3) << 4)) - 32;
                            q1 = ((memory[address + ql_base + 32 + l] & 15) |
                                  (((memory[address + 128 + qh_base + l] >> 2) & 3) << 4)) - 32;
                            q2 = ((memory[address + ql_base + l] >> 4) |
                                  (((memory[address + 128 + qh_base + l] >> 4) & 3) << 4)) - 32;
                            q3 = ((memory[address + ql_base + 32 + l] >> 4) |
                                  (((memory[address + 128 + qh_base + l] >> 6) & 3) << 4)) - 32;
                            qvalue = $signed(memory[INPUT_OFFSET + block * 258 + 2 + base + l]);
                            part = q0 * qvalue * $signed(memory[address + 192 + sg]);
                            qvalue = $signed(memory[INPUT_OFFSET + block * 258 + 2 + base + l + 32]);
                            part = part + q1 * qvalue * $signed(memory[address + 192 + sg + 2]);
                            qvalue = $signed(memory[INPUT_OFFSET + block * 258 + 2 + base + l + 64]);
                            part = part + q2 * qvalue * $signed(memory[address + 192 + sg + 4]);
                            qvalue = $signed(memory[INPUT_OFFSET + block * 258 + 2 + base + l + 96]);
                            part = part + q3 * qvalue * $signed(memory[address + 192 + sg + 6]);
                            isum = isum + part;
                        end
                    end
                    weight_d = {memory[address + 209], memory[address + 208]};
                    input_d = {memory[INPUT_OFFSET + block * 258 + 1],
                               memory[INPUT_OFFSET + block * 258]};
                    score = score + isum * fp16_product_q30(weight_d, input_d);
                end
                if (score > expected_best_score) begin
                    expected_best_score = score;
                    expected_token = row;
                end
            end

            old_done = status[28];
            send_register(0, WEIGHT_OFFSET >> 3);
            send_register(1, INPUT_OFFSET >> 3);
            send_register(2, OUTPUT_OFFSET >> 3);
            send_register(3, 1024);
            send_register(4, 3);
            send_register(5, 2);
            wait_done(old_done);
            if (status[23:8] !== expected_token[15:0])
                $fatal(1, "Q6 argmax mismatch: got %0d expected %0d", status[23:8], expected_token);
            if (dut.best_score !== expected_best_score)
                $fatal(1, "Q6 score mismatch: got %0d expected %0d", dut.best_score, expected_best_score);
            $display("PASS: Q6_K exact Q30 argmax output");
        end
    endtask

    initial begin
        for (i = 0; i < MEMORY_BYTES; i = i + 1)
            memory[i] = 0;
        repeat (10) @(posedge clk);
        rst_n = 1'b1;
        repeat (5) @(posedge clk);
        if (!status[31])
            $fatal(1, "accelerator presence bit is absent");
        run_q4_test();
        run_q4_large_input_test();
        run_q4_test();
        run_q4_large_input_test();
        run_q4_test();
        run_q4_large_input_test();
        run_q6_test();
        run_q4_test();
        run_q6_test();
        $display("PASS: LFM accelerator regression complete");
        $finish;
    end

endmodule

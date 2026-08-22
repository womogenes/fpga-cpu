`timescale 1ns/1ps

module lfm_matvec_accelerator_tb;
    function automatic logic [32:0] round_integer_to_fp32_precision(
        input logic [31:0] magnitude);
        logic [32:0] significand;
        logic [31:0] low_mask;
        integer msb;
        integer shift;
        integer bit_index;
        begin
            msb = 0;
            for (bit_index = 0; bit_index < 32; bit_index = bit_index + 1)
                if (magnitude[bit_index]) msb = bit_index;
            if (magnitude == 0 || msb <= 23) begin
                round_integer_to_fp32_precision = {1'b0, magnitude};
            end else begin
                shift = msb - 23;
                significand = {1'b0, magnitude >> shift};
                low_mask = shift > 1 ?
                    ((32'd1 << (shift - 1)) - 32'd1) : 32'd0;
                if (magnitude[shift - 1] &&
                    (|(magnitude & low_mask) || significand[0]))
                    significand = significand + 33'd1;
                round_integer_to_fp32_precision = significand << shift;
            end
        end
    endfunction

    function automatic logic [31:0] q30_float_reference(
        input logic signed [63:0] value);
        logic sign;
        logic [64:0] magnitude;
        logic [65:0] significand;
        logic [64:0] low_mask;
        logic signed [32:0] rounded_high;
        logic [32:0] rounded_low;
        logic signed [64:0] split_value;
        integer msb;
        integer shift;
        integer exponent;
        integer bit_index;
        begin
            rounded_high = value[63] ?
                -$signed(round_integer_to_fp32_precision(
                    ~value[63:32] + 32'd1)) :
                 $signed(round_integer_to_fp32_precision(value[63:32]));
            rounded_low = round_integer_to_fp32_precision(value[31:0]);
            split_value =
                ($signed({{32{rounded_high[32]}}, rounded_high}) <<< 32) +
                $signed({32'b0, rounded_low});
            if (split_value == 0) begin
                q30_float_reference = 32'b0;
            end else begin
                sign = split_value[64];
                magnitude = sign ? (~$unsigned(split_value) + 65'd1) :
                                   $unsigned(split_value);
                msb = 0;
                for (bit_index = 0; bit_index < 65;
                     bit_index = bit_index + 1)
                    if (magnitude[bit_index]) msb = bit_index;
                exponent = msb + 97;
                significand = '0;
                if (msb <= 23) begin
                    significand = {1'b0, magnitude} << (23 - msb);
                end else begin
                    shift = msb - 23;
                    significand = {1'b0, magnitude >> shift};
                    low_mask = shift > 1 ?
                        ((65'd1 << (shift - 1)) - 65'd1) : 65'd0;
                    if (magnitude[shift - 1] &&
                        (|(magnitude & low_mask) || significand[0]))
                        significand = significand + 66'd1;
                    if (significand[24]) begin
                        significand = significand >> 1;
                        exponent = exponent + 1;
                    end
                end
                q30_float_reference =
                    {sign, exponent[7:0], significand[22:0]};
            end
        end
    endfunction

    localparam logic [29:0] MODEL_BASE = 30'h2f00_0000;
    localparam integer MEMORY_BYTES = 4 * 1024 * 1024;
    localparam integer INPUT_OFFSET = 32'h0000_0000;
    // Match the 0xae0 page offset of blk.1.ffn_down.weight, the first
    // production tensor whose hardware result diverged.
    localparam integer WEIGHT_OFFSET = 32'h0000_2ae0;
    localparam integer OUTPUT_OFFSET = 32'h0030_0000;
    localparam integer RESIDUAL_SUM_OFFSET = 32'h0031_0000;
    localparam logic [29:0] RESIDUAL_SUM_ADDRESS = 30'h3e28_0000;

    logic clk = 1'b0;
    logic fp_clk = 1'b0;
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
    always #5.555 fp_clk = ~fp_clk;
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
        .fp_clk_i(fp_clk),
        .rst_ni(rst_n),
        .fp_rst_ni(rst_n),
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
                memory_index = read_address == RESIDUAL_SUM_ADDRESS ?
                    RESIDUAL_SUM_OFFSET : read_address - MODEL_BASE;
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
                    memory_index = read_address + 8 == RESIDUAL_SUM_ADDRESS ?
                        RESIDUAL_SUM_OFFSET : read_address + 8 - MODEL_BASE;
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
                memory_index = aw_address == RESIDUAL_SUM_ADDRESS ?
                    RESIDUAL_SUM_OFFSET : aw_address - MODEL_BASE;
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

    task automatic run_residual_test;
        integer element;
        integer byte_index;
        integer pass;
        logic old_done;
        logic [31:0] status_sum;
        begin
            for (element = 0; element < 2048; element = element + 1) begin
                for (byte_index = 0; byte_index < 4;
                     byte_index = byte_index + 1) begin
                    memory[INPUT_OFFSET + element * 4 + byte_index] =
                        (32'h3f80_0000 >> (byte_index * 8)) & 8'hff;
                    memory[INPUT_OFFSET + 8192 + element * 4 + byte_index] =
                        (32'h4000_0000 >> (byte_index * 8)) & 8'hff;
                    memory[OUTPUT_OFFSET + element * 4 + byte_index] = 0;
                end
            end
            for (byte_index = 0; byte_index < 8; byte_index = byte_index + 1)
                memory[RESIDUAL_SUM_OFFSET + byte_index] = 0;
            old_done = status[28];
            send_register(1, INPUT_OFFSET >> 3);
            send_register(2, OUTPUT_OFFSET >> 3);
            send_register(3, 2048);
            send_register(4, 5);
            send_register(6, 0);
            send_register(5, 7);
            wait_done(old_done);
            if (status[29]) $fatal(1, "residual command failed");
            status_sum = {14'b0, status[25:8]};
            send_register(6, 25'h100_0000);
            status_sum[31:18] = status[21:8];
            if (status_sum !== 32'h4690_0000)
                $fatal(1, "residual status sum mismatch: %08x", status_sum);
            for (element = 0; element < 2048; element = element + 1)
                for (byte_index = 0; byte_index < 4;
                     byte_index = byte_index + 1)
                    if (memory[OUTPUT_OFFSET + element * 4 + byte_index] !==
                        ((32'h4040_0000 >> (byte_index * 8)) & 8'hff))
                        $fatal(1, "residual element %0d byte %0d mismatch",
                               element, byte_index);
            for (byte_index = 0; byte_index < 4; byte_index = byte_index + 1)
                if (memory[RESIDUAL_SUM_OFFSET + byte_index] !==
                    ((32'h4690_0000 >> (byte_index * 8)) & 8'hff))
                    $fatal(1, "residual sum byte %0d mismatch", byte_index);

            // Start a different residual command immediately. The residual
            // done level from the first command can remain synchronized high
            // for several AXI clocks; it must not publish the preceding sum.
            for (element = 0; element < 2048; element = element + 1) begin
                for (byte_index = 0; byte_index < 4;
                     byte_index = byte_index + 1) begin
                    memory[INPUT_OFFSET + element * 4 + byte_index] =
                        (32'h3f80_0000 >> (byte_index * 8)) & 8'hff;
                    memory[INPUT_OFFSET + 8192 + element * 4 + byte_index] =
                        (32'h3f80_0000 >> (byte_index * 8)) & 8'hff;
                    memory[OUTPUT_OFFSET + element * 4 + byte_index] = 0;
                end
            end
            old_done = status[28];
            send_register(6, 0);
            send_register(5, 7);
            wait_done(old_done);
            if (status[29]) $fatal(1, "second residual command failed");
            status_sum = {14'b0, status[25:8]};
            send_register(6, 25'h100_0000);
            status_sum[31:18] = status[21:8];
            if (status_sum !== 32'h4600_0000)
                $fatal(1, "second residual status sum mismatch: %08x",
                       status_sum);
            for (element = 0; element < 2048; element = element + 1)
                for (byte_index = 0; byte_index < 4;
                     byte_index = byte_index + 1)
                    if (memory[OUTPUT_OFFSET + element * 4 + byte_index] !==
                        ((32'h4000_0000 >> (byte_index * 8)) & 8'hff))
                        $fatal(1,
                               "second residual element %0d byte %0d mismatch",
                               element, byte_index);
            for (byte_index = 0; byte_index < 4; byte_index = byte_index + 1)
                if (memory[RESIDUAL_SUM_OFFSET + byte_index] !==
                    ((32'h4600_0000 >> (byte_index * 8)) & 8'hff))
                    $fatal(1, "second residual sum byte %0d mismatch",
                           byte_index);

            // Exercise enough immediate back-to-back commands to cover
            // repeated CDC phase relationships.  The scalar now follows the
            // vector through the acknowledged result mailbox, so every DONE
            // must leave the current command's sum in memory.
            for (pass = 0; pass < 16; pass = pass + 1) begin
                old_done = status[28];
                send_register(6, 0);
                send_register(5, 7);
                wait_done(old_done);
                if (status[29])
                    $fatal(1, "residual command %0d failed", pass + 3);
                status_sum = {14'b0, status[25:8]};
                send_register(6, 25'h100_0000);
                status_sum[31:18] = status[21:8];
                if (status_sum !== 32'h4600_0000)
                    $fatal(1,
                           "residual command %0d status sum mismatch: %08x",
                           pass + 3, status_sum);
                for (byte_index = 0; byte_index < 4;
                     byte_index = byte_index + 1)
                    if (memory[RESIDUAL_SUM_OFFSET + byte_index] !==
                        ((32'h4600_0000 >> (byte_index * 8)) & 8'hff))
                        $fatal(1,
                               "residual command %0d sum byte %0d mismatch",
                               pass + 3, byte_index);
            end
            $display("PASS: consecutive integrated residual add/sum (%0d cycles)",
                     dut.cycle_count);
        end
    endtask

    task automatic send_register(input logic [2:0] index, input logic [24:0] value);
        begin
            toggle = ~toggle;
            // Model worst-case bus skew at the CDC boundary: the toggle can
            // arrive one destination clock before the payload.  The DUT must
            // not acknowledge or consume the command until the payload has
            // had an additional settling cycle.
            command[31] = toggle;
            @(posedge clk);
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
                $fatal(1, "accelerator timeout state=%0d status=%08x row=%0d block=%0d position=%0d bytes=%0d reader_busy=%0d reader_done=%0d",
                       dut.state, status, dut.row_index, dut.row_block,
                       dut.weight_position, dut.weight_byte_index,
                       dut.reader_busy, dut.reader_done);
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

    task automatic run_q1_test;
        integer block;
        integer input_block_index;
        integer row;
        integer group;
        integer j;
        integer address;
        integer bit_index;
        integer bit_value;
        integer input_value;
        integer dot;
        longint signed expected;
        longint signed actual;
        logic [15:0] input_d;
        logic [15:0] weight_d;
        logic old_done;
        begin
            for (input_block_index = 0; input_block_index < 8;
                 input_block_index = input_block_index + 1) begin
                address = INPUT_OFFSET + input_block_index * 34;
                input_d = 16'h2a00 + input_block_index * 16'h0030;
                write_u16(address, input_d);
                for (j = 0; j < 32; j = j + 1)
                    memory[address + 2 + j] =
                        byte'(((input_block_index * 43 + j * 17) % 255) - 127);
            end
            for (row = 0; row < 3; row = row + 1) begin
                for (block = 0; block < 2; block = block + 1) begin
                    address = WEIGHT_OFFSET + (row * 2 + block) * 18;
                    weight_d = 16'h2800 + row * 16'h0080 + block * 16'h0020;
                    write_u16(address, weight_d);
                    for (j = 0; j < 16; j = j + 1)
                        memory[address + 2 + j] =
                            (row * 71 + block * 29 + j * 53) & 8'hff;
                end
            end

            old_done = status[28];
            send_register(0, WEIGHT_OFFSET >> 3);
            send_register(1, INPUT_OFFSET >> 3);
            send_register(2, OUTPUT_OFFSET >> 3);
            send_register(3, 256);
            send_register(4, 3);
            send_register(5, 3);
            wait_done(old_done);

            for (row = 0; row < 3; row = row + 1) begin
                expected = 0;
                for (block = 0; block < 2; block = block + 1) begin
                    address = WEIGHT_OFFSET + (row * 2 + block) * 18;
                    weight_d = {memory[address + 1], memory[address]};
                    for (group = 0; group < 4; group = group + 1) begin
                        dot = 0;
                        input_block_index = block * 4 + group;
                        for (j = 0; j < 32; j = j + 1) begin
                            bit_index = group * 32 + j;
                            bit_value = (memory[address + 2 + bit_index / 8] >>
                                         (bit_index % 8)) & 1;
                            input_value = $signed(memory[INPUT_OFFSET +
                                input_block_index * 34 + 2 + j]);
                            dot = dot + (bit_value ? input_value : -input_value);
                        end
                        input_d = {memory[INPUT_OFFSET + input_block_index * 34 + 1],
                                   memory[INPUT_OFFSET + input_block_index * 34]};
                        expected = expected +
                            dot * fp16_product_q30(weight_d, input_d);
                    end
                end
                actual = 0;
                for (j = 0; j < 8; j = j + 1)
                    actual[j*8 +: 8] = memory[OUTPUT_OFFSET + row * 8 + j];
                if (actual[63:32] !== 32'b0 ||
                    actual[31:0] !== q30_float_reference(expected))
                    $fatal(1, "Q1 row %0d float mismatch: got %h expected %h",
                           row, actual[31:0], q30_float_reference(expected));
            end
            $display("PASS: Q1_0 correctly rounded FP32 matrix-vector output");
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

    // Larger Q1 runs covering multi-block rows, negative and zero scales,
    // and the single-block-per-row case that stresses the row-result FIFO
    // backpressure of the streaming Q1 pipeline.
    task automatic run_q1_stress(input integer ne0, input integer ne1);
        integer block;
        integer input_block_index;
        integer row;
        integer group;
        integer j;
        integer k;
        integer address;
        integer bit_index;
        integer bit_value;
        integer input_value;
        integer dot;
        longint signed expected;
        longint signed actual;
        longint signed expected_best;
        integer expected_token;
        integer group_address;
        integer block_in_group;
        logic [7:0] packed_bytes [0:71];
        logic [15:0] input_d;
        logic [15:0] weight_d;
        logic old_done;
        begin
            for (input_block_index = 0; input_block_index < ne0 / 32;
                 input_block_index = input_block_index + 1) begin
                address = INPUT_OFFSET + input_block_index * 34;
                input_d = 16'h2c00 + ((input_block_index * 37) % 512);
                if (input_block_index % 5 == 3)
                    input_d = input_d | 16'h8000;
                write_u16(address, input_d);
                for (j = 0; j < 32; j = j + 1)
                    memory[address + 2 + j] =
                        byte'(((input_block_index * 61 + j * 13) % 255) - 127);
            end
            expected_best = 64'sh8000_0000_0000_0000;
            expected_token = 0;
            for (row = 0; row < ne1; row = row + 1) begin
                for (block = 0; block < ne0 / 128; block = block + 1) begin
                    address = WEIGHT_OFFSET + (row * (ne0 / 128) + block) * 18;
                    weight_d = 16'h2600 + ((row * 97 + block * 41) % 1024);
                    if ((row + block) % 3 == 1)
                        weight_d = weight_d | 16'h8000;
                    if (row == 0 && block == 0)
                        weight_d = 16'h0000;
                    if (ne1 > 65536 && row + 1 == ne1)
                        weight_d = 16'h3800;
                    write_u16(address, weight_d);
                    for (j = 0; j < 16; j = j + 1) begin
                        if (ne1 > 65536 && row + 1 == ne1) begin
                            memory[address + 2 + j] = 0;
                            for (k = 0; k < 8; k = k + 1) begin
                                bit_index = j * 8 + k;
                                input_block_index = block * 4 + bit_index / 32;
                                input_value = $signed(memory[INPUT_OFFSET +
                                    input_block_index * 34 + 2 + bit_index % 32]);
                                if (input_value >= 0)
                                    memory[address + 2 + j] =
                                        memory[address + 2 + j] | (1 << k);
                            end
                        end else begin
                            memory[address + 2 + j] =
                                (row * 19 + block * 7 + j * 111) & 8'hff;
                        end
                    end
                end
            end

            old_done = status[28];
            send_register(0, WEIGHT_OFFSET >> 3);
            send_register(1, INPUT_OFFSET >> 3);
            send_register(2, OUTPUT_OFFSET >> 3);
            send_register(3, ne0[24:0]);
            send_register(4, ne1[24:0]);
            send_register(5, 3);
            wait_done(old_done);

            for (row = 0; row < ne1; row = row + 1) begin
                expected = 0;
                for (block = 0; block < ne0 / 128; block = block + 1) begin
                    address = WEIGHT_OFFSET + (row * (ne0 / 128) + block) * 18;
                    weight_d = {memory[address + 1], memory[address]};
                    for (group = 0; group < 4; group = group + 1) begin
                        dot = 0;
                        input_block_index = block * 4 + group;
                        for (j = 0; j < 32; j = j + 1) begin
                            bit_index = group * 32 + j;
                            bit_value = (memory[address + 2 + bit_index / 8] >>
                                         (bit_index % 8)) & 1;
                            input_value = $signed(memory[INPUT_OFFSET +
                                input_block_index * 34 + 2 + j]);
                            dot = dot + (bit_value ? input_value : -input_value);
                        end
                        input_d = {memory[INPUT_OFFSET + input_block_index * 34 + 1],
                                   memory[INPUT_OFFSET + input_block_index * 34]};
                        expected = expected +
                            dot * fp16_product_q30(weight_d, input_d);
                    end
                end
                actual = 0;
                for (j = 0; j < 8; j = j + 1)
                    actual[j*8 +: 8] = memory[OUTPUT_OFFSET + row * 8 + j];
                if (actual[63:32] !== 32'b0 ||
                    actual[31:0] !== q30_float_reference(expected))
                    $fatal(1, "Q1 stress %0dx%0d row %0d float mismatch: got %h expected %h",
                           ne0, ne1, row, actual[31:0],
                           q30_float_reference(expected));
                if (expected > expected_best) begin
                    expected_best = expected;
                    expected_token = row;
                end
            end
            $display("PASS: Q1_0 FP32 %0dx%0d stress output (%0d cycles, %0d weights)",
                     ne0, ne1, dut.cycle_count, ne0 * ne1);

            // Compact FP32 output packs two consecutive rows into each AXI
            // beat. Exercise both the address calculation and low/high word
            // ordering before the packed-weight rewrite below changes the
            // source matrix in place.
            if (ne1 <= 2048 && (ne1 & 1) == 0) begin
                for (j = 0; j < ne1 * 8 + 8; j = j + 1)
                    memory[OUTPUT_OFFSET + j] = 8'ha5;
                old_done = status[28];
                send_register(0, WEIGHT_OFFSET >> 3);
                send_register(1, INPUT_OFFSET >> 3);
                send_register(2, OUTPUT_OFFSET >> 3);
                send_register(3, ne0[24:0]);
                send_register(4, ne1[24:0]);
                send_register(6, 64);
                send_register(5, 3);
                wait_done(old_done);
                for (row = 0; row < ne1; row = row + 1) begin
                    expected = 0;
                    for (block = 0; block < ne0 / 128;
                         block = block + 1) begin
                        address = WEIGHT_OFFSET +
                            (row * (ne0 / 128) + block) * 18;
                        weight_d = {memory[address + 1], memory[address]};
                        for (group = 0; group < 4; group = group + 1) begin
                            dot = 0;
                            input_block_index = block * 4 + group;
                            for (j = 0; j < 32; j = j + 1) begin
                                bit_index = group * 32 + j;
                                bit_value =
                                    (memory[address + 2 + bit_index / 8] >>
                                     (bit_index % 8)) & 1;
                                input_value = $signed(memory[INPUT_OFFSET +
                                    input_block_index * 34 + 2 + j]);
                                dot = dot +
                                    (bit_value ? input_value : -input_value);
                            end
                            input_d = {memory[INPUT_OFFSET +
                                             input_block_index * 34 + 1],
                                       memory[INPUT_OFFSET +
                                             input_block_index * 34]};
                            expected = expected +
                                dot * fp16_product_q30(weight_d, input_d);
                        end
                    end
                    actual = 0;
                    for (j = 0; j < 4; j = j + 1)
                        actual[j*8 +: 8] =
                            memory[OUTPUT_OFFSET + row * 4 + j];
                    if (actual[31:0] !== q30_float_reference(expected))
                        $fatal(1,
                               "compact Q1 %0dx%0d row %0d mismatch: got %h expected %h",
                               ne0, ne1, row, actual[31:0],
                               q30_float_reference(expected));
                end
                for (j = 0; j < 8; j = j + 1)
                    if (memory[OUTPUT_OFFSET + ne1 * 4 + j] !== 8'ha5)
                        $fatal(1,
                               "compact Q1 %0dx%0d wrote beyond packed output at byte %0d",
                               ne0, ne1, j);
                $display("PASS: compact Q1_0 FP32 %0dx%0d output (%0d cycles)",
                         ne0, ne1, dut.cycle_count);
                send_register(6, 0);
            end

            old_done = status[28];
            send_register(0, WEIGHT_OFFSET >> 3);
            send_register(1, INPUT_OFFSET >> 3);
            send_register(2, OUTPUT_OFFSET >> 3);
            send_register(3, ne0[24:0]);
            send_register(4, ne1[24:0]);
            send_register(5, 4);
            wait_done(old_done);
            if (status[25:8] !== expected_token[17:0])
                $fatal(1, "Q1 argmax %0dx%0d mismatch: got %0d expected %0d",
                       ne0, ne1, status[25:8], expected_token);
            if (dut.best_score !== expected_best)
                $fatal(1, "Q1 argmax score mismatch: got %0d expected %0d",
                       dut.best_score, expected_best);
            $display("PASS: Q1_0 exact %0dx%0d hardware argmax token %0d",
                     ne0, ne1, expected_token);

            if (ne0 % 512 == 0 && ne1 <= 96) begin
                for (row = 0; row < ne1; row = row + 1) begin
                    for (group = 0; group < ne0 / 512; group = group + 1) begin
                        group_address = WEIGHT_OFFSET +
                            (row * (ne0 / 128) + group * 4) * 18;
                        for (block = 0; block < 4; block = block + 1) begin
                            packed_bytes[block * 2] =
                                memory[group_address + block * 18];
                            packed_bytes[block * 2 + 1] =
                                memory[group_address + block * 18 + 1];
                            for (j = 0; j < 16; j = j + 1)
                                packed_bytes[8 + block * 16 + j] =
                                    memory[group_address + block * 18 + 2 + j];
                        end
                        for (j = 0; j < 72; j = j + 1)
                            memory[group_address + j] = packed_bytes[j];
                    end
                end

                old_done = status[28];
                send_register(0, WEIGHT_OFFSET >> 3);
                send_register(1,
                    ((MODEL_BASE + INPUT_OFFSET) >> 3) & 25'h1ffffff);
                send_register(6,
                    4 | (((MODEL_BASE + INPUT_OFFSET) >> 28) & 3) |
                    32 | ((((MODEL_BASE + OUTPUT_OFFSET) >> 28) & 3) << 3));
                send_register(2,
                    ((MODEL_BASE + OUTPUT_OFFSET) >> 3) & 25'h1ffffff);
                send_register(3, ne0[24:0]);
                send_register(4, ne1[24:0]);
                send_register(5, 5);
                wait_done(old_done);

                expected_best = 64'sh8000_0000_0000_0000;
                expected_token = 0;
                for (row = 0; row < ne1; row = row + 1) begin
                    expected = 0;
                    for (block = 0; block < ne0 / 128;
                         block = block + 1) begin
                        group = block / 4;
                        block_in_group = block % 4;
                        group_address = WEIGHT_OFFSET +
                            (row * (ne0 / 128) + group * 4) * 18;
                        weight_d = {
                            memory[group_address + block_in_group * 2 + 1],
                            memory[group_address + block_in_group * 2]};
                        for (j = 0; j < 4; j = j + 1) begin
                            dot = 0;
                            input_block_index = block * 4 + j;
                            for (k = 0; k < 32; k = k + 1) begin
                                bit_index = j * 32 + k;
                                bit_value =
                                    (memory[group_address + 8 +
                                            block_in_group * 16 + bit_index / 8] >>
                                     (bit_index % 8)) & 1;
                                input_value = $signed(memory[INPUT_OFFSET +
                                    input_block_index * 34 + 2 + k]);
                                dot = dot +
                                    (bit_value ? input_value : -input_value);
                            end
                            input_d = {memory[INPUT_OFFSET +
                                             input_block_index * 34 + 1],
                                       memory[INPUT_OFFSET +
                                             input_block_index * 34]};
                            expected = expected +
                                dot * fp16_product_q30(weight_d, input_d);
                        end
                    end
                    actual = 0;
                    for (j = 0; j < 8; j = j + 1)
                        actual[j*8 +: 8] =
                            memory[OUTPUT_OFFSET + row * 8 + j];
                    if (actual[63:32] !== 32'b0 ||
                        actual[31:0] !== q30_float_reference(expected))
                        $fatal(1,
                               "packed Q1 %0dx%0d row %0d float mismatch: got %h expected %h",
                               ne0, ne1, row, actual[31:0],
                               q30_float_reference(expected));
                    if (expected > expected_best) begin
                        expected_best = expected;
                        expected_token = row;
                    end
                end
                $display("PASS: packed Q1_0 FP32 %0dx%0d output (%0d cycles)",
                         ne0, ne1, dut.cycle_count);

                old_done = status[28];
                send_register(0, WEIGHT_OFFSET >> 3);
                send_register(1, INPUT_OFFSET >> 3);
                send_register(2, OUTPUT_OFFSET >> 3);
                send_register(3, ne0[24:0]);
                send_register(4, ne1[24:0]);
                send_register(5, 6);
                wait_done(old_done);
                if (status[25:8] !== expected_token[17:0] ||
                    dut.best_score !== expected_best)
                    $fatal(1, "packed Q1 argmax mismatch");
                $display("PASS: packed Q1_0 exact argmax token %0d",
                         expected_token);
                send_register(1, INPUT_OFFSET >> 3);
                send_register(6, 0);
            end
        end
    endtask

    task automatic run_ffn_pack_test;
        integer element;
        integer block;
        integer byte_index;
        logic old_done;
        begin
            for (element = 0; element < 12288; element = element + 1) begin
                memory[INPUT_OFFSET + element * 8 + 0] = 8'h00;
                memory[INPUT_OFFSET + element * 8 + 1] = 8'h00;
                memory[INPUT_OFFSET + element * 8 + 2] = 8'h80;
                memory[INPUT_OFFSET + element * 8 + 3] = 8'h3f;
                memory[INPUT_OFFSET + element * 8 + 4] = 8'h00;
                memory[INPUT_OFFSET + element * 8 + 5] = 8'h00;
                memory[INPUT_OFFSET + element * 8 + 6] = 8'h00;
                memory[INPUT_OFFSET + element * 8 + 7] = 8'h00;
            end
            for (byte_index = 0; byte_index < 6528; byte_index = byte_index + 1)
                memory[OUTPUT_OFFSET + byte_index] = 0;
            old_done = status[28];
            send_register(1, INPUT_OFFSET >> 3);
            send_register(2, OUTPUT_OFFSET >> 3);
            send_register(3, 6144);
            send_register(4, 1);
            send_register(6, 0);
            send_register(5, 7);
            while (status[28] == old_done) @(posedge clk);
            if (status[29]) $fatal(1, "FFN pack command failed");
            for (block = 0; block < 192; block = block + 1) begin
                if (memory[OUTPUT_OFFSET + block * 34] !== 8'he5 ||
                    memory[OUTPUT_OFFSET + block * 34 + 1] !== 8'h1d)
                    $fatal(1, "FFN scale block %0d mismatch: got %02x%02x",
                           block, memory[OUTPUT_OFFSET + block * 34 + 1],
                           memory[OUTPUT_OFFSET + block * 34]);
                for (byte_index = 0; byte_index < 32; byte_index = byte_index + 1)
                    if (memory[OUTPUT_OFFSET + block * 34 + 2 + byte_index] !== 8'h7f)
                        $fatal(1, "FFN q block %0d byte %0d mismatch",
                               block, byte_index);
            end
            $display("PASS: FFN activation and exact Q8_0 packing (%0d cycles)",
                     dut.cycle_count);

            // The production path stores up and gate as consecutive packed
            // FP32 vectors. Each 64-bit source beat must feed both words to
            // the transform instead of discarding its high half.
            for (element = 0; element < 12288; element = element + 1) begin
                memory[INPUT_OFFSET + element * 4 + 0] = 8'h00;
                memory[INPUT_OFFSET + element * 4 + 1] = 8'h00;
                memory[INPUT_OFFSET + element * 4 + 2] = 8'h80;
                memory[INPUT_OFFSET + element * 4 + 3] = 8'h3f;
            end
            for (byte_index = 0; byte_index < 6528;
                 byte_index = byte_index + 1)
                memory[OUTPUT_OFFSET + byte_index] = 0;
            old_done = status[28];
            send_register(1, INPUT_OFFSET >> 3);
            send_register(2, OUTPUT_OFFSET >> 3);
            send_register(3, 6144);
            send_register(4, 1);
            send_register(6, 64);
            send_register(5, 7);
            while (status[28] == old_done) @(posedge clk);
            if (status[29]) $fatal(1, "compact FFN pack command failed");
            for (block = 0; block < 192; block = block + 1) begin
                if (memory[OUTPUT_OFFSET + block * 34] !== 8'he5 ||
                    memory[OUTPUT_OFFSET + block * 34 + 1] !== 8'h1d)
                    $fatal(1,
                           "compact FFN scale block %0d mismatch: got %02x%02x",
                           block, memory[OUTPUT_OFFSET + block * 34 + 1],
                           memory[OUTPUT_OFFSET + block * 34]);
                for (byte_index = 0; byte_index < 32;
                     byte_index = byte_index + 1)
                    if (memory[OUTPUT_OFFSET + block * 34 + 2 +
                               byte_index] !== 8'h7f)
                        $fatal(1,
                               "compact FFN q block %0d byte %0d mismatch",
                               block, byte_index);
            end
            $display("PASS: compact FFN source and exact Q8_0 packing (%0d cycles)",
                     dut.cycle_count);
            send_register(6, 0);
        end
    endtask

    task automatic run_norm_pack_test;
        integer element;
        integer block;
        integer byte_index;
        logic [31:0] value;
        logic [31:0] input_values [0:31];
        logic signed [7:0] expected [0:31];
        logic old_done;
        begin
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
            expected = '{-127, -119, -111, -103, -95, -87, -79, -71,
                         -63, -56, -48, -40, -32, -24, -16, -8,
                         0, 8, 16, 24, 32, 40, 48, 56,
                         63, 71, 79, 87, 95, 103, 111, 119};
            // Four Q8 blocks exercise the accelerator's 136-byte AXI write
            // grouping, matching the model's 2048-element norm commands.
            for (element = 0; element < 128; element = element + 2) begin
                value = input_values[element % 32];
                for (byte_index = 0; byte_index < 4; byte_index = byte_index + 1)
                    memory[INPUT_OFFSET + (element / 2) * 8 + byte_index] =
                        value[byte_index * 8 +: 8];
                value = input_values[(element + 1) % 32];
                for (byte_index = 0; byte_index < 4; byte_index = byte_index + 1)
                    memory[INPUT_OFFSET + (element / 2) * 8 + 4 + byte_index] =
                        value[byte_index * 8 +: 8];
                value = 32'h3f80_0000;
                for (byte_index = 0; byte_index < 4; byte_index = byte_index + 1)
                    memory[INPUT_OFFSET + (64 + element / 2) * 8 + byte_index] =
                        value[byte_index * 8 +: 8];
                for (byte_index = 0; byte_index < 4; byte_index = byte_index + 1)
                    memory[INPUT_OFFSET + (64 + element / 2) * 8 + 4 + byte_index] =
                        value[byte_index * 8 +: 8];
            end
            old_done = status[28];
            send_register(1, INPUT_OFFSET >> 3);
            send_register(2, OUTPUT_OFFSET >> 3);
            send_register(3, 128);
            send_register(4, 2);
            send_register(7, 32'h3f80_0000 & 25'h1ff_ffff);
            send_register(6, (32'h3f80_0000 >> 25) << 7);
            send_register(5, 7);
            wait_done(old_done);
            if (status[29]) $fatal(1, "norm pack command failed");
            for (block = 0; block < 4; block = block + 1) begin
                if (memory[OUTPUT_OFFSET + block * 34] !== 8'h08 ||
                    memory[OUTPUT_OFFSET + block * 34 + 1] !== 8'h24)
                    $fatal(1, "norm scale block %0d mismatch", block);
                for (byte_index = 0; byte_index < 32; byte_index = byte_index + 1)
                    if ($signed(memory[OUTPUT_OFFSET + block * 34 + 2 + byte_index]) !==
                        expected[byte_index])
                        $fatal(1, "norm block %0d q[%0d] mismatch",
                               block, byte_index);
            end
            $display("PASS: RMS norm apply and exact Q8_0 packing (%0d cycles)",
                     dut.cycle_count);

            // Vector mode consumes only the input vector and synthesizes
            // unit weight/scale internally. It must produce the identical
            // bytes while halving source traffic for attention packing.
            for (byte_index = 0; byte_index < 136; byte_index = byte_index + 1)
                memory[OUTPUT_OFFSET + byte_index] = 0;
            old_done = status[28];
            send_register(1, INPUT_OFFSET >> 3);
            send_register(2, OUTPUT_OFFSET >> 3);
            send_register(3, 128);
            send_register(4, 3);
            send_register(7, 32'h3f80_0000 & 25'h1ff_ffff);
            send_register(6, (32'h3f80_0000 >> 25) << 7);
            send_register(5, 7);
            wait_done(old_done);
            if (status[29]) $fatal(1, "vector pack command failed");
            for (block = 0; block < 4; block = block + 1) begin
                if (memory[OUTPUT_OFFSET + block * 34] !== 8'h08 ||
                    memory[OUTPUT_OFFSET + block * 34 + 1] !== 8'h24)
                    $fatal(1, "vector scale block %0d mismatch", block);
                for (byte_index = 0; byte_index < 32; byte_index = byte_index + 1)
                    if ($signed(memory[OUTPUT_OFFSET + block * 34 + 2 + byte_index]) !==
                        expected[byte_index])
                        $fatal(1, "vector block %0d q[%0d] mismatch",
                               block, byte_index);
            end
            $display("PASS: vector exact Q8_0 packing (%0d cycles)",
                     dut.cycle_count);
        end
    endtask

    task automatic run_watchdog_recovery_test;
        logic old_done;
        logic [17:0] expected_diagnostic;
        integer timeout;
        begin
            old_done = status[28];
            force dut.i_ffn_packer.fp_input_ready = 1'b0;
            send_register(1, INPUT_OFFSET >> 3);
            send_register(2, OUTPUT_OFFSET >> 3);
            send_register(3, 128);
            send_register(4, 2);
            send_register(7, 32'h3f80_0000 & 25'h1ff_ffff);
            send_register(6, (32'h3f80_0000 >> 25) << 7);
            send_register(5, 7);
            wait (dut.state == 6'd38);
            @(negedge clk);
            force dut.cycle_count = 64'h0000_0000_2000_0000;
            expected_diagnostic = {
                dut.state,
                dut.ffn_debug_state_sync,
                dut.reader_busy,
                dut.reader_done,
                dut.ffn_busy,
                dut.ffn_done,
                dut.ffn_group_drain,
                dut.ffn_block_valid,
                dut.wr_active,
                |dut.wr_outstanding
            };
            @(posedge clk);
            release dut.cycle_count;
            timeout = 0;
            while (status[28] == old_done && timeout < 100000) begin
                @(posedge clk);
                timeout = timeout + 1;
            end
            release dut.i_ffn_packer.fp_input_ready;
            if (timeout == 100000)
                $fatal(1, "watchdog did not release the DDR bus");
            if (!status[29] || status[25:8] != expected_diagnostic)
                $fatal(1, "watchdog diagnostic mismatch status=%08x", status);
            if (request || grant)
                $fatal(1, "watchdog exposed DONE before DDR release");
            $display("PASS: watchdog recovered stalled FFN_STREAM command");
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
        if (!status[26])
            $fatal(1, "accelerator v2 capability bit is absent");
        if (!status[0])
            $fatal(1, "accelerator absolute-output capability bit is absent");
        if (!status[1])
            $fatal(1, "accelerator FFN packing capability bit is absent");
        if (!status[2])
            $fatal(1, "accelerator norm packing capability bit is absent");
        run_q4_test();
        run_q4_large_input_test();
        run_q4_test();
        run_q4_large_input_test();
        run_q4_test();
        run_q4_large_input_test();
        run_q1_test();
        run_q6_test();
        run_q4_test();
        run_q1_test();
        run_q6_test();
        run_q1_stress(1024, 37);
        run_q1_stress(128, 64);
        run_q1_stress(2048, 96);
        run_q1_stress(6144, 96);
        run_q1_stress(128, 65538);
        run_q4_test();
        run_q6_test();
        run_ffn_pack_test();
        run_norm_pack_test();
        run_residual_test();
        run_watchdog_recovery_test();
        $display("PASS: LFM and Bonsai accelerator regression complete");
        $finish;
    end

endmodule

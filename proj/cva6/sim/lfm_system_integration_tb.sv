`timescale 1ns/1ps

// Integration regression: the exclusive DDR arbiter with the streaming
// accelerator on one side and continuous CPU traffic on the other. Verifies
// that a Q1 command issued while the CPU hammers DDR is granted, completes,
// toggles DONE, and releases the bus so the CPU resumes.
module lfm_system_integration_tb;

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

    always #2.5 clk = ~clk;

    AXI_BUS #(
        .AXI_ADDR_WIDTH(64), .AXI_DATA_WIDTH(64),
        .AXI_ID_WIDTH(5), .AXI_USER_WIDTH(1)
    ) cpu_bus();
    AXI_BUS #(
        .AXI_ADDR_WIDTH(64), .AXI_DATA_WIDTH(64),
        .AXI_ID_WIDTH(5), .AXI_USER_WIDTH(1)
    ) accel_bus();
    AXI_BUS #(
        .AXI_ADDR_WIDTH(64), .AXI_DATA_WIDTH(64),
        .AXI_ID_WIDTH(5), .AXI_USER_WIDTH(1)
    ) accel_cut_bus();
    AXI_BUS #(
        .AXI_ADDR_WIDTH(64), .AXI_DATA_WIDTH(64),
        .AXI_ID_WIDTH(5), .AXI_USER_WIDTH(1)
    ) ddr_bus();

    lfm_matvec_accelerator #(.ID_WIDTH(5)) dut (
        .clk_i(clk),
        .fp_clk_i(clk),
        .rst_ni(rst_n),
        .fp_rst_ni(rst_n),
        .command_i(command),
        .status_o(status),
        .bus_request_o(request),
        .bus_grant_i(grant),
        .axi(accel_bus)
    );

    // Mirror the deployed topology: a full register slice between the
    // accelerator and the exclusive arbiter.
    axi_cut_intf #(
        .BYPASS(1'b0),
        .ADDR_WIDTH(64),
        .DATA_WIDTH(64),
        .ID_WIDTH(5),
        .USER_WIDTH(1)
    ) i_cut (
        .clk_i(clk),
        .rst_ni(rst_n),
        .in(accel_bus),
        .out(accel_cut_bus)
    );

    lfm_axi_arbiter i_arbiter (
        .clk_i(clk),
        .rst_ni(rst_n),
        .accelerator_request_i(request),
        .accelerator_grant_o(grant),
        .cpu(cpu_bus),
        .accelerator(accel_cut_bus),
        .ddr(ddr_bus)
    );

    // ------------------------------------------------------------------
    // DDR memory model on the arbiter's DDR master port.
    logic read_active;
    logic [29:0] read_address;
    logic [8:0] read_beats;
    logic [4:0] read_id;
    logic aw_have;
    logic [29:0] aw_address;
    logic [4:0] aw_id;
    logic w_have;
    logic [63:0] w_data;
    logic [7:0] w_strb;

    assign ddr_bus.ar_ready = !read_active;
    assign ddr_bus.aw_ready = !aw_have && !ddr_bus.b_valid;
    assign ddr_bus.w_ready = !w_have && !ddr_bus.b_valid;
    assign ddr_bus.b_resp = 2'b00;
    assign ddr_bus.b_user = '0;
    assign ddr_bus.r_resp = 2'b00;
    assign ddr_bus.r_user = '0;
    assign ddr_bus.b_id = aw_id;
    assign ddr_bus.r_id = read_id;

    // Random mid-burst stalls model real MIG behavior (refresh pauses,
    // reordering delays) that the steady TB stream would otherwise hide.
    logic [31:0] stall_lfsr;
    logic        r_stall;
    logic        b_stall;

    integer i;
    integer memory_index;
    always_ff @(posedge clk) begin
        if (!rst_n) begin
            read_active <= 1'b0;
            read_address <= '0;
            read_beats <= '0;
            read_id <= '0;
            ddr_bus.r_data <= '0;
            ddr_bus.r_last <= 1'b0;
            ddr_bus.r_valid <= 1'b0;
            aw_have <= 1'b0;
            aw_address <= '0;
            aw_id <= '0;
            w_have <= 1'b0;
            w_data <= '0;
            w_strb <= '0;
            ddr_bus.b_valid <= 1'b0;
            stall_lfsr <= 32'hdeadcafe;
            r_stall <= 1'b0;
            b_stall <= 1'b0;
        end else begin
            stall_lfsr <= {stall_lfsr[30:0],
                           stall_lfsr[31] ^ stall_lfsr[21] ^
                           stall_lfsr[1] ^ stall_lfsr[0]};
            r_stall <= stall_lfsr[2:0] == 3'd0;
            b_stall <= stall_lfsr[5:3] == 3'd0;
            if (ddr_bus.ar_valid && ddr_bus.ar_ready) begin
                read_active <= 1'b1;
                read_address <= ddr_bus.ar_addr[29:0];
                read_beats <= {1'b0, ddr_bus.ar_len} + 1'b1;
                read_id <= ddr_bus.ar_id;
            end
            if (read_active && !ddr_bus.r_valid && !r_stall) begin
                memory_index = read_address - MODEL_BASE;
                for (i = 0; i < 8; i = i + 1)
                    ddr_bus.r_data[i*8 +: 8] <=
                        (memory_index + i < MEMORY_BYTES && memory_index >= 0) ?
                        memory[memory_index + i] : 8'h5a;
                ddr_bus.r_last <= read_beats == 1;
                ddr_bus.r_valid <= 1'b1;
            end
            if (ddr_bus.r_valid && ddr_bus.r_ready) begin
                if (ddr_bus.r_last) begin
                    ddr_bus.r_valid <= 1'b0;
                    read_active <= 1'b0;
                end else begin
                    read_address <= read_address + 8;
                    read_beats <= read_beats - 1'b1;
                    if (r_stall) begin
                        // Insert a bubble mid-burst; the next beat is
                        // presented on a later cycle.
                        ddr_bus.r_valid <= 1'b0;
                    end else begin
                        memory_index = read_address + 8 - MODEL_BASE;
                        for (i = 0; i < 8; i = i + 1)
                            ddr_bus.r_data[i*8 +: 8] <=
                                (memory_index + i < MEMORY_BYTES && memory_index >= 0) ?
                                memory[memory_index + i] : 8'h5a;
                        ddr_bus.r_last <= read_beats == 2;
                        ddr_bus.r_valid <= 1'b1;
                    end
                end
            end

            if (ddr_bus.aw_valid && ddr_bus.aw_ready) begin
                aw_have <= 1'b1;
                aw_address <= ddr_bus.aw_addr[29:0];
                aw_id <= ddr_bus.aw_id;
            end
            if (ddr_bus.w_valid && ddr_bus.w_ready) begin
                w_have <= 1'b1;
                w_data <= ddr_bus.w_data;
                w_strb <= ddr_bus.w_strb;
            end
            if (aw_have && w_have && !ddr_bus.b_valid && !b_stall) begin
                memory_index = aw_address - MODEL_BASE;
                for (i = 0; i < 8; i = i + 1)
                    if (w_strb[i] && memory_index + i < MEMORY_BYTES &&
                        memory_index >= 0)
                        memory[memory_index + i] <= w_data[i*8 +: 8];
                aw_have <= 1'b0;
                w_have <= 1'b0;
                ddr_bus.b_valid <= 1'b1;
            end
            if (ddr_bus.b_valid && ddr_bus.b_ready)
                ddr_bus.b_valid <= 1'b0;
        end
    end

    // ------------------------------------------------------------------
    // CPU traffic generator: continuous interleaved reads and writes so the
    // arbiter always has traffic to drain, mimicking uncached SMP harts.
    logic [29:0] cpu_addr;
    logic [2:0] cpu_phase;
    logic cpu_pause;
    logic [5:0] cpu_pause_count;
    integer cpu_reads_done;
    integer cpu_writes_done;

    assign cpu_bus.aw_id = 5'h03;
    assign cpu_bus.aw_addr = {{34{1'b0}}, cpu_addr};
    assign cpu_bus.aw_len = 8'd0;
    assign cpu_bus.aw_size = 3'd3;
    assign cpu_bus.aw_burst = 2'b01;
    assign cpu_bus.aw_lock = 1'b0;
    assign cpu_bus.aw_cache = 4'b0010;
    assign cpu_bus.aw_prot = '0;
    assign cpu_bus.aw_qos = '0;
    assign cpu_bus.aw_region = '0;
    assign cpu_bus.aw_atop = '0;
    assign cpu_bus.aw_user = '0;
    assign cpu_bus.aw_valid = !cpu_pause && cpu_phase == 3'd2;
    assign cpu_bus.w_data = 64'hdead_beef_0123_4567;
    assign cpu_bus.w_strb = 8'hff;
    assign cpu_bus.w_last = 1'b1;
    assign cpu_bus.w_user = '0;
    assign cpu_bus.w_valid = cpu_phase == 3'd3;
    assign cpu_bus.b_ready = 1'b1;
    assign cpu_bus.ar_id = 5'h04;
    assign cpu_bus.ar_addr = {{34{1'b0}}, cpu_addr};
    assign cpu_bus.ar_len = 8'd1;
    assign cpu_bus.ar_size = 3'd3;
    assign cpu_bus.ar_burst = 2'b01;
    assign cpu_bus.ar_lock = 1'b0;
    assign cpu_bus.ar_cache = 4'b0010;
    assign cpu_bus.ar_prot = '0;
    assign cpu_bus.ar_qos = '0;
    assign cpu_bus.ar_region = '0;
    assign cpu_bus.ar_user = '0;
    assign cpu_bus.ar_valid = !cpu_pause && cpu_phase == 3'd0;
    assign cpu_bus.r_ready = 1'b1;

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            cpu_addr <= 30'h3f00_0000;
            cpu_phase <= '0;
            cpu_pause <= 1'b0;
            cpu_pause_count <= '0;
            cpu_reads_done <= 0;
            cpu_writes_done <= 0;
        end else begin
            cpu_pause_count <= cpu_pause_count + 1'b1;
            // A short periodic gap in new CPU traffic mirrors real pipeline
            // bubbles; the arbiter needs them only for its outstanding drain.
            cpu_pause <= cpu_pause_count[5:4] == 2'b11;
            case (cpu_phase)
                3'd0: if (cpu_bus.ar_valid && cpu_bus.ar_ready) cpu_phase <= 3'd1;
                      else if (cpu_pause) cpu_phase <= 3'd2;
                3'd1: if (cpu_bus.r_valid && cpu_bus.r_ready && cpu_bus.r_last) begin
                          cpu_phase <= 3'd2;
                          cpu_reads_done <= cpu_reads_done + 1;
                      end
                3'd2: if (cpu_bus.aw_valid && cpu_bus.aw_ready) cpu_phase <= 3'd3;
                      else if (cpu_pause) cpu_phase <= 3'd0;
                3'd3: if (cpu_bus.w_valid && cpu_bus.w_ready) cpu_phase <= 3'd4;
                3'd4: if (cpu_bus.b_valid) begin
                          cpu_phase <= 3'd0;
                          cpu_writes_done <= cpu_writes_done + 1;
                          cpu_addr <= cpu_addr + 8;
                      end
                default: cpu_phase <= 3'd0;
            endcase
        end
    end

    // ------------------------------------------------------------------
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

    task automatic run_q1_under_traffic(input integer ne0, input integer ne1);
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
        integer timeout;
        begin
            for (input_block_index = 0; input_block_index < ne0 / 32;
                 input_block_index = input_block_index + 1) begin
                address = INPUT_OFFSET + input_block_index * 34;
                input_d = 16'h2c00 + ((input_block_index * 37) % 512);
                write_u16(address, input_d);
                for (j = 0; j < 32; j = j + 1)
                    memory[address + 2 + j] =
                        byte'(((input_block_index * 61 + j * 13) % 255) - 127);
            end
            for (row = 0; row < ne1; row = row + 1) begin
                for (block = 0; block < ne0 / 128; block = block + 1) begin
                    address = WEIGHT_OFFSET + (row * (ne0 / 128) + block) * 18;
                    weight_d = 16'h2600 + ((row * 97 + block * 41) % 1024);
                    write_u16(address, weight_d);
                    for (j = 0; j < 16; j = j + 1)
                        memory[address + 2 + j] =
                            (row * 19 + block * 7 + j * 111) & 8'hff;
                end
            end

            old_done = status[28];
            send_register(0, WEIGHT_OFFSET >> 3);
            send_register(1, INPUT_OFFSET >> 3);
            send_register(2, OUTPUT_OFFSET >> 3);
            send_register(3, ne0[24:0]);
            send_register(4, ne1[24:0]);
            send_register(5, 3);
            timeout = 0;
            while (status[28] == old_done && timeout < 4000000) begin
                @(posedge clk);
                timeout = timeout + 1;
            end
            if (timeout == 4000000)
                $fatal(1, "integration timeout: state=%0d request=%0d grant=%0d status=%08x reads=%0d writes=%0d",
                       dut.state, request, grant, status,
                       cpu_reads_done, cpu_writes_done);
            if (status[29])
                $fatal(1, "integration error flag set: status=%08x", status);

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
                    $fatal(1,
                           "integration Q1 %0dx%0d row %0d mismatch: got %h expected %h",
                           ne0, ne1, row, actual[31:0],
                           q30_float_reference(expected));
            end
            if (request || grant)
                $fatal(1, "bus not released after completion");
            $display("PASS: integration Q1 %0dx%0d under CPU traffic (cpu reads=%0d writes=%0d)",
                     ne0, ne1, cpu_reads_done, cpu_writes_done);
        end
    endtask

    integer k;
    initial begin
        for (k = 0; k < MEMORY_BYTES; k = k + 1)
            memory[k] = 0;
        repeat (10) @(posedge clk);
        rst_n = 1'b1;
        repeat (20) @(posedge clk);
        run_q1_under_traffic(256, 3);
        run_q1_under_traffic(2048, 24);
        run_q1_under_traffic(128, 16);
        $display("PASS: accelerator/arbiter integration regression complete");
        $finish;
    end

endmodule

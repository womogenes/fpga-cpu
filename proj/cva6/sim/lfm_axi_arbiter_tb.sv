`timescale 1ns/1ps

module lfm_axi_arbiter_tb;
    logic clk = 1'b0;
    logic rst_n = 1'b0;
    logic request = 1'b0;
    logic grant;

    AXI_BUS #(.AXI_ADDR_WIDTH(64), .AXI_DATA_WIDTH(64), .AXI_ID_WIDTH(5), .AXI_USER_WIDTH(1)) cpu();
    AXI_BUS #(.AXI_ADDR_WIDTH(64), .AXI_DATA_WIDTH(64), .AXI_ID_WIDTH(5), .AXI_USER_WIDTH(1)) accelerator();
    AXI_BUS #(.AXI_ADDR_WIDTH(64), .AXI_DATA_WIDTH(64), .AXI_ID_WIDTH(5), .AXI_USER_WIDTH(1)) ddr();

    always #2.5 clk = ~clk;

    lfm_axi_arbiter dut (
        .clk_i(clk),
        .rst_ni(rst_n),
        .accelerator_request_i(request),
        .accelerator_grant_o(grant),
        .cpu(cpu),
        .accelerator(accelerator),
        .ddr(ddr)
    );

    task automatic clear_masters;
        begin
            cpu.aw_id = '0;
            cpu.aw_addr = '0;
            cpu.aw_len = '0;
            cpu.aw_size = 3'd3;
            cpu.aw_burst = 2'b01;
            cpu.aw_lock = 1'b0;
            cpu.aw_cache = '0;
            cpu.aw_prot = '0;
            cpu.aw_qos = '0;
            cpu.aw_region = '0;
            cpu.aw_atop = '0;
            cpu.aw_user = '0;
            cpu.aw_valid = 1'b0;
            cpu.w_data = '0;
            cpu.w_strb = '1;
            cpu.w_last = 1'b1;
            cpu.w_user = '0;
            cpu.w_valid = 1'b0;
            cpu.b_ready = 1'b1;
            cpu.ar_id = '0;
            cpu.ar_addr = '0;
            cpu.ar_len = '0;
            cpu.ar_size = 3'd3;
            cpu.ar_burst = 2'b01;
            cpu.ar_lock = 1'b0;
            cpu.ar_cache = '0;
            cpu.ar_prot = '0;
            cpu.ar_qos = '0;
            cpu.ar_region = '0;
            cpu.ar_user = '0;
            cpu.ar_valid = 1'b0;
            cpu.r_ready = 1'b1;

            accelerator.aw_id = '1;
            accelerator.aw_addr = '0;
            accelerator.aw_len = '0;
            accelerator.aw_size = 3'd3;
            accelerator.aw_burst = 2'b01;
            accelerator.aw_lock = 1'b0;
            accelerator.aw_cache = '0;
            accelerator.aw_prot = '0;
            accelerator.aw_qos = '0;
            accelerator.aw_region = '0;
            accelerator.aw_atop = '0;
            accelerator.aw_user = '0;
            accelerator.aw_valid = 1'b0;
            accelerator.w_data = '0;
            accelerator.w_strb = '1;
            accelerator.w_last = 1'b1;
            accelerator.w_user = '0;
            accelerator.w_valid = 1'b0;
            accelerator.b_ready = 1'b1;
            accelerator.ar_id = '1;
            accelerator.ar_addr = '0;
            accelerator.ar_len = '0;
            accelerator.ar_size = 3'd3;
            accelerator.ar_burst = 2'b01;
            accelerator.ar_lock = 1'b0;
            accelerator.ar_cache = '0;
            accelerator.ar_prot = '0;
            accelerator.ar_qos = '0;
            accelerator.ar_region = '0;
            accelerator.ar_user = '0;
            accelerator.ar_valid = 1'b0;
            accelerator.r_ready = 1'b1;
        end
    endtask

    initial begin
        clear_masters();
        ddr.aw_ready = 1'b1;
        ddr.w_ready = 1'b0;
        ddr.b_id = '0;
        ddr.b_resp = '0;
        ddr.b_user = '0;
        ddr.b_valid = 1'b0;
        ddr.ar_ready = 1'b1;
        ddr.r_id = '0;
        ddr.r_data = '0;
        ddr.r_resp = '0;
        ddr.r_last = 1'b0;
        ddr.r_user = '0;
        ddr.r_valid = 1'b0;

        repeat (4) @(posedge clk);
        @(negedge clk);
        rst_n = 1'b1;
        @(posedge clk);

        // Accept AW while W is stalled, then request the accelerator. The
        // arbiter must finish this CPU write before switching masters.
        @(negedge clk);
        cpu.aw_valid = 1'b1;
        cpu.w_valid = 1'b1;
        @(posedge clk);
        #1;
        @(negedge clk);
        cpu.aw_valid = 1'b0;
        request = 1'b1;
        @(posedge clk);
        #1;
        if (grant || !ddr.w_valid)
            $fatal(1, "CPU write data was not drained before grant");
        @(negedge clk);
        ddr.w_ready = 1'b1;
        @(posedge clk);
        #1;
        @(negedge clk);
        cpu.w_valid = 1'b0;
        ddr.b_valid = 1'b1;
        @(posedge clk);
        #1;
        if (grant)
            $fatal(1, "grant changed before the drained response retired");
        @(negedge clk);
        ddr.b_valid = 1'b0;
        @(posedge clk);
        #1;
        if (!grant)
            $fatal(1, "accelerator was not granted after CPU drain");

        @(negedge clk);
        request = 1'b0;
        @(posedge clk);
        @(posedge clk);
        #1;
        if (grant)
            $fatal(1, "accelerator grant did not release");

        // A CPU AW/W presented after request belongs to a future transaction.
        // Neither independent channel may leak into the accelerator epoch.
        @(negedge clk);
        ddr.w_ready = 1'b1;
        cpu.aw_valid = 1'b1;
        cpu.w_valid = 1'b1;
        request = 1'b1;
        #1;
        if (ddr.aw_valid || ddr.w_valid || cpu.aw_ready || cpu.w_ready)
            $fatal(1, "blocked CPU write leaked across the handoff");
        @(posedge clk);
        @(posedge clk);
        #1;
        if (!grant)
            $fatal(1, "held future CPU write prevented accelerator grant");

        $display("PASS: AXI arbiter drains coupled write channels safely");
        $finish;
    end
endmodule

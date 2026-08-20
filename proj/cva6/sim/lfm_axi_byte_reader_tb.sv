`timescale 1ns/1ps

// Exercise the byte reader with multiple read bursts outstanding.  The main
// accelerator testbench only accepts a new AR after the previous burst's last
// beat, which leaves the deployed two-burst bookkeeping untested.
module lfm_axi_byte_reader_tb;
    localparam integer ID_WIDTH = 5;
    localparam logic [29:0] BASE = 30'h2f00_0000;

    logic clk = 1'b0;
    logic rst_n = 1'b0;
    logic start = 1'b0;
    logic [29:0] address;
    logic [31:0] byte_count;
    logic [3:0] unit_bytes;
    logic busy;
    logic done;
    logic error;
    logic [63:0] data;
    logic byte_valid;
    logic byte_ready;
    logic [ID_WIDTH-1:0] arid;
    logic [29:0] araddr;
    logic [7:0] arlen;
    logic [2:0] arsize;
    logic [1:0] arburst;
    logic arvalid;
    logic arready;
    logic [ID_WIDTH-1:0] rid;
    logic [63:0] rdata;
    logic [1:0] rresp;
    logic rlast;
    logic rvalid;
    logic rready;

    always #2.5 clk = ~clk;

    lfm_axi_byte_reader #(.ID_WIDTH(ID_WIDTH)) dut (
        .clk_i(clk),
        .rst_ni(rst_n),
        .start_i(start),
        .address_i(address),
        .byte_count_i(byte_count),
        .unit_bytes_i(unit_bytes),
        .busy_o(busy),
        .done_o(done),
        .error_o(error),
        .data_o(data),
        .byte_valid_o(byte_valid),
        .byte_ready_i(byte_ready),
        .arid_o(arid),
        .araddr_o(araddr),
        .arlen_o(arlen),
        .arsize_o(arsize),
        .arburst_o(arburst),
        .arvalid_o(arvalid),
        .arready_i(arready),
        .rid_i(rid),
        .rdata_i(rdata),
        .rresp_i(rresp),
        .rlast_i(rlast),
        .rvalid_i(rvalid),
        .rready_o(rready)
    );

    function automatic logic [7:0] pattern(input logic [29:0] byte_address);
        pattern = byte_address[7:0] ^ byte_address[15:8] ^ 8'h5a;
    endfunction

    function automatic logic [63:0] pattern_word(input logic [29:0] beat_address);
        integer lane;
        begin
            for (lane = 0; lane < 8; lane = lane + 1)
                pattern_word[lane * 8 +: 8] = pattern(beat_address + lane);
        end
    endfunction

    // Four-entry address queue.  AR stays ready while the current response is
    // active, unlike the legacy memory model, so two or more bursts become
    // outstanding exactly as they do through the physical MIG.
    logic [29:0] burst_address [0:3];
    logic [8:0] burst_beats [0:3];
    logic [1:0] queue_head;
    logic [1:0] queue_tail;
    logic [2:0] queue_count;
    logic [31:0] lfsr;
    logic push_burst;
    logic pop_burst;

    assign arready = queue_count < 4;
    assign push_burst = arvalid && arready;
    assign pop_burst = rvalid && rready && rlast;
    assign rid = {ID_WIDTH{1'b1}};
    assign rresp = 2'b00;

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            queue_head <= '0;
            queue_tail <= '0;
            queue_count <= '0;
            rdata <= '0;
            rlast <= 1'b0;
            rvalid <= 1'b0;
            lfsr <= 32'h91e1_0da5;
        end else begin
            lfsr <= {lfsr[30:0],
                     lfsr[31] ^ lfsr[21] ^ lfsr[1] ^ lfsr[0]};

            if (push_burst) begin
                burst_address[queue_tail] <= araddr;
                burst_beats[queue_tail] <= {1'b0, arlen} + 1'b1;
                queue_tail <= queue_tail + 1'b1;
            end

            if (!rvalid && queue_count != 0 && lfsr[3:0] != 0) begin
                rdata <= pattern_word(burst_address[queue_head]);
                rlast <= burst_beats[queue_head] == 1;
                rvalid <= 1'b1;
            end else if (rvalid && rready) begin
                rvalid <= 1'b0;
                if (rlast) begin
                    queue_head <= queue_head + 1'b1;
                end else begin
                    burst_address[queue_head] <=
                        burst_address[queue_head] + 8;
                    burst_beats[queue_head] <=
                        burst_beats[queue_head] - 1'b1;
                end
            end

            case ({push_burst, pop_burst})
                2'b10: queue_count <= queue_count + 1'b1;
                2'b01: queue_count <= queue_count - 1'b1;
                default: begin end
            endcase
        end
    end

    logic [29:0] expected_address;
    integer consumed_bytes;
    integer cycle_count;
    integer lane;

    always_comb begin
        // Randomized consumer backpressure, including multi-cycle gaps.
        byte_ready = lfsr[6:4] != 3'b000 && lfsr[10:8] != 3'b111;
    end

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            expected_address <= '0;
            consumed_bytes <= 0;
            cycle_count <= 0;
        end else begin
            cycle_count <= cycle_count + 1;
            if (byte_valid && byte_ready) begin
                for (lane = 0; lane < unit_bytes; lane = lane + 1)
                    if (data[lane * 8 +: 8] !==
                        pattern(expected_address + lane))
                        $fatal(1,
                               "data mismatch byte=%0d lane=%0d got=%02x expected=%02x",
                               consumed_bytes, lane,
                               data[lane * 8 +: 8],
                               pattern(expected_address + lane));
                expected_address <= expected_address + unit_bytes;
                consumed_bytes <= consumed_bytes + unit_bytes;
            end
            if (cycle_count > 200000)
                $fatal(1,
                       "reader timeout consumed=%0d busy=%0d bursts=%0d beats=%0d current=%0d queued=%0d",
                       consumed_bytes, busy, dut.bursts_in_flight,
                       dut.beats_in_flight, dut.current_burst_beats,
                       dut.queued_burst_beats);
        end
    end

    task automatic run_case(
        input logic [29:0] test_address,
        input integer test_bytes,
        input integer test_unit
    );
        begin
            while (busy || queue_count != 0 || rvalid) @(posedge clk);
            address = test_address;
            byte_count = test_bytes;
            unit_bytes = test_unit;
            expected_address = test_address;
            consumed_bytes = 0;
            start = 1'b1;
            @(posedge clk);
            start = 1'b0;
            while (!done) @(posedge clk);
            if (error) $fatal(1, "reader reported AXI error");
            if (consumed_bytes != test_bytes)
                $fatal(1, "consumed %0d bytes, expected %0d",
                       consumed_bytes, test_bytes);
            $display("PASS: reader address=%08x bytes=%0d unit=%0d",
                     test_address, test_bytes, test_unit);
        end
    endtask

    initial begin
        address = BASE;
        byte_count = 0;
        unit_bytes = 8;
        repeat (8) @(posedge clk);
        rst_n = 1'b1;
        repeat (4) @(posedge clk);

        run_case(BASE, 72, 8);
        run_case(BASE, 72 * 28, 8);
        run_case(BASE, 72 * 29, 8);
        run_case(BASE + 30'h0000_0ae0, 72 * 74, 8);
        run_case(BASE + 30'h0000_0ff8, 9216, 8);
        run_case(BASE, 9216, 2);
        run_case(BASE, 9216, 1);
        $display("PASS: AXI byte reader multi-burst regression");
        $finish;
    end
endmodule

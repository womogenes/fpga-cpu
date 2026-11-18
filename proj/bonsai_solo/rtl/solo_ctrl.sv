// PicoRV32 controller for the Bonsai solo build. Everything runs in the
// 200 MHz MIG UI clock domain, so the accelerator command/status words need
// no clock crossing.
//
// Address map
//   0x0000_0000  64 KiB BRAM (boot ROM image + firmware loaded over UART)
//   0x1000_0000  MMIO
//     0x00 UART TX       W  byte; the store stalls until the transmitter is free
//     0x04 UART RX       R  {rx_valid, 23'b0, byte}; a valid read pops the FIFO
//     0x08 UART status   R  bit0 rx non-empty, bit1 tx ready
//     0x10 accel command RW command_i
//     0x14 accel status  R  status_o
//     0x18 cycles low    R  latches the high word
//     0x1c cycles high   R
//     0x20 system        R  bit0 DDR calibration complete
//     0x24 LEDs          W
//     0x40-0x5c beat buffer words 0..7 (one 256-bit DDR beat)
//     0x60 beat address  W  byte address, 32-byte aligned
//     0x64 beat control  W  bit0 write buffer to DDR, bit1 read DDR into
//                           buffer; the store stalls until the beat completes
//     0x68 beat strobe   W  byte enables for beat writes
//   0x8000_0000-0xbfff_ffff  single-word DDR window (slow, for setup)
module solo_ctrl #(
    parameter string BOOT_HEX = "boot.hex",
    parameter integer BRAM_WORDS = 16384,
    parameter integer RX_FIFO_BYTES = 4096,
    parameter integer CLOCK_HZ = 200000000,
    parameter integer BAUD = 3000000
) (
    input  logic        clk_i,
    input  logic        rst_ni,
    input  logic        calib_done_i,
    input  logic        uart_rx_i,
    output logic        uart_tx_o,
    output logic [31:0] accel_command_o,
    input  logic [31:0] accel_status_i,
    output logic [7:0]  led_o,
    AXI_BUS.Master      ddr
);

    // ------------------------------------------------------------------ CPU
    logic        mem_valid;
    logic        mem_instr;
    logic        mem_ready;
    logic [31:0] mem_addr;
    logic [31:0] mem_wdata;
    logic [3:0]  mem_wstrb;
    logic [31:0] mem_rdata;

    picorv32 #(
        .ENABLE_COUNTERS     (1),
        .ENABLE_COUNTERS64   (1),
        .ENABLE_REGS_16_31   (1),
        .ENABLE_REGS_DUALPORT(1),
        .LATCHED_MEM_RDATA   (0),
        .TWO_STAGE_SHIFT     (1),
        .BARREL_SHIFTER      (1),
        .TWO_CYCLE_COMPARE   (1),
        .TWO_CYCLE_ALU       (1),
        .COMPRESSED_ISA      (0),
        .CATCH_MISALIGN      (1),
        .CATCH_ILLINSN       (1),
        .ENABLE_PCPI         (0),
        .ENABLE_MUL          (0),
        .ENABLE_FAST_MUL     (1),
        .ENABLE_DIV          (1),
        .ENABLE_IRQ          (0),
        .ENABLE_TRACE        (0),
        .PROGADDR_RESET      (32'h0000_0000),
        .STACKADDR           (32'h0001_0000)
    ) i_cpu (
        .clk         (clk_i),
        .resetn      (rst_ni),
        .trap        (),
        .mem_valid   (mem_valid),
        .mem_instr   (mem_instr),
        .mem_ready   (mem_ready),
        .mem_addr    (mem_addr),
        .mem_wdata   (mem_wdata),
        .mem_wstrb   (mem_wstrb),
        .mem_rdata   (mem_rdata),
        .mem_la_read (),
        .mem_la_write(),
        .mem_la_addr (),
        .mem_la_wdata(),
        .mem_la_wstrb(),
        .pcpi_valid  (),
        .pcpi_insn   (),
        .pcpi_rs1    (),
        .pcpi_rs2    (),
        .pcpi_wr     (1'b0),
        .pcpi_rd     (32'h0),
        .pcpi_wait   (1'b0),
        .pcpi_ready  (1'b0),
        .irq         (32'h0),
        .eoi         (),
        .trace_valid (),
        .trace_data  ()
    );

    wire sel_bram = mem_valid && mem_addr[31:28] == 4'h0;
    wire sel_mmio = mem_valid && mem_addr[31:28] == 4'h1;
    wire sel_ddr  = mem_valid && mem_addr[31:30] == 2'b10;

    // ----------------------------------------------------------------- BRAM
    localparam integer BRAM_BITS = $clog2(BRAM_WORDS);
    logic [31:0] bram [BRAM_WORDS];
    logic [31:0] bram_rdata;
    logic        bram_ready;

    initial $readmemh(BOOT_HEX, bram);

    always_ff @(posedge clk_i) begin
        bram_ready <= sel_bram && !bram_ready;
        if (sel_bram && !bram_ready) begin
            bram_rdata <= bram[mem_addr[BRAM_BITS+1:2]];
            for (int i = 0; i < 4; i++)
                if (mem_wstrb[i])
                    bram[mem_addr[BRAM_BITS+1:2]][8*i +: 8] <= mem_wdata[8*i +: 8];
        end
    end

    // ----------------------------------------------------------------- UART
    logic [7:0] rx_byte;
    logic       rx_strobe;
    logic       tx_ready;
    logic       tx_valid;

    uart_rx #(.CLOCK_HZ(CLOCK_HZ), .BAUD(BAUD)) i_uart_rx (
        .clk   (clk_i),
        .reset (!rst_ni),
        .serial(uart_rx_i),
        .data  (rx_byte),
        .valid (rx_strobe)
    );

    uart_tx #(.CLOCK_HZ(CLOCK_HZ), .BAUD(BAUD)) i_uart_tx (
        .clk   (clk_i),
        .reset (!rst_ni),
        .data  (mem_wdata[7:0]),
        .valid (tx_valid),
        .ready (tx_ready),
        .serial(uart_tx_o)
    );

    localparam integer RX_BITS = $clog2(RX_FIFO_BYTES);
    logic [7:0]       rx_fifo [RX_FIFO_BYTES];
    logic [RX_BITS:0] rx_head;
    logic [RX_BITS:0] rx_tail;
    logic [7:0]       rx_head_byte;
    wire              rx_nonempty = rx_head != rx_tail;

    always_ff @(posedge clk_i) begin
        if (rx_strobe) rx_fifo[rx_tail[RX_BITS-1:0]] <= rx_byte;
        rx_head_byte <= rx_fifo[rx_head[RX_BITS-1:0]];
    end

    // ----------------------------------------------------------------- MMIO
    logic [63:0]  cycles;
    logic [31:0]  cycles_high_latched;
    logic [255:0] beat_buffer;
    logic [29:0]  beat_address;
    logic [31:0]  beat_strobe;
    logic         mmio_ready;
    logic [31:0]  mmio_rdata;
    logic         rx_read_wait;
    logic         rx_read_valid;

    // DDR bridge requests from the MMIO beat port or the word window.
    typedef enum logic [2:0] {
        BR_IDLE, BR_AW, BR_W, BR_B, BR_AR, BR_R
    } bridge_state_t;
    bridge_state_t bridge_state;
    logic         bridge_done;
    logic         bridge_word;
    logic [29:0]  bridge_address;
    logic [255:0] bridge_wdata;
    logic [31:0]  bridge_strobe;
    logic [255:0] bridge_rdata;
    logic         ddr_word_ready;

    wire beat_ctrl_write = sel_mmio && mem_addr[7:0] == 8'h64 && mem_wstrb != 4'b0;
    wire tx_write = sel_mmio && mem_addr[7:0] == 8'h00 && mem_wstrb != 4'b0;
    assign tx_valid = tx_write && tx_ready && !mmio_ready;

    always_ff @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
            cycles              <= '0;
            cycles_high_latched <= '0;
            accel_command_o     <= '0;
            led_o               <= '0;
            beat_buffer         <= '0;
            beat_address        <= '0;
            beat_strobe         <= '1;
            mmio_ready          <= 1'b0;
            mmio_rdata          <= '0;
            rx_head             <= '0;
            rx_tail             <= '0;
            rx_read_wait        <= 1'b0;
            rx_read_valid       <= 1'b0;
        end else begin
            cycles     <= cycles + 64'd1;
            mmio_ready <= 1'b0;
            if (rx_strobe) rx_tail <= rx_tail + 1'b1;

            if (sel_mmio && !mmio_ready) begin
                unique case (mem_addr[7:0])
                    8'h00: begin
                        if (mem_wstrb != 4'b0 && tx_ready) mmio_ready <= 1'b1;
                        else if (mem_wstrb == 4'b0) mmio_ready <= 1'b1;
                    end
                    8'h04: begin
                        // Wait one cycle so rx_head_byte reflects rx_head.
                        // Validity is sampled in the same cycle as the
                        // byte, so a byte landing mid-read is not returned
                        // as a stale FIFO slot.
                        if (!rx_read_wait) begin
                            rx_read_wait  <= 1'b1;
                            rx_read_valid <= rx_nonempty;
                        end else begin
                            rx_read_wait <= 1'b0;
                            mmio_rdata   <= {rx_read_valid, 23'b0, rx_head_byte};
                            if (rx_read_valid) rx_head <= rx_head + 1'b1;
                            mmio_ready   <= 1'b1;
                        end
                    end
                    8'h08: begin
                        mmio_rdata <= {30'b0, tx_ready, rx_nonempty};
                        mmio_ready <= 1'b1;
                    end
                    8'h10: begin
                        if (mem_wstrb != 4'b0) accel_command_o <= mem_wdata;
                        mmio_rdata <= accel_command_o;
                        mmio_ready <= 1'b1;
                    end
                    8'h14: begin
                        mmio_rdata <= accel_status_i;
                        mmio_ready <= 1'b1;
                    end
                    8'h18: begin
                        mmio_rdata          <= cycles[31:0];
                        cycles_high_latched <= cycles[63:32];
                        mmio_ready          <= 1'b1;
                    end
                    8'h1c: begin
                        mmio_rdata <= cycles_high_latched;
                        mmio_ready <= 1'b1;
                    end
                    8'h20: begin
                        mmio_rdata <= {31'b0, calib_done_i};
                        mmio_ready <= 1'b1;
                    end
                    8'h24: begin
                        if (mem_wstrb != 4'b0) led_o <= mem_wdata[7:0];
                        mmio_rdata <= {24'b0, led_o};
                        mmio_ready <= 1'b1;
                    end
                    8'h60: begin
                        if (mem_wstrb != 4'b0) beat_address <= mem_wdata[29:0];
                        mmio_rdata <= {2'b0, beat_address};
                        mmio_ready <= 1'b1;
                    end
                    8'h64: begin
                        if (mem_wstrb == 4'b0) begin
                            mmio_rdata <= 32'b0;
                            mmio_ready <= 1'b1;
                        end else if (bridge_done) begin
                            if (mem_wdata[1]) beat_buffer <= bridge_rdata;
                            mmio_ready <= 1'b1;
                        end
                    end
                    8'h68: begin
                        if (mem_wstrb != 4'b0) beat_strobe <= mem_wdata;
                        mmio_rdata <= beat_strobe;
                        mmio_ready <= 1'b1;
                    end
                    default: begin
                        if (mem_addr[7:5] == 3'b010) begin
                            if (mem_wstrb != 4'b0)
                                beat_buffer[32*mem_addr[4:2] +: 32] <= mem_wdata;
                            mmio_rdata <= beat_buffer[32*mem_addr[4:2] +: 32];
                        end else begin
                            mmio_rdata <= 32'hdead_beef;
                        end
                        mmio_ready <= 1'b1;
                    end
                endcase
            end
        end
    end

    // ----------------------------------------------------- DDR AXI bridge
    // One 256-bit beat per request, ID 0. The word window places the 32-bit
    // word in its lane; the beat port moves the whole buffer.
    wire bridge_start = (beat_ctrl_write || sel_ddr) &&
                        bridge_state == BR_IDLE && !bridge_done;

    always_ff @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
            bridge_state   <= BR_IDLE;
            bridge_done    <= 1'b0;
            bridge_word    <= 1'b0;
            bridge_address <= '0;
            bridge_wdata   <= '0;
            bridge_strobe  <= '0;
            bridge_rdata   <= '0;
        end else begin
            // bridge_done holds until the CPU-side access retires.
            if (bridge_done && !(beat_ctrl_write || sel_ddr)) bridge_done <= 1'b0;
            if (bridge_done && (mmio_ready || ddr_word_ready)) bridge_done <= 1'b0;
            unique case (bridge_state)
                BR_IDLE: if (bridge_start) begin
                    if (sel_ddr) begin
                        bridge_word    <= 1'b1;
                        bridge_address <= {mem_addr[29:5], 5'b0};
                        bridge_wdata   <= {8{mem_wdata}};
                        bridge_strobe  <= 32'(mem_wstrb) << (4 * mem_addr[4:2]);
                        bridge_state   <= mem_wstrb != 4'b0 ? BR_AW : BR_AR;
                    end else begin
                        bridge_word    <= 1'b0;
                        bridge_address <= {beat_address[29:5], 5'b0};
                        bridge_wdata   <= beat_buffer;
                        bridge_strobe  <= beat_strobe;
                        bridge_state   <= mem_wdata[0] ? BR_AW : BR_AR;
                    end
                end
                BR_AW: if (ddr.aw_ready) bridge_state <= BR_W;
                BR_W:  if (ddr.w_ready) bridge_state <= BR_B;
                BR_B:  if (ddr.b_valid) begin
                    bridge_state <= BR_IDLE;
                    bridge_done  <= 1'b1;
                end
                BR_AR: if (ddr.ar_ready) bridge_state <= BR_R;
                BR_R:  if (ddr.r_valid) begin
                    bridge_rdata <= ddr.r_data;
                    bridge_state <= BR_IDLE;
                    bridge_done  <= 1'b1;
                end
                default: bridge_state <= BR_IDLE;
            endcase
        end
    end

    always_ff @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) ddr_word_ready <= 1'b0;
        else ddr_word_ready <= sel_ddr && bridge_done && bridge_word && !ddr_word_ready;
    end

    assign ddr.aw_id     = '0;
    assign ddr.aw_addr   = {{(64-32){1'b0}}, 2'b10, bridge_address};
    assign ddr.aw_len    = 8'd0;
    assign ddr.aw_size   = 3'd5;
    assign ddr.aw_burst  = 2'b01;
    assign ddr.aw_lock   = 1'b0;
    assign ddr.aw_cache  = 4'b0011;
    assign ddr.aw_prot   = 3'b000;
    assign ddr.aw_qos    = 4'b0;
    assign ddr.aw_region = 4'b0;
    assign ddr.aw_atop   = '0;
    assign ddr.aw_user   = '0;
    assign ddr.aw_valid  = bridge_state == BR_AW;
    assign ddr.w_data    = bridge_wdata;
    assign ddr.w_strb    = bridge_strobe;
    assign ddr.w_last    = 1'b1;
    assign ddr.w_user    = '0;
    assign ddr.w_valid   = bridge_state == BR_W;
    assign ddr.b_ready   = bridge_state == BR_B;
    assign ddr.ar_id     = '0;
    assign ddr.ar_addr   = {{(64-32){1'b0}}, 2'b10, bridge_address};
    assign ddr.ar_len    = 8'd0;
    assign ddr.ar_size   = 3'd5;
    assign ddr.ar_burst  = 2'b01;
    assign ddr.ar_lock   = 1'b0;
    assign ddr.ar_cache  = 4'b0011;
    assign ddr.ar_prot   = 3'b000;
    assign ddr.ar_qos    = 4'b0;
    assign ddr.ar_region = 4'b0;
    assign ddr.ar_user   = '0;
    assign ddr.ar_valid  = bridge_state == BR_AR;
    assign ddr.r_ready   = bridge_state == BR_R;

    // ---------------------------------------------------------- read mux
    always_comb begin
        mem_ready = 1'b0;
        mem_rdata = 32'h0;
        if (bram_ready) begin
            mem_ready = 1'b1;
            mem_rdata = bram_rdata;
        end else if (mmio_ready) begin
            mem_ready = 1'b1;
            mem_rdata = mmio_rdata;
        end else if (ddr_word_ready) begin
            mem_ready = 1'b1;
            mem_rdata = bridge_rdata[32*mem_addr[4:2] +: 32];
        end
    end

endmodule

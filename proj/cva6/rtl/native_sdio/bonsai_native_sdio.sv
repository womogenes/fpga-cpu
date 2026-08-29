// AXI interface wrapper for the pinned native 4-bit SD controller.
// The native-SD build selects this wrapper; the default build remains SPI.
module bonsai_native_sdio #(
    parameter int unsigned AxiAddrWidth = 64,
    parameter int unsigned RegAxiIdWidth = 4,
    parameter int unsigned DmaAxiIdWidth = 2,
    parameter int unsigned RegAxiDataWidth = 64,
    parameter int unsigned DmaAddrWidth = 32,
    parameter int unsigned FifoAddrBits = 7
) (
    input logic clk_i,
    input logic rst_ni,
    AXI_BUS.Slave regs,
    AXI_BUS.Master dma,
    inout wire sd_cmd,
    inout wire [3:0] sd_dat,
    output logic sd_clk,
    output logic sd_reset,
    input logic sd_cd,
    output logic irq_o
);

    logic [1:0] regs_bresp;
    logic regs_bvalid;
    logic [31:0] regs_rdata;
    logic [1:0] regs_rresp;
    logic regs_rvalid;
    logic [RegAxiIdWidth-1:0] write_id_q;
    logic [RegAxiIdWidth-1:0] read_id_q;
    logic read_lane_q;
    logic dma_write_lane_q;
    logic dma_read_lane_q;
    logic dma_write_lane;
    logic dma_read_lane;

    logic [DmaAddrWidth-1:0] dma_awaddr;
    logic [7:0] dma_awlen;
    logic dma_awvalid;
    logic [31:0] dma_wdata;
    logic dma_wlast;
    logic dma_wvalid;
    logic dma_bready;
    logic [DmaAddrWidth-1:0] dma_araddr;
    logic [7:0] dma_arlen;
    logic dma_arvalid;
    logic dma_rready;

    always_ff @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
            write_id_q <= '0;
            read_id_q <= '0;
            read_lane_q <= 1'b0;
            dma_write_lane_q <= 1'b0;
            dma_read_lane_q <= 1'b0;
        end else begin
            if (regs.aw_valid && regs.aw_ready)
                write_id_q <= regs.aw_id;
            if (regs.ar_valid && regs.ar_ready) begin
                read_id_q <= regs.ar_id;
                read_lane_q <= regs.ar_addr[2];
            end
            if (dma_awvalid && dma.aw_ready) begin
                // The controller commonly presents AW and the first W beat in
                // the same cycle. Record the lane for the *next* beat when
                // both handshakes fire; otherwise beat 1 repeats beat 0's
                // lane, overwrites it, and shifts every later 32-bit word.
                if (dma_wvalid && dma.w_ready)
                    dma_write_lane_q <= ~dma_awaddr[2];
                else
                    dma_write_lane_q <= dma_awaddr[2];
            end
            else if (dma_wvalid && dma.w_ready)
                dma_write_lane_q <= ~dma_write_lane_q;
            if (dma_arvalid && dma.ar_ready) begin
                if (dma.r_valid && dma_rready)
                    dma_read_lane_q <= ~dma_araddr[2];
                else
                    dma_read_lane_q <= dma_araddr[2];
            end
            else if (dma.r_valid && dma_rready)
                dma_read_lane_q <= ~dma_read_lane_q;
        end
    end

    // Select the correct lane for a first data beat accepted concurrently
    // with its address. Subsequent beats use the registered next-beat lane.
    assign dma_write_lane = dma_awvalid && dma.aw_ready ?
                            dma_awaddr[2] : dma_write_lane_q;
    assign dma_read_lane = dma_arvalid && dma.ar_ready ?
                           dma_araddr[2] : dma_read_lane_q;

    // Register accesses are naturally aligned 32-bit MMIO transactions. This
    // lane adapter avoids pulling another generic width-converter hierarchy
    // into an already timing-constrained design.
    assign regs.b_id = write_id_q;
    assign regs.b_resp = regs_bresp;
    assign regs.b_user = '0;
    assign regs.b_valid = regs_bvalid;
    assign regs.r_id = read_id_q;
    assign regs.r_data = read_lane_q ? {regs_rdata, 32'b0} : {32'b0, regs_rdata};
    assign regs.r_resp = regs_rresp;
    assign regs.r_last = 1'b1;
    assign regs.r_user = '0;
    assign regs.r_valid = regs_rvalid;

    assign dma.aw_id = '0;
    assign dma.aw_addr = {{(AxiAddrWidth-DmaAddrWidth){1'b0}}, dma_awaddr};
    assign dma.aw_len = dma_awlen;
    assign dma.aw_size = 3'd2;
    assign dma.aw_burst = 2'b01;
    assign dma.aw_lock = '0;
    assign dma.aw_cache = '0;
    assign dma.aw_prot = '0;
    assign dma.aw_qos = '0;
    assign dma.aw_region = '0;
    assign dma.aw_atop = '0;
    assign dma.aw_user = '0;
    assign dma.aw_valid = dma_awvalid;
    assign dma.w_data = dma_write_lane ? {dma_wdata, 32'b0} : {32'b0, dma_wdata};
    assign dma.w_strb = dma_write_lane ? 8'hf0 : 8'h0f;
    assign dma.w_last = dma_wlast;
    assign dma.w_user = '0;
    assign dma.w_valid = dma_wvalid;
    assign dma.b_ready = dma_bready;
    assign dma.ar_id = '0;
    assign dma.ar_addr = {{(AxiAddrWidth-DmaAddrWidth){1'b0}}, dma_araddr};
    assign dma.ar_len = dma_arlen;
    assign dma.ar_size = 3'd2;
    assign dma.ar_burst = 2'b01;
    assign dma.ar_lock = '0;
    assign dma.ar_cache = '0;
    assign dma.ar_prot = '0;
    assign dma.ar_qos = '0;
    assign dma.ar_region = '0;
    assign dma.ar_user = '0;
    assign dma.ar_valid = dma_arvalid;
    assign dma.r_ready = dma_rready;

    sdc_controller #(
        .dma_addr_bits(DmaAddrWidth),
        .fifo_addr_bits(FifoAddrBits),
        .sdio_card_detect_level(1),
        .voltage_controll_reg(3300),
        .capabilies_reg(16'b0000_0000_0000_0011)
    ) i_sdc_controller (
        .async_resetn(rst_ni),
        .clock(clk_i),
        .s_axi_awaddr(regs.aw_addr[15:0]),
        .s_axi_awvalid(regs.aw_valid),
        .s_axi_awready(regs.aw_ready),
        // Prefer the low word for an aligned 64-bit store (WSTRB=ff). This is
        // required for the 32-bit DMA-address register at 0x60. Select the high
        // word only for a narrow write whose strobes exclusively occupy that
        // lane, such as a 32-bit store to address +4.
        .s_axi_wdata((|regs.w_strb[7:4]) && !(|regs.w_strb[3:0]) ?
                     regs.w_data[63:32] : regs.w_data[31:0]),
        .s_axi_wvalid(regs.w_valid),
        .s_axi_wready(regs.w_ready),
        .s_axi_bresp(regs_bresp),
        .s_axi_bvalid(regs_bvalid),
        .s_axi_bready(regs.b_ready),
        .s_axi_araddr(regs.ar_addr[15:0]),
        .s_axi_arvalid(regs.ar_valid),
        .s_axi_arready(regs.ar_ready),
        .s_axi_rdata(regs_rdata),
        .s_axi_rresp(regs_rresp),
        .s_axi_rvalid(regs_rvalid),
        .s_axi_rready(regs.r_ready),
        .m_axi_awaddr(dma_awaddr),
        .m_axi_awlen(dma_awlen),
        .m_axi_awvalid(dma_awvalid),
        .m_axi_awready(dma.aw_ready),
        .m_axi_wdata(dma_wdata),
        .m_axi_wlast(dma_wlast),
        .m_axi_wvalid(dma_wvalid),
        .m_axi_wready(dma.w_ready),
        .m_axi_bresp(dma.b_resp),
        .m_axi_bvalid(dma.b_valid),
        .m_axi_bready(dma_bready),
        .m_axi_araddr(dma_araddr),
        .m_axi_arlen(dma_arlen),
        .m_axi_arvalid(dma_arvalid),
        .m_axi_arready(dma.ar_ready),
        .m_axi_rdata(dma_read_lane ? dma.r_data[63:32] : dma.r_data[31:0]),
        .m_axi_rlast(dma.r_last),
        .m_axi_rresp(dma.r_resp),
        .m_axi_rvalid(dma.r_valid),
        .m_axi_rready(dma_rready),
        .sdio_cmd(sd_cmd),
        .sdio_dat(sd_dat),
        .sdio_clk(sd_clk),
        .sdio_reset(sd_reset),
        .sdio_cd(sd_cd),
        .interrupt(irq_o)
    );

endmodule

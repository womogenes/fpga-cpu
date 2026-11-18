// Bonsai solo: Genesys 2 image that only runs Bonsai-1.7B.
//
// MIG DDR3 (200 MHz UI, 256-bit AXI) is shared by the accelerator and the
// PicoRV32 controller through the exclusive arbiter; the accelerator owns the
// bus while a command runs. All logic except the FFN-packer FMA (125 MHz)
// runs in the UI clock domain.
module solo_top #(
    parameter string BOOT_HEX = "boot.hex"
) (
    input  logic        sys_clk_p,
    input  logic        sys_clk_n,
    input  logic        cpu_resetn,

    inout  wire  [31:0] ddr3_dq,
    inout  wire  [3:0]  ddr3_dqs_n,
    inout  wire  [3:0]  ddr3_dqs_p,
    output logic [14:0] ddr3_addr,
    output logic [2:0]  ddr3_ba,
    output logic        ddr3_ras_n,
    output logic        ddr3_cas_n,
    output logic        ddr3_we_n,
    output logic        ddr3_reset_n,
    output logic [0:0]  ddr3_ck_p,
    output logic [0:0]  ddr3_ck_n,
    output logic [0:0]  ddr3_cke,
    output logic [0:0]  ddr3_cs_n,
    output logic [3:0]  ddr3_dm,
    output logic [0:0]  ddr3_odt,

    input  logic        rx,
    output logic        tx,
    output logic [7:0]  led,
    output logic        fan_pwm
);

    logic clk;
    logic ui_reset;
    logic calib_done;

    logic [4:0]   m_axi_awid, m_axi_bid, m_axi_arid, m_axi_rid;
    logic [29:0]  m_axi_awaddr, m_axi_araddr;
    logic [7:0]   m_axi_awlen, m_axi_arlen;
    logic [2:0]   m_axi_awsize, m_axi_arsize, m_axi_awprot, m_axi_arprot;
    logic [1:0]   m_axi_awburst, m_axi_arburst, m_axi_bresp, m_axi_rresp;
    logic         m_axi_awlock, m_axi_arlock;
    logic [3:0]   m_axi_awcache, m_axi_arcache, m_axi_awqos, m_axi_arqos;
    logic         m_axi_awvalid, m_axi_awready, m_axi_wlast, m_axi_wvalid;
    logic         m_axi_wready, m_axi_bready, m_axi_bvalid, m_axi_arvalid;
    logic         m_axi_arready, m_axi_rready, m_axi_rlast, m_axi_rvalid;
    logic [255:0] m_axi_wdata, m_axi_rdata;
    logic [31:0]  m_axi_wstrb;
    logic [7:0]   ctrl_led;

    // --------------------------------------------------------------- resets
    // Logic leaves reset once the UI clock is stable and DDR has calibrated.
    (* ASYNC_REG = "TRUE" *) logic [2:0] rst_sync;
    logic rst_n;
    always_ff @(posedge clk or posedge ui_reset) begin
        if (ui_reset) rst_sync <= '0;
        else rst_sync <= {rst_sync[1:0], calib_done};
    end
    assign rst_n = rst_sync[2];

    // 125 MHz for the fpnew FMA inside the FFN packer.
    logic fp_clk_unbuffered, fp_clk, fp_feedback, fp_locked;
    MMCME2_BASE #(
        .CLKIN1_PERIOD   (5.0),
        .CLKFBOUT_MULT_F (5.0),
        .CLKOUT0_DIVIDE_F(8.0),
        .DIVCLK_DIVIDE   (1)
    ) i_fp_mmcm (
        .CLKIN1   (clk),
        .CLKFBIN  (fp_feedback),
        .CLKFBOUT (fp_feedback),
        .CLKOUT0  (fp_clk_unbuffered),
        .CLKOUT0B (), .CLKOUT1 (), .CLKOUT1B (), .CLKOUT2 (), .CLKOUT2B (),
        .CLKOUT3  (), .CLKOUT3B (), .CLKOUT4 (), .CLKOUT5 (), .CLKOUT6 (),
        .CLKFBOUTB(),
        .LOCKED   (fp_locked),
        .PWRDWN   (1'b0),
        .RST      (ui_reset)
    );
    BUFG i_fp_bufg (.I(fp_clk_unbuffered), .O(fp_clk));

    (* ASYNC_REG = "TRUE" *) logic [2:0] fp_rst_sync;
    always_ff @(posedge fp_clk or negedge fp_locked) begin
        if (!fp_locked) fp_rst_sync <= '0;
        else fp_rst_sync <= {fp_rst_sync[1:0], rst_n};
    end

    solo_core #(.BOOT_HEX(BOOT_HEX)) i_core (
        .clk_i       (clk),
        .fp_clk_i    (fp_clk),
        .rst_ni      (rst_n),
        .fp_rst_ni   (fp_rst_sync[2]),
        .calib_done_i(calib_done),
        .rx, .tx, .ctrl_led,
        .m_axi_awid, .m_axi_awaddr, .m_axi_awlen, .m_axi_awsize,
        .m_axi_awburst, .m_axi_awlock, .m_axi_awcache, .m_axi_awprot,
        .m_axi_awqos, .m_axi_awvalid, .m_axi_awready, .m_axi_wdata,
        .m_axi_wstrb, .m_axi_wlast, .m_axi_wvalid, .m_axi_wready,
        .m_axi_bready, .m_axi_bid, .m_axi_bresp, .m_axi_bvalid,
        .m_axi_arid, .m_axi_araddr, .m_axi_arlen, .m_axi_arsize,
        .m_axi_arburst, .m_axi_arlock, .m_axi_arcache, .m_axi_arprot,
        .m_axi_arqos, .m_axi_arvalid, .m_axi_arready, .m_axi_rready,
        .m_axi_rid, .m_axi_rdata, .m_axi_rresp, .m_axi_rlast, .m_axi_rvalid
    );

    logic [26:0] heartbeat;
    always_ff @(posedge clk) heartbeat <= heartbeat + 1'b1;
    assign led     = {ctrl_led[7:2], heartbeat[26], calib_done};
    assign fan_pwm = 1'b1;

    // ------------------------------------------------------------------ MIG
    xlnx_mig_7_ddr3 i_ddr (
        .sys_clk_p,
        .sys_clk_n,
        .ddr3_dq,
        .ddr3_dqs_n,
        .ddr3_dqs_p,
        .ddr3_addr,
        .ddr3_ba,
        .ddr3_ras_n,
        .ddr3_cas_n,
        .ddr3_we_n,
        .ddr3_reset_n,
        .ddr3_ck_p,
        .ddr3_ck_n,
        .ddr3_cke,
        .ddr3_cs_n,
        .ddr3_dm,
        .ddr3_odt,
        .mmcm_locked        (),
        .app_sr_req         (1'b0),
        .app_ref_req        (1'b0),
        .app_zq_req         (1'b0),
        .app_sr_active      (),
        .app_ref_ack        (),
        .app_zq_ack         (),
        .ui_clk             (clk),
        .ui_clk_sync_rst    (ui_reset),
        .aresetn            (!ui_reset),
        .s_axi_awid         (m_axi_awid),
        .s_axi_awaddr       (m_axi_awaddr),
        .s_axi_awlen        (m_axi_awlen),
        .s_axi_awsize       (m_axi_awsize),
        .s_axi_awburst      (m_axi_awburst),
        .s_axi_awlock       (m_axi_awlock),
        .s_axi_awcache      (m_axi_awcache),
        .s_axi_awprot       (m_axi_awprot),
        .s_axi_awqos        (m_axi_awqos),
        .s_axi_awvalid      (m_axi_awvalid),
        .s_axi_awready      (m_axi_awready),
        .s_axi_wdata        (m_axi_wdata),
        .s_axi_wstrb        (m_axi_wstrb),
        .s_axi_wlast        (m_axi_wlast),
        .s_axi_wvalid       (m_axi_wvalid),
        .s_axi_wready       (m_axi_wready),
        .s_axi_bready       (m_axi_bready),
        .s_axi_bid          (m_axi_bid),
        .s_axi_bresp        (m_axi_bresp),
        .s_axi_bvalid       (m_axi_bvalid),
        .s_axi_arid         (m_axi_arid),
        .s_axi_araddr       (m_axi_araddr),
        .s_axi_arlen        (m_axi_arlen),
        .s_axi_arsize       (m_axi_arsize),
        .s_axi_arburst      (m_axi_arburst),
        .s_axi_arlock       (m_axi_arlock),
        .s_axi_arcache      (m_axi_arcache),
        .s_axi_arprot       (m_axi_arprot),
        .s_axi_arqos        (m_axi_arqos),
        .s_axi_arvalid      (m_axi_arvalid),
        .s_axi_arready      (m_axi_arready),
        .s_axi_rready       (m_axi_rready),
        .s_axi_rid          (m_axi_rid),
        .s_axi_rdata        (m_axi_rdata),
        .s_axi_rresp        (m_axi_rresp),
        .s_axi_rlast        (m_axi_rlast),
        .s_axi_rvalid       (m_axi_rvalid),
        .init_calib_complete(calib_done),
        .device_temp        (),
        .sys_rst            (cpu_resetn)
    );

endmodule

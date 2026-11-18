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

    localparam integer ADDR_WIDTH = 64;
    localparam integer DATA_WIDTH = 256;
    localparam integer ID_WIDTH   = 5;
    localparam integer USER_WIDTH = 1;

    logic clk;
    logic ui_reset;
    logic calib_done;

    AXI_BUS #(
        .AXI_ADDR_WIDTH(ADDR_WIDTH), .AXI_DATA_WIDTH(DATA_WIDTH),
        .AXI_ID_WIDTH(ID_WIDTH), .AXI_USER_WIDTH(USER_WIDTH)
    ) ctrl_bus (), accel_bus (), accel_cut (), mig_bus ();

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

    // ---------------------------------------------------------- accelerator
    logic [31:0] accel_command;
    logic [31:0] accel_status;
    logic        bus_request;
    logic        bus_grant;

    lfm_matvec_accelerator #(
        .ID_WIDTH  (ID_WIDTH),
        .DATA_WIDTH(DATA_WIDTH)
    ) i_accelerator (
        .clk_i        (clk),
        .fp_clk_i     (fp_clk),
        .rst_ni       (rst_n),
        .fp_rst_ni    (fp_rst_sync[2]),
        .command_i    (accel_command),
        .status_o     (accel_status),
        .bus_request_o(bus_request),
        .bus_grant_i  (bus_grant),
        .axi          (accel_bus)
    );

    axi_cut_intf #(
        .BYPASS    (1'b0),
        .ADDR_WIDTH(ADDR_WIDTH),
        .DATA_WIDTH(DATA_WIDTH),
        .ID_WIDTH  (ID_WIDTH),
        .USER_WIDTH(USER_WIDTH)
    ) i_accel_cut (
        .clk_i (clk),
        .rst_ni(rst_n),
        .in    (accel_bus),
        .out   (accel_cut)
    );

    lfm_axi_arbiter i_arbiter (
        .clk_i                (clk),
        .rst_ni               (rst_n),
        .accelerator_request_i(bus_request),
        .accelerator_grant_o  (bus_grant),
        .cpu                  (ctrl_bus),
        .accelerator          (accel_cut),
        .ddr                  (mig_bus)
    );

    // ----------------------------------------------------------- controller
    logic [7:0] ctrl_led;

    solo_ctrl #(.BOOT_HEX(BOOT_HEX)) i_ctrl (
        .clk_i          (clk),
        .rst_ni         (rst_n),
        .calib_done_i   (calib_done),
        .uart_rx_i      (rx),
        .uart_tx_o      (tx),
        .accel_command_o(accel_command),
        .accel_status_i (accel_status),
        .led_o          (ctrl_led),
        .ddr            (ctrl_bus)
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
        .s_axi_awid         (mig_bus.aw_id),
        .s_axi_awaddr       (mig_bus.aw_addr[29:0]),
        .s_axi_awlen        (mig_bus.aw_len),
        .s_axi_awsize       (mig_bus.aw_size),
        .s_axi_awburst      (mig_bus.aw_burst),
        .s_axi_awlock       (mig_bus.aw_lock),
        .s_axi_awcache      (mig_bus.aw_cache),
        .s_axi_awprot       (mig_bus.aw_prot),
        .s_axi_awqos        (mig_bus.aw_qos),
        .s_axi_awvalid      (mig_bus.aw_valid),
        .s_axi_awready      (mig_bus.aw_ready),
        .s_axi_wdata        (mig_bus.w_data),
        .s_axi_wstrb        (mig_bus.w_strb),
        .s_axi_wlast        (mig_bus.w_last),
        .s_axi_wvalid       (mig_bus.w_valid),
        .s_axi_wready       (mig_bus.w_ready),
        .s_axi_bready       (mig_bus.b_ready),
        .s_axi_bid          (mig_bus.b_id),
        .s_axi_bresp        (mig_bus.b_resp),
        .s_axi_bvalid       (mig_bus.b_valid),
        .s_axi_arid         (mig_bus.ar_id),
        .s_axi_araddr       (mig_bus.ar_addr[29:0]),
        .s_axi_arlen        (mig_bus.ar_len),
        .s_axi_arsize       (mig_bus.ar_size),
        .s_axi_arburst      (mig_bus.ar_burst),
        .s_axi_arlock       (mig_bus.ar_lock),
        .s_axi_arcache      (mig_bus.ar_cache),
        .s_axi_arprot       (mig_bus.ar_prot),
        .s_axi_arqos        (mig_bus.ar_qos),
        .s_axi_arvalid      (mig_bus.ar_valid),
        .s_axi_arready      (mig_bus.ar_ready),
        .s_axi_rready       (mig_bus.r_ready),
        .s_axi_rid          (mig_bus.r_id),
        .s_axi_rdata        (mig_bus.r_data),
        .s_axi_rresp        (mig_bus.r_resp),
        .s_axi_rlast        (mig_bus.r_last),
        .s_axi_rvalid       (mig_bus.r_valid),
        .init_calib_complete(calib_done),
        .device_temp        (),
        .sys_rst            (cpu_resetn)
    );
    assign mig_bus.b_user = '0;
    assign mig_bus.r_user = '0;

endmodule

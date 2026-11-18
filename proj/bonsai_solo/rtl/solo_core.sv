// Bonsai solo core: accelerator, register slice, arbiter and controller,
// with the DDR side exposed as flat AXI4 ports (MIG in hardware, a C++
// memory model in simulation).
module solo_core #(
    parameter string BOOT_HEX = "boot.hex"
) (
    input  logic         clk_i,
    input  logic         fp_clk_i,
    input  logic         rst_ni,
    input  logic         fp_rst_ni,
    input  logic         calib_done_i,
    input  logic         rx,
    output logic         tx,
    output logic [7:0]   ctrl_led,

    output logic [4:0]   m_axi_awid,
    output logic [29:0]  m_axi_awaddr,
    output logic [7:0]   m_axi_awlen,
    output logic [2:0]   m_axi_awsize,
    output logic [1:0]   m_axi_awburst,
    output logic         m_axi_awlock,
    output logic [3:0]   m_axi_awcache,
    output logic [2:0]   m_axi_awprot,
    output logic [3:0]   m_axi_awqos,
    output logic         m_axi_awvalid,
    input  logic         m_axi_awready,
    output logic [255:0] m_axi_wdata,
    output logic [31:0]  m_axi_wstrb,
    output logic         m_axi_wlast,
    output logic         m_axi_wvalid,
    input  logic         m_axi_wready,
    output logic         m_axi_bready,
    input  logic [4:0]   m_axi_bid,
    input  logic [1:0]   m_axi_bresp,
    input  logic         m_axi_bvalid,
    output logic [4:0]   m_axi_arid,
    output logic [29:0]  m_axi_araddr,
    output logic [7:0]   m_axi_arlen,
    output logic [2:0]   m_axi_arsize,
    output logic [1:0]   m_axi_arburst,
    output logic         m_axi_arlock,
    output logic [3:0]   m_axi_arcache,
    output logic [2:0]   m_axi_arprot,
    output logic [3:0]   m_axi_arqos,
    output logic         m_axi_arvalid,
    input  logic         m_axi_arready,
    output logic         m_axi_rready,
    input  logic [4:0]   m_axi_rid,
    input  logic [255:0] m_axi_rdata,
    input  logic [1:0]   m_axi_rresp,
    input  logic         m_axi_rlast,
    input  logic         m_axi_rvalid
);

    localparam integer ADDR_WIDTH = 64;
    localparam integer DATA_WIDTH = 256;
    localparam integer ID_WIDTH   = 5;
    localparam integer USER_WIDTH = 1;

    wire clk = clk_i;
    wire rst_n = rst_ni;

    // rst_ni reaches the accelerator under a 4-cycle multicycle constraint
    // (scripts/timing_hook.tcl); the controller leaves reset 6 cycles later
    // through this short chain, so the CPU only starts once every
    // accelerator register is out of reset.
    logic [5:0] ctrl_rst_chain;
    always_ff @(posedge clk or negedge rst_ni) begin
        if (!rst_ni) ctrl_rst_chain <= '0;
        else ctrl_rst_chain <= {ctrl_rst_chain[4:0], 1'b1};
    end
    wire ctrl_rst_n = ctrl_rst_chain[5];

    AXI_BUS #(
        .AXI_ADDR_WIDTH(ADDR_WIDTH), .AXI_DATA_WIDTH(DATA_WIDTH),
        .AXI_ID_WIDTH(ID_WIDTH), .AXI_USER_WIDTH(USER_WIDTH)
    ) ctrl_bus (), accel_bus (), accel_cut (), mig_bus ();

    // ---------------------------------------------------------- accelerator
    logic [31:0] accel_command;
    logic [31:0] accel_status;
    logic        bus_request;
    logic        bus_grant;

    lfm_matvec_accelerator #(
        .ID_WIDTH  (ID_WIDTH),
        .DATA_WIDTH(DATA_WIDTH)
    ) i_lfm_matvec_accelerator (
        .clk_i        (clk),
        .fp_clk_i     (fp_clk_i),
        .rst_ni       (rst_n),
        .fp_rst_ni    (fp_rst_ni),
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
    solo_ctrl #(.BOOT_HEX(BOOT_HEX)) i_ctrl (
        .clk_i          (clk),
        .rst_ni         (ctrl_rst_n),
        .calib_done_i   (calib_done_i),
        .uart_rx_i      (rx),
        .uart_tx_o      (tx),
        .accel_command_o(accel_command),
        .accel_status_i (accel_status),
        .led_o          (ctrl_led),
        .ddr            (ctrl_bus)
    );


    assign m_axi_awid    = mig_bus.aw_id;
    assign m_axi_awaddr  = mig_bus.aw_addr[29:0];
    assign m_axi_awlen   = mig_bus.aw_len;
    assign m_axi_awsize  = mig_bus.aw_size;
    assign m_axi_awburst = mig_bus.aw_burst;
    assign m_axi_awlock  = mig_bus.aw_lock;
    assign m_axi_awcache = mig_bus.aw_cache;
    assign m_axi_awprot  = mig_bus.aw_prot;
    assign m_axi_awqos   = mig_bus.aw_qos;
    assign m_axi_awvalid = mig_bus.aw_valid;
    assign mig_bus.aw_ready = m_axi_awready;
    assign m_axi_wdata   = mig_bus.w_data;
    assign m_axi_wstrb   = mig_bus.w_strb;
    assign m_axi_wlast   = mig_bus.w_last;
    assign m_axi_wvalid  = mig_bus.w_valid;
    assign mig_bus.w_ready = m_axi_wready;
    assign m_axi_bready  = mig_bus.b_ready;
    assign mig_bus.b_id    = m_axi_bid;
    assign mig_bus.b_resp  = m_axi_bresp;
    assign mig_bus.b_valid = m_axi_bvalid;
    assign mig_bus.b_user  = '0;
    assign m_axi_arid    = mig_bus.ar_id;
    assign m_axi_araddr  = mig_bus.ar_addr[29:0];
    assign m_axi_arlen   = mig_bus.ar_len;
    assign m_axi_arsize  = mig_bus.ar_size;
    assign m_axi_arburst = mig_bus.ar_burst;
    assign m_axi_arlock  = mig_bus.ar_lock;
    assign m_axi_arcache = mig_bus.ar_cache;
    assign m_axi_arprot  = mig_bus.ar_prot;
    assign m_axi_arqos   = mig_bus.ar_qos;
    assign m_axi_arvalid = mig_bus.ar_valid;
    assign mig_bus.ar_ready = m_axi_arready;
    assign m_axi_rready  = mig_bus.r_ready;
    assign mig_bus.r_id    = m_axi_rid;
    assign mig_bus.r_data  = m_axi_rdata;
    assign mig_bus.r_resp  = m_axi_rresp;
    assign mig_bus.r_last  = m_axi_rlast;
    assign mig_bus.r_valid = m_axi_rvalid;
    assign mig_bus.r_user  = '0;

endmodule

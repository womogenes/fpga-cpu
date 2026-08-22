module lfm_accelerator_timing_top (
    input  logic        clk_i,
    input  logic        fp_clk_i,
    input  logic        rst_ni,
    input  logic [31:0] command_i,
    output logic [31:0] status_o,
    output logic        bus_request_o,
    input  logic        bus_grant_i,
    output logic [4:0]  aw_id_o,
    output logic [63:0] aw_addr_o,
    output logic [7:0]  aw_len_o,
    output logic [2:0]  aw_size_o,
    output logic [1:0]  aw_burst_o,
    output logic        aw_lock_o,
    output logic [3:0]  aw_cache_o,
    output logic [2:0]  aw_prot_o,
    output logic [3:0]  aw_qos_o,
    output logic [3:0]  aw_region_o,
    output logic [5:0]  aw_atop_o,
    output logic        aw_user_o,
    output logic        aw_valid_o,
    input  logic        aw_ready_i,
    output logic [63:0] w_data_o,
    output logic [7:0]  w_strb_o,
    output logic        w_last_o,
    output logic        w_user_o,
    output logic        w_valid_o,
    input  logic        w_ready_i,
    input  logic [4:0]  b_id_i,
    input  logic [1:0]  b_resp_i,
    input  logic        b_user_i,
    input  logic        b_valid_i,
    output logic        b_ready_o,
    output logic [4:0]  ar_id_o,
    output logic [63:0] ar_addr_o,
    output logic [7:0]  ar_len_o,
    output logic [2:0]  ar_size_o,
    output logic [1:0]  ar_burst_o,
    output logic        ar_lock_o,
    output logic [3:0]  ar_cache_o,
    output logic [2:0]  ar_prot_o,
    output logic [3:0]  ar_qos_o,
    output logic [3:0]  ar_region_o,
    output logic        ar_user_o,
    output logic        ar_valid_o,
    input  logic        ar_ready_i,
    input  logic [4:0]  r_id_i,
    input  logic [63:0] r_data_i,
    input  logic [1:0]  r_resp_i,
    input  logic        r_last_i,
    input  logic        r_user_i,
    input  logic        r_valid_i,
    output logic        r_ready_o
);

    AXI_BUS #(
        .AXI_ADDR_WIDTH(64),
        .AXI_DATA_WIDTH(64),
        .AXI_ID_WIDTH(5),
        .AXI_USER_WIDTH(1)
    ) axi();

    assign aw_id_o     = axi.aw_id;
    assign aw_addr_o   = axi.aw_addr;
    assign aw_len_o    = axi.aw_len;
    assign aw_size_o   = axi.aw_size;
    assign aw_burst_o  = axi.aw_burst;
    assign aw_lock_o   = axi.aw_lock;
    assign aw_cache_o  = axi.aw_cache;
    assign aw_prot_o   = axi.aw_prot;
    assign aw_qos_o    = axi.aw_qos;
    assign aw_region_o = axi.aw_region;
    assign aw_atop_o   = axi.aw_atop;
    assign aw_user_o   = axi.aw_user;
    assign aw_valid_o  = axi.aw_valid;
    assign axi.aw_ready = aw_ready_i;
    assign w_data_o    = axi.w_data;
    assign w_strb_o    = axi.w_strb;
    assign w_last_o    = axi.w_last;
    assign w_user_o    = axi.w_user;
    assign w_valid_o   = axi.w_valid;
    assign axi.w_ready = w_ready_i;
    assign axi.b_id    = b_id_i;
    assign axi.b_resp  = b_resp_i;
    assign axi.b_user  = b_user_i;
    assign axi.b_valid = b_valid_i;
    assign b_ready_o   = axi.b_ready;
    assign ar_id_o     = axi.ar_id;
    assign ar_addr_o   = axi.ar_addr;
    assign ar_len_o    = axi.ar_len;
    assign ar_size_o   = axi.ar_size;
    assign ar_burst_o  = axi.ar_burst;
    assign ar_lock_o   = axi.ar_lock;
    assign ar_cache_o  = axi.ar_cache;
    assign ar_prot_o   = axi.ar_prot;
    assign ar_qos_o    = axi.ar_qos;
    assign ar_region_o = axi.ar_region;
    assign ar_user_o   = axi.ar_user;
    assign ar_valid_o  = axi.ar_valid;
    assign axi.ar_ready = ar_ready_i;
    assign axi.r_id    = r_id_i;
    assign axi.r_data  = r_data_i;
    assign axi.r_resp  = r_resp_i;
    assign axi.r_last  = r_last_i;
    assign axi.r_user  = r_user_i;
    assign axi.r_valid = r_valid_i;
    assign r_ready_o   = axi.r_ready;

    lfm_matvec_accelerator #(
        .ID_WIDTH(5)
    ) i_accelerator (
        .clk_i,
        .fp_clk_i,
        .rst_ni,
        .fp_rst_ni(rst_ni),
        .command_i,
        .status_o,
        .bus_request_o,
        .bus_grant_i,
        .axi
    );

endmodule

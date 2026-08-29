module IOBUF (
    input wire T,
    input wire I,
    inout wire IO,
    output wire O
);
    assign IO = T ? 1'bz : I;
    assign O = IO;
endmodule

module native_sdio_wrapper_lint;
    logic clk;
    logic rst_n;
    tri sd_cmd;
    tri [3:0] sd_dat;
    logic sd_clk;
    logic sd_reset;
    logic sd_cd;
    logic irq;

    AXI_BUS #(
        .AXI_ADDR_WIDTH(64),
        .AXI_DATA_WIDTH(64),
        .AXI_ID_WIDTH(2),
        .AXI_USER_WIDTH(1)
    ) regs();

    AXI_BUS #(
        .AXI_ADDR_WIDTH(64),
        .AXI_DATA_WIDTH(64),
        .AXI_ID_WIDTH(2),
        .AXI_USER_WIDTH(1)
    ) dma();

    bonsai_native_sdio #(
        .RegAxiIdWidth(2),
        .DmaAxiIdWidth(2),
        .RegAxiDataWidth(64)
    ) i_dut (
        .clk_i(clk),
        .rst_ni(rst_n),
        .regs(regs),
        .dma(dma),
        .sd_cmd(sd_cmd),
        .sd_dat(sd_dat),
        .sd_clk(sd_clk),
        .sd_reset(sd_reset),
        .sd_cd(sd_cd),
        .irq_o(irq)
    );
endmodule

// Exclusive two-master arbiter for the CVA6 DDR UI.
//
// The CPU normally owns DDR. When the accelerator requests it, new CPU address
// transactions are stopped and already accepted transactions are drained. The
// accelerator is granted the complete AXI bus only after that drain completes.
// This keeps the accelerator out of CVA6's ID-sensitive main crossbar.
module lfm_axi_arbiter (
    input  logic clk_i,
    input  logic rst_ni,
    input  logic accelerator_request_i,
    output logic accelerator_grant_o,
    AXI_BUS.Slave cpu,
    AXI_BUS.Slave accelerator,
    AXI_BUS.Master ddr
);

    logic [7:0] cpu_reads_outstanding;
    logic [7:0] cpu_writes_outstanding;
    logic [7:0] cpu_write_data_pending;

    always_ff @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
            accelerator_grant_o   <= 1'b0;
            cpu_reads_outstanding <= '0;
            cpu_writes_outstanding <= '0;
            cpu_write_data_pending <= '0;
        end else begin
            case ({cpu.ar_valid && cpu.ar_ready,
                   cpu.r_valid && cpu.r_ready && cpu.r_last})
                2'b10: cpu_reads_outstanding <= cpu_reads_outstanding + 1'b1;
                2'b01: cpu_reads_outstanding <= cpu_reads_outstanding - 1'b1;
                default: cpu_reads_outstanding <= cpu_reads_outstanding;
            endcase
            case ({cpu.aw_valid && cpu.aw_ready,
                   cpu.b_valid && cpu.b_ready})
                2'b10: cpu_writes_outstanding <= cpu_writes_outstanding + 1'b1;
                2'b01: cpu_writes_outstanding <= cpu_writes_outstanding - 1'b1;
                default: cpu_writes_outstanding <= cpu_writes_outstanding;
            endcase
            case ({cpu.aw_valid && cpu.aw_ready,
                   cpu.w_valid && cpu.w_ready && cpu.w_last})
                2'b10: cpu_write_data_pending <= cpu_write_data_pending + 1'b1;
                2'b01: cpu_write_data_pending <= cpu_write_data_pending - 1'b1;
                default: cpu_write_data_pending <= cpu_write_data_pending;
            endcase

            if (!accelerator_grant_o) begin
                if (accelerator_request_i &&
                    cpu_reads_outstanding == 0 &&
                    cpu_writes_outstanding == 0 &&
                    cpu_write_data_pending == 0)
                    accelerator_grant_o <= 1'b1;
            end else if (!accelerator_request_i) begin
                accelerator_grant_o <= 1'b0;
            end
        end
    end

    always_comb begin
        ddr.aw_id     = '0;
        ddr.aw_addr   = '0;
        ddr.aw_len    = '0;
        ddr.aw_size   = '0;
        ddr.aw_burst  = '0;
        ddr.aw_lock   = '0;
        ddr.aw_cache  = '0;
        ddr.aw_prot   = '0;
        ddr.aw_qos    = '0;
        ddr.aw_region = '0;
        ddr.aw_atop   = '0;
        ddr.aw_user   = '0;
        ddr.aw_valid  = 1'b0;
        ddr.w_data    = '0;
        ddr.w_strb    = '0;
        ddr.w_last    = 1'b0;
        ddr.w_user    = '0;
        ddr.w_valid   = 1'b0;
        ddr.b_ready   = 1'b0;
        ddr.ar_id     = '0;
        ddr.ar_addr   = '0;
        ddr.ar_len    = '0;
        ddr.ar_size   = '0;
        ddr.ar_burst  = '0;
        ddr.ar_lock   = '0;
        ddr.ar_cache  = '0;
        ddr.ar_prot   = '0;
        ddr.ar_qos    = '0;
        ddr.ar_region = '0;
        ddr.ar_user   = '0;
        ddr.ar_valid  = 1'b0;
        ddr.r_ready   = 1'b0;

        cpu.aw_ready = 1'b0;
        cpu.w_ready  = 1'b0;
        cpu.b_id     = '0;
        cpu.b_resp   = '0;
        cpu.b_user   = '0;
        cpu.b_valid  = 1'b0;
        cpu.ar_ready = 1'b0;
        cpu.r_id     = '0;
        cpu.r_data   = '0;
        cpu.r_resp   = '0;
        cpu.r_last   = 1'b0;
        cpu.r_user   = '0;
        cpu.r_valid  = 1'b0;

        accelerator.aw_ready = 1'b0;
        accelerator.w_ready  = 1'b0;
        accelerator.b_id     = '0;
        accelerator.b_resp   = '0;
        accelerator.b_user   = '0;
        accelerator.b_valid  = 1'b0;
        accelerator.ar_ready = 1'b0;
        accelerator.r_id     = '0;
        accelerator.r_data   = '0;
        accelerator.r_resp   = '0;
        accelerator.r_last   = 1'b0;
        accelerator.r_user   = '0;
        accelerator.r_valid  = 1'b0;

        if (accelerator_grant_o) begin
            ddr.aw_id     = accelerator.aw_id;
            ddr.aw_addr   = accelerator.aw_addr;
            ddr.aw_len    = accelerator.aw_len;
            ddr.aw_size   = accelerator.aw_size;
            ddr.aw_burst  = accelerator.aw_burst;
            ddr.aw_lock   = accelerator.aw_lock;
            ddr.aw_cache  = accelerator.aw_cache;
            ddr.aw_prot   = accelerator.aw_prot;
            ddr.aw_qos    = accelerator.aw_qos;
            ddr.aw_region = accelerator.aw_region;
            ddr.aw_atop   = accelerator.aw_atop;
            ddr.aw_user   = accelerator.aw_user;
            ddr.aw_valid  = accelerator.aw_valid;
            accelerator.aw_ready = ddr.aw_ready;
            ddr.w_data    = accelerator.w_data;
            ddr.w_strb    = accelerator.w_strb;
            ddr.w_last    = accelerator.w_last;
            ddr.w_user    = accelerator.w_user;
            ddr.w_valid   = accelerator.w_valid;
            accelerator.w_ready = ddr.w_ready;
            accelerator.b_id    = ddr.b_id;
            accelerator.b_resp  = ddr.b_resp;
            accelerator.b_user  = ddr.b_user;
            accelerator.b_valid = ddr.b_valid;
            ddr.b_ready   = accelerator.b_ready;
            ddr.ar_id     = accelerator.ar_id;
            ddr.ar_addr   = accelerator.ar_addr;
            ddr.ar_len    = accelerator.ar_len;
            ddr.ar_size   = accelerator.ar_size;
            ddr.ar_burst  = accelerator.ar_burst;
            ddr.ar_lock   = accelerator.ar_lock;
            ddr.ar_cache  = accelerator.ar_cache;
            ddr.ar_prot   = accelerator.ar_prot;
            ddr.ar_qos    = accelerator.ar_qos;
            ddr.ar_region = accelerator.ar_region;
            ddr.ar_user   = accelerator.ar_user;
            ddr.ar_valid  = accelerator.ar_valid;
            accelerator.ar_ready = ddr.ar_ready;
            accelerator.r_id    = ddr.r_id;
            accelerator.r_data  = ddr.r_data;
            accelerator.r_resp  = ddr.r_resp;
            accelerator.r_last  = ddr.r_last;
            accelerator.r_user  = ddr.r_user;
            accelerator.r_valid = ddr.r_valid;
            ddr.r_ready   = accelerator.r_ready;
        end else begin
            ddr.aw_id     = cpu.aw_id;
            ddr.aw_addr   = cpu.aw_addr;
            ddr.aw_len    = cpu.aw_len;
            ddr.aw_size   = cpu.aw_size;
            ddr.aw_burst  = cpu.aw_burst;
            ddr.aw_lock   = cpu.aw_lock;
            ddr.aw_cache  = cpu.aw_cache;
            ddr.aw_prot   = cpu.aw_prot;
            ddr.aw_qos    = cpu.aw_qos;
            ddr.aw_region = cpu.aw_region;
            ddr.aw_atop   = cpu.aw_atop;
            ddr.aw_user   = cpu.aw_user;
            ddr.aw_valid  = cpu.aw_valid && !accelerator_request_i;
            cpu.aw_ready  = ddr.aw_ready && !accelerator_request_i;
            ddr.w_data    = cpu.w_data;
            ddr.w_strb    = cpu.w_strb;
            ddr.w_last    = cpu.w_last;
            ddr.w_user    = cpu.w_user;
            // AXI write address and data channels are independent. Once an
            // accelerator request blocks new AW transactions, forward only W
            // beats belonging to an address that DDR already accepted. This
            // prevents an orphan CPU beat from pairing with an accelerator AW.
            ddr.w_valid   = cpu.w_valid &&
                            (cpu_write_data_pending != 0 ||
                             (cpu.aw_valid && cpu.aw_ready));
            cpu.w_ready   = ddr.w_ready &&
                            (cpu_write_data_pending != 0 ||
                             (cpu.aw_valid && cpu.aw_ready));
            cpu.b_id      = ddr.b_id;
            cpu.b_resp    = ddr.b_resp;
            cpu.b_user    = ddr.b_user;
            cpu.b_valid   = ddr.b_valid;
            ddr.b_ready   = cpu.b_ready;
            ddr.ar_id     = cpu.ar_id;
            ddr.ar_addr   = cpu.ar_addr;
            ddr.ar_len    = cpu.ar_len;
            ddr.ar_size   = cpu.ar_size;
            ddr.ar_burst  = cpu.ar_burst;
            ddr.ar_lock   = cpu.ar_lock;
            ddr.ar_cache  = cpu.ar_cache;
            ddr.ar_prot   = cpu.ar_prot;
            ddr.ar_qos    = cpu.ar_qos;
            ddr.ar_region = cpu.ar_region;
            ddr.ar_user   = cpu.ar_user;
            ddr.ar_valid  = cpu.ar_valid && !accelerator_request_i;
            cpu.ar_ready  = ddr.ar_ready && !accelerator_request_i;
            cpu.r_id      = ddr.r_id;
            cpu.r_data    = ddr.r_data;
            cpu.r_resp    = ddr.r_resp;
            cpu.r_last    = ddr.r_last;
            cpu.r_user    = ddr.r_user;
            cpu.r_valid   = ddr.r_valid;
            ddr.r_ready   = cpu.r_ready;
        end
    end

endmodule

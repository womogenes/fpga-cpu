// AXI4 burst reader with a byte-wide output stream. Addresses and lengths must
// be eight-byte aligned. Bursts never cross a 4 KiB boundary.
module lfm_axi_byte_reader #(
    parameter integer ID_WIDTH = 5
) (
    input  logic                clk_i,
    input  logic                rst_ni,
    input  logic                start_i,
    input  logic [29:0]         address_i,
    input  logic [31:0]         byte_count_i,
    input  logic                word_mode_i,
    output logic                busy_o,
    output logic                done_o,
    output logic                error_o,
    output logic [15:0]         data_o,
    output logic                byte_valid_o,
    input  logic                byte_ready_i,
    output logic [ID_WIDTH-1:0] arid_o,
    output logic [29:0]         araddr_o,
    output logic [7:0]          arlen_o,
    output logic [2:0]          arsize_o,
    output logic [1:0]          arburst_o,
    output logic                arvalid_o,
    input  logic                arready_i,
    input  logic [ID_WIDTH-1:0] rid_i,
    input  logic [63:0]         rdata_i,
    input  logic [1:0]          rresp_i,
    input  logic                rlast_i,
    input  logic                rvalid_i,
    output logic                rready_o
);

    logic [29:0] next_address;
    logic [31:0] beats_unissued;
    logic [8:0]  burst_beats;
    logic [8:0]  beats_in_flight;
    logic [63:0] beat_data;
    logic [2:0]  byte_index;
    logic        beat_valid;
    logic        last_unit;
    logic        consume_last;

    function automatic [8:0] choose_burst(
        input logic [29:0] address,
        input logic [31:0] remaining
    );
        logic [8:0] until_boundary;
        logic [8:0] result;
        begin
            until_boundary = (13'd4096 - {1'b0, address[11:0]}) >> 3;
            if (until_boundary == 0)
                until_boundary = 9'd256;
            result = remaining > 256 ? 9'd256 : remaining[8:0];
            if (result > until_boundary)
                result = until_boundary;
            choose_burst = result;
        end
    endfunction

    assign arid_o       = {ID_WIDTH{1'b1}};
    assign arsize_o     = 3'd3;
    assign arburst_o    = 2'b01;
    assign data_o       = beat_data[15:0];
    assign byte_valid_o = beat_valid;
    assign last_unit    = (!word_mode_i && byte_index == 3'd7) ||
                          (word_mode_i && byte_index == 3'd6);
    assign consume_last = beat_valid && byte_ready_i && last_unit;
    // Replace a consumed beat directly from RDATA when the next AXI beat is
    // already waiting. This removes the former one-cycle bubble per 64 bits.
    assign rready_o     = busy_o && (!beat_valid || consume_last);

    always_ff @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
            busy_o          <= 1'b0;
            done_o          <= 1'b0;
            error_o         <= 1'b0;
            next_address    <= '0;
            beats_unissued  <= '0;
            burst_beats     <= '0;
            beats_in_flight <= '0;
            beat_data       <= '0;
            byte_index      <= '0;
            beat_valid      <= 1'b0;
            araddr_o        <= '0;
            arlen_o         <= '0;
            arvalid_o       <= 1'b0;
        end else begin
            done_o <= 1'b0;

            if (start_i && !busy_o) begin
                busy_o          <= 1'b1;
                error_o         <= (address_i[2:0] != 0) ||
                                   (byte_count_i[2:0] != 0) ||
                                   (byte_count_i == 0);
                next_address    <= address_i;
                beats_unissued  <= byte_count_i >> 3;
                burst_beats     <= '0;
                beats_in_flight <= '0;
                beat_valid      <= 1'b0;
                arvalid_o       <= 1'b0;
            end

            if (busy_o && !arvalid_o && beats_in_flight == 0 &&
                beats_unissued != 0) begin
                burst_beats <= choose_burst(next_address, beats_unissued);
                araddr_o    <= next_address;
                arlen_o     <= choose_burst(next_address, beats_unissued) - 1'b1;
                arvalid_o   <= 1'b1;
            end

            if (arvalid_o && arready_i) begin
                arvalid_o       <= 1'b0;
                beats_in_flight <= burst_beats;
                beats_unissued  <= beats_unissued - burst_beats;
                next_address    <= next_address + (burst_beats << 3);
            end

            if (rvalid_i && rready_o) begin
                beat_data  <= rdata_i;
                byte_index <= 3'd0;
                beat_valid <= 1'b1;
                if (rresp_i != 2'b00 || rid_i != {ID_WIDTH{1'b1}})
                    error_o <= 1'b1;
                if (beats_in_flight == 1) begin
                    if (!rlast_i)
                        error_o <= 1'b1;
                end else if (rlast_i) begin
                    error_o <= 1'b1;
                end
                beats_in_flight <= beats_in_flight - 1'b1;
            end

            if (beat_valid && byte_ready_i) begin
                if (last_unit) begin
                    if (!(rvalid_i && rready_o)) begin
                        beat_valid <= 1'b0;
                        if (beats_unissued == 0 && beats_in_flight == 0 &&
                            !arvalid_o) begin
                            busy_o <= 1'b0;
                            done_o <= 1'b1;
                        end
                    end
                end else begin
                    if (word_mode_i) begin
                        beat_data  <= {16'b0, beat_data[63:16]};
                        byte_index <= byte_index + 2'd2;
                    end else begin
                        beat_data  <= {8'b0, beat_data[63:8]};
                        byte_index <= byte_index + 1'b1;
                    end
                end
            end
        end
    end

endmodule

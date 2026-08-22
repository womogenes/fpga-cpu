// AXI4 burst reader with a variable-width little-endian output stream.
// Addresses must be eight-byte aligned; lengths may end on any byte. Bursts
// never cross a 4 KiB boundary. A 16-byte elastic buffer permits unaligned
// 1-, 2-, and 8-byte units while retaining the full 64-bit AXI bandwidth.
module lfm_axi_byte_reader #(
    parameter integer ID_WIDTH = 5
) (
    input  logic                clk_i,
    input  logic                rst_ni,
    input  logic                start_i,
    input  logic [29:0]         address_i,
    input  logic [31:0]         byte_count_i,
    input  logic [3:0]          unit_bytes_i,
    output logic                busy_o,
    output logic                done_o,
    output logic                error_o,
    output logic [63:0]         data_o,
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

    logic [29:0]  next_address;
    logic [31:0]  beats_unissued;
    logic [31:0]  bytes_unreceived;
    logic [8:0]   burst_beats;
    logic [9:0]   beats_in_flight;
    logic [1:0]   bursts_in_flight;
    logic [8:0]   current_burst_beats;
    logic [8:0]   queued_burst_beats;
    logic [127:0] data_buffer;
    logic [4:0]   buffered_bytes;
    logic         consume_unit;
    logic [4:0]   bytes_after_consume;
    logic [127:0] buffer_after_consume;
    logic         accept_beat;
    logic [4:0]   accepted_bytes;

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
    assign data_o       = data_buffer[63:0];
    assign byte_valid_o = buffered_bytes >= unit_bytes_i && unit_bytes_i != 0;
    assign consume_unit = byte_valid_o && byte_ready_i;
    assign bytes_after_consume = buffered_bytes -
                                 (consume_unit ? unit_bytes_i : 4'd0);
    assign buffer_after_consume = consume_unit ?
                                  (data_buffer >> (unit_bytes_i * 8)) :
                                  data_buffer;
    // Drive AXI ready only from registered reader state.  Depending on the
    // current consumer handshake here creates a long combinational path from
    // the matrix engine through the reader and into the AXI register slice.
    // With 8-byte units, an accepted beat and consumed unit leave the buffer
    // at eight bytes, so this still sustains one 64-bit beat per cycle.  For
    // narrower units the consumer is already slower than the AXI input.
    assign rready_o = busy_o && buffered_bytes <= 8;
    assign accept_beat = rvalid_i && rready_o;
    assign accepted_bytes = bytes_unreceived >= 8 ? 5'd8 :
                            {1'b0, bytes_unreceived[3:0]};

    always_ff @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
            busy_o          <= 1'b0;
            done_o          <= 1'b0;
            error_o         <= 1'b0;
            next_address    <= '0;
            beats_unissued  <= '0;
            bytes_unreceived <= '0;
            burst_beats     <= '0;
            beats_in_flight <= '0;
            bursts_in_flight <= '0;
            current_burst_beats <= '0;
            queued_burst_beats <= '0;
            data_buffer     <= '0;
            buffered_bytes  <= '0;
            araddr_o        <= '0;
            arlen_o         <= '0;
            arvalid_o       <= 1'b0;
        end else begin
            done_o <= 1'b0;

            // Keep an error visible through the completion pulse so the
            // consumer can latch it, then retire it while idle.  A later
            // command must not inherit a completed command's AXI error.
            if (!busy_o && !start_i)
                error_o <= 1'b0;

            if (start_i && !busy_o) begin
                busy_o          <= 1'b1;
                error_o         <= (address_i[2:0] != 0) ||
                                   (byte_count_i == 0) ||
                                   !((unit_bytes_i == 1) ||
                                     (unit_bytes_i == 2) ||
                                     (unit_bytes_i == 8));
                next_address    <= address_i;
                beats_unissued  <= (byte_count_i + 7) >> 3;
                bytes_unreceived <= byte_count_i;
                burst_beats     <= '0;
                beats_in_flight <= '0;
                bursts_in_flight <= '0;
                current_burst_beats <= '0;
                queued_burst_beats <= '0;
                data_buffer     <= '0;
                buffered_bytes  <= '0;
                arvalid_o       <= 1'b0;
            end

            /* Keep one burst queued behind the active response stream. AXI
               preserves response order for our single ID, so this removes
               the DDR command gap at every 2 KiB boundary without requiring
               a data reorder buffer. */
            if (busy_o && !arvalid_o && bursts_in_flight < 2 &&
                beats_unissued != 0) begin
                burst_beats <= choose_burst(next_address, beats_unissued);
                araddr_o    <= next_address;
                arlen_o     <= choose_burst(next_address, beats_unissued) - 1'b1;
                arvalid_o   <= 1'b1;
            end

            if (arvalid_o && arready_i) begin
                arvalid_o       <= 1'b0;
                beats_unissued  <= beats_unissued - burst_beats;
                next_address    <= next_address + (burst_beats << 3);
                if (bursts_in_flight == 0) begin
                    current_burst_beats <= burst_beats;
                end else begin
                    queued_burst_beats <= burst_beats;
                end
            end

            if (consume_unit || accept_beat) begin
                if (accept_beat)
                    data_buffer <= buffer_after_consume |
                                   ({64'b0, rdata_i} << (bytes_after_consume * 8));
                else
                    data_buffer <= buffer_after_consume;
                buffered_bytes <= bytes_after_consume +
                                  (accept_beat ? accepted_bytes : 5'd0);
            end

            if (accept_beat) begin
                bytes_unreceived <= bytes_unreceived - accepted_bytes;
                if (rresp_i != 2'b00 || rid_i != {ID_WIDTH{1'b1}})
                    error_o <= 1'b1;
                if (current_burst_beats == 1) begin
                    if (!rlast_i)
                        error_o <= 1'b1;
                end else if (rlast_i) begin
                    error_o <= 1'b1;
                end
                if (current_burst_beats == 1) begin
                    current_burst_beats <= queued_burst_beats;
                    queued_burst_beats <= '0;
                end else begin
                    current_burst_beats <= current_burst_beats - 1'b1;
                end
            end

            // Beat occupancy and burst occupancy have different retirement
            // events.  The old combined case used "last beat" for both.  If
            // a queued AR was accepted alongside a non-last R beat, it added
            // the new burst without subtracting the R beat, leaving a phantom
            // beat in flight and keeping busy_o asserted forever.
            case ({arvalid_o && arready_i, accept_beat})
                2'b10: begin
                    beats_in_flight <= beats_in_flight + burst_beats;
                end
                2'b01: begin
                    beats_in_flight <= beats_in_flight - 1'b1;
                end
                2'b11: begin
                    beats_in_flight <= beats_in_flight + burst_beats - 1'b1;
                end
                default: begin end
            endcase

            case ({arvalid_o && arready_i,
                   accept_beat && current_burst_beats == 1})
                2'b10: begin
                    bursts_in_flight <= bursts_in_flight + 1'b1;
                end
                2'b01: begin
                    bursts_in_flight <= bursts_in_flight - 1'b1;
                end
                2'b11: begin
                    /* One burst completes as another is accepted. */
                    bursts_in_flight <= bursts_in_flight;
                    if (bursts_in_flight == 1) begin
                        current_burst_beats <= burst_beats;
                        queued_burst_beats <= '0;
                    end else begin
                        current_burst_beats <= queued_burst_beats;
                        queued_burst_beats <= burst_beats;
                    end
                end
                default: begin end
            endcase

            if (consume_unit && bytes_after_consume == 0 && !accept_beat &&
                beats_unissued == 0 && beats_in_flight == 0 && !arvalid_o) begin
                busy_o <= 1'b0;
                done_o <= 1'b1;
            end
        end
    end

endmodule

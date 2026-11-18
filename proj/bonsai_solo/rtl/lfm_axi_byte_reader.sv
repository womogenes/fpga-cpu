// AXI4 burst reader with a variable-width little-endian output stream.
// Addresses must be beat-aligned; lengths may end on any byte. Bursts never
// cross a 4 KiB boundary. A two-beat elastic buffer permits 1-, 2-, 8-, and
// full-beat units while retaining the full AXI bandwidth.
module lfm_axi_byte_reader #(
    parameter integer ID_WIDTH = 5,
    parameter integer DATA_WIDTH = 128
) (
    input  logic                clk_i,
    input  logic                rst_ni,
    input  logic                start_i,
    input  logic [29:0]         address_i,
    input  logic [31:0]         byte_count_i,
    input  logic [5:0]          unit_bytes_i,
    output logic                busy_o,
    output logic                done_o,
    output logic                error_o,
    output logic [DATA_WIDTH-1:0] data_o,
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
    input  logic [DATA_WIDTH-1:0] rdata_i,
    input  logic [1:0]          rresp_i,
    input  logic                rlast_i,
    input  logic                rvalid_i,
    output logic                rready_o
);

    logic [29:0]  next_address;
    logic [31:0]  beats_unissued;
    logic [31:0]  bytes_unreceived;
    logic [8:0]   burst_beats;
    logic [10:0]  beats_in_flight;
    logic [3:0]   bursts_in_flight;
    logic [8:0]   current_burst_beats;
    logic [8:0]   queued_burst_beats;
    localparam integer DATA_BYTES = DATA_WIDTH / 8;
    localparam integer BEAT_SHIFT = $clog2(DATA_BYTES);
    logic [DATA_WIDTH-1:0] beat_data0, beat_data1;
    // These metadata bits feed both the local availability predicates and
    // several physically distant accelerator consumers.  Bound their fanout
    // so implementation can retain local replicas instead of routing one
    // carry-chain source across the entire accelerator region.
    (* max_fanout = 16 *) logic [5:0] beat_bytes0, beat_bytes1;
    (* max_fanout = 8 *) logic beat_read_pointer;
    logic         beat_write_pointer;
    logic [DATA_WIDTH-1:0] head_beat_data, second_beat_data;
    logic [5:0]   head_beat_bytes, second_beat_bytes;
    (* max_fanout = 16 *) logic [5:0] beat_offset;
    logic [1:0]   buffered_beats;
    logic [2*DATA_WIDTH-1:0] beat_window;
    logic [5:0]   head_remaining;
    logic         offset_in_head;
    logic         enough_unit;
    logic         enough_1, enough_2, enough_4;
    logic         enough_8, enough_16, enough_32;
    logic         consume_unit;
    logic         pop_beat;
    logic         pop_two_beats;
    logic         accept_beat;
    logic [6:0]   accepted_bytes;

    function automatic [8:0] choose_burst(
        input logic [29:0] address,
        input logic [31:0] remaining
    );
        logic [8:0] until_boundary;
        logic [8:0] result;
        begin
            until_boundary = (13'd4096 - {1'b0, address[11:0]}) >> BEAT_SHIFT;
            if (until_boundary == 0)
                until_boundary = 9'd256;
            result = remaining > 256 ? 9'd256 : remaining[8:0];
            if (result > until_boundary)
                result = until_boundary;
            choose_burst = result;
        end
    endfunction

    assign arid_o       = {ID_WIDTH{1'b1}};
    assign arsize_o     = BEAT_SHIFT;
    assign arburst_o    = 2'b01;
    // Retain beats as a two-entry FIFO and select from their concatenated
    // read-only window instead of shifting and reinserting a wide buffer
    // on every clock. Q4/Q6 switch unit widths within a beat, so a unit may
    // straddle the FIFO entries even though each individual width divides 16.
    // The output selector is not part of the FIFO register feedback path.
    // Use independent one-bit read/write pointers so the wide payload write
    // enables depend only on the AXI handshake and a single registered bit.
    // The former shift FIFO selected beat_data0/1 from buffered_beats through
    // the complete byte-valid/retirement cone.  On the routed 200 MHz design
    // that became an eight-level buffered_beats -> beat_data CE path.
    assign head_beat_data = beat_read_pointer ? beat_data1 : beat_data0;
    assign second_beat_data = beat_read_pointer ? beat_data0 : beat_data1;
    assign head_beat_bytes = beat_read_pointer ? beat_bytes1 : beat_bytes0;
    assign second_beat_bytes = beat_read_pointer ? beat_bytes0 : beat_bytes1;
    assign beat_window = {second_beat_data, head_beat_data};
    assign data_o = beat_window >> (beat_offset * 8);
    // Compute availability for each supported constant width independently.
    // Empty-FIFO transition cycles deliberately retain payload metadata, so
    // explicitly validate the offset before using the unsigned remainder.
    // This prevents stale metadata from becoming a valid unit while retaining
    // one short subtract chain; the wider absolute-end formulation required
    // three CARRY4 levels on the routed consumer-control path.
    assign offset_in_head = beat_offset <= head_beat_bytes;
    assign head_remaining = head_beat_bytes - beat_offset;
    assign enough_1 = (buffered_beats != 0) &&
                      offset_in_head &&
                      ((head_remaining >= 6'd1) ||
                       ((buffered_beats == 2) &&
                        (second_beat_bytes >= 6'd1 - head_remaining)));
    assign enough_2 = (buffered_beats != 0) &&
                      offset_in_head &&
                      ((head_remaining >= 6'd2) ||
                       ((buffered_beats == 2) &&
                        (second_beat_bytes >= 6'd2 - head_remaining)));
    assign enough_4 = (buffered_beats != 0) &&
                      offset_in_head &&
                      ((head_remaining >= 6'd4) ||
                       ((buffered_beats == 2) &&
                        (second_beat_bytes >= 6'd4 - head_remaining)));
    assign enough_8 = (buffered_beats != 0) &&
                      offset_in_head &&
                      ((head_remaining >= 6'd8) ||
                       ((buffered_beats == 2) &&
                        (second_beat_bytes >= 6'd8 - head_remaining)));
    assign enough_16 = (buffered_beats != 0) &&
                       offset_in_head &&
                       ((head_remaining >= 6'd16) ||
                        ((buffered_beats == 2) &&
                         (second_beat_bytes >= 6'd16 - head_remaining)));
    assign enough_32 = (buffered_beats != 0) &&
                       offset_in_head &&
                       ((head_remaining >= 6'd32) ||
                        ((buffered_beats == 2) &&
                         (second_beat_bytes >= 6'd32 - head_remaining)));
    always_comb begin
        case (unit_bytes_i)
            6'd1:  enough_unit = enough_1;
            6'd2:  enough_unit = enough_2;
            6'd4:  enough_unit = enough_4;
            6'd8:  enough_unit = enough_8;
            6'd16: enough_unit = enough_16;
            6'd32: enough_unit = enough_32;
            default: enough_unit = 1'b0;
        endcase
    end
    assign byte_valid_o = enough_unit;
    assign consume_unit = byte_valid_o && byte_ready_i;
    assign pop_beat = consume_unit &&
                      beat_offset + unit_bytes_i >= head_beat_bytes;
    // A variable-width unit may consume the remainder of the head beat and
    // all of a short final beat. Retire both entries in that case; retaining
    // the fully consumed partial beat leaves byte_valid low forever.
    assign pop_two_beats = pop_beat && buffered_beats == 2 &&
                           {1'b0, beat_offset} + {1'b0, unit_bytes_i} >=
                               {1'b0, head_beat_bytes} +
                               {1'b0, second_beat_bytes};
    // Drive AXI ready only from registered reader state.  Depending on the
    // current consumer handshake here creates a long combinational path from
    // the matrix engine through the reader and into the AXI register slice.
    // With full-beat units, an accepted beat and consumed unit leave the buffer
    // at one beat, so this still sustains one AXI beat per cycle. For
    // narrower units the consumer is already slower than the AXI input.
    assign rready_o = busy_o && buffered_beats < 2;
    assign accept_beat = rvalid_i && rready_o;
    assign accepted_bytes = bytes_unreceived >= DATA_BYTES ? DATA_BYTES :
                            bytes_unreceived[6:0];

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
            beat_data0      <= '0;
            beat_data1      <= '0;
            beat_bytes0     <= '0;
            beat_bytes1     <= '0;
            beat_read_pointer <= 1'b0;
            beat_write_pointer <= 1'b0;
            beat_offset     <= '0;
            buffered_beats  <= '0;
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
                // Narrow consumers are eight-byte based, but the DDR port
                // may be wider. Align the AXI request down and skip the
                // leading lanes in the first returned beat.
                // The reader is an internal accelerator component and every
                // caller supplies one of the supported unit widths. Keeping
                // a live legality decoder here lets synthesis share its
                // unit-width comparisons with FIFO retirement, placing ten
                // logic levels on the beat-data clock-enable path. Validate
                // only command properties that can actually vary at runtime.
                error_o         <= (address_i[2:0] != 0) ||
                                   (byte_count_i == 0);
                next_address    <= address_i & ~(DATA_BYTES - 1);
                beats_unissued  <= (byte_count_i +
                                   address_i[BEAT_SHIFT-1:0] +
                                   DATA_BYTES - 1) >> BEAT_SHIFT;
                bytes_unreceived <= byte_count_i +
                                    address_i[BEAT_SHIFT-1:0];
                burst_beats     <= '0;
                beats_in_flight <= '0;
                bursts_in_flight <= '0;
                current_burst_beats <= '0;
                queued_burst_beats <= '0;
                beat_data0      <= '0;
                beat_data1      <= '0;
                beat_bytes0     <= '0;
                beat_bytes1     <= '0;
                beat_read_pointer <= 1'b0;
                beat_write_pointer <= 1'b0;
                beat_offset     <= address_i[BEAT_SHIFT-1:0];
                buffered_beats  <= '0;
                arvalid_o       <= 1'b0;
            end

            /* Queue eight 4 KiB bursts. With ROW_BANK_COLUMN DDR mapping this
               spans four physical banks, allowing all four MIG bank machines
               to hide activate/precharge latency. AXI preserves response
               order for our single ID, and RLAST identifies each retirement,
               so this needs no response reorder buffer. */
            if (busy_o && !arvalid_o && bursts_in_flight < 8 &&
                beats_unissued != 0) begin
                burst_beats <= choose_burst(next_address, beats_unissued);
                araddr_o    <= next_address;
                arlen_o     <= choose_burst(next_address, beats_unissued) - 1'b1;
                arvalid_o   <= 1'b1;
            end

            if (arvalid_o && arready_i) begin
                arvalid_o       <= 1'b0;
                beats_unissued  <= beats_unissued - burst_beats;
                next_address    <= next_address +
                                   (burst_beats << BEAT_SHIFT);
                if (bursts_in_flight == 0) begin
                    current_burst_beats <= burst_beats;
                end else begin
                    queued_burst_beats <= burst_beats;
                end
            end

            if (consume_unit && !pop_beat)
                beat_offset <= beat_offset + unit_bytes_i;
            else if (pop_two_beats)
                beat_offset <= '0;
            else if (pop_beat)
                beat_offset <= beat_offset + unit_bytes_i - head_beat_bytes;

            // At full consumer speed a head beat retires on the same clock
            // that its replacement arrives, keeping occupancy at one and
            // accepting one full AXI beat per clock. A stalled consumer can
            // retain two complete beats without involving a data shifter.
            if (pop_two_beats) begin
                // rready_o is low at occupancy two, so an AXI beat cannot be
                // accepted on the same edge as this double retirement.
                buffered_beats <= '0;
            end else case ({accept_beat, pop_beat})
                2'b10: begin
                    if (!beat_write_pointer) begin
                        beat_data0 <= rdata_i;
                        beat_bytes0 <= accepted_bytes[5:0];
                    end else begin
                        beat_data1 <= rdata_i;
                        beat_bytes1 <= accepted_bytes[5:0];
                    end
                    beat_write_pointer <= ~beat_write_pointer;
                    buffered_beats <= buffered_beats + 1'b1;
                end
                2'b01: begin
                    beat_read_pointer <= ~beat_read_pointer;
                    buffered_beats <= buffered_beats - 1'b1;
                end
                2'b11: begin
                    // At occupancy one the tail slot receiving this beat is
                    // exactly the new head after the simultaneous retirement.
                    if (!beat_write_pointer) begin
                        beat_data0 <= rdata_i;
                        beat_bytes0 <= accepted_bytes[5:0];
                    end else begin
                        beat_data1 <= rdata_i;
                        beat_bytes1 <= accepted_bytes[5:0];
                    end
                    beat_read_pointer <= ~beat_read_pointer;
                    beat_write_pointer <= ~beat_write_pointer;
                    buffered_beats <= buffered_beats;
                end
                default: begin end
            endcase

            if (accept_beat) begin
                bytes_unreceived <= bytes_unreceived - accepted_bytes;
                if (rresp_i != 2'b00 || rid_i != {ID_WIDTH{1'b1}})
                    error_o <= 1'b1;
                /* RLAST is authoritative with multiple same-ID bursts. */
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

            case ({arvalid_o && arready_i, accept_beat && rlast_i})
                2'b10: begin
                    bursts_in_flight <= bursts_in_flight + 1'b1;
                end
                2'b01: begin
                    bursts_in_flight <= bursts_in_flight - 1'b1;
                end
                2'b11: begin
                    /* One burst completes as another is accepted. */
                    bursts_in_flight <= bursts_in_flight;
                end
                default: begin end
            endcase

            // Declare completion from the registered drained state, rather
            // than requiring the FIFO's final pop to coincide with the AXI
            // counters already reading zero.  The final response and final
            // consumer pop can retire on adjacent edges; in that case the
            // old event-qualified test missed completion permanently even
            // though every requested byte had been consumed.
            if (busy_o && buffered_beats == 0 && bytes_unreceived == 0 &&
                beats_unissued == 0 && beats_in_flight == 0 && !arvalid_o &&
                !accept_beat) begin
                busy_o <= 1'b0;
                done_o <= 1'b1;
            end
        end
    end

endmodule

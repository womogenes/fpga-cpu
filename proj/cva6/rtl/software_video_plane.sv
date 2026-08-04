module software_video_plane #(
    parameter integer SOURCE_WIDTH = 160,
    parameter integer SOURCE_HEIGHT = 90,
    parameter integer SCALE_SHIFT = 3
) (
    input  logic        clk,
    input  logic        reset,
    input  logic [31:0] command_bus,
    input  logic        force_terminal,
    input  logic [10:0] pixel_x,
    input  logic [9:0]  pixel_y,
    output logic        ready,
    output logic        active,
    output logic [7:0]  red,
    output logic [7:0]  green,
    output logic [7:0]  blue
);

    localparam integer FRAME_PIXELS = SOURCE_WIDTH * SOURCE_HEIGHT;
    localparam integer FRAME_ADDRESS_WIDTH = $clog2(FRAME_PIXELS);
    localparam integer MEMORY_ADDRESS_WIDTH = $clog2(2 * FRAME_PIXELS);

    logic [6:0] frame_buffer [0:2 * FRAME_PIXELS - 1];

    (* ASYNC_REG = "TRUE" *) logic [31:0] command_meta;
    (* ASYNC_REG = "TRUE" *) logic [31:0] command_sync;
    logic command_toggle;
    logic loading;
    logic unpacking;
    logic [1:0] packet_index;
    logic [27:0] packet_data;
    logic [FRAME_ADDRESS_WIDTH-1:0] write_index;
    logic write_bank;
    logic display_bank;
    logic pending_bank;
    logic pending_swap;
    logic graphics_enabled;

    logic [10:0] next_pixel_x;
    logic [9:0] next_pixel_y;
    logic next_active;
    logic next_bank;
    logic [MEMORY_ADDRESS_WIDTH-1:0] read_address;
    logic pixel_active;
    logic [6:0] pixel_data;

    always_comb begin
        if (pixel_x == 1649) begin
            next_pixel_x = 0;
            next_pixel_y = (pixel_y == 749) ? 0 : pixel_y + 1'b1;
        end else begin
            next_pixel_x = pixel_x + 1'b1;
            next_pixel_y = pixel_y;
        end
        next_active = (next_pixel_x < 1280) && (next_pixel_y < 720);
        next_bank = ((next_pixel_x == 0) && (next_pixel_y == 0) && pending_swap) ? pending_bank : display_bank;
        if (next_active)
            read_address = next_bank * FRAME_PIXELS + (next_pixel_y >> SCALE_SHIFT) * SOURCE_WIDTH + (next_pixel_x >> SCALE_SHIFT);
        else
            read_address = next_bank * FRAME_PIXELS;
    end

    always_ff @(posedge clk) begin
        command_meta <= command_bus;
        command_sync <= command_meta;

        if (reset) begin
            command_toggle <= 0;
            loading <= 0;
            unpacking <= 0;
            packet_index <= 0;
            packet_data <= 0;
            write_index <= 0;
            write_bank <= 1;
            display_bank <= 0;
            pending_bank <= 0;
            pending_swap <= 0;
            graphics_enabled <= 0;
            pixel_active <= 0;
            pixel_data <= 0;
        end else begin
            pixel_active <= next_active;
            pixel_data <= frame_buffer[read_address];

            if ((next_pixel_x == 0) && (next_pixel_y == 0) && pending_swap) begin
                display_bank <= pending_bank;
                pending_swap <= 0;
                graphics_enabled <= 1;
            end

            if (unpacking) begin
                frame_buffer[write_bank * FRAME_PIXELS + write_index] <= packet_data[packet_index * 7 +: 7];
                if (write_index == FRAME_PIXELS - 1) begin
                    unpacking <= 0;
                    loading <= 0;
                    pending_bank <= write_bank;
                    pending_swap <= 1;
                    write_index <= 0;
                end else begin
                    write_index <= write_index + 1'b1;
                    if (packet_index == 3)
                        unpacking <= 0;
                    else
                        packet_index <= packet_index + 1'b1;
                end
            end

            if (command_sync[31] != command_toggle) begin
                command_toggle <= command_sync[31];
                if (loading) begin
                    if (!unpacking && command_sync[30:28] == 3'b000) begin
                        packet_data <= command_sync[27:0];
                        packet_index <= 0;
                        unpacking <= 1;
                    end
                end else if (command_sync[30:28] == 3'b001) begin
                    case (command_sync[6:0])
                        7'h00: begin
                            graphics_enabled <= 0;
                            pending_swap <= 0;
                        end
                        7'h01: begin
                            write_bank <= ~display_bank;
                            write_index <= 0;
                            loading <= 1;
                        end
                        7'h02: graphics_enabled <= 1;
                        default: begin end
                    endcase
                end
            end
        end
    end

    always_comb begin
        ready = !loading && !pending_swap;
        active = graphics_enabled && !force_terminal;
        if (pixel_active) begin
            red = {pixel_data[6:5], pixel_data[6:5], pixel_data[6:5], pixel_data[6:5]};
            green = {pixel_data[4:2], pixel_data[4:2], pixel_data[4:3]};
            blue = {pixel_data[1:0], pixel_data[1:0], pixel_data[1:0], pixel_data[1:0]};
        end else begin
            red = 0;
            green = 0;
            blue = 0;
        end
    end

endmodule

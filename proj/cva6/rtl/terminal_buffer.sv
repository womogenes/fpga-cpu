module terminal_buffer #(
    parameter string FONT_FILE = "../../../rtl/font8x16.mem"
) (
    input  logic        clk,
    input  logic        reset,
    input  logic [7:0]  rx_data,
    input  logic        rx_valid,
    output logic        rx_ready,
    input  logic [10:0] pixel_x,
    input  logic [9:0]  pixel_y,
    output logic [7:0]  red,
    output logic [7:0]  green,
    output logic [7:0]  blue
);

    localparam integer COLUMNS = 160;
    localparam integer ROWS = 45;
    localparam integer CELLS = COLUMNS * ROWS;
    localparam integer TOTAL_CELLS = 2 * CELLS;
    localparam integer FONT_FIRST = 32;
    localparam integer FONT_GLYPHS = 95;
    localparam logic [15:0] DEFAULT_CELL = 16'h0720;

    (* ram_style = "block" *) logic [15:0] text_ram [0:TOTAL_CELLS-1];
    (* rom_style = "distributed" *) logic [7:0] font_rom [0:FONT_GLYPHS*16-1];

    integer initialize_index;
    initial begin
        for (initialize_index = 0; initialize_index < TOTAL_CELLS; initialize_index = initialize_index + 1)
            text_ram[initialize_index] = DEFAULT_CELL;
        $readmemh(FONT_FILE, font_rom);
    end

    function automatic [13:0] screen_base(input logic alternate);
        screen_base = alternate ? CELLS : 0;
    endfunction

    function automatic [13:0] line_offset(input logic [9:0] line);
        logic [13:0] widened;
        begin
            widened = line;
            line_offset = (widened << 7) + (widened << 5);
        end
    endfunction

    function automatic [7:0] count_or_one(input logic [15:0] value, input logic [7:0] maximum);
        logic [15:0] requested;
        begin
            requested = (value == 0) ? 1 : value;
            count_or_one = (requested > maximum) ? maximum : requested[7:0];
        end
    endfunction

    function automatic [8:0] next_sgr(input logic [8:0] old_state, input logic [15:0] code);
        logic inverse;
        logic [3:0] background;
        logic [3:0] foreground;
        begin
            {inverse, background, foreground} = old_state;
            case (code)
                0: begin foreground = 4'h7; background = 4'h0; inverse = 1'b0; end
                1: foreground[3] = 1'b1;
                2, 22: foreground[3] = 1'b0;
                7: inverse = 1'b1;
                27: inverse = 1'b0;
                30, 31, 32, 33, 34, 35, 36, 37: foreground[2:0] = code - 30;
                39: foreground = 4'h7;
                40, 41, 42, 43, 44, 45, 46, 47: background = code - 40;
                49: background = 4'h0;
                90, 91, 92, 93, 94, 95, 96, 97: foreground = code - 82;
                100, 101, 102, 103, 104, 105, 106, 107: background = code - 92;
                default: begin end
            endcase
            next_sgr = {inverse, background, foreground};
        end
    endfunction

    function automatic [23:0] ansi_color(input logic [3:0] index);
        case (index)
            4'h0: ansi_color = 24'h000000;
            4'h1: ansi_color = 24'hcc3333;
            4'h2: ansi_color = 24'h33cc66;
            4'h3: ansi_color = 24'hcccc33;
            4'h4: ansi_color = 24'h3366cc;
            4'h5: ansi_color = 24'hcc33cc;
            4'h6: ansi_color = 24'h33cccc;
            4'h7: ansi_color = 24'he0eeff;
            4'h8: ansi_color = 24'h666666;
            4'h9: ansi_color = 24'hff6666;
            4'ha: ansi_color = 24'h66ff99;
            4'hb: ansi_color = 24'hffff66;
            4'hc: ansi_color = 24'h6699ff;
            4'hd: ansi_color = 24'hff66ff;
            4'he: ansi_color = 24'h66ffff;
            default: ansi_color = 24'hffffff;
        endcase
    endfunction

    logic [7:0] cursor_x;
    logic [5:0] cursor_y;
    logic [5:0] scroll_top;
    logic [5:0] scroll_bottom;
    logic [3:0] current_foreground;
    logic [3:0] current_background;
    logic current_inverse;
    logic cursor_visible;
    logic [2:0] cursor_style;
    logic wrap_enabled;
    logic wrap_pending;
    logic origin_mode;
    logic alternate_screen;

    logic [7:0] saved_cursor_x;
    logic [5:0] saved_cursor_y;
    logic [5:0] saved_scroll_top;
    logic [5:0] saved_scroll_bottom;
    logic [3:0] saved_foreground;
    logic [3:0] saved_background;
    logic saved_inverse;
    logic saved_wrap_pending;
    logic saved_origin_mode;
    logic saved_cursor_visible;
    logic [2:0] saved_cursor_style;

    logic [7:0] normal_cursor_x;
    logic [5:0] normal_cursor_y;
    logic [5:0] normal_scroll_top;
    logic [5:0] normal_scroll_bottom;
    logic [3:0] normal_foreground;
    logic [3:0] normal_background;
    logic normal_inverse;
    logic normal_wrap_enabled;
    logic normal_wrap_pending;
    logic normal_origin_mode;
    logic normal_cursor_visible;
    logic [2:0] normal_cursor_style;

    typedef enum logic [2:0] {TEXT, ESCAPE, CSI, STRING_IGNORE, STRING_ESCAPE, ESCAPE_IGNORE_ONE} parser_state_t;
    parser_state_t parser_state;
    logic [15:0] parameters [0:7];
    logic [2:0] parameter_index;
    logic private_mode;
    logic csi_space;

    typedef enum logic [1:0] {OP_IDLE, OP_CLEAR, OP_COPY} operation_state_t;
    operation_state_t operation_state;
    logic [13:0] operation_source;
    logic [13:0] operation_destination;
    logic [13:0] operation_remaining;
    logic operation_backward;
    logic operation_copy_pending;
    logic [13:0] operation_pending_destination;
    logic [13:0] operation_read_address;
    logic [15:0] operation_read_data;
    logic [13:0] after_clear_address;
    logic [13:0] after_clear_count;

    logic direct_write_valid;
    logic [13:0] direct_write_address;
    logic [15:0] direct_write_data;
    logic deferred_write_valid;
    logic [13:0] deferred_write_address;
    logic [15:0] deferred_write_data;

    logic memory_write_enable;
    logic [13:0] memory_write_address;
    logic [15:0] memory_write_data;

    logic [13:0] render_address;
    logic [15:0] render_cell;
    logic [10:0] render_x;
    logic [9:0] render_y;
    logic render_active;
    logic [10:0] next_pixel_x;
    logic [9:0] next_pixel_y;

    assign rx_ready = (operation_state == OP_IDLE) && !deferred_write_valid;
    assign operation_read_address = operation_source;

    always_comb begin
        memory_write_enable = 1'b0;
        memory_write_address = 0;
        memory_write_data = DEFAULT_CELL;
        if (operation_state == OP_CLEAR) begin
            memory_write_enable = 1'b1;
            memory_write_address = operation_destination;
            memory_write_data = {current_background, current_foreground, 8'h20};
        end else if (operation_state == OP_COPY && operation_copy_pending) begin
            memory_write_enable = 1'b1;
            memory_write_address = operation_pending_destination;
            memory_write_data = operation_read_data;
        end else if (deferred_write_valid) begin
            memory_write_enable = 1'b1;
            memory_write_address = deferred_write_address;
            memory_write_data = deferred_write_data;
        end else if (direct_write_valid) begin
            memory_write_enable = 1'b1;
            memory_write_address = direct_write_address;
            memory_write_data = direct_write_data;
        end
    end

    always_comb begin
        if (pixel_x == 1649) begin
            next_pixel_x = 0;
            next_pixel_y = (pixel_y == 749) ? 0 : pixel_y + 1'b1;
        end else begin
            next_pixel_x = pixel_x + 1'b1;
            next_pixel_y = pixel_y;
        end
        if (next_pixel_x < 1280 && next_pixel_y < 720)
            render_address = screen_base(alternate_screen) + line_offset(next_pixel_y >> 4) + (next_pixel_x >> 3);
        else
            render_address = screen_base(alternate_screen);
    end

    always_ff @(posedge clk) begin
        render_cell <= text_ram[render_address];
        operation_read_data <= text_ram[operation_read_address];
        if (memory_write_enable)
            text_ram[memory_write_address] <= memory_write_data;
        render_x <= next_pixel_x;
        render_y <= next_pixel_y;
        render_active <= (next_pixel_x < 1280) && (next_pixel_y < 720);
    end

    task automatic start_clear(input logic [13:0] address, input logic [13:0] count);
        begin
            if (count != 0) begin
                operation_state <= OP_CLEAR;
                operation_destination <= address;
                operation_remaining <= count;
                operation_copy_pending <= 1'b0;
                after_clear_count <= 0;
            end
        end
    endtask

    task automatic start_copy(
        input logic [13:0] source,
        input logic [13:0] destination,
        input logic [13:0] count,
        input logic backward,
        input logic [13:0] clear_address,
        input logic [13:0] clear_count
    );
        begin
            if (count == 0) begin
                start_clear(clear_address, clear_count);
            end else begin
                operation_state <= OP_COPY;
                operation_source <= backward ? source + count - 1'b1 : source;
                operation_destination <= backward ? destination + count - 1'b1 : destination;
                operation_remaining <= count;
                operation_backward <= backward;
                operation_copy_pending <= 1'b0;
                after_clear_address <= clear_address;
                after_clear_count <= clear_count;
            end
        end
    endtask

    task automatic scroll_up(input logic [7:0] requested_lines);
        logic [7:0] lines;
        logic [13:0] base;
        logic [13:0] moving_cells;
        begin
            lines = (requested_lines > scroll_bottom - scroll_top + 1) ? scroll_bottom - scroll_top + 1 : requested_lines;
            base = screen_base(alternate_screen);
            moving_cells = line_offset(scroll_bottom - scroll_top + 1 - lines);
            start_copy(
                base + line_offset(scroll_top + lines),
                base + line_offset(scroll_top),
                moving_cells,
                1'b0,
                base + line_offset(scroll_bottom - lines + 1),
                line_offset(lines)
            );
        end
    endtask

    task automatic scroll_down(input logic [7:0] requested_lines);
        logic [7:0] lines;
        logic [13:0] base;
        logic [13:0] moving_cells;
        begin
            lines = (requested_lines > scroll_bottom - scroll_top + 1) ? scroll_bottom - scroll_top + 1 : requested_lines;
            base = screen_base(alternate_screen);
            moving_cells = line_offset(scroll_bottom - scroll_top + 1 - lines);
            start_copy(
                base + line_offset(scroll_top),
                base + line_offset(scroll_top + lines),
                moving_cells,
                1'b1,
                base + line_offset(scroll_top),
                line_offset(lines)
            );
        end
    endtask

    task automatic linefeed;
        begin
            wrap_pending <= 1'b0;
            if (cursor_y == scroll_bottom)
                scroll_up(1);
            else if (cursor_y < ROWS - 1)
                cursor_y <= cursor_y + 1'b1;
        end
    endtask

    task automatic save_cursor;
        begin
            saved_cursor_x <= cursor_x;
            saved_cursor_y <= cursor_y;
            saved_scroll_top <= scroll_top;
            saved_scroll_bottom <= scroll_bottom;
            saved_foreground <= current_foreground;
            saved_background <= current_background;
            saved_inverse <= current_inverse;
            saved_wrap_pending <= wrap_pending;
            saved_origin_mode <= origin_mode;
            saved_cursor_visible <= cursor_visible;
            saved_cursor_style <= cursor_style;
        end
    endtask

    task automatic restore_cursor;
        begin
            cursor_x <= saved_cursor_x;
            cursor_y <= saved_cursor_y;
            scroll_top <= saved_scroll_top;
            scroll_bottom <= saved_scroll_bottom;
            current_foreground <= saved_foreground;
            current_background <= saved_background;
            current_inverse <= saved_inverse;
            wrap_pending <= saved_wrap_pending;
            origin_mode <= saved_origin_mode;
            cursor_visible <= saved_cursor_visible;
            cursor_style <= saved_cursor_style;
        end
    endtask

    task automatic save_normal_screen_state;
        begin
            normal_cursor_x <= cursor_x;
            normal_cursor_y <= cursor_y;
            normal_scroll_top <= scroll_top;
            normal_scroll_bottom <= scroll_bottom;
            normal_foreground <= current_foreground;
            normal_background <= current_background;
            normal_inverse <= current_inverse;
            normal_wrap_enabled <= wrap_enabled;
            normal_wrap_pending <= wrap_pending;
            normal_origin_mode <= origin_mode;
            normal_cursor_visible <= cursor_visible;
            normal_cursor_style <= cursor_style;
        end
    endtask

    task automatic restore_normal_screen_state;
        begin
            cursor_x <= normal_cursor_x;
            cursor_y <= normal_cursor_y;
            scroll_top <= normal_scroll_top;
            scroll_bottom <= normal_scroll_bottom;
            current_foreground <= normal_foreground;
            current_background <= normal_background;
            current_inverse <= normal_inverse;
            wrap_enabled <= normal_wrap_enabled;
            wrap_pending <= normal_wrap_pending;
            origin_mode <= normal_origin_mode;
            cursor_visible <= normal_cursor_visible;
            cursor_style <= normal_cursor_style;
        end
    endtask

    integer parser_index;
    always_ff @(posedge clk) begin
        direct_write_valid <= 1'b0;

        if (reset) begin
            cursor_x <= 0;
            cursor_y <= 0;
            scroll_top <= 0;
            scroll_bottom <= ROWS - 1;
            current_foreground <= 4'h7;
            current_background <= 4'h0;
            current_inverse <= 1'b0;
            cursor_visible <= 1'b1;
            cursor_style <= 0;
            wrap_enabled <= 1'b1;
            wrap_pending <= 1'b0;
            origin_mode <= 1'b0;
            alternate_screen <= 1'b0;
            saved_cursor_x <= 0;
            saved_cursor_y <= 0;
            saved_scroll_top <= 0;
            saved_scroll_bottom <= ROWS - 1;
            saved_foreground <= 4'h7;
            saved_background <= 4'h0;
            saved_inverse <= 1'b0;
            saved_wrap_pending <= 1'b0;
            saved_origin_mode <= 1'b0;
            saved_cursor_visible <= 1'b1;
            saved_cursor_style <= 0;
            normal_cursor_x <= 0;
            normal_cursor_y <= 0;
            normal_scroll_top <= 0;
            normal_scroll_bottom <= ROWS - 1;
            normal_foreground <= 4'h7;
            normal_background <= 4'h0;
            normal_inverse <= 1'b0;
            normal_wrap_enabled <= 1'b1;
            normal_wrap_pending <= 1'b0;
            normal_origin_mode <= 1'b0;
            normal_cursor_visible <= 1'b1;
            normal_cursor_style <= 0;
            parser_state <= TEXT;
            parameter_index <= 0;
            private_mode <= 1'b0;
            csi_space <= 1'b0;
            for (parser_index = 0; parser_index < 8; parser_index = parser_index + 1)
                parameters[parser_index] <= 0;
            operation_state <= OP_CLEAR;
            operation_source <= 0;
            operation_destination <= 0;
            operation_remaining <= CELLS;
            operation_backward <= 1'b0;
            operation_copy_pending <= 1'b0;
            operation_pending_destination <= 0;
            after_clear_address <= 0;
            after_clear_count <= 0;
            deferred_write_valid <= 1'b0;
            deferred_write_address <= 0;
            deferred_write_data <= DEFAULT_CELL;
            direct_write_address <= 0;
            direct_write_data <= DEFAULT_CELL;
        end else begin
            if (operation_state == OP_CLEAR) begin
                if (operation_remaining == 1) begin
                    operation_state <= OP_IDLE;
                    operation_remaining <= 0;
                end else begin
                    operation_destination <= operation_destination + 1'b1;
                    operation_remaining <= operation_remaining - 1'b1;
                end
            end else if (operation_state == OP_COPY) begin
                if (operation_remaining != 0) begin
                    operation_pending_destination <= operation_destination;
                    operation_copy_pending <= 1'b1;
                    operation_remaining <= operation_remaining - 1'b1;
                    if (operation_backward) begin
                        operation_source <= operation_source - 1'b1;
                        operation_destination <= operation_destination - 1'b1;
                    end else begin
                        operation_source <= operation_source + 1'b1;
                        operation_destination <= operation_destination + 1'b1;
                    end
                end else if (operation_copy_pending) begin
                    operation_copy_pending <= 1'b0;
                    if (after_clear_count != 0) begin
                        operation_state <= OP_CLEAR;
                        operation_destination <= after_clear_address;
                        operation_remaining <= after_clear_count;
                        after_clear_count <= 0;
                    end else begin
                        operation_state <= OP_IDLE;
                    end
                end
            end else if (deferred_write_valid) begin
                deferred_write_valid <= 1'b0;
            end else if (rx_valid) begin
                case (parser_state)
                    TEXT: begin
                        if (rx_data == 8'h1b) begin
                            parser_state <= ESCAPE;
                        end else if (rx_data == 8'h0d) begin
                            cursor_x <= 0;
                            wrap_pending <= 1'b0;
                        end else if (rx_data == 8'h0a || rx_data == 8'h0b || rx_data == 8'h0c) begin
                            linefeed();
                        end else if (rx_data == 8'h08 || rx_data == 8'h7f) begin
                            if (cursor_x != 0)
                                cursor_x <= cursor_x - 1'b1;
                            wrap_pending <= 1'b0;
                        end else if (rx_data == 8'h09) begin
                            cursor_x <= ((cursor_x + 8) >= COLUMNS) ? COLUMNS - 1 : ((cursor_x + 8) & 8'hf8);
                            wrap_pending <= 1'b0;
                        end else if (rx_data >= 8'h20 && rx_data <= 8'h7e) begin
                            if (wrap_pending && wrap_enabled) begin
                                cursor_x <= 1;
                                wrap_pending <= 1'b0;
                                if (cursor_y == scroll_bottom) begin
                                    deferred_write_valid <= 1'b1;
                                    deferred_write_address <= screen_base(alternate_screen) + line_offset(cursor_y);
                                    deferred_write_data <= current_inverse ? {current_foreground, current_background, rx_data} : {current_background, current_foreground, rx_data};
                                    scroll_up(1);
                                end else begin
                                    cursor_y <= cursor_y + 1'b1;
                                    direct_write_valid <= 1'b1;
                                    direct_write_address <= screen_base(alternate_screen) + line_offset(cursor_y + 1'b1);
                                    direct_write_data <= current_inverse ? {current_foreground, current_background, rx_data} : {current_background, current_foreground, rx_data};
                                end
                            end else begin
                                direct_write_valid <= 1'b1;
                                direct_write_address <= screen_base(alternate_screen) + line_offset(cursor_y) + cursor_x;
                                direct_write_data <= current_inverse ? {current_foreground, current_background, rx_data} : {current_background, current_foreground, rx_data};
                                if (cursor_x == COLUMNS - 1)
                                    wrap_pending <= wrap_enabled;
                                else
                                    cursor_x <= cursor_x + 1'b1;
                            end
                        end
                    end
                    ESCAPE: begin
                        case (rx_data)
                            "[": begin
                                parser_state <= CSI;
                                parameter_index <= 0;
                                private_mode <= 1'b0;
                                csi_space <= 1'b0;
                                for (parser_index = 0; parser_index < 8; parser_index = parser_index + 1)
                                    parameters[parser_index] <= 0;
                            end
                            "]", "P", "^", "_": parser_state <= STRING_IGNORE;
                            "(", ")", "*", "+": parser_state <= ESCAPE_IGNORE_ONE;
                            "7": begin save_cursor(); parser_state <= TEXT; end
                            "8": begin restore_cursor(); parser_state <= TEXT; end
                            "D": begin linefeed(); parser_state <= TEXT; end
                            "E": begin cursor_x <= 0; linefeed(); parser_state <= TEXT; end
                            "M": begin
                                if (cursor_y == scroll_top)
                                    scroll_down(1);
                                else if (cursor_y != 0)
                                    cursor_y <= cursor_y - 1'b1;
                                wrap_pending <= 1'b0;
                                parser_state <= TEXT;
                            end
                            "c": begin
                                cursor_x <= 0;
                                cursor_y <= 0;
                                scroll_top <= 0;
                                scroll_bottom <= ROWS - 1;
                                current_foreground <= 4'h7;
                                current_background <= 4'h0;
                                current_inverse <= 1'b0;
                                cursor_visible <= 1'b1;
                                cursor_style <= 0;
                                wrap_enabled <= 1'b1;
                                wrap_pending <= 1'b0;
                                origin_mode <= 1'b0;
                                start_clear(screen_base(alternate_screen), CELLS);
                                parser_state <= TEXT;
                            end
                            default: parser_state <= TEXT;
                        endcase
                    end
                    ESCAPE_IGNORE_ONE: parser_state <= TEXT;
                    STRING_IGNORE: begin
                        if (rx_data == 8'h07)
                            parser_state <= TEXT;
                        else if (rx_data == 8'h1b)
                            parser_state <= STRING_ESCAPE;
                    end
                    STRING_ESCAPE: begin
                        if (rx_data == "\\")
                            parser_state <= TEXT;
                        else
                            parser_state <= STRING_IGNORE;
                    end
                    CSI: begin
                        if (rx_data >= "0" && rx_data <= "9") begin
                            parameters[parameter_index] <= parameters[parameter_index] * 10 + rx_data - "0";
                        end else if (rx_data == ";") begin
                            if (parameter_index != 7)
                                parameter_index <= parameter_index + 1'b1;
                        end else if (rx_data == "?") begin
                            private_mode <= 1'b1;
                        end else if (rx_data == 8'h20) begin
                            csi_space <= 1'b1;
                        end else if (rx_data >= 8'h40 && rx_data <= 8'h7e) begin
                            logic [7:0] amount;
                            logic [7:0] available;
                            logic [5:0] requested_top;
                            logic [5:0] requested_bottom;
                            logic [13:0] base;
                            logic [8:0] sgr_state;
                            parser_state <= TEXT;
                            base = screen_base(alternate_screen);
                            amount = count_or_one(parameters[0], 8'hff);
                            case (rx_data)
                                "A": begin
                                    available = cursor_y - (origin_mode ? scroll_top : 0);
                                    cursor_y <= cursor_y - ((amount > available) ? available : amount);
                                    wrap_pending <= 1'b0;
                                end
                                "B", "e": begin
                                    available = (origin_mode ? scroll_bottom : ROWS - 1) - cursor_y;
                                    cursor_y <= cursor_y + ((amount > available) ? available : amount);
                                    wrap_pending <= 1'b0;
                                end
                                "C", "a": begin
                                    available = COLUMNS - 1 - cursor_x;
                                    cursor_x <= cursor_x + ((amount > available) ? available : amount);
                                    wrap_pending <= 1'b0;
                                end
                                "D": begin
                                    available = cursor_x;
                                    cursor_x <= cursor_x - ((amount > available) ? available : amount);
                                    wrap_pending <= 1'b0;
                                end
                                "E": begin
                                    available = (origin_mode ? scroll_bottom : ROWS - 1) - cursor_y;
                                    cursor_y <= cursor_y + ((amount > available) ? available : amount);
                                    cursor_x <= 0;
                                    wrap_pending <= 1'b0;
                                end
                                "F": begin
                                    available = cursor_y - (origin_mode ? scroll_top : 0);
                                    cursor_y <= cursor_y - ((amount > available) ? available : amount);
                                    cursor_x <= 0;
                                    wrap_pending <= 1'b0;
                                end
                                "G", "`": begin
                                    cursor_x <= (amount > COLUMNS) ? COLUMNS - 1 : amount - 1'b1;
                                    wrap_pending <= 1'b0;
                                end
                                "d": begin
                                    requested_top = origin_mode ? scroll_top : 0;
                                    requested_bottom = origin_mode ? scroll_bottom : ROWS - 1;
                                    cursor_y <= (amount > requested_bottom - requested_top + 1) ? requested_bottom : requested_top + amount - 1'b1;
                                    wrap_pending <= 1'b0;
                                end
                                "H", "f": begin
                                    requested_top = origin_mode ? scroll_top : 0;
                                    requested_bottom = origin_mode ? scroll_bottom : ROWS - 1;
                                    amount = count_or_one(parameters[0], ROWS);
                                    cursor_y <= (amount > requested_bottom - requested_top + 1) ? requested_bottom : requested_top + amount - 1'b1;
                                    amount = count_or_one(parameters[1], COLUMNS);
                                    cursor_x <= (amount > COLUMNS) ? COLUMNS - 1 : amount - 1'b1;
                                    wrap_pending <= 1'b0;
                                end
                                "J": begin
                                    case (parameters[0])
                                        0: start_clear(base + line_offset(cursor_y) + cursor_x, CELLS - (line_offset(cursor_y) + cursor_x));
                                        1: start_clear(base, line_offset(cursor_y) + cursor_x + 1'b1);
                                        2, 3: start_clear(base, CELLS);
                                        default: begin end
                                    endcase
                                    wrap_pending <= 1'b0;
                                end
                                "K": begin
                                    case (parameters[0])
                                        0: start_clear(base + line_offset(cursor_y) + cursor_x, COLUMNS - cursor_x);
                                        1: start_clear(base + line_offset(cursor_y), cursor_x + 1'b1);
                                        2: start_clear(base + line_offset(cursor_y), COLUMNS);
                                        default: begin end
                                    endcase
                                    wrap_pending <= 1'b0;
                                end
                                "@": begin
                                    amount = count_or_one(parameters[0], COLUMNS - cursor_x);
                                    start_copy(
                                        base + line_offset(cursor_y) + cursor_x,
                                        base + line_offset(cursor_y) + cursor_x + amount,
                                        COLUMNS - cursor_x - amount,
                                        1'b1,
                                        base + line_offset(cursor_y) + cursor_x,
                                        amount
                                    );
                                    wrap_pending <= 1'b0;
                                end
                                "P": begin
                                    amount = count_or_one(parameters[0], COLUMNS - cursor_x);
                                    start_copy(
                                        base + line_offset(cursor_y) + cursor_x + amount,
                                        base + line_offset(cursor_y) + cursor_x,
                                        COLUMNS - cursor_x - amount,
                                        1'b0,
                                        base + line_offset(cursor_y) + COLUMNS - amount,
                                        amount
                                    );
                                    wrap_pending <= 1'b0;
                                end
                                "X": begin
                                    amount = count_or_one(parameters[0], COLUMNS - cursor_x);
                                    start_clear(base + line_offset(cursor_y) + cursor_x, amount);
                                    wrap_pending <= 1'b0;
                                end
                                "L": if (cursor_y >= scroll_top && cursor_y <= scroll_bottom) begin
                                    amount = count_or_one(parameters[0], scroll_bottom - cursor_y + 1);
                                    start_copy(
                                        base + line_offset(cursor_y),
                                        base + line_offset(cursor_y + amount),
                                        line_offset(scroll_bottom - cursor_y + 1 - amount),
                                        1'b1,
                                        base + line_offset(cursor_y),
                                        line_offset(amount)
                                    );
                                    wrap_pending <= 1'b0;
                                end
                                "M": if (cursor_y >= scroll_top && cursor_y <= scroll_bottom) begin
                                    amount = count_or_one(parameters[0], scroll_bottom - cursor_y + 1);
                                    start_copy(
                                        base + line_offset(cursor_y + amount),
                                        base + line_offset(cursor_y),
                                        line_offset(scroll_bottom - cursor_y + 1 - amount),
                                        1'b0,
                                        base + line_offset(scroll_bottom - amount + 1),
                                        line_offset(amount)
                                    );
                                    wrap_pending <= 1'b0;
                                end
                                "S": scroll_up(count_or_one(parameters[0], scroll_bottom - scroll_top + 1));
                                "T": scroll_down(count_or_one(parameters[0], scroll_bottom - scroll_top + 1));
                                "m": begin
                                    sgr_state = {current_inverse, current_background, current_foreground};
                                    for (parser_index = 0; parser_index < 8; parser_index = parser_index + 1)
                                        if (parser_index <= parameter_index)
                                            sgr_state = next_sgr(sgr_state, parameters[parser_index]);
                                    {current_inverse, current_background, current_foreground} <= sgr_state;
                                end
                                "r": begin
                                    requested_top = (parameters[0] == 0) ? 0 : parameters[0] - 1'b1;
                                    requested_bottom = (parameters[1] == 0) ? ROWS - 1 : parameters[1] - 1'b1;
                                    if (requested_top < requested_bottom && requested_bottom < ROWS) begin
                                        scroll_top <= requested_top;
                                        scroll_bottom <= requested_bottom;
                                        cursor_x <= 0;
                                        cursor_y <= origin_mode ? requested_top : 0;
                                        wrap_pending <= 1'b0;
                                    end
                                end
                                "s": save_cursor();
                                "u": restore_cursor();
                                "q": if (csi_space) begin
                                    case (parameters[0])
                                        0, 1: cursor_style <= 0;
                                        2: cursor_style <= 1;
                                        3: cursor_style <= 2;
                                        4: cursor_style <= 3;
                                        5: cursor_style <= 4;
                                        6: cursor_style <= 5;
                                        default: begin end
                                    endcase
                                end
                                "h": if (private_mode) begin
                                    for (parser_index = 0; parser_index < 8; parser_index = parser_index + 1) begin
                                        if (parser_index <= parameter_index) begin
                                            case (parameters[parser_index])
                                                6: begin origin_mode <= 1'b1; cursor_x <= 0; cursor_y <= scroll_top; end
                                                7: wrap_enabled <= 1'b1;
                                                25: cursor_visible <= 1'b1;
                                                47, 1047, 1049: if (!alternate_screen) begin
                                                    save_normal_screen_state();
                                                    alternate_screen <= 1'b1;
                                                    cursor_x <= 0;
                                                    cursor_y <= 0;
                                                    scroll_top <= 0;
                                                    scroll_bottom <= ROWS - 1;
                                                    wrap_pending <= 1'b0;
                                                    start_clear(CELLS, CELLS);
                                                end
                                                default: begin end
                                            endcase
                                        end
                                    end
                                end
                                "l": if (private_mode) begin
                                    for (parser_index = 0; parser_index < 8; parser_index = parser_index + 1) begin
                                        if (parser_index <= parameter_index) begin
                                            case (parameters[parser_index])
                                                6: begin origin_mode <= 1'b0; cursor_x <= 0; cursor_y <= 0; end
                                                7: begin wrap_enabled <= 1'b0; wrap_pending <= 1'b0; end
                                                25: cursor_visible <= 1'b0;
                                                47, 1047, 1049: if (alternate_screen) begin
                                                    alternate_screen <= 1'b0;
                                                    restore_normal_screen_state();
                                                end
                                                default: begin end
                                            endcase
                                        end
                                    end
                                end
                                default: begin end
                            endcase
                        end
                    end
                    default: parser_state <= TEXT;
                endcase
            end
        end
    end

    logic [7:0] character;
    logic [3:0] foreground_index;
    logic [3:0] background_index;
    logic [10:0] font_address;
    logic [7:0] font_bits;
    logic [2:0] glyph_x;
    logic [3:0] glyph_y;
    logic glyph_pixel;
    logic cursor_cell;
    logic cursor_blink_on;
    logic cursor_draw;
    logic cursor_block;
    logic [23:0] blink_counter;
    logic [23:0] foreground_color;
    logic [23:0] background_color;
    logic [23:0] pixel_color;

    always_ff @(posedge clk) begin
        if (reset)
            blink_counter <= 0;
        else
            blink_counter <= blink_counter + 1'b1;
    end

    always_comb begin
        character = render_cell[7:0];
        foreground_index = render_cell[11:8];
        background_index = render_cell[15:12];
        glyph_x = render_x[2:0];
        glyph_y = render_y[3:0];
        font_address = 0;
        font_bits = 0;
        if (character >= FONT_FIRST && character < FONT_FIRST + FONT_GLYPHS) begin
            font_address = (character - FONT_FIRST) * 16 + glyph_y;
            font_bits = font_rom[font_address];
        end
        glyph_pixel = font_bits[7 - glyph_x];
        foreground_color = ansi_color(foreground_index);
        background_color = ansi_color(background_index);
        pixel_color = glyph_pixel ? foreground_color : background_color;
        cursor_cell = render_active && (render_x >> 3) == cursor_x && (render_y >> 4) == cursor_y;
        cursor_blink_on = cursor_style[0] || blink_counter[23];
        cursor_block = cursor_style <= 1;
        case (cursor_style)
            0, 1: cursor_draw = 1'b1;
            2, 3: cursor_draw = glyph_y >= 14;
            4, 5: cursor_draw = glyph_x < 2;
            default: cursor_draw = 1'b1;
        endcase
        if (cursor_visible && cursor_cell && cursor_blink_on && cursor_draw) begin
            if (cursor_block)
                pixel_color = glyph_pixel ? background_color : foreground_color;
            else
                pixel_color = foreground_color;
        end
        if (render_active) begin
            red = pixel_color[23:16];
            green = pixel_color[15:8];
            blue = pixel_color[7:0];
        end else begin
            red = 0;
            green = 0;
            blue = 0;
        end
    end

endmodule

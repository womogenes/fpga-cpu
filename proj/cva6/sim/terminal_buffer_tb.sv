`timescale 1ns/1ps

module terminal_buffer_tb;
    localparam integer COLUMNS = 160;
    localparam integer ROWS = 45;
    localparam integer CELLS = COLUMNS * ROWS;

    logic clk = 1'b0;
    logic reset = 1'b1;
    logic [7:0] rx_data = 0;
    logic rx_valid = 1'b0;
    logic rx_ready;
    logic [10:0] pixel_x = 0;
    logic [9:0] pixel_y = 0;
    logic [7:0] red;
    logic [7:0] green;
    logic [7:0] blue;

    always #5 clk = ~clk;

    terminal_buffer #(
        .FONT_FILE("rtl/font8x16.mem")
    ) dut (
        .clk,
        .reset,
        .rx_data,
        .rx_valid,
        .rx_ready,
        .pixel_x,
        .pixel_y,
        .red,
        .green,
        .blue
    );

    task automatic wait_ready;
        begin
            while (!rx_ready)
                @(posedge clk);
        end
    endtask

    task automatic send_byte(input logic [7:0] value);
        begin
            wait_ready();
            @(negedge clk);
            rx_data = value;
            rx_valid = 1'b1;
            @(negedge clk);
            rx_valid = 1'b0;
            @(negedge clk);
        end
    endtask

    task automatic send_text(input string value);
        integer index;
        begin
            for (index = 0; index < value.len(); index = index + 1)
                send_byte(value[index]);
        end
    endtask

    task automatic send_csi(input string value);
        begin
            send_byte(8'h1b);
            send_byte("[");
            send_text(value);
        end
    endtask

    task automatic reset_screen;
        begin
            send_byte(8'h1b);
            send_byte("c");
            wait_ready();
        end
    endtask

    task automatic end_frame;
        begin
            @(negedge clk);
            pixel_x = 1649;
            pixel_y = 749;
            @(negedge clk);
            pixel_x = 0;
            pixel_y = 0;
        end
    endtask

    task automatic expect_char(
        input integer bank,
        input integer row,
        input integer column,
        input logic [7:0] expected
    );
        logic [7:0] actual;
        begin
            actual = dut.text_ram[bank * CELLS + row * COLUMNS + column][7:0];
            if (actual !== expected)
                $fatal(1, "cell bank=%0d row=%0d column=%0d: expected %02x, got %02x", bank, row, column, expected, actual);
        end
    endtask

    task automatic expect_attributes(
        input integer bank,
        input integer row,
        input integer column,
        input logic [3:0] foreground,
        input logic [3:0] background
    );
        logic [7:0] actual;
        begin
            actual = dut.text_ram[bank * CELLS + row * COLUMNS + column][15:8];
            if (actual !== {background, foreground})
                $fatal(1, "attributes bank=%0d row=%0d column=%0d: expected %02x, got %02x", bank, row, column, {background, foreground}, actual);
        end
    endtask

    initial begin
        repeat (4) @(posedge clk);
        reset = 1'b0;
        wait_ready();

        if (dut.cursor_x !== 0 || dut.cursor_y !== 0)
            $fatal(1, "reset cursor is not at 0,0");

        send_text("ABC");
        expect_char(0, 0, 0, "A");
        expect_char(0, 0, 1, "B");
        expect_char(0, 0, 2, "C");
        if (dut.cursor_x !== 3 || dut.cursor_y !== 0)
            $fatal(1, "print cursor did not advance");

        reset_screen();
        send_csi("31;44m");
        send_byte("X");
        expect_char(0, 0, 0, "X");
        expect_attributes(0, 0, 0, 4'h1, 4'h4);

        reset_screen();
        send_text("ABCDE");
        send_csi("1;3H");
        send_csi("2@");
        wait_ready();
        expect_char(0, 0, 0, "A");
        expect_char(0, 0, 1, "B");
        expect_char(0, 0, 2, " ");
        expect_char(0, 0, 3, " ");
        expect_char(0, 0, 4, "C");
        expect_char(0, 0, 6, "E");
        send_csi("1;2H");
        send_csi("P");
        wait_ready();
        expect_char(0, 0, 0, "A");
        expect_char(0, 0, 1, " ");
        expect_char(0, 0, 3, "C");

        reset_screen();
        send_csi("1;1H");
        send_byte("A");
        send_csi("2;1H");
        send_byte("B");
        send_csi("3;1H");
        send_byte("C");
        send_csi("1;3r");
        send_csi("2;1H");
        send_csi("L");
        wait_ready();
        expect_char(0, 0, 0, "A");
        expect_char(0, 1, 0, " ");
        expect_char(0, 2, 0, "B");
        send_csi("M");
        wait_ready();
        expect_char(0, 0, 0, "A");
        expect_char(0, 1, 0, "B");
        expect_char(0, 2, 0, " ");

        reset_screen();
        send_csi("1;1H");
        send_byte("A");
        send_csi("2;1H");
        send_byte("B");
        send_csi("3;1H");
        send_byte("C");
        send_csi("1;3r");
        send_csi("3;1H");
        send_byte(8'h0a);
        wait_ready();
        expect_char(0, 0, 0, "B");
        expect_char(0, 1, 0, "C");
        expect_char(0, 2, 0, " ");

        reset_screen();
        send_csi("1;160H");
        send_byte("X");
        send_byte("Y");
        expect_char(0, 0, 159, "X");
        expect_char(0, 1, 0, "Y");
        if (dut.cursor_x !== 1 || dut.cursor_y !== 1)
            $fatal(1, "160-column autowrap failed");

        reset_screen();
        send_text("SHELL");
        send_csi("?1049h");
        wait_ready();
        if (!dut.alternate_screen)
            $fatal(1, "alternate screen did not enable");
        send_text("VIM");
        expect_char(1, 0, 0, "V");
        send_csi("s");
        send_csi("20;80H");
        send_csi("?1049l");
        if (dut.alternate_screen)
            $fatal(1, "alternate screen did not disable");
        expect_char(0, 0, 0, "S");
        expect_char(0, 0, 4, "L");
        if (dut.cursor_x !== 5 || dut.cursor_y !== 0)
            $fatal(1, "normal cursor state was not restored");

        reset_screen();
        send_csi("3 q");
        if (dut.cursor_style !== 2)
            $fatal(1, "cursor style was not set");
        send_csi("?25l");
        if (dut.cursor_visible)
            $fatal(1, "cursor visibility did not disable");
        send_csi("?25h");
        if (!dut.cursor_visible)
            $fatal(1, "cursor visibility did not enable");

        if (dut.cursor_blink_half_period_frames !== 30)
            $fatal(1, "cursor blink did not reset to a 1 Hz period");
        send_csi("?2z");
        if (dut.cursor_blink_half_period_frames !== 2 || !dut.cursor_blink_phase)
            $fatal(1, "cursor blink period was not configured");
        end_frame();
        if (!dut.cursor_blink_phase)
            $fatal(1, "cursor blink phase changed too early");
        end_frame();
        if (dut.cursor_blink_phase)
            $fatal(1, "cursor blink phase did not change");
        end_frame();
        end_frame();
        if (!dut.cursor_blink_phase)
            $fatal(1, "cursor blink phase did not return");

        reset_screen();
        send_byte(8'h1b);
        send_byte("]");
        send_text("ignored title");
        send_byte(8'h07);
        send_byte("Z");
        expect_char(0, 0, 0, "Z");

        send_csi("2J");
        wait_ready();
        expect_char(0, 0, 0, " ");
        expect_char(0, ROWS - 1, COLUMNS - 1, " ");

        $display("PASS: 160x45 hardware terminal regression");
        $finish;
    end
endmodule

`timescale 1ns/1ps

module software_video_plane_tb;
    localparam integer SOURCE_WIDTH = 4;
    localparam integer SOURCE_HEIGHT = 2;
    localparam integer FRAME_PIXELS = SOURCE_WIDTH * SOURCE_HEIGHT;

    logic clk = 0;
    logic reset = 1;
    logic [31:0] command_bus = 0;
    logic force_terminal = 0;
    logic [10:0] pixel_x = 0;
    logic [9:0] pixel_y = 0;
    logic ready;
    logic active;
    logic [7:0] red;
    logic [7:0] green;
    logic [7:0] blue;

    always #5 clk = ~clk;

    software_video_plane #(
        .SOURCE_WIDTH(SOURCE_WIDTH),
        .SOURCE_HEIGHT(SOURCE_HEIGHT),
        .SCALE_SHIFT(3)
    ) dut (
        .clk,
        .reset,
        .command_bus,
        .force_terminal,
        .pixel_x,
        .pixel_y,
        .ready,
        .active,
        .red,
        .green,
        .blue
    );

    task automatic send_packet(input logic [27:0] pixels);
        command_bus <= {~command_bus[31], 3'b000, pixels};
        repeat (10) @(posedge clk);
    endtask

    task automatic send_command(input logic [6:0] command);
        command_bus <= {~command_bus[31], 3'b001, 21'b0, command};
        repeat (5) @(posedge clk);
    endtask

    initial begin
        repeat (8) @(posedge clk);
        reset <= 0;
        repeat (4) @(posedge clk);

        send_command(7'h01);
        if (ready)
            $fatal(1, "video plane stayed ready while loading");
        send_packet({7'd3, 7'd2, 7'd1, 7'd0});
        send_packet({7'd7, 7'd6, 7'd5, 7'd4});
        for (int index = 0; index < FRAME_PIXELS; index++) begin
            if (dut.frame_buffer[FRAME_PIXELS + index] !== index[6:0])
                $fatal(1, "packed pixel %0d was unpacked incorrectly", index);
        end

        if (active)
            $fatal(1, "graphics became active before vertical blank");
        if (ready)
            $fatal(1, "video plane became ready before frame swap");

        pixel_x <= 1649;
        pixel_y <= 749;
        @(posedge clk);
        pixel_x <= 0;
        pixel_y <= 0;
        repeat (2) @(posedge clk);

        if (!active)
            $fatal(1, "graphics did not activate after the frame swap");
        if (!ready)
            $fatal(1, "video plane did not become ready after frame swap");

        force_terminal <= 1;
        #1;
        if (active)
            $fatal(1, "force-terminal input did not override graphics");
        force_terminal <= 0;

        send_command(7'h00);
        repeat (3) @(posedge clk);
        if (active)
            $fatal(1, "software terminal command did not disable graphics");

        $display("software_video_plane_tb passed");
        $finish;
    end
endmodule

module uart_hdmi_terminal #(
    parameter integer UART_BAUD = 3000000
) (
    input  logic      clk_200,
    input  logic      reset,
    input  logic      uart_tx,
    input  logic [31:0] software_video_bus,
    input  logic      force_terminal,
    output logic      software_video_ready,
    output logic      software_video_command_ack,
    output wire       hdmi_clk_p,
    output wire       hdmi_clk_n,
    output wire [2:0] hdmi_data_p,
    output wire [2:0] hdmi_data_n
);

    wire mmcm_feedback;
    wire pixel_clk_unbuffered;
    wire serial_clk_unbuffered;
    wire pixel_clk;
    wire serial_clk;
    wire mmcm_locked;
    logic [3:0] reset_shift = 4'hf;
    wire video_reset = reset_shift[3];

    MMCME2_ADV #(
        .BANDWIDTH("OPTIMIZED"),
        .CLKFBOUT_MULT_F(37.125),
        .CLKIN1_PERIOD(5.0),
        .CLKOUT0_DIVIDE_F(10.0),
        .CLKOUT1_DIVIDE(2),
        .DIVCLK_DIVIDE(10),
        .REF_JITTER1(0.01)
    ) i_video_mmcm (
        .CLKFBIN(mmcm_feedback),
        .CLKIN1(clk_200),
        .CLKIN2(1'b0),
        .CLKINSEL(1'b1),
        .DADDR(7'b0),
        .DCLK(1'b0),
        .DEN(1'b0),
        .DI(16'b0),
        .DWE(1'b0),
        .PSCLK(1'b0),
        .PSEN(1'b0),
        .PSINCDEC(1'b0),
        .PWRDWN(1'b0),
        .RST(reset),
        .CLKFBOUT(mmcm_feedback),
        .CLKOUT0(pixel_clk_unbuffered),
        .CLKOUT1(serial_clk_unbuffered),
        .LOCKED(mmcm_locked)
    );

    BUFG i_pixel_bufg (.I(pixel_clk_unbuffered), .O(pixel_clk));
    BUFG i_serial_bufg (.I(serial_clk_unbuffered), .O(serial_clk));

    always_ff @(posedge pixel_clk or negedge mmcm_locked) begin
        if (!mmcm_locked)
            reset_shift <= 4'hf;
        else
            reset_shift <= {reset_shift[2:0], 1'b0};
    end

    logic [10:0] horizontal;
    logic [9:0] vertical;
    wire active = (horizontal < 1280) && (vertical < 720);
    wire hsync = (horizontal >= 1390) && (horizontal < 1430);
    wire vsync = (vertical >= 725) && (vertical < 730);

    always_ff @(posedge pixel_clk) begin
        if (video_reset) begin
            horizontal <= 0;
            vertical <= 0;
        end else if (horizontal == 1649) begin
            horizontal <= 0;
            vertical <= (vertical == 749) ? 0 : vertical + 1'b1;
        end else begin
            horizontal <= horizontal + 1'b1;
        end
    end

    logic [7:0] decoded_data;
    logic decoded_valid;
    uart_rx #(.BAUD(UART_BAUD)) i_uart_rx (
        .clk(pixel_clk),
        .reset(video_reset),
        .serial(uart_tx),
        .data(decoded_data),
        .valid(decoded_valid)
    );

    logic [7:0] rx_data;
    logic rx_valid;
    logic rx_ready;
    logic fifo_overflow;
    byte_fifo #(.DEPTH(64)) i_uart_fifo (
        .clk(pixel_clk),
        .reset(video_reset),
        .push_data(decoded_data),
        .push(decoded_valid),
        .pop(rx_valid && rx_ready),
        .data(rx_data),
        .valid(rx_valid),
        .overflow(fifo_overflow)
    );

    logic [7:0] terminal_red;
    logic [7:0] terminal_green;
    logic [7:0] terminal_blue;
    terminal_buffer i_buffer (
        .clk(pixel_clk),
        .reset(video_reset),
        .rx_data,
        .rx_valid,
        .rx_ready,
        .pixel_x(horizontal),
        .pixel_y(vertical),
        .red(terminal_red),
        .green(terminal_green),
        .blue(terminal_blue)
    );

    logic graphics_active;
    logic [7:0] graphics_red;
    logic [7:0] graphics_green;
    logic [7:0] graphics_blue;
    software_video_plane i_software_video (
        .clk(pixel_clk),
        .reset(video_reset),
        .command_bus(software_video_bus),
        .force_terminal,
        .pixel_x(horizontal),
        .pixel_y(vertical),
        .ready(software_video_ready),
        .command_ack(software_video_command_ack),
        .active(graphics_active),
        .red(graphics_red),
        .green(graphics_green),
        .blue(graphics_blue)
    );

    wire [7:0] red = graphics_active ? graphics_red : terminal_red;
    wire [7:0] green = graphics_active ? graphics_green : terminal_green;
    wire [7:0] blue = graphics_active ? graphics_blue : terminal_blue;

    logic [9:0] tmds_blue;
    logic [9:0] tmds_green;
    logic [9:0] tmds_red;
    tmds_encoder i_blue_encoder (
        .clk(pixel_clk),
        .reset(video_reset),
        .data(blue),
        .control({vsync, hsync}),
        .data_enable(active),
        .encoded(tmds_blue)
    );
    tmds_encoder i_green_encoder (
        .clk(pixel_clk),
        .reset(video_reset),
        .data(green),
        .control(2'b00),
        .data_enable(active),
        .encoded(tmds_green)
    );
    tmds_encoder i_red_encoder (
        .clk(pixel_clk),
        .reset(video_reset),
        .data(red),
        .control(2'b00),
        .data_enable(active),
        .encoded(tmds_red)
    );

    wire [2:0] serialized;
    tmds_serializer i_blue_serializer (.pixel_clk, .serial_clk, .reset(video_reset), .data(tmds_blue), .serial(serialized[0]));
    tmds_serializer i_green_serializer (.pixel_clk, .serial_clk, .reset(video_reset), .data(tmds_green), .serial(serialized[1]));
    tmds_serializer i_red_serializer (.pixel_clk, .serial_clk, .reset(video_reset), .data(tmds_red), .serial(serialized[2]));

    wire forwarded_pixel_clock;
    ODDR #(.DDR_CLK_EDGE("SAME_EDGE")) i_clock_forward (
        .C(pixel_clk),
        .CE(1'b1),
        .D1(1'b1),
        .D2(1'b0),
        .R(1'b0),
        .S(1'b0),
        .Q(forwarded_pixel_clock)
    );

    OBUFDS i_clock_output (.I(forwarded_pixel_clock), .O(hdmi_clk_p), .OB(hdmi_clk_n));
    OBUFDS i_blue_output (.I(serialized[0]), .O(hdmi_data_p[0]), .OB(hdmi_data_n[0]));
    OBUFDS i_green_output (.I(serialized[1]), .O(hdmi_data_p[1]), .OB(hdmi_data_n[1]));
    OBUFDS i_red_output (.I(serialized[2]), .O(hdmi_data_p[2]), .OB(hdmi_data_n[2]));

endmodule

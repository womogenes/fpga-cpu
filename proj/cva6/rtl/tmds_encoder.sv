module tmds_encoder (
    input  logic       clk,
    input  logic       reset,
    input  logic [7:0] data,
    input  logic [1:0] control,
    input  logic       data_enable,
    output logic [9:0] encoded
);

    logic [3:0] ones_data;
    logic use_xnor;
    logic [8:0] q_m;
    logic [3:0] ones_q_m;
    logic [3:0] zeros_q_m;
    logic signed [5:0] disparity;
    integer bit_index;

    always_comb begin
        ones_data = data[0] + data[1] + data[2] + data[3] + data[4] + data[5] + data[6] + data[7];
        use_xnor = (ones_data > 4) || ((ones_data == 4) && !data[0]);
        q_m[0] = data[0];
        for (bit_index = 1; bit_index < 8; bit_index = bit_index + 1)
            q_m[bit_index] = use_xnor ? ~(q_m[bit_index-1] ^ data[bit_index]) : (q_m[bit_index-1] ^ data[bit_index]);
        q_m[8] = !use_xnor;
        ones_q_m = q_m[0] + q_m[1] + q_m[2] + q_m[3] + q_m[4] + q_m[5] + q_m[6] + q_m[7];
        zeros_q_m = 8 - ones_q_m;
    end

    always_ff @(posedge clk) begin
        if (reset) begin
            encoded <= 10'b1101010100;
            disparity <= 0;
        end else if (!data_enable) begin
            disparity <= 0;
            case (control)
                2'b00: encoded <= 10'b1101010100;
                2'b01: encoded <= 10'b0010101011;
                2'b10: encoded <= 10'b0101010100;
                default: encoded <= 10'b1010101011;
            endcase
        end else if ((disparity == 0) || (ones_q_m == zeros_q_m)) begin
            encoded[9] <= ~q_m[8];
            encoded[8] <= q_m[8];
            if (q_m[8]) begin
                encoded[7:0] <= q_m[7:0];
                disparity <= disparity + $signed({1'b0, ones_q_m}) - $signed({1'b0, zeros_q_m});
            end else begin
                encoded[7:0] <= ~q_m[7:0];
                disparity <= disparity + $signed({1'b0, zeros_q_m}) - $signed({1'b0, ones_q_m});
            end
        end else if ((!disparity[5] && (ones_q_m > zeros_q_m)) || (disparity[5] && (zeros_q_m > ones_q_m))) begin
            encoded[9] <= 1'b1;
            encoded[8] <= q_m[8];
            encoded[7:0] <= ~q_m[7:0];
            disparity <= disparity + $signed({4'b0, q_m[8], 1'b0}) + $signed({2'b0, zeros_q_m}) - $signed({2'b0, ones_q_m});
        end else begin
            encoded[9] <= 1'b0;
            encoded[8] <= q_m[8];
            encoded[7:0] <= q_m[7:0];
            disparity <= disparity - $signed({4'b0, !q_m[8], 1'b0}) + $signed({2'b0, ones_q_m}) - $signed({2'b0, zeros_q_m});
        end
    end

endmodule

module bonsai_fp32_alu (
    input  logic        fused_i,
    input  logic        multiply_i,
    input  logic [31:0] a_i,
    input  logic [31:0] b_i,
    input  logic [31:0] c_i,
    output logic [31:0] result_o
);

    function automatic logic [27:0] shift_right_jam28(
        input logic [27:0] value,
        input integer amount);
        logic sticky;
        integer bit_index;
        begin
            if (amount <= 0) begin
                shift_right_jam28 = value;
            end else if (amount >= 28) begin
                shift_right_jam28 = {27'b0, |value};
            end else begin
                sticky = 1'b0;
                for (bit_index = 0; bit_index < 28; bit_index = bit_index + 1)
                    if (bit_index < amount) sticky = sticky | value[bit_index];
                shift_right_jam28 = value >> amount;
                shift_right_jam28[0] = shift_right_jam28[0] | sticky;
            end
        end
    endfunction

    function automatic logic [56:0] shift_right_jam57(
        input logic [56:0] value,
        input integer amount);
        logic sticky;
        integer bit_index;
        begin
            if (amount <= 0) begin
                shift_right_jam57 = value;
            end else if (amount >= 57) begin
                shift_right_jam57 = {56'b0, |value};
            end else begin
                sticky = 1'b0;
                for (bit_index = 0; bit_index < 57; bit_index = bit_index + 1)
                    if (bit_index < amount) sticky = sticky | value[bit_index];
                shift_right_jam57 = value >> amount;
                shift_right_jam57[0] = shift_right_jam57[0] | sticky;
            end
        end
    endfunction

    function automatic logic [31:0] pack_rounded(
        input logic sign,
        input integer exponent,
        input logic [27:0] extended);
        logic [24:0] rounded;
        logic [27:0] shifted;
        logic increment;
        logic [7:0] packed_exponent;
        integer subnormal_shift;
        integer final_exponent;
        begin
            shifted = extended;
            final_exponent = exponent;
            if (final_exponent < -126) begin
                subnormal_shift = -126 - final_exponent;
                shifted = shift_right_jam28(shifted, subnormal_shift);
                final_exponent = -126;
            end
            increment = shifted[2] && (shifted[1] || shifted[0] || shifted[3]);
            rounded = {1'b0, shifted[26:3]} + increment;
            if (rounded[24]) begin
                rounded = rounded >> 1;
                final_exponent = final_exponent + 1;
            end
            if (final_exponent > 127) begin
                pack_rounded = {sign, 8'hff, 23'b0};
            end else if (rounded[23:0] == 0) begin
                pack_rounded = {sign, 31'b0};
            end else if (final_exponent == -126 && !rounded[23]) begin
                pack_rounded = {sign, 8'b0, rounded[22:0]};
            end else begin
                packed_exponent = final_exponent + 127;
                pack_rounded = {sign, packed_exponent, rounded[22:0]};
            end
        end
    endfunction

    function automatic logic [31:0] fused_multiply_add_fp32(
        input logic [31:0] a,
        input logic [31:0] b,
        input logic [31:0] c);
        logic product_sign;
        logic c_sign;
        logic [7:0] a_exp_bits;
        logic [7:0] b_exp_bits;
        logic [7:0] c_exp_bits;
        logic [22:0] a_frac;
        logic [22:0] b_frac;
        logic [22:0] c_frac;
        logic [23:0] a_mantissa;
        logic [23:0] b_mantissa;
        logic [23:0] c_mantissa;
        logic [47:0] product;
        logic [47:0] normalized_product;
        logic [56:0] product_extended;
        logic [56:0] c_extended;
        logic [57:0] sum;
        logic [27:0] rounded_input;
        logic result_sign;
        integer a_exponent;
        integer b_exponent;
        integer c_exponent;
        integer product_exponent;
        integer exponent;
        integer normalize;
        begin
            product_sign = a[31] ^ b[31];
            c_sign = c[31];
            a_exp_bits = a[30:23];
            b_exp_bits = b[30:23];
            c_exp_bits = c[30:23];
            a_frac = a[22:0];
            b_frac = b[22:0];
            c_frac = c[22:0];
            if ((a_exp_bits == 8'hff && a_frac != 0) ||
                (b_exp_bits == 8'hff && b_frac != 0) ||
                (c_exp_bits == 8'hff && c_frac != 0) ||
                ((a_exp_bits == 8'hff || b_exp_bits == 8'hff) &&
                 (((a_exp_bits == 0 && a_frac == 0) ||
                   (b_exp_bits == 0 && b_frac == 0)) ||
                  (c_exp_bits == 8'hff && product_sign != c_sign)))) begin
                fused_multiply_add_fp32 = 32'h7fc00000;
            end else if (a_exp_bits == 8'hff || b_exp_bits == 8'hff) begin
                fused_multiply_add_fp32 = {product_sign, 8'hff, 23'b0};
            end else if (c_exp_bits == 8'hff) begin
                fused_multiply_add_fp32 = c;
            end else if ((a_exp_bits == 0 && a_frac == 0) ||
                         (b_exp_bits == 0 && b_frac == 0)) begin
                fused_multiply_add_fp32 = add_fp32({product_sign, 31'b0}, c);
            end else if (c_exp_bits == 0 && c_frac == 0) begin
                fused_multiply_add_fp32 = multiply_fp32(a, b);
            end else begin
                a_exponent = a_exp_bits == 0 ? -126 : a_exp_bits - 127;
                b_exponent = b_exp_bits == 0 ? -126 : b_exp_bits - 127;
                c_exponent = c_exp_bits == 0 ? -126 : c_exp_bits - 127;
                a_mantissa = a_exp_bits == 0 ? {1'b0, a_frac} : {1'b1, a_frac};
                b_mantissa = b_exp_bits == 0 ? {1'b0, b_frac} : {1'b1, b_frac};
                c_mantissa = c_exp_bits == 0 ? {1'b0, c_frac} : {1'b1, c_frac};
                for (normalize = 0; normalize < 23; normalize = normalize + 1) begin
                    if (!a_mantissa[23]) begin
                        a_mantissa = a_mantissa << 1;
                        a_exponent = a_exponent - 1;
                    end
                    if (!b_mantissa[23]) begin
                        b_mantissa = b_mantissa << 1;
                        b_exponent = b_exponent - 1;
                    end
                    if (!c_mantissa[23]) begin
                        c_mantissa = c_mantissa << 1;
                        c_exponent = c_exponent - 1;
                    end
                end
                product = a_mantissa * b_mantissa;
                product_exponent = a_exponent + b_exponent;
                if (product[47]) begin
                    normalized_product = product;
                    product_exponent = product_exponent + 1;
                end else begin
                    normalized_product = product << 1;
                end
                product_extended = {1'b0, normalized_product, 8'b0};
                c_extended = {1'b0, c_mantissa, 32'b0};
                if (product_exponent > c_exponent) begin
                    c_extended = shift_right_jam57(
                        c_extended, product_exponent - c_exponent);
                    exponent = product_exponent;
                end else if (c_exponent > product_exponent) begin
                    product_extended = shift_right_jam57(
                        product_extended, c_exponent - product_exponent);
                    exponent = c_exponent;
                end else begin
                    exponent = product_exponent;
                end
                if (product_sign == c_sign) begin
                    sum = {1'b0, product_extended} + {1'b0, c_extended};
                    result_sign = product_sign;
                end else if (product_extended >= c_extended) begin
                    sum = {1'b0, product_extended} - {1'b0, c_extended};
                    result_sign = product_sign;
                end else begin
                    sum = {1'b0, c_extended} - {1'b0, product_extended};
                    result_sign = c_sign;
                end
                if (sum == 0) begin
                    fused_multiply_add_fp32 = 32'b0;
                end else begin
                    if (sum[56]) begin
                        sum[56:0] = shift_right_jam57(sum[56:0], 1);
                        exponent = exponent + 1;
                    end else begin
                        for (normalize = 0; normalize < 56;
                             normalize = normalize + 1) begin
                            if (!sum[55] && exponent > -126) begin
                                sum = sum << 1;
                                exponent = exponent - 1;
                            end
                        end
                    end
                    rounded_input = {1'b0, sum[55:32], sum[31], sum[30],
                                     |sum[29:0]};
                    fused_multiply_add_fp32 =
                        pack_rounded(result_sign, exponent, rounded_input);
                end
            end
        end
    endfunction

    function automatic logic [31:0] multiply_fp32(
        input logic [31:0] a,
        input logic [31:0] b);
        logic sign;
        logic [7:0] a_exp;
        logic [7:0] b_exp;
        logic [22:0] a_frac;
        logic [22:0] b_frac;
        logic [23:0] a_mantissa;
        logic [23:0] b_mantissa;
        logic [47:0] product;
        logic [27:0] extended;
        integer exponent;
        integer normalize;
        begin
            sign = a[31] ^ b[31];
            a_exp = a[30:23];
            b_exp = b[30:23];
            a_frac = a[22:0];
            b_frac = b[22:0];
            if ((a_exp == 8'hff && a_frac != 0) ||
                (b_exp == 8'hff && b_frac != 0)) begin
                multiply_fp32 = 32'h7fc00000;
            end else if ((a_exp == 8'hff && (b_exp == 0 && b_frac == 0)) ||
                         (b_exp == 8'hff && (a_exp == 0 && a_frac == 0))) begin
                multiply_fp32 = 32'h7fc00000;
            end else if (a_exp == 8'hff || b_exp == 8'hff) begin
                multiply_fp32 = {sign, 8'hff, 23'b0};
            end else if ((a_exp == 0 && a_frac == 0) ||
                         (b_exp == 0 && b_frac == 0)) begin
                multiply_fp32 = {sign, 31'b0};
            end else begin
                a_mantissa = a_exp == 0 ? {1'b0, a_frac} : {1'b1, a_frac};
                b_mantissa = b_exp == 0 ? {1'b0, b_frac} : {1'b1, b_frac};
                exponent = (a_exp == 0 ? -126 : a_exp - 127) +
                           (b_exp == 0 ? -126 : b_exp - 127);
                for (normalize = 0; normalize < 23; normalize = normalize + 1) begin
                    if (!a_mantissa[23]) begin
                        a_mantissa = a_mantissa << 1;
                        exponent = exponent - 1;
                    end
                    if (!b_mantissa[23]) begin
                        b_mantissa = b_mantissa << 1;
                        exponent = exponent - 1;
                    end
                end
                product = a_mantissa * b_mantissa;
                if (product[47]) begin
                    extended = {1'b0, product[47:24], product[23], product[22],
                                |product[21:0]};
                    exponent = exponent + 1;
                end else begin
                    extended = {1'b0, product[46:23], product[22], product[21],
                                |product[20:0]};
                end
                multiply_fp32 = pack_rounded(sign, exponent, extended);
            end
        end
    endfunction

    function automatic logic [31:0] add_fp32(
        input logic [31:0] a,
        input logic [31:0] b);
        logic a_sign;
        logic b_sign;
        logic [7:0] a_exp_bits;
        logic [7:0] b_exp_bits;
        logic [22:0] a_frac;
        logic [22:0] b_frac;
        logic [23:0] a_mantissa;
        logic [23:0] b_mantissa;
        logic [27:0] a_extended;
        logic [27:0] b_extended;
        logic [28:0] sum;
        logic sign;
        integer a_exponent;
        integer b_exponent;
        integer exponent;
        integer normalize;
        begin
            a_sign = a[31];
            b_sign = b[31];
            a_exp_bits = a[30:23];
            b_exp_bits = b[30:23];
            a_frac = a[22:0];
            b_frac = b[22:0];
            if ((a_exp_bits == 8'hff && a_frac != 0) ||
                (b_exp_bits == 8'hff && b_frac != 0) ||
                (a_exp_bits == 8'hff && b_exp_bits == 8'hff &&
                 a_sign != b_sign)) begin
                add_fp32 = 32'h7fc00000;
            end else if (a_exp_bits == 8'hff) begin
                add_fp32 = a;
            end else if (b_exp_bits == 8'hff) begin
                add_fp32 = b;
            end else if (a_exp_bits == 0 && a_frac == 0 &&
                         b_exp_bits == 0 && b_frac == 0) begin
                add_fp32 = {a_sign & b_sign, 31'b0};
            end else if (a_exp_bits == 0 && a_frac == 0) begin
                add_fp32 = b;
            end else if (b_exp_bits == 0 && b_frac == 0) begin
                add_fp32 = a;
            end else begin
                a_exponent = a_exp_bits == 0 ? -126 : a_exp_bits - 127;
                b_exponent = b_exp_bits == 0 ? -126 : b_exp_bits - 127;
                a_mantissa = a_exp_bits == 0 ? {1'b0, a_frac} : {1'b1, a_frac};
                b_mantissa = b_exp_bits == 0 ? {1'b0, b_frac} : {1'b1, b_frac};
                a_extended = {1'b0, a_mantissa, 3'b0};
                b_extended = {1'b0, b_mantissa, 3'b0};
                if (a_exponent > b_exponent) begin
                    b_extended = shift_right_jam28(
                        b_extended, a_exponent - b_exponent);
                    exponent = a_exponent;
                end else if (b_exponent > a_exponent) begin
                    a_extended = shift_right_jam28(
                        a_extended, b_exponent - a_exponent);
                    exponent = b_exponent;
                end else begin
                    exponent = a_exponent;
                end
                if (a_sign == b_sign) begin
                    sum = {1'b0, a_extended} + {1'b0, b_extended};
                    sign = a_sign;
                end else if (a_extended >= b_extended) begin
                    sum = {1'b0, a_extended} - {1'b0, b_extended};
                    sign = a_sign;
                end else begin
                    sum = {1'b0, b_extended} - {1'b0, a_extended};
                    sign = b_sign;
                end
                if (sum == 0) begin
                    add_fp32 = 32'b0;
                end else begin
                    if (sum[27]) begin
                        sum[27:0] = shift_right_jam28(sum[27:0], 1);
                        exponent = exponent + 1;
                    end else begin
                        for (normalize = 0; normalize < 24;
                             normalize = normalize + 1) begin
                            if (!sum[26] && exponent > -126) begin
                                sum = sum << 1;
                                exponent = exponent - 1;
                            end
                        end
                    end
                    add_fp32 = pack_rounded(sign, exponent, sum[27:0]);
                end
            end
        end
    endfunction

    always_comb begin
        if (fused_i)
            result_o = fused_multiply_add_fp32(a_i, b_i, c_i);
        else
            result_o = multiply_i ? multiply_fp32(a_i, b_i) :
                                    add_fp32(a_i, b_i);
    end

endmodule

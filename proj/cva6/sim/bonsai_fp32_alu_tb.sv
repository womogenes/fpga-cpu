`timescale 1ns/1ps

module bonsai_fp32_alu_tb;
    logic multiply;
    logic fused;
    logic [31:0] a;
    logic [31:0] b;
    logic [31:0] c;
    logic [31:0] result;
    logic [31:0] expected_add;
    logic [31:0] expected_multiply;
    logic [31:0] expected_fused;
    integer vectors;
    integer failures;
    integer descriptor;
    integer scanned;
    reg [1023:0] vector_path;

    bonsai_fp32_alu dut (
        .fused_i(fused),
        .multiply_i(multiply),
        .a_i(a),
        .b_i(b),
        .c_i(c),
        .result_o(result)
    );

    initial begin
        if (!$value$plusargs("vectors=%s", vector_path)) begin
            $display("missing +vectors=PATH");
            $fatal(1);
        end
        descriptor = $fopen(vector_path, "r");
        if (descriptor == 0) begin
            $display("cannot open %s", vector_path);
            $fatal(1);
        end
        vectors = 0;
        failures = 0;
        fused = 1'b0;
        multiply = 1'b0;
        while (!$feof(descriptor)) begin
            scanned = $fscanf(descriptor, "%h %h %h %h %h %h\n", a, b, c,
                              expected_add, expected_multiply,
                              expected_fused);
            if (scanned == 6) begin
                fused = 1'b0;
                multiply = 1'b0;
                #1;
                if (result !== expected_add) begin
                    if (failures < 20)
                        $display("add mismatch a=%h b=%h got=%h want=%h",
                                 a, b, result, expected_add);
                    failures = failures + 1;
                end
                multiply = 1'b1;
                #1;
                if (result !== expected_multiply) begin
                    if (failures < 20)
                        $display("mul mismatch a=%h b=%h got=%h want=%h",
                                 a, b, result, expected_multiply);
                    failures = failures + 1;
                end
                fused = 1'b1;
                #1;
                if (result !== expected_fused) begin
                    if (failures < 20)
                        $display("fma mismatch a=%h b=%h c=%h got=%h want=%h",
                                 a, b, c, result, expected_fused);
                    failures = failures + 1;
                end
                vectors = vectors + 1;
            end
        end
        $fclose(descriptor);
        $display("fp32 vectors=%0d failures=%0d", vectors, failures);
        if (failures != 0) $fatal(1);
        $finish;
    end
endmodule

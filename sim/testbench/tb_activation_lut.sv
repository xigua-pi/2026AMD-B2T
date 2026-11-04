`timescale 1ns/1ps

module tb_activation_lut;
  logic signed [15:0] preactivation;
  logic [8:0] lut_index;
  logic [15:0] sigmoid_q16;
  logic signed [15:0] tanh_q15;

  logic [15:0] expected_index [0:65535];
  logic [15:0] expected_sigmoid [0:65535];
  logic signed [15:0] expected_tanh [0:65535];
  integer raw;
  integer errors = 0;

  h144_activation_lut dut (
      .preactivation(preactivation),
      .lut_index(lut_index),
      .sigmoid_q16(sigmoid_q16),
      .tanh_q15(tanh_q15)
  );

  initial begin
    $readmemh("unit_vectors/lut_expected_index_u16.mem", expected_index);
    $readmemh("unit_vectors/lut_expected_sigmoid_u16.mem", expected_sigmoid);
    $readmemh("unit_vectors/lut_expected_tanh_i16.mem", expected_tanh);
    for (raw = 0; raw < 65536; raw = raw + 1) begin
      preactivation = raw[15:0];
      #1;
      if (lut_index !== expected_index[raw][8:0] ||
          sigmoid_q16 !== expected_sigmoid[raw] ||
          tanh_q15 !== expected_tanh[raw]) begin
        if (errors < 20)
          $display("Mismatch raw=%0d signed=%0d index=%0d/%0d sig=%0d/%0d tanh=%0d/%0d",
                   raw, preactivation, lut_index, expected_index[raw],
                   sigmoid_q16, expected_sigmoid[raw], tanh_q15, expected_tanh[raw]);
        errors = errors + 1;
      end
    end
    if (errors != 0)
      $fatal(1, "Activation LUT failed: %0d/65536 mismatches", errors);
    $display("PASS: LUT index, sigmoid and tanh matched all 65536 INT16 inputs");
    $finish;
  end
endmodule

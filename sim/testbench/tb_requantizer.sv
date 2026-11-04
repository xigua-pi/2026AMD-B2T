`timescale 1ns/1ps

module tb_requantizer;
  localparam int CASES = 10035;

  logic signed [31:0] accumulator;
  logic signed [15:0] weight_exponent = 16'sd0;
  logic signed [7:0] input_exponent = 8'sd0;
  logic signed [7:0] target_exponent;
  logic output_is_int16;
  logic signed [15:0] result;

  logic signed [31:0] accumulators [0:CASES-1];
  logic signed [7:0] shifts [0:CASES-1];
  logic [7:0] width_flags [0:CASES-1];
  logic signed [15:0] expected [0:CASES-1];
  integer index;
  integer errors = 0;

  h144_requantizer dut (
      .accumulator(accumulator),
      .weight_exponent(weight_exponent),
      .input_exponent(input_exponent),
      .target_exponent(target_exponent),
      .output_is_int16(output_is_int16),
      .result(result)
  );

  initial begin
    $readmemh("unit_vectors/requant_accumulator_i32.mem", accumulators);
    $readmemh("unit_vectors/requant_shift_i8.mem", shifts);
    $readmemh("unit_vectors/requant_output_is_i16.mem", width_flags);
    $readmemh("unit_vectors/requant_expected_i16.mem", expected);
    for (index = 0; index < CASES; index = index + 1) begin
      accumulator = accumulators[index];
      target_exponent = shifts[index];
      output_is_int16 = width_flags[index][0];
      #1;
      if (result !== expected[index]) begin
        if (errors < 20)
          $display("Mismatch case=%0d acc=%0d shift=%0d i16=%0d expected=%0d actual=%0d",
                   index, accumulator, target_exponent, output_is_int16,
                   expected[index], result);
        errors = errors + 1;
      end
    end
    if (errors != 0)
      $fatal(1, "Requantizer failed: %0d/%0d mismatches", errors, CASES);
    $display("PASS: requantizer matched %0d directed/random golden cases", CASES);
    $finish;
  end
endmodule

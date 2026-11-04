`timescale 1ns/1ps

module h144_requantizer (
    input  logic signed [31:0] accumulator,
    input  logic signed [15:0] weight_exponent,
    input  logic signed [7:0] input_exponent,
    input  logic signed [7:0] target_exponent,
    input  logic output_is_int16,
    output logic signed [15:0] result
);
  import h144_fixed_math_pkg::*;

  integer right_shift;
  logic signed [63:0] extended;
  logic signed [63:0] rounded;
  logic signed [7:0] saturated_i8;

  always_comb begin
    right_shift = $signed(target_exponent) -
                  ($signed(weight_exponent) + $signed(input_exponent));
    extended = {{32{accumulator[31]}}, accumulator};
    rounded = round_shift_even64(extended, right_shift);
    saturated_i8 = saturate_i8_sym(rounded);
    if (output_is_int16)
      result = saturate_i16_sym(rounded);
    else
      result = {{8{saturated_i8[7]}}, saturated_i8};
  end
endmodule

`timescale 1ns/1ps

package h144_fixed_math_pkg;
  function automatic signed [63:0] round_shift_even64(
      input signed [63:0] value,
      input integer right_shift
  );
    logic negative;
    logic [63:0] magnitude;
    logic [63:0] quotient;
    logic [63:0] remainder;
    logic [63:0] mask;
    logic [63:0] half;
    logic increment;
    logic [63:0] rounded;
    begin
      if (right_shift <= 0) begin
        round_shift_even64 = value <<< (-right_shift);
      end else if (right_shift >= 63) begin
        round_shift_even64 = 64'sd0;
      end else begin
        negative = value < 0;
        magnitude = negative ? $unsigned(-value) : $unsigned(value);
        quotient = magnitude >> right_shift;
        mask = (64'h1 << right_shift) - 1;
        remainder = magnitude & mask;
        half = 64'h1 << (right_shift - 1);
        increment = (remainder > half) ||
                    ((remainder == half) && quotient[0]);
        rounded = quotient + increment;
        round_shift_even64 = negative ? -$signed(rounded) : $signed(rounded);
      end
    end
  endfunction

  function automatic signed [31:0] saturate_i32(input signed [63:0] value);
    begin
      if (value > 64'sd2147483647)
        saturate_i32 = 32'sh7fffffff;
      else if (value < -64'sd2147483648)
        saturate_i32 = 32'sh80000000;
      else
        saturate_i32 = value[31:0];
    end
  endfunction

  function automatic signed [15:0] saturate_i16_sym(input signed [63:0] value);
    begin
      if (value > 64'sd32767)
        saturate_i16_sym = 16'sd32767;
      else if (value < -64'sd32767)
        saturate_i16_sym = -16'sd32767;
      else
        saturate_i16_sym = value[15:0];
    end
  endfunction

  function automatic signed [7:0] saturate_i8_sym(input signed [63:0] value);
    begin
      if (value > 64'sd127)
        saturate_i8_sym = 8'sd127;
      else if (value < -64'sd127)
        saturate_i8_sym = -8'sd127;
      else
        saturate_i8_sym = value[7:0];
    end
  endfunction

  function automatic [8:0] lut_index_512(input signed [15:0] preactivation);
    integer signed value_int;
    integer signed clipped_int;
    integer unsigned numerator_int;
    integer unsigned quotient_int;
    integer unsigned remainder_int;
    integer unsigned increment_int;
    begin
      value_int = $signed(preactivation);
      if (value_int < -16384)
        clipped_int = -16384;
      else if (value_int > 16384)
        clipped_int = 16384;
      else
        clipped_int = value_int;

      // denominator=32768=2^15; numerator=(clipped-minimum)*511.
      numerator_int = (clipped_int + 16384) * 511;
      quotient_int = numerator_int >> 15;
      remainder_int = numerator_int & 32767;
      increment_int = (remainder_int > 16384) ||
                      ((remainder_int == 16384) && (quotient_int & 1));
      if (quotient_int + increment_int > 511)
        lut_index_512 = 9'd511;
      else
        lut_index_512 = quotient_int[8:0] + increment_int[0];
    end
  endfunction
endpackage

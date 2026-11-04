`timescale 1ns/1ps

module h144_activation_lut #(
    parameter SIGMOID_FILE = "accelerator/sigmoid_lut_u16.mem",
    parameter TANH_FILE = "accelerator/tanh_lut_i16.mem",
    parameter bit PIPELINED_INDEX = 1'b0
) (
    input  logic clk,
    input  logic rst_n,
    input  logic signed [15:0] preactivation,
    output logic [8:0] lut_index,
    output logic [15:0] sigmoid_q16,
    output logic signed [15:0] tanh_q15
);
  import h144_fixed_math_pkg::*;

  (* rom_style = "distributed" *) logic [15:0] sigmoid_rom [0:511];
  (* rom_style = "distributed" *) logic signed [15:0] tanh_rom [0:511];
  logic [8:0] computed_index;
  logic [16:0] clipped_offset;
  logic [24:0] numerator;
  logic [24:0] numerator_q;
  logic [9:0] quotient;
  logic [14:0] remainder;
  logic increment;
  logic [9:0] rounded_index;

  initial begin
    $readmemh(SIGMOID_FILE, sigmoid_rom);
    $readmemh(TANH_FILE, tanh_rom);
  end

  always_comb begin
    computed_index = lut_index_512(preactivation);
    if ($signed(preactivation) < -16'sd16384)
      clipped_offset = 17'd0;
    else if ($signed(preactivation) > 16'sd16384)
      clipped_offset = 17'd32768;
    else
      clipped_offset = $signed(preactivation) + 17'sd16384;
    numerator = clipped_offset * 9'd511;
    quotient = numerator_q >> 15;
    remainder = numerator_q[14:0];
    increment = (remainder > 15'd16384) ||
                ((remainder == 15'd16384) && quotient[0]);
    rounded_index = quotient + increment;
    sigmoid_q16 = sigmoid_rom[lut_index];
    tanh_q15 = tanh_rom[lut_index];
  end

  generate
    if (PIPELINED_INDEX) begin : g_pipelined_index
      always_ff @(posedge clk) begin
        if (!rst_n) begin
          numerator_q <= 25'd0;
          lut_index <= 9'd0;
        end else begin
          // 第一级：裁剪并计算 (x-min)*511；第二级：除以 2^15 后 ties-to-even。
          numerator_q <= numerator;
          if (rounded_index > 10'd511)
            lut_index <= 9'd511;
          else
            lut_index <= rounded_index[8:0];
        end
      end
    end else begin : g_combinational_index
      always_comb lut_index = computed_index;
    end
  endgenerate
endmodule

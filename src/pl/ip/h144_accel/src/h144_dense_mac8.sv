`timescale 1ns/1ps

module h144_dense_mac8 #(
    parameter WEIGHT_BANK0_FILE = "accelerator/dense_weights_bank0_u64.mem",
    parameter WEIGHT_BANK1_FILE = "accelerator/dense_weights_bank1_u64.mem",
    parameter WEIGHT_BANK2_FILE = "accelerator/dense_weights_bank2_u64.mem",
    parameter EXPONENT_FILE = "accelerator/dense_exponents_i16.mem",
    parameter BIAS_FILE = "accelerator/dense_bias_i32.mem"
) (
    input  logic clk,
    input  logic rst_n,

    input  logic x_write_enable,
    input  logic [7:0] x_write_address,
    input  logic signed [15:0] x_write_data,

    input  logic start,
    input  logic [15:0] weight_word_base,
    input  logic [12:0] parameter_base,
    input  logic [7:0] chunks_per_output,
    input  logic [8:0] output_count,
    input  logic signed [7:0] input_exponent,
    input  logic signed [7:0] target_exponent,
    input  logic output_is_int16,

    output logic busy,
    output logic output_valid,
    output logic [8:0] output_index,
    output logic signed [15:0] output_data,
    output logic done,
    output logic accumulator_overflow
);
  import h144_layout_pkg::*;
  import h144_fixed_math_pkg::*;

  typedef enum logic [3:0] {
    STATE_IDLE,
    STATE_LOAD_PARAMETER,
    STATE_ISSUE_FIRST,
    STATE_ISSUE,
    STATE_MULTIPLY,
    STATE_ACCUMULATE,
    STATE_REQUANTIZE,
    STATE_ROUND_SHIFT,
    STATE_OUTPUT
  } state_t;

  // 分 bank 避免 34,368 words 的单一 ROM 被综合为 65,536 深度。
  (* rom_style = "block" *) logic [63:0] weight_rom0 [0:DENSE_WEIGHT_BANK0_WORDS-1];
  (* rom_style = "block" *) logic [63:0] weight_rom1 [0:DENSE_WEIGHT_BANK1_WORDS-1];
  (* rom_style = "block" *) logic [63:0] weight_rom2 [0:DENSE_WEIGHT_BANK2_WORDS-1];
  (* rom_style = "block" *) logic signed [15:0] exponent_rom [0:DENSE_PARAMETER_ROWS-1];
  (* rom_style = "block" *) logic signed [31:0] bias_rom [0:DENSE_PARAMETER_ROWS-1];
  logic signed [15:0] x_memory [0:191];

  state_t state;
  logic [15:0] cfg_weight_base;
  logic [12:0] cfg_parameter_base;
  logic [7:0] cfg_chunks;
  logic [8:0] cfg_outputs;
  logic signed [7:0] cfg_input_exponent;
  logic signed [7:0] cfg_target_exponent;
  logic cfg_output_is_int16;

  logic [8:0] current_output;
  logic [7:0] current_chunk;
  logic [63:0] weight_word;
  logic [63:0] weight_bank0_q;
  logic [63:0] weight_bank1_q;
  logic [63:0] weight_bank2_q;
  logic [1:0] weight_bank_select_q;
  logic signed [15:0] weight_exponent_q;
  logic signed [31:0] bias_q;
  logic signed [63:0] accumulator;

  logic signed [7:0] lane_weight [0:7];
  logic signed [15:0] lane_input [0:7];
  logic signed [23:0] lane_product [0:7];
  logic signed [31:0] pair_sum [0:3];
  logic signed [31:0] quad_sum [0:1];
  logic signed [31:0] product_sum;
  logic signed [31:0] product_sum_q;
  logic signed [63:0] row_sum;
  logic signed [31:0] row_sum_i32;
  logic signed [31:0] final_sum_i32_q;
  logic signed [15:0] requantized;
  logic [16:0] requested_weight_address;

  integer lane;
  integer right_shift;
  logic signed [63:0] rounded_q;
  logic [32:0] round_magnitude_q;
  logic [32:0] round_quotient_next;
  logic [32:0] round_magnitude_rounded;
  logic signed [32:0] round_signed_result;
  logic [4:0] round_shift_count_q;
  logic round_sticky_q;
  logic round_negative_q;
  logic round_increment;
  logic signed [7:0] saturated_i8;

  initial begin
    $readmemh(WEIGHT_BANK0_FILE, weight_rom0);
    $readmemh(WEIGHT_BANK1_FILE, weight_rom1);
    $readmemh(WEIGHT_BANK2_FILE, weight_rom2);
    $readmemh(EXPONENT_FILE, exponent_rom);
    $readmemh(BIAS_FILE, bias_rom);
  end

  always_comb begin
    requested_weight_address = cfg_weight_base + current_output * cfg_chunks;
    if (state == STATE_ISSUE)
      requested_weight_address = requested_weight_address + current_chunk;
  end

  // 三个 ROM 各自保持单端口同步读模板，下一周期只对寄存后的输出做选择。
  // 这种写法可被 Vivado 稳定推断为三个独立 Block RAM bank。
  always_comb begin
    case (weight_bank_select_q)
      2'd0: weight_word = weight_bank0_q;
      2'd1: weight_word = weight_bank1_q;
      default: weight_word = weight_bank2_q;
    endcase
  end

  always_comb begin
    for (lane = 0; lane < 8; lane = lane + 1) begin
      lane_weight[lane] = weight_word[lane*8 +: 8];
      lane_input[lane] = x_memory[current_chunk*8 + lane];
      lane_product[lane] = lane_weight[lane] * lane_input[lane];
    end
    // 平衡加法树，避免 8 个乘积形成串行加法链。
    pair_sum[0] = {{8{lane_product[0][23]}}, lane_product[0]} +
                  {{8{lane_product[1][23]}}, lane_product[1]};
    pair_sum[1] = {{8{lane_product[2][23]}}, lane_product[2]} +
                  {{8{lane_product[3][23]}}, lane_product[3]};
    pair_sum[2] = {{8{lane_product[4][23]}}, lane_product[4]} +
                  {{8{lane_product[5][23]}}, lane_product[5]};
    pair_sum[3] = {{8{lane_product[6][23]}}, lane_product[6]} +
                  {{8{lane_product[7][23]}}, lane_product[7]};
    quad_sum[0] = pair_sum[0] + pair_sum[1];
    quad_sum[1] = pair_sum[2] + pair_sum[3];
    product_sum = quad_sum[0] + quad_sum[1];

    row_sum = accumulator + {{32{product_sum_q[31]}}, product_sum_q};
    row_sum_i32 = saturate_i32(row_sum);
    right_shift = $signed(cfg_target_exponent) -
                  ($signed(weight_exponent_q) + $signed(cfg_input_exponent));
    saturated_i8 = saturate_i8_sym(rounded_q);
    if (cfg_output_is_int16)
      requantized = saturate_i16_sym(rounded_q);
    else
      requantized = {{8{saturated_i8[7]}}, saturated_i8};
  end

  always_comb begin
    round_quotient_next = round_magnitude_q >> 1;
    // 当前被移出的位在最后一次迭代时是 guard；此前移出的位汇总为 sticky。
    round_increment = round_magnitude_q[0] &&
                      (round_sticky_q || round_quotient_next[0]);
    round_magnitude_rounded = round_quotient_next + round_increment;
    if (round_negative_q)
      round_signed_result = -$signed(round_magnitude_rounded);
    else
      round_signed_result = $signed(round_magnitude_rounded);
  end

  always_ff @(posedge clk) begin
    if (x_write_enable)
      x_memory[x_write_address] <= x_write_data;

    if (!rst_n) begin
      state <= STATE_IDLE;
      busy <= 1'b0;
      output_valid <= 1'b0;
      done <= 1'b0;
      accumulator_overflow <= 1'b0;
      current_output <= 9'd0;
      current_chunk <= 8'd0;
      accumulator <= 64'sd0;
      product_sum_q <= 32'sd0;
      final_sum_i32_q <= 32'sd0;
      rounded_q <= 64'sd0;
      round_magnitude_q <= 33'd0;
      round_shift_count_q <= 5'd0;
      round_sticky_q <= 1'b0;
      round_negative_q <= 1'b0;
    end else begin
      output_valid <= 1'b0;
      done <= 1'b0;

      case (state)
        STATE_IDLE: begin
          busy <= 1'b0;
          if (start) begin
            cfg_weight_base <= weight_word_base;
            cfg_parameter_base <= parameter_base;
            cfg_chunks <= chunks_per_output;
            cfg_outputs <= output_count;
            cfg_input_exponent <= input_exponent;
            cfg_target_exponent <= target_exponent;
            cfg_output_is_int16 <= output_is_int16;
            current_output <= 9'd0;
            accumulator_overflow <= 1'b0;
            busy <= 1'b1;
            state <= STATE_LOAD_PARAMETER;
          end
        end

        STATE_LOAD_PARAMETER: begin
          bias_q <= bias_rom[cfg_parameter_base + current_output];
          weight_exponent_q <= exponent_rom[cfg_parameter_base + current_output];
          state <= STATE_ISSUE_FIRST;
        end

        STATE_ISSUE_FIRST: begin
          accumulator <= {{32{bias_q[31]}}, bias_q};
          current_chunk <= 8'd0;
          weight_bank0_q <= weight_rom0[requested_weight_address[13:0]];
          weight_bank1_q <= weight_rom1[requested_weight_address[13:0]];
          weight_bank2_q <= weight_rom2[requested_weight_address[10:0]];
          if (requested_weight_address < DENSE_WEIGHT_BANK1_BASE)
            weight_bank_select_q <= 2'd0;
          else if (requested_weight_address < DENSE_WEIGHT_BANK2_BASE)
            weight_bank_select_q <= 2'd1;
          else
            weight_bank_select_q <= 2'd2;
          state <= STATE_MULTIPLY;
        end

        STATE_ISSUE: begin
          weight_bank0_q <= weight_rom0[requested_weight_address[13:0]];
          weight_bank1_q <= weight_rom1[requested_weight_address[13:0]];
          weight_bank2_q <= weight_rom2[requested_weight_address[10:0]];
          if (requested_weight_address < DENSE_WEIGHT_BANK1_BASE)
            weight_bank_select_q <= 2'd0;
          else if (requested_weight_address < DENSE_WEIGHT_BANK2_BASE)
            weight_bank_select_q <= 2'd1;
          else
            weight_bank_select_q <= 2'd2;
          state <= STATE_MULTIPLY;
        end

        STATE_MULTIPLY: begin
          product_sum_q <= product_sum;
          state <= STATE_ACCUMULATE;
        end

        STATE_ACCUMULATE: begin
          if (current_chunk + 1 >= cfg_chunks) begin
            final_sum_i32_q <= row_sum_i32;
            if ((row_sum > 64'sd2147483647) || (row_sum < -64'sd2147483648))
              accumulator_overflow <= 1'b1;
            state <= STATE_REQUANTIZE;
          end else begin
            accumulator <= row_sum;
            current_chunk <= current_chunk + 1'b1;
            state <= STATE_ISSUE;
          end
        end

        STATE_REQUANTIZE: begin
          round_negative_q <= final_sum_i32_q[31];
          if (final_sum_i32_q[31])
            round_magnitude_q <=
                $unsigned(-$signed({final_sum_i32_q[31], final_sum_i32_q}));
          else
            round_magnitude_q <= {1'b0, final_sum_i32_q};
          round_sticky_q <= 1'b0;
          if (right_shift <= 0) begin
            // 当前冻结模型的所有 dense requant shift 均在 [2,14]；保留 0 位移兜底。
            rounded_q <= {{32{final_sum_i32_q[31]}}, final_sum_i32_q};
            state <= STATE_OUTPUT;
          end else begin
            round_shift_count_q <= right_shift[4:0];
            state <= STATE_ROUND_SHIFT;
          end
        end

        STATE_ROUND_SHIFT: begin
          if (round_shift_count_q == 1) begin
            rounded_q <= {{31{round_signed_result[32]}}, round_signed_result};
            round_shift_count_q <= 5'd0;
            state <= STATE_OUTPUT;
          end else begin
            round_sticky_q <= round_sticky_q | round_magnitude_q[0];
            round_magnitude_q <= round_quotient_next;
            round_shift_count_q <= round_shift_count_q - 1'b1;
          end
        end

        STATE_OUTPUT: begin
          output_index <= current_output;
          output_data <= requantized;
          output_valid <= 1'b1;
          if (current_output + 1 >= cfg_outputs) begin
            busy <= 1'b0;
            done <= 1'b1;
            state <= STATE_IDLE;
          end else begin
            current_output <= current_output + 1'b1;
            state <= STATE_LOAD_PARAMETER;
          end
        end

        default: state <= STATE_IDLE;
      endcase
    end
  end
endmodule

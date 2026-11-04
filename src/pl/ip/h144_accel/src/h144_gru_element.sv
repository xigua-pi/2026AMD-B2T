`timescale 1ns/1ps

module h144_gru_element (
    input  logic clk,
    input  logic rst_n,
    input  logic start,
    input logic low_precision,
    input  logic signed [15:0] input_reset,
    input  logic signed [15:0] input_update,
    input  logic signed [15:0] input_new,
    input  logic signed [15:0] hidden_reset,
    input  logic signed [15:0] hidden_update,
    input  logic signed [15:0] hidden_new,
    input  logic signed [15:0] previous_hidden,
    output logic busy,
    output logic done,
    output logic [15:0] reset_q16,
    output logic [15:0] update_q16,
    output logic signed [15:0] candidate_q15,
    output logic signed [15:0] next_hidden_q14
);
  import h144_fixed_math_pkg::*;

  typedef enum logic [3:0] {
    STATE_IDLE,
    STATE_RESET_INDEX,
    STATE_RESET_INDEX_WAIT,
    STATE_RESET_STORE,
    STATE_UPDATE_INDEX,
    STATE_UPDATE_INDEX_WAIT,
    STATE_UPDATE_STORE,
    STATE_RESET_PRODUCT,
    STATE_RESET_ROUND,
    STATE_CANDIDATE_INDEX,
    STATE_CANDIDATE_INDEX_WAIT,
    STATE_CANDIDATE_STORE,
    STATE_OUTPUT_PRODUCTS,
    STATE_OUTPUT_ROUND,
    STATE_OUTPUT
  } state_t;

  state_t state;
  logic low_precision_q;
  logic [15:0] reduced_reset, reduced_update;
  logic signed [63:0] rounded_reset, rounded_update;
  logic signed [15:0] in_reset_q;
  logic signed [15:0] in_update_q;
  logic signed [15:0] in_new_q;
  logic signed [15:0] hid_reset_q;
  logic signed [15:0] hid_update_q;
  logic signed [15:0] hid_new_q;
  logic signed [15:0] previous_hidden_q;

  logic signed [15:0] activation_input;
  logic [8:0] activation_index;
  logic [15:0] sigmoid_value;
  logic signed [15:0] tanh_value;
  logic signed [63:0] reset_product;
  logic signed [63:0] reset_product_q;
  logic signed [15:0] reset_hidden_q11;
  logic signed [15:0] reset_hidden_q11_q;
  logic signed [63:0] candidate_product;
  logic signed [63:0] previous_product;
  logic signed [63:0] candidate_product_q;
  logic signed [63:0] previous_product_q;
  logic signed [15:0] candidate_term_q14;
  logic signed [15:0] previous_term_q14;
  logic signed [15:0] candidate_term_q14_q;
  logic signed [15:0] previous_term_q14_q;
  logic signed [63:0] gate_sum;
  logic signed [63:0] hidden_sum;

  h144_activation_lut #(.PIPELINED_INDEX(1'b1)) activation (
      .clk(clk), .rst_n(rst_n),
      .preactivation(activation_input),
      .lut_index(activation_index),
      .sigmoid_q16(sigmoid_value),
      .tanh_q15(tanh_value)
  );

  always_comb begin
    // round-even再饱和：UQ0.8 / UQ0.12，扩回原Q16总线，低位确定为零。
    rounded_reset = round_shift_even64($signed({1'b0, sigmoid_value}), 8);
    rounded_update = round_shift_even64($signed({1'b0, sigmoid_value}), 4);
    reduced_reset = (rounded_reset > 255) ? 16'hff00 : {rounded_reset[7:0],8'b0};
    reduced_update = (rounded_update > 4095) ? 16'hfff0 : {rounded_update[11:0],4'b0};
    reset_product = $signed({1'b0, reset_q16}) * $signed(hid_new_q);
    reset_hidden_q11 = saturate_i16_sym(round_shift_even64(reset_product_q, 16));
    // 1.0 in Q0.16 is 65536, which requires 17 unsigned bits.
    candidate_product = $signed({1'b0, 17'd65536 - {1'b0, update_q16}}) *
                        $signed(candidate_q15);
    previous_product = $signed({1'b0, update_q16}) * $signed(previous_hidden_q);
    candidate_term_q14 = saturate_i16_sym(round_shift_even64(candidate_product_q, 17));
    previous_term_q14 = saturate_i16_sym(round_shift_even64(previous_product_q, 16));

    case (state)
      STATE_RESET_INDEX: begin
        gate_sum = {{48{in_reset_q[15]}}, in_reset_q} +
                   {{48{hid_reset_q[15]}}, hid_reset_q};
        activation_input = saturate_i16_sym(gate_sum);
      end
      STATE_UPDATE_INDEX: begin
        gate_sum = {{48{in_update_q[15]}}, in_update_q} +
                   {{48{hid_update_q[15]}}, hid_update_q};
        activation_input = saturate_i16_sym(gate_sum);
      end
      STATE_CANDIDATE_INDEX: begin
        gate_sum = {{48{in_new_q[15]}}, in_new_q} +
                   {{48{reset_hidden_q11_q[15]}}, reset_hidden_q11_q};
        activation_input = saturate_i16_sym(gate_sum);
      end
      default: begin
        gate_sum = 64'sd0;
        activation_input = 16'sd0;
      end
    endcase
    hidden_sum = {{48{candidate_term_q14_q[15]}}, candidate_term_q14_q} +
                 {{48{previous_term_q14_q[15]}}, previous_term_q14_q};
  end

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      state <= STATE_IDLE;
      busy <= 1'b0;
      done <= 1'b0;
      reset_q16 <= 16'd0;
      update_q16 <= 16'd0;
      candidate_q15 <= 16'sd0;
      next_hidden_q14 <= 16'sd0;
      reset_product_q <= 64'sd0;
      reset_hidden_q11_q <= 16'sd0;
      candidate_product_q <= 64'sd0;
      previous_product_q <= 64'sd0;
      candidate_term_q14_q <= 16'sd0;
      previous_term_q14_q <= 16'sd0;
    end else begin
      done <= 1'b0;
      case (state)
        STATE_IDLE: begin
          busy <= 1'b0;
          if (start) begin
            low_precision_q <= low_precision;
            in_reset_q <= input_reset;
            in_update_q <= input_update;
            in_new_q <= input_new;
            hid_reset_q <= hidden_reset;
            hid_update_q <= hidden_update;
            hid_new_q <= hidden_new;
            previous_hidden_q <= previous_hidden;
            busy <= 1'b1;
            state <= STATE_RESET_INDEX;
          end
        end
        STATE_RESET_INDEX: begin
          state <= STATE_RESET_INDEX_WAIT;
        end
        STATE_RESET_INDEX_WAIT: begin
          state <= STATE_RESET_STORE;
        end
        STATE_RESET_STORE: begin
          reset_q16 <= low_precision_q ? reduced_reset : sigmoid_value;
          state <= STATE_UPDATE_INDEX;
        end
        STATE_UPDATE_INDEX: begin
          state <= STATE_UPDATE_INDEX_WAIT;
        end
        STATE_UPDATE_INDEX_WAIT: begin
          state <= STATE_UPDATE_STORE;
        end
        STATE_UPDATE_STORE: begin
          update_q16 <= low_precision_q ? reduced_update : sigmoid_value;
          state <= STATE_RESET_PRODUCT;
        end
        STATE_RESET_PRODUCT: begin
          reset_product_q <= reset_product;
          state <= STATE_RESET_ROUND;
        end
        STATE_RESET_ROUND: begin
          reset_hidden_q11_q <= reset_hidden_q11;
          state <= STATE_CANDIDATE_INDEX;
        end
        STATE_CANDIDATE_INDEX: begin
          state <= STATE_CANDIDATE_INDEX_WAIT;
        end
        STATE_CANDIDATE_INDEX_WAIT: begin
          state <= STATE_CANDIDATE_STORE;
        end
        STATE_CANDIDATE_STORE: begin
          candidate_q15 <= tanh_value;
          state <= STATE_OUTPUT_PRODUCTS;
        end
        STATE_OUTPUT_PRODUCTS: begin
          candidate_product_q <= candidate_product;
          previous_product_q <= previous_product;
          state <= STATE_OUTPUT_ROUND;
        end
        STATE_OUTPUT_ROUND: begin
          candidate_term_q14_q <= candidate_term_q14;
          previous_term_q14_q <= previous_term_q14;
          state <= STATE_OUTPUT;
        end
        STATE_OUTPUT: begin
          next_hidden_q14 <= saturate_i16_sym(hidden_sum);
          busy <= 1'b0;
          done <= 1'b1;
          state <= STATE_IDLE;
        end
        default: state <= STATE_IDLE;
      endcase
    end
  end
endmodule

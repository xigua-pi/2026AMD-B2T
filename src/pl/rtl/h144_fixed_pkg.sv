`timescale 1ns/1ps

package h144_fixed_pkg;
  // Auto-generated from fixed_point_specification.json.
  localparam int INPUT_DIM = 192;
  localparam int BOTTLENECK_DIM = 80;
  localparam int HIDDEN_DIM = 144;
  localparam int OUTPUT_DIM = 32;
  localparam int ADAPTER_RANK = 8;
  localparam int TEMPORAL_KERNEL = 13;
  localparam int SKIP_LEN = 5;
  localparam int LUT_ENTRIES = 512;
  localparam int ACC_BITS = 32;
  localparam int WEIGHT_BITS = 8;
  localparam int signed INPUT_Q_EXP = -4;
  localparam int signed ADAPTER_MID_Q_EXP = -3;
  localparam int signed ADAPTER_OUT_Q_EXP = -4;
  localparam int signed DEPTHWISE_Q_EXP = -5;
  localparam int signed BOTTLENECK_Q_EXP = -5;
  localparam int signed GRU2_INPUT_Q_EXP = -6;
  localparam int signed HIDDEN1_Q_EXP = -14;
  localparam int signed HIDDEN2_Q_EXP = -14;
  localparam int signed PREACT1_Q_EXP = -11;
  localparam int signed PREACT2_Q_EXP = -11;
  localparam int signed RESET_Q_EXP = -16;
  localparam int signed UPDATE_Q_EXP = -16;
  localparam int signed CANDIDATE_Q_EXP = -15;
  localparam int signed LOGIT_Q_EXP = -8;
endpackage

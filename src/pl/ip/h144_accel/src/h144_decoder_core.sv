`timescale 1ns/1ps

module h144_decoder_core #(
    parameter ADAPTER_BIAS_FILE = "accelerator/adapter_bias_i8.mem",
    parameter START_STATE_1_FILE = "accelerator/start_state_1_i16.mem",
    parameter START_STATE_2_FILE = "accelerator/start_state_2_i16.mem"
) (
    input  logic clk,
    input  logic rst_n,
    input  logic sequence_reset,
    input  logic [3:0] session_id,
    input logic [1:0] precision_mode,
    output logic [31:0] low_precision_elements,
    output logic [31:0] total_precision_elements,

    input  logic s_axis_tvalid,
    output logic s_axis_tready,
    input  logic signed [7:0] s_axis_tdata,
    input  logic s_axis_tlast,
    input  logic flush_valid,
    output logic flush_ready,

    output logic m_axis_tvalid,
    input  logic m_axis_tready,
    output logic signed [15:0] m_axis_tdata,
    output logic [5:0] m_axis_tindex,
    output logic m_axis_tlast,

    output logic sequence_ready,
    output logic frame_done,
    output logic result_frame_done,
    output logic logits_emitted,
    output logic protocol_error,
    output logic accumulator_overflow,
    output logic [31:0] last_frame_cycles
);
  import h144_fixed_pkg::*;
  import h144_layout_pkg::*;
  import h144_fixed_math_pkg::*;

  typedef enum logic [5:0] {
    ST_INIT_HIDDEN,
    ST_INIT_HISTORY,
    ST_INPUT,
    ST_ADAPTER_V_LOAD,
    ST_ADAPTER_V_START,
    ST_ADAPTER_V_WAIT,
    ST_ADAPTER_U_LOAD,
    ST_ADAPTER_U_START,
    ST_ADAPTER_U_WAIT,
    ST_ADAPTER_COMBINE,
    ST_FLUSH_SHIFT,
    ST_DEPTH_LOAD,
    ST_DEPTH_START,
    ST_DEPTH_WAIT,
    ST_POINT_LOAD,
    ST_POINT_START,
    ST_POINT_WAIT,
    ST_GRU1_IH_LOAD,
    ST_GRU1_IH_START,
    ST_GRU1_IH_WAIT,
    ST_GRU1_HH_LOAD,
    ST_GRU1_HH_START,
    ST_GRU1_HH_WAIT,
    ST_GRU1_UPDATE_START,
    ST_GRU1_UPDATE_WAIT,
    ST_SKIP_CHECK,
    ST_GRU2_INPUT,
    ST_GRU2_IH_LOAD,
    ST_GRU2_IH_START,
    ST_GRU2_IH_WAIT,
    ST_GRU2_HH_LOAD,
    ST_GRU2_HH_START,
    ST_GRU2_HH_WAIT,
    ST_GRU2_UPDATE_START,
    ST_GRU2_UPDATE_WAIT,
    ST_READOUT_LOAD,
    ST_READOUT_START,
    ST_READOUT_WAIT,
    ST_EMIT,
    ST_WARMUP_FINISH,
    ST_FINISH
  } state_t;

  state_t state;

  logic signed [7:0] input_frame [0:191];
  logic signed [7:0] adapter_mid [0:7];
  logic signed [7:0] adapter_low_rank [0:191];
  logic signed [7:0] adapter_output [0:191];
  logic signed [7:0] temporal_history [0:12][0:191];
  logic signed [7:0] depthwise_output [0:191];
  logic signed [7:0] bottleneck_output [0:79];
  logic signed [15:0] gru1_input_linear [0:431];
  logic signed [15:0] gru1_hidden_linear [0:431];
  logic signed [15:0] hidden1 [0:143];
  logic signed [7:0] gru2_input [0:143];
  logic signed [15:0] gru2_input_linear [0:431];
  logic signed [15:0] gru2_hidden_linear [0:431];
  logic signed [15:0] hidden2 [0:143];
  logic signed [15:0] logits [0:31];
  logic signed [7:0] previous_input1 [0:79];
  logic signed [7:0] previous_input2 [0:143];
  logic [8:0] input_delta1, input_delta2;
  logic [16:0] hidden_delta1, hidden_delta2;
  logic [16:0] current_hidden_delta1, current_hidden_delta2;
  logic has_history1, has_history2;
  logic low_mode1, low_mode2, gru_low_precision;
  logic signed [8:0] input_difference1, input_difference2;
  logic [8:0] abs_input_difference1, abs_input_difference2;
  logic signed [16:0] hidden_difference;
  logic [16:0] abs_hidden_difference;

  (* rom_style = "distributed" *) logic signed [7:0] adapter_bias_rom [0:1919];
  (* rom_style = "distributed" *) logic signed [15:0] start_state_1_rom [0:143];
  (* rom_style = "distributed" *) logic signed [15:0] start_state_2_rom [0:143];

  logic [7:0] input_index;
  logic [7:0] load_index;
  logic [8:0] depth_channel;
  logic [7:0] hidden_index;
  logic [5:0] emit_index;
  logic [7:0] init_hidden_index;
  logic [3:0] init_history_tap;
  logic [7:0] init_history_channel;
  logic [2:0] skip_counter;
  logic [2:0] warmup_count;
  logic [3:0] frame_session;
  logic [31:0] frame_cycle_counter;

  logic mac_x_we;
  logic [7:0] mac_x_addr;
  logic signed [15:0] mac_x_data;
  logic mac_start;
  logic [15:0] mac_weight_base;
  logic [12:0] mac_parameter_base;
  logic [7:0] mac_chunks;
  logic [8:0] mac_outputs;
  logic signed [7:0] mac_input_exp;
  logic signed [7:0] mac_target_exp;
  logic mac_output_i16;
  logic mac_busy;
  logic mac_output_valid;
  logic [8:0] mac_output_index;
  logic signed [15:0] mac_output_data;
  logic mac_done;
  logic mac_overflow;

  logic gru_start;
  logic signed [15:0] gru_input_reset;
  logic signed [15:0] gru_input_update;
  logic signed [15:0] gru_input_new;
  logic signed [15:0] gru_hidden_reset;
  logic signed [15:0] gru_hidden_update;
  logic signed [15:0] gru_hidden_new;
  logic signed [15:0] gru_previous_hidden;
  logic gru_busy;
  logic gru_done;
  logic [15:0] gru_reset_q16;
  logic [15:0] gru_update_q16;
  logic signed [15:0] gru_candidate_q15;
  logic signed [15:0] gru_next_hidden_q14;

  logic signed [15:0] adapter_sum;
  logic signed [63:0] gru2_shifted;

  initial begin
    $readmemh(ADAPTER_BIAS_FILE, adapter_bias_rom);
    $readmemh(START_STATE_1_FILE, start_state_1_rom);
    $readmemh(START_STATE_2_FILE, start_state_2_rom);
  end

  h144_dense_mac8 dense_mac (
      .clk(clk), .rst_n(rst_n),
      .x_write_enable(mac_x_we), .x_write_address(mac_x_addr), .x_write_data(mac_x_data),
      .start(mac_start), .weight_word_base(mac_weight_base),
      .parameter_base(mac_parameter_base), .chunks_per_output(mac_chunks),
      .output_count(mac_outputs), .input_exponent(mac_input_exp),
      .target_exponent(mac_target_exp), .output_is_int16(mac_output_i16),
      .busy(mac_busy), .output_valid(mac_output_valid), .output_index(mac_output_index),
      .output_data(mac_output_data), .done(mac_done),
      .accumulator_overflow(mac_overflow)
  );

  h144_gru_element gru_element (
      .clk(clk), .rst_n(rst_n), .start(gru_start), .low_precision(gru_low_precision),
      .input_reset(gru_input_reset), .input_update(gru_input_update),
      .input_new(gru_input_new), .hidden_reset(gru_hidden_reset),
      .hidden_update(gru_hidden_update), .hidden_new(gru_hidden_new),
      .previous_hidden(gru_previous_hidden), .busy(gru_busy), .done(gru_done),
      .reset_q16(gru_reset_q16), .update_q16(gru_update_q16),
      .candidate_q15(gru_candidate_q15), .next_hidden_q14(gru_next_hidden_q14)
  );

  always_comb begin
    s_axis_tready = (state == ST_INPUT);
    flush_ready = (state == ST_INPUT);
    sequence_ready = (state == ST_INPUT) && (input_index == 0);
    m_axis_tvalid = (state == ST_EMIT);
    m_axis_tdata = logits[emit_index];
    m_axis_tindex = emit_index;
    m_axis_tlast = (state == ST_EMIT) && (emit_index == 31);

    mac_x_we = 1'b0;
    mac_x_addr = load_index;
    mac_x_data = 16'sd0;
    mac_start = 1'b0;

    case (state)
      ST_ADAPTER_V_LOAD: begin
        mac_x_we = 1'b1;
        mac_x_data = {{8{input_frame[load_index][7]}}, input_frame[load_index]};
      end
      ST_ADAPTER_U_LOAD: begin
        mac_x_we = 1'b1;
        mac_x_data = {{8{adapter_mid[load_index][7]}}, adapter_mid[load_index]};
      end
      ST_DEPTH_LOAD: begin
        mac_x_we = 1'b1;
        if (load_index < 13)
          mac_x_data = {{8{temporal_history[load_index][depth_channel][7]}},
                        temporal_history[load_index][depth_channel]};
        else
          mac_x_data = 16'sd0;
      end
      ST_POINT_LOAD: begin
        mac_x_we = 1'b1;
        mac_x_data = {{8{depthwise_output[load_index][7]}}, depthwise_output[load_index]};
      end
      ST_GRU1_IH_LOAD: begin
        mac_x_we = 1'b1;
        mac_x_data = {{8{bottleneck_output[load_index][7]}}, bottleneck_output[load_index]};
      end
      ST_GRU1_HH_LOAD: begin
        mac_x_we = 1'b1;
        mac_x_data = hidden1[load_index];
      end
      ST_GRU2_IH_LOAD: begin
        mac_x_we = 1'b1;
        mac_x_data = {{8{gru2_input[load_index][7]}}, gru2_input[load_index]};
      end
      ST_GRU2_HH_LOAD: begin
        mac_x_we = 1'b1;
        mac_x_data = hidden2[load_index];
      end
      ST_READOUT_LOAD: begin
        mac_x_we = 1'b1;
        mac_x_data = hidden2[load_index];
      end
      default: begin end
    endcase

    mac_weight_base = 16'd0;
    mac_parameter_base = 13'd0;
    mac_chunks = 8'd1;
    mac_outputs = 9'd1;
    mac_input_exp = 8'sd0;
    mac_target_exp = 8'sd0;
    mac_output_i16 = 1'b0;

    case (state)
      ST_ADAPTER_V_START, ST_ADAPTER_V_WAIT: begin
        mac_weight_base = ADAPTER_V_WEIGHT_WORD_BASE +
                          frame_session * ADAPTER_V_SESSION_WEIGHT_WORD_STRIDE;
        mac_parameter_base = ADAPTER_V_PARAMETER_BASE +
                             frame_session * ADAPTER_V_SESSION_PARAMETER_STRIDE;
        mac_chunks = ADAPTER_V_CHUNKS_PER_OUTPUT;
        mac_outputs = ADAPTER_V_OUTPUTS_PER_SESSION;
        mac_input_exp = INPUT_Q_EXP;
        mac_target_exp = ADAPTER_MID_Q_EXP;
        mac_start = (state == ST_ADAPTER_V_START);
      end
      ST_ADAPTER_U_START, ST_ADAPTER_U_WAIT: begin
        mac_weight_base = ADAPTER_U_WEIGHT_WORD_BASE +
                          frame_session * ADAPTER_U_SESSION_WEIGHT_WORD_STRIDE;
        mac_parameter_base = ADAPTER_U_PARAMETER_BASE +
                             frame_session * ADAPTER_U_SESSION_PARAMETER_STRIDE;
        mac_chunks = ADAPTER_U_CHUNKS_PER_OUTPUT;
        mac_outputs = ADAPTER_U_OUTPUTS_PER_SESSION;
        mac_input_exp = ADAPTER_MID_Q_EXP;
        mac_target_exp = ADAPTER_OUT_Q_EXP;
        mac_start = (state == ST_ADAPTER_U_START);
      end
      ST_DEPTH_START, ST_DEPTH_WAIT: begin
        mac_weight_base = DEPTHWISE_WEIGHT_WORD_BASE + depth_channel * DEPTHWISE_CHUNKS_PER_OUTPUT;
        mac_parameter_base = DEPTHWISE_PARAMETER_BASE + depth_channel;
        mac_chunks = DEPTHWISE_CHUNKS_PER_OUTPUT;
        mac_outputs = 9'd1;
        mac_input_exp = ADAPTER_OUT_Q_EXP;
        mac_target_exp = DEPTHWISE_Q_EXP;
        mac_start = (state == ST_DEPTH_START);
      end
      ST_POINT_START, ST_POINT_WAIT: begin
        mac_weight_base = POINTWISE_WEIGHT_WORD_BASE;
        mac_parameter_base = POINTWISE_PARAMETER_BASE;
        mac_chunks = POINTWISE_CHUNKS_PER_OUTPUT;
        mac_outputs = POINTWISE_OUTPUTS_PER_SESSION;
        mac_input_exp = DEPTHWISE_Q_EXP;
        mac_target_exp = BOTTLENECK_Q_EXP;
        mac_start = (state == ST_POINT_START);
      end
      ST_GRU1_IH_START, ST_GRU1_IH_WAIT: begin
        mac_weight_base = GRU1_IH_WEIGHT_WORD_BASE;
        mac_parameter_base = GRU1_IH_PARAMETER_BASE;
        mac_chunks = GRU1_IH_CHUNKS_PER_OUTPUT;
        mac_outputs = GRU1_IH_OUTPUTS_PER_SESSION;
        mac_input_exp = BOTTLENECK_Q_EXP;
        mac_target_exp = PREACT1_Q_EXP;
        mac_output_i16 = 1'b1;
        mac_start = (state == ST_GRU1_IH_START);
      end
      ST_GRU1_HH_START, ST_GRU1_HH_WAIT: begin
        mac_weight_base = GRU1_HH_WEIGHT_WORD_BASE;
        mac_parameter_base = GRU1_HH_PARAMETER_BASE;
        mac_chunks = GRU1_HH_CHUNKS_PER_OUTPUT;
        mac_outputs = GRU1_HH_OUTPUTS_PER_SESSION;
        mac_input_exp = HIDDEN1_Q_EXP;
        mac_target_exp = PREACT1_Q_EXP;
        mac_output_i16 = 1'b1;
        mac_start = (state == ST_GRU1_HH_START);
      end
      ST_GRU2_IH_START, ST_GRU2_IH_WAIT: begin
        mac_weight_base = GRU2_IH_WEIGHT_WORD_BASE;
        mac_parameter_base = GRU2_IH_PARAMETER_BASE;
        mac_chunks = GRU2_IH_CHUNKS_PER_OUTPUT;
        mac_outputs = GRU2_IH_OUTPUTS_PER_SESSION;
        mac_input_exp = GRU2_INPUT_Q_EXP;
        mac_target_exp = PREACT2_Q_EXP;
        mac_output_i16 = 1'b1;
        mac_start = (state == ST_GRU2_IH_START);
      end
      ST_GRU2_HH_START, ST_GRU2_HH_WAIT: begin
        mac_weight_base = GRU2_HH_WEIGHT_WORD_BASE;
        mac_parameter_base = GRU2_HH_PARAMETER_BASE;
        mac_chunks = GRU2_HH_CHUNKS_PER_OUTPUT;
        mac_outputs = GRU2_HH_OUTPUTS_PER_SESSION;
        mac_input_exp = HIDDEN2_Q_EXP;
        mac_target_exp = PREACT2_Q_EXP;
        mac_output_i16 = 1'b1;
        mac_start = (state == ST_GRU2_HH_START);
      end
      ST_READOUT_START, ST_READOUT_WAIT: begin
        mac_weight_base = READOUT_WEIGHT_WORD_BASE;
        mac_parameter_base = READOUT_PARAMETER_BASE;
        mac_chunks = READOUT_CHUNKS_PER_OUTPUT;
        mac_outputs = READOUT_OUTPUTS_PER_SESSION;
        mac_input_exp = HIDDEN2_Q_EXP;
        mac_target_exp = LOGIT_Q_EXP;
        mac_output_i16 = 1'b1;
        mac_start = (state == ST_READOUT_START);
      end
      default: begin end
    endcase

    input_difference1 = $signed(bottleneck_output[load_index]) - $signed(previous_input1[load_index]);
    input_difference2 = $signed(gru2_input[load_index]) - $signed(previous_input2[load_index]);
    abs_input_difference1 = input_difference1[8] ? -input_difference1 : input_difference1;
    abs_input_difference2 = input_difference2[8] ? -input_difference2 : input_difference2;
    hidden_difference = $signed(gru_next_hidden_q14) - $signed(gru_previous_hidden);
    abs_hidden_difference = hidden_difference[16] ? -hidden_difference : hidden_difference;
    gru_low_precision = ((state == ST_GRU1_UPDATE_START) || (state == ST_GRU1_UPDATE_WAIT)) ? low_mode1 : low_mode2;
    gru_start = (state == ST_GRU1_UPDATE_START) || (state == ST_GRU2_UPDATE_START);
    if ((state == ST_GRU1_UPDATE_START) || (state == ST_GRU1_UPDATE_WAIT)) begin
      gru_input_reset = gru1_input_linear[hidden_index];
      gru_input_update = gru1_input_linear[hidden_index + 144];
      gru_input_new = gru1_input_linear[hidden_index + 288];
      gru_hidden_reset = gru1_hidden_linear[hidden_index];
      gru_hidden_update = gru1_hidden_linear[hidden_index + 144];
      gru_hidden_new = gru1_hidden_linear[hidden_index + 288];
      gru_previous_hidden = hidden1[hidden_index];
    end else begin
      gru_input_reset = gru2_input_linear[hidden_index];
      gru_input_update = gru2_input_linear[hidden_index + 144];
      gru_input_new = gru2_input_linear[hidden_index + 288];
      gru_hidden_reset = gru2_hidden_linear[hidden_index];
      gru_hidden_update = gru2_hidden_linear[hidden_index + 144];
      gru_hidden_new = gru2_hidden_linear[hidden_index + 288];
      gru_previous_hidden = hidden2[hidden_index];
    end

    adapter_sum = {{8{input_frame[load_index][7]}}, input_frame[load_index]} +
                  {{8{adapter_low_rank[load_index][7]}}, adapter_low_rank[load_index]} +
                  {{8{adapter_bias_rom[frame_session*192 + load_index][7]}},
                      adapter_bias_rom[frame_session*192 + load_index]};
    gru2_shifted = 64'sd0;
    if (state == ST_GRU2_INPUT)
      gru2_shifted = round_shift_even64(
          {{48{hidden1[load_index][15]}}, hidden1[load_index]}, 8
      );
  end

  integer tap;
  always_ff @(posedge clk) begin
    if (!rst_n) begin
      state <= ST_INIT_HIDDEN;
      input_delta1 <= 0; input_delta2 <= 0;
      hidden_delta1 <= 17'd65535; hidden_delta2 <= 17'd65535;
      current_hidden_delta1 <= 0; current_hidden_delta2 <= 0;
      has_history1 <= 0; has_history2 <= 0;
      low_mode1 <= 0; low_mode2 <= 0;
      low_precision_elements <= 0; total_precision_elements <= 0;

      input_index <= 8'd0;
      load_index <= 8'd0;
      depth_channel <= 9'd0;
      hidden_index <= 8'd0;
      emit_index <= 6'd0;
      init_hidden_index <= 8'd0;
      init_history_tap <= 4'd0;
      init_history_channel <= 8'd0;
      skip_counter <= 3'd0;
      warmup_count <= 3'd0;
      frame_session <= 4'd0;
      frame_cycle_counter <= 32'd0;
      last_frame_cycles <= 32'd0;
      frame_done <= 1'b0;
      result_frame_done <= 1'b0;
      logits_emitted <= 1'b0;
      protocol_error <= 1'b0;
      accumulator_overflow <= 1'b0;
    end else begin
      frame_done <= 1'b0;
      result_frame_done <= 1'b0;
      logits_emitted <= 1'b0;
      if (state != ST_INPUT && state != ST_INIT_HIDDEN && state != ST_INIT_HISTORY)
        frame_cycle_counter <= frame_cycle_counter + 1'b1;
      if (mac_overflow)
        accumulator_overflow <= 1'b1;

      case (state)
        ST_INIT_HIDDEN: begin
          previous_input2[init_hidden_index] <= 0;
          if (init_hidden_index < 80) previous_input1[init_hidden_index] <= 0;
          hidden1[init_hidden_index] <= start_state_1_rom[init_hidden_index];
          hidden2[init_hidden_index] <= start_state_2_rom[init_hidden_index];
          if (init_hidden_index == 143) begin
            init_hidden_index <= 8'd0;
            init_history_tap <= 4'd0;
            init_history_channel <= 8'd0;
            state <= ST_INIT_HISTORY;
          end else
            init_hidden_index <= init_hidden_index + 1'b1;
        end

        ST_INIT_HISTORY: begin
          temporal_history[init_history_tap][init_history_channel] <= 8'sd0;
          if (init_history_channel == 191) begin
            init_history_channel <= 8'd0;
            if (init_history_tap == 12) begin
              init_history_tap <= 4'd0;
              input_index <= 8'd0;
              skip_counter <= 3'd0;
              warmup_count <= 3'd0;
              state <= ST_INPUT;
            end else
              init_history_tap <= init_history_tap + 1'b1;
          end else
            init_history_channel <= init_history_channel + 1'b1;
        end

        ST_INPUT: begin
          if (sequence_reset) begin
            init_hidden_index <= 8'd0;
      input_delta1 <= 0; input_delta2 <= 0;
      hidden_delta1 <= 17'd65535; hidden_delta2 <= 17'd65535;
      current_hidden_delta1 <= 0; current_hidden_delta2 <= 0;
      has_history1 <= 0; has_history2 <= 0;
      low_mode1 <= 0; low_mode2 <= 0;
      low_precision_elements <= 0; total_precision_elements <= 0;
            state <= ST_INIT_HIDDEN;
          end else if (flush_valid) begin
            load_index <= 8'd0;
            frame_cycle_counter <= 32'd0;
            accumulator_overflow <= 1'b0;
            state <= ST_FLUSH_SHIFT;
          end else if (s_axis_tvalid) begin
            input_frame[input_index] <= s_axis_tdata;
            if (input_index == 0)
              frame_session <= session_id;
            if ((input_index == 191) || s_axis_tlast) begin
              if ((input_index != 191) || !s_axis_tlast)
                protocol_error <= 1'b1;
              input_index <= 8'd0;
              load_index <= 8'd0;
              frame_cycle_counter <= 32'd0;
              accumulator_overflow <= 1'b0;
              state <= ST_ADAPTER_V_LOAD;
            end else
              input_index <= input_index + 1'b1;
          end
        end

        ST_ADAPTER_V_LOAD:
          if (load_index == 191) begin load_index <= 0; state <= ST_ADAPTER_V_START; end
          else load_index <= load_index + 1'b1;
        ST_ADAPTER_V_START: state <= ST_ADAPTER_V_WAIT;
        ST_ADAPTER_V_WAIT: begin
          if (mac_output_valid) adapter_mid[mac_output_index] <= mac_output_data[7:0];
          if (mac_done) begin load_index <= 0; state <= ST_ADAPTER_U_LOAD; end
        end

        ST_ADAPTER_U_LOAD:
          if (load_index == 7) begin load_index <= 0; state <= ST_ADAPTER_U_START; end
          else load_index <= load_index + 1'b1;
        ST_ADAPTER_U_START: state <= ST_ADAPTER_U_WAIT;
        ST_ADAPTER_U_WAIT: begin
          if (mac_output_valid) adapter_low_rank[mac_output_index] <= mac_output_data[7:0];
          if (mac_done) begin load_index <= 0; state <= ST_ADAPTER_COMBINE; end
        end

        ST_ADAPTER_COMBINE: begin
          adapter_output[load_index] <= saturate_i8_sym(adapter_sum);
          for (tap = 0; tap < 12; tap = tap + 1)
            temporal_history[tap][load_index] <= temporal_history[tap+1][load_index];
          temporal_history[12][load_index] <= saturate_i8_sym(adapter_sum);
          if (load_index == 191) begin
            load_index <= 0;
            if (warmup_count < 6) begin
              warmup_count <= warmup_count + 1'b1;
              state <= ST_WARMUP_FINISH;
            end else begin
              depth_channel <= 0;
              state <= ST_DEPTH_LOAD;
            end
          end else load_index <= load_index + 1'b1;
        end

        ST_FLUSH_SHIFT: begin
          for (tap = 0; tap < 12; tap = tap + 1)
            temporal_history[tap][load_index] <= temporal_history[tap+1][load_index];
          temporal_history[12][load_index] <= 8'sd0;
          if (load_index == 191) begin
            load_index <= 0;
            depth_channel <= 0;
            state <= ST_DEPTH_LOAD;
          end else load_index <= load_index + 1'b1;
        end

        ST_DEPTH_LOAD:
          if (load_index == 15) begin load_index <= 0; state <= ST_DEPTH_START; end
          else load_index <= load_index + 1'b1;
        ST_DEPTH_START: state <= ST_DEPTH_WAIT;
        ST_DEPTH_WAIT: begin
          if (mac_output_valid)
            depthwise_output[depth_channel] <= mac_output_data[7:0];
          if (mac_done) begin
            if (depth_channel == 191) begin
              depth_channel <= 0;
              load_index <= 0;
              state <= ST_POINT_LOAD;
            end else begin
              depth_channel <= depth_channel + 1'b1;
              load_index <= 0;
              state <= ST_DEPTH_LOAD;
            end
          end
        end

        ST_POINT_LOAD:
          if (load_index == 191) begin load_index <= 0; state <= ST_POINT_START; end
          else load_index <= load_index + 1'b1;
        ST_POINT_START: state <= ST_POINT_WAIT;
        ST_POINT_WAIT: begin
          if (mac_output_valid) bottleneck_output[mac_output_index] <= mac_output_data[7:0];
          if (mac_done) begin load_index <= 0; state <= ST_GRU1_IH_LOAD; end
        end

        ST_GRU1_IH_LOAD: begin
          if (load_index == 0 || abs_input_difference1 > input_delta1) input_delta1 <= abs_input_difference1;
          previous_input1[load_index] <= bottleneck_output[load_index];
          if (load_index == 79) begin load_index <= 0; state <= ST_GRU1_IH_START; end
          else load_index <= load_index + 1'b1;
        end
        ST_GRU1_IH_START: state <= ST_GRU1_IH_WAIT;
        ST_GRU1_IH_WAIT: begin
          if (mac_output_valid) gru1_input_linear[mac_output_index] <= mac_output_data;
          if (mac_done) begin load_index <= 0; state <= ST_GRU1_HH_LOAD; end
        end

        ST_GRU1_HH_LOAD:
          if (load_index == 143) begin load_index <= 0; state <= ST_GRU1_HH_START; end
          else load_index <= load_index + 1'b1;
        ST_GRU1_HH_START: state <= ST_GRU1_HH_WAIT;
        ST_GRU1_HH_WAIT: begin
          if (mac_output_valid) gru1_hidden_linear[mac_output_index] <= mac_output_data;
          if (mac_done) begin
            current_hidden_delta1 <= 0;
            low_mode1 <= (precision_mode == 1) || ((precision_mode == 2) && has_history1 &&
                          input_delta1 <= 8 && hidden_delta1 <= 2048);
            hidden_index <= 0; state <= ST_GRU1_UPDATE_START;
          end
        end

        ST_GRU1_UPDATE_START: state <= ST_GRU1_UPDATE_WAIT;
        ST_GRU1_UPDATE_WAIT: if (gru_done) begin
          total_precision_elements <= total_precision_elements + 1;
          if (low_mode1) low_precision_elements <= low_precision_elements + 1;
          if (abs_hidden_difference > current_hidden_delta1) current_hidden_delta1 <= abs_hidden_difference;
          if (hidden_index == 143) begin
            hidden_delta1 <= (abs_hidden_difference > current_hidden_delta1) ? abs_hidden_difference : current_hidden_delta1;
            has_history1 <= 1;
          end
          hidden1[hidden_index] <= gru_next_hidden_q14;
          if (hidden_index == 143) begin hidden_index <= 0; state <= ST_SKIP_CHECK; end
          else begin hidden_index <= hidden_index + 1'b1; state <= ST_GRU1_UPDATE_START; end
        end

        ST_SKIP_CHECK: begin
          if (skip_counter == 0) begin load_index <= 0; state <= ST_GRU2_INPUT; end
          else state <= ST_FINISH;
        end

        ST_GRU2_INPUT: begin
          gru2_input[load_index] <= saturate_i8_sym(gru2_shifted);
          if (load_index == 143) begin load_index <= 0; state <= ST_GRU2_IH_LOAD; end
          else load_index <= load_index + 1'b1;
        end

        ST_GRU2_IH_LOAD: begin
          if (load_index == 0 || abs_input_difference2 > input_delta2) input_delta2 <= abs_input_difference2;
          previous_input2[load_index] <= gru2_input[load_index];
          if (load_index == 143) begin load_index <= 0; state <= ST_GRU2_IH_START; end
          else load_index <= load_index + 1'b1;
        end
        ST_GRU2_IH_START: state <= ST_GRU2_IH_WAIT;
        ST_GRU2_IH_WAIT: begin
          if (mac_output_valid) gru2_input_linear[mac_output_index] <= mac_output_data;
          if (mac_done) begin load_index <= 0; state <= ST_GRU2_HH_LOAD; end
        end

        ST_GRU2_HH_LOAD:
          if (load_index == 143) begin load_index <= 0; state <= ST_GRU2_HH_START; end
          else load_index <= load_index + 1'b1;
        ST_GRU2_HH_START: state <= ST_GRU2_HH_WAIT;
        ST_GRU2_HH_WAIT: begin
          if (mac_output_valid) gru2_hidden_linear[mac_output_index] <= mac_output_data;
          if (mac_done) begin
            current_hidden_delta2 <= 0;
            low_mode2 <= (precision_mode == 1) || ((precision_mode == 2) && has_history2 &&
                          input_delta2 <= 16 && hidden_delta2 <= 2048);
            hidden_index <= 0; state <= ST_GRU2_UPDATE_START;
          end
        end

        ST_GRU2_UPDATE_START: state <= ST_GRU2_UPDATE_WAIT;
        ST_GRU2_UPDATE_WAIT: if (gru_done) begin
          total_precision_elements <= total_precision_elements + 1;
          if (low_mode2) low_precision_elements <= low_precision_elements + 1;
          if (abs_hidden_difference > current_hidden_delta2) current_hidden_delta2 <= abs_hidden_difference;
          if (hidden_index == 143) begin
            hidden_delta2 <= (abs_hidden_difference > current_hidden_delta2) ? abs_hidden_difference : current_hidden_delta2;
            has_history2 <= 1;
          end
          hidden2[hidden_index] <= gru_next_hidden_q14;
          if (hidden_index == 143) begin hidden_index <= 0; load_index <= 0; state <= ST_READOUT_LOAD; end
          else begin hidden_index <= hidden_index + 1'b1; state <= ST_GRU2_UPDATE_START; end
        end

        ST_READOUT_LOAD:
          if (load_index == 143) begin load_index <= 0; state <= ST_READOUT_START; end
          else load_index <= load_index + 1'b1;
        ST_READOUT_START: state <= ST_READOUT_WAIT;
        ST_READOUT_WAIT: begin
          if (mac_output_valid) logits[mac_output_index] <= mac_output_data;
          if (mac_done) begin emit_index <= 0; state <= ST_EMIT; end
        end

        ST_EMIT: if (m_axis_tready) begin
          if (emit_index == 31) begin
            logits_emitted <= 1'b1;
            emit_index <= 0;
            state <= ST_FINISH;
          end else emit_index <= emit_index + 1'b1;
        end

        ST_WARMUP_FINISH: begin
          frame_done <= 1'b1;
          last_frame_cycles <= frame_cycle_counter;
          state <= ST_INPUT;
        end

        ST_FINISH: begin
          frame_done <= 1'b1;
          result_frame_done <= 1'b1;
          last_frame_cycles <= frame_cycle_counter;
          if (skip_counter == 4) skip_counter <= 0;
          else skip_counter <= skip_counter + 1'b1;
          state <= ST_INPUT;
        end

        default: state <= ST_INIT_HIDDEN;
      endcase
    end
  end
endmodule

`timescale 1ns/1ps

module tb_h144_decoder_core #(
    parameter integer FRAMES_TO_TEST = 32
);
  localparam int TOTAL_FRAMES = FRAMES_TO_TEST;
  localparam int DOWN_FRAMES = (TOTAL_FRAMES+4)/5;
  localparam int INPUT_DIM = 192;
  localparam int BOTTLENECK_DIM = 80;
  localparam int HIDDEN_DIM = 144;
  localparam int OUTPUT_DIM = 32;

  logic clk = 1'b0;
  logic rst_n = 1'b0;
  logic sequence_reset = 1'b0;
  logic [3:0] session_id = 4'd0;
  logic s_axis_tvalid = 1'b0;
  logic s_axis_tready;
  logic signed [7:0] s_axis_tdata = 8'sd0;
  logic s_axis_tlast = 1'b0;
  logic flush_valid = 1'b0;
  logic flush_ready;
  logic m_axis_tvalid;
  logic m_axis_tready = 1'b1;
  logic signed [15:0] m_axis_tdata;
  logic [5:0] m_axis_tindex;
  logic m_axis_tlast;
  logic sequence_ready;
  logic frame_done;
  logic result_frame_done;
  logic logits_emitted;
  logic protocol_error;
  logic accumulator_overflow;
  logic [31:0] last_frame_cycles;

  logic signed [7:0] input_trace [0:TOTAL_FRAMES*INPUT_DIM-1];
  logic signed [7:0] adapter_trace [0:TOTAL_FRAMES*INPUT_DIM-1];
  logic signed [7:0] depthwise_trace [0:TOTAL_FRAMES*INPUT_DIM-1];
  logic signed [7:0] bottleneck_trace [0:TOTAL_FRAMES*BOTTLENECK_DIM-1];
  logic signed [15:0] gru1_trace [0:TOTAL_FRAMES*HIDDEN_DIM-1];
  logic signed [7:0] gru2_input_trace [0:DOWN_FRAMES*HIDDEN_DIM-1];
  logic signed [15:0] gru2_trace [0:DOWN_FRAMES*HIDDEN_DIM-1];
  logic signed [15:0] logits_trace [0:DOWN_FRAMES*OUTPUT_DIM-1];

  integer frame_index;
  integer element_index;
  integer downsample_index = 0;
  integer target_index_for_monitor = -1;
  integer received_logit_index = 0;
  integer errors = 0;
  integer logit_errors = 0;
  integer adapter_errors = 0;
  integer depthwise_errors = 0;
  integer bottleneck_errors = 0;
  integer gru1_errors = 0;
  integer gru2_input_errors = 0;
  integer gru2_errors = 0;
  integer maximum_frame_cycles = 0;
  integer input_frames_to_send;
  integer flushes_to_send;

  always #5 clk = ~clk;

  h144_decoder_core dut (
      .clk(clk), .rst_n(rst_n), .sequence_reset(sequence_reset), .session_id(session_id),
      .s_axis_tvalid(s_axis_tvalid), .s_axis_tready(s_axis_tready),
      .s_axis_tdata(s_axis_tdata), .s_axis_tlast(s_axis_tlast),
      .flush_valid(flush_valid), .flush_ready(flush_ready),
      .m_axis_tvalid(m_axis_tvalid), .m_axis_tready(m_axis_tready),
      .m_axis_tdata(m_axis_tdata), .m_axis_tindex(m_axis_tindex),
      .m_axis_tlast(m_axis_tlast), .sequence_ready(sequence_ready),
      .frame_done(frame_done), .result_frame_done(result_frame_done),
      .logits_emitted(logits_emitted),
      .protocol_error(protocol_error), .accumulator_overflow(accumulator_overflow),
      .last_frame_cycles(last_frame_cycles)
  );

  always @(posedge clk) begin
    if (m_axis_tvalid && m_axis_tready) begin
      if (m_axis_tindex !== received_logit_index[5:0]) begin
        $display("Logit index mismatch frame=%0d expected-index=%0d actual-index=%0d",
                 target_index_for_monitor, received_logit_index, m_axis_tindex);
        errors = errors + 1;
        logit_errors = logit_errors + 1;
      end
      if (m_axis_tdata !== logits_trace[downsample_index*OUTPUT_DIM + received_logit_index]) begin
        $display("Logit mismatch frame=%0d output=%0d expected=%0d actual=%0d",
                 target_index_for_monitor, received_logit_index,
                 logits_trace[downsample_index*OUTPUT_DIM + received_logit_index],
                 m_axis_tdata);
        errors = errors + 1;
        logit_errors = logit_errors + 1;
      end
      if (received_logit_index == OUTPUT_DIM-1)
        received_logit_index = 0;
      else
        received_logit_index = received_logit_index + 1;
    end
  end

  task automatic compare_current_frame(input integer current_frame);
    integer base;
    begin
      base = current_frame * INPUT_DIM;
      for (element_index = 0; element_index < INPUT_DIM; element_index = element_index + 1) begin
        if (dut.temporal_history[6][element_index] !== adapter_trace[base + element_index]) begin
        if (adapter_errors < 10) $display("Adapter mismatch t=%0d i=%0d exp=%0d got=%0d",
                                    current_frame, element_index,
                                    adapter_trace[base+element_index], dut.temporal_history[6][element_index]);
          errors = errors + 1;
          adapter_errors = adapter_errors + 1;
        end
        if (dut.depthwise_output[element_index] !== depthwise_trace[base + element_index]) begin
          if (depthwise_errors < 10) $display("Depthwise mismatch t=%0d i=%0d exp=%0d got=%0d",
                                    current_frame, element_index,
                                    depthwise_trace[base+element_index], dut.depthwise_output[element_index]);
          errors = errors + 1;
          depthwise_errors = depthwise_errors + 1;
        end
      end
      base = current_frame * BOTTLENECK_DIM;
      for (element_index = 0; element_index < BOTTLENECK_DIM; element_index = element_index + 1)
        if (dut.bottleneck_output[element_index] !== bottleneck_trace[base + element_index]) begin
          if (bottleneck_errors < 10) $display("Bottleneck mismatch t=%0d i=%0d exp=%0d got=%0d",
                                    current_frame, element_index,
                                    bottleneck_trace[base+element_index], dut.bottleneck_output[element_index]);
          errors = errors + 1;
          bottleneck_errors = bottleneck_errors + 1;
        end
      base = current_frame * HIDDEN_DIM;
      for (element_index = 0; element_index < HIDDEN_DIM; element_index = element_index + 1)
        if (dut.hidden1[element_index] !== gru1_trace[base + element_index]) begin
          if (gru1_errors < 10) $display("GRU1 mismatch t=%0d i=%0d exp=%0d got=%0d",
                                    current_frame, element_index,
                                    gru1_trace[base+element_index], dut.hidden1[element_index]);
          errors = errors + 1;
          gru1_errors = gru1_errors + 1;
        end
      if ((current_frame % 5) == 0) begin
        base = downsample_index * HIDDEN_DIM;
        for (element_index = 0; element_index < HIDDEN_DIM; element_index = element_index + 1) begin
          if (dut.gru2_input[element_index] !== gru2_input_trace[base + element_index]) begin
            if (gru2_input_errors < 10) $display("GRU2 input mismatch t=%0d i=%0d exp=%0d got=%0d",
                                      current_frame, element_index,
                                      gru2_input_trace[base+element_index], dut.gru2_input[element_index]);
            errors = errors + 1;
            gru2_input_errors = gru2_input_errors + 1;
          end
          if (dut.hidden2[element_index] !== gru2_trace[base + element_index]) begin
            if (gru2_errors < 10) $display("GRU2 mismatch t=%0d i=%0d exp=%0d got=%0d",
                                      current_frame, element_index,
                                      gru2_trace[base+element_index], dut.hidden2[element_index]);
            errors = errors + 1;
            gru2_errors = gru2_errors + 1;
          end
        end
      end
    end
  endtask

  task automatic compare_latest_adapter(input integer input_frame_index);
    integer base;
    begin
      base = input_frame_index * INPUT_DIM;
      for (element_index = 0; element_index < INPUT_DIM; element_index = element_index + 1)
        if (dut.adapter_output[element_index] !== adapter_trace[base + element_index]) begin
          if (adapter_errors < 10) $display("Latest adapter mismatch input=%0d i=%0d exp=%0d got=%0d",
                                            input_frame_index, element_index,
                                            adapter_trace[base+element_index], dut.adapter_output[element_index]);
          errors = errors + 1;
          adapter_errors = adapter_errors + 1;
        end
    end
  endtask

  initial begin
    if (FRAMES_TO_TEST < 13 || FRAMES_TO_TEST > 4096)
      $fatal(1, "Synthetic full-sequence FRAMES must be in [13,4096]");
    $readmemh("trace_record0/input.mem", input_trace);
    $readmemh("trace_record0/adapter.mem", adapter_trace);
    $readmemh("trace_record0/depthwise.mem", depthwise_trace);
    $readmemh("trace_record0/bottleneck.mem", bottleneck_trace);
    $readmemh("trace_record0/gru1_hidden.mem", gru1_trace);
    $readmemh("trace_record0/gru2_input.mem", gru2_input_trace);
    $readmemh("trace_record0/gru2_hidden.mem", gru2_trace);
    $readmemh("trace_record0/logits_down.mem", logits_trace);

    repeat (4) @(posedge clk);
    rst_n = 1'b1;
    wait (sequence_ready);

    // 完整合成序列先输入全部帧，再flush六次，验证warmup与句末边界。
    input_frames_to_send = TOTAL_FRAMES;
    flushes_to_send = 6;

    for (frame_index = 0; frame_index < input_frames_to_send; frame_index = frame_index + 1) begin
      if (frame_index >= 6) begin
        target_index_for_monitor = frame_index - 6;
        downsample_index = target_index_for_monitor / 5;
      end else begin
        target_index_for_monitor = -1;
      end
      wait (s_axis_tready);
      for (element_index = 0; element_index < INPUT_DIM; element_index = element_index + 1) begin
        @(negedge clk);
        s_axis_tvalid = 1'b1;
        s_axis_tdata = input_trace[frame_index*INPUT_DIM + element_index];
        s_axis_tlast = (element_index == INPUT_DIM-1);
      end
      @(negedge clk);
      s_axis_tvalid = 1'b0;
      s_axis_tlast = 1'b0;
      wait (frame_done);
      #1;
      if (last_frame_cycles > maximum_frame_cycles)
        maximum_frame_cycles = last_frame_cycles;
      compare_latest_adapter(frame_index);
      if (frame_index >= 6) begin
        if (!result_frame_done) $fatal(1, "Missing result_frame_done for target %0d", target_index_for_monitor);
        compare_current_frame(target_index_for_monitor);
      end else if (result_frame_done) begin
        $fatal(1, "Unexpected result during six-frame look-ahead warmup");
      end
      if (protocol_error) $fatal(1, "Input stream protocol error");
      if (accumulator_overflow) $fatal(1, "Unexpected INT32 accumulator overflow");
      @(posedge clk);
    end


    for (frame_index = 0; frame_index < flushes_to_send; frame_index = frame_index + 1) begin
      target_index_for_monitor = TOTAL_FRAMES - 6 + frame_index;
      downsample_index = target_index_for_monitor / 5;
      wait (flush_ready);
      @(negedge clk);
      flush_valid = 1'b1;
      @(negedge clk);
      flush_valid = 1'b0;
      wait (frame_done);
      #1;
      if (!result_frame_done) $fatal(1, "Missing result_frame_done during flush");
      if (last_frame_cycles > maximum_frame_cycles)
        maximum_frame_cycles = last_frame_cycles;
      compare_current_frame(target_index_for_monitor);
      if (accumulator_overflow) $fatal(1, "Unexpected INT32 accumulator overflow");
      @(posedge clk);
    end

    $display("ERROR_SUMMARY logits=%0d adapter=%0d depthwise=%0d bottleneck=%0d gru1=%0d gru2_input=%0d gru2=%0d",
             logit_errors, adapter_errors, depthwise_errors, bottleneck_errors,
             gru1_errors, gru2_input_errors, gru2_errors);
    if (errors != 0)
      $fatal(1, "Full decoder core failed: %0d mismatches over %0d frames", errors, FRAMES_TO_TEST);
    $display("PASS: full H144 RTL matched all layer traces for %0d consecutive frames", FRAMES_TO_TEST);
    $display("MAX_FRAME_CYCLES=%0d LATENCY_US_AT_100MHZ=%0f",
             maximum_frame_cycles, maximum_frame_cycles / 100.0);
    $finish;
  end
endmodule

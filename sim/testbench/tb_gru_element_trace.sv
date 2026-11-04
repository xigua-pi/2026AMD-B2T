`timescale 1ns/1ps

module tb_gru_element_trace #(
  parameter integer GRU1_STEPS = 32
);
  localparam int GRU2_STEPS = (GRU1_STEPS+4)/5;
  localparam int HIDDEN = 144;
  localparam int GATES = 432;

  logic clk = 1'b0;
  logic rst_n = 1'b0;
  logic start = 1'b0;
  logic signed [15:0] input_reset;
  logic signed [15:0] input_update;
  logic signed [15:0] input_new;
  logic signed [15:0] hidden_reset;
  logic signed [15:0] hidden_update;
  logic signed [15:0] hidden_new;
  logic signed [15:0] previous_hidden;
  logic busy;
  logic done;
  logic [15:0] reset_q16;
  logic [15:0] update_q16;
  logic signed [15:0] candidate_q15;
  logic signed [15:0] next_hidden_q14;

  logic signed [15:0] g1_input_linear [0:GRU1_STEPS*GATES-1];
  logic signed [15:0] g1_hidden_linear [0:GRU1_STEPS*GATES-1];
  logic signed [15:0] g1_previous [0:GRU1_STEPS*HIDDEN-1];
  logic [15:0] g1_reset [0:GRU1_STEPS*HIDDEN-1];
  logic [15:0] g1_update [0:GRU1_STEPS*HIDDEN-1];
  logic signed [15:0] g1_candidate [0:GRU1_STEPS*HIDDEN-1];
  logic signed [15:0] g1_hidden [0:GRU1_STEPS*HIDDEN-1];

  logic signed [15:0] g2_input_linear [0:GRU2_STEPS*GATES-1];
  logic signed [15:0] g2_hidden_linear [0:GRU2_STEPS*GATES-1];
  logic signed [15:0] g2_previous [0:GRU2_STEPS*HIDDEN-1];
  logic [15:0] g2_reset [0:GRU2_STEPS*HIDDEN-1];
  logic [15:0] g2_update [0:GRU2_STEPS*HIDDEN-1];
  logic signed [15:0] g2_candidate [0:GRU2_STEPS*HIDDEN-1];
  logic signed [15:0] g2_hidden [0:GRU2_STEPS*HIDDEN-1];

  integer time_index;
  integer hidden_index;
  integer flat_hidden;
  integer flat_gate;
  integer errors = 0;
  integer checked = 0;

  always #5 clk = ~clk;

  h144_gru_element dut (
      .clk(clk), .rst_n(rst_n), .start(start),
      .input_reset(input_reset), .input_update(input_update), .input_new(input_new),
      .hidden_reset(hidden_reset), .hidden_update(hidden_update), .hidden_new(hidden_new),
      .previous_hidden(previous_hidden), .busy(busy), .done(done),
      .reset_q16(reset_q16), .update_q16(update_q16),
      .candidate_q15(candidate_q15), .next_hidden_q14(next_hidden_q14)
  );

  task automatic drive_and_check(
      input logic signed [15:0] ir,
      input logic signed [15:0] iz,
      input logic signed [15:0] inew,
      input logic signed [15:0] hr,
      input logic signed [15:0] hz,
      input logic signed [15:0] hnew,
      input logic signed [15:0] prev,
      input logic [15:0] exp_r,
      input logic [15:0] exp_z,
      input logic signed [15:0] exp_n,
      input logic signed [15:0] exp_h
  );
    begin
      input_reset = ir;
      input_update = iz;
      input_new = inew;
      hidden_reset = hr;
      hidden_update = hz;
      hidden_new = hnew;
      previous_hidden = prev;
      start = 1'b1;
      @(posedge clk);
      start = 1'b0;
      wait (done);
      #1;
      if (reset_q16 !== exp_r || update_q16 !== exp_z ||
          candidate_q15 !== exp_n || next_hidden_q14 !== exp_h) begin
        if (errors < 20)
          $display("Mismatch checked=%0d r=%0d/%0d z=%0d/%0d n=%0d/%0d h=%0d/%0d",
                   checked, reset_q16, exp_r, update_q16, exp_z,
                   candidate_q15, exp_n, next_hidden_q14, exp_h);
        errors = errors + 1;
      end
      checked = checked + 1;
      @(posedge clk);
    end
  endtask

  initial begin
    $readmemh("trace_record0/gru1_gates_input_linear.mem", g1_input_linear);
    $readmemh("trace_record0/gru1_gates_hidden_linear.mem", g1_hidden_linear);
    $readmemh("trace_record0/gru1_gates_previous_hidden.mem", g1_previous);
    $readmemh("trace_record0/gru1_gates_reset.mem", g1_reset);
    $readmemh("trace_record0/gru1_gates_update.mem", g1_update);
    $readmemh("trace_record0/gru1_gates_candidate.mem", g1_candidate);
    $readmemh("trace_record0/gru1_hidden.mem", g1_hidden);
    $readmemh("trace_record0/gru2_gates_input_linear.mem", g2_input_linear);
    $readmemh("trace_record0/gru2_gates_hidden_linear.mem", g2_hidden_linear);
    $readmemh("trace_record0/gru2_gates_previous_hidden.mem", g2_previous);
    $readmemh("trace_record0/gru2_gates_reset.mem", g2_reset);
    $readmemh("trace_record0/gru2_gates_update.mem", g2_update);
    $readmemh("trace_record0/gru2_gates_candidate.mem", g2_candidate);
    $readmemh("trace_record0/gru2_hidden.mem", g2_hidden);

    repeat (4) @(posedge clk);
    rst_n = 1'b1;
    @(posedge clk);

    for (time_index = 0; time_index < GRU1_STEPS; time_index = time_index + 1)
      for (hidden_index = 0; hidden_index < HIDDEN; hidden_index = hidden_index + 1) begin
        flat_hidden = time_index*HIDDEN + hidden_index;
        flat_gate = time_index*GATES + hidden_index;
        drive_and_check(
            g1_input_linear[flat_gate],
            g1_input_linear[flat_gate + HIDDEN],
            g1_input_linear[flat_gate + 2*HIDDEN],
            g1_hidden_linear[flat_gate],
            g1_hidden_linear[flat_gate + HIDDEN],
            g1_hidden_linear[flat_gate + 2*HIDDEN],
            g1_previous[flat_hidden],
            g1_reset[flat_hidden], g1_update[flat_hidden],
            g1_candidate[flat_hidden], g1_hidden[flat_hidden]
        );
      end

    for (time_index = 0; time_index < GRU2_STEPS; time_index = time_index + 1)
      for (hidden_index = 0; hidden_index < HIDDEN; hidden_index = hidden_index + 1) begin
        flat_hidden = time_index*HIDDEN + hidden_index;
        flat_gate = time_index*GATES + hidden_index;
        drive_and_check(
            g2_input_linear[flat_gate],
            g2_input_linear[flat_gate + HIDDEN],
            g2_input_linear[flat_gate + 2*HIDDEN],
            g2_hidden_linear[flat_gate],
            g2_hidden_linear[flat_gate + HIDDEN],
            g2_hidden_linear[flat_gate + 2*HIDDEN],
            g2_previous[flat_hidden],
            g2_reset[flat_hidden], g2_update[flat_hidden],
            g2_candidate[flat_hidden], g2_hidden[flat_hidden]
        );
      end

    if (errors != 0)
      $fatal(1, "GRU element failed: %0d/%0d mismatches", errors, checked);
    $display("PASS: GRU gate/hidden equations matched %0d synthetic recurrent elements", checked);
    $finish;
  end
endmodule

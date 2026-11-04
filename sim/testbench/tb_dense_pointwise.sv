`timescale 1ns/1ps

module tb_dense_pointwise #(
  parameter integer TRACE_FRAMES = 32
);
  import h144_fixed_pkg::*;
  import h144_layout_pkg::*;

  logic clk = 1'b0;
  logic rst_n = 1'b0;
  logic x_write_enable = 1'b0;
  logic [7:0] x_write_address = 8'd0;
  logic signed [15:0] x_write_data = 16'sd0;
  logic start = 1'b0;
  logic busy;
  logic output_valid;
  logic [8:0] output_index;
  logic signed [15:0] output_data;
  logic done;
  logic accumulator_overflow;

  logic signed [7:0] depthwise_trace [0:TRACE_FRAMES*192-1];
  logic signed [7:0] bottleneck_trace [0:TRACE_FRAMES*80-1];
  integer index;
  integer errors = 0;
  integer cycles = 0;

  always #5 clk = ~clk;

  h144_dense_mac8 dut (
      .clk(clk),
      .rst_n(rst_n),
      .x_write_enable(x_write_enable),
      .x_write_address(x_write_address),
      .x_write_data(x_write_data),
      .start(start),
      .weight_word_base(POINTWISE_WEIGHT_WORD_BASE),
      .parameter_base(POINTWISE_PARAMETER_BASE),
      .chunks_per_output(POINTWISE_CHUNKS_PER_OUTPUT),
      .output_count(POINTWISE_OUTPUTS_PER_SESSION),
      .input_exponent(DEPTHWISE_Q_EXP),
      .target_exponent(BOTTLENECK_Q_EXP),
      .output_is_int16(1'b0),
      .busy(busy),
      .output_valid(output_valid),
      .output_index(output_index),
      .output_data(output_data),
      .done(done),
      .accumulator_overflow(accumulator_overflow)
  );

  always @(posedge clk) begin
    if (rst_n)
      cycles <= cycles + 1;
    if (output_valid) begin
      if (output_data[7:0] !== bottleneck_trace[output_index]) begin
        $display("Mismatch output=%0d expected=%0d actual=%0d",
                 output_index, bottleneck_trace[output_index], $signed(output_data));
        errors <= errors + 1;
      end
    end
    if (cycles > 10000)
      $fatal(1, "Timeout");
  end

  initial begin
    $readmemh("trace_record0/depthwise.mem", depthwise_trace);
    $readmemh("trace_record0/bottleneck.mem", bottleneck_trace);

    repeat (4) @(posedge clk);
    rst_n <= 1'b1;
    @(posedge clk);

    for (index = 0; index < 192; index = index + 1) begin
      x_write_enable <= 1'b1;
      x_write_address <= index[7:0];
      x_write_data <= {{8{depthwise_trace[index][7]}}, depthwise_trace[index]};
      @(posedge clk);
    end
    x_write_enable <= 1'b0;
    start <= 1'b1;
    @(posedge clk);
    start <= 1'b0;

    wait (done);
    @(posedge clk);
    if (accumulator_overflow)
      $fatal(1, "Unexpected INT32 accumulator overflow");
    if (errors != 0)
      $fatal(1, "Pointwise MAC bit-exact check failed: %0d mismatches", errors);
    $display("PASS: pointwise 80x192 MAC and requantization are bit-exact, cycles=%0d", cycles);
    $finish;
  end
endmodule

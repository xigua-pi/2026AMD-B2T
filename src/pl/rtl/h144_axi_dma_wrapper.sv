`timescale 1ns/1ps

// PYNQ-Z2 部署包装器：
// 1) 将 AXI DMA 的整句 8-bit 字节流重新划分为 192 通道神经帧；
// 2) 在输入结束后自动执行 6 帧零填充，以补偿因果卷积的尾部延迟；
// 3) 将每 32 个 logit 的内部包合并为一笔整句 S2MM 传输；
// 4) 提供固定地址的 AXI4-Lite 控制、状态和性能计数器。
module h144_axi_dma_wrapper (
    (* X_INTERFACE_PARAMETER = "XIL_INTERFACENAME aclk, ASSOCIATED_BUSIF S_AXI:S_AXIS:M_AXIS, ASSOCIATED_RESET aresetn, FREQ_HZ 50000000" *)
    (* X_INTERFACE_INFO = "xilinx.com:signal:clock:1.0 aclk CLK" *)
    input  logic        aclk,
    (* X_INTERFACE_PARAMETER = "XIL_INTERFACENAME aresetn, POLARITY ACTIVE_LOW" *)
    (* X_INTERFACE_INFO = "xilinx.com:signal:reset:1.0 aresetn RST" *)
    input  logic        aresetn,

    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI AWADDR" *) input  logic [7:0]  s_axi_awaddr,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI AWVALID" *) input  logic        s_axi_awvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI AWREADY" *) output logic        s_axi_awready,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI WDATA" *) input  logic [31:0] s_axi_wdata,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI WSTRB" *) input  logic [3:0]  s_axi_wstrb,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI WVALID" *) input  logic        s_axi_wvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI WREADY" *) output logic        s_axi_wready,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI BRESP" *) output logic [1:0]  s_axi_bresp,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI BVALID" *) output logic        s_axi_bvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI BREADY" *) input  logic        s_axi_bready,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI ARADDR" *) input  logic [7:0]  s_axi_araddr,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI ARVALID" *) input  logic        s_axi_arvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI ARREADY" *) output logic        s_axi_arready,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI RDATA" *) output logic [31:0] s_axi_rdata,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI RRESP" *) output logic [1:0]  s_axi_rresp,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI RVALID" *) output logic        s_axi_rvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI RREADY" *) input  logic        s_axi_rready,

    (* X_INTERFACE_PARAMETER = "XIL_INTERFACENAME S_AXIS, TDATA_NUM_BYTES 1, HAS_TKEEP 1, HAS_TLAST 1" *)
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 S_AXIS TDATA" *) input  logic [7:0]  s_axis_tdata,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 S_AXIS TKEEP" *) input  logic [0:0]  s_axis_tkeep,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 S_AXIS TVALID" *) input  logic        s_axis_tvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 S_AXIS TREADY" *) output logic        s_axis_tready,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 S_AXIS TLAST" *) input  logic        s_axis_tlast,

    (* X_INTERFACE_PARAMETER = "XIL_INTERFACENAME M_AXIS, TDATA_NUM_BYTES 2, HAS_TKEEP 1, HAS_TLAST 1" *)
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 M_AXIS TDATA" *) output logic [15:0] m_axis_tdata,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 M_AXIS TKEEP" *) output logic [1:0]  m_axis_tkeep,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 M_AXIS TVALID" *) output logic        m_axis_tvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 M_AXIS TREADY" *) input  logic        m_axis_tready,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 M_AXIS TLAST" *) output logic        m_axis_tlast,

    output logic [3:0] status_leds
);
  localparam logic [31:0] DESIGN_VERSION = 32'h0002_0001;
  localparam int INPUT_DIM = 192;
  localparam int TAIL_FLUSH_FRAMES = 6;

  typedef enum logic [3:0] {
    ENGINE_IDLE,
    ENGINE_WAIT_INITIAL_READY,
    ENGINE_RESET_SEQUENCE,
    ENGINE_WAIT_REINITIALIZED,
    ENGINE_FEED_INPUT,
    ENGINE_WAIT_INPUT_FRAME,
    ENGINE_REQUEST_FLUSH,
    ENGINE_WAIT_FLUSH,
    ENGINE_COMPLETE
  } engine_state_t;

  engine_state_t engine_state;

  logic [3:0]  cfg_session_id;
  logic [31:0] cfg_input_frames;
  logic [31:0] cfg_output_words;
  logic        start_pulse;

  logic        engine_busy;
  logic        engine_done;
  logic        input_packet_error;
  logic        output_count_error;
  logic [7:0]  byte_in_frame;
  logic [31:0] input_byte_count;
  logic [31:0] input_frame_count;
  logic [31:0] output_word_count;
  logic [31:0] frame_done_count;
  logic [31:0] result_frame_count;
  logic [31:0] flush_completed_count;
  logic [31:0] total_cycle_count;
  logic [31:0] maximum_frame_cycles;

  logic        core_sequence_reset;
  logic        core_s_tvalid;
  logic        core_s_tready;
  logic signed [7:0] core_s_tdata;
  logic        core_s_tlast;
  logic        core_flush_valid;
  logic        core_flush_ready;
  logic        core_m_tvalid;
  logic        core_m_tready;
  logic signed [15:0] core_m_tdata;
  logic [5:0]  core_m_tindex;
  logic        core_m_tlast;
  logic        core_sequence_ready;
  logic        core_frame_done;
  logic        core_result_frame_done;
  logic        core_logits_emitted;
  logic        core_protocol_error;
  logic        core_accumulator_overflow;
  logic [31:0] core_last_frame_cycles;
  logic [1:0] cfg_precision_mode;
  logic [31:0] core_low_precision_elements, core_total_precision_elements;

  logic        aw_pending;
  logic [7:0]  awaddr_q;
  logic        w_pending;
  logic [31:0] wdata_q;
  logic [3:0]  wstrb_q;

  function automatic logic [31:0] apply_write_strobes(
      input logic [31:0] old_value,
      input logic [31:0] new_value,
      input logic [3:0] strobes
  );
    integer byte_index;
    begin
      apply_write_strobes = old_value;
      for (byte_index = 0; byte_index < 4; byte_index = byte_index + 1)
        if (strobes[byte_index])
          apply_write_strobes[byte_index*8 +: 8] = new_value[byte_index*8 +: 8];
    end
  endfunction

  function automatic logic [31:0] read_register(input logic [7:0] address);
    logic any_error;
    begin
      any_error = core_protocol_error | core_accumulator_overflow |
                  input_packet_error | output_count_error;
      case (address[7:2])
        6'h00: read_register = 32'd0; // CONTROL
        6'h01: read_register = {28'd0, any_error, core_sequence_ready,
                                engine_done, engine_busy};
        6'h02: read_register = {28'd0, cfg_session_id};
        6'h03: read_register = cfg_input_frames;
        6'h04: read_register = cfg_output_words;
        6'h05: read_register = input_byte_count;
        6'h06: read_register = output_word_count;
        6'h07: read_register = input_frame_count;
        6'h08: read_register = frame_done_count;
        6'h09: read_register = result_frame_count;
        6'h0a: read_register = flush_completed_count;
        6'h0b: read_register = total_cycle_count;
        6'h0c: read_register = maximum_frame_cycles;
        6'h0d: read_register = {28'd0, output_count_error, input_packet_error,
                                core_accumulator_overflow, core_protocol_error};
        6'h0e: read_register = DESIGN_VERSION;
        6'h0f: read_register = {28'd0, engine_state};
        6'h10: read_register = {30'd0,cfg_precision_mode};
        6'h11: read_register = core_low_precision_elements;
        6'h12: read_register = core_total_precision_elements;
        default: read_register = 32'hdeadc0de;
      endcase
    end
  endfunction

  // AXI4-Lite：控制写入和状态读取均与加速器共用 50 MHz 时钟域。
  always_ff @(posedge aclk) begin
    if (!aresetn) begin
      cfg_precision_mode <= 2;
      cfg_session_id <= 4'd0;
      cfg_input_frames <= 32'd988;
      cfg_output_words <= 32'd6336;
      start_pulse <= 1'b0;
      aw_pending <= 1'b0;
      awaddr_q <= 8'd0;
      w_pending <= 1'b0;
      wdata_q <= 32'd0;
      wstrb_q <= 4'd0;
      s_axi_awready <= 1'b0;
      s_axi_wready <= 1'b0;
      s_axi_bresp <= 2'b00;
      s_axi_bvalid <= 1'b0;
      s_axi_arready <= 1'b0;
      s_axi_rdata <= 32'd0;
      s_axi_rresp <= 2'b00;
      s_axi_rvalid <= 1'b0;
    end else begin
      start_pulse <= 1'b0;
      s_axi_awready <= 1'b0;
      s_axi_wready <= 1'b0;
      s_axi_arready <= 1'b0;

      if (!aw_pending && s_axi_awvalid) begin
        awaddr_q <= s_axi_awaddr;
        aw_pending <= 1'b1;
        s_axi_awready <= 1'b1;
      end
      if (!w_pending && s_axi_wvalid) begin
        wdata_q <= s_axi_wdata;
        wstrb_q <= s_axi_wstrb;
        w_pending <= 1'b1;
        s_axi_wready <= 1'b1;
      end
      if (aw_pending && w_pending && !s_axi_bvalid) begin
        case (awaddr_q[7:2])
          6'h00: if (wstrb_q[0] && wdata_q[0]) start_pulse <= 1'b1;
          6'h02: if (wstrb_q[0]) cfg_session_id <= wdata_q[3:0];
          6'h03: cfg_input_frames <= apply_write_strobes(
              cfg_input_frames, wdata_q, wstrb_q);
          6'h04: cfg_output_words <= apply_write_strobes(
              cfg_output_words, wdata_q, wstrb_q);
          default: begin end
        endcase
        aw_pending <= 1'b0;
        w_pending <= 1'b0;
        s_axi_bresp <= 2'b00;
        s_axi_bvalid <= 1'b1;
      end
      if (s_axi_bvalid && s_axi_bready)
        s_axi_bvalid <= 1'b0;

      if (!s_axi_rvalid && s_axi_arvalid) begin
        s_axi_arready <= 1'b1;
        s_axi_rdata <= read_register(s_axi_araddr);
        s_axi_rresp <= 2'b00;
        s_axi_rvalid <= 1'b1;
      end else if (s_axi_rvalid && s_axi_rready) begin
        s_axi_rvalid <= 1'b0;
      end
    end
  end

  always_comb begin
    core_sequence_reset = (engine_state == ENGINE_RESET_SEQUENCE);
    core_flush_valid = (engine_state == ENGINE_REQUEST_FLUSH);

    core_s_tvalid = (engine_state == ENGINE_FEED_INPUT) && s_axis_tvalid;
    core_s_tdata = $signed(s_axis_tdata);
    core_s_tlast = (byte_in_frame == INPUT_DIM-1);
    s_axis_tready = (engine_state == ENGINE_FEED_INPUT) && core_s_tready;

    m_axis_tvalid = core_m_tvalid;
    m_axis_tdata = core_m_tdata;
    m_axis_tkeep = 2'b11;
    m_axis_tlast = core_m_tvalid && (cfg_output_words != 0) &&
                   (output_word_count == cfg_output_words-1);
    core_m_tready = m_axis_tready;

    status_leds[0] = engine_busy;
    status_leds[1] = engine_done;
    status_leds[2] = core_protocol_error | core_accumulator_overflow |
                     input_packet_error | output_count_error;
    status_leds[3] = core_sequence_ready;
  end

  // 整句事务控制与性能计数。尾部固定冲刷 6 帧，与软件 golden graph 一致。
  always_ff @(posedge aclk) begin
    if (!aresetn) begin
      engine_state <= ENGINE_IDLE;
      engine_busy <= 1'b0;
      engine_done <= 1'b0;
      input_packet_error <= 1'b0;
      output_count_error <= 1'b0;
      byte_in_frame <= 8'd0;
      input_byte_count <= 32'd0;
      input_frame_count <= 32'd0;
      output_word_count <= 32'd0;
      frame_done_count <= 32'd0;
      result_frame_count <= 32'd0;
      flush_completed_count <= 32'd0;
      total_cycle_count <= 32'd0;
      maximum_frame_cycles <= 32'd0;
    end else begin
      if (engine_busy)
        total_cycle_count <= total_cycle_count + 1'b1;

      if (core_m_tvalid && core_m_tready)
        output_word_count <= output_word_count + 1'b1;

      if (core_frame_done) begin
        frame_done_count <= frame_done_count + 1'b1;
        if (core_result_frame_done)
          result_frame_count <= result_frame_count + 1'b1;
        if (core_last_frame_cycles > maximum_frame_cycles)
          maximum_frame_cycles <= core_last_frame_cycles;
      end

      case (engine_state)
        ENGINE_IDLE: begin
          if (start_pulse) begin
            engine_busy <= 1'b1;
            engine_done <= 1'b0;
            input_packet_error <= 1'b0;
            output_count_error <= 1'b0;
            byte_in_frame <= 8'd0;
            input_byte_count <= 32'd0;
            input_frame_count <= 32'd0;
            output_word_count <= 32'd0;
            frame_done_count <= 32'd0;
            result_frame_count <= 32'd0;
            flush_completed_count <= 32'd0;
            total_cycle_count <= 32'd0;
            maximum_frame_cycles <= 32'd0;
            engine_state <= ENGINE_WAIT_INITIAL_READY;
          end
        end

        ENGINE_WAIT_INITIAL_READY:
          if (core_sequence_ready)
            engine_state <= ENGINE_RESET_SEQUENCE;

        ENGINE_RESET_SEQUENCE:
          engine_state <= ENGINE_WAIT_REINITIALIZED;

        ENGINE_WAIT_REINITIALIZED:
          if (core_sequence_ready)
            engine_state <= ENGINE_FEED_INPUT;

        ENGINE_FEED_INPUT: begin
          if (s_axis_tvalid && s_axis_tready) begin
            input_byte_count <= input_byte_count + 1'b1;
            if (!s_axis_tkeep[0])
              input_packet_error <= 1'b1;

            if (byte_in_frame == INPUT_DIM-1) begin
              byte_in_frame <= 8'd0;
              input_frame_count <= input_frame_count + 1'b1;
            end else begin
              byte_in_frame <= byte_in_frame + 1'b1;
            end

            if (s_axis_tlast) begin
              if ((byte_in_frame != INPUT_DIM-1) ||
                  (input_frame_count + 1'b1 != cfg_input_frames))
                input_packet_error <= 1'b1;
              engine_state <= ENGINE_WAIT_INPUT_FRAME;
            end else if ((byte_in_frame == INPUT_DIM-1) &&
                         (input_frame_count + 1'b1 >= cfg_input_frames)) begin
              input_packet_error <= 1'b1;
            end
          end
        end

        ENGINE_WAIT_INPUT_FRAME:
          if (core_frame_done)
            engine_state <= ENGINE_REQUEST_FLUSH;

        ENGINE_REQUEST_FLUSH:
          if (core_flush_ready)
            engine_state <= ENGINE_WAIT_FLUSH;

        ENGINE_WAIT_FLUSH: begin
          if (core_frame_done) begin
            flush_completed_count <= flush_completed_count + 1'b1;
            if (flush_completed_count == TAIL_FLUSH_FRAMES-1)
              engine_state <= ENGINE_COMPLETE;
            else
              engine_state <= ENGINE_REQUEST_FLUSH;
          end
        end

        ENGINE_COMPLETE: begin
          if (output_word_count != cfg_output_words)
            output_count_error <= 1'b1;
          engine_busy <= 1'b0;
          engine_done <= 1'b1;
          engine_state <= ENGINE_IDLE;
        end

        default: engine_state <= ENGINE_IDLE;
      endcase
    end
  end

  h144_decoder_core decoder_core (
      .clk(aclk),
      .rst_n(aresetn),
      .sequence_reset(core_sequence_reset),
      .session_id(cfg_session_id),
      .precision_mode(cfg_precision_mode),
      .low_precision_elements(core_low_precision_elements),
      .total_precision_elements(core_total_precision_elements),
      .s_axis_tvalid(core_s_tvalid),
      .s_axis_tready(core_s_tready),
      .s_axis_tdata(core_s_tdata),
      .s_axis_tlast(core_s_tlast),
      .flush_valid(core_flush_valid),
      .flush_ready(core_flush_ready),
      .m_axis_tvalid(core_m_tvalid),
      .m_axis_tready(core_m_tready),
      .m_axis_tdata(core_m_tdata),
      .m_axis_tindex(core_m_tindex),
      .m_axis_tlast(core_m_tlast),
      .sequence_ready(core_sequence_ready),
      .frame_done(core_frame_done),
      .result_frame_done(core_result_frame_done),
      .logits_emitted(core_logits_emitted),
      .protocol_error(core_protocol_error),
      .accumulator_overflow(core_accumulator_overflow),
      .last_frame_cycles(core_last_frame_cycles)
  );
endmodule

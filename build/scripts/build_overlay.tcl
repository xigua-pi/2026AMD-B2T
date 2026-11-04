# 从当前脚本定位工程，不依赖原开发机器的盘符。
set root [file normalize [file join [file dirname [info script]] ".." ".."]]
set vivado_root [file join $root src pl]
set memory_dir [file join $root data golden_ref rtl_vectors accelerator]
set export_dir [file join $root board bitstream]
set report_dir [file join $root build reports]
set project_dir [file join $root build vivado_project]
set project_name h144_dma_overlay
set design_name h144_overlay
set base_reference [file join $root build scripts pynq_3_0_1_base_reference.tcl]
if {[file exists [file join $project_dir "${project_name}.xpr"]]} {
    error "Project already exists; choose a new output directory before rebuilding."
}
file mkdir $export_dir
file mkdir $report_dir

create_project -force $project_name $project_dir -part xc7z020clg400-1
set_property target_language Verilog [current_project]
set_property simulator_language Mixed [current_project]

set rtl_sources [list \
    [file join $vivado_root "rtl" "h144_fixed_pkg.sv"] \
    [file join $vivado_root "rtl" "h144_layout_pkg.sv"] \
    [file join $vivado_root "rtl" "h144_fixed_math_pkg.sv"] \
    [file join $vivado_root "rtl" "h144_requantizer.sv"] \
    [file join $vivado_root "rtl" "h144_activation_lut.sv"] \
    [file join $vivado_root "rtl" "h144_dense_mac8.sv"] \
    [file join $vivado_root "rtl" "h144_gru_element.sv"] \
    [file join $vivado_root "rtl" "h144_decoder_core.sv"] \
    [file join $vivado_root "rtl" "h144_axi_dma_wrapper.sv"] \
    [file join $vivado_root "rtl" "h144_axi_dma_wrapper_bd.v"]]
add_files -norecurse $rtl_sources

# 明确加入 ROM 初始化文件，同时在综合运行目录保留 accelerator/ 相对目录。
set memory_files [glob -nocomplain [file join $memory_dir "*.mem"]]
add_files -norecurse $memory_files
foreach memory_file $memory_files {
    set_property file_type {Memory Initialization Files} [get_files $memory_file]
}
add_files -fileset constrs_1 -norecurse \
    [file join $vivado_root "constraints" "pynq_z2_overlay_leds.xdc"]
update_compile_order -fileset sources_1

create_bd_design $design_name

set DDR [create_bd_intf_port -mode Master -vlnv xilinx.com:interface:ddrx_rtl:1.0 DDR]
set FIXED_IO [create_bd_intf_port -mode Master \
    -vlnv xilinx.com:display_processing_system7:fixedio_rtl:1.0 FIXED_IO]
set status_leds [create_bd_port -dir O -from 3 -to 0 status_leds]

# 从板上正在运行的 PYNQ 3.0.1 base.tcl 提取已验证的 DDR/MIO/时钟参数。
set reference_handle [open $base_reference "r"]
set reference_text [read $reference_handle]
close $reference_handle
set ps_marker [string first "# Create instance: ps7_0" $reference_text]
set ps_property_start [string first "set_property -dict" $reference_text $ps_marker]
set ps_property_end [string first {] $ps7_0} $reference_text $ps_property_start]
if {$ps_marker < 0 || $ps_property_start < 0 || $ps_property_end < 0} {
    error "Could not extract ps7_0 configuration from $base_reference"
}

set ps7_0 [create_bd_cell -type ip -vlnv xilinx.com:ip:processing_system7:5.5 ps7_0]
set ps_property_end [expr {$ps_property_end + [string length {] $ps7_0}] - 1}]
set ps_property_command [string range $reference_text $ps_property_start $ps_property_end]
eval $ps_property_command

# 只增加本 Overlay 所需的 AXI 端口，并把 PL 主时钟固定为 50 MHz。
set_property -dict [list \
    CONFIG.PCW_USE_M_AXI_GP0 {1} \
    CONFIG.PCW_USE_M_AXI_GP1 {0} \
    CONFIG.PCW_USE_S_AXI_GP0 {0} \
    CONFIG.PCW_USE_S_AXI_GP1 {0} \
    CONFIG.PCW_USE_S_AXI_HP0 {1} \
    CONFIG.PCW_USE_S_AXI_HP1 {0} \
    CONFIG.PCW_USE_S_AXI_HP2 {0} \
    CONFIG.PCW_USE_S_AXI_HP3 {0} \
    CONFIG.PCW_USE_S_AXI_ACP {0} \
    CONFIG.PCW_FPGA0_PERIPHERAL_FREQMHZ {50.000000} \
    CONFIG.PCW_CLK0_FREQ {50000000}] $ps7_0

set rst_0 [create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset:5.0 rst_0]
set dma_0 [create_bd_cell -type ip -vlnv xilinx.com:ip:axi_dma:* axi_dma_0]
set_property -dict [list \
    CONFIG.c_include_sg {0} \
    CONFIG.c_sg_include_stscntrl_strm {0} \
    CONFIG.c_sg_length_width {26} \
    CONFIG.c_m_axi_mm2s_data_width {64} \
    CONFIG.c_m_axi_s2mm_data_width {64} \
    CONFIG.c_m_axis_mm2s_tdata_width {8} \
    CONFIG.c_s_axis_s2mm_tdata_width {16} \
    CONFIG.c_include_mm2s_dre {0} \
    CONFIG.c_include_s2mm_dre {0} \
    CONFIG.c_include_mm2s_sf {1} \
    CONFIG.c_include_s2mm_sf {1} \
    CONFIG.c_mm2s_burst_size {16} \
    CONFIG.c_s2mm_burst_size {16}] $dma_0

set ctrl_ic [create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 ctrl_interconnect]
set_property -dict [list CONFIG.NUM_SI {1} CONFIG.NUM_MI {2}] $ctrl_ic
set mem_ic [create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 mem_interconnect]
set_property -dict [list CONFIG.NUM_SI {2} CONFIG.NUM_MI {1}] $mem_ic

set accel_0 [create_bd_cell -type module -reference h144_axi_dma_wrapper_bd h144_accel_0]

connect_bd_intf_net [get_bd_intf_ports DDR] [get_bd_intf_pins ps7_0/DDR]
connect_bd_intf_net [get_bd_intf_ports FIXED_IO] [get_bd_intf_pins ps7_0/FIXED_IO]
connect_bd_intf_net [get_bd_intf_pins ps7_0/M_AXI_GP0] [get_bd_intf_pins ctrl_interconnect/S00_AXI]
connect_bd_intf_net [get_bd_intf_pins ctrl_interconnect/M00_AXI] [get_bd_intf_pins axi_dma_0/S_AXI_LITE]
connect_bd_intf_net [get_bd_intf_pins ctrl_interconnect/M01_AXI] [get_bd_intf_pins h144_accel_0/S_AXI]
connect_bd_intf_net [get_bd_intf_pins axi_dma_0/M_AXI_MM2S] [get_bd_intf_pins mem_interconnect/S00_AXI]
connect_bd_intf_net [get_bd_intf_pins axi_dma_0/M_AXI_S2MM] [get_bd_intf_pins mem_interconnect/S01_AXI]
connect_bd_intf_net [get_bd_intf_pins mem_interconnect/M00_AXI] [get_bd_intf_pins ps7_0/S_AXI_HP0]
connect_bd_intf_net [get_bd_intf_pins axi_dma_0/M_AXIS_MM2S] [get_bd_intf_pins h144_accel_0/S_AXIS]
connect_bd_intf_net [get_bd_intf_pins h144_accel_0/M_AXIS] [get_bd_intf_pins axi_dma_0/S_AXIS_S2MM]
connect_bd_net [get_bd_ports status_leds] [get_bd_pins h144_accel_0/status_leds]

set pl_clk [get_bd_pins ps7_0/FCLK_CLK0]
set peripheral_resetn [get_bd_pins rst_0/peripheral_aresetn]
set interconnect_resetn [get_bd_pins rst_0/interconnect_aresetn]
connect_bd_net $pl_clk \
    [get_bd_pins ps7_0/M_AXI_GP0_ACLK] \
    [get_bd_pins ps7_0/S_AXI_HP0_ACLK] \
    [get_bd_pins rst_0/slowest_sync_clk] \
    [get_bd_pins axi_dma_0/s_axi_lite_aclk] \
    [get_bd_pins axi_dma_0/m_axi_mm2s_aclk] \
    [get_bd_pins axi_dma_0/m_axi_s2mm_aclk] \
    [get_bd_pins ctrl_interconnect/ACLK] \
    [get_bd_pins ctrl_interconnect/S00_ACLK] \
    [get_bd_pins ctrl_interconnect/M00_ACLK] \
    [get_bd_pins ctrl_interconnect/M01_ACLK] \
    [get_bd_pins mem_interconnect/ACLK] \
    [get_bd_pins mem_interconnect/S00_ACLK] \
    [get_bd_pins mem_interconnect/S01_ACLK] \
    [get_bd_pins mem_interconnect/M00_ACLK] \
    [get_bd_pins h144_accel_0/aclk]
connect_bd_net [get_bd_pins ps7_0/FCLK_RESET0_N] [get_bd_pins rst_0/ext_reset_in]
connect_bd_net $interconnect_resetn \
    [get_bd_pins ctrl_interconnect/ARESETN] \
    [get_bd_pins ctrl_interconnect/S00_ARESETN] \
    [get_bd_pins ctrl_interconnect/M00_ARESETN] \
    [get_bd_pins ctrl_interconnect/M01_ARESETN] \
    [get_bd_pins mem_interconnect/ARESETN] \
    [get_bd_pins mem_interconnect/S00_ARESETN] \
    [get_bd_pins mem_interconnect/S01_ARESETN] \
    [get_bd_pins mem_interconnect/M00_ARESETN]
connect_bd_net $peripheral_resetn \
    [get_bd_pins axi_dma_0/axi_resetn] \
    [get_bd_pins h144_accel_0/aresetn]

assign_bd_address
validate_bd_design
save_bd_design
set_property synth_checkpoint_mode None [get_files ${design_name}.bd]
write_bd_tcl -force [file join $export_dir "h144_overlay_design.tcl"]

generate_target all [get_files ${design_name}.bd]
set wrapper_files [make_wrapper -files [get_files ${design_name}.bd] -top]
add_files -norecurse $wrapper_files
set_property top ${design_name}_wrapper [current_fileset]
update_compile_order -fileset sources_1

# 保证 RTL 中 accelerator/*.mem 的相对引用在综合子进程中可解析。
set synth_run_dir [get_property DIRECTORY [get_runs synth_1]]
file mkdir [file join $synth_run_dir "accelerator"]
foreach memory_file $memory_files {
    file copy -force $memory_file [file join $synth_run_dir "accelerator" [file tail $memory_file]]
}

if {[lsearch -exact $argv "--create-only"] >= 0} {
    puts "H144_CREATE_ONLY_VALIDATED"
    close_project
    exit
}
cd $project_dir
launch_runs synth_1 -jobs 8
wait_on_run synth_1
if {[get_property STATUS [get_runs synth_1]] ne {synth_design Complete!}} {
    error "Synthesis failed: [get_property STATUS [get_runs synth_1]]"
}
open_run synth_1
report_utilization -file [file join $report_dir "post_synth_utilization.rpt"]
report_timing_summary -delay_type max -max_paths 30 \
    -file [file join $report_dir "post_synth_timing.rpt"]
close_design

launch_runs impl_1 -to_step write_bitstream -jobs 8
wait_on_run impl_1
if {[get_property STATUS [get_runs impl_1]] ne {write_bitstream Complete!}} {
    error "Implementation failed: [get_property STATUS [get_runs impl_1]]"
}
open_run impl_1
report_utilization -hierarchical -file [file join $report_dir "post_route_utilization_hier.rpt"]
report_utilization -file [file join $report_dir "post_route_utilization.rpt"]
report_timing_summary -delay_type min_max -max_paths 50 -report_unconstrained \
    -file [file join $report_dir "post_route_timing.rpt"]
report_power -file [file join $report_dir "post_route_power_estimate.rpt"]

set wns [get_property SLACK [get_timing_paths -delay_type max -max_paths 1]]
set whs [get_property SLACK [get_timing_paths -delay_type min -max_paths 1]]
set bitstream_source [file join [get_property DIRECTORY [get_runs impl_1]] \
    "${design_name}_wrapper.bit"]
file copy -force $bitstream_source [file join $export_dir "h144_dma_overlay.bit"]

set hwh_candidates [glob -nocomplain -types f \
    [file join $project_dir "${project_name}.gen" "sources_1" "bd" $design_name \
        "hw_handoff" "*.hwh"]]
if {[llength $hwh_candidates] != 1} {
    error "Expected exactly one HWH file, found: $hwh_candidates"
}
file copy -force [lindex $hwh_candidates 0] [file join $export_dir "h144_dma_overlay.hwh"]

set summary [open [file join $report_dir "overlay_build_summary.txt"] "w"]
puts $summary "part=xc7z020clg400-1"
puts $summary "board=PYNQ-Z2"
puts $summary "vivado=[version -short]"
puts $summary "pl_clock_mhz=50.0"
puts $summary "post_route_setup_wns_ns=$wns"
puts $summary "post_route_hold_whs_ns=$whs"
puts $summary "bitstream=[file join $export_dir h144_dma_overlay.bit]"
puts $summary "hwh=[file join $export_dir h144_dma_overlay.hwh]"
close $summary

puts "DMA_OVERLAY_BUILD_COMPLETE"
puts "BITSTREAM=[file join $export_dir h144_dma_overlay.bit]"
puts "HWH=[file join $export_dir h144_dma_overlay.hwh]"
puts "POST_ROUTE_SETUP_WNS_NS=$wns"
puts "POST_ROUTE_HOLD_WHS_NS=$whs"
exit

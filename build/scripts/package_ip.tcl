# 为科研仓库生成可在 IP Catalog 中复用的自定义 AXI 加速器 IP。
set root [file normalize [file join [file dirname [info script]] .. ..]]
set output [file join $root src pl ip h144_accel]
set project_dir [file join $root build ip_packager]
if {[file exists [file join $output component.xml]]} {
    error "Packaged IP already exists; preserve or rename it before packaging again."
}
create_project -force h144_ip_packager $project_dir -part xc7z020clg400-1
set_property target_language Verilog [current_project]
add_files -norecurse [glob [file join $root src pl rtl *.sv]]
add_files -norecurse [glob [file join $root src pl rtl *.v]]
set_property top h144_axi_dma_wrapper_bd [current_fileset]
update_compile_order -fileset sources_1
ipx::package_project -root_dir $output -vendor bmi.research -library bci -taxonomy /UserIP -import_files -set_current true
set core [ipx::current_core]
set_property name h144_accel $core
set_property display_name {B2Tnet Activity-Aware GRU Decoder} $core
set_property description {B2Tnet W8A8/H16/ACC32 decoder, fixed activity gate policy, read-only mode2, version 0x00020001.} $core
set_property version 2.0 $core
foreach name {S_AXI S_AXIS M_AXIS} {
    if {[llength [ipx::get_bus_interfaces $name -of_objects $core]] > 0} {
        ipx::associate_bus_interfaces -busif $name -clock aclk $core
    }
}
if {[llength [ipx::get_memory_maps S_AXI -of_objects $core]] == 0} {
    set map [ipx::add_memory_map S_AXI $core]
    set block [ipx::add_address_block registers $map]
    set_property range 4096 $block
    set_property width 32 $block
    set_property usage register $block
    set_property slave_memory_map_ref S_AXI [ipx::get_bus_interfaces S_AXI -of_objects $core]
}
file mkdir [file join $output accelerator]
foreach source [glob [file join $root data golden_ref rtl_vectors accelerator *.mem]] {
    file copy -force $source [file join $output accelerator [file tail $source]]
}
set synth [ipx::get_file_groups xilinx_anylanguagesynthesis -of_objects $core]
set sim [ipx::get_file_groups xilinx_anylanguagebehavioralsimulation -of_objects $core]
foreach group [list $synth $sim] {
    if {[llength $group]} {
        foreach source [glob [file join $output accelerator *.mem]] {
            set file [ipx::add_file accelerator/[file tail $source] $group]
            set_property type memoryInitializationFile $file
        }
    }
}
ipx::create_xgui_files $core
ipx::update_checksums $core
ipx::check_integrity -quiet $core
ipx::save_core $core
puts "H144_IP_PACKAGE_READY=$output"
close_project
exit

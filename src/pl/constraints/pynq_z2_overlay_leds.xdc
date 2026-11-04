# PYNQ-Z2 四个用户 LED。LED0=busy，LED1=done，LED2=error，LED3=core ready。
set_property -dict {PACKAGE_PIN R14 IOSTANDARD LVCMOS33} [get_ports {status_leds[0]}]
set_property -dict {PACKAGE_PIN P14 IOSTANDARD LVCMOS33} [get_ports {status_leds[1]}]
set_property -dict {PACKAGE_PIN N16 IOSTANDARD LVCMOS33} [get_ports {status_leds[2]}]
set_property -dict {PACKAGE_PIN M14 IOSTANDARD LVCMOS33} [get_ports {status_leds[3]}]


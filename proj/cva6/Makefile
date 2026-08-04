.PHONY: help init sim lint gateware image ram-image stage2-loader upload-ram upload-ram-model upload-model monitor deploy-remote remote-boot remote-boot-model deploy-model-remote remote-model status

ROOT := $(abspath ../..)
OUTPUT := $(CURDIR)/build/sdk
HOST_TOOLS := $(CURDIR)/build/host-tools
BUILDROOT := $(CURDIR)/sdk/buildroot
BUILDROOT_EXTERNAL := $(CURDIR)/sdk/br2-ext-tree
BUILDROOT_DEFCONFIG := $(CURDIR)/config/buildroot64_defconfig
BC_VERSION := 1.08.2
BC_ARCHIVE := $(CURDIR)/build/downloads/bc-$(BC_VERSION).tar.gz
UART_BAUD ?= 3000000
MODEL_UPLOAD_RATE_KIB ?= 10
SERIAL ?= /dev/ttyUSB0
REMOTE_HOST ?= thinkmax
REMOTE_CVA6_DIR ?= ~/fpga/fpga-cpu/build/cva6
REMOTE_SERIAL ?= /dev/ttyUSB0
MODEL ?= $(abspath $(ROOT)/../inference/models/LFM2.5-350M-Q4_0.gguf)

help:
	@echo "make init      Initialize all pinned CVA6 dependencies"
	@echo "make sim       Run the hardware terminal regression test"
	@echo "make lint      Elaborate the HDMI terminal RTL with Vivado"
	@echo "make gateware Build the CVA6 Genesys 2 HDMI bitstream"
	@echo "make image    Build the 64-bit Linux userspace and SD-card image"
	@echo "make ram-image Build Linux with the root filesystem embedded in DRAM firmware"
	@echo "make stage2-loader Build the flow-controlled fast RAM loader"
	@echo "make upload-ram Upload the existing DRAM Linux firmware over UART"
	@echo "make upload-ram-model Upload Linux and LFM weights through the fast stage-2 loader"
	@echo "make upload-model Upload LFM weights into the running RAM filesystem"
	@echo "make monitor  Open the board's serial console"
	@echo "make deploy-remote Copy board-ready artifacts to thinkmax"
	@echo "make remote-boot Deploy, program the remote FPGA, upload Linux, and attach"
	@echo "make remote-boot-model Boot remote Linux with LFM weights staged in reserved DDR"
	@echo "make remote-model Upload the LFM weights into a running remote RAM filesystem"
	@echo "make status   Show upstream revisions and submodule state"

init:
	git -C $(ROOT) submodule update --init --recursive -- proj/cva6/hardware proj/cva6/sdk
	./tools/apply_patches.sh

sim:
	mkdir -p build/sim
	iverilog -g2012 -s terminal_buffer_tb -o build/sim/terminal_buffer_tb rtl/terminal_buffer.sv sim/terminal_buffer_tb.sv
	cd $(CURDIR) && vvp build/sim/terminal_buffer_tb
	iverilog -g2012 -s uart_rx_tb -o build/sim/uart_rx_tb rtl/uart_rx.sv sim/uart_rx_tb.sv
	cd $(CURDIR) && vvp build/sim/uart_rx_tb
	iverilog -g2012 -s software_video_plane_tb -o build/sim/software_video_plane_tb rtl/software_video_plane.sv sim/software_video_plane_tb.sv
	cd $(CURDIR) && vvp build/sim/software_video_plane_tb
	verilator --binary --timing -Wno-fatal -Wno-WIDTHEXPAND -Wno-WIDTHTRUNC -Wno-UNUSEDSIGNAL -Wno-UNUSEDPARAM -Wno-DECLFILENAME -Wno-TIMESCALEMOD -Ihardware/vendor/pulp-platform/axi/include --top-module lfm_matvec_accelerator_tb --Mdir build/sim/lfm/obj hardware/vendor/pulp-platform/axi/src/axi_pkg.sv hardware/vendor/pulp-platform/axi/src/axi_intf.sv rtl/lfm_axi_byte_reader.sv rtl/lfm_matvec_accelerator.sv sim/lfm_matvec_accelerator_tb.sv
	build/sim/lfm/obj/Vlfm_matvec_accelerator_tb

lint:
	cd hardware/corev_apu/fpga && CVA6_HDMI_ROOT=$(CURDIR) CVA6_UART_BAUD=$(UART_BAUD) vivado -nojournal -nolog -mode batch -source $(CURDIR)/scripts/lint_hdmi.tcl

gateware:
	UART_BAUD=$(UART_BAUD) ./tools/build_gateware.sh

monitor:
	SERIAL=$(SERIAL) UART_BAUD=$(UART_BAUD) ./tools/monitor.sh

$(BC_ARCHIVE):
	mkdir -p $(dir $@)
	wget -O $@ https://ftp.gnu.org/gnu/bc/bc-$(BC_VERSION).tar.gz

$(HOST_TOOLS)/bin/bc: $(BC_ARCHIVE)
	mkdir -p $(HOST_TOOLS)/src
	tar -xf $< -C $(HOST_TOOLS)/src
	cd $(HOST_TOOLS)/src/bc-$(BC_VERSION) && ./configure --prefix=$(HOST_TOOLS)
	$(MAKE) -C $(HOST_TOOLS)/src/bc-$(BC_VERSION)
	$(MAKE) -C $(HOST_TOOLS)/src/bc-$(BC_VERSION) install

image: $(HOST_TOOLS)/bin/bc
	mkdir -p $(OUTPUT)
	PATH="$(HOST_TOOLS)/bin:$(PATH)" $(MAKE) -C $(BUILDROOT) BR2_EXTERNAL="$(BUILDROOT_EXTERNAL)" BR2_DEFCONFIG="$(BUILDROOT_DEFCONFIG)" HOSTCXX=$(CURDIR)/tools/host-gxx defconfig
	PATH="$(HOST_TOOLS)/bin:$(PATH)" $(MAKE) -j$$(nproc) -C $(BUILDROOT) BR2_EXTERNAL="$(BUILDROOT_EXTERNAL)" BINARIES_DIR=$(OUTPUT) HOSTCXX=$(CURDIR)/tools/host-gxx

ram-image: image
	./tools/build_ram_image.sh

stage2-loader:
	./tools/build_stage2_loader.sh

upload-ram: stage2-loader
	test -f $(OUTPUT)/ram-linux.bin || { echo "Run 'make ram-image' first." >&2; exit 1; }
	SERIAL=$(SERIAL) UART_BAUD=$(UART_BAUD) ./tools/ram_upload.py --stage2 $(OUTPUT)/stage2-loader.bin $(OUTPUT)/ram-linux.bin

upload-ram-model: stage2-loader
	test -f $(OUTPUT)/ram-linux.bin || { echo "Run 'make ram-image' first." >&2; exit 1; }
	test -f $(MODEL) || { echo "Model not found: $(MODEL)" >&2; exit 1; }
	SERIAL=$(SERIAL) UART_BAUD=$(UART_BAUD) ./tools/ram_upload.py --stage2 $(OUTPUT)/stage2-loader.bin --model $(MODEL) $(OUTPUT)/ram-linux.bin

upload-model:
	test -f $(MODEL) || { echo "Model not found: $(MODEL)" >&2; exit 1; }
	./tools/model_upload.py --serial $(SERIAL) --baud $(UART_BAUD) --rate-kib $(MODEL_UPLOAD_RATE_KIB) $(MODEL)

deploy-remote: stage2-loader
	test -f build/gateware/cva6_genesys2_hdmi.bit || { echo "Run 'make gateware' first." >&2; exit 1; }
	test -f $(OUTPUT)/ram-linux.bin || { echo "Run 'make ram-image' first." >&2; exit 1; }
	ssh $(REMOTE_HOST) 'mkdir -p $(REMOTE_CVA6_DIR)'
	rsync -az build/gateware/cva6_genesys2_hdmi.bit $(OUTPUT)/ram-linux.bin $(OUTPUT)/stage2-loader.bin tools/ram_upload.py $(REMOTE_HOST):$(REMOTE_CVA6_DIR)/

remote-boot: deploy-remote
	ssh -tt $(REMOTE_HOST) 'cd $(REMOTE_CVA6_DIR) && openFPGALoader -b genesys2 --freq 1000000 cva6_genesys2_hdmi.bit && python3 ram_upload.py --serial $(REMOTE_SERIAL) --baud $(UART_BAUD) --stage2 stage2-loader.bin ram-linux.bin'

remote-boot-model: deploy-remote deploy-model-remote
	ssh -tt $(REMOTE_HOST) 'cd $(REMOTE_CVA6_DIR) && openFPGALoader -b genesys2 --freq 1000000 cva6_genesys2_hdmi.bit && python3 ram_upload.py --serial $(REMOTE_SERIAL) --baud $(UART_BAUD) --stage2 stage2-loader.bin --model LFM2.5-350M-Q4_0.gguf ram-linux.bin'

deploy-model-remote:
	test -f $(MODEL) || { echo "Model not found: $(MODEL)" >&2; exit 1; }
	ssh $(REMOTE_HOST) 'mkdir -p $(REMOTE_CVA6_DIR)'
	rsync -az $(MODEL) $(REMOTE_HOST):$(REMOTE_CVA6_DIR)/LFM2.5-350M-Q4_0.gguf
	rsync -az tools/model_upload.py $(REMOTE_HOST):$(REMOTE_CVA6_DIR)/

remote-model: deploy-model-remote
	ssh -tt $(REMOTE_HOST) 'cd $(REMOTE_CVA6_DIR) && python3 model_upload.py --serial $(REMOTE_SERIAL) --baud $(UART_BAUD) --rate-kib $(MODEL_UPLOAD_RATE_KIB) LFM2.5-350M-Q4_0.gguf'

status:
	git -C hardware status --short --branch
	git -C hardware submodule status --recursive
	git -C sdk status --short --branch
	git -C sdk submodule status --recursive

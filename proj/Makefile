.PHONY: all setup build gateware diagnostic software framebuffer-demo flash load upload sync clean

all: gateware software

setup:
	./tools/setup.sh

build: gateware

gateware:
	./tools/build.sh

diagnostic:
	PATH="$(CURDIR)/.venv/bin:$(PATH)" .venv/bin/python hdl/diagnostics/hdmi_colorbars.py
	cp build/genesys2_hdmi_diag/gateware/genesys2_hdmi_diag.bit build/known_good_hdmi.bit

software framebuffer-demo:
	$(MAKE) -C sw/framebuffer_demo

flash load:
	./tools/load.sh

upload:
	./tools/upload.sh

sync:
	./tools/sync.sh

clean:
	$(RM) -r build sw/framebuffer_demo/build

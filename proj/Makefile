.PHONY: all setup build gateware diagnostic software framebuffer-demo flash load upload sync clean

# Allow `make upload <program>` while retaining `make upload` as a shorthand
# for framebuffer_demo. The second command-line goal is a selector, not a
# recipe of its own.
ifeq ($(firstword $(MAKECMDGOALS)),upload)
ifneq ($(words $(MAKECMDGOALS)),1)
ifneq ($(words $(MAKECMDGOALS)),2)
$(error usage: make upload [program])
endif
UPLOAD_GOAL := $(word 2,$(MAKECMDGOALS))
.PHONY: $(UPLOAD_GOAL)
$(UPLOAD_GOAL):
	@:
endif
endif

PROGRAM ?= $(or $(UPLOAD_GOAL),framebuffer_demo)

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
	PROGRAM="$(PROGRAM)" ./tools/upload.sh

sync:
	./tools/sync.sh

clean:
	$(RM) -r build sw/framebuffer_demo/build

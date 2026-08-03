UART_BAUD ?= 3000000
UART_FIFO_DEPTH ?= 512

.PHONY: all setup build gateware diagnostic software framebuffer-demo text-demo micropython-repl bad-apple flash load monitor upload sync clean

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
	UART_BAUD="$(UART_BAUD)" UART_FIFO_DEPTH="$(UART_FIFO_DEPTH)" ./tools/build.sh

diagnostic:
	PATH="$(CURDIR)/.venv/bin:$(PATH)" .venv/bin/python hdl/diagnostics/hdmi_colorbars.py
	cp build/genesys2_hdmi_diag/gateware/genesys2_hdmi_diag.bit build/known_good_hdmi.bit

software framebuffer-demo:
	$(MAKE) -C sw/framebuffer_demo

text-demo:
	$(MAKE) -C sw/text_demo

micropython-repl:
	$(MAKE) -C sw/micropython_repl

bad-apple:
	$(MAKE) -C sw/bad_apple

flash load:
	./tools/load.sh

monitor:
	./tools/serial.sh

upload:
	PROGRAM="$(PROGRAM)" ./tools/upload.sh

sync:
	./tools/sync.sh

clean:
	$(RM) -r build sw/framebuffer_demo/build sw/text_demo/build sw/micropython_repl/build sw/bad_apple/build

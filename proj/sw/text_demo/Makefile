FONT_SIZE ?= 8x16
FONT_SCALE ?= 1
DEMO_LINES_PER_SECOND ?= 60
USE_LIBC := 1

include ../text_console/text_console.mk
include ../runtime/app.mk

CFLAGS += -DDEMO_LINES_PER_SECOND=$(DEMO_LINES_PER_SECOND)

$(OUTPUT_DIR)/main.o: $(TEXT_CONSOLE_CONFIG)

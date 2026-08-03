FONT_SIZE ?= 8x16
FONT_SCALE ?= 1
DEMO_LINES_PER_SECOND ?= 60
USE_LIBC := 1
SUPPORTED_FONT_SIZES := 5x8 6x12 8x16 12x24 16x32 32x64

ifeq ($(filter $(FONT_SIZE),$(SUPPORTED_FONT_SIZES)),)
$(error unsupported FONT_SIZE '$(FONT_SIZE)'; choose $(SUPPORTED_FONT_SIZES))
endif

FONT_WIDTH := $(word 1,$(subst x, ,$(FONT_SIZE)))
FONT_HEIGHT := $(word 2,$(subst x, ,$(FONT_SIZE)))
FONT_BDF := assets/spleen-$(FONT_SIZE)-ascii.bdf
FONT_BIN = $(OUTPUT_DIR)/spleen_font.bin
FONT_CONFIG = $(OUTPUT_DIR)/.font-scale
EXTRA_OBJECTS = $(OUTPUT_DIR)/spleen_font.o

include ../runtime/app.mk

CFLAGS += \
	-DFONT_WIDTH=$(FONT_WIDTH) \
	-DFONT_HEIGHT=$(FONT_HEIGHT) \
	-DFONT_SCALE=$(FONT_SCALE) \
	-DDEMO_LINES_PER_SECOND=$(DEMO_LINES_PER_SECOND)

$(OUTPUT_DIR)/main.o: $(FONT_CONFIG)

$(FONT_CONFIG): FORCE | $(OUTPUT_DIR)
	@if ! test -f $@ || ! test "$$(cat $@)" = "$(FONT_SIZE):$(FONT_SCALE):$(DEMO_LINES_PER_SECOND)"; then \
		printf '%s\n' '$(FONT_SIZE):$(FONT_SCALE):$(DEMO_LINES_PER_SECOND)' > $@; \
	fi

$(FONT_BIN): $(FONT_BDF) bdf_to_bin.py $(FONT_CONFIG) | $(OUTPUT_DIR)
	python3 bdf_to_bin.py --width $(FONT_WIDTH) --height $(FONT_HEIGHT) $< $@

$(OUTPUT_DIR)/spleen_font.o: $(FONT_BIN)
	cd $(OUTPUT_DIR) && $(OBJCOPY) \
		-I binary -O elf32-littleriscv -B riscv \
		--rename-section .data=.rodata,alloc,load,readonly,data,contents \
		spleen_font.bin spleen_font.o

FORCE:

.PHONY: FORCE

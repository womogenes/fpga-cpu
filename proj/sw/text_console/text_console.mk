TEXT_CONSOLE_DIR := $(patsubst %/,%,$(dir $(lastword $(MAKEFILE_LIST))))
OUTPUT_DIR ?= build
FONT_SIZE ?= 8x16
FONT_SCALE ?= 1
TEXT_CONSOLE_MAX_FPS ?= 60
SUPPORTED_FONT_SIZES := 5x8 6x12 8x16 12x24 16x32 32x64

ifeq ($(filter $(FONT_SIZE),$(SUPPORTED_FONT_SIZES)),)
$(error unsupported FONT_SIZE '$(FONT_SIZE)'; choose $(SUPPORTED_FONT_SIZES))
endif

FONT_WIDTH := $(word 1,$(subst x, ,$(FONT_SIZE)))
FONT_HEIGHT := $(word 2,$(subst x, ,$(FONT_SIZE)))
TEXT_CONSOLE_FONT_BDF := $(TEXT_CONSOLE_DIR)/assets/spleen-$(FONT_SIZE)-ascii.bdf
TEXT_CONSOLE_FONT_BIN := $(OUTPUT_DIR)/spleen_font.bin
TEXT_CONSOLE_CONFIG := $(OUTPUT_DIR)/.text-console-config

EXTRA_CFLAGS += \
	-I$(TEXT_CONSOLE_DIR)/include \
	-DFONT_WIDTH=$(FONT_WIDTH) \
	-DFONT_HEIGHT=$(FONT_HEIGHT) \
	-DFONT_SCALE=$(FONT_SCALE) \
	-DTEXT_CONSOLE_MAX_FPS=$(TEXT_CONSOLE_MAX_FPS)

EXTRA_OBJECTS += \
	$(OUTPUT_DIR)/text_console.o \
	$(OUTPUT_DIR)/spleen_font.o

$(OUTPUT_DIR)/text_console.o: $(TEXT_CONSOLE_DIR)/text_console.c $(TEXT_CONSOLE_CONFIG) | $(OUTPUT_DIR)
	$(CC) $(CFLAGS) -MD -MP -c $< -o $@

$(TEXT_CONSOLE_CONFIG): FORCE | $(OUTPUT_DIR)
	@if ! test -f $@ || ! test "$$(cat $@)" = "$(FONT_SIZE):$(FONT_SCALE):$(TEXT_CONSOLE_MAX_FPS)"; then \
		printf '%s\n' '$(FONT_SIZE):$(FONT_SCALE):$(TEXT_CONSOLE_MAX_FPS)' > $@; \
	fi

$(TEXT_CONSOLE_FONT_BIN): $(TEXT_CONSOLE_FONT_BDF) $(TEXT_CONSOLE_DIR)/bdf_to_bin.py $(TEXT_CONSOLE_CONFIG) | $(OUTPUT_DIR)
	python3 $(TEXT_CONSOLE_DIR)/bdf_to_bin.py \
		--width $(FONT_WIDTH) --height $(FONT_HEIGHT) $< $@

$(OUTPUT_DIR)/spleen_font.o: $(TEXT_CONSOLE_FONT_BIN)
	cd $(OUTPUT_DIR) && $(OBJCOPY) \
		-I binary -O elf32-littleriscv -B riscv \
		--rename-section .data=.rodata,alloc,load,readonly,data,contents \
		spleen_font.bin spleen_font.o

FORCE:

.PHONY: FORCE

OPTIMIZATION := -O3
PATTERN ?= spacefiller2
PATTERN_RLE := assets/$(PATTERN).rle
PATTERN_CONFIG = $(OUTPUT_DIR)/.pattern-name
PATTERN_BIN = $(OUTPUT_DIR)/initial_pattern.bin
EXTRA_OBJECTS = $(OUTPUT_DIR)/initial_pattern.o

include ../runtime/app.mk

$(PATTERN_CONFIG): FORCE | $(OUTPUT_DIR)
	@if ! test -f $@ || ! test "$$(cat $@)" = "$(PATTERN)"; then \
		printf '%s\n' '$(PATTERN)' > $@; \
	fi

$(PATTERN_BIN): $(PATTERN_RLE) rle_to_bin.py $(PATTERN_CONFIG) | $(OUTPUT_DIR)
	python3 rle_to_bin.py $< $@

$(OUTPUT_DIR)/initial_pattern.o: $(PATTERN_BIN)
	cd $(OUTPUT_DIR) && $(OBJCOPY) \
		-I binary -O elf32-littleriscv -B riscv \
		--rename-section .data=.rodata,alloc,load,readonly,data,contents \
		initial_pattern.bin initial_pattern.o

FORCE:

.PHONY: FORCE

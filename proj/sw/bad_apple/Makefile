OPTIMIZATION := -O3
VIDEO ?= assets/badapple.mp4
VIDEO_WIDTH ?= 160
VIDEO_HEIGHT ?= 90
VIDEO_FPS ?= 30
VIDEO_THRESHOLD ?= 128
VIDEO_CONFIG = $(OUTPUT_DIR)/.video-config
VIDEO_BIN = $(OUTPUT_DIR)/video.bin
UZLIB_SOURCES := tinflate.c header.c adler32.c crc32.c
UZLIB_OBJECTS = $(addprefix $(OUTPUT_DIR)/uzlib/,$(UZLIB_SOURCES:.c=.o))
EXTRA_OBJECTS = $(OUTPUT_DIR)/video.o $(UZLIB_OBJECTS)
EXTRA_CFLAGS := -Ivendor/uzlib

include ../runtime/app.mk

$(VIDEO_CONFIG): FORCE | $(OUTPUT_DIR)
	@new='$(abspath $(VIDEO)) $(VIDEO_WIDTH)x$(VIDEO_HEIGHT) $(VIDEO_FPS)fps threshold=$(VIDEO_THRESHOLD)'; \
	if ! test -f $@ || ! test "$$(cat $@)" = "$$new"; then \
		printf '%s\n' "$$new" > $@; \
	fi

$(VIDEO_BIN): $(VIDEO) video_to_bin.py $(VIDEO_CONFIG) | $(OUTPUT_DIR)
	python3 video_to_bin.py $(VIDEO) $@ \
		--width $(VIDEO_WIDTH) --height $(VIDEO_HEIGHT) \
		--fps $(VIDEO_FPS) --threshold $(VIDEO_THRESHOLD)

$(OUTPUT_DIR)/video.o: $(VIDEO_BIN)
	cd $(OUTPUT_DIR) && $(OBJCOPY) \
		-I binary -O elf32-littleriscv -B riscv \
		--rename-section .data=.rodata.video,alloc,load,readonly,data,contents \
		video.bin video.o

$(OUTPUT_DIR)/uzlib/%.o: vendor/uzlib/%.c
	mkdir -p $(@D)
	$(CC) $(CFLAGS) -MD -MP -c $< -o $@

$(OUTPUT_DIR)/main.o: $(VIDEO_CONFIG) vendor/uzlib/uzlib.h

FORCE:

.PHONY: FORCE

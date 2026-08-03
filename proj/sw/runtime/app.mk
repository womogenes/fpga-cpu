.DEFAULT_GOAL := all

BUILD_DIR ?= ../../build/fpga_soc
OUTPUT_DIR ?= build
RUNTIME_DIR := ../runtime
TARGET ?= $(notdir $(CURDIR))
EXTRA_OBJECTS ?=
EXTRA_CFLAGS ?=
EXTRA_LDFLAGS ?=
USE_LIBC ?= 0

CROSS_COMPILE ?= riscv64-elf-
CC := $(CROSS_COMPILE)gcc
OBJCOPY := $(CROSS_COMPILE)objcopy
OPTIMIZATION ?= -Os
ARCH_FLAGS := -march=rv32i2p0_m -mabi=ilp32

CFLAGS := \
	-std=gnu99 $(OPTIMIZATION) -g3 \
	$(ARCH_FLAGS) \
	-D__vexriscv__ \
	-ffreestanding -fno-builtin -fno-stack-protector \
	-ffunction-sections -fdata-sections \
	-Wall -Wextra -Wstrict-prototypes \
	-I$(BUILD_DIR)/software/include \
	-I$(RUNTIME_DIR)/include \
	$(EXTRA_CFLAGS)

LDFLAGS := \
	$(ARCH_FLAGS) \
	-nostdlib -nodefaultlibs -no-pie \
	-Wl,--no-dynamic-linker \
	-Wl,--build-id=none \
	-Wl,--gc-sections \
	-L$(BUILD_DIR)/software/include \
	$(EXTRA_LDFLAGS)

ELF := $(OUTPUT_DIR)/$(TARGET).elf
BIN := $(OUTPUT_DIR)/$(TARGET).bin
MAP := $(OUTPUT_DIR)/$(TARGET).elf.map
OBJECTS := \
	$(OUTPUT_DIR)/crt0.o \
	$(OUTPUT_DIR)/runtime.o \
	$(OUTPUT_DIR)/main.o \
	$(EXTRA_OBJECTS)

ifeq ($(USE_LIBC),1)
OBJECTS += $(OUTPUT_DIR)/newlib_syscalls.o
LIBS := -Wl,--start-group -lc_nano -lgcc -Wl,--end-group
endif

all: $(BIN)

$(OUTPUT_DIR):
	mkdir -p $@

$(BIN): $(ELF)
	$(OBJCOPY) -O binary $< $@
	chmod -x $@

$(ELF): $(OBJECTS) $(RUNTIME_DIR)/linker.ld
	$(CC) $(LDFLAGS) -T $(RUNTIME_DIR)/linker.ld -N -o $@ \
		$(OBJECTS) \
		$(LIBS) \
		-Wl,-Map,$(MAP)
	chmod -x $@

$(OUTPUT_DIR)/crt0.o: $(RUNTIME_DIR)/crt0.S | $(OUTPUT_DIR)
	$(CC) $(CFLAGS) -MD -MP -c $< -o $@

$(OUTPUT_DIR)/runtime.o: $(RUNTIME_DIR)/runtime.c | $(OUTPUT_DIR)
	$(CC) $(CFLAGS) -MD -MP -c $< -o $@

$(OUTPUT_DIR)/newlib_syscalls.o: $(RUNTIME_DIR)/newlib_syscalls.c | $(OUTPUT_DIR)
	$(CC) $(CFLAGS) -MD -MP -c $< -o $@

$(OUTPUT_DIR)/main.o: main.c | $(OUTPUT_DIR)
	$(CC) $(CFLAGS) -MD -MP -c $< -o $@

-include $(OBJECTS:.o=.d)

clean:
	$(RM) -r $(OUTPUT_DIR)

.PHONY: all clean

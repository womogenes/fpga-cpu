BUILD_DIR ?= ../../build/fpga_soc
OUTPUT_DIR ?= build
RUNTIME_DIR := ../runtime
TARGET ?= $(notdir $(CURDIR))

CROSS_COMPILE ?= riscv64-elf-
CC := $(CROSS_COMPILE)gcc
OBJCOPY := $(CROSS_COMPILE)objcopy
OPTIMIZATION ?= -Os

CFLAGS := \
	-std=gnu99 $(OPTIMIZATION) -g3 \
	-march=rv32i2p0_m -mabi=ilp32 \
	-D__vexriscv__ \
	-ffreestanding -fno-builtin -fno-stack-protector \
	-ffunction-sections -fdata-sections \
	-Wall -Wextra -Wstrict-prototypes \
	-I$(BUILD_DIR)/software/include \
	-I$(RUNTIME_DIR)/include

LDFLAGS := \
	-nostdlib -nodefaultlibs -no-pie \
	-Wl,--no-dynamic-linker \
	-Wl,--build-id=none \
	-Wl,--gc-sections \
	-L$(BUILD_DIR)/software/include

ELF := $(OUTPUT_DIR)/$(TARGET).elf
BIN := $(OUTPUT_DIR)/$(TARGET).bin
MAP := $(OUTPUT_DIR)/$(TARGET).elf.map
OBJECTS := \
	$(OUTPUT_DIR)/crt0.o \
	$(OUTPUT_DIR)/runtime.o \
	$(OUTPUT_DIR)/main.o

all: $(BIN)

$(OUTPUT_DIR):
	mkdir -p $@

$(BIN): $(ELF)
	$(OBJCOPY) -O binary $< $@
	chmod -x $@

$(ELF): $(OBJECTS) $(RUNTIME_DIR)/linker.ld
	$(CC) $(LDFLAGS) -T $(RUNTIME_DIR)/linker.ld -N -o $@ \
		$(OBJECTS) \
		-Wl,-Map,$(MAP)
	chmod -x $@

$(OUTPUT_DIR)/crt0.o: $(RUNTIME_DIR)/crt0.S | $(OUTPUT_DIR)
	$(CC) $(CFLAGS) -MD -MP -c $< -o $@

$(OUTPUT_DIR)/runtime.o: $(RUNTIME_DIR)/runtime.c | $(OUTPUT_DIR)
	$(CC) $(CFLAGS) -MD -MP -c $< -o $@

$(OUTPUT_DIR)/main.o: main.c | $(OUTPUT_DIR)
	$(CC) $(CFLAGS) -MD -MP -c $< -o $@

-include $(OBJECTS:.o=.d)

clean:
	$(RM) -r $(OUTPUT_DIR)

.PHONY: all clean

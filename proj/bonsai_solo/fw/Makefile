CC := riscv64-elf-gcc
OBJCOPY := riscv64-elf-objcopy
CFLAGS := -march=rv32im -mabi=ilp32 -O2 -ffreestanding -fno-builtin \
	-nostdlib -nostartfiles -Wall -Wextra -fno-pic -mcmodel=medlow
BUILD := build

all: $(BUILD)/boot.hex $(BUILD)/bonsai.bin $(BUILD)/sim_boot.hex $(BUILD)/monitor_ram.bin

$(BUILD):
	mkdir -p $@

$(BUILD)/monitor.elf: start.S monitor.c support.c hw.h monitor.ld | $(BUILD)
	$(CC) $(CFLAGS) -Os -T monitor.ld start.S monitor.c support.c -lgcc -o $@

$(BUILD)/bonsai.elf: bonsai.c support.c hw.h bonsai_math.h app.ld | $(BUILD)
	$(CC) $(CFLAGS) -T app.ld bonsai.c support.c -lgcc -o $@

$(BUILD)/%.bin: $(BUILD)/%.elf
	$(OBJCOPY) -O binary $< $@

# 64 KiB BRAM image, one 32-bit word per line, monitor at address 0.
$(BUILD)/boot.hex: $(BUILD)/monitor.bin
	python3 -c "import sys,struct; d=open('$<','rb').read(); d+=bytes(-len(d)%4); w=struct.unpack('<%dI'%(len(d)//4),d); open('$@','w').write(''.join('%08x\n'%x for x in w+(0,)*(16384-len(w))))"

clean:
	rm -rf $(BUILD)

# Simulation image: monitor plus the Bonsai app preloaded at 0x1000.
$(BUILD)/sim_boot.hex: $(BUILD)/monitor.bin $(BUILD)/bonsai.bin
	python3 -c "import struct; m=open('$(BUILD)/monitor.bin','rb').read(); a=open('$(BUILD)/bonsai.bin','rb').read(); d=bytearray(65536); d[0:len(m)]=m; d[0x1000:0x1000+len(a)]=a; w=struct.unpack('<16384I',bytes(d)); open('$@','w').write(''.join('%08x\n'%x for x in w))"

# Relocated monitor for 0xc000, uploaded and started by the ROM monitor.
$(BUILD)/monitor_ram.elf: start.S monitor.c support.c hw.h monitor_ram.ld | $(BUILD)
	$(CC) $(CFLAGS) -Os -T monitor_ram.ld start.S monitor.c support.c -lgcc -o $@

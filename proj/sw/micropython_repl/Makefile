.DEFAULT_GOAL := all

BUILD_DIR ?= ../../build/fpga_soc
BUILD ?= build
OUTPUT_DIR := $(BUILD)
CROSS_COMPILE ?= riscv64-elf-

export MICROPY_GIT_TAG := v1.28.0
export MICROPY_GIT_HASH := e0e9fbb17e

include vendor/micropython/py/mkenv.mk

QSTR_DEFS := qstrdefsport.h
MICROPY_ROM_TEXT_COMPRESSION := 1
include $(TOP)/py/py.mk
include ../text_console/text_console.mk

ARCH_FLAGS := -march=rv32i2p0_m -mabi=ilp32
INC += -I.
INC += -I$(TOP)
INC += -I$(BUILD)
INC += -I$(BUILD_DIR)/software/include
INC += -I../runtime/include

CFLAGS += \
	$(ARCH_FLAGS) $(INC) $(EXTRA_CFLAGS) \
	-std=gnu99 -Os -g3 -DNDEBUG \
	-D__vexriscv__ \
	-DMICROPY_HEAP_SIZE=8388608 \
	-DMICROPY_FLOAT_IMPL=MICROPY_FLOAT_IMPL_FLOAT \
	-fsingle-precision-constant \
	-ffreestanding -fno-builtin -fno-stack-protector \
	-ffunction-sections -fdata-sections \
	-Wall -Wextra -Wno-unused-parameter

CFLAGS += \
	-DLFS2_NO_MALLOC -DLFS2_NO_DEBUG -DLFS2_NO_WARN \
	-DLFS2_NO_ERROR -DLFS2_NO_ASSERT \
	-DLFS2_DEFINES=extmod/littlefs-include/lfs2_defines.h \
	-DFFCONF_H=\"lib/oofatfs/ffconf.h\"

CSUPEROPT := -O3
AFLAGS := $(ARCH_FLAGS)
LDFLAGS += \
	$(ARCH_FLAGS) -nostdlib -nodefaultlibs -no-pie \
	-Wl,--no-dynamic-linker \
	-Wl,--build-id=none \
	-Wl,--gc-sections \
	-Wl,--defsym,_program_data_end=0x7fdf0000 \
	-Wl,--defsym,_fstack=0x7fe00000 \
	-L$(BUILD_DIR)/software/include

LIBS := -Wl,--start-group $(shell $(CC) $(ARCH_FLAGS) -print-libgcc-file-name) -Wl,--end-group

PORT_SRC_C := \
	main.c \
	mphalport.c \
	modmachine.c \
	ram_filesystem.c \
	libm_shims.c

EXTMOD_SRC_C := \
	extmod/modos.c \
	extmod/modtime.c \
	extmod/modvfs.c \
	extmod/vfs.c \
	extmod/vfs_blockdev.c \
	extmod/vfs_lfs.c \
	extmod/vfs_reader.c \
	lib/littlefs/lfs2.c \
	lib/littlefs/lfs2_util.c \
	shared/timeutils/timeutils.c

SHARED_SRC_C := \
	shared/libc/printf.c \
	shared/libc/string0.c \
	shared/readline/readline.c \
	shared/runtime/gchelper_native.c \
	shared/runtime/interrupt_char.c \
	shared/runtime/pyexec.c \
	shared/runtime/stdout_helpers.c \
	shared/runtime/sys_stdio_mphal.c

LIBM_SRC_C := $(addprefix lib/libm/,\
	acoshf.c asinfacosf.c asinhf.c atan2f.c atanf.c atanhf.c \
	ef_rem_pio2.c ef_sqrt.c erf_lgamma.c fmodf.c kf_cos.c \
	kf_rem_pio2.c kf_sin.c kf_tan.c log1pf.c math.c nearbyintf.c \
	roundf.c sf_cos.c sf_erf.c sf_frexp.c sf_ldexp.c sf_modf.c \
	sf_sin.c sf_tan.c wf_lgamma.c wf_tgamma.c)

$(BUILD)/lib/libm/%.o: CFLAGS += -Wno-sign-compare

SRC_QSTR += $(PORT_SRC_C) $(EXTMOD_SRC_C) $(SHARED_SRC_C)
QSTR_GLOBAL_DEPENDENCIES += mphalport.h

OBJ += $(PY_O)
OBJ += $(addprefix $(BUILD)/,$(PORT_SRC_C:.c=.o))
OBJ += $(addprefix $(BUILD)/,$(EXTMOD_SRC_C:.c=.o))
OBJ += $(addprefix $(BUILD)/,$(SHARED_SRC_C:.c=.o))
OBJ += $(addprefix $(BUILD)/,$(LIBM_SRC_C:.c=.o))
OBJ += $(BUILD)/shared/runtime/gchelper_rv32i.o
OBJ += $(BUILD)/crt0.o $(BUILD)/project_runtime.o
OBJ += $(EXTRA_OBJECTS)

ELF := $(BUILD)/micropython_repl.elf
BIN := $(BUILD)/micropython_repl.bin
MAP := $(BUILD)/micropython_repl.elf.map

all: $(BIN)

$(BIN): $(ELF)
	$(OBJCOPY) -O binary $< $@
	chmod -x $@

$(ELF): $(OBJ) ../runtime/linker.ld
	$(CC) $(LDFLAGS) -T ../runtime/linker.ld -N -o $@ $(OBJ) $(LIBS) \
		-Wl,-Map,$(MAP)
	chmod -x $@
	$(SIZE) $@

$(BUILD)/crt0.o: ../runtime/crt0.S | $(BUILD)
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILD)/project_runtime.o: ../runtime/runtime.c | $(BUILD)
	$(CC) $(CFLAGS) -MD -MP -c $< -o $@

.PHONY: all

include $(TOP)/py/mkrules.mk

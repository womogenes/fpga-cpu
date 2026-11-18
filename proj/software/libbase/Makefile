include ../include/generated/variables.mak
include $(SOC_DIRECTORY)/software/common.mak

# Keep LiteX's standard libbase sources, but use direct hardware-FIFO polling
# for UART. The interrupt implementation has a fixed 128-byte software ring
# that drops bytes from SFL's 255-byte serial-boot frames at high baud rates.
UPSTREAM_LIBBASE_DIRECTORY := $(SOC_DIRECTORY)/software/libbase
CFLAGS += -I$(UPSTREAM_LIBBASE_DIRECTORY)

OBJECTS =  \
	crc16.o    \
	crc32.o    \
	console.o  \
	format.o   \
	system.o   \
	progress.o \
	memtest.o  \
	uart.o     \
	startup.o  \
	spiflash.o \
	i2c.o      \
	isr.o      \
	hyperram.o

all: libbase.a

libbase.a: $(OBJECTS)
	$(AR) crs libbase.a $(OBJECTS)

-include $(OBJECTS:.o=.d)

uart.o: $(UPSTREAM_LIBBASE_DIRECTORY)/uart.c
	$(call compile,-DUART_POLLING)

%.o: $(UPSTREAM_LIBBASE_DIRECTORY)/%.c
	$(compile)

.PHONY: all clean

clean:
	$(RM) $(OBJECTS)
	$(RM) libbase.a .*~ *~

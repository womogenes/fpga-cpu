CC ?= cc
CFLAGS ?= -O2 -Wall -Wextra -std=gnu11

.PHONY: all clean

all: mandelbrot badapple snake hello

mandelbrot: mandelbrot.c video_hw.c video_hw.h
	$(CC) $(CFLAGS) -o $@ mandelbrot.c video_hw.c -lm

badapple: badapple.c video_hw.c video_hw.h
	$(CC) $(CFLAGS) -o $@ badapple.c video_hw.c -lz

snake: snake.c
	$(CC) $(CFLAGS) -o $@ snake.c

hello: hello.c
	$(CC) $(CFLAGS) -o $@ hello.c

clean:
	rm -f mandelbrot badapple snake hello

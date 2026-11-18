# Text demo

This application exercises the shared `sw/text_console` renderer on the
1280x720 RGB565 framebuffer. The selected Spleen BDF source is converted to a
compact bitmap and linked into the application binary automatically.

Build with the default 8x16 font:

```sh
make -C sw/text_demo
```

Scale every source pixel to a 2x2 block, producing 16x32 character cells:

```sh
make -C sw/text_demo FONT_SCALE=2
```

Select another native font without resampling it:

```sh
make -C sw/text_demo FONT_SIZE=16x32
```

Available sizes are `5x8`, `6x12`, `8x16`, `12x24`, `16x32`, and `32x64`.
The default 8x16 size fits 160 columns and 45 complete rows; 16x32 fits 80
columns and 22 complete rows. Only printable ASCII (characters 32 through 126)
is embedded.

## Formatted output

This demo opts into the project's small Newlib configuration. Standard
`printf()` writes to UART, while the demo's small `console_printf()` wrapper
formats into a fixed buffer and passes the result to `text_console_write()`:

```c
printf("UART value: %u\n", value);
console_printf("SCREEN VALUE: %u\n", value);
```

Integer decimal and hexadecimal formatting are enabled. Floating-point
formatting is intentionally omitted by Newlib Nano unless explicitly linked.
Characters paint both their foreground and background pixels. Scrolling uses
word-wide framebuffer copies and a small character grid; the video DMA base
remains fixed. The prime-number demo targets 60 lines per second. Its rate can
be adjusted at build time:

```sh
make -C sw/text_demo DEMO_LINES_PER_SECOND=60
```

Spleen is copyright 2018-2026 Frederic Cambus and distributed under the BSD
2-Clause license. See `../text_console/LICENSE.spleen` for its license text.

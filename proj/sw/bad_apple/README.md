# Bad Apple video player

This software-only application converts a source video into a compact monochrome format, embeds it in the firmware, and loops it on the 1280×720 RGB565 framebuffer. The default 160×90 source geometry scales exactly 8× in each direction.

Build from `proj/`:

```sh
make bad-apple
make upload bad_apple
```

The build requires `ffmpeg`. Override the source or conversion settings without changing the player:

```sh
make -C sw/bad_apple VIDEO=/path/to/video.mp4 VIDEO_WIDTH=160 VIDEO_HEIGHT=90 VIDEO_FPS=30 VIDEO_THRESHOLD=128
```

`video_to_bin.py` bit-packs each thresholded frame, XORs it against the preceding frame, applies a small PackBits-style encoding, and wraps the complete stream in DEFLATE compression to reduce firmware upload time. The player expands it into DDR once at startup. The runtime then updates only changed source pixels, expanding each one into a framebuffer block. MVF1 is generic: any source video with a width divisible by eight and dimensions that divide 1280×720 can use the same player.

The inflater is uzlib from MicroPython and retains its upstream permissive license notice.

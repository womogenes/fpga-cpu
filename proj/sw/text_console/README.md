# Text console

This shared software renderer provides a fixed-grid ASCII terminal on the
1280x720 RGB565 framebuffer. It owns the Spleen BDF assets, converts the
selected font during each application's build, and links the packed glyphs
directly into that application.

Include `text_console.mk` before `runtime/app.mk`, then call
`text_console_init()`, `text_console_write()`, and `text_console_present()`.
`text_console_service()` performs a rate-limited presentation when output is
being generated continuously.

The renderer understands carriage return, newline, tab, backspace, VT100
cursor-left, and erase-to-end-of-line. Scrolling moves framebuffer pixels but
never changes or restarts the framebuffer DMA base.

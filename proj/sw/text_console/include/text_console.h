#ifndef PROJECT_TEXT_CONSOLE_H
#define PROJECT_TEXT_CONSOLE_H

#include <stddef.h>
#include <stdint.h>

struct text_console_config {
  volatile uint16_t* framebuffer;
  uint16_t foreground;
  uint16_t background;
};

uint16_t text_console_rgb565(uint8_t red, uint8_t green, uint8_t blue);
void text_console_init(const struct text_console_config* config);
void text_console_set_cursor_enabled(int enabled);
void text_console_write(const char* data, size_t length);
void text_console_service(void);
void text_console_present(void);

#endif

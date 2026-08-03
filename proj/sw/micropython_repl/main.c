#include <generated/mem.h>
#include <runtime.h>
#include <stdint.h>
#include <text_console.h>

#include "py/builtin.h"
#include "py/compile.h"
#include "py/gc.h"
#include "py/mperrno.h"
#include "py/runtime.h"
#include "shared/runtime/gchelper.h"
#include "shared/runtime/pyexec.h"

#include "ram_filesystem.h"

#ifndef MICROPY_HEAP_SIZE
#define MICROPY_HEAP_SIZE (8u * 1024u * 1024u)
#endif

#define MICROPY_STACK_LIMIT (48u * 1024u)

static uint8_t gc_heap[MICROPY_HEAP_SIZE]
    DDR_NOINIT __attribute__((aligned(16)));

static void print_fatal(const char* message) {
  mp_printf(&mp_plat_print, "fatal: %s\r\n", message);
  text_console_present();
}

int main(void) {
  struct text_console_config console_config = {
      .framebuffer = (volatile uint16_t*)VIDEO_FRAMEBUFFER_BASE,
      .foreground = text_console_rgb565(224, 238, 255),
      .background = text_console_rgb565(0, 0, 0),
  };

  uart_init();
  text_console_init(&console_config);
  text_console_set_cursor_enabled(1);
  mp_cstack_init_with_sp_here(MICROPY_STACK_LIMIT);
  gc_init(gc_heap, gc_heap + sizeof(gc_heap));

  for (;;) {
    mp_init();
    ram_filesystem_mount();

    for (;;) {
      int result;
      if (pyexec_mode_kind == PYEXEC_MODE_RAW_REPL)
        result = pyexec_raw_repl();
      else
        result = pyexec_friendly_repl();
      if (result != 0)
        break;
    }

    mp_printf(&mp_plat_print, "MPY: soft reboot\r\n");
    text_console_present();
    gc_sweep_all();
    mp_deinit();
  }
}

void gc_collect(void) {
  gc_collect_start();
  gc_helper_collect_regs_and_stack();
  gc_collect_end();
}

void nlr_jump_fail(void* value) {
  (void)value;
  print_fatal("uncaught non-local return");
  for (;;) {
  }
}

__attribute__((noreturn)) void __fatal_error(const char* message) {
  print_fatal(message);
  for (;;) {
  }
}

void abort(void) {
  __fatal_error("abort");
}

#ifndef NDEBUG
void __assert_func(const char* file, int line, const char* function,
                   const char* expression) {
  mp_printf(&mp_plat_print, "assertion failed: %s (%s:%d, %s)\r\n",
            expression, file, line, function);
  text_console_present();
  for (;;) {
  }
}
#endif

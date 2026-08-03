#ifndef MICROPY_INCLUDED_FPGA_CPU_MPCONFIGPORT_H
#define MICROPY_INCLUDED_FPGA_CPU_MPCONFIGPORT_H

#include <alloca.h>
#include <stdint.h>

#define MICROPY_CONFIG_ROM_LEVEL (MICROPY_CONFIG_ROM_LEVEL_CORE_FEATURES)
#define MICROPY_ENABLE_COMPILER (1)
#define MICROPY_ENABLE_GC (1)
#define MICROPY_ENABLE_FINALISER (1)
#define MICROPY_ENABLE_EXTERNAL_IMPORT (1)
#define MICROPY_PERSISTENT_CODE_LOAD (0)
#define MICROPY_HELPER_REPL (1)
#define MICROPY_REPL_AUTO_INDENT (1)
#define MICROPY_READLINE_HISTORY_SIZE (16)
#define MICROPY_KBD_EXCEPTION (1)
#define MICROPY_ENABLE_EMERGENCY_EXCEPTION_BUF (1)
#define MICROPY_LONGINT_IMPL (MICROPY_LONGINT_IMPL_MPZ)
#define MICROPY_MALLOC_USES_ALLOCATED_SIZE (1)
#define MICROPY_OPT_COMPUTED_GOTO (1)
#define MICROPY_EMIT_RV32 (0)
#define MICROPY_EMIT_INLINE_RV32 (0)
#define MICROPY_PY_THREAD (0)
#define MICROPY_PY_MACHINE (0)
#define MICROPY_PY_BUILTINS_HELP (1)
#define MICROPY_PY_BUILTINS_HELP_MODULES (1)
#define MICROPY_PY_OS (1)
#define MICROPY_PY_OS_STATVFS (1)
#define MICROPY_PY_OS_URANDOM (1)
#define MICROPY_PY_TIME (1)
#define MICROPY_VFS (1)
#define MICROPY_VFS_LFS2 (1)
#define MICROPY_READER_VFS (1)
#define MICROPY_PY_SYS_PLATFORM "fpga-cpu"
#define MICROPY_HW_BOARD_NAME "FPGA CPU"
#define MICROPY_HW_MCU_NAME "RV32IM"

#define MICROPY_VM_HOOK_COUNT (32)
#define MICROPY_VM_HOOK_INIT \
  static unsigned int vm_hook_divisor = MICROPY_VM_HOOK_COUNT;
#define MICROPY_VM_HOOK_POLL            \
  do {                                  \
    if (--vm_hook_divisor == 0u) {      \
      vm_hook_divisor = MICROPY_VM_HOOK_COUNT; \
      mp_hal_event_poll_hook();         \
    }                                   \
  } while (0)
#define MICROPY_VM_HOOK_LOOP MICROPY_VM_HOOK_POLL;
#define MICROPY_VM_HOOK_RETURN MICROPY_VM_HOOK_POLL;
#define MICROPY_EVENT_POLL_HOOK mp_hal_event_poll_hook();

#define MP_SSIZE_MAX (0x7fffffff)

typedef long mp_off_t;

void mp_hal_event_poll_hook(void);

#define MP_STATE_PORT MP_STATE_VM

#endif

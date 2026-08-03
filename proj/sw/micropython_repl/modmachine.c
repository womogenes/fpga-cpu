#include <generated/csr.h>

#include "py/obj.h"
#include "py/runtime.h"

static mp_obj_t machine_reset(void) {
  /* Bit 0 is LiteX's one-cycle whole-SoC reset pulse. */
  ctrl_reset_write(1u << CSR_CTRL_RESET_SOC_RST_OFFSET);

  /* The reset should arrive immediately; do not continue if it is delayed. */
  for (;;) {
    __asm__ volatile("nop");
  }
  __builtin_unreachable();
}
static MP_DEFINE_CONST_FUN_OBJ_0(machine_reset_obj, machine_reset);

static const mp_rom_map_elem_t machine_module_globals_table[] = {
    {MP_ROM_QSTR(MP_QSTR___name__), MP_ROM_QSTR(MP_QSTR_machine)},
    {MP_ROM_QSTR(MP_QSTR_reset), MP_ROM_PTR(&machine_reset_obj)},
};
static MP_DEFINE_CONST_DICT(machine_module_globals,
                            machine_module_globals_table);

const mp_obj_module_t machine_module = {
    .base = {&mp_type_module},
    .globals = (mp_obj_dict_t*)&machine_module_globals,
};

MP_REGISTER_MODULE(MP_QSTR_machine, machine_module);

#include <runtime.h>
#include <stdint.h>
#include <string.h>

#include "extmod/vfs.h"
#include "extmod/vfs_lfs.h"
#include "py/binary.h"
#include "py/mperrno.h"
#include "py/objarray.h"
#include "py/runtime.h"

#include "ram_filesystem.h"

#define RAM_FILESYSTEM_SIZE (2u * 1024u * 1024u)
#define RAM_FILESYSTEM_BLOCK_SIZE (4096u)
#define RAM_FILESYSTEM_BLOCK_COUNT \
  (RAM_FILESYSTEM_SIZE / RAM_FILESYSTEM_BLOCK_SIZE)

typedef struct {
  mp_obj_base_t base;
} ram_block_device_t;

static uint8_t ram_filesystem_storage[RAM_FILESYSTEM_SIZE]
    DDR_NOINIT __attribute__((aligned(RAM_FILESYSTEM_BLOCK_SIZE)));

static mp_obj_t ram_block_device_readblocks(size_t argument_count,
                                             const mp_obj_t* arguments) {
  mp_int_t block = mp_obj_get_int(arguments[1]);
  mp_int_t offset = argument_count == 4 ? mp_obj_get_int(arguments[3]) : 0;
  mp_buffer_info_t buffer;
  mp_get_buffer_raise(arguments[2], &buffer, MP_BUFFER_WRITE);

  if (block < 0 || offset < 0) return MP_OBJ_NEW_SMALL_INT(-MP_EINVAL);

  size_t address = (size_t)block * RAM_FILESYSTEM_BLOCK_SIZE + (size_t)offset;
  if (address > RAM_FILESYSTEM_SIZE ||
      buffer.len > RAM_FILESYSTEM_SIZE - address)
    return MP_OBJ_NEW_SMALL_INT(-MP_EINVAL);

  memcpy(buffer.buf, ram_filesystem_storage + address, buffer.len);
  return MP_OBJ_NEW_SMALL_INT(0);
}
static MP_DEFINE_CONST_FUN_OBJ_VAR_BETWEEN(ram_block_device_readblocks_obj, 3,
                                            4, ram_block_device_readblocks);

static mp_obj_t ram_block_device_writeblocks(size_t argument_count,
                                              const mp_obj_t* arguments) {
  mp_int_t block = mp_obj_get_int(arguments[1]);
  mp_int_t offset = argument_count == 4 ? mp_obj_get_int(arguments[3]) : 0;
  mp_buffer_info_t buffer;
  mp_get_buffer_raise(arguments[2], &buffer, MP_BUFFER_READ);

  if (block < 0 || offset < 0) return MP_OBJ_NEW_SMALL_INT(-MP_EINVAL);

  size_t address = (size_t)block * RAM_FILESYSTEM_BLOCK_SIZE + (size_t)offset;
  if (address > RAM_FILESYSTEM_SIZE ||
      buffer.len > RAM_FILESYSTEM_SIZE - address)
    return MP_OBJ_NEW_SMALL_INT(-MP_EINVAL);

  memcpy(ram_filesystem_storage + address, buffer.buf, buffer.len);
  return MP_OBJ_NEW_SMALL_INT(0);
}
static MP_DEFINE_CONST_FUN_OBJ_VAR_BETWEEN(ram_block_device_writeblocks_obj, 3,
                                            4, ram_block_device_writeblocks);

static mp_obj_t ram_block_device_ioctl(mp_obj_t self, mp_obj_t command_object,
                                        mp_obj_t argument_object) {
  (void)self;
  mp_int_t command = mp_obj_get_int(command_object);
  mp_int_t argument = mp_obj_get_int(argument_object);

  switch (command) {
    case MP_BLOCKDEV_IOCTL_INIT:
    case MP_BLOCKDEV_IOCTL_DEINIT:
    case MP_BLOCKDEV_IOCTL_SYNC:
      return MP_OBJ_NEW_SMALL_INT(0);
    case MP_BLOCKDEV_IOCTL_BLOCK_COUNT:
      return MP_OBJ_NEW_SMALL_INT(RAM_FILESYSTEM_BLOCK_COUNT);
    case MP_BLOCKDEV_IOCTL_BLOCK_SIZE:
      return MP_OBJ_NEW_SMALL_INT(RAM_FILESYSTEM_BLOCK_SIZE);
    case MP_BLOCKDEV_IOCTL_BLOCK_ERASE:
      if (argument < 0 || (size_t)argument >= RAM_FILESYSTEM_BLOCK_COUNT)
        return MP_OBJ_NEW_SMALL_INT(-MP_EINVAL);
      memset(ram_filesystem_storage +
                 (size_t)argument * RAM_FILESYSTEM_BLOCK_SIZE,
             0xff, RAM_FILESYSTEM_BLOCK_SIZE);
      return MP_OBJ_NEW_SMALL_INT(0);
    default:
      return mp_const_none;
  }
}
static MP_DEFINE_CONST_FUN_OBJ_3(ram_block_device_ioctl_obj,
                                  ram_block_device_ioctl);

static const mp_rom_map_elem_t ram_block_device_locals_table[] = {
    {MP_ROM_QSTR(MP_QSTR_readblocks),
     MP_ROM_PTR(&ram_block_device_readblocks_obj)},
    {MP_ROM_QSTR(MP_QSTR_writeblocks),
     MP_ROM_PTR(&ram_block_device_writeblocks_obj)},
    {MP_ROM_QSTR(MP_QSTR_ioctl), MP_ROM_PTR(&ram_block_device_ioctl_obj)},
};
static MP_DEFINE_CONST_DICT(ram_block_device_locals,
                            ram_block_device_locals_table);

static MP_DEFINE_CONST_OBJ_TYPE(ram_block_device_type, MP_QSTR_RAMBlockDevice,
                                MP_TYPE_FLAG_NONE, locals_dict,
                                &ram_block_device_locals);

static const ram_block_device_t ram_block_device = {
    .base = {&ram_block_device_type},
};

void ram_filesystem_mount(void) {
  mp_obj_t block_device = MP_OBJ_FROM_PTR(&ram_block_device);
  mp_obj_t mount_point = MP_OBJ_NEW_QSTR(MP_QSTR__slash_);
  int result = mp_vfs_mount_and_chdir_protected(block_device, mount_point);

  if (result == -MP_ENODEV) {
    mp_obj_t mkfs = mp_load_attr(MP_OBJ_FROM_PTR(&mp_type_vfs_lfs2),
                                 MP_QSTR_mkfs);
    mp_call_function_1(mkfs, block_device);
    result = mp_vfs_mount_and_chdir_protected(block_device, mount_point);
  }

  if (result != 0) mp_raise_OSError(-result);

  mp_obj_list_append(mp_sys_path, MP_OBJ_NEW_QSTR(MP_QSTR_));
  mp_obj_list_append(mp_sys_path, MP_OBJ_NEW_QSTR(MP_QSTR__slash_lib));
}

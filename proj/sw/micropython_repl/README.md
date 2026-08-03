# MicroPython REPL

This application runs a MicroPython v1.28.0 REPL directly on the RV32IM CPU. It polls UART for input and mirrors output to UART and the HDMI text console.

Build only the software image:

```sh
make -C sw/micropython_repl
```

From `proj/`, upload it with:

```sh
make upload micropython_repl
```

The normal friendly REPL supports expressions, multiline definitions, history, completion, arbitrary-size integers, and software floating point. Ctrl-A enters raw REPL mode, Ctrl-B returns to friendly mode, Ctrl-C interrupts running Python, and Ctrl-D performs a soft reboot.

## Files and modules

The root directory is a 2 MiB littlefs filesystem backed by DDR. Normal file APIs, directories, Python imports, `os`, `time`, `vfs`, and builtin `help()` are enabled:

```python
with open("hello.py", "w") as file:
    file.write("answer = 42\n")

import hello
print(hello.answer)

import os
print(os.listdir())

import time
time.sleep_ms(100)
```

The import path searches the current directory and `/lib`. Files survive a `Ctrl-D` MicroPython soft reboot because the RAM disk is outside the garbage-collected heap. A full SoC reset, FPGA reconfiguration, or power cycle may lose them. The block-device boundary is intentionally generic so persistent SD or flash storage can replace the RAM disk later.

Reset the whole SoC back into the LiteX BIOS from the REPL with:

```python
import machine
machine.reset()
```

If `litex_term` is still attached with a firmware image, it sees the BIOS download request and uploads the image again automatically. This uses the existing LiteX reset CSR and does not reconfigure the FPGA.

The default HDMI font is 8x16. Override it without rebuilding hardware:

```sh
make -C sw/micropython_repl FONT_SIZE=16x32
```

The port reserves an 8 MiB garbage-collected DDR heap, a 2 MiB DDR RAM disk, and a 64 KiB DDR stack region. Most of the remaining DDR address space remains available to future features.

The vendored MicroPython subset is pinned to v1.28.0, commit `e0e9fbb17ed6fd06bb76e266ae554784c9c80804`, under its MIT license. The littlefs sources retain their BSD-3-Clause notices.

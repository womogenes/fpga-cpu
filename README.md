# FPGA CPU projects

- `proj/` contains the original Genesys 2 LiteX CPU, HDMI framebuffer, firmware, and build/programming tools.
- `proj/cva6/` contains the 64-bit CVA6 Linux workstation, hardware HDMI terminal, software-video plane, NumPy/OpenBLAS and standalone LFM inference support, RAM/SD images, and its pinned hardware and SDK dependencies.

Each sibling project owns its dependencies and generated-output ignores.

# Notes for agents

- Use sentence case for markdown headers and titles in general
- Ensure markdown tables are formatted properly for viewing in a monospace text editor
- Generally, do not use Latex in chat as it will not be rendered properly
- Fix markdown files so that single lines are not split across multiple for column-wrap

## "Remote mode"

If the user asks to operate in "remote mode," the FPGA is attached to a remote server and is hooked up to it via JTAG/USB UART.

You should ask the user for the hostname/IP of the remote server if it is not clear already.

Every time we synthesize, we should do it locally and then scp the `.obj` file over. openFPGALoader should already be installed on the remote server. Use that to flash onto the FPGA.

## Goal (night of 8/3)

commit your work thus far (just the cva6 stuff)

it's time to work highly autonomously and produce something extra cool.

note we don't have an SD card yet, but will have one tomorrow. don't implement this yet, it can all stay in DRAM for now.

you have a camera now by the way! it's pointed at the projector. you should use it to interact with serial and see the output and figure out if it's doing what you intended.

things we want at the end:

- real linux filesystem with home at /root/
- bash
- python >=3.12 installed and working (python and python3 both bring up shell). all standard libraries are included.
  - ipython with syntax highlighting would be super epic.
- vim and tmux work as intended to edit files + run stuff in parallel
- we can write a c program on the remote filesystem to play bad apple, like we did in the other thread. we should also be able to play something like an optimized 30fps mandelbrot zoom (with color!)
- running fairly fast
- games! snake, space invaders, tetris (colors!), etc. these can all be regular c programs on the remote filesystem.
- optimize the syscalls and whatever so that this all runs rather fast.

all this must be validated through serial input and HDMI visual output. latency should be minimal.

generally, synthesize hardware few times (optimally no more times) and design it so that all future customization/debugging can be done in software. this is to make iteration loops faster, and is generally good for

if we're feeling extra ambitious:

- install numpy and add hardware acceleration
- run a small LLM on the linux system. qwen3 0.6B? smollm2.5-350m? up to you. there might be some helpful context in `/home/willi/coding/ai/bonsai/inference`, including its codex thread.

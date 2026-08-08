# CVA6 Linux workstation

This is a real writable Linux filesystem. A RAM boot disappears at power-off; run `sd-install --help` to install the live system and staged models onto the persistent onboard SD card.

Run `python` for the Python 3.14 shell, `vim` to edit files, and `tmux` to keep multiple shells alive. Both `python` and `python3` start Python. NumPy is linked to threaded OpenBLAS for the four CVA6 floating-point units; run `numpy-bench` for configuration and speed measurements.

Run `lfm-chat` to chat with the standalone LFM2.5-350M Q4 model, or run `lfm-chat --self-test` to check its tokenizer without weights. A RAM boot uses weights staged directly in reserved DDR; an SD boot stages the persistent file from `/root/models` once at startup. Model matrix-vector operations are offloaded to the FPGA accelerator, and the client prints measured prompt and decode rates after each reply. Run `bonsai-chat` for the larger Bonsai 1.7B Q1 model.

The hardware display demos are `mandelbrot` and `badapple`. They switch HDMI from the terminal to the 160x90 RGB232 software-video plane and return to the terminal on Ctrl-C. Run `video-bench` to measure packed frame submission without rendering work. Raising physical switch 7 forces the terminal immediately even if software crashes.

Games include `snake`, `tetris`, and `ascii-invaders`. Run `games` for a reminder.

The C examples live in `/root/examples`. Run `cd /root/examples && make`, edit any source in Vim, and rebuild natively with TinyCC without touching FPGA hardware.

Run `workstation-check` for a quick Linux, Bash, Python, NumPy, Vim, tmux, tokenizer, and native C compiler smoke test.

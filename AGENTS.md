# Notes for agents

- Use sentence case for markdown headers and titles in general
- Ensure markdown tables are formatted properly for viewing in a monospace text editor
- Generally, do not use Latex in chat as it will not be rendered properly

## "Remote mode"

If the user asks to operate in "remote mode," the FPGA is attached to a remote server and is hooked up to it via JTAG/USB UART.

You should ask the user for the hostname/IP of the remote server if it is not clear already.

Every time we synthesize, we should do it locally and then scp the `.obj` file over. openFPGALoader should already be installed on the remote server. Use that to flash onto the FPGA.

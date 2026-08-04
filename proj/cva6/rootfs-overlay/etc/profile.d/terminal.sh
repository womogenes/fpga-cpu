#!/bin/sh

if [ -t 0 ]; then
    export TERM=xterm
    export LANG=en_US.UTF-8
    export LC_ALL=en_US.UTF-8
    stty rows 45 cols 160
fi

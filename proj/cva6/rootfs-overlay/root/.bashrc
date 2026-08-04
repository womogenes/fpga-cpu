export TERM=xterm
export LANG=en_US.UTF-8
export LC_ALL=en_US.UTF-8
export EDITOR=vim
export VISUAL=vim
export PS1='\[\e[1;36m\]cva6\[\e[0m\]:\[\e[1;34m\]\w\[\e[0m\]# '
alias ll='ls -alF --color=auto'
alias python='python3'
alias cls='clear'
alias games='printf "snake\ntetris\nascii-invaders\nmandelbrot\nbadapple\n"'

stty rows 45 cols 160 2>/dev/null || true

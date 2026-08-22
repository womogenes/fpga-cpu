#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "usage: $0 ARTIFACT_DIR" >&2
    exit 2
fi

artifact_dir="$(cd "$1" && pwd)"
test -s "$artifact_dir/handoff.ready"
tmux has-session -t bonsai-console
tmux send-keys -t bonsai-console Enter
sleep 2
tmux capture-pane -pt bonsai-console -S -80 | tee "$artifact_dir/handoff-final-console.log"
grep -q 'you> ' "$artifact_dir/handoff-final-console.log"
ffmpeg -hide_banner -loglevel error -f v4l2 -input_format mjpeg \
    -video_size 1280x720 -i /dev/video0 -frames:v 1 -y \
    "$artifact_dir/handoff-camera0.jpg" || true
ffmpeg -hide_banner -loglevel error -f v4l2 -input_format mjpeg \
    -video_size 1920x1080 -i /dev/video2 -frames:v 1 -y \
    "$artifact_dir/handoff-camera2.jpg" || true
date --iso-8601=seconds >"$artifact_dir/handoff-final.ready"

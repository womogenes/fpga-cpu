#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "usage: $0 ARTIFACT_DIR" >&2
    exit 2
fi

artifact_dir="$(cd "$1" && pwd)"
ready="$artifact_dir/handoff.ready"

if [[ -s "$ready" ]] && tmux has-session -t bonsai-console 2>/dev/null; then
    echo "$(date --iso-8601=seconds) Handoff is already ready."
    exit 0
fi

if tmux has-session -t bonsai-console 2>/dev/null && \
        tmux capture-pane -pt bonsai-console -S -80 | grep -q 'you> '; then
    date --iso-8601=seconds >"$ready"
    echo "$(date --iso-8601=seconds) Recovered the ready marker from the live console."
    exit 0
fi

if systemctl --user is-active --quiet bonsai-handoff.service; then
    echo "$(date --iso-8601=seconds) Initial handoff is still running."
    exit 0
fi

echo "$(date --iso-8601=seconds) Handoff is not ready; restarting the deployment."
systemctl --user restart bonsai-handoff.service

#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
venv="$project_dir/.venv"
python="${PYTHON:-python3}"
stamp="$venv/.requirements.sha256"
requirements_hash="$(sha256sum "$project_dir/requirements.txt" | cut -d' ' -f1)"

if [[ ! -x "$venv/bin/python" ]]; then
    if ! "$python" -m venv "$venv"; then
        "$python" -m venv --clear --without-pip "$venv"
    fi
fi

if ! "$venv/bin/python" -m pip --version >/dev/null 2>&1; then
    get_pip="$(mktemp)"
    trap 'rm -f "$get_pip"' EXIT
    "$venv/bin/python" - "$get_pip" <<'PY'
import sys
import urllib.request

urllib.request.urlretrieve("https://bootstrap.pypa.io/get-pip.py", sys.argv[1])
PY
    "$venv/bin/python" "$get_pip"
    rm -f "$get_pip"
    trap - EXIT
fi

if [[ -f "$stamp" ]] && [[ "$(<"$stamp")" == "$requirements_hash" ]]; then
    echo "LiteX environment is already up to date: $venv"
    exit 0
fi

"$venv/bin/python" -m pip install --upgrade pip
"$venv/bin/python" -m pip install -r "$project_dir/requirements.txt"
printf '%s\n' "$requirements_hash" > "$stamp"

echo "LiteX environment is ready: $venv"

#!/usr/bin/env bash
# GUI を起動する（先に setup_ubuntu.sh を実行しておく）
set -euo pipefail
cd "$(dirname "$0")"
source .venv/bin/activate
exec python scripts/gui.py "$@"

#!/usr/bin/env bash
# 合成データ生成ソフトの実行環境を作る（Ubuntu 22.04 / 24.04）
#   bash setup_ubuntu.sh
# 1. GUI（Qt）に必要なシステムパッケージ  2. Python 仮想環境 .venv と依存パッケージ
# 3. テストと動作確認（オーバルを 2 m 走らせて runs/check に画像を出力）
set -euo pipefail
cd "$(dirname "$0")"

echo "== 1/3 システムパッケージ（sudo のパスワードを聞かれます）"
sudo apt-get update
sudo apt-get install -y python3-venv python3-pip \
  libxkbcommon-x11-0 libxcb-icccm4 libxcb-image0 libxcb-keysyms1 \
  libxcb-render-util0 libxcb-shape0 libxcb-xinerama0 libxcb-randr0 libegl1 libgl1 \
  libfontconfig1 libdbus-1-3 fonts-noto-cjk
# Qt 6.5 以降の xcb（X11）表示に必要。universe リポジトリにある
sudo apt-get install -y libxcb-cursor0 || echo "注意: libxcb-cursor0 を入れられませんでした（sudo add-apt-repository universe の後に再実行）"

echo "== 2/3 Python 仮想環境（.venv）"
python3 -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

echo "== 3/3 動作確認"
python -m pytest -q tests
python scripts/simulate.py --course config/courses/oval.yaml --out runs/check --distance 2
python - <<'PY'
import PySide6, cv2, numpy
print(f"Python OK: numpy {numpy.__version__}, OpenCV {cv2.__version__}, PySide6 {PySide6.__version__}")
PY
echo
echo "完了。runs/check/camera と runs/check/bev に画像ができていれば OK。"
echo "GUI の起動:  bash run_gui.sh"

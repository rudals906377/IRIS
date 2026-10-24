#!/usr/bin/env bash
# IRIS 생성 서버 한 번에 설치(리눅스, 엔비디아 GPU). 처음 한 번만 실행:  bash setup.sh
set -e
cd "$(dirname "$0")"
command -v python3 >/dev/null || { echo "[문제] python3 이 없습니다"; exit 1; }
[ -d venv ] || python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
echo "GPU용 PyTorch 설치 중(약 2.5GB)..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126 || pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
# mediapipe가 쓰는 그래픽 라이브러리(없으면 libEGL 오류)
if command -v apt-get >/dev/null && ! ldconfig -p | grep -q libEGL.so.1; then
  echo "libEGL 이 없어 설치를 시도합니다(관리자 암호를 물을 수 있음)"; sudo apt-get install -y libegl1 libgles2 libgl1 libglib2.0-0 || true
fi
echo
python check.py
echo
echo '위에 "모두 준비됨" 이 보이면  bash run.sh  를 실행하세요.'

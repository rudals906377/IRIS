#!/usr/bin/env bash
# 시험 자료 내려받기(저장소에는 올리지 않음: web/public/testdata 는 .gitignore)
# - Intel IoT DevKit sample-videos(CC BY 4.0): 웹캠 구도의 인물 영상
# - IDM-VTON 예시 사진(연구용, 시험 전용): 정면 인물 사진 → 얼굴을 크게 잘라 메이크업 시험 영상(testdata/face/)
# 필요: python3(imageio-ffmpeg, pillow), git
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
TD="$ROOT/web/public/testdata"
mkdir -p "$TD/face"
FF=$(python3 -c "import imageio_ffmpeg;print(imageio_ffmpeg.get_ffmpeg_exe())")
TMP=$(mktemp -d)
for f in head-pose-face-detection-female.mp4 head-pose-face-detection-male.mp4; do
  curl -sL -o "$TMP/$f" "https://raw.githubusercontent.com/intel-iot-devkit/sample-videos/master/$f"
done
"$FF" -hide_banner -loglevel error -y -i "$TMP/head-pose-face-detection-female.mp4" -t 40 -an -c:v libvpx-vp9 -b:v 0 -crf 33 -row-mt 1 -deadline realtime -cpu-used 8 "$TD/female.webm"
"$FF" -hide_banner -loglevel error -y -ss 100 -i "$TMP/head-pose-face-detection-male.mp4" -t 34 -an -c:v libvpx-vp9 -b:v 0 -crf 33 -row-mt 1 -deadline realtime -cpu-used 8 "$TD/male-turn.webm"
if [ ! -d "$TMP/idm/.git" ]; then
  GIT_LFS_SKIP_SMUDGE=1 git clone -q --depth 1 --filter=blob:none --sparse https://github.com/yisol/idm-vton "$TMP/idm"
  git -C "$TMP/idm" sparse-checkout set gradio_demo/example/human
fi
mkdir -p "$TD/human"
for n in 00034_00 00035_00 00055_00 00121_00 01992_00; do cp "$TMP/idm/gradio_demo/example/human/$n.jpg" "$TD/human/"; done
for n in 00121_00 01992_00 00035_00 00055_00 00034_00; do
  "$FF" -hide_banner -loglevel error -y -loop 1 -t 2 -i "$TMP/idm/gradio_demo/example/human/$n.jpg" -vf "scale=576:768,fps=15" -c:v libvpx-vp9 -b:v 0 -crf 30 -deadline realtime -cpu-used 8 "$TD/still_$n.webm"
done
cp "$TMP/idm/gradio_demo/example/human/sam1 (1).jpg" "$TD/human/sam1.jpg"
# 얼굴 확대 영상(메이크업 시험용). 잘라 낼 범위는 얼굴 점으로 한 번 재어 둔 값
python3 - "$TD" "$FF" <<'PY'
import subprocess, sys
from PIL import Image
td, ff = sys.argv[1], sys.argv[2]
boxes = {'00034_00': (255, 0, 470, 286), '00055_00': (294, 0, 533, 319), '01992_00': (178, 0, 419, 321), 'sam1': (372, 38, 703, 479)}
for n, box in boxes.items():
    Image.open(f'{td}/human/{n}.jpg').convert('RGB').crop(box).resize((720, 960), Image.LANCZOS).save(f'{td}/face/{n}.png')
    subprocess.run([ff, '-hide_banner', '-loglevel', 'error', '-y', '-loop', '1', '-t', '2', '-i', f'{td}/face/{n}.png', '-vf', 'fps=15',
                    '-c:v', 'libvpx-vp9', '-b:v', '0', '-crf', '24', '-deadline', 'realtime', '-cpu-used', '8', f'{td}/face/{n}.webm'], check=True)
PY
cp "$ROOT/web/node_modules/@mediapipe/tasks-vision/vision_bundle.mjs" "$TD/" 2>/dev/null || true
echo "시험 자료 준비 완료: $TD"
ls "$TD"

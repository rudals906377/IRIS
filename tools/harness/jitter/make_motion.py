# 얼굴 점 떨림·지연 측정용 합성 영상: 얼굴 사진을 알려진 이동·회전·크기로 움직인다(정답을 안다).
# 0~2초 정지, 2~4초 느린 이동, 4~6초 빠른 흔들기(초당 3회), 6~8초 정지.
# 사용법: python3 make_motion.py <얼굴 사진(testdata/face/*.png)> <출력 폴더>
# 결과: motion.webm(web/public/testdata 에 복사해 쓴다), poses.json(프레임별 정답 변환)
import json, math, subprocess, sys
import imageio_ffmpeg
from PIL import Image
src, out = sys.argv[1], sys.argv[2]
W, H, N = 640, 480, 240
base = Image.open(src).convert('RGB').resize((360, 480), Image.LANCZOS)
frames, poses = [], []
for i in range(N):
    t = i / 30
    if t < 2: dx, dy, th, s = 0, 0, 0, 1
    elif t < 4: u = (t - 2) / 2; dx, dy, th, s = 60 * u, 20 * math.sin(u * math.pi), 6 * u, 1 + 0.08 * u
    elif t < 6: u = t - 4; dx, dy, th, s = 60 + 35 * math.sin(2 * math.pi * 3 * u), 0, 6 + 5 * math.sin(2 * math.pi * 3 * u + 1), 1.08
    else: dx, dy, th, s = 60, 0, 6, 1.08
    poses.append([dx, dy, th, s])
    im = base.rotate(th, resample=Image.BICUBIC, center=(180, 240), expand=False, fillcolor=(230, 230, 230))
    im = im.resize((round(360 * s), round(480 * s)), Image.BICUBIC)
    canvas = Image.new('RGB', (W, H), (230, 230, 230))
    canvas.paste(im, (round(260 + dx - im.width / 2), round(240 + dy - im.height / 2)))
    frames.append(canvas)
ff = imageio_ffmpeg.get_ffmpeg_exe()
p = subprocess.Popen([ff, '-hide_banner', '-loglevel', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{W}x{H}', '-r', '30', '-i', '-',
                      '-c:v', 'libvpx-vp9', '-b:v', '0', '-crf', '20', '-deadline', 'realtime', '-cpu-used', '8', f'{out}/motion.webm'], stdin=subprocess.PIPE)
for f in frames: p.stdin.write(f.tobytes())
p.stdin.close(); p.wait()
json.dump({'poses': poses}, open(f'{out}/poses.json', 'w'))
print('완료', N, '프레임')

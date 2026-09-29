"""분석 결과(부위 라벨 색 + 기준점)를 한 장의 결과표로 그린다.
사용법: python3 render.py <작업 폴더>  → <작업 폴더>/sheet.jpg
"""
import json, os, sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont
os.chdir(sys.argv[1] if len(sys.argv) > 1 else '.')
meta = json.load(open('meta.json')); res = {r['i']: r for r in json.load(open('result.json'))}
font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 11)
cols = {1: (80, 80, 255), 2: (255, 60, 60), 3: (60, 200, 60), 4: (255, 200, 0)}
tiles = []
for m in meta:
    i, w, h = m['i'], m['w'], m['h']
    img = np.fromfile(f'img{i}.rgba', np.uint8).reshape(h, w, 4)[..., :3].astype(np.float32)
    r = res[i]
    try:
        lab = np.fromfile(f'lab{i}.bin', np.uint8).reshape(h, w)
    except FileNotFoundError:
        lab = np.fromfile(f'mask{i}.bin', np.uint8).reshape(h, w)
    over = img.copy()
    for k, c in cols.items():
        sel = lab == k
        over[sel] = over[sel] * 0.5 + np.array(c) * 0.5
    im = Image.fromarray(over.clip(0, 255).astype(np.uint8))
    d = ImageDraw.Draw(im)
    for k, p in (r.get('kp') or {}).items():
        x, y = p['x'], p['y']
        d.ellipse((x-4, y-4, x+4, y+4), fill=(255, 0, 255), outline=(0, 0, 0))
        d.text((x+5, y-6), k.replace('sleeve', 'sl'), fill=(0, 0, 0), font=font)
    d.text((4, 4), f"#{i} {r.get('sleeve')} c={r.get('conf')}", fill=(0, 0, 0), font=font)
    tiles.append(im.resize((round(w * 384 / h), 384)))
cols_n = 7
tw = max(t.size[0] for t in tiles)
rows = (len(tiles) + cols_n - 1) // cols_n
sheet = Image.new('RGB', (tw * cols_n, 384 * rows), 'white')
for k, t in enumerate(tiles):
    sheet.paste(t, ((k % cols_n) * tw, (k // cols_n) * 384))
sheet.save('sheet.jpg', quality=85)
print(sheet.size)

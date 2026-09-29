"""상품 사진들을 최대 512px로 줄여 RGBA 원시 파일로 저장한다(노드 분석 시험용).
사용법: python3 prep.py <작업 폴더> <사진 경로들...>
"""
import json, sys
from pathlib import Path
import numpy as np
from PIL import Image

work = Path(sys.argv[1]); work.mkdir(parents=True, exist_ok=True)
meta = []
for i, f in enumerate(sys.argv[2:]):
    im = Image.open(f).convert('RGBA')
    s = 512 / max(im.size)
    im = im.resize((round(im.size[0] * s), round(im.size[1] * s)), Image.LANCZOS)
    np.asarray(im).tofile(work / f'img{i}.rgba')
    meta.append({'i': i, 'src': f, 'w': im.size[0], 'h': im.size[1]})
(work / 'meta.json').write_text(json.dumps(meta))
print(len(meta), '장 준비')

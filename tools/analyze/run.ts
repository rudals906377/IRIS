/// <reference types="node" />
// 브라우저와 같은 분석 코드(web/src/engine/analyze)를 노드에서 돌려 결과를 저장한다.
// 사용법: node tools/analyze/run.ts <작업 폴더>   (prep.py로 준비한 폴더)
import { readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { removeBackground } from '../../web/src/engine/analyze/background.ts';
import { analyzeTop } from '../../web/src/engine/analyze/top.ts';

const work = process.argv[2] ?? '.';
const meta = JSON.parse(readFileSync(join(work, 'meta.json'), 'utf8'));
const out = [];
for (const m of meta) {
  const data = new Uint8ClampedArray(readFileSync(join(work, `img${m.i}.rgba`)));
  const t0 = performance.now();
  const bg = removeBackground({ data, width: m.w, height: m.h });
  const a = analyzeTop(bg.mask, m.w, m.h, data);
  const ms = performance.now() - t0;
  writeFileSync(join(work, `mask${m.i}.bin`), bg.mask);
  if (a) writeFileSync(join(work, `lab${m.i}.bin`), a.labels);
  out.push({ i: m.i, src: m.src, ok: !!a, sleeve: a?.sleeve, conf: a ? +a.confidence.toFixed(2) : 0, ms: +ms.toFixed(1), warnings: a?.warnings, kp: a?.keypoints });
  console.log(m.i, a ? `${a.sleeve}${a.highNeck ? '·하이넥' : ''} 신뢰도 ${a.confidence.toFixed(2)}` : '실패', a?.warnings.join(' | ') ?? '', '←', m.src);
}
writeFileSync(join(work, 'result.json'), JSON.stringify(out));

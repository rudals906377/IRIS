// 스모크 시험: 모든 효과(피부·윤곽·립·섀도 펄·헤어·타투·네일)를 켜고 영상을 돌려
// 셰이더 컴파일 오류·예외가 없는지 확인한다. 커밋 전에 실행(셰이더 오류는 빌드로는 안 잡힌다).
// 사용법: node tools/harness/smoke.mjs   (미리보기 서버가 떠 있어야 함)
import { BASE, launch, waitReady } from './common.mjs';

const cases = [
  'src=sample&look=glam&lstyle=gradient&over=0.5&pearl=0.6&hair=11&tattoo=moon&place=forearmL&nail=0&nstyle=french',
  'src=sample&look=smoky&lstyle=blur&hair=3',
];
const browser = await launch();
let failed = false;
for (const q of cases) {
  const page = await browser.newPage({ viewport: { width: 900, height: 900 } });
  const errors = [];
  page.on('pageerror', (e) => errors.push(e.message));
  page.on('console', (m) => {
    if (m.type() === 'error' && /셰이더|shader|WebGL|TypeError|ReferenceError/i.test(m.text())) errors.push(m.text());
  });
  await page.goto(`${BASE}/?delegate=CPU&${q}`);
  try {
    await waitReady(page, 90000);
    await page.waitForTimeout(3000);
  } catch {
    errors.push('준비 시간 초과(얼굴을 못 찾았거나 프레임이 처리되지 않음)');
  }
  const frames = await page.evaluate(() => window.iris?.metrics.framesTotal ?? 0);
  console.log(errors.length ? '실패' : '통과', `프레임 ${frames}`, q);
  for (const e of errors) console.log('  ', e.slice(0, 300));
  if (errors.length) failed = true;
  await page.close();
}
await browser.close();
process.exit(failed ? 1 : 0);

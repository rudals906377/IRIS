// 크롬 확장 프로그램을 빌드하고 설치용 zip을 만든다.
//   node scripts/build-ext.mjs [zip 경로]   (기본: ../release/iris-extension.zip)
import { execFileSync } from 'node:child_process';
import { cpSync, existsSync, mkdirSync, readdirSync, rmSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const out = resolve(root, 'dist-ext');
const zipPath = resolve(root, process.argv[2] ?? '../release/iris-extension.zip');

execFileSync('node', ['scripts/copy-wasm.mjs'], { cwd: root, stdio: 'inherit' });
execFileSync('npx', ['vite', 'build', '--config', 'vite.ext.config.ts'], { cwd: root, stdio: 'inherit' });

cpSync(resolve(root, 'extension/manifest.json'), resolve(out, 'manifest.json'));
cpSync(resolve(root, 'extension/icons'), resolve(out, 'icons'), { recursive: true });
// 시험용 자료와 이 확장에서 쓰지 않는 WASM 변형(비SIMD·모듈)은 뺀다(크롬은 SIMD 지원).
rmSync(resolve(out, 'testdata'), { recursive: true, force: true });
const wasmDir = resolve(out, 'mediapipe/wasm');
if (existsSync(wasmDir)) {
  for (const f of readdirSync(wasmDir)) {
    if (f.includes('nosimd') || f.includes('module')) rmSync(resolve(wasmDir, f));
  }
}

mkdirSync(dirname(zipPath), { recursive: true });
rmSync(zipPath, { force: true });
// 압축은 어디에나 있는 파이썬 zipfile 모듈로(폴더 안의 내용이 zip 최상위에 오도록)
execFileSync('python3', ['-c', `
import os, sys, zipfile
src, dst = sys.argv[1], sys.argv[2]
with zipfile.ZipFile(dst, 'w', zipfile.ZIP_DEFLATED) as z:
    for base, _, files in os.walk(src):
        for f in files:
            p = os.path.join(base, f)
            z.write(p, os.path.relpath(p, src))
`, out, zipPath], { stdio: 'inherit' });
console.log('확장 프로그램 zip:', zipPath);

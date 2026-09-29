import { defineConfig } from 'vite';

export default defineConfig({
  // 상대 경로로 빌드해야 GitHub Pages 하위 경로(/IRIS/)에서도 그대로 동작한다.
  base: './',
  build: { target: 'es2022' },
});

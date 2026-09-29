import { defineConfig } from 'vite';

// 크롬 확장 프로그램 빌드: 같은 웹 앱(index.html) + 서비스 워커(background.js)
export default defineConfig({
  base: './',
  build: {
    outDir: 'dist-ext',
    emptyOutDir: true,
    target: 'es2022',
    rollupOptions: {
      input: {
        index: new URL('./index.html', import.meta.url).pathname,
        background: new URL('./extension/background.ts', import.meta.url).pathname,
      },
      output: {
        entryFileNames: (chunk) => (chunk.name === 'background' ? 'background.js' : 'assets/[name]-[hash].js'),
      },
    },
  },
});

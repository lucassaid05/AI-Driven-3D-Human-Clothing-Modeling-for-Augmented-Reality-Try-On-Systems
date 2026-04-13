import { defineConfig } from 'vite';
import { resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = fileURLToPath(new URL('.', import.meta.url));

export default defineConfig({
  root,
  base: './',
  publicDir: 'public',
  server: {
    open: '/apps/ar-tryon/',
    port: 5173,
  },
  build: {
    outDir: 'dist',
    emptyOutDir: true,
    rollupOptions: {
      input: {
        'ar-tryon': resolve(root, 'apps/ar-tryon/index.html'),
      },
    },
  },
  resolve: {
    alias: {
      '@': resolve(root, 'src'),
    },
  },
});

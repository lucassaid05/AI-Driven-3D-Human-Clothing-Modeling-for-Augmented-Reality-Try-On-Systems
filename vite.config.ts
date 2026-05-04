import { defineConfig } from 'vite';
import { resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import basicSsl from '@vitejs/plugin-basic-ssl';

const root = fileURLToPath(new URL('.', import.meta.url));

/** Python server-local.py (GLB generation, garment list, etc.). Override with VITE_DEV_API_ORIGIN if needed. */
const devApiOrigin = process.env.VITE_DEV_API_ORIGIN ?? 'http://127.0.0.1:8000';

export default defineConfig({
  root,
  base: './',
  publicDir: 'public',
  // Self-signed HTTPS so mobile browsers allow getUserMedia on https://<LAN-IP>:5173 (camera blocks plain http:// on LAN).
  plugins: [basicSsl()],
  server: {
    // Listen on all interfaces so phones on the same Wi‑Fi can open https://<PC-LAN-IP>:5173/
    host: true,
    // Vite 5 rejects Host headers that are not localhost unless allowed — LAN IPs get 403 without this.
    allowedHosts: true,
    open: '/apps/ar-tryon/',
    port: 5173,
    strictPort: false,
    headers: {
      'Permissions-Policy': 'camera=(self), microphone=(self), fullscreen=(self), display-capture=(self)',
    },
    proxy: {
      // Same-origin fetches from the UI use location.origin; Vite has no /api — forward to the Python server.
      '/api': { target: devApiOrigin, changeOrigin: true },
      '/generated': { target: devApiOrigin, changeOrigin: true },
    },
  },
  preview: {
    host: true,
    allowedHosts: true,
    port: 4173,
    strictPort: false,
    headers: {
      'Permissions-Policy': 'camera=(self), microphone=(self), fullscreen=(self), display-capture=(self)',
    },
    proxy: {
      '/api': { target: devApiOrigin, changeOrigin: true },
      '/generated': { target: devApiOrigin, changeOrigin: true },
    },
  },
  build: {
    outDir: 'dist',
    emptyOutDir: true,
    rollupOptions: {
      input: {
        'ar-tryon': resolve(root, 'apps/ar-tryon/index.html'),
      },
      output: {
        // Stable names so Python/static servers can link /dist/apps/ar-tryon/ without editing HTML each build.
        entryFileNames: 'assets/ar-tryon.js',
        chunkFileNames: 'assets/ar-tryon-[name]-[hash].js',
        assetFileNames: 'assets/ar-tryon[extname]',
      },
    },
  },
  resolve: {
    alias: {
      '@': resolve(root, 'src'),
    },
  },
});

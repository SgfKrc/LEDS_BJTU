import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  server: {
    host: '127.0.0.1',
    port: 5180,
    proxy: {
      '/healthz': 'http://127.0.0.1:8090',
      '/v1': 'http://127.0.0.1:8090',
    },
  },
  build: { outDir: 'dist', sourcemap: true },
});

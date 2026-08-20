import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// 与既有 frontend 共用同一个 FastAPI 后端；端口错开避免并行开发时冲突。
const apiTarget = process.env.QLH_VITE_API_TARGET || 'http://localhost:8000'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5174,
    proxy: {
      '/api': {
        target: apiTarget,
        changeOrigin: true,
      },
    },
  },
  build: {
    outDir: 'dist',
    sourcemap: false,
  },
})

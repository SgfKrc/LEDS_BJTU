import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

const apiTarget = process.env.QLH_VITE_API_TARGET || 'http://localhost:8000'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: apiTarget,
        changeOrigin: true,
      },
    },
  },
})

import tailwindcss from '@tailwindcss/vite'
import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    // Forward /api/* to Flask so the browser sees one origin (no CORS in dev).
    proxy: { '/api': 'http://127.0.0.1:5000' },
  },
})

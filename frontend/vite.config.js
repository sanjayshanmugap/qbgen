import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// https://vite.dev/config/
export default defineConfig({
  base: '/',
  plugins: [react()],
  server: {
    proxy: {
      '/process_clues': {
        target: 'http://localhost:8080',
        changeOrigin: true,
      },
      '/get_sets': {
        target: 'http://localhost:8080',
        changeOrigin: true,
      },
      '/process_set_clues': {
        target: 'http://localhost:8080',
        changeOrigin: true,
      },
      '/generate_apkg': {
        target: 'http://localhost:8080',
        changeOrigin: true,
      },
    },
  },
  build: {
    outDir: 'dist',
  },
});

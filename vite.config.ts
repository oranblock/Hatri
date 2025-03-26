import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import { resolve } from 'path';

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      '@': resolve(__dirname, 'src'),
    },
    dedupe: ['react', 'react-dom'] // Explicitly dedupe React to avoid duplicates
  },
  build: {
    outDir: 'dist',
    sourcemap: true,
    rollupOptions: {
      input: {
        main: resolve(__dirname, 'index.html'),
      },
    }
  },
  server: {
    port: 3000,
    open: true,
    cors: true,
  },
  // Optimize dependencies to prevent hooks errors
  optimizeDeps: {
    include: [
      'react', 
      'react-dom', 
      '@tensorflow/tfjs', 
      '@tensorflow-models/coco-ssd',
      '@tensorflow-models/blazeface',
      'lucide-react'
    ],
    force: true // Force dependency pre-bundling
  },
});
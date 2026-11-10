import { sveltekit } from '@sveltejs/kit/vite';
import { defineConfig } from 'vite';
export default defineConfig({
  plugins: [sveltekit()],
  server: { host: 'localhost', port: 5800, strictPort: true,
    proxy: { '/api': { target: process.env.GR8_CONSOLE_BACKEND || 'http://localhost:8800' } } }
});

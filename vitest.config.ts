import { defineConfig } from 'vitest/config';

// Separate from vite.config.ts (whose root is the web UI): tests live next to the sources.
export default defineConfig({
  test: {
    root: '.',
    include: ['src/**/*.test.ts'],
    exclude: ['src/web/**', '**/node_modules/**'],
  },
});

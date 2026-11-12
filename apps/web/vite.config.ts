import { defineConfig } from 'vite';

// Cross-origin isolation lets the page share memory between workers, which
// the multi-threaded search needs. Without these headers the bots fall back
// to one thread.
const isolation = {
  'Cross-Origin-Opener-Policy': 'same-origin',
  'Cross-Origin-Embedder-Policy': 'require-corp',
};

export default defineConfig({
  server: { headers: isolation },
  preview: { headers: isolation },
  // The search threads are workers started from inside the bot worker.
  worker: { format: 'es' },
});

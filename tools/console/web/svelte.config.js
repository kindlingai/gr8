import adapter from '@sveltejs/adapter-static';

/** @type {import('@sveltejs/kit').Config} */
const config = {
  kit: {
    // Static SPA build: FastAPI serves web/build with an index.html fallback,
    // so there is no node server in production — just uvicorn.
    adapter: adapter({ fallback: 'index.html' })
  }
};

export default config;

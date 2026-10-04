import { defineConfig } from "vite";

export default defineConfig({
  base: "./",
  server: { port: 5173, strictPort: true, host: "127.0.0.1" },
  build: { target: "es2022", outDir: "dist", assetsInlineLimit: 0 },
  test: { include: ["tests/**/*.test.ts"] },
});

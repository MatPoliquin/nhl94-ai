import { defineConfig } from "vitest/config";

export default defineConfig({
  base: "./",
  worker: { format: "es" },
  test: { include: ["tests/*.test.ts"] },
  server: {
    fs: { strict: true },
    headers: { "X-Content-Type-Options": "nosniff" },
  },
});

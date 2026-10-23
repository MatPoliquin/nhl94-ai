import { defineConfig } from "vitest/config";
import { fileURLToPath } from "node:url";

export default defineConfig({
  base: "./",
  worker: { format: "es" },
  build: {
    rolldownOptions: {
      input: {
        arena: fileURLToPath(new URL("./index.html", import.meta.url)),
        bots: fileURLToPath(new URL("./bots.html", import.meta.url)),
        leaderboards: fileURLToPath(new URL("./leaderboards.html", import.meta.url)),
        help: fileURLToPath(new URL("./help.html", import.meta.url)),
        about: fileURLToPath(new URL("./about.html", import.meta.url)),
      },
    },
  },
  test: { include: ["tests/*.test.ts"] },
  server: {
    fs: { strict: true },
    headers: { "X-Content-Type-Options": "nosniff" },
  },
});

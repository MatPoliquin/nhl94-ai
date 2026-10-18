import { access } from "node:fs/promises";
import { fileURLToPath } from "node:url";

for (const name of ["genesis.js", "genesis.wasm", "LICENSE.txt", "source.tar.gz", "build.json"]) {
  try {
    await access(fileURLToPath(new URL(`../public/core/${name}`, import.meta.url)));
  } catch (error) {
    throw new Error("WASM core artifacts are missing. Run npm run build:core before building the site.", { cause: error });
  }
}

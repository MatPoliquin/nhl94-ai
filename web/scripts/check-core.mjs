import { access, readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";

for (const name of ["genesis.js", "genesis.wasm", "LICENSE.txt", "source.tar.gz", "build.json"]) {
  try {
    await access(fileURLToPath(new URL(`../public/core/${name}`, import.meta.url)));
  } catch (error) {
    throw new Error("WASM core artifacts are missing. Run npm run build:core before building the site.", { cause: error });
  }

  const metadata = JSON.parse(await readFile(new URL("../public/core/build.json", import.meta.url), "utf8"));
  if (metadata.stateAbi !== "little-endian-lp64" || metadata.browserMemory !== "wasm32-lowered") {
    throw new Error("The core does not support the bundled native save. Run npm run build:core.");
  }
  await access(new URL("../public/states/SabresVsMightyDucks.ManualGoalie.Start.state", import.meta.url));
}

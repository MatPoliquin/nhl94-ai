import { createHash } from "node:crypto";
import { cp, mkdir, readFile, stat, writeFile } from "node:fs/promises";
import { availableParallelism } from "node:os";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { spawnSync } from "node:child_process";

const root = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const revision = "f66bcb1e780acdcced95a05e04550bb8a91c1850";
const sdkVersion = "3.1.74";
const cache = join(root, ".cache");
const source = join(cache, `Genesis-Plus-GX-${revision}`);
const archive = join(cache, "genesis-source.tar.gz");
const output = join(root, "public/core");

function run(command, args, cwd = root) {
  const result = spawnSync(command, args, { cwd, stdio: "inherit" });
  if (result.error) throw result.error;
  if (result.status !== 0) {
    throw new Error(`${command} failed with status ${result.status}`);
  }
}

const compiler = spawnSync("emcc", ["--version"], { encoding: "utf8" });
if (compiler.error || compiler.status !== 0) {
  throw new Error("emcc is missing. Activate Emscripten 3.1.74; see README.md.");
}
if (!compiler.stdout.includes(` ${sdkVersion} `)) {
  throw new Error(`This build requires Emscripten ${sdkVersion}.`);
}

await mkdir(cache, { recursive: true });
await mkdir(output, { recursive: true });
try {
  await stat(join(source, "Makefile.libretro"));
} catch (error) {
  if (error.code !== "ENOENT") throw error;
  const url = `https://codeload.github.com/libretro/Genesis-Plus-GX/tar.gz/${revision}`;
  console.log(`Downloading pinned core ${revision}`);
  const response = await fetch(url);
  if (!response.ok) throw new Error(`Core download failed: HTTP ${response.status}`);
  await writeFile(archive, new Uint8Array(await response.arrayBuffer()));
  run("tar", ["-xzf", archive, "-C", cache]);
}

const patch = join(root, "native/core-globals.patch");
const patchHash = createHash("sha256").update(await readFile(patch)).digest("hex");
const patchStamp = join(source, ".arena-globals-patch");
try {
  const applied = await readFile(patchStamp, "utf8");
  if (applied !== patchHash) throw new Error("Core patch changed. Remove the named core cache and rebuild.");
} catch (error) {
  if (error.code !== "ENOENT") throw error;
  const patchedFiles = ["libretro/osd.h", "core/membnk.h", "libretro/libretro.c"];
  run("tar", [
    "-xzf", archive, "-C", cache,
    ...patchedFiles.map((file) => `Genesis-Plus-GX-${revision}/${file}`),
  ]);
  for (const file of patchedFiles) {
    const path = join(source, file);
    await writeFile(path, (await readFile(path, "utf8")).replaceAll("\r\n", "\n"));
  }
  run("patch", ["--batch", "--no-backup-if-mismatch", "-p1", "-i", patch], source);
  await writeFile(patchStamp, patchHash);
}

run("make", [
  "--silent", "--always-make", "-f", "Makefile.libretro", `-j${Math.min(availableParallelism(), 8)}`,
  "platform=emscripten", "CC=emcc", "AR=emar", "HAVE_CHD=0",
  "TARGET=genesis.a",
  "HAVE_SYS_PARAM=1", "CPPFLAGS=-sUSE_ZLIB=1 -fno-common", `GIT_VERSION= ${revision.slice(0, 12)}`,
], source);

run("emcc", [
  join(root, "native/bridge.c"), join(source, "genesis.a"),
  join(source, "libretro/libretro-common/streams/file_stream.c"),
  join(source, "libretro/libretro-common/streams/file_stream_transforms.c"),
  `-I${join(source, "libretro")}`, `-I${join(source, "libretro/libretro-common/include")}`,
  "-O2", "--no-entry",
  "-sUSE_ZLIB=1", "-sMODULARIZE=1", "-sEXPORT_ES6=1",
  "-sENVIRONMENT=web,worker,node", "-sALLOW_MEMORY_GROWTH=1",
  "-sINITIAL_MEMORY=33554432", "-sMAXIMUM_MEMORY=134217728", "-sSTACK_SIZE=1048576",
  "-sFILESYSTEM=1", "-o", join(output, "genesis.js"),
]);

await cp(join(source, "LICENSE.txt"), join(output, "LICENSE.txt"));
const bridgeHash = createHash("sha256")
  .update(await readFile(join(root, "native/bridge.c"))).digest("hex");
await writeFile(join(output, "build.json"), JSON.stringify({
  core: "Genesis Plus GX", revision, emscripten: sdkVersion, bridgeHash, patchHash,
  scope: "local smoke test; not Classic; not verified netplay",
}, null, 2) + "\n");

// Include the upstream and wrapper sources alongside the compiled core.
run("tar", [
  "--exclude=*.o", "--exclude=*.bc", "--exclude=*.a",
  "--exclude=*.orig", "--exclude=*.rej", "-czf", join(output, "source.tar.gz"),
  "-C", cache, `Genesis-Plus-GX-${revision}`,
  "-C", root, "native/bridge.c", "native/core-globals.patch", "scripts/build-core.mjs", "package.json", "README.md",
]);
console.log(`Built local core in ${output}`);

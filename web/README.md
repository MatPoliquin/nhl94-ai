# NHL94 Browser Lab

A **local emulator smoke test**, not the Classic AI port and not a netplay release.
The app is TypeScript/Vite; the emulator is Genesis Plus GX compiled to
single-threaded WebAssembly with a small C/libretro bridge.

## Run locally

Requirements: Node.js 22.12+ (or a supported newer release), npm, Emscripten
**3.1.74**, make, patch, tar and internet access for the first dependency/core
download. Activate your Emscripten SDK before building:

```bash
source /path/to/emsdk/emsdk_env.sh
# From the nhl94-ai repository:
cd web
npm ci
npm run build:core
npm run dev -- --port 5173 --strictPort
```

Open <http://127.0.0.1:5173/>. The development server binds only to localhost.
It does not require Python, stable-retro, a neural inference runtime or an account.

If you need an SDK, install/activate version 3.1.74 following
[the Emscripten instructions](https://emscripten.org/docs/getting_started/downloads.html).
The SDK can also supply Node.js. Do not install a second global Node runtime just
for this app if the activated SDK already supplies a suitable version.

The build downloads the pinned core into `.cache/`, not the sibling stable-retro
checkout. A small reproducible patch turns two legacy header-level global
definitions into extern declarations with one definition each. Sources, patch
and license accompany the generated core under `public/core/`.

## Try the smoke test

1. Select your legally obtained, uncompressed original NHL94 Genesis ROM.
2. Press **Play**. Navigate the game's normal menus using controller 1.
3. For the diagnostic opponent, select a two-player game. Once on the rink,
   enable **Diagnostic bot on the other controller**.
4. Use the controller dropdown to choose which physical port your keyboard or
   gamepad controls. The game itself still determines home/away assignment.
5. Pause, step exactly one native frame, inspect RAM or restart from ROM boot.
6. Download a replay; load it again and press Play to reproduce its input stream.

Supported ROM SHA-256:

```text
ccc0298c118d580863c877abd470c71186ef0a1b60f1e7c6ff1e5a38f48fd053
```

Raw `.bin`, `.md` and `.gen` files with this content are accepted. ZIP, interleaved
SMD, other revisions and roster hacks are rejected rather than decoded using
potentially wrong RAM offsets. No ROM is supplied, uploaded or stored persistently.

Controls:

| Input | Genesis action |
| --- | --- |
| Arrow keys | D-pad |
| Z / X / C | A / B / C |
| Enter | Start |
| Standard gamepad X / A / B | A / B / C |
| Standard gamepad Start | Start |

On-screen buttons provide the same inputs. Only standard-mapped gamepads are
supported. Sound is opt-in and starts from a user gesture. Hiding the tab pauses
the simulation; returning does not automatically resume it.

## Current boundaries

- The diagnostic bot chases the puck and pulses C. **It is not Classic.**
- Both controllers execute at native-frame cadence. The normal playback target
  is the core-reported frame rate, approximately 59.92 Hz, not display refresh.
- ROM restart creates a fresh core instance. Replay input starts from ROM boot,
  not from the current screen, and is tied to the exact ROM/WASM build hashes.
- Recordings are bounded to 108,000 native frames; replay files to 4 MiB.
- The RAM checksum is diagnostic, not a canonical complete-state checksum.
- Importing foreign save states is not exposed in the UI.
- No external bot scripts, TensorFlow.js models, online rooms or WebRTC yet.
- Worker isolation keeps emulator work off the UI thread; it is not a security
  sandbox for untrusted bot JavaScript.
- This is a local development build, not a public security-audited release.

## Validate

```bash
npm test
npm run build

# ROM-dependent WASM replay/snapshot tests:
NHL94_ROM=/absolute/path/to/your/nhl94.bin npm test

# Browser tests, after installing the test browsers once:
npx playwright install chromium firefox
NHL94_ROM=/absolute/path/to/your/nhl94.bin npm run test:browser
```

Without `NHL94_ROM`, the ROM-dependent tests are explicitly skipped. Tests do not
copy the ROM into public assets or require it in the repository.
The ROM integration test compares two WASM instances over 10,000 input frames
and tests local snapshot continuation. Browser tests exercise Chromium/Firefox
file validation, the real worker, controller inputs, pause/step and replay.
Browser tests build and exercise the production static assets, not just the Vite
development server. This is not yet proof of complete cross-browser gameplay-state
determinism.

`npm run build` checks that core artifacts exist, type-checks, then builds static
assets into `dist/`. It does not deploy anything. After TypeScript/CSS edits,
Vite reloads automatically; rebuild the core only after native/build changes.
Generated assets, dependencies, caches and test output are ignored by Git.

Genesis Plus GX has a
[non-commercial license](https://github.com/ekeeke/Genesis-Plus-GX/blob/master/LICENSE.txt)
and source/notice requirements. Review those obligations before public deployment
or monetization.

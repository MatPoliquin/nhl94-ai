# NHL94 AI CONTEST

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

### Site pages

Top-right links open the information pages in a new tab so the arena's loaded
match is preserved (hiding its tab still pauses playback):

- [Bots](bots.html): available browser bots and descriptions, generated from
  the same catalogue as the controller selectors.
- [Help](help.html): ROM loading, controllers, input setup, playback, stats and replays.
- [About](about.html): research purpose, EA non-affiliation, contact and community links.

The Vite multi-page build emits `index.html`, `bots.html`, `help.html` and
`about.html`. Links and assets remain relative for GitHub Pages project subpaths;
these pages do not require a backend or client-side router.

The shared palette is defined by CSS custom properties in [style.css](style.css):
deep navy/slate surfaces, ice-blue actions, red Away accents, mint ready states
and amber warnings. It does not recolor the emulated game image.

Titles and section headings use locally bundled Barlow Condensed Bold/SemiBold
from the pinned Fontsource package; body text and controls retain the system font.
Font files are served with the site, not requested from a font CDN. The
[SIL Open Font License](public/fonts/barlow-condensed-LICENSE.txt) is included
in the static build.

If you need an SDK, install/activate version 3.1.74 following
[the Emscripten instructions](https://emscripten.org/docs/getting_started/downloads.html).
The SDK can also supply Node.js. Do not install a second global Node runtime just
for this app if the activated SDK already supplies a suitable version.

The build downloads the pinned core into `.cache/`, not the sibling stable-retro
checkout. A small reproducible patch turns two legacy header-level global
definitions into extern declarations with one definition each. Sources, patch
and license accompany the generated core under `public/core/`.

The core uses the native little-endian LP64 save layout and Emscripten
`MEMORY64=2` lowering to ordinary 32-bit browser memory. This matches the supplied
native save's pointer-sized sound/CPU records without requiring browser Memory64
support. The public wrapper still exposes numeric memory handles.

The default state is copied unchanged to
[`public/states/SabresVsMightyDucks.ManualGoalie.Start.state`](public/states/SabresVsMightyDucks.ManualGoalie.Start.state).
Its compressed SHA-256 is pinned in [start-state.ts](start-state.ts); missing,
changed or incompatible state data fails explicitly rather than falling back to
ROM boot. Review permissions for distributing game-derived save assets before
publishing them.

## Try the smoke test

1. Select your legally obtained, uncompressed original NHL94 Genesis ROM.
2. Step 2 lists four controller slots. Controller 1 defaults to **Human / Team 1
   (Home)** and controller 2 to **Diagnostic bot / Team 2 (Away)**.
   Each enabled slot offers **Human**, the available bots, then **None**, with a
   Home/Away team selector alongside it. Controllers 3 and 4 are greyed out as
   None until four-player support is implemented.
3. In **Step 3: Input device**, use the keyboard/on-screen controls or select a
   gamepad. The same panel contains gamepad setup and live input diagnostics.
4. Loading the ROM restores Sabres vs Mighty Ducks with manual goalies.
   Press **Play** to begin; normal menu navigation is not required.
5. Pause, step exactly one native frame, inspect Game Stats or restart the match.
6. In **Step 4: Inspect and replay**, download a replay; load it again and press
   Play to reproduce its input stream.

Controller setup changes pause playback. With one Human slot, keyboard, on-screen
buttons and the selected gamepad all control that slot. With two Human slots,
keyboard/on-screen input controls slot 1 and the selected gamepad controls slot 2.
Each bot slot has its own bot instance, evaluated against the same pre-frame state.

Home/Away choices directly update the native joystick/team/selected-skater
assignments and player control flags, following the Python controller setup
semantics. None releases its controller; a team with no assigned controller uses
the native CPU. Same-team slots select different players. The supplied manual
goalie options are preserved.

For Bot vs Bot, select a bot in both slots and press Play. No Human slot is needed
to navigate menus. Replay playback ignores and disables live controller selectors
and uses the recorded physical-port inputs and assignment events. Replay
download/import is in Step 4.

New recordings use `nhl94-browser-inputs-v2`, identifying the compressed starting
state and recording every controller change with its frame number, including
multiple changes while paused. Starting the match presents one neutral native
initialization frame to restore video before recorded frame zero. Restart repeats
this initialization. Compatible v1 recordings still start from ROM boot; the
existing exact ROM/core-build checks reject recordings from incompatible builds.

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

On-screen buttons provide the same inputs. Sound is opt-in and starts from a user
gesture. Hiding the tab pauses the simulation; returning does not automatically
resume it.

### Gamepads and 8BitDo M30 receivers

The Step 3 input-device panel detects all gamepads exposed by the browser, even before a
ROM is loaded. Standard-mapped pads use the defaults above. Non-standard pads,
including receivers that expose a raw layout, must be configured explicitly:

1. Click the page and press a controller button so the browser can expose it.
2. Select it under **Gamepad device**.
3. Expand **Gamepad setup and live inputs**. Raw button indices and axis values
   should change when you use the controller.
4. Click **Set up buttons / D-pad** and follow the eight prompts: Up, Down,
   Left, Right, A, B, C, Start. Release between presses. D-pad buttons or
   positive/negative axis directions can be learned.
5. Check the decoded **Genesis input** and then play. Custom mappings last for
   the current tab and can also override standard mappings.

Disconnecting the selected pad releases its inputs and pauses live playback;
the app does not silently switch to another connected device. Keyboard and
on-screen controls remain available.

Linux recognizing the device does not guarantee that Firefox exposes it through
the Gamepad API. If the panel says **No gamepad exposed by the browser**, focus
the page and press a button first. In Firefox, check that
`dom.gamepad.enabled` is enabled in `about:config`; privacy/fingerprinting
settings may also restrict exposure. Try a regular Firefox/Chrome window rather
than an embedded webview. Device detection and raw-layout support are separate
issues; the panel distinguishes both without assuming a particular M30 mode.

### Game Stats

The Home/Away table directly below Bot Configuration replaces the raw RAM panel.
It updates from the same native team records used by NHL94's pause-menu Game
Statistics screen:

- Score, shots on goal and shooting percentage.
- Power-play goals/opportunities, time and shots; shorthanded goals.
- Breakaway, one-timer and penalty-shot goals/attempts.
- Faceoffs won, body checks, penalties/minutes and attack-zone time.
- Completed/attempted passes and passing percentage.

Times are `minutes:seconds`; percentages use the native integer calculation.
These are the ROM's counters, not inferred button presses or bot decisions.
Home/Away remains tied to physical teams even when controller routing changes.
They are cumulative match totals, not individual player or period summaries.
Before a ROM is loaded, values are unavailable (`-`); replay and restart use the
emulator's corresponding totals.

## Current boundaries

- The diagnostic bot chases the puck and pulses C. **It is not Classic.**
- **Bot Configuration**, below the on-screen controls, previews 60/15/10/6 Hz
  frequency choices, AI Goalie and Consider player ratings. These controls are
  disabled placeholders and do not affect bot execution.
- Both controllers execute at native-frame cadence. The normal playback target
  is the core-reported frame rate, approximately 59.92 Hz, not display refresh.
- Match restart creates a fresh core instance and restores the bundled start.
  Replays identify their starting state and are tied to exact ROM/WASM hashes.
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

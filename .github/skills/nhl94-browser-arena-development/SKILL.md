---
name: nhl94-browser-arena-development
description: "Develop and debug the NHL94 browser arena in web/. Use for Genesis Plus GX WASM builds, the C/libretro bridge, TypeScript emulator workers, RAM observations, keyboard/gamepad mapping including 8BitDo M30, native-frame timing, input replay determinism, Python Classic-to-TypeScript parity, WebRTC bot-versus-bot lockstep, TensorFlow.js adapters or GitHub Pages asset loading. Distinguish implemented smoke-test features from planned capabilities."
---

# NHL94 browser arena development

Use this workflow for browser emulator and AI integration. Routine styling can
follow [web instructions](../../instructions/web.instructions.md) without the
full emulator workflow. For Python-only training or tactical changes, use the
existing [task](../nhl94-task-development/SKILL.md) or
[Classic](../nhl94-classic-ai-development/SKILL.md) skill instead.

## 1. Establish the affected contract

Read the relevant portion of the [browser README](../../../web/README.md) and
[package scripts](../../../web/package.json). Inspect the actual runtime path
before changing it. Do not assume a previous session's paths or test results
describe the current checkout.

| Surface | Source map |
| --- | --- |
| UI, pacing, lifecycle and audio | [main](../../../web/main.ts), [HTML](../../../web/index.html), [styles](../../../web/style.css), [audio](../../../web/audio.ts) |
| Worker requests, failures and ownership | [client](../../../web/client.ts), [protocol](../../../web/protocol.ts), [worker](../../../web/emulator.worker.ts) |
| Frame stepping, snapshots, memory and native callbacks | [engine](../../../web/engine.ts), [C bridge](../../../web/native/bridge.c) |
| Pinned core, Emscripten and source distribution | [core build](../../../web/scripts/build-core.mjs), [core patch](../../../web/native/core-globals.patch), [artifact check](../../../web/scripts/check-core.mjs) |
| Trusted default save, native team ownership and replay initialization | [start state](../../../web/start-state.ts), [native bridge](../../../web/native/bridge.c), [startup tests](../../../web/tests/start-state.test.ts), [browser replay parity](../../../web/tests/browser/start-state.spec.ts) |
| Browser observations, native team stats and ROM assumptions | [observations](../../../web/observations.ts), [game stats](../../../web/game-stats.ts), [ROM validation](../../../web/rom.ts) |
| Python Genesis reference semantics | [platform backend](../../../nhl94_ai/game/platforms/genesis.py), [RAM declarations](../../../nhl94_ai/game/ram.py), [game state](../../../nhl94_ai/game/state.py) |
| Controller masks, keyboard, gamepad detection and setup | [inputs](../../../web/inputs.ts), [gamepad panel](../../../web/gamepad.ts) |
| Bot boundary, controller slots/team routing and replay validation | [bot](../../../web/bot.ts), [controllers](../../../web/controllers.ts), [replay](../../../web/replay.ts) |
| Focused checks | [contract tests](../../../web/tests/contracts.test.ts), [WASM tests](../../../web/tests/core.test.ts), [browser tests](../../../web/tests/browser/smoke.spec.ts), [browser config](../../../web/playwright.config.ts) |

At initial creation this app is a local smoke test with a diagnostic bot.
Classic, online play and TensorFlow.js are not implemented. Verify current
sources before changing that description; never relabel the diagnostic bot as
Classic or treat future design as tested behavior.

For a runtime defect, capture a focused failing case first. Record only the
necessary ROM/core identifiers, controller assignment, frame/input sequence,
browser and failure. Store private fixtures, measurements and run history outside
the repository; keep reusable tests with their implementation.

## 2. Preserve simulation and memory semantics

- One step applies both physical controller masks to one native frame. Use the
  core-reported frame rate, not display refresh, as the simulation clock.
- Keep bot observation cadence separate from tactical decisions and rendering.
  Preserve button holds, releases and fresh edges across pause, restart and
  replay. Do not change cadence to conceal latency or performance problems.
- Verify physical port, physical team and selected-player slot independently.
  Controller 1 is not unconditionally the home team; joystick count is not
  skater count. Check both assignments.
- The current LSB_FIRST WASM core exposes word-swapped work RAM. Verify
  offsets, byte order, signedness and units against Python Genesis semantics
  and known values. Do not apply CPU-address big-endian reads blindly to the
  emulator heap or duplicate offsets without a contract test.
- Preserve heap lifetime and bounds across memory growth; do not hold stale
  typed-array views. Validate frame counts, controller masks and local-file
  formats before crossing the native boundary.
- Treat native/browser save compatibility as unproven until tested. Restore bot
  history/RNG/recurrent state too when adding match continuation; an emulator
  snapshot alone is not a complete bot snapshot.
- The bundled Sabres/Ducks save requires the native LP64 record layout. The build
  uses `MEMORY64=2` to preserve it while lowering to 32-bit browser memory.
  Keep the state hash, one-frame startup initialization and recorded controller
  events consistent across restart and replay; do not silently replay save-based
  recordings from ROM boot or ignore ownership changes made while paused.
- RAM checksums are diagnostic, not complete-state digests. Matching checksums
  or two Node WASM instances do not prove cross-browser gameplay determinism
  or honest bot execution.

## 3. Build and verify only the changed surfaces

Work from repository-root `web/`. Activate the SDK using its real installed path;
do not encode a session-specific SDK path in source or instructions. Use the
Node/Emscripten versions specified by the README and build script. Never silently
upgrade the pinned core, toolchain or patch to make a failure disappear.

Choose the smallest relevant commands; these are selectors, not an instruction
to install dependencies or run every suite for every edit:

```bash
# Controller, observation, bot or replay contract changes:
npm test -- tests/contracts.test.ts

# TypeScript/type checking and production static assets:
npm run build

# Native/core changes or missing generated core assets:
npm run build:core
NHL94_ROM=/absolute/path/to/local/nhl94.bin npm test -- tests/core.test.ts

# UI/worker integration; runs a production build and Chromium/Firefox tests:
NHL94_ROM=/absolute/path/to/local/nhl94.bin npm run test:browser
```

If dependencies are missing or manifests changed, restore the declared
dependencies using the package manager and lockfile. Install test browsers only
if missing. Without `NHL94_ROM`, ROM-dependent tests are skipped, not passed.
Do not download, bundle or commit a commercial ROM to unblock validation.

The current core test compares 10,000 input frames across two WASM instances,
checks observations/video and tests snapshot continuation. Extend its scenarios
for the changed behavior; boot/menu traces alone do not cover full gameplay,
goalies, both teams or stoppages.

For UI changes, use browser tools to exercise the affected controls and failure
states. For gamepad changes, inspect `navigator.getGamepads()` after user input
and distinguish no browser-exposed device from an unconfigured non-standard
mapping. Use the live gamepad panel and explicit button/D-pad setup for raw
devices; an M30's label/mode is not evidence that its buttons match the default
layout. Test press/release and disconnect; report
hardware testing as unavailable when no physical controller is exposed.

If starting a local dev server, verify the chosen port is responsive. Reuse an
existing task server rather than replacing an unrelated process. Keep long-lived
servers only when needed for user iteration; clean up temporary test servers.

## 4. Apply the feature-specific gate

### Classic TypeScript port

Use the [Python agent interface](../../../nhl94_ai/agents/base.py) and
[Classic behavior guide](../../../docs/CLASSIC.md). Freeze/fingerprint the Python
controller, helpers and configuration used as the reference. Port state decoding,
movement, passing, finishing, defense, goalie handling, scheduling and action
lifecycles, not a simplified puck chase under the same name.

Replay identical recorded observations through Python and TypeScript; compare
every native-frame button output, including both sides, reset, held-input edges,
goalies and interruptions. Then compare integrated emulator traces. Explain
mismatches before changing the reference. Similar scores do not establish parity.
Do not change Python tactics while debugging a structural port.

### Remote bot input exchange

This is planned behavior, not an existing networking module. Keep each user's
bot local; exchange validated session/frame-numbered inputs, not bot code, ROMs,
models or arbitrary save states. Verify identical ROM/core/options/starting state
before ready/start. Both copies must apply the same two inputs to the same frame.

Start with explicit lockstep waiting and bounded messages, frame windows and
queues. Make timeout, disconnect and desync visible; do not substitute neutral
input, a CPU opponent or a successful result. Measure WAN throughput before
promising normal playback speed. Batching, input delay or rollback require
explicit semantics and tests, not silent changes to bot cadence.

Static Pages hosting does not supply signaling or TURN. Manual offer/answer
exchange still may need STUN/TURN for connectivity. Never embed reusable relay
secrets; disclose peer-IP exposure. Peer results remain unverified even if
replays agree. A Worker is not a sandbox for hostile JavaScript.

### TensorFlow.js policies

Add only when requested; preserve a runtime-independent bot interface.
Define a supported conversion path because current PyTorch/SB3 artifacts do not
load directly into TensorFlow.js. Match observation ordering, normalization,
action postprocessing, decision cadence, reset and recurrent state; reject
unsupported contracts.

Compare inference and decoded actions against recorded Python references on the
chosen browser backend. Keep emulator determinism separate from inference
numerics: online peers need to agree on recorded inputs, not rerun each other's
models. Do not add a neural runtime to a UI-only task.

### Static deployment

Build from `web/` and publish `web/dist/`, not TypeScript source or the Python
package. Test repository-subpath asset URLs, worker/WASM loading and incompatible
cached builds. Keep generated outputs ignored and preserve the core's required
notices and corresponding sources.

The baseline is single-threaded WASM; do not require pthreads, SharedArrayBuffer
or custom response headers without verifying the hosting requirements. Do not
commit, push or deploy just to validate a local change unless requested.

## 5. Report the evidence and limits

Summarize the affected contract, commands actually run and observed results.
Separate unit success, native/browser parity, visual checks, physical hardware
testing and live two-network testing. State skipped or unavailable evidence
explicitly; do not describe one as proof of another.

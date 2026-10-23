# Controller and display troubleshooting

Use with the [browser arena skill](../SKILL.md). Start from observable signals
and fix the layer where they first diverge; do not change bot tactics or emulator
configuration to conceal an input or presentation problem.

## Source map

| Layer | Source and regression checks |
| --- | --- |
| Browser exposure, selected device and setup | [gamepad panel](../../../../web/gamepad.ts), [browser tests](../../../../web/tests/browser/gamepad.spec.ts) |
| Automatic presets and saved custom mappings | [profiles](../../../../web/gamepad-profiles.ts), [profile tests](../../../../web/tests/gamepad-profiles.test.ts) |
| Button/axis decoding and Genesis masks | [inputs](../../../../web/inputs.ts), [contract tests](../../../../web/tests/contracts.test.ts) |
| Human/bot slots, physical ports and native ownership | [controllers](../../../../web/controllers.ts), [main](../../../../web/main.ts), [worker](../../../../web/emulator.worker.ts), [engine](../../../../web/engine.ts), [bridge](../../../../web/native/bridge.c) |
| Frame transport, canvas sizing and full-image checks | [protocol](../../../../web/protocol.ts), [styles](../../../../web/style.css), [smoke tests](../../../../web/tests/browser/smoke.spec.ts) |

## 1. Establish the failing layer

Record the browser, ROM/core/start identity if relevant, controller assignments,
Play/Pause state, device ID, browser mapping mode, button/axis counts and one
failing physical action. Preserve existing ROMs, states and working mappings.

Trace:

```text
Browser Gamepad API
  -> selected device and decoded Genesis input
  -> Human slot and physical-port routing
  -> worker-applied masks
  -> native selected player and visible response
```

- No device in `navigator.getGamepads()`: exposure/focus/API problem. A Linux
  input device or a device working in another tab does not prove this page sees
  it. Activate the arena and press a button before changing OS permissions.
  A hidden VS Code browser can expose no hardware while regular Firefox does.
- Raw buttons/axes change, but Genesis input stays neutral: mapping or active
  setup problem. Inspect the chosen profile, not emulator ownership.
- Genesis input changes, but the game does not respond: inspect Play/Pause,
  routing and applied inputs before changing bindings.
- Native player coordinates or game state change, but the action looks absent:
  verify the selected team/player and complete rendered image before concluding
  that inputs failed.

The website's Play and the emulated Genesis Start button are different. Setup,
configuration changes and hidden tabs pause frontend playback. Finish setup and
press Play explicitly; do not silently resume a hidden or previously paused match.

## 2. Verify mappings, not controller names

Browser-standard layouts map automatically. The supported raw M30 / 6B receiver
preset is guarded by USB identity `0ca3:0024`, 10 buttons and 3 axes. Its physical
A/B/C/Start buttons are 0/1/5/9; D-pad directions use axes 0/1. Read
[the profile implementation](../../../../web/gamepad-profiles.ts) for the current
authority. Do not generalize this layout to every M30 mode, OS, receiver or other
controller sharing the same USB ID. In particular, C is not button 2 on this
verified receiver.

- Ask for raw values or the existing working mapping block, rather than making
  the user repeat the eight-step wizard to diagnose one button.
- Snapshot Gamepad/GamepadButton fields explicitly. Real browser properties can
  be prototype getters; object spread may not copy them.
- Verify fresh presses, holds, releases, opposite axes, neutral values and
  unsupported hat-axis encodings. Do not infer directions from arbitrary
  out-of-range values or turn Mode/unused axes into Start.
- If the browser cannot expose the physical device, distinguish mock tests from
  hardware evidence. If local hardware capture is needed, first verify the
  selected gamepad's identity. Use a bounded, non-exclusive read of that device,
  announce readiness before asking for one physical press, ignore initialization
  events and close the helper afterward. Do not capture keyboards, grab the
  controller exclusively or change input permissions to obtain a mapping.

## 3. Preserve profile and reconnection behavior

Custom mappings override both standard mappings and automatic presets. Profiles
are stored locally by device ID, browser mapping mode and button/axis counts,
not the temporary gamepad index. Check reload and index-change/reconnection
behavior, and ensure a different layout does not inherit the old mapping.

Reset removes the saved override and restores an automatic preset if available.
Corrupt or blocked storage must log and show a warning; a current-tab mapping
may still work, but must not be presented as successfully persisted. Do not
silently clear other site data or conceal a profile validation failure.

Disconnect must pause and release the selected input. Do not silently switch to
another controller. Reset/setup changes must not leave stale held inputs.

## 4. Verify the actual native input

With one Human slot, keyboard/on-screen and the selected gamepad are combined
for that slot. With two Human slots, keyboard/on-screen targets slot 1 and the
gamepad targets slot 2. With no Human slot, those inputs are ignored. None releases
that controller; teams with no assigned controller return to the game's CPU.
A bot slot does not become a Human input sink.

Check the requested team, native physical port and selected skater independently.
Home is not synonymous with port 1. Inspect the worker's acknowledged frame
packet inputs or exported physical-port replay masks. Wait for the requested
frame acknowledgement: a packet from before an asynchronous Step request often
contains zero inputs and is not evidence that the new input was dropped.

Test Human versus Classic, not only Human versus the diagnostic bot. Require a
mapped direction to reach the correct port, move the actual selected player and
release to neutral. Verify A/B/C and Start masks separately; a live input label
or a recorded mask alone does not prove the intended native action occurred.

## 5. Separate framebuffer defects from CSS clipping

Compare native width/height and framebuffer pixels with the canvas and its
container's `getBoundingClientRect()` bounds. If the HUD exists in the pixels
but falls below the visible container, fix layout rather than RAM/VDP offsets,
save contents or video callback cropping.

The game can switch between 256- and 320-pixel native widths. A canvas with
100% width/height inside an aspect-ratio grid can enlarge its automatic grid
track through its intrinsic minimum size and then be clipped by overflow.
Keep the zero-minimum grid tracks and canvas minimum sizes in
[the screen styles](../../../../web/style.css); preserve full-image containment.

Verify all four frame edges, score and timer after ROM load and video-size
changes, at desktop and mobile widths. An initially empty or 320-wide canvas is
not a sufficient test of a 256x224 saved-match frame.

## 6. Run focused checks

From `web/`, select the smallest applicable checks:

```bash
npm test -- tests/gamepad-profiles.test.ts tests/contracts.test.ts tests/controllers.test.ts
npm run build
NHL94_ROM=/absolute/path/to/nhl94.bin \
  npx playwright test tests/browser/gamepad.spec.ts tests/browser/smoke.spec.ts
```

Run a production build before preview-based browser tests. Report skipped
ROM/hardware checks explicitly. Keep physical button confirmation, synthetic
Gamepad tests, native input/player movement and visual HUD checks distinct.

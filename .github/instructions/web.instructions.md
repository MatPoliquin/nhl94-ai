---
description: "Use when editing the NHL94 browser website: TypeScript/Vite UI, Genesis Plus GX WASM, controller input, bot adapters, replays, browser tests or static deployment."
applyTo: "web/**"
---

# NHL94 browser application

- Keep this independent application in repository-root `web/`, not inside the
  Python package. Use [package.json](../../web/package.json) and its lockfile for
  browser dependencies; keep Python training/UI imports out of the app.
- Start with the [browser README](../../web/README.md) for current capabilities,
  prerequisites and commands. For emulator/AI workflows, use
  [the browser arena skill](../skills/nhl94-browser-arena-development/SKILL.md).
- Preserve one-native-frame stepping, simultaneous controller inputs, held-button
  and release edges, and the distinction between physical ports and home/away
  teams. Display pacing must not change simulation or bot decision cadence.
- Keep controller masks, observation schemas and replay formats explicit and
  validated. Make intentional incompatibility visible rather than guessing.
- Surface worker, bot, file and audio failures in the UI. Keep cancellation and
  recovery responsive; do not silently replace invalid input or failed bots.
- The diagnostic bot is not Classic. Classic, online play and TensorFlow.js
  must not be presented as implemented until their runtime paths are verified.
- Keep ROMs, credentials, dependencies, caches and generated build outputs out
  of commits. Preserve license notices and corresponding source distribution.
- Keep the baseline compatible with static Pages hosting and repository-subpath
  URLs. Do not add a backend, deployment, threading requirement or external
  service without a requirement that calls for it.
- Run the smallest relevant checks from `web/`: focused Vitest tests and
  `npm run build`; use `npm run test:browser` for UI/worker integration.
  Set `NHL94_ROM` only to a legally supplied local file when emulator behavior
  needs validation. Explicitly report skipped ROM tests and untested hardware.
- Rebuild WASM when native sources/build settings change or core assets are
  missing, not for ordinary TypeScript/CSS edits. Restore dependencies only
  after manifest changes or a missing-dependency failure.
- Validate user-visible changes with the browser, including errors and controls,
  not only a successful build. Documentation-only edits do not need app tests.

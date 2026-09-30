# Inherited behavior kept separate from the refactor

- `StochasticFrameSkip` stops early on termination but continues its repeated
  steps on truncation. The public Gym return still distinguishes the flags.
  `test_frame_skip_stops_on_truncation` is an expected failure documenting the
  desired behavior. Fixing the timing is a separate behavior change.
- The older PvP display references unqualified `FB_WIDTH`/`FB_HEIGHT` when creating
  its surface. That constructor can raise `NameError`. The active single-player
  debug view is separate. This inherited issue has a local lint annotation.
- Legacy custom MLP classification and self-play observation handling retain their
  original limitations. The old self-play opponent encoder uses its default input
  configuration; arbitrary custom observation groups and temporal opponents need
  a separate compatibility/timing change before being relied on.
- Export supports Box observations; optional vision/Mamba backends and all possible
  network/action combinations have not been exhaustively exercised.

Smoke tests establish successful execution and structural equivalence, not playing
strength or convergence. The frozen traces in `tests/fixtures/traces.json` provide
a fixed five-case comparison set: three variants with filtered actions, full-team
multidiscrete no-op controls, and full-team hockey intents. They record seed 7,
start states, frame skip 4, actions, rewards, and observation hashes. The opponent
is the game's built-in AI. Use longer benchmark runs for tactical changes.

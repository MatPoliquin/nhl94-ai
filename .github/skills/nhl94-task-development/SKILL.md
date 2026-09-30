---
name: nhl94-task-development
description: >
  Develop and debug NHL94 training rewards, PPO training health, live replay,
  randomized episode starts, and RAM-backed environment behavior. Use for
  DefenseZone or similar tasks, misleading statistics or stale live models,
  implausible starting states, possession or controller bugs, and frame-skip
  or action-timing questions.
---

# NHL94 training-task development

Use the existing task and ROM contracts before inventing new reset machinery.
Keep implementation in `nhl94_ai/`; do not modify the separate stable-retro or
runtime repositories, or regenerate `models/`, unless explicitly requested.
Answer design questions without editing gameplay when that is the requested scope.

## Start with the relevant source, not a repository-wide search

Paths below are relative to the repository root. Read only the rows needed for
the task, then follow specific symbols.

| Question | Start here |
| --- | --- |
| Current reward, terminal priority, shared self-play behavior | `nhl94_ai/tasks/defensezone.py`, `nhl94_ai/tasks/selfplay.py`, `nhl94_ai/tasks/legacy.py`, `nhl94_ai/tasks/registry.py` |
| Coherent procedural reset and per-instance RAM writes | `nhl94_ai/tasks/defense_setup.py`; reusable sampling in `nhl94_ai/tasks/setup.py` |
| Reset order, reward cadence, action repetition, episode statistics | `nhl94_ai/env/observation.py`, `nhl94_ai/env/factory.py`, `nhl94_ai/env/wrappers.py` |
| Possession, control, field ordering, normalization | `nhl94_ai/game/state.py`, `nhl94_ai/env/encoding.py`, `nhl94_ai/game/specs.py` |
| Buttons, intents and press/release sequences | `nhl94_ai/env/actions.py`, `nhl94_ai/env/intents.py` |
| Target-only positioning and controller diagnostics | `nhl94_ai/env/target_control.py`, `nhl94_ai/ui/targets.py`; Architecture's target-position section |
| Hyperparameters and checkpoint restoration | `configs/training/nhl94.json`, `nhl94_ai/config.py`, `nhl94_ai/models/factory.py`, `nhl94_ai/artifacts.py` |
| Target Gaussian noise, squashing and entropy | `nhl94_ai/models/mlp.py`, `configs/training/defense-target-hyperparams.json`, `tests/unit/test_squashed_policy.py` |
| PPO logs, evaluation sample size and model freshness | `nhl94_ai/training/live.py`, `nhl94_ai/training/events.py`, `nhl94_ai/ui/live.py`, `configs/training/defense-target.json` |
| Reported episode totals and replay/checkpoint regressions | `nhl94_ai/evaluation/metrics.py`, `tests/unit/test_episode_statistics.py`, `tests/unit/test_live_snapshots.py`, `tests/unit/test_target_overlay.py` |
| Shot-counter decoding and misleading high words | `nhl94_ai/game/ram.py::decode_shot_count`, `nhl94_ai/game/state.py::Team.begin_frame`, `tests/unit/test_episode_statistics.py::ShotDecodingTests` |
| Existing test fixtures and reset assertions | `tests/unit/test_defense_setup.py`, `tests/unit/test_defensezone.py`, `tests/unit/test_environment_contracts.py`, `tests/ram_fixture.py` |

Read [Architecture](../../../docs/ARCHITECTURE.md) for the current task contract
and [Artifacts](../../../docs/ARTIFACTS.md) before changing a model/input schema.
For ROM details, jump to the relevant sections of the
[NHL94 deep dive](../../../docs/nhl94%20Deep%20dive.md):
2.2 object layout, 2.3 flag byte order, 3 button semantics, 6.1 assignments,
and 12 integration mapping pitfalls. Do not read the entire disassembly just to
rediscover these. Consult the cited ROM routines only when a necessary detail
is not established by the local reference or implementation.

## Establish the behavior being changed

1. Identify the actual game, save name, controller count, task, action type and
   resolved model configuration. `num_players` counts controllers, not skaters.
2. Define valid initial ownership, geometric constraints, terminal outcomes and
   same-frame reward priority. DefenseZone has only possession, shot and goal
   rewards; do not add positional shaping unless explicitly requested.
3. Change only the requested dimensions. A reset fix does not automatically call
   for a curriculum, new observations, different tactics or another action space.
4. Preserve the task's shared callers, including self-play. Verify each supported
   variant; do not assume a full-team RAM layout applies to reduced-player games.

## Reuse the verified RAM/reset workflow

- Use the configured training interpreter/virtualenv. Missing NumPy or Gymnasium
  in system Python does not prove the project's environment is unavailable.
  Check an existing environment before installing dependencies; use `.[dev]`
  and the relevant extras when restoration is actually needed.
- Create a bounded diagnostic environment through `env.factory.make_retro`;
  close it in `finally`. Do not launch a long training run just to inspect a reset.
- Raw Retro `reset()` may return an empty info dictionary. After reset, read
  `env.data.lookup_all()` for the saved RAM values. Inspect a field's actual
  address/type with `env.data.get_variable(name)` before interpreting it.
- For scoped reset writes, the existing initializer uses
  `env.data.memory.extract(address, type)` and
  `env.data.memory.assign(address, type, value)`. These operate on this emulator
  instance without editing integration files or adding hundreds of observation
  aliases. Keep address/type knowledge localized rather than duplicating it.
- Sample related positions jointly: carrier, reachable goal-side defender,
  support and goalies. Bound rejection sampling and raise on exhaustion instead
  of returning overlapping or out-of-bounds fallback positions.
- Check the base save's live roles, possession, controller assignment, period and
  clock before writing. A filename is not proof of a compatible game situation.

### Pitfalls already established in this repository

| Pitfall | Required check |
| --- | --- |
| Moving only integer X/Y retains inconsistent physics | Reset full fixed-point current **and previous** positions, height and full-word velocities. Legacy velocity observations are only high bytes. Reuse `_place_object`. |
| Repositioned players or goalies launch despite zero initial velocity | Rebuild all 16 objects' `Ylist`, `OOlistpos` and `OOlist` via `_rebuild_object_order`. Collision candidates use cached positions while impulses use current positions. |
| Clearing too much assignment scratch breaks collision geometry | Scratch ends at `+0x49`. Preserve radii `+0x4A/+0x4C`; clear wall-contact components `+0x4E/+0x50` separately. |
| A receiver carries the puck while spinning in support AI | Preserve one `assnearest` chain per team. The initial carrier needs `asspuckc -> assnearest -> role`, not just carrier -> role. Follow real possession changes after reset. |
| Teleporting a player leaves a shot, fall or stale CPU target active | Reinitialize verified animation/contact/assignment state; preserve team/direction flags, roles, roster and attributes. Never clear an entire player record blindly. |
| Starting a fresh carrier assignment can immediately pass or shoot | Reuse the carrier initialization in `init_full_team_defense`, including its normal awareness-based decision delay. Verify ownership after the wrapper's reset frame, not only immediately after RAM writes. |
| Changing a controller slot can leave a stale encoded star for a frame | Compare the actual controller slot with decoded control after reset. The current initializer retains controller selection and varies which skater fills the primary-defender role. |
| A controller-slot word is negative | This is no selection (`FFFF` or byte-cleared `FFxx`), not a goalie or negative roster index. Preserve the target schema; clear control flags/undefined relative features and wait neutrally for the ROM. |
| Goalie slots and aliases differ between variants | Full-team goalies occupy 5/11. Inspect live role words and field metadata before adapting to 1-on-1 or 2-on-2; even a named goalie field can be misaddressed. |
| Visual stars can describe stale touch/possession state | Prefer authoritative owner slots when available. Negative owners mean no carrier; the legacy decoder also uses `-1` for an absent field, so check raw field presence when distinguishing missing data from a loose puck. |
| A shot animation is not a recorded shot | Compare decoded `Stats.shots` with `last_stats.shots`, not C or a persistent shooter flag. `game.ram.decode_shot_count` strips the legacy unrelated high word at the data boundary; do not repeat masks in rewards. |
| Live totals grow once per reset without actual events | Reuse `reset_team_totals` to read saved-counter baselines through VecEnv wrappers, before the first action. Preserve first-action/terminal events. The defensive save already contains one home pass. |
| Clock and flag observations can be composite words | The legacy clock can include the period in its high word. A 68000 bit operation on memory tests a byte; a bit at a word's starting address appears in the high byte of a big-endian word observation. |
| Gym seeding alone leaves ROM randomness unchanged | Drive procedural sampling and the ROM RNG seed from the environment RNG. Check repeatability across resets, as well as variation across seeds. |
| Requested buttons are mistaken for completed actions | Button observations report submitted input, not successful ROM actions. Holding C is not repeated fresh presses; existing boost intents do not automatically pulse it. |
| A world target drifts ahead of the scrolling ice | Reuse `ui/targets.py` and the wrapper's `target_camera_x/y`, derived from queued scroll values sampled before the emulator step. Current `Hpos/Vpos` can describe a future frame. See Architecture for the 256x224 cropped-viewport projection. |

## Check timing, attribution and compatibility

- The observation wrapper computes reward once per emulator frame, then calls
  `EndFrame`. Frame skipping sums those rewards and stops on termination.
  Test an event is charged once, not once per repeated action.
- `Monitor` is inside frame skipping in the training factory: episode lengths
  count emulator frames, while SB3 timesteps count policy/environment decisions.
  Account for early termination before multiplying by the configured frame skip.
- DefenseZone recovery requires a friendly skater: goalie catches/holds give no
  recovery bonus and are nonterminal unless a new opponent shot, goal or clock cutoff occurs.
  Shots and goals end the episode with exactly -1 and override same-frame
  possession; skater recovery without either gives +1. Detect shots from the
  decoded counter delta, not animation flags or the saved counter baseline.
  Test shot/catch/recovery priority and exact termination across frame skips.
  A high recovery reward can still come from an autonomous teammate, including
  one receiving a goalie outlet. Compare matched starts with passive/scripted
  baselines and distinguish controlled-player contribution from team success.
- MLP inputs retain fixed roster slots with control flags. Preserve field
  ordering, not just observation length. Keep editable training defaults and
  `nhl94_ai/data/training/` copies synchronized.
- Model-specific overrides take precedence over common hyperparameters.
  Checkpoint loading restores saved architecture/PPO settings; editing JSON does
  not necessarily retune a resumed model. Do not silently repurpose old weights.
- `TARGET_POSITION` holds a two-float target for the policy interval, not fixed
  buttons: its controller executes every emulator frame. Preserve its versioned
  controller observation block and check `tests.integration.target_control`;
  B requests switching but does not guarantee the desired skater.
- Target bounds belong to `TaskDefinition`. DefenseZone maps outputs directly
  to `x=-120..120, y=-270..-88`; use the shared coordinate conversion helpers
  rather than assuming every task scales Y by 270. Bounds are an artifact contract.

## Diagnose training health and live replay

- Separate optimizer stability from useful positioning. Reasonable KL and
  clipping do not establish improvement over the passive/scripted baselines.
  Check evaluation sample size before interpreting reward swings or a best score:
  with five episodes, swapping one +1 recovery for a -1 concession shifts the
  mean by 0.4 before other penalties. Use a larger, fixed held-out seed set for
  policy comparisons rather than treating a lucky maximum as convergence.
- Check the loaded action distribution before interpreting `train/std`.
  Legacy Gaussian targets are clipped to `[-1, 1]`; the target example now uses
  `SquashedMlpPolicy`, tanh after sampling, and initial pre-squash std 0.5.
  Its bounded entropy uses fresh reparameterized samples, not raw Gaussian
  entropy or old rollout actions. Large latent means/noise can still saturate
  targets at boundaries. Preserve log-probability corrections and verify
  rollout/replay/save-load parity when changing the distribution.
  Account for stochastic training versus deterministic evaluation before
  interpreting reward gaps. Loading old weights retains their saved distribution.
- The live viewer follows `_latest_live`, refreshed at each evaluation even
  when reward is lower or tied. `_best_live` still uses strict score improvement;
  the final checkpoint is separate. Do not reconnect replay to the best path.
  Latest means the latest evaluated snapshot, not every optimizer update.
- For stale replay, trace `latest_model_path`, `model_version`, and the displayed
  `loaded_model_timesteps`. Publish the version only after ZIP and sidecar saves
  complete, and keep viewer loads under the same shared lock. Preserve metadata
  validation, target-controller reset/cadence, and headless runs without extra
  viewer snapshots. Use `test_live_snapshots` for lower/tied-score reloads,
  best-checkpoint preservation, failed loads, and session wiring.

## Validate the real reset boundary

Start with the affected contracts, selecting only relevant modules:

```bash
python -m unittest tests.unit.test_defense_setup tests.unit.test_defensezone tests.unit.test_environment_contracts -v
```

For emulator-backed changes, run the explicit ROM checks with locally installed
ROMs. The first exercises actual reset observations, not just sampled geometry.
The physics check follows motion and CPU ownership handoffs after reset.
The smoke check verifies unrelated gameplay traces remain unchanged; do not
use `--record` to make a regression pass.

```bash
python -m tests.integration.defense_starts
python -m tests.integration.defense_physics
python -m tests.integration.smoke
```

Assert valid ownership, nonterminal starts, spacing, a reachable goal-side
defender, goalie placement, actual/encoded control agreement, variation and
seed reproducibility. For reward changes, also cover same-frame events,
frame skips 1/4/10, reset counter baselines and shared defense task names.
Use `tests/ram_fixture.py` for overlapping typed RAM writes in unit tests, but
do not treat a fake-memory result as proof of ROM behavior.

Finish implementation work with the repository checks:

```bash
python -m unittest discover -s tests -v
python -m pylint nhl94_ai tests
```

Document intentional behavior shifts and save-state constraints in the existing
architecture/task docs. Report missing dependencies or ROMs explicitly. Reset
invariants passing does not establish that the resulting policy learns reliably.

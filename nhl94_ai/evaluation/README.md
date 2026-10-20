# Evaluation and gameplay tools

This package runs games, measures agents, records native inputs, and compares
decisions from restored emulator states. It contains source code, not generated
evaluation results. Reports and recordings belong in the output directory
selected by the caller.

## Where to start

| Purpose | Modules | Entry point |
| --- | --- | --- |
| Play a game | `play.py` | `nhl94 play` |
| Evaluate an agent or checkpoint | `runner.py`, `session.py` | `nhl94 evaluate`; `session.py` supports legacy run configuration |
| Scripted agent comparisons | `benchmark.py` | `nhl94 benchmark` |
| Genesis CPU and teammate benchmarks | `cpu_benchmark.py`, `teammates.py` | `nhl94 benchmark-cpu`, `nhl94 benchmark-teammates` |
| SNES policy comparisons | `snes_benchmark.py` | `nhl94 benchmark-snes` |
| External opponents | `external.py` | `nhl94 external` |
| Genesis input recording and replay | `human_session.py`, `human_recording.py`, `human_replay.py` | `nhl94 record-human`, `nhl94 replay-match` (`replay-human` is an alias) |
| SNES input recording and replay | `snes_session.py`, `snes_recording.py`, `snes_replay.py` | `nhl94 record-snes`, `nhl94 replay-snes` |
| Existing offense and goalie policies | `offense_policy.py`, `goalie_policy.py` | `nhl94 offense`, `nhl94 goalie` |

Use `--help` on a command to inspect its options. Most standalone experiment
runners also support `python -m nhl94_ai.evaluation.<module> --help`.

## Experiments and measurements

- `carry_*`, `chance_creation_*`, `possession_ablation_replay.py`, and
  `refinement_replay.py` examine carrying and offensive decisions.
- `receiver_replay.py`, `reception_replay.py`, `one_timer_*`, `rebound_replay.py`,
  and `cross_crease_*` examine passes, receptions, and coordinated finishes.
- `shot_direction_*`, `shot_placement_*`, `chance_selection_*`, and `retention*`
  collect or evaluate alternatives and existing decision models.
- `goalie.py` collects goalie starts; `goalie_outcomes.py` measures their outcomes.
- `fixture_regression*`, `lifecycle_replay.py`, and `carry_replay_gate.py` check
  fixtures, controller history, and native replay cadence.
- `*_metrics.py`, `*_outcomes.py`, `offense_attempts.py`, and `opportunity.py`
  aggregate events and outcomes. `frames.py` supports visual comparisons;
  `playback_profile.py` profiles playback.

Replay experiments compare decisions from the same emulator state and controller
history. Their bounded outcomes answer local tactical questions; they are not
automatically measures of full-game strength. Keep each runner's sampling rules,
termination conditions, protocol version, and limitations with that runner.

## Shared code

- [`replay_support.py`](replay_support.py) owns Genesis CPU period setup,
  native stepping, verified snapshot restoration, observation serialization,
  replay source fingerprints, and compact JSON/gzip report writing.
- [`../game/genesis_match.py`](../game/genesis_match.py) owns Genesis match RAM
  declarations, joystick selection, roster decoding, and home/away state views.
  Gameplay environments import these helpers directly.
- `benchmark.py` and `cpu_benchmark.py` retain their historical helper imports.
  `refinement_replay.py` still exposes `advance`, `read_view`, and `restore`;
  `possession_ablation_replay.py` still exposes `observation` and `source_hashes`.
  New callers should import shared operations from their owning modules.

Source fingerprints must cover shared helpers as well as experiment code.
`replay_source_hashes` includes nested game/platform modules and accepts extra
paths relative to `nhl94_ai`. When moving measured code, update explicit source
lists and replay gates too. Existing report structures and command/module paths
are kept stable; fingerprints change when their source changes.

## Compatibility and outputs

`compare.py` intentionally reports that the retired train-then-compare workflow
is unavailable. The model evaluators remain useful for existing checkpoints even
while training is disabled. `lifecycle_history.py` supports replaying archived
controller implementations. These files are not unused simply because normal
gameplay does not call them.

Generated `__pycache__` files are disposable and ignored by Git. Recordings,
model artifacts, and local reports have separate ignored destinations; do not
store new experiment outputs alongside these source modules.

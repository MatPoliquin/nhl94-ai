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

## Matched teammate experiments

`python -m nhl94_ai.evaluation.teammate_benchmark` snapshots source, runs the
single/two/three-bot CPU matrix in a fresh interpreter, and compares paired
results. For example:

```bash
python -m nhl94_ai.evaluation.teammate_benchmark snapshot --output tmp/team-baseline-source
python -m nhl94_ai.evaluation.teammate_benchmark snapshot --output tmp/team-harness
python -m nhl94_ai.evaluation.teammate_benchmark run --source tmp/team-baseline-source --harness tmp/team-harness --output tmp/team-baseline --seed 20000 --count 20 --periods 3 --seconds 300 --frame-skip 4 --trace
python -m nhl94_ai.evaluation.teammate_benchmark compare --before tmp/team-baseline/report.json --after tmp/team-candidate/report.json --output tmp/team-comparison.json
```

Use different immutable directories for every candidate. `--variant` accepts a
JSON object of tactical options for team profiles only. Duration, cadence,
starting save, seed lists and controller layout cannot be overridden there.
`--periods`, `--seconds`, `--frame-skip` and `--seeds` apply to every profile,
including the one-controller control (`strategy=single`, runner `team_size=2`).
Explicit seed lists may be longer than twenty. A resumed job must match its
complete policy, harness, runtime, seed and settings contract. Scored support,
off-puck movement, intent arbitration, defensive support, primary race policies,
position-based three-bot roles, and
`--teammate-cue-anticipation` are separate
opt-in experiments. Cue anticipation times the bot receiver's one-timer cue from
the shared pass intent's predicted contact instead of native release
confirmation. The shared
Classic single-bot profile receives none of these options. The public gameplay
and single-match teammate evaluator parsers expose matching `--teammate-*` flags; movement
requires scored support and experimental variants require coordinated control.

Coordinated receiving preserves a released pass while all controller selections
are temporarily unknown, with neutral input during the gap and unchanged expiry,
interception and confirmed-selection-loss handling. This correctness fix is on
by default. `--no-teammate-transient-receiving` disables only gap preservation
for comparison; `--teammate-transient-receiving` explicitly enables it.
Single and independent modes do not use shared receiving.

Receiving-controller ownership always follows the settled native selection
table. A one-timer cue is recorded only after its C
edge is applied to the intended receiver; suppressed requests do not consume
the cue. Native `one-timer-armed` feedback is recorded separately, including
CPU receivers that acquire a controller when the game arms their shot.
Applied cues are not automatically retried when native arming is absent.
These lifecycle checks do not certify the contact forecast or playing strength.

The report separates live possession, requested B inputs, applied B/C edges,
native goalie owner exits/receptions, sampled modeled pass opportunities, and
native one-timer attempts. Baselines recorded before intent provenance existed
report unclassified edges; do not infer intent from possession or held labels.
Dangerous possession entries are a geometry proxy. Paired intervals use only
complete, matching seeds/settings/assets within a controller layout. The
study must declare its own seed cap, margins and multiplicity policy before
examining candidate results. Install `.[evaluation]` (or `.[dev]`) for SciPy
Student-t quantiles. `compare --unit game --confidence 0.9979166666666667`
uses raw game totals and Bonferroni coverage for 24 estimates; `--unit per120`
is separately labeled and scales the displayed thresholds to preserve the same
decision. The default Python comparison remains compatible with historical
per-120-second screens. Regulation acceptance thresholds apply only to games
with three 300-second periods; short screens are labeled separately.

Policy packages load from `--source`; all evaluator modules load from the
separately pinned `--harness`. Neither snapshot is rewritten. Reports include
native totals by period, and `*.events.jsonl.gz` stores bounded pre-release
and flight observations for tracked passes, including successes and failures.
Event endings can occur beyond the retained window and are labeled accordingly.
`teammate_study` reports cross-layout context separately from policy pairing,
keeps game-level pass denominators, and provides variance-only sample sizing.
Periods and individual passes are never counted as independent seeds.

`teammate_trace_compare.py` validates entire traces and records the first input
and diagnostic differences. Later live trajectory differences are expected
after an intended input change. Exact single-bot input and final-RAM comparisons
remain a separate gate. Input-only replay tests determinism, not policy parity.

## Compatibility and outputs

`compare.py` intentionally reports that the retired train-then-compare workflow
is unavailable. The model evaluators remain useful for existing checkpoints even
while training is disabled. `lifecycle_history.py` supports replaying archived
controller implementations. These files are not unused simply because normal
gameplay does not call them.

Generated `__pycache__` files are disposable and ignored by Git. Recordings,
model artifacts, and local reports have separate ignored destinations; do not
store new experiment outputs alongside these source modules.

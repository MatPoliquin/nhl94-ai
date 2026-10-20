# Architecture

The repository contains the `nhl94_ai` Python package at its root. `models/`
holds model assets; `nhl94_ai/models/` contains network definitions. The browser
application is a separate TypeScript project in `web/`.

| Package | Responsibility | Extension point |
| --- | --- | --- |
| `game` | Variants, RAM decoding, geometry, and features | `specs.GAMES` |
| `env` | Environment assembly, observations, actions, and timing | `factory.build_single_nhl94_env`, `encoding.ObservationEncoder` |
| `tasks` | Episode setup, rewards, completion, and restrictions | `registry.register_task` |
| `agents` | Scripted and learned gameplay controllers | `Agent.reset`, `Agent.act` |
| `models` | Neural policy definitions | `factory.register_model` |
| `training` | PPO, collection, imitation, and curriculum | `service.train`, `curriculum.PHASE_RUNNERS` |
| `evaluation` | Evaluation, playback, comparison, and external opponents | `runner.evaluate`, `external.play_match` |
| `ui` | Optional display and recording | Consumers of state and training events |
| `cli` | Command dispatch and configuration | `nhl94 --help` |

Game decoding does not import UI or training code. Display and export dependencies
load only when requested. Environment bindings are per-instance objects.

`GameSpec.platform` selects the console backend in `game/platforms/`. Both
consoles use `NHL94GameState` and `ClassicAIV1Model`; backend hooks own telemetry
registration, player/engine decoding, rink dimensions, action encoding, movement
projection, goalie mechanics, and episode clock semantics. Profiles are selected per environment,
without mutating global constants. Genesis retains its public button ordering,
observation normalization, and ROM-derived forecasts. SNES uses native array
ownership/selection and signed word velocities, with a separate normalization
identifier and explicitly estimated motion/pass forecasts. Its integration
scenario is packaged with this project; ROMs and save states stay in stable-retro.

SNES supports full-team Classic `FILTERED` `PostPlay` against CPU on either side,
human versus Classic, and Classic versus Classic. Reset writes native menu,
joystick/player assignment, and team cursor tables before advancing the supplied
save. Gameplay uses controller inputs and native selection/animation feedback;
the shared goalie controller receives console-specific takeover and save mechanics.
Two scripted agents keep independent schedulers and world-coordinate team views.
Configuration validation rejects unmapped training tasks, RNG, learned artifacts,
and Genesis goalie experiments.

## Services

`training.service.train(config)` accepts an options dictionary, `RunConfig`, or
resolved namespace and returns a `TrainingResult`. Headless and live modes share
the environment factory, trainer, and callbacks.

`evaluation.runner.evaluate(agent, config, env=env)` returns rewards, lengths,
termination and truncation flags, and budget status. The caller owns and closes
the supplied environment.

`training.collect.collect_demonstrations(agent, config)` writes a dataset and its
metadata. Supplied agent instances use one worker and reset between episodes;
registered teachers support worker processes. BC and DAgger are available through
`training.bc.train_bc` and `training.dagger.run_dagger`.

`training.curriculum.run_curriculum(path, dry_run=False)` resolves phase settings
and chains artifacts. `nhl94 external` loads an explicitly requested adapter;
ordinary gameplay has no dependency on an external adapter package.

## Configuration

Precedence is parser defaults, JSON options, then explicit CLI arguments.
Declared paths inside an options file resolve relative to that file. CLI paths
resolve relative to the current working directory. `--state` is an emulator save
name. Boolean options in JSON must be booleans.

Editable examples are in `configs/`. Built-in training defaults are package data
under `nhl94_ai/data/training/`; keep those copies synchronized when changing
defaults. Dependencies and optional extras are declared in `pyproject.toml`.

`GameSpec` distinguishes skaters per team from `num_players`, which counts
controllers. `EnvironmentConfig` validates supported game, action, goalie, and
self-play combinations before emulator workers start.

## Model input variants

`--model_input` selects ordered observation groups from the model-input registry.
The selected layout and normalization are recorded with the checkpoint. Equal
tensor sizes do not imply compatible layouts. Loading a checkpoint restores its
saved architecture; changing a configuration file does not resize a loaded model.

The `pvg` layout is a compact player-goalie observation contract. Use `pvg-v3`
for its earlier layout and `default` or `legacy` for their respective full-team
layouts. [Artifact contracts](ARTIFACTS.md) describe compatibility checks.

## Timing and reset

The inner environment processes input, advances one emulator frame, refreshes
state, computes task rewards and completion, and encodes observations. Frame-skip
wrappers repeat policy actions around that loop. Macro timers use native frames.

Classic's frame dispatcher observes each native frame independently of its
tactical interval. `agents/scheduling.py` owns those clocks and cached input;
`agents/lifecycle.py` owns shot, pass, and one-timer transitions. Standalone
callers use `predict_frame(state, frame_skip=4)` per native frame to match normal
dispatch. See [Classic](CLASSIC.md).

`agents.base.configure_scripted_frames` attaches reactive scripted agents to
the frame-skip wrapper for evaluation and collection. Demonstrations record
the actual substeps; learned button policies retain their configured interval.
Reset clears controller history, pending actions, and recurrent policy state.

`team_vs_game` extends the two-controller button path to native Genesis
Two-Teammates play. `game.ram.configure_teammates` owns scoped reset assignments
and automatic-goalie settings. `teammate_controller_info` supplies an authoritative
selection for one controller without changing physical team identities or the
other controller's fields. The legacy composite clock is replaced only in these
views by its native low-word clock. Existing neural input layouts are unchanged.

`agents.teammates.TeammateCoordinator` observes the same RAM snapshot through two or three
independent game states, keeps controller execution histories separate, assigns
pursuit/support roles and emits 24 or 36 bot buttons before the next native frame. Native
pass feedback reserves a controlled receiver; input requests are distinct from
completed passes and native one-timer attempts. Gameplay selection changes use
controller buttons. `evaluation.teammates` runs bounded seeded CPU comparisons
and records/replays optional per-frame input traces from a verified reset boundary.

Executors persist across role changes. Support frames observe fresh feedback and
commit the buttons actually applied, without generating a discarded Classic
decision. A shared pass commitment records native release, actual receiver,
interception point, expiry and reception/shot phase. Controlled receivers reserve
input ownership; one-timer cues require estimated contact with the native shot
pose. `--teammate-support-options` opts into bounded reachable support options
and goal-side recovery coverage in playback or teammate benchmarks. Support
forecasts advance the receiver, carrier, teammates, defenders, goalies and puck
to a common estimated release time before evaluating the pass flight. The
receiver follows the same turning/braking steering used in execution; forecasts
of other actors hold observed steering and do not predict new CPU decisions.
Staying and coasting are explicit options, with bounded target persistence and
immediate invalidation when possession or receiver availability changes.
These tactics remain confined to the opt-in teammate mode. Benchmark `--state`
also accepts an absolute path to an externally prepared, gzip native save; its
initial state and RAM hashes are included in the report.
Reception-to-shot evaluation retains observed possession through a bounded carry
window and allows delayed shot counters to settle. Normal releases use native
animation, ownership and puck-motion evidence; one-timers use native attempt
counters. Confirmed shot follow-through is excluded from ordinary turnovers.

RAM decoding belongs in `game/ram.py` and `game/state.py`. Task reset helpers
belong in `tasks/setup.py` and the relevant task initializer. Changes to ordered
observations or action semantics require matching schema version changes.

Python's coordinated teammate mode also uses `agents.teammate_pressure.SharedPressure`
against an opposing skater carrying the puck. Three bots jointly assign `contain`,
`close`, and `cover`; two bots can assign both attacking roles only if other
on-ice teammates retain observed coverage. The planner evaluates the whole group
so that a deeper covering skater does not unnecessarily replace the nearest
attacker. Coverage holds its existing depth and adjusts to the carrier-to-net line.

Traps start in neutral/attacking ice or bounded board-side situations, with
separate approach targets and an initial containment phase before contact.
Every native frame rechecks coverage, dangerous receivers, player availability,
arrival estimates, and predicted collisions over a 24-frame horizon. Motion
forecasts are estimates; uncontrolled teammates receive only observed-motion
credit. Advanced receivers require existing shot-lane cover. Unsafe or stalled
traps return to ordinary pressure/coverage, and passes, loose pucks, or selection
changes invalidate the shared plan. Assignments have a short stability lease,
but safety exits take precedence.

`ClassicAIV1Model.predict_team_defense` executes the shared assignments through
the existing defense controller. Individual switching and approach boosts are
disabled during the trap; only one attacker may request contact. A covered closer
can approach from behind while retaining the ordinary energy, facing, impact,
route, and collision checks. Exiting the trap releases commitments and reruns
primary selection, so a loose puck gets a fresh retriever.

`python -m tests.integration.teammate_pressure` checks two- and three-bot traps
using native feedback. `evaluation.teammates --no-pressure` provides the previous
coordinated behavior for seeded comparisons. Reports include `pressure_outcomes`
with activation, fallback, spacing, and subsequent possession/shot observations;
these are contextual measurements, not claims that a trap caused a turnover.

## Target-position control

`--action_type TARGET_POSITION` supports full-team, single-controller `MlpPolicy`
with a two-float normalized action. DefenseZone maps that action into
`x = -120..120`, `y = -270..-88`. `(0, 0)` maps to `(0, -179)`.

`env.target_control.TargetPositionController` converts destinations into filtered
buttons. The policy selects targets at its decision interval; the controller
handles skating, routing, switching, and native feedback each emulator frame.
Other players remain under the game's AI. Playback retains the checkpoint's
decision interval while rendering individual frames.

Target checkpoints require matching metadata for coordinates, bounds, controller
settings, and ordered fields. They cannot be loaded as button policies. Start a
fresh target model with `configs/training/defense-target.json`; loading existing
weights preserves their saved network and action distribution.

## Tasks and extension

Tasks declare setup, rewards, completion, and supported actions through the task
registry. Procedural resets operate on the current emulator instance and preserve
coherent ownership, controller slots, object ordering, roles, and physical state.
Use the shared setup helpers rather than writing unrelated coordinates separately.

DefenseZone supports button or target-position control. PvG supports the compact
player-goalie input. Choose a matching game/save, controller count, action type,
and observation layout for the task. Preserve shared self-play callers when
changing task behavior.

## EA 4-Way Play

`config.resolve_play_controllers` resolves teammate layout for both playback
entrypoints and configuration validation. The logical count is team size plus
one human in `team_vs_player`. Logical slots map directly to zero-based emulator
ports; EA 4-Way Play provides four physical input channels. Unused channels are
neutral and unassigned. `make_retro` explicitly requests device 1025 on connector
0 only for the new modes. This configures the adapter across both Genesis
connectors; connector IDs and the four input-channel IDs are different concepts.

The stable-retro patch lives in the sibling repository. It increases native
capacity to four, exposes `RetroEmulator.max_players`, `controller_devices`, and
`set_controller_port_device(port, device)`, and accepts `controller_devices` in
`RetroEnv`. It bounds-checks native callbacks and movie access. Existing defaults
remain unchanged. Feature detection fails before extra-controller environments
start if the extension is older.

Use this source checkout and its patched sibling together; a stock installed
wheel is insufficient. In an already configured Unix Makefiles build using the
development interpreter, rebuild only the frontend objects and extension:

```bash
cd /home/mat/github/stable-retro
make -f CMakeFiles/retro-base.dir/build.make CMakeFiles/retro-base.dir/build -j4
make -f CMakeFiles/stable_retro.dir/build.make CMakeFiles/stable_retro.dir/build -j4
/home/mat/vretro/bin/python -c 'import stable_retro; print(stable_retro.__file__, stable_retro.RetroEmulator.max_players)'
```

For a fresh checkout, apply the sibling repository changes, install its documented
build prerequisites, configure CMake with `-DPYTHON_EXECUTABLE=/path/to/python`
and `-DPython_EXECUTABLE=/path/to/python`, and follow its source installation
instructions. Build the `genesis` target for Genesis Plus GX. The targeted make
commands above assume its other native build dependencies have already been
built; a fresh installation must build those dependencies first. Start a fresh
Python process after rebuilding. No Gymnasium or Stable-Baselines3 patch is used.

`game.multitap.prepare_multitap_save` initializes the saved port-B IO direction
register to the value used by the ROM's adapter boot routine. It validates the
GENPLUS-GX 1.7.5 state format and expected IO values. `configure_teammates` then
sets native polling, teams, unique selected skaters and automatic goalies at
reset. Gameplay uses controller input exclusively. Snapshot restore retains the
configured adapter and restores saved IO without calling device setup again.

The ROM temporarily substitutes C3/C4 into C1/C2 during input and selection
routines. A video-frame boundary can expose those unfinished RAM tables.
`controller_snapshot` retains the raw tables in diagnostics and reports unknown
selections for inconsistent snapshots; it neither guesses suspended selections
nor writes ownership back. Bots produce neutral input while those selections
are unavailable, and persistent inconsistency fails explicitly. Goalie slots
use their own telemetry and are excluded from the skater-flag comparison.

The observation wrapper accepts 24/36/48 binary buttons only for the corresponding
scripted layout, maintains separate input histories, and steps the emulator once.
Neural observation layouts and ordinary one-/two-controller action contracts are
unchanged. Gymnasium returns scalar reward and five step values; the Monitor and
VecEnv stack retains batching, four-value VecEnv steps and terminal auto-reset.
The debug display appends the human's twelve buttons after all bot channels.
In both teammate modes, `--side` selects the bots' physical team; the human,
when present, owns the opposite team. Reset selects distinct skaters from
slots 0..4 for home or 6..10 for away, and initializes the human from the
opposing range. The wrapper keeps the display in physical home/away order;
the coordinator supplies each bot's team perspective. Teammate playback never
calls the single-controller `restore_away_control` gameplay repair.

New layouts use v2 teammate traces with physical capacity, logical port mapping,
teams, opponent, seed, reset setup, and ROM/native/core hashes. Replay validates
this metadata before stepping. Existing v1 traces retain their historical layout.
V2 uses its existing `side` and physical `teams` fields for away layouts;
omitted `side` retains the historical home meaning. No schema change is needed.
`python -m tests.integration.away_teammates` checks both sides, native away input,
display routing, and a full away regulation game with input replay.
`python -m tests.integration.away_teammates_comparison` runs the fixed 24-match
Sabres/Ducks, two/three-bot, pressure-on/off comparison for seeds 94..96,
writing paired initial-state checks and outcome totals to
`tmp/away-teammates/comparison.json`.
Explicit native checks run with `python -m tests.integration.multitap`; legacy
checks remain in `python -m tests.integration.teammates` and
`python -m tests.integration.human_play`.

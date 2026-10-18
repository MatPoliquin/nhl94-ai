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

RAM decoding belongs in `game/ram.py` and `game/state.py`. Task reset helpers
belong in `tasks/setup.py` and the relevant task initializer. Changes to ordered
observations or action semantics require matching schema version changes.

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

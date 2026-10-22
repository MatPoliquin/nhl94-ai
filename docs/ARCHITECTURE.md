# Architecture

The Python application lives in `nhl94_ai/`. Top-level `models/` holds model
assets; `nhl94_ai/models/` contains policy definitions. The separate
[TypeScript browser application](../web/README.md) uses a WebAssembly emulator
and its own Classic implementation; it does not require the Python runtime.

| Package | Responsibility | Extension point |
| --- | --- | --- |
| `game` | Variants, console backends, RAM decoding, geometry, and features | `specs.GAMES` |
| `env` | Environment assembly, observations, actions, and timing | `factory.build_single_nhl94_env`, `encoding.ObservationEncoder` |
| `tasks` | Episode setup, progression, completion, and reward stubs | `registry.register_task` |
| `agents` | Scripted and learned gameplay controllers | `base.Agent.reset`, `base.Agent.act` |
| `models` | Neural policy definitions | `factory.register_model` |
| `training` | Unavailable workflow stubs and compatibility names | `unavailable.TrainingUnavailableError` |
| `evaluation` | Playback, evaluation, recording, replay, and external opponents | `runner.evaluate`, `external.play_match` |
| `ui` | Optional display and recording | State and agent diagnostics |
| `cli` | Command dispatch and configuration | `nhl94 --help` |

Game decoding does not import UI or training code. Display and export dependencies
load only when requested. Environment bindings and controller state are per instance.
For commands, see the [README](../README.md), [Classic guide](CLASSIC.md),
[evaluation guide](../nhl94_ai/evaluation/README.md), and [test guide](../tests/README.md).

## Console backends

`GameSpec.platform` selects a backend in `game/platforms/`. Genesis and SNES
share `NHL94GameState` and `ClassicAIV1Model`. Backends own telemetry registration,
player/engine decoding, rink dimensions, action encoding, motion projection,
goalie mechanics, and episode clocks. Selecting a backend does not mutate global
constants. `GameSpec.skaters_per_team` describes the roster layout; `num_players`
counts controllers.

Genesis retains its button ordering, observation normalization, and ROM-derived
forecasts. SNES uses native array ownership/selection, signed word velocities,
a separate normalization identifier, and estimated motion/pass forecasts. Its
integration scenario is packaged here; Python ROMs and saves live in stable-retro.
SNES reset configures native menu and controller tables before advancing the save;
gameplay then uses controller input and native feedback. Supported modes and
console restrictions are documented in [Classic](CLASSIC.md#snes-support).

## Services

Training, demonstration collection, imitation, curriculum, and train-then-compare
workflows are unavailable pending an ML redesign. Python entrypoints raise
`TrainingUnavailableError` (`NotImplementedError`); CLI entrypoints return a
nonzero status with an explanation. ES support is removed.

Compatible existing non-ES checkpoints remain usable for inference and evaluation.
`models.decision_ppo` retains legacy loading/prediction classes with learning
disabled; `training.decision_ppo` re-exports their serialized names. `datasets`
retains artifact readers/writers, and `run_options` retains legacy configuration
parsing. These compatibility helpers do not restore training workflows.

`evaluation.runner.evaluate(agent, config, env=env)` returns rewards, lengths,
termination/truncation flags, and budget status. The caller owns and closes the
environment. Task-based evaluation reports unavailable rewards as JSON `null`
with `reward_status: unavailable`; goals, shots, passes, and episode endings remain
measurable. Native offense/goalie outcome protocols have their own scoring rules.

`nhl94 external` loads an explicitly requested adapter. Ordinary gameplay has no
external-adapter dependency. The [evaluation guide](../nhl94_ai/evaluation/README.md)
maps playback, policy evaluation, benchmarks, and replay modules.

## Configuration

Precedence is parser defaults, JSON options, then explicit CLI arguments.
Declared paths inside an options file resolve relative to that file; CLI paths
resolve relative to the working directory. `--state` normally names an emulator
save; individual evaluation runners may also accept native save paths. JSON
boolean options must be booleans.

Editable examples live in `configs/`. Packaged legacy training defaults live in
`nhl94_ai/data/training/`; keep corresponding copies synchronized when changing
defaults. Dependencies and extras are declared in `pyproject.toml`.
`EnvironmentConfig` validates game, controller, action, goalie, and self-play
combinations before emulator workers start. Classic playback's additional file
settings and overrides are described in [decision timing](CLASSIC.md#decision-timing).

## Agents

`agents.base.Agent` exposes `reset()` and `act(AgentInput) -> AgentOutput`.
Inputs carry game state and/or encoded observations; outputs carry actions,
diagnostics, and an optional decision snapshot. `ScriptedAgent` adapts Classic;
`LearnedAgent` retains recurrent state until reset. `FrameRepeatAgent` holds a
policy action while playback advances and renders individual frames.

`ClassicAIV1Model` coordinates tactical components and frame ownership.
`classic_options` resolves and validates options; `classic_components` constructs
components; `classic_resources` holds reusable loaded artifacts. Mutable action
state belongs to each controller. Possession, passing, defense, and goalie modules
make tactical decisions; scheduling and lifecycle modules govern their execution.
Requested inputs are diagnostic intent. Applied buttons, native ownership,
animation feedback, and counters establish what the game executed.

## Model input variants

`--model_input` selects ordered observation groups from the input registry.
Checkpoints record the layout and normalization; equal tensor sizes do not imply
compatible schemas. Loading restores the saved architecture, so editing a
configuration file does not resize a loaded model.

`pvg` is the compact player-goalie layout, `pvg-v3` its earlier version, and
`default`/`legacy` the respective full-team layouts. Changes to ordered observations
or action semantics require matching schema version changes. See
[artifact contracts](ARTIFACTS.md) for checkpoint, dataset, and export compatibility.

## Timing and reset

Each inner environment step applies input, advances one emulator frame, refreshes
game state, runs task progression and reward/completion logic, and encodes the
next observation. Frame-skip wrappers repeat actions around that loop. Macro
and native lifecycle timers count emulator frames.

In reactive mode, Classic observes every native frame independently of its
tactical interval. `agents/scheduling.py` owns clocks and cached input;
`agents/lifecycle.py` owns shot, pass, and one-timer transitions.
`agents.base.configure_scripted_frames` attaches the scripted agent to the
frame-skip wrapper and can record actual substeps. Learned button policies retain
their configured interval. Strict Classic control intervals also throttle
observation and hold raw buttons; see [Classic](CLASSIC.md#decision-timing).
Reset clears controller history, pending actions, and recurrent policy state.

## Coordinated teammates

`agents.teammates.TeammateCoordinator` reads the same RAM snapshot through two or
three controller-specific game states, with separate execution histories.
It assigns pursuit/support roles and emits 24 or 36 bot buttons before the next
native frame. Controllers persist across role changes. Support frames observe
feedback and commit applied buttons without generating discarded Classic decisions.

`game.ram.configure_teammates` owns reset assignments and automatic-goalie setup.
`teammate_controller_info` supplies controller selections without changing physical
team identities. The coordinator supplies each bot's team perspective; the display
keeps physical home/away order. Both teammate modes support either side and use
controller input for gameplay selection changes, without the single-controller
`restore_away_control` repair.

`SharedReceiving` reserves receiver input through the pass/reception lifecycle.
Commitments track native release, actual receiver, expiry, and reception/shot
phase. Temporarily unknown selections produce neutral input and preserve released
passes by default without extending expiry. Applied one-timer cues and native
arming are separate evidence. `SupportPlanner` handles optional reachable support
positions; `SharedPressure` coordinates containment, contact, and coverage through
the existing defense controller. Native feedback invalidates stale assignments.
Motion forecasts estimate future positions rather than predict new CPU decisions.

Gameplay options belong in [Classic](CLASSIC.md#ai-teammates); experimental
protocols, measurements, and replay comparisons belong in the
[evaluation guide](../nhl94_ai/evaluation/README.md#matched-teammate-experiments).

## Target-position control

`TARGET_POSITION` supports full-team Genesis with one `MlpPolicy` controller and
a two-float normalized action. DefenseZone maps it into
`x = -120..120`, `y = -270..-88`; `(0, 0)` maps to `(0, -179)`.
`env.target_control.TargetPositionController` converts destinations into filtered
buttons, handling skating, routing, switching, and feedback each native frame.
The policy selects targets at its recorded decision interval; other players use
the game's AI. Target checkpoints require matching coordinate, bound, controller,
and observation metadata and cannot be loaded as button policies.

## Tasks and extension

Tasks declare setup, optional frame progression, reward availability, completion,
and supported actions through `tasks.registry`. Built-in rewards currently return
`0.0` and advertise `reward_available=False`. The observation wrapper calls the
optional `TaskDefinition.advance` once per native frame before completion checks,
preserving stall/possession limits and opportunity completion.

RAM decoding belongs in `game/ram.py`, `game/state.py`, and console backends.
Procedural resets use `tasks/setup.py` and task initializers on the current emulator
instance, preserving ownership, controller slots, object ordering, and physical
state. Match game/save, controller count, action type, and observation layout;
preserve shared self-play callers when changing task behavior.

## EA 4-Way Play

`config.resolve_play_controllers` derives the logical count: two or three bots,
plus one human in `team_vs_player`. For more than two controllers, `make_retro`
requires the patched stable-retro extension and requests device 1025 on connector
0. EA 4-Way Play spans both Genesis connectors and exposes four input channels;
connector IDs and input-channel IDs are distinct. Logical slots map directly to
zero-based input channels. Unused channels are neutral and unassigned.

The required stable-retro patch lives in the sibling checkout. It exposes
`RetroEmulator.max_players >= 4`, `set_controller_port_device(port, device)`, and
`controller_devices` support in `RetroEnv`. `game.multitap.require_multitap`
checks capabilities before construction. Build/install that patched checkout
with the project's Python environment and its documented native prerequisites,
including the Genesis core; restart Python after rebuilding. A stock wheel
without these capabilities cannot run three/four-controller layouts.

`game.multitap.prepare_multitap_save` validates the GENPLUS-GX 1.7.5 save format
and initializes port-B IO for adapter polling at reset. `configure_teammates`
then assigns teams, unique selected skaters, and automatic goalies. Snapshot
restore preserves adapter/IO state without repeating device setup.

The ROM can temporarily alias C3/C4 into C1/C2 tables across a video-frame boundary.
`controller_snapshot` retains raw tables for diagnostics and reports inconsistent
selections as unknown; it does not guess ownership or repair live RAM. Bots emit
neutral input during the gap, and persistent inconsistency fails explicitly.
Goalie telemetry is separate from the skater-selection check.

The observation wrapper accepts 24/36/48 binary buttons for the corresponding
layout, tracks each controller's history, and advances the emulator once. Human
buttons follow bot channels. Gymnasium retains scalar rewards and five step
values; the Monitor/VecEnv stack retains batching, four-value VecEnv steps, and
terminal auto-reset. Ordinary neural observation/action contracts stay unchanged.

Expanded layouts use v2 teammate input traces with physical capacity, logical
ports, teams, opponent, seed, reset setup, and ROM/native/core hashes. Replay checks
this metadata before stepping. Existing v1 traces retain their historical layout;
an omitted `side` retains the historical home meaning. Native checks live in
`tests.integration.multitap`, `teammates`, `away_teammates`, and `human_play`;
see the [test guide](../tests/README.md) for prerequisites and execution conventions.

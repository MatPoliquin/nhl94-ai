# NHL94 AI refactoring plan

Status: implemented in the working tree, including removal of the temporary
forwarding scripts and duplicate configuration directories. See
[REFACTOR_RESULTS.md](REFACTOR_RESULTS.md) for verification and compatibility limits.

## Objective

Organize NHL94 AI as an installable Python package with clear boundaries between
hockey logic, emulator interaction, agents, training, evaluation, and display.
Preserve working gameplay, training commands, configurations, and saved models
while migrating in small, verifiable steps.

Adding a task, agent, or neural network should require changes in its own module
and an explicit registration or configuration entry. The same agent should be
usable for gameplay, evaluation, self-play, and demonstration collection.

## Agreed constraints

- Put the Python package at `nhl94_ai/` directly under the repository root. There
  will be no `src/` directory.
- Work in this repository. `stable-retro` remains the emulator dependency, and
  `retro-ai-runtime` owns the C++ runtime. Changes to those repositories are
  separate work.
- Keep existing README files unchanged during the initial code refactor.
  Documentation reorganization is a later pass.
- Preserve the existing Git history; importing history from stable-retro-scripts
  is unnecessary.
- Preserve current commands, reward names, state names, observation ordering,
  action encodings, and checkpoint compatibility during migration.
- Keep model binaries unchanged. Moving or reorganizing assets must preserve
  their contents.
- Keep behavior changes, tactical improvements, dependency upgrades, and reward
  tuning separate from structural changes.
- Preserve support for 1-on-1, 2-on-2, and full-team NHL94.

## Current starting point

The Python project has been copied into this repository. Other-game wrappers,
configurations, model assets, screenshots, RPG tooling, and Pong-specific policy
code have been removed. README files still contain the original documentation.

Checks completed after that cleanup:

- Three NHL94 model-input tests and five PyTorch checks passed.
- Each NHL94 variant reset and completed 16 ClassicAIV3 action steps.
- A 16-step PPO training run completed and saved a checkpoint.
- Eight gameplay, training, and imitation command entry points loaded with
  `--help`.
- The export entry point could not load because ONNX is missing from the local
  environment.
- Pylint reported six findings in files unchanged by the cleanup.

These are smoke checks, not a complete regression suite or evidence of playing
strength. Capture reusable checks before moving core code.

The main areas to untangle are:

| Current area | Responsibilities currently combined |
| --- | --- |
| `nhl94_obs.py` | Observations, action translation, timed moves, task execution, perspective transforms, opponent inference |
| `nhl94_gamestate.py` | State decoding, normalization, geometry, tactical features, historical state |
| `models.py` and `models_utils.py` | Network definitions, policy selection, scripted controllers, loading, diagnostics |
| `train_live.py` | Training, evaluation, self-play snapshots, threading, rendering |
| `train_curriculum.py` | Configuration, conversion to CLI arguments, phase execution, terminal display |
| `compare_models.py` | Training orchestration, evaluation, presentation, video recording |

## Target layout

```text
nhl94-ai/
├── nhl94_ai/
│   ├── __init__.py
│   ├── game/          # Variants, state decoding, geometry, derived features
│   ├── env/           # Gym environment, observations, actions, self-play
│   ├── tasks/         # Episode setup, rewards, completion, action restrictions
│   ├── agents/        # Scripted agents and adapters for learned controllers
│   ├── models/        # Neural networks and SB3 policy definitions
│   ├── training/      # RL, behavior cloning, DAgger, curriculum execution
│   ├── evaluation/    # Matches, benchmarks, metrics
│   ├── ui/            # Gameplay, debug, training, comparison, recording
│   ├── cli/           # Thin command entry points
│   ├── config.py
│   └── artifacts.py
├── configs/
│   ├── training/
│   └── curricula/
├── tests/
│   ├── unit/
│   └── integration/
├── models/            # Bundled model assets
├── docs/              # Documentation and screenshots, migrated later
├── .github/
├── .gitignore
├── pyproject.toml
├── README.md
└── LICENSE
```

Temporary forwarding scripts, duplicate configuration directories, and
requirements aliases were removed after verifying the package commands. The
checkpoint import aliases inside the package remain necessary for saved models.
This plan now lives under `docs/`.

## Architecture rules

- `game/` provides hockey state and calculations without importing display,
  training, or model-loading code. Keep emulator information decoding testable
  using plain recorded dictionaries.
- `env/` coordinates emulator stepping, state updates, task evaluation,
  observations, and actions. Keep their ordering explicit.
- `tasks/` operates on state transitions and owns its episode-specific trackers.
- `agents/` consumes state and/or encoded observations and produces actions plus
  optional diagnostics. Model construction happens at the application boundary.
- Training and evaluation construct environments and agents explicitly. Replace
  the mutable global wrapper manager with ordinary factories as callers migrate.
- UI code consumes progress events and snapshots. Training can run without
  importing or starting the UI.
- CLI parsers convert user input into validated configuration and call reusable
  functions. Internal callers pass configuration objects directly.
- Use small explicit registries where implementations need selection by name.
  Introduce abstractions when an existing pair of implementations needs them.

## Phase 1: Preserve behavior with repeatable checks

- [x] Add a reusable emulator smoke runner for all three supported variants.
- [x] Record focused state fixtures for possession, controlled-player changes,
  period direction, goalie control, and self-play perspective transforms.
- [x] Check observation field order, dimensions, normalization, and sequence reset.
- [x] Check action layouts, timed moves, cooldown reset, and frame-skip behavior.
- [x] Check rewards and termination against representative state transitions.
- [x] Exercise checkpoint loading and action prediction, including an existing
  custom-policy checkpoint when available.
- [x] Run a small training/save/reload cycle and a small imitation-learning cycle.
- [x] Define a fixed evaluation set for later behavior changes, recording seeds,
  start states, opponents, game variants, and frame settings.

**Completion check:** automated checks reproduce the current outputs for fixed
inputs and short controlled traces. Emulator-dependent checks are clearly
separated from unit tests and use locally supplied ROMs. Stochastic training is
checked for successful execution and valid artifacts rather than identical weights.

## Phase 2: Establish the package

- [x] Add `pyproject.toml` and the root-level `nhl94_ai/` package.
- [x] Configure package discovery explicitly so repository directories such as
  `models/`, `configs/`, and `tests/` do not become accidental Python packages.
- [x] Move one subsystem at a time, preserving behavior and existing imports
  through compatibility modules where necessary.
- [x] Turn existing executable scripts into forwarding entry points as their
  implementations move into the package.
- [x] Separate optional display and export dependencies and defer their imports
  until those features are requested.
- [x] Keep dependency versions stable while migrating dependency declarations.
- [x] Verify editable installation and a built wheel in an isolated environment.

**Completion check:** current commands still work, imports work outside the
repository's working directory, and installed-package checks do not depend on
`sys.path` modifications from the old script layout. Existing checkpoints load.

## Phase 3: Centralize configuration and game variants

- [x] Introduce validated configuration objects for environments, agents,
  training, evaluation, and curriculum phases.
- [x] Introduce one `GameSpec` definition per supported variant, including its
  environment ID, skaters per team, and default states.
- [x] Distinguish controller count from skaters per team in internal names and
  types. Preserve existing CLI flags through translation.
- [x] Define configuration precedence: defaults, file, then explicit CLI overrides.
- [x] Resolve paths relative to their declaring configuration file. Keep an
  adapter for existing script-relative configuration paths during migration.
- [x] Validate incompatible observation, action, policy, and self-play settings
  before creating emulator workers.
- [x] Move training and curriculum configuration under `configs/`, retaining
  compatibility paths while existing workflows are migrated.
- [x] Define how installed commands obtain defaults without depending on a
  checkout-relative `../hyperparams` path.

**Completion check:** old configurations resolve to equivalent settings, and
invalid combinations fail with useful errors before any training starts.

## Phase 4: Extract environment components

Migrate these independently behind the existing environment interface:

| Component | Responsibility |
| --- | --- |
| State decoder | Convert emulator information into hockey state |
| Geometry and derived features | Distances, passing lanes, shot lanes, tactical features |
| Observation encoder | Produce inputs with defined ordering, dtype, normalization, and schema version |
| Action controller | Translate button/intent actions and advance timed input sequences |
| Task definition | Initialize episodes and compute rewards, completion, and action restrictions |
| Opponent controller | Produce the second controller's actions using a defined perspective |
| Environment factory | Assemble the components, wrappers, and vector environments |

- [x] Replace positional reward-function tuples with named task definitions,
  initially adapting existing functions without changing their calculations.
- [x] Extract action and perspective logic from `nhl94_obs.py`.
- [x] Give learner and opponent controllers independent episode state.
- [x] Make emulator-frame counts and agent-decision counts explicit. Preserve
  current timing first; review any timing corrections separately.
- [x] Define reset behavior for histories, reward trackers, macros, and opponents.
- [x] Preserve the distinction between termination and truncation in the public
  environment interface; record any discovered behavioral bugs separately.
- [x] Route training, gameplay, and imitation through the same environment factory.

**Completion check:** fixed traces produce equivalent observations, actions, and
rewards; resets do not carry episode state forward; all game variants and
supported action modes pass their integration checks.

## Phase 5: Separate agents from neural networks

- [x] Define an agent interface with episode reset, action selection, declared
  input/action schemas, and optional diagnostic output.
- [x] Add adapters for learned policies and the existing multi-model controller.
- [x] Expose ClassicAI versions through the same agent interface. Keep historical
  versions available as reproducible baselines.
- [x] Extract shared scripted-agent geometry, targeting, and controller mechanics
  where their behavior is actually equivalent.
- [x] Keep training and serialization capabilities separate from gameplay agents;
  scripted controllers should not require unsupported `learn()` methods.
- [x] Split neural network definitions into focused model modules and use an
  explicit model-construction registry.
- [x] Preserve checkpoint import aliases until old custom-policy loading has a
  tested replacement or migration path.

**Completion check:** the same agent implementation can be used in gameplay,
evaluation, opponent control, and demonstration collection without special-case
branches in each runner.

## Phase 6: Unify training, evaluation, and curriculum execution

Proposed service interfaces:

```python
train(config) -> TrainingResult
evaluate(agent, config) -> EvaluationResult
collect_demonstrations(agent, config) -> Dataset
run_curriculum(config) -> CurriculumResult
```

- [x] Extract reusable RL, behavior-cloning, and DAgger services.
- [x] Extract evaluation and match metrics into `evaluation/`.
- [x] Replace curriculum configuration-to-CLI-to-configuration conversions with
  direct calls using validated phase configurations.
- [x] Register phase implementations explicitly and share result/artifact types.
- [x] Move progress reporting, terminal rendering, live display, and recording
  into UI consumers of progress events or snapshots.
- [x] Make environment cleanup and display shutdown explicit on success,
  cancellation, and failure.
- [x] Use the same evaluation configuration for live monitoring and headless runs.

**Completion check:** ordinary training, live training, and curriculum phases use
the same underlying services. Headless runs need no display initialization, and
UI attachment does not change training configuration or action timing.

## Phase 7: Version artifacts and export contracts

- [x] Save resolved configuration, random seeds, code revision, and dependency
  versions with each run.
- [x] Record game variant, observation ordering and normalization, action schema,
  sequence length, and frame settings with checkpoints and datasets.
- [x] Build on the existing observation schema versioning and validate schema
  compatibility when loading models or imitation datasets.
- [x] Record whether an older artifact's schema is known, explicitly supplied,
  or unresolved. Avoid silently guessing compatibility.
- [x] Validate exported predictions against the Python model on representative
  inputs, in addition to checking file creation.
- [x] Define export metadata that the separate runtime can consume. Any runtime
  implementation changes belong in a follow-up task in that repository.

**Completion check:** incompatible inputs/actions fail early with useful errors;
supported checkpoints reload; exported models have a documented, reproducible
input/output contract.

## Phase 8: Complete the repository cleanup

- [x] Remove unused generic wrappers and duplicated utilities after checking
  their callers and checkpoint compatibility.
- [x] Consolidate dependency and tool configuration in `pyproject.toml` where
  supported, updating CI in the same change.
- [x] Run fast unit tests and package-install checks in ordinary CI. Keep ROM-
  dependent emulator checks available as a separate local or suitably configured
  integration job.
- [x] Resolve the inherited lint findings in a separate cleanup change.
- [x] Remove forwarding scripts only after their replacement commands and
  checkpoint-loading paths have been verified and documented.
- [x] In the deferred documentation pass, update README examples, consolidate
  NHL94 documentation and screenshots under `docs/`, remove other-game documents,
  and remove the obsolete `retro_ai_lib/` migration notice.

Proposed commands after the CLI migration:

```bash
nhl94 train --config configs/training/default.json
nhl94 play --agent classic-v1
nhl94 curriculum configs/curricula/nhl94.json
```

These commands are implemented. Add `--live` to training to attach the display.

**Completion check:** the root matches the agreed layout, documented commands
work from an installed package, and no active implementation depends on the old
script directory structure.

## Execution approach

1. Finish Phase 1 before moving core code.
2. Establish the package foundation in Phase 2, then migrate modules alongside
   their focused extractions. Avoid a single repository-wide move.
3. Introduce configuration and environment boundaries before consolidating their
   callers. Artifact metadata can be introduced alongside those boundaries.
4. Keep each change reviewable: one subsystem, its compatibility adapters, and
   the relevant checks.
5. Compare behavior before and after each extraction. If a pre-existing bug is
   found, document it and make its correction a separate behavior change.

Implementation and validation details are recorded in [REFACTOR_RESULTS.md](REFACTOR_RESULTS.md).
Use `nhl94` or `python -m nhl94_ai` for commands and `configs/` for configuration examples.

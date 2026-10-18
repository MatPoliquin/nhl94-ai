# NHL94 AI

Agents, gameplay, evaluation, and training for NHL94 on Genesis: 1-on-1,
2-on-2, and full-team hockey. The Python package lives in `nhl94_ai/` at the
repository root. The `models/` directory holds bundled checkpoints.

## Install

Use your Python environment with stable-retro and the NHL94 game integrations
installed. The emulator and imported ROMs belong to stable-retro; this repository
contains the agents and training tools.

```bash
python -m pip install -e .
# Optional display, vision models, and export tools:
python -m pip install -e '.[display,vision,export]'
```

## Commands

```bash
# Watch Classic with selective deep-csave goalie control; requires the display extra.
nhl94 play --agent classic-v1 --env NHL94-Genesis-v0

# Watch Classic as Quebec against Montreal's built-in CPU.
nhl94 play --agent classic-v1 --env NHL94-Genesis-v0 --state CanadiensVsNordiques.start --side away --max_playback_speed 1.0

# Record the full 1080p window, including the decision inspector and debug panels.
nhl94 play --agent classic-v1 --env NHL94-Genesis-v0 --record-mp4 ../recordings/new-session.mp4

# Play against Classic v1 at half speed (you control the away team).
nhl94 play --agent classic-v1 --mode player_vs_model --env NHL94-Genesis-v0 --max_playback_speed 0.5

# Train without a display. Add --live to watch progress.
nhl94 train --config configs/training/default.json --num_env 1

# Train Ducks versus the CPU shootout goalie.
nhl94 train --config configs/training/pvg.json

# Run bounded evaluation against the game's built-in opponent.
nhl94 evaluate --agent classic-v1 --episodes 1 --max_steps 100

# Inspect a curriculum before starting its configured training runs.
nhl94 curriculum configs/curricula/nhl94.json --dry-run

nhl94 collect --help
nhl94 bc --help
nhl94 dagger --help
nhl94 export --help
```

`python -m nhl94_ai` is equivalent to `nhl94`. Commands and packaged defaults
work outside the checkout. Relative paths in a JSON options file are resolved
against that file's directory. Explicit CLI flags override file settings.
`--config` accepts either hyperparameters or an options object; see
[configuration](docs/ARCHITECTURE.md#configuration).

## External opponents

Install an adapter package separately, then pass its Python factory explicitly:

```bash
nhl94 external --adapter my_external_package.adapter:create \
  --pairs 10 --seed 26534 --periods 3 --output /tmp/external-results.json
```

Each seed runs with Classic on both controller assignments. Use `--side home`
or `--side away` for one assignment and `--watch` for a live window. Choose a
saved roster with `--fixture`; `--periods 1 --seconds 30` runs a short smoke test.
Reports require an explicit output path. Normal play does not load adapters.

An adapter factory receives `protocol_version=1`, `side` (physical controller 1
or 2), `seed`, and an optional `config` path. It returns an object with `reset()`,
`act(frame)`, `close()`, JSON-compatible `metadata`, and a `ram` mapping of field
names to `(address, type)` declarations. The wrapper accepts integer Genesis RAM
reads and supplies `frame.number`, `frame.side`, `frame.phase`, and
`frame.observations`. `act` returns twelve binary buttons in this order:
`B A MODE START UP DOWN LEFT RIGHT C Y X Z`. Adapters receive every native frame,
including intermissions; Classic keeps its configured tactical interval.
Adapter code executes in the current Python process. Bot implementations,
runtimes, configuration and named examples belong in the adapter package.

## Learned policies

The training commands support collection, imitation learning, PPO, and self-play.
Use matching game, action, observation, and checkpoint settings when loading a
policy. Keep each model together with its metadata sidecar.

```bash
# Inspect the available offense and goalie workflows.
nhl94 offense --help
nhl94 goalie --help

# Train a target-position defensive policy.
nhl94 train --config configs/training/defense-target.json --live

# Watch a compatible button policy.
nhl94 play --env NHL94-Genesis-v0 --nn MlpPolicy --model_1 /path/to/model.zip
```

The offense workflow supports GRU and MLP policies, full-period or attack tasks,
and carry-only control with Classic executing passes and shots. Goalie policies
use a separate checkpoint format and run inside Classic's manual-goalie control.
These optional policies do not change the default scripted agent.

See [artifact formats](docs/ARTIFACTS.md) for schema compatibility and
[target-position control](docs/ARCHITECTURE.md#target-position-control) for
coordinate-based policies.

## Classic

`classic-v1` (also `classic`) is the scripted gameplay agent. Full-team playback
uses selective manual-goalie control and `deep-csave` positioning by default.
`--goalie-positioning classic` selects the original manual goalie;
`--goalie-policy off` disables manual-goalie AI.

The agent reacts to native emulator frames while retaining its configured
tactical decision interval. Use the [Classic guide](docs/CLASSIC.md) for timing,
goalie options, and supported optional behavior. `nhl94 play --help` lists the
available flags.

## Development

```bash
python -m pip install -e '.[dev]'
python -m unittest discover -s tests -v
python -m pylint nhl94_ai tests
python -m build --wheel

# Run emulator checks with local game integrations and ROMs installed.
python tests/integration/smoke.py
python tests/integration/workflows.py
```

Unit tests run without ROM files. Emulator checks in `tests/integration/` run
explicitly with the matching stable-retro integrations installed.

- [Architecture and extension points](docs/ARCHITECTURE.md)
- [Artifact and export contracts](docs/ARTIFACTS.md)
- [Gameplay examples](docs/NHL94-README.md)
- [Current limitations](docs/KNOWN_ISSUES.md)
- [Browser app](web/README.md)

The C++ runtime is maintained separately in `retro-ai-runtime`.

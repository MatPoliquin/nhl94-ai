# NHL94 AI

Agents, gameplay, evaluation, and training for NHL94 on Genesis: 1-on-1,
2-on-2, and full-team hockey. Classic also plays full-team NHL94 on SNES
against the built-in CPU using the same tactical agent. The Python package lives in `nhl94_ai/` at the
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

# Watch two coordinated AI teammates against the built-in CPU (automatic goalies).
nhl94 play --agent classic-v1 --env NHL94-Genesis-v0 --mode team_vs_game --max_playback_speed 1

# Watch the shared Classic agent on SNES against the CPU.
nhl94 play --agent classic-v1 --env NHL94-Snes-v0

# SNES: away versus CPU, human versus AI, or two Classic agents.
nhl94 play --agent classic-v1 --env NHL94-Snes-v0 --side away
nhl94 play --agent classic-v1 --env NHL94-Snes-v0 --mode player_vs_model
nhl94 play --agent classic-v1 --env NHL94-Snes-v0 --mode model_vs_model

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

The Python play window automatically supports the 8BitDo M30 through the
USB receiver identified as `6B controller` (`0ca3:0024`). The D-pad moves;
A clears/holds, B passes/switches, C shoots/checks, and Start pauses the game.
X/Y/Z and Select/Mode map to the matching Genesis buttons. Keyboard controls
remain available alongside the pad. F1 toggles human input; losing window focus
or unplugging the pad releases its input. Reconnecting works without restarting.
Use `--no-gamepad` for keyboard-only play. Other M30 connection modes have not
been mapped.

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

To play against an installed adapter yourself, use `--human` and select your
side. This starts a playable window with the same keyboard and M30 controls as
`nhl94 play`:

```bash
# You: Ducks (away/controller 2). Adapter: Sabres (home/controller 1).
nhl94 external --adapter my_external_package.adapter:create \
  --human --fixture sabres-ducks --side away
```

Human input and adapter callbacks run every native frame at 60 Hz, with the
window paced at normal speed. `--frame-skip` only applies to Classic in AI
matches. `--side home` puts you on the home team; human mode requires one side.
Games last three five-minute regulation periods by default, with automatic
intermission advancement. Space pauses playback, Start/Enter pauses inside the
game, and Esc exits. Add `--no-gamepad` for keyboard only. `--watch` remains
spectator mode when `--human` is absent.

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

## AI teammates

`team_vs_game` runs two or three home Classic bots against the built-in CPU.
`team_vs_player` adds one opposing human through the Python keyboard/gamepad display.
`--team-size {2,3}` defaults to 2; `--team-strategy {coordinated,independent}`
defaults to coordinated. Bots occupy controllers 1 through N; the human occupies
controller N+1. The coordinator gives the third bot a separate deep coverage role.
Each bot observes the same native frame with its own selection and input history.
The display shows each bot's selected slot, role and target.

These modes support original full-team `NHL94-Genesis-v0`, home bots, `FILTERED`
buttons, automatic goalies and three regulation periods. Use
`SabresVsMightyDucks.ManualGoalie.Start`. On-ice rosters remain five skaters plus
a goalie per side. Learned checkpoints, training, SNES, manual-goalie coordination
and overtime are unsupported. Human play requires the display and uses arrows,
X (pass/switch), C (shoot/check), and Enter (native pause).

Three or four controllers require the patched stable-retro frontend with explicit
EA 4-Way Play support. See [dependency build and reset details](docs/ARCHITECTURE.md#ea-4-way-play).
Two bots versus CPU retain the original two-controller path and work with older
stable-retro. Reset configures RAM and the adapter's polling register in memory;
ROM files and stored saves are unchanged. An explicit legacy `--num_players`
value must match the derived bot-plus-human count; it can normally be omitted.

```bash
nhl94 play --agent classic-v1 --env NHL94-Genesis-v0 --mode team_vs_game --team-size 3
nhl94 play --agent classic-v1 --env NHL94-Genesis-v0 --mode team_vs_player --team-size 3
nhl94 play --agent classic-v1 --env NHL94-Genesis-v0 --mode team_vs_game --team-size 3 --team-strategy independent
```

Seeded evaluation supports `coordinated`, `independent` (separate Classic bots),
and the existing `single` baseline. `--team-size 3` selects three home bots; the
single and away-side benchmarks retain their original one-/two-controller scope.
The default evaluation save is `SabresVsMightyDucks.ManualGoalie.Start`.

```bash
nhl94 benchmark-teammates --strategy coordinated --trials 3 --seed 94 \
  --output /tmp/teammates.json

# Record native inputs and decisions for a short first period.
nhl94 benchmark-teammates --seconds 30 --trace /tmp/teammates-inputs.jsonl \
  --output /tmp/teammates-short.json

# Replay those inputs from the verified reset boundary.
nhl94 benchmark-teammates --replay /tmp/teammates-inputs.jsonl \
  --output /tmp/teammates-replay.json
```

Multitap input traces use `nhl94-teammate-inputs-v2`, including controller mapping,
opponent type, reset setup and native/core hashes. Replay consumes the combined
recorded buttons; it does not rerun bots. Historical v1 traces retain their
one-/two-controller interpretation. The Python `play_match(human_input=...)`
interface accepts a scripted 12-button human input source and records that channel
when `trace` is supplied. Keyboard playback offers the existing video recorder.


Evaluation defaults to one complete first period; `--periods 3` runs regulation.
`--side away` evaluates the visitor with the existing one-/two-controller
strategies and `--workers 4` runs independent trials in four processes. Playback remains home-only. Top-level scores and shots are
ordered AI first, CPU second; period scoreboards retain home/visitor order.
Reports retain physical team IDs and live lineups.
`--teammate-support-options` opts into estimated support positions that consider
ordinary passes, one-timers, staying, braking and coasting. With three bots this
planner owns the outlet; the other supporter retains deep coverage. It remains
experimental.
Two-bot input traces additionally include sampled pass candidates and rejection reasons, live
contact estimates, native touch attribution and applied C edges. These forecasts
describe heuristic opportunities; native counters determine completed shots.
Reports include completion status, native-frame counts, scores, shots, pass and
one-timer attempts, observed controlled-teammate receptions, skater turnovers,
roles, input hashes and ROM/source provenance. Incomplete runs remain explicitly
incomplete. Input traces are local JSONL recordings for debugging, not learned
policy training datasets. This scripted prototype has no established strength
advantage over the single-controller agent.

## Classic

`classic-v1` (also `classic`) is the scripted gameplay agent. Full-team playback
uses selective manual-goalie control by default, with `deep-csave` positioning
on Genesis and shared classic positioning on SNES.
`--goalie-positioning classic` selects the original manual goalie;
`--goalie-policy off` disables manual-goalie AI.

The agent reacts to native emulator frames while retaining its configured
tactical decision interval. Use the [Classic guide](docs/CLASSIC.md) for timing,
goalie options, and supported optional behavior. `nhl94 play --help` lists the
available flags.

SNES uses `DucksVsSabres.ManualGoalie.Start` from the experimental stable-retro
integration. The project supplies its action scenario, so that integration does
not need its own `metadata.json` or `scenario.json`. SNES supports home or away
versus CPU, human versus Classic, and Classic versus Classic, with `FILTERED`
actions and `PostPlay`. Reset configures native controller assignments from the
supplied save; multiplayer needs no additional save. Controller 1 uses `--side`
(home by default); controller 2 uses the other team. In `player_vs_model`, the
human input controls controller 2. Manual goalies use X to take over, A to save,
Y to dive, and B to return/pass, with native feedback for catches and outlets.
`--goalie-policy off` leaves goalie positioning to the game. ROM RNG seeding,
training tasks, learned checkpoints, deep goalie controllers, and Genesis-only
experiments remain unsupported on SNES. Movement and pass forecasts use explicit
SNES estimates where live ratings or native mechanics have not been mapped.

## Development

```bash
python -m pip install -e '.[dev]'
python -m unittest discover -s tests -v
python -m pylint nhl94_ai tests
python -m build --wheel

# Run emulator checks with local game integrations and ROMs installed.
python tests/integration/smoke.py
python tests/integration/workflows.py
python -m tests.integration.snes
python -m tests.integration.teammates
```

Unit tests run without ROM files. Emulator checks in `tests/integration/` run
explicitly with the matching stable-retro integrations installed.

- [Architecture and extension points](docs/ARCHITECTURE.md)
- [Artifact and export contracts](docs/ARTIFACTS.md)
- [Gameplay examples](docs/NHL94-README.md)
- [Current limitations](docs/KNOWN_ISSUES.md)
- [Browser app](web/README.md)

The C++ runtime is maintained separately in `retro-ai-runtime`.

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
# Watch a scripted agent; requires the display extra.
nhl94 play --agent classic-v1 --env NHL94-Genesis-v0

# Watch Classic as Quebec against Montreal's built-in CPU.
nhl94 play --agent classic-v1 --env NHL94-Genesis-v0 --state CanadiensVsNordiques.start --side away --max_playback_speed 1.0

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

The `PvG` (Player vs Goalie) example uses
`MightyDucksVsAllStarCampbell.Shootout.NearGoalie.Start`, not a manual-goalie or
`.2P` save. Its named `pvg` input has 52 features: the selected player, opposing
goalie and net, with player-centered, rink-aligned geometry and live attributes
for both participants. Input variants are defined in `configs/model_input.json`;
`--model_input NAME` lets other tasks reuse them. The existing full-team input is
named `default`. The example trains a fresh `[128, 128]` MLP with PPO; environment
count and timestep budget are configured in `configs/training/pvg.json`.
A home goal gives +1, and creating a usable net opening before shot release
gives +0.05 at most once per episode. Windup while holding C remains eligible.
Failures have no penalty. C-pressed status and hold duration are included in
the player input. Each episode ends when the ROM finishes that shootout attempt.
Starts slightly vary player/goalie positions and player velocity around the
save, keeping the puck with its carrier; explicit seeds reproduce the variations.
Mean reward includes shaping; use goal totals divided by attempts for goal rate.
Add `--live` to watch, or `--num_env 1` to reduce parallelism. Checkpoints and logs
are saved under `~/OUTPUT/pvg/`. See [PvG](docs/ARCHITECTURE.md#pvg-player-vs-goalie).
Earlier 310-input PvG checkpoints require `--model_input default`; 50-input
checkpoints use `--model_input pvg-v3`. Start a fresh model for the 52-input
schema rather than reinterpreting those weights.

The sole scripted controller is [Classic V1](docs/CLASSIC_V1.md), formerly V4.
Its default offense restores the established carry/pass priorities and reception
cadence after the native-lookahead policy regressed. Use `--offense-lookahead`
to opt into that experimental policy; `--uncertain-carry` and `--chance-creation`
also use its native forecasting. See the [restoration evidence](docs/CLASSIC_V1.md#default-offense-restoration-2026-10-08).
Its defense chooses a safe lane/recovery target before selecting a skater and
reacts every emulator frame. The target appears as a green square on the ice.
Offense now evaluates advancement passes and purposeful cuts, with an amber
offensive target; its configured decision interval is unchanged. The initial
progressive-offense change regressed from 42 to 28 goals in the matched 50-period
comparison. The later one-timer fix increased scoring from 20 to 30 goals in
24 matched periods, but concessions also rose from 13 to 18; this is not evidence
of universally stronger play. `classic` is also an alias
for it; the original V1–V3 controllers have been removed.

For full-team Classic AI-versus-CPU playback, `--side away` transfers controller 1
to the away team and releases the home skater to the built-in CPU. Use a
single-controller home save, not a `.2P` save. Home remains the default; away
playback currently requires `FILTERED` buttons and the `PostPlay` task.

In `player_vs_model`, Classic controls P1/home and the keyboard controls P2/away.
Use **arrow keys** to skate, **X** to pass/switch/poke, **C** to shoot/boost/check,
**Z** to clear/hold, **Enter** to pause and **Esc** to quit. The two-controller
save is selected automatically. This mode uses `FILTERED` buttons; the AI's
green defensive target stays visible. See [gameplay examples](docs/NHL94-README.md).

Use `nhl94 train --live` for the live display. It replays the latest snapshot,
refreshed at each evaluation even when reward falls; the best checkpoint is
saved separately. Curricula show the display unless
`headless` is set in their configuration. Evolution Strategies remains available
through `python -m nhl94_ai.training.rl --alg es`.

### Target-only defensive policy

```bash
# Fresh PPO model: outputs a destination, not buttons.
nhl94 train --config configs/training/defense-target.json --live
```

This example uses separate `[256, 256]` actor and critic MLPs, configured in
`configs/training/defense-target-hyperparams.json`. Shared training defaults remain
`[128, 128]`. The target example samples a **tanh-squashed Gaussian**, starting at
standard deviation 0.5 before squashing; its entropy bonus measures the bounded
action distribution rather than encouraging unlimited raw Gaussian spread.
Inputs, rewards, and other PPO settings are unchanged. This applies to fresh
models; old checkpoints keep their original distributions when loaded.

`TARGET_POSITION` uses normal player switching and scripted skating, braking,
boost pulses, and local poke attempts. The policy can revise its destination
every four emulator frames; the controller executes every frame. DefenseZone
maps all outputs into our defensive zone, from the end boards to our blue line,
instead of allowing targets anywhere on the rink. Live viewing and playback
show the chosen target as a green square on both the game image and rink diagram,
with the actual/desired skaters identified on the diagram.
Existing button policies and defensive starts are unchanged. DefenseZone ends
with +1 for skater recovery or -1 for an opponent shot or goal. Goalie possession
alone gives no reward and does not end the episode; a newly recorded shot does.

See [target-position control](docs/ARCHITECTURE.md#target-position-control)
for playback, evaluation, controller limitations, and checkpoint compatibility.

## Development

```bash
python -m pip install -e '.[dev]'
python -m unittest discover -s tests -v
python -m pylint nhl94_ai tests
python -m build --wheel

# With local NHL94 ROMs installed:
python tests/integration/smoke.py
python tests/integration/workflows.py
```

Unit tests require no ROM. Emulator regression traces cover all three variants
and replay frozen inputs to preserve the original observations, rewards, and
timing. `python -m tests.integration.classic_defense` checks live movement
telemetry and per-frame/interval execution parity; `python -m
tests.integration.classic_offense` verifies actual advancement receptions and
cut movement; `python -m tests.integration.play_v1` checks deterministic playback.

- [Architecture and extension points](docs/ARCHITECTURE.md)
- [Artifact and runtime export contracts](docs/ARTIFACTS.md)
- [Training-task development skill](.github/skills/nhl94-task-development/SKILL.md)
- [Refactor plan](docs/REFACTOR_PLAN.md) and [verification](docs/REFACTOR_RESULTS.md)
- [Known inherited issues](docs/KNOWN_ISSUES.md)
- [NHL94 gameplay examples](docs/NHL94-README.md)

The C++ runtime is maintained separately in `retro-ai-runtime`.

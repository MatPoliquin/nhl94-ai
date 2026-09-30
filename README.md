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

# Play against Classic v1 at half speed (you control the away team).
nhl94 play --agent classic-v1 --mode player_vs_model --env NHL94-Genesis-v0 --max_playback_speed 0.5

# Train without a display. Add --live to watch progress.
nhl94 train --config configs/training/default.json --num_env 1

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

The sole scripted controller is [Classic V1](docs/CLASSIC_V1.md), formerly V4.
Its defense chooses a safe lane/recovery target before selecting a skater and
reacts every emulator frame. The target appears as a green square on the ice.
Offensive tactics and decision timing are unchanged. `classic` is also an alias
for it; the original V1–V3 controllers have been removed.

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
tests.integration.play_v1` checks deterministic defensive playback.

- [Architecture and extension points](docs/ARCHITECTURE.md)
- [Artifact and runtime export contracts](docs/ARTIFACTS.md)
- [Training-task development skill](.github/skills/nhl94-task-development/SKILL.md)
- [Refactor plan](docs/REFACTOR_PLAN.md) and [verification](docs/REFACTOR_RESULTS.md)
- [Known inherited issues](docs/KNOWN_ISSUES.md)
- [NHL94 gameplay examples](docs/NHL94-README.md)

The C++ runtime is maintained separately in `retro-ai-runtime`.

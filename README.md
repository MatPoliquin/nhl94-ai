# NHL94 AI

> **Work in progress**: The ML less baseline bot is available now. Training workflows are unavailable pending a redesign.


* Benchmark various Neural Networks architectures, LLMs and ML/RL algos on NHL94
* Host AI vs AI contest for NHL94
* Provide challenging, fun but beatable AI opponents that win through smart tactics rather than frame-perfect, superhuman reactions.

Supported games:
* NHL 94 Sega Genesis (full game, 2 on 2, 1 on 1 versions)
* NHL 94 Snes



## Install

```bash
sudo apt update
sudo apt-get install python3 python3-pip python3-venv git zlib1g-dev libopenmpi-dev ffmpeg cmake libgl1-mesa-dev

python3 -m venv .venv
source .venv/bin/activate

python3 -m pip install -e ".[display]"

git clone ---depth 1 https://github.com/Farama-Foundation/stable-retro.git ../stable-retro
python3 -m pip install --no-deps -e ../stable-retro
```

## Why NHL94?

Sparse scoring rewards, puck physics, player-specific ratings, PvP, and team coordination make NHL 94 a rich testbed for reinforcement learning and neural-network policies.

Supports up to four players/bots in configurations including:
* human vs. up to three bots
* bot vs original in game CPU
* bot vs bot
* 2 bots vs 2 bots
* 1 bot vs 3 bots


## NHL 94 Rom

You need to provide your own rom

```
python3 -m retro.import .
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

# Record seeded SNES periods, then replay native inputs without running the AI.
nhl94 record-snes --output /tmp/snes-session --periods 3 --seed 7
nhl94 replay-snes /tmp/snes-session/period-001/inputs.jsonl.gz

# Compare two Classic configurations on SNES, swapping sides for each seed.
nhl94 benchmark-snes --output /tmp/snes-benchmark --policy-a /path/to/a.json --policy-b /path/to/b.json --seeds 20 --frame-skip 4

# Watch Classic as Quebec against Montreal's built-in CPU.
nhl94 play --agent classic-v1 --env NHL94-Genesis-v0 --state CanadiensVsNordiques.start --side away --max_playback_speed 1.0

# Record the full 1080p window, including the decision inspector and debug panels.
nhl94 play --agent classic-v1 --env NHL94-Genesis-v0 --record-mp4 ../recordings/new-session.mp4

# Play against Classic v1 at half speed (you control the away team).
nhl94 play --agent classic-v1 --mode player_vs_model --env NHL94-Genesis-v0 --max_playback_speed 0.5

# Run bounded evaluation against the game's built-in opponent.
nhl94 evaluate --agent classic-v1 --episodes 1 --max_steps 100

nhl94 export --help
```


## External opponents

Install an adapter package separately, then pass its Python factory explicitly:

```bash
nhl94 external --adapter my_external_package.adapter:create \
  --pairs 10 --seed 26534 --periods 3 --output /tmp/external-results.json
```

To play against an installed adapter yourself, use `--human` and select your
side. This starts a playable window with the same keyboard and M30 controls as
`nhl94 play`:

```bash
# You: Ducks (away/controller 2). Adapter: Sabres (home/controller 1).
nhl94 external --adapter my_external_package.adapter:create \
  --human --fixture sabres-ducks --side away
```

## Learned policies

Task rewards currently return zero; reward-based evaluation totals are marked unavailable. Goals and other gameplay statistics remain available.

Training, collection, imitation learning, curriculum, and train-then-compare commands are explicit stubs that fail without creating runs or artifacts. Offense and goalie evaluation remain available.
Use matching game, action, observation, and checkpoint settings when loading a
policy. Keep each model together with its metadata sidecar.

```bash
# Inspect evaluation options for existing offense and goalie checkpoints.
nhl94 offense --help
nhl94 goalie --help

# Watch a compatible button policy.
nhl94 play --env NHL94-Genesis-v0 --nn MlpPolicy --model_1 /path/to/model.zip
```

## AI teammates

In Python teammate modes, `--side home|away` selects the bots' team (default:
home). Two or three Classic bots occupy controllers 1 through `--team-size`;
in `team_vs_player`, the human uses the next controller on the opposite team.
Goalies are automatic and `coordinated` is the default strategy, including
the shared sandwich defense. For Ducks bots against Sabres, use
`--state SabresVsMightyDucks.ManualGoalie.Start --side away`.
`benchmark-teammates` uses the same bot-side meaning.

```bash
nhl94 play --agent classic-v1 --env NHL94-Genesis-v0 --mode team_vs_game --team-size 3
nhl94 play --agent classic-v1 --env NHL94-Genesis-v0 --mode team_vs_player --team-size 3
nhl94 play --agent classic-v1 --env NHL94-Genesis-v0 --mode team_vs_game --team-size 3 --team-strategy independent
```


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

The [evaluation package guide](nhl94_ai/evaluation/README.md) maps gameplay,
benchmarks, recordings, replay experiments, and their shared helpers.
The [test guide](tests/README.md) covers unit tests, native integration checks,
and reusable fixtures.

## Citations
@misc{nhl94-ai,
  author = {Poliquin, Mathieu},
  title = {nhl94-ai, custom AI players for NHL 94 - Sega Genesis},
  year = {2026},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/MatPoliquin/nhl94-ai},
}

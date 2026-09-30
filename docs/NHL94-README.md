# NHL94 gameplay examples

The supported variants are `NHL941on1-Genesis-v0`, `NHL942on2-Genesis-v0`, and
`NHL94-Genesis-v0`. Run these examples from the repository root after installing
the package and its display extra. Game integrations and ROMs belong to stable-retro.

## Watch a scripted agent

```bash
nhl94 play --agent classic-v1 --env NHL941on1-Genesis-v0
nhl94 play --agent classic-v1 --env NHL942on2-Genesis-v0
nhl94 play --agent classic-v1 --env NHL94-Genesis-v0
```

## Train a scoring policy

```bash
nhl94 train --env NHL941on1-Genesis-v0 --nn MlpPolicy --rf ScoreGoal \
  --num_env 12 --num_timesteps 100000000 \
  --hyperparams configs/training/nhl94.json
```

Add `--live` to attach the training display. To watch a saved policy, pass its
checkpoint to `nhl94 play --model_1` with the same environment, architecture,
action type, and observation configuration used during training. See
[artifact contracts](ARTIFACTS.md) for older checkpoints with unknown schemas.

## Play against Classic v1

```bash
nhl94 play --agent classic-v1 --mode player_vs_model --env NHL94-Genesis-v0 \
  --max_playback_speed 0.5
```

Classic plays **controller 1 / home**; you play **controller 2 / away**. The mode
selects two controllers automatically and loads the corresponding `.2P` save.
Custom saves must assign controller 1 to home and controller 2 to away.
All three supported game variants work with `FILTERED` buttons; hockey-intent
and target-coordinate action modes are not supported for keyboard opponents.

| Key | Action |
| --- | --- |
| Arrow keys | Skate / aim |
| X | Pass with possession; switch or poke without it |
| C | Shoot with possession; boost/check without it |
| Z | Clear with possession; hold/hook without it |
| Enter | Pause |
| F1 | Disable/enable only your P2 keyboard input; the AI keeps playing |
| Esc | Quit |

Focus the game window for keyboard input. The AI's green defensive target remains
visible on the ice and rink diagram. Use `--max_playback_speed 1.0` for normal
speed, `0.5` for half speed, or `0.25` for quarter speed. Viewing speed does not
change either controller's in-game decision timing.

## Play against the game

```bash
nhl94 play --mode player_vs_game --env NHL94-Genesis-v0 --nn MlpPolicy \
  --rf PostPlay --hyperparams configs/training/nhl94.json
```

The debug display implementation is in
[`nhl94_ai/ui/debug.py`](../nhl94_ai/ui/debug.py).

## Project history

- [NHL94 Discord, including its AI subgroup](https://discord.gg/SDnKEXujDs)
- [Earlier 1-on-1 AI and reward-function demonstration](https://www.youtube.com/watch?v=UBXXn2amGUU)

# Classic gameplay

`classic-v1` and `classic` select the `ClassicAIV1` scripted controller. It supports
the NHL94 variants registered by the project; full-team goalie options require
`NHL94-Genesis-v0`, filtered buttons, and a compatible save.

## Play

```bash
nhl94 play --agent classic-v1 --env NHL94-Genesis-v0
nhl94 play --agent classic-v1 --mode player_vs_model --env NHL94-Genesis-v0
```

In human-versus-agent mode Classic uses controller 1/home; the human uses
controller 2/away. See [gameplay examples](NHL94-README.md) for keyboard controls.

## Decision timing

Normal dispatch calls the agent each native emulator frame. Defense, puck
feedback, input edges, and committed action lifecycles react at that cadence;
tactical offense uses the configured decision interval, normally four frames.
Playback speed changes display pacing without changing either interval.

Standalone Python callers use `predict_frame(state, frame_skip=4)` per native
frame. `--classic-control-interval N` explicitly samples and holds raw input
every N frames instead. It is separate from the tactical frame-skip setting.

## Goalies

Full-team playback defaults to selective manual goalie control and `deep-csave`
positioning. With no explicit save it selects the manual-goalie Sabres/Ducks
start. Explicit non-manual saves retain automatic goalies unless requested.

- `--goalie-policy off` disables manual goalie AI.
- `--goalie-positioning classic` selects the original manual positioning/saves.
- `--goalie-positioning deep` selects deeper pre-shot positioning.
- `--goalie-positioning deep-csave` combines that position with the C-save rule.
- `--goalie-model MODEL.zip` uses a compatible trained goalie checkpoint and its
  `.goalie.json` sidecar inside enabled manual goalie play.

`--goalie-planner` offers optional released-shot physics planners. A learned
goalie, alternative save timing, or a slower goalie interval can replace or
conflict with another controller. Unsupported combinations fail validation.
`nhl94 play --help` lists the accepted settings.

## Optional behavior

Flags such as `--cross-crease`, `--opportunistic-cross-crease`, `--deke`,
`--offense-lookahead`, `--uncertain-carry`, `--chance-creation`, and
`--classic-refinements` explicitly enable optional behavior. They are disabled
by default and can have compatibility restrictions. `--retention-model` loads
a compatible movement-filter artifact while preserving pass and shot priority.

Defense is implemented in `agents/defense.py`, default possession in
`agents/possession.py`, shared motion in `agents/motion.py`, passing in
`agents/passing.py`, and goalie control in `agents/goalie.py`. Scheduling and
action lifecycles are separate from those decisions. Diagnostic requests describe
the controller's intended input; native ownership and counters determine what
the game executed.

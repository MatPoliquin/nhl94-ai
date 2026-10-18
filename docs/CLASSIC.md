# Classic gameplay

`classic-v1` and `classic` select the `ClassicAIV1` scripted controller. It supports
the NHL94 variants registered by the project; full-team goalie options require
filtered buttons and a compatible Genesis or SNES save.

## Play

```bash
nhl94 play --agent classic-v1 --env NHL94-Genesis-v0
nhl94 play --agent classic-v1 --mode player_vs_model --env NHL94-Genesis-v0
nhl94 play --agent classic-v1 --env NHL94-Snes-v0
nhl94 play --agent classic-v1 --env NHL94-Snes-v0 --side away
nhl94 play --agent classic-v1 --env NHL94-Snes-v0 --mode player_vs_model
nhl94 play --agent classic-v1 --env NHL94-Snes-v0 --mode model_vs_model
```

In human-versus-agent mode Classic uses controller 1; the human uses controller
2. The default assignment is home/away. SNES `--side away` reverses these teams.
See [gameplay examples](NHL94-README.md) for keyboard controls. The SNES keyboard
uses S for X (select goalie), C for A (shoot/save), Z for Y (clear/dive), and X
for B (pass/switch/return).

SNES uses the shared offense, defense, scheduling, and pass/shot lifecycle with
a console backend. It supports either side versus CPU and two controllers for
human-versus-Classic or Classic-versus-Classic play. It defaults to
`DucksVsSabres.ManualGoalie.Start`, configuring native assignment tables at reset
for the requested sides and controller count. Its native `B` passes/switches, `A`
shoots/checks, and `Y` clears; these map to the same hockey control channels as
Genesis `B`, `C`, and `A`. The [original SNES manual](https://www.gamesdatabase.org/Media/SYSTEM/Nintendo_SNES/manual/Formated/NHL_-94_-_1993_-_Electronic_Arts.pdf)
describes these controls. Manual-goalie AI shares positioning, threat prediction,
save decisions, and outlets with Genesis. The backend supplies X takeover,
native A/Y save animation feedback, and motion estimates. Each multiplayer agent
keeps its own scheduler and action state, observing the same native frame in
world coordinates. Learned policies, training tasks, deep goalie controllers,
and optional Genesis experiments fail validation on SNES. Forecasts are estimates
where SNES ratings, animation timing, or receiver selection are not mapped.

## Decision timing

Normal dispatch calls the agent each native emulator frame. Defense, puck
feedback, input edges, and committed action lifecycles react at that cadence;
tactical offense uses the configured decision interval, normally four frames.
Playback speed changes display pacing without changing either interval.

Standalone Python callers use `predict_frame(state, frame_skip=4)` per native
frame. `--classic-control-interval N` explicitly samples and holds raw input
every N frames instead. It is separate from the tactical frame-skip setting.

## Goalies

Full-team playback defaults to selective manual goalie control. Genesis uses
`deep-csave` positioning and SNES uses `classic`. With no explicit save it selects
the console's manual-goalie Ducks/Sabres start. Explicit non-manual saves retain
automatic goalies unless requested. The following deep, learned, and planner
options apply to Genesis.

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

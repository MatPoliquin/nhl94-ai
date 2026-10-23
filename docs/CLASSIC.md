# Classic gameplay

`classic-v1` and `classic` select the `ClassicAIV1` scripted controller. It supports
the NHL94 variants registered by the project; full-team goalie options require
filtered buttons and a compatible Genesis or SNES save.

## Play

See the [README commands](../README.md#commands) for playback examples. Run them
from the repository root after installing the package and its display extra.
Genesis supports `NHL941on1-Genesis-v0`, `NHL942on2-Genesis-v0`, and
`NHL94-Genesis-v0`; SNES uses `NHL94-Snes-v0`. Python gameplay requires the
appropriate ROMs and saves in stable-retro.

For Genesis, add `--seed 12000` to vary episodes reproducibly. Any unsigned
32-bit seed is accepted; each reset uses the next value, wrapping at `2**32`.
Omit the option to retain the save's RNG state. Seeding changes randomness,
not saved lineups, positions, or ratings, and works on either side with automatic
or manual goalies. SNES playback does not accept `--seed`.

### Human controls

In human-versus-agent mode Classic uses controller 1; the human uses controller
2. The default assignment is home/away. SNES `--side away` reverses these teams.
Genesis selects a matching `.2P` save when available; custom two-player saves
must assign controller 1 to home and controller 2 to away. Human-versus-Classic
requires `FILTERED` buttons.

| Key | Genesis action |
| --- | --- |
| Arrow keys | Skate / aim |
| X | Pass with possession; switch or poke without it |
| C | Shoot with possession; boost/check without it |
| Z | Clear with possession; hold/hook without it |
| Enter | Game Start / pause |
| Space or P | Pause/resume emulator playback |
| F1 | Toggle human input; in human-versus-Classic, the AI keeps playing |
| Esc | Quit |

The SNES keyboard uses S for X (select goalie), C for A (shoot/save), Z for Y
(clear/dive), and X for B (pass/switch/return).

Focus the game window for keyboard input. Use `--max_playback_speed 1.0` for
normal speed, `0.5` for half speed, or `0.25` for quarter speed. This changes
display pacing without changing in-game decision timing. The debug display and
decision inspector are implemented in [`ui/debug.py`](../nhl94_ai/ui/debug.py).

To play directly against the built-in CPU:

```bash
nhl94 play --mode player_vs_game --env NHL94-Genesis-v0 --nn MlpPolicy \
  --rf PostPlay --hyperparams configs/training/nhl94.json
```

For existing learned policies, see [playback examples](../README.md#learned-policies)
and [artifact compatibility](ARTIFACTS.md). Match the checkpoint's environment,
architecture, action type, and observation configuration.

### AI teammates

`team_vs_game` runs two or three Classic bots against the built-in CPU;
`team_vs_player` adds an opposing human on the next controller. In both modes,
`--side home|away` selects the bots' team (default: home). These Genesis-only
modes use automatic goalies and three regulation periods. See
[AI teammates](../README.md#ai-teammates) for commands, saves, and replay.

The default `coordinated` strategy assigns a carrier/pursuer, an outlet, and
(with three bots) a deeper coverage skater, with shared sandwich defense.
`--team-strategy independent` runs separate Classic instances. Shared receiving
supports every bot as sender or receiver; unavailable skaters are excluded from
pursuit. `--teammate-support-options` enables optional outlet positioning while
the third bot retains distinct coverage.

Coordinated receiving keeps released passes through temporarily unknown
controller selections without extending their deadlines. Inputs stay neutral
until selection is known; interception, stoppage and confirmed selection loss
still cancel the commitment. Cues count only after C reaches the receiver's
actual controller, and native one-timer arming is tracked separately.
These correctness checks are enabled by default. The comparison option
`--no-teammate-transient-receiving` disables only preservation across unknown
selections. Earlier cue timing and new positioning remain separate opt-in tactics.

### SNES support

SNES shares Classic's offense, defense, pass/shot lifecycle, and manual-goalie
decisions with Genesis. It supports either side versus CPU, human-versus-Classic,
and Classic-versus-Classic play. With default manual-goalie settings, it loads
`DucksVsSabres.ManualGoalie.Start` and configures the requested sides and controller
count at reset. Learned policies, training tasks, deep goalie controllers, and
optional Genesis experiments fail validation on SNES. Forecasts are estimates
where SNES ratings, animation timing, or receiver selection are not mapped.
See [architecture](ARCHITECTURE.md) for console backends and controller ownership.

## Decision timing

Classic playback automatically reads [`configs/classic_bot.json`](../configs/classic_bot.json).
Edit it before starting a game, or select another file with `--classic-config PATH`.
The reader accepts `//` and `/* ... */` comments. Missing settings use the defaults;
unknown keys and invalid types or values fail with an error. Installed packages
without the checkout file use the same defaults.

- `decision_hz`: `60` (default), `15`, `10`, or `6`. At 60 Hz Classic observes
  every native frame, runs eligible tactical decisions, and updates inputs.
  Lower rates sample the current state every 4, 6, or 10 frames and hold all
  buttons between samples, including manual goalie inputs. They require
  `FILTERED` buttons and are unsupported in teammate modes.
- `ai_goalie`: `true` (default) enables the normal selective manual goalie default
  for supported modes/saves; `false` leaves goalie control to the game.
- `player_ratings_consideration`: `true` (default) uses live ratings. `false` uses
  neutral ratings for both teams across offense, defense, goalie and teammate
  decisions. Actual game ratings and physics remain unchanged; positions,
  velocities, energy and handedness are still observed.

Explicit `--classic-control-interval` (including `0`) overrides `decision_hz`;
explicit `--goalie-policy` overrides `ai_goalie`. An explicit control interval
retains the separate tactical frame-skip setting. These defaults apply to
playback, not benchmarks.

Normal dispatch calls the agent each native emulator frame. Defense, puck
feedback, input edges, and committed action lifecycles react at that cadence;
tactical offense uses the configured decision interval: one frame for 60 Hz
playback, or four frames for callers without the playback rate setting. Exclusive
actions such as goalie control and pass reception can temporarily own a frame.
Faster tactics preserve the normal four-frame shot hold and scale decision-counted
follow-through, setup and recovery cooldown budgets; native-frame timers stay unchanged.
Playback speed changes display pacing without changing either interval.

Standalone Python callers use `predict_frame(state, frame_skip=4)` per native
frame. `--classic-control-interval N` explicitly samples and holds raw input
every N frames instead. It is separate from the tactical frame-skip setting.
For a full 60 Hz external-opponent benchmark, use `nhl94 external --frame-skip 1`;
the external adapter receives every native frame, and Classic plans every eligible frame.

## Goalies

Full-team Classic playback against the CPU or a human defaults to selective
manual goalie control; SNES Classic-versus-Classic also uses this default.
Teammate modes use automatic goalies. Genesis uses `deep-csave` positioning and
SNES uses `classic`. With no explicit save, manual-goalie playback selects the
console's manual-goalie Ducks/Sabres start. Explicit non-manual saves retain
automatic goalies unless requested. The deep, learned, and planner options below
apply to Genesis.

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

For implementation details, see [architecture](ARCHITECTURE.md); for benchmarks
and replay diagnostics, see the [evaluation guide](../nhl94_ai/evaluation/README.md).
Diagnostic requests describe the controller's intended input; native ownership
and counters determine what the game executed.

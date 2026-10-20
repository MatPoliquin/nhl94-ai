# Plan: support away-team bots in Python

Status: implemented and verified in Python.

Verification: 82 targeted unit tests and six native test methods passed, including
all eight side/count/opponent layouts, physical controller movement, display input
routing, sandwich-defense safety, and a three-period away game with native
highlights and matching input-replay RAM hashes. The 24-run comparison completed
with all 12 pressure-on/off pairs matching initial RAM, ratings, lineup, ROM and
save hashes. Results are in `tmp/away-teammates/comparison.json`; full-game evidence
is in `tmp/away-teammates/away-regulation-highlights.json` and its input trace.
The comparison report records the later menu-only ownership-timeout fix and both
source hashes. That fix changes exception handling, not generated inputs.

## Goal

Allow two or three Classic bots to play on either team, against the built-in CPU
or a human, through `nhl94 play` and `nhl94 benchmark-teammates`.

In teammate modes, `--side` selects the **bots' team**. A human opponent controls
the other team. Preserve the existing home-side defaults and coordinated defense.
This work covers Python only.

## Current blockers

This is partly an application restriction, but removing the restrictions alone
would produce incorrect native assignments:

| Location | Current limitation |
|---|---|
| `nhl94_ai/config.py`, `EnvironmentConfig.from_args` | Requires home bots in all teammate playback modes. |
| `nhl94_ai/evaluation/play.py`, `NHL94Player.__init__` | Separately restricts away playback; startup text also labels teammate bots as home. |
| `nhl94_ai/game/ram.py`, `configure_teammates` | Rejects away multitap setups. Third-bot selection searches home slots `0..4`; human selection is hard-coded to away slot `6`. |
| `nhl94_ai/evaluation/teammates.py`, `play_match` | Rejects away three-bot and human-opponent evaluations. |
| Tests | Some tests explicitly expect away multitap configurations to be rejected. |

Useful existing pieces already accept a side: `controller_snapshot`,
`teammate_controller_info`, and `TeammateCoordinator`. The coordinator flips its
team perspective for away bots. Reuse these paths and verify them with native play.

## Intended controller layout

Keep input-channel order independent of home/away identity:

| Mode | Bots | Controller teams when bots are away |
|---|---:|---|
| Against CPU | 2 | Away, away; home controlled by CPU |
| Against CPU | 3 | Away, away, away; home controlled by CPU |
| Against human | 2 | Away, away, home human |
| Against human | 3 | Away, away, away, home human |

Home skater slots are `0..4`; away skater slots are `6..10`. Goalies are `5` and
`11`. Initial bot selections must be distinct skaters on the requested team.

## Implementation sequence

### 1. Correct reset-time controller assignments

Update `configure_teammates` before relaxing public validation.

- Calculate bot and opponent slot ranges from `side`.
- Select all bot skaters, including the third, from the bot range.
- Select the human's initial skater from the opposing range.
- Preserve deterministic selection and the existing home behavior.
- Keep automatic goalies and release joystick flags from unselected players.
- Retain validation of the supported full-team exhibition starting save. Supporting
  arbitrary midgame saves is outside this change.

Keep adapter initialization at reset. Do not repair assignments by writing RAM
during gameplay, or reuse the single-controller `restore_away_control` path for
multitap play.

### 2. Enable both sides through the CLI and benchmark

Update validation in `config.py`, `evaluation/play.py`, and
`evaluation/teammates.py` together.

- Accept home or away for two- and three-bot teammate modes.
- Keep the existing Genesis, Classic, button-schema, goalie, and controller-count
  requirements.
- Keep unsupported single-controller/multitap combinations rejected.
- Make help and startup messages identify the bots' side, opponent side, and
  human controller accurately.
- Keep `coordinated` as the default strategy. Away bots must use the same
  sandwich planner and safety checks as home bots.

### 3. Verify human input, display, and side-dependent state

Audit teammate playback and its environment/display adapters.

- Append human input after the bot channels, as today, irrespective of team.
- Route that input to the opposing physical team.
- Verify selected-player display, possession, attack direction, score labels,
  passing, and shooting from the away perspective.
- Verify that role assignment, including `contain`, `close`, and `cover`, follows
  actual selected players after switches and period changes.
- Preserve neutral input during transient native controller-table aliases.

### 4. Preserve recording and replay

Audit teammate trace headers, `trace_layout`, reset preparation, and replay.

- Record and validate the bot side, human side when present, controller count,
  and physical team assignments.
- Ensure replay reconstructs the same reset and side before applying input.
- Keep existing home traces readable; treat omitted side metadata according to
  the existing format's home-side meaning.
- Introduce a new schema version only if the existing layout cannot represent
  away ownership unambiguously.
- Check input replay against final RAM hashes for away three-bot games and
  away-bots-versus-human games.

### 5. Add tests, then run a bounded native comparison

**Unit coverage**

- Parameterize side × bot count × CPU/human opponent.
- Assert unique selections from the correct skater range and correct joystick flags.
- Replace rejection tests for newly supported configurations; retain tests for
  invalid counts, goalies, and unsupported modes.
- Exercise away transient aliases, cross-team selection detection, and human routing.
- Check mirrored defensive roles and target directions.

**Native coverage**

- Run short smoke tests for every supported layout before longer evaluations.
- Exercise faceoffs, switching, passing, loose-puck recovery, and human input on
  the home team against three away bots.
- Complete an away three-bot game through period transitions and highlights.
- Record and replay an away three-bot CPU run and a deterministic scripted-human run.
- Confirm the opponent remains CPU-controlled when no human is configured.

**Comparison**

Run both Sabres and Ducks with two and three bots, using seeds `94`, `95`, and
`96`, sandwiches enabled and disabled, for 120 game-clock seconds per run:
24 runs total. Compare paired initial RAM/ratings and report completion, goals,
shots, trap attempts, recoveries, and unsafe/transient exits. Do not infer improved
play from a small sample alone.

## Commands that must work after implementation

Watch three Ducks bots against CPU Sabres:

```bash
nhl94 play --agent classic-v1 --env NHL94-Genesis-v0 \
  --state SabresVsMightyDucks.ManualGoalie.Start \
  --mode team_vs_game --team-size 3 --side away \
  --team-strategy coordinated --max_playback_speed 1
```

Play as the Sabres against three Ducks bots:

```bash
nhl94 play --agent classic-v1 --env NHL94-Genesis-v0 \
  --state SabresVsMightyDucks.ManualGoalie.Start \
  --mode team_vs_player --team-size 3 --side away \
  --team-strategy coordinated --max_playback_speed 1
```

Benchmark three Ducks bots:

```bash
nhl94 benchmark-teammates --side away --team-size 3 \
  --seed 94 --trials 3 --seconds 120 --workers 3 \
  --output tmp/ducks-three-pressure.json
```

These are acceptance examples, not commands supported by the current code.

## Completion criteria

- All three commands work with correct physical team ownership.
- Existing home configurations and old replay formats still pass.
- No gameplay RAM assignment repairs are introduced.
- Native away matches complete and input replays reproduce final state.
- The comparison includes both teams with both bot counts.
- Documentation explains `--side` consistently; no Python teammate restriction
  remains solely because the bots are away.

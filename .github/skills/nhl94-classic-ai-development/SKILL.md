---
name: nhl94-classic-ai-development
description: >
  Develop and evaluate NHL94 Classic AI tactics, carrying, passing, one-timers,
  dekes, cross-crease and goalie behavior. Use for Classic strategy changes,
  excessive detours, benchmark regressions, native action timing, same-state
  counterfactual replays, or teammate-score/debug inspection. Require native
  cadence/provenance checks before interpreting replay or playing-strength data.
---

# NHL94 Classic AI development

Answer design questions without changing gameplay. For implementation requests,
separate a reproduced mechanics defect from a tactical improvement or strength
claim. Keep implementation and reusable checks in `nhl94_ai/`, not skill scripts
or `src/`. Preserve public flags, task/state names, button ordering, neural input
layouts and saved-model compatibility. Do not regenerate `models/` or modify the
separate stable-retro/runtime repositories unless explicitly requested.

For PPO, rewards, randomized resets and save initialization, use
[nhl94-task-development](../nhl94-task-development/SKILL.md).
Read only the relevant portions of [Classic V1](../../../docs/CLASSIC_V1.md),
[Architecture](../../../docs/ARCHITECTURE.md) and the
[ROM deep dive](../../../docs/nhl94%20Deep%20dive.md); do not begin with a broad
repository/disassembly search.

## Start at the affected contract

| Question | Source and focused validation |
| --- | --- |
| Live decisions, cached input, reception interrupts and one-timer lifecycle | `nhl94_ai/agents/classic_v1.py`, `nhl94_ai/agents/base.py`; `tests/unit/test_classic_v1.py`, `tests/unit/test_v1_one_timers.py` |
| Carry forecasting, defender reach, uncertainty and feasible exits | `nhl94_ai/agents/carry.py`, `nhl94_ai/agents/skating.py`, `nhl94_ai/agents/offense.py`; `tests/unit/test_carry_continuations.py`, `tests/unit/test_uncertain_carry.py` |
| Native passing selector, reception and finishing credit | `nhl94_ai/agents/passing.py`, `nhl94_ai/game/geometry.py`; `tests/unit/test_pass_geometry.py`, `tests/integration/classic_carry.py` |
| Optional finishers and goalie interactions | `nhl94_ai/agents/deke.py`, `nhl94_ai/agents/cross_crease.py`, `nhl94_ai/agents/goalie.py`; corresponding unit/explicit integration fixtures |
| RAM attributes, physical slots, controller assignment and native counters | `nhl94_ai/game/ram.py`, `nhl94_ai/game/state.py`; ROM deep dive integration-mapping sections |
| Production CPU measurements and event attribution | `nhl94_ai/evaluation/cpu_benchmark.py`, `nhl94_ai/evaluation/offense_metrics.py`, `nhl94_ai/evaluation/pass_outcomes.py` |
| Same-state replay, native endings and executable cadence gate | `nhl94_ai/evaluation/carry_replay.py`, `nhl94_ai/evaluation/carry_outcomes.py`, `nhl94_ai/evaluation/carry_replay_gate.py`; corresponding unit tests |
| Teammate scores, cached/historical labels and pause | `nhl94_ai/ui/debug.py`, `nhl94_ai/ui/targets.py`, `nhl94_ai/evaluation/play.py`; `tests/unit/test_debug_playback.py`, `tests/integration/debug_playback.py` |

## Establish the experiment before editing

Record game/save, AI side, physical teams/lineups, controller count, action schema,
decision cadence, goalie policy, optional finisher/risk flags and seed range.
`num_players=1` means one joystick, not one skater. Ordinary full-team state uses
`NHL94GameState(5)`. The usual save is
`NHL94-Genesis-v0` / `SabresVsMightyDucks.ManualGoalie.Start`.

Reproduce the specific defect first and retain its counterexample as a test.
State whether a route is certified safe, sampled-model viable or fallback.
Unsuccessful pursuit samples do not prove safety; failed certification does not
prove a bad play. Risk must remain visible even at zero opportunity value.
Replanning is not an escape: native position integration precedes new input,
and threatened routes need an achievable exit or release before contact.
Do not tune shooting weights to conceal a movement/admission defect.

## Verify the caller before expensive simulations

Standalone `ClassicAIV1Model.predict_game_state` is not production frame dispatch.
Setting a `frame_skip` attribute on that controller does not install the wrapper.
Call `predict_frame(state, frame_skip=4)` every emulator frame; production
`ScriptedAgent` does this when its configured frame skip is set. Observe real
decision-tick changes, including reception interrupts. Preserve decision/defense
clocks, input edges and cached input when injecting a diagnostic first choice.

Use the installed training/development interpreter, not an unconfigured system
Python. If no completed current-source default reference exists, generate a
small **correctness reference**, not a strength comparison:

```bash
python -m nhl94_ai benchmark-cpu \
  --agent classic-v1 --matchups sabres-ducks-manual ducks-sabres-manual \
  --trials 1 --seed 20262004 --seconds 300 --frame-skip 4 \
  --action-type FILTERED --goalie-policy off --workers 1 \
  --output /path/to/session/default-reference.json
```

Then run the fail-closed native gate before a long replay collection:

```bash
python -m nhl94_ai.evaluation.carry_replay_gate \
  --benchmark /path/to/session/default-reference.json \
  --output /path/to/session/replay-gate.json
```

The gate requires the standard default configuration, complete seed/side
coverage and current source hashes, then reruns a complete away period.
Clock, native frame count and applied-action SHA256 must match. It also checks
sources after the run and writes success evidence only after all checks pass.
A stale reference must be regenerated; never rewrite its historical hashes.
Gate failure blocks interpretation of replays, not a reason to record/relabel
a failing trace. The gate validates this caller/fixture, not playing strength.

## Validate mechanics, then useful attacks

Run the smallest relevant unit regression, then explicit native fixtures.
Use both rink ends, supported action schemas/cadences and live feedback when
those surfaces change. An isolated or idle-opponent fixture only diagnoses
mechanics; include actively responding defenders before judging viability.
Do not use `--record` to make a regression pass.

For counterfactuals, restore full emulator/RNG state and clone actual controller
history. Verify identical initial RAM for every branch and restore the shared
default-driven prefix after probes. Declare selection criteria before outcomes:
include input disagreements plus viable one-timer/safe-pass opportunity controls
when policies agree. Report overlaps, missing strata and bounded search lengths.
Forced rejected passes are probes, not normal-policy alternatives.

Follow shots to goals, catches/recoveries, misses, stoppages or explicit timeouts.
Use fresh native counters plus shooter attribution, not B/C or stale
`shot_player` alone. Use native `ltplayer` for touches; `last_puck_player` is
shot/controller history. Decode flags in GameState and consume named booleans.
Normal-release evidence is not exhaustive proof that an unobserved shot never
happened. Separate controlled possession, autonomous teammates, both goalies and
loose pucks; record completed receptions, chance windows, native attempts and
recorded shots. A retained or loose puck is not automatically a useful attack.
Keep shot follow-through separate from confirmed ordinary turnovers.

## Measure playing strength without changing the policy

Unit/native checks and selected possessions do not establish a stronger policy.
When strength is being evaluated, complete the usual comparison: **20 seeds per
AI side per policy**, 300-clock-second first periods, four-frame offense and
`FILTERED`: **80 periods total**. Use the same saves, seeds, teams/ratings and
feature settings for both policies. Reused seeds are development comparisons;
a held-out claim requires a disjoint declared seed range.

```bash
python -m nhl94_ai benchmark-cpu \
  --agent classic-v1 --matchups sabres-ducks-manual ducks-sabres-manual \
  --trials 20 --seed 20262004 --seconds 300 --frame-skip 4 \
  --action-type FILTERED --goalie-policy off --workers 4 \
  --output docs/benchmarks/classic-v1-CHANGE-default.json
```

Run the matching candidate command with its **explicit** opt-in flag and a
separate output. Freeze sources during measurements; capture hashes/versions and
validate completed periods, identical starts/lineups, active actors and balanced
one-timer accounting. Do independent documentation/analysis while commands run,
not edits to measured policy/helpers. Never call old prototype results a
measurement of a new revision.

Report each team as well as combined GF-GA and W/D/L. Include scoreless periods,
native one-timer attempts/goals, ordinary and zone turnovers, recorded/opponent
shots and relevant phase-tagged events. Counts can overlap; recorded-shot
follow-through metrics differ from complete release tracking. More shots,
possession or contact counts alone do not prove better decisions or causality.
Explain genuine uncertainty without substituting endless pilots for a requested
full benchmark. Keep experimental tactics opt-in unless promotion is requested.

## Finish with persistent evidence

Update existing architecture/tactic docs, preserve negative/historical reports,
and explicitly supersede invalid interpretations instead of hiding them.
Do not store temporary scores, seed sets or task status as durable conventions.
Finish implementation changes with:

```bash
python -m unittest discover -s tests -v
python -m pylint nhl94_ai tests
```

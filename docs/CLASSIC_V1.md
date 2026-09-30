# Classic V1

V1 retains the former V4 offensive controller and adds target-first reactive
defense, starting in `nhl94_ai/agents/classic_v1.py`. The original V1–V3 implementations
have been removed. Use `classic-v1` (or `classic`); old versioned command names
are not retained.

Historical results below and the archived benchmark files keep their original
V4/V2/V3 labels, sources, and hashes. They are not new V1 benchmark runs, and
comparisons against the removed controllers cannot be rerun from this tree.

Offense retains the pre-Roy decision tree, one-timer setup movement and
accuracy-aware positioning, along with pass-outcome diagnostics:

1. Pass to a nearby skater when the goalie owns the puck.
2. Shoot across the goalie at middle height from close range, moving closer for
   less accurate shooters. Generate a fresh C press and hold aim through the swing.
3. Otherwise, take an open cross-slot pass to a suitable shooter and attempt a one-timer.
4. Briefly move across a potential receiver to create a passing opportunity;
   otherwise carry toward the slot and across the goalie.
5. Without possession, choose a defensive destination, then a skater to execute it.

Possession comes from the engine's puck-owner slot. An opponent taking possession
ends the shot follow-through. Otherwise, aim is held for six decisions after
the shot press, including after the puck becomes loose.
Shot left/right refers to the physical goal mouth, even when
attacking the lower net. There is no inheritance from other agents and no model asset.
Offensive timings use agent decisions, normally every four emulator frames.
Defense and its cooldowns run every emulator frame.

Optional read-only goalie motion observations from the experiment remain
available, but V1 uses goalie position for aim. Legacy velocity
fields and neural input arrays remain unchanged.

## Target-first defense

`agents/defense.py` keeps target selection separate from player selection:

| Mode | Target and safety rule |
| --- | --- |
| `protect-lane` | Stay between the predicted carrier and our goal; bias toward the goal-mouth side less covered by the goalie. |
| `contain-boards` | Close the inside escape from the boards while remaining goal-side; do not leave an open slot receiver without another defender. |
| `recover-safe` | Reach a predicted low puck with time to settle possession before an opponent can contest it. |
| `intercept-pass` | Reach an intermediate trajectory point before reception, including switch delay and a safety margin. |
| `deny-reception` | When interception is unsafe, cover the predicted receiver's shooting lane immediately, before the puck arrives. |

One shared movement estimator uses actual momentum, facing, effective speed,
agility, weight and roster energy. It models turning/braking costs and the ROM's
energy-dependent speed limit; missing skater feedback uses conservative own-team
and optimistic opponent estimates, identified in diagnostics. Missing full puck
motion/height never justifies claiming a safe interception.

Prediction is bounded to 64 frames, sampled every four frames, with friction
and height estimates. Prediction stops at uncertain wall/net contacts rather
than inventing exact bounces. A live pass-target slot is only accepted when the
observed puck trajectory reaches that receiver; stale pass targets are ignored.
There is no assumed reception delay for a one-timer threat. Arrival estimates
are heuristics, not a perfect simulator or a promise that an attacker will
continue moving straight.

Safe recovery requires an arrival margin of at least five frames (ten with
missing skater feedback), plus time to settle the puck. It rejects losing races,
uncontrolled fast receptions and chasing beyond the puck with no covering
defender. The planner considers only the current skater or a plausibly obtainable
switch when declaring a puck interceptable.

Selection considers travel time, switching delay, current momentum and the lane
left behind. Locked/inactive skaters are excluded using the live selection flags,
not the legacy animation field. B cannot address an arbitrary slot: its nearest
projected-puck search includes the current skater, in which case it requests a
sweep check instead. The projection uses signed velocity high bytes; equally
distant candidates favor the later slot.

The ideal defender is shown separately from the obtainable choice. Selection
compares the current skater with the predicted B selection, so an unreachable
ideal no longer prevents switching to another useful teammate. More than four
frames of advantage are required after a six-frame input/selection allowance
and any remaining cooldown. This anti-churn margin cannot override a safe puck
interception deadline.

Requests use a fresh B press followed by release. The normal switch interval is
12 frames; the controller observes the actual slot for up to eight frames rather
than treating a press as success. Unconfirmed requests retry with a backoff of
four frames per consecutive attempt, capped at 24 frames, not a permanent
two-attempt lockout. Unexpected selections are reported and replanned from the
actual skater. The current skater keeps moving toward the target while waiting;
boost/poke requests do not interrupt pending switch input. No direct player-control
or gameplay RAM writes are used by the agent.

Hockey-intent possession gates use the engine owner when available, including
negative loose-puck sentinels. Stale visual possession stars can no longer suppress
a defensive switch or turn its input into a passing macro. Legacy star-based
gating remains the fallback when the owner field is absent; neural observations
and action schemas are unchanged.

Boost requires current facing alignment, sufficient energy, an unobstructed
route and braking room; it is not pressed continuously. Pokes require a close,
aligned predicted contact while remaining goal-side. Both use fresh button
edges. An opponent taking possession cancels an offensive pending sequence;
friendly possession releases defensive B/C requests within the current interval.

### Debugging and execution

Normal `nhl94 play --agent classic-v1` shows the **green tactical target square on
the ice**, plus the same square in the rink diagram. Cyan identifies the desired
defender; the overlay also shows the actual defender, waypoint, projected puck
path, reason, arrival estimates, receiver and missing feedback. Ideal/obtainable
skaters, the predicted B selection, switch status and the last observed result
explain why it switches or retains control. It uses the existing pre-step camera
projection and clears on resets.

`AgentOutput.diagnostics["classic_defense"]` exposes the same information without
a display. Evaluation includes the last defensive decision per episode; CPU
benchmarks count tactical frames, switch/poke/boost requests and controlled
recoveries. Requests are not counted as successful switches or checks.
Play, scripted evaluation, collection, DAgger teacher observations and raw
benchmarks all use `predict_frame`. The learned target-position controller,
neural observation ordering, tasks/rewards and bundled model assets are unchanged.
`python -m tests.integration.classic_defense` exercises a useful but non-ideal
switch through the real ROM in both action formats, checks selection feedback
in all three variants, and compares per-frame/four-frame caller behavior.

### Initial paired CPU measurement

Eight complete 300-second periods per matchup used the same seeds
290900–290907 before and after the defense change, with filtered buttons and a
four-frame offensive interval. This was an exploratory development comparison,
not a held-out estimate or full-game win rate. It predates the switching fixes
described above:

| Matchup | Goals conceded, before → after | Shots conceded, before → after | Opponent one-timers, before → after |
| --- | --- | --- | --- |
| Pittsburgh vs Ottawa | 7 → 2 | 32 → 30 | 12 → 5 |
| Ottawa vs Pittsburgh | 15 → 3 | 52 → 32 | 11 → 12 |
| Quebec vs Montreal | 7 → 6 | 46 → 34 | 8 → 12 |
| Total, 24 periods | 29 → 11 | 130 → 96 | 31 → 29 |

Overall concessions improved in this sample, but one-timer suppression did not
improve on every roster. Different defensive play also changes offensive
opportunities even though the offensive decision rules are preserved.
See the [before](benchmarks/classic-v1-defense-before.json) and
[after](benchmarks/classic-v1-defense-after.json) reports for individual periods
and source hashes.

### Switching follow-up: 50 Ottawa periods against Pittsburgh

After the switching fixes, Classic V1 controlled Ottawa against Pittsburgh's
actual in-game CPU for 50 complete five-minute first periods. These used fresh
ROM seeds 20260930–20260979, fixed starting rosters, filtered buttons, per-frame
defense and a four-frame offensive interval. Gameplay code was not changed
during the measurement.

| Metric | Result |
| --- | --- |
| Goals conceded | 27 total; 0.54 per period |
| Shutout periods | 29/50 (58%) |
| Periods conceding one goal | 15/50 |
| Periods conceding two goals | 6/50 |
| Most goals conceded in a period | 2 |
| Shots conceded | 217 total; 4.34 per period |
| Ottawa goals scored | 35 |
| Period outcomes | 19 wins, 17 draws, 14 losses |
| Pittsburgh one-timers / one-timer goals | 66 / 6 |

The defense does **not** stop every goal: Pittsburgh scored in 21 of 50 periods.
These are team results, including the CPU-controlled goalie and other skaters,
not proof that the controlled defender blocked every shot in a shutout. They are
not full-game results or a paired estimate of the switching fix's effect; the
earlier measurements used different seeds and fewer trials.

All 50 periods reached a zero clock and produced distinct action traces. The
[raw report](benchmarks/classic-v1-switching-senators-penguins-50-periods.json)
contains every score, seed, lineup, action hash and source hash. Reproduce with:

```bash
nhl94 benchmark-cpu --agent classic-v1 --matchups senators-penguins \
  --trials 50 --seed 20260930 --seconds 300 --frame-skip 4 \
  --action-type FILTERED --workers 4 \
  --output docs/benchmarks/classic-v1-switching-senators-penguins-50-periods.json
```

## One-timers

V1 preserves its close-range direct shot as the first choice. Otherwise, it
considers a teammate across the slot: the passer must be beyond attack-relative
Y=100, the receiver beyond Y=175–190 depending on accuracy and below Y=245,
and the pass must be 30–150 units long with less than 80 units of vertical
separation. Receivers must be within 70 units of the middle, on the opposite
side of the centerline, with more than 30 units of lateral separation.
Opposing skaters must be at least 14 units from the pass segment, including its
endpoints. Falling receivers are excluded. This is a first-valid-receiver scan,
not an option-scoring system.

When no pass or close shot is available, a carrier between Y=140 and 215 can
move toward the opposite side of a potential receiver for at most eight
decisions. The receiver must already be near shooting range. A 48-decision
cooldown prevents repeated sideways movement; a turnover or receiver knockdown
cancels the setup. This creates a passing angle without waiting indefinitely.

Live effective shot accuracy is read from each skater's RAM record. Direct-shot
depth is `218 + max(0, 15 - accuracy) * 0.4`; the minimum one-timer receiving
depth is `175 + max(0, 15 - accuracy)`. An unknown attribute falls back to 15;
a real zero remains zero. These are heuristic position thresholds, not scoring
probabilities. The additional fields do not change neural observation ordering
or shapes. Goalie position determines normal-shot aim; goalie ratings
are not used to assign a scoring probability.

V1 aims the pass and presses B, then releases B. While the puck is loose it
generates fresh C presses to activate the ROM's one-timer routine. The ROM
chooses the actual pass recipient from the direction and handles one-timer aim.
V1 stops pressing C when the intended receiver enters the one-timer animation.
It cancels on an interception, clean reception, receiver knockdown, or a
18-decision timeout. A 24-decision pass cooldown prevents immediate retries.
Pending state stores player slots so it remains valid when state objects are
rebuilt. Episode reset clears the sequence and cooldown.

Both action formats are supported. Pass aim now selects the nearest of eight
pad directions: a shallow cross-ice pass uses left/right, rather than inheriting
the diagonal input used for skating toward a point. The shared intent processor
uses the same geometry for targeted passes and its `ONE_TIMER` macro.
In hockey-intent mode, the targeted-pass
macro now holds B and its selected direction for four emulator frames before
release. A one-frame B pulse could release before the ROM sampled the requested
direction and send the pass along the player's old facing. This fixes targeted
passes and the existing `ONE_TIMER` macro for all intent users. Episode reset
also clears any held B/aim. The six-field action schema remains unchanged.

## CPU benchmark

The installed command runs the actual in-game opponent:

```bash
nhl94 benchmark-cpu --agent classic-v1 --trials 20 --seed 12000 --workers 4 \
  --output /tmp/classic-v1-cpu.json
```

It covers Pittsburgh against Ottawa, Ottawa against Pittsburgh, and Quebec
against Montreal. `--matchups` selects a subset; `--action-type HOCKEY_INTENT_DPAD`
tests the intent processor. Every trial runs to the actual first-period clock
reaching zero. Incomplete trials fail rather than count partial scores.
The ROM's roster pointers identify starting players, including Patrick Roy.

Away trials transfer controller 1 and its player-control flags to the away
team, releasing the home team to the CPU. A second human controller with no
input is never substituted for the CPU. Away observations use corrected RAM
control; home trials use normal play's star-derived inference. Both step the
emulator directly, bypassing the UI and PostPlay early ending. Reports include
scores, one-timer counters, setup decisions, lineups, source and action hashes.
Both benchmarks also record the outcome of every deliberate V1 setup pass.
A fresh ROM pass counter validates the selected recipient; a team one-timer
counter plus the shooter slot confirms execution. Interceptions, possession
lost before release, receptions without a shot, returns to the passer, and
timeouts are distinct outcomes. A wrong recipient is an additional flag and
can overlap any outcome. Unfinished plays at the period boundary are recorded
separately. These diagnostics observe each emulator frame without changing input.

Replaying the previous V4's 20 Montreal periods (seeds 9000–9019) reproduced
every action hash, score, shot count, and frame count. Of 39 deliberate setups,
23 were intercepted, nine produced one-timer shots, three were received without
a shot, three timed out in flight, and one returned to the passer. Only two
selected the wrong recipient. Pass interception, rather than direction selection,
was the main failure in that sample. ROM one-timer totals can also include
incidental teammate shots, so they are reported separately from setup outcomes.

## Historical default-play regression and restoration

A report of worse play used the former `classic-v4` name; its current equivalent is:

```bash
nhl94 play --agent classic-v1 --env NHL94-Genesis-v0
```

It runs Pittsburgh against Ottawa with filtered buttons. Replaying the actual
`NHL94Player` path with rendering disabled preserves the save's original RNG,
normal observations, reset behavior, and four-frame decision cadence. Unlike
the CPU benchmark, this replay ends at the first PostPlay terminal transition,
with nine seconds remaining; these scores are not complete-period benchmarks.

| Controller | Pittsburgh goals–Ottawa goals | Pittsburgh shots | Pittsburgh one-timers |
| --- | --- | ---: | ---: |
| Previous V4, now restored | 1–1 | 3 | 1 |
| Roy experiment | 1–0 | 1 | 0 |
| Early rebound recovery only | 1–0 | 1 | 0 |
| Goalie changes only | 1–1 | 3 | 1 |
| Roy experiment without the deke | 1–0 | 1 | 0 |

The experimental controller won this session but produced less offense. Its
actions were identical to the rebound-only and no-deke variants; no deke
occurred in this replay. The previous controller and goalie-only variant also
had identical actions. That isolates the action-sequence change in this
particular replay to early rebound recovery, without establishing that it
always worsens scores or that every lost shot was directly caused by it.

A separate diagnostic reused seeds 12000–12009, with ten complete periods
per CPU matchup. Pittsburgh's record was 6–2–2 for the previous controller,
3–4–3 for the Roy experiment, 4–4–2 for rebound-only, and 5–3–2 for goalie-only.
The goalie-only variant also worsened Montreal results (1–4–5 versus the
previous controller's 2–3–5). These ablations did not establish a consistently
better subset of the changes, so all experimental tactics were removed from
the default. This restores the exact prior controller source, while preserving
the diagnostic tooling and one-timer features that predated the experiment.

The [play replay](benchmarks/classic-v4-roy-play-regression.json) and
[seeded ablations](benchmarks/classic-v4-roy-ablation.json) retain controller
sources, input hashes, and results. The defensive change intentionally replaces
that complete-play trajectory. `tests/integration/play_v1.py` now checks
deterministic reactive playback; unit tests separately preserve offensive
decision/held-action parity. This complements the raw-emulator benchmarks.

## Rolled-back experiment: goalie timing and rebound recovery

The following results describe the experimental 256-line controller, which is
no longer the default. “Current” in this historical table refers to that
experiment; “previous” is the controller now restored. The experiment used a
bounded deke, projected goalie position, high shots against stacked pads, and
immediate recovery after the puck left the shooter.

The controller was frozen before validation on seeds 12000–12019. Each CPU
fixture ran 20 complete first periods with filtered buttons and four-frame
action repeat. The previous 223-line V4 was rerun on the same seeds and protocol.
Records below are wins–draws–losses, from V4's perspective.

| V4 team / CPU opponent | Previous V4 | Current V4 | Previous → current goals for–against |
| --- | --- | --- | --- |
| Pittsburgh / Ottawa | 7–8–5 | 7–7–6 | 13–12 → 13–10 |
| Ottawa / Pittsburgh | 4–6–10 | 4–5–11 | 11–23 → 12–27 |
| Quebec / Montreal (Roy) | 3–6–11 | 5–9–6 | 10–20 → 15–15 |

Against Montreal, V4's shots increased from 32 to 45 while Montreal's fell
from 126 to 109. One-timer shots increased from 12 to 13 and one-timer goals
from three to five. This is an improvement in this sample, not a reliable
winning record. Ottawa's result deteriorated slightly. Across all 60 CPU
periods, goals changed from 34–55 to 40–52 and one-timer shots from 32 to 40.
The combined test does not isolate the effect of goalie timing from rebound
recovery, and changing one decision changes the subsequent game trajectory.

| Scripted opponent | Periods | V4 wins–draws–losses | Goals for–against |
| --- | ---: | --- | --- |
| Classic V2 | 40 | 20–14–6 | 35–16 |
| Classic V3 | 40 | 22–12–6 | 38–14 |

Each scripted comparison uses both team assignments for every seed. V4 remains
a single decision tree, with no V2/V3 inheritance or learned parameters.

Reports: [current CPU](benchmarks/classic-v4-roy-cpu.json),
[previous V4 on matching CPU seeds](benchmarks/classic-v4-roy-previous-cpu.json),
[V2/V3 comparison](benchmarks/classic-v4-roy-h2h.json), and
[the earlier Montreal pass diagnostic](benchmarks/classic-v4-roy-pass-diagnostic.json).
The baseline reports include the frozen controller source. The current CPU
report records 18 interceptions and 13 confirmed one-timer shots among 41
Montreal setups; nine setups selected a different recipient. These overlapping
recipient flags remain useful diagnostics, not a guarantee that pass aim is solved.

An additional [30 intent-control CPU periods](benchmarks/classic-v4-roy-cpu-intents.json),
seeds 12000–12009, all completed and produced 24 one-timer shots and nine
one-timer goals. Records were 5–3–2 as Pittsburgh, 0–4–6 as Ottawa, and 0–8–2
as Quebec against Montreal. These are separate results: the filtered-button
advantage cannot be assumed for hockey intents.

Development used seeds 11000–11005 for CPU games and 11000–11003 for paired
scripted games. A slot-protection prototype helped Ottawa but worsened the
Montreal result. A moving-defender pass filter did not show a consistent
benefit. Neither is included in the final controller. The
[development archive](benchmarks/classic-v4-roy-development.json) retains each
tested controller, results, and notes about superseded velocity assumptions.
The final validation above was not used to retune the controller.

## Previous version: setup movement and accuracy changes

The final 223-line controller was frozen before testing seeds 9000–9019.
Twenty complete first periods per CPU matchup gave the following records
(wins–draws–losses). The previous V4 was rerun on the same seeds and protocol.

| V4 team / CPU opponent | Previous V4 | Current V4 | Current goals for–against | Previous → current one-timers |
| --- | --- | --- | --- | --- |
| Pittsburgh / Ottawa | 5–7–8 | 7–8–5 | 13–11 | 4 → 10 |
| Ottawa / Pittsburgh | 3–7–10 | 4–10–6 | 10–16 | 5 → 14 |
| Quebec / Montreal (Roy) | 6–8–6 | 7–4–9 | 17–23 | 7 → 9 |

Across the 60 CPU periods, V4's record changed from 14–22–24 to 18–22–20.
Actual one-timer shots increased from 16 to 33 and one-timer goals from 5 to 14.
Deliberate setup passes increased from 31 to 109. More setup passes do not
guarantee more goals: total goals scored fell from 43 to 40, while conceded
goals fell from 62 to 50. Montreal remains a weakness and its record worsened
in this sample. No claim is made that V4 reliably beats every CPU roster.

One-timers remain opportunistic: 37/60 current CPU periods had no recorded
one-timer shot, versus 46/60 previously. The policy now creates and attempts
more plays, but interceptions, missed receptions, and game situations still
prevent many from becoming shots.

On the same fresh seed range, paired home/away matches against the scripted
opponents produced:

| Opponent | Periods | V4 wins–draws–losses | Goals for–against | V4 one-timers |
| --- | ---: | --- | --- | ---: |
| Classic V2 | 40 | 14–16–10 | 22–14 | 27 |
| Classic V3 | 40 | 16–17–7 | 32–15 | 34 |

Reports: [current CPU](benchmarks/classic-v4-roster-cpu.json),
[previous V4 on the same CPU seeds](benchmarks/classic-v4-roster-previous-cpu.json),
and [V2/V3 comparison](benchmarks/classic-v4-roster-h2h.json).
The baseline report includes its frozen source. Timings were about 29 µs per
V4 decision versus 256/295 µs for V2/V3 on this machine.

An additional [30 CPU periods using hockey intents](benchmarks/classic-v4-roster-cpu-intents.json),
seeds 9000–9009, all completed and produced 17 one-timer shots and eight
one-timer goals. Records were 4–4–2 as Pittsburgh, 1–8–1 as Ottawa, and 1–4–5
as Quebec against Montreal. These are separate intent results, not an extension
of the filtered-control sample.

Development used seeds 6000–6005. An earlier candidate then failed its V2
check on seeds 8000–8019 (10–13–17), despite improved Ottawa CPU results.
Those [CPU](benchmarks/classic-v4-roster-initial-cpu.json) and
[head-to-head](benchmarks/classic-v4-roster-initial-h2h.json) reports are retained.
Investigation found shallow passes using diagonal skating input; correcting
pass aim was followed by the separate 9000–9019 validation above. This is a
finite sample of fixed starting lineups, not a universal strength guarantee.

## Previous version: initial one-timer results

The earlier controller was tested on seeds 3000–3019, using both team assignments
against each opponent. All 120 first-period trials completed.

| Opponent | V4 wins | Draws | V4 losses | V4 goals | Opponent goals |
| --- | ---: | ---: | ---: | ---: | ---: |
| V4 with one-timers disabled | 17 | 12 | 11 | 33 | 29 |
| Classic V2 | 17 | 15 | 8 | 31 | 21 |
| Classic V3 | 18 | 15 | 7 | 29 | 14 |

The ROM recorded **55 one-timer attempts and 22 one-timer goals** for V4 across
these trials. These are team counters, including any incidental CPU teammate
one-timers; they are separate from the 100 deliberate V4 setup decisions.
Setup-decision instrumentation applies to V4 only, while ROM attempt/goal
counters apply to every opponent. That controller was 188 source lines.
Mean decisions remained around 25 µs, versus 244 µs for V2 and 272 µs for V3.

The [one-timer report](benchmarks/classic-v4-one-timers.json) contains every match,
counter, and source hash. `classic-v4-direct` (now `classic-v1-direct`) is a benchmark-only ablation that
disables the pass/setup branch. At that revision, an emulator regression verified
that disabling it reproduced the original V4 exactly. The current ablation
also uses the new accuracy-aware shooting and corrected outlet aim, so it is
no longer the original controller.
The head-to-head results favor the new branch in this sample; they do not
establish superiority on every configuration.

## Previous version: CPU matchup diagnostic

A report of a period without one-timers prompted a separate CPU evaluation.
The controller was unchanged. Seeds 4200–4209 each played one 300-second first
period in three configurations, with filtered buttons repeated for four frames.
All 30 periods completed. Records below are from V4's perspective.

| V4 team | CPU opponent | Wins | Draws | Losses | Goals for–against | One-timers | Periods without one-timers |
| --- | --- | ---: | ---: | ---: | --- | ---: | ---: |
| Pittsburgh | Ottawa | 5 | 3 | 2 | 8–2 | 3 | 7/10 |
| Ottawa | Pittsburgh | 1 | 3 | 6 | 5–13 | 1 | 9/10 |
| Quebec | Montreal, with Patrick Roy | 2 | 3 | 5 | 4–8 | 1 | 9/10 |

Starting names were read from the ROM roster using the live player indices;
Montreal's goalie was Patrick Roy. Existing save-state rosters and attributes
were retained. To play the away team against the CPU, the probe transferred
controller 1 and its player-control flags to the away team, then used the
benchmark's control/possession corrections and physical away-team view.
Home trials used normal star-derived control inference. The probes stepped the
raw emulator directly to clock zero, bypassing the UI and PostPlay early ending.
These are diagnostic agent trials, not exact reproductions of the play UI.

The [CPU diagnostic report](benchmarks/classic-v4-cpu-diagnostic.json) includes
individual periods, initial lineups, live accuracy values, source hashes,
protocol details, and the temporary probe scripts needed to repeat the run.
Ten seeds per fixed lineup are a small sample; these tests do not isolate the
effect of goalie ratings from the rest of each opposing team. They do show that
the earlier advantage over V2/V3 does not establish strength against the CPU
with Ottawa or against Montreal.

One-timer execution worked, but that policy seldom chose a setup. Its narrow
cross-slot conditions and priority for direct shots leave few opportunities;
it had no movement rule that deliberately created a passing opportunity.
A separate seed-42 Pittsburgh CPU probe produced zero setups with both normal
control inference and corrected RAM control. Fixing control inference alone
therefore does not resolve the infrequency.

That revision used goalie position when aiming but did not use goalie ratings
or skater shot accuracy when choosing a shot or pass. The new behavior described
above addresses setup movement and skater accuracy. More attempted one-timers
alone are not evidence of a stronger agent.

## Original V4 results, before one-timers

The original version led both opponents over 80 fresh-seed trials, seeds 2000–2019,
with both team assignments for every seed. All periods completed.

| Opponent | V4 wins | Draws | V4 losses | V4 goals | Opponent goals |
| --- | ---: | ---: | ---: | ---: | ---: |
| Classic V2 | 19 | 13 | 8 | 29 | 15 |
| Classic V3 | 19 | 12 | 9 | 36 | 21 |

V4's home/away records were 10–6–4 / 9–7–4 against V2 and
12–4–4 / 7–8–5 against V3 (wins–draws–losses). The advantage therefore appeared
on both team assignments, rather than depending on a stronger team.

Mean decision times were 25.58 µs versus V2's 241.10 µs, and 25.51 µs versus
V3's 271.35 µs: approximately 9.4× and 10.6× faster, respectively. That V4 occupied
126 source lines, compared with V2's 820 and V3's 1,167 including inherited V2
code. These counts include comments and blank lines and exclude shared helpers.

The [complete report](benchmarks/classic-v4.json) includes all individual
matches and source hashes. This is measured superiority on the protocol below,
not a guarantee of winning every match or every configuration.

## Use

```bash
nhl94 play --agent classic-v1 --env NHL94-Genesis-v0
nhl94 evaluate --agent classic-v1 --episodes 5
nhl94 collect --agent classic-v1 --action_type FILTERED --output /tmp/v1-demos
```

The internal controller name is `ClassicAIV1`; public aliases are `classic-v1` and `classic`.
The registry also exposes it to DAgger. Supported outputs are `FILTERED` buttons
and `HOCKEY_INTENT_DPAD`. Full-team hockey intents use the existing wrapper
macros, so their timings and playing strength are not identical to raw buttons.

## Run the head-to-head ablation

```bash
nhl94 benchmark --agent classic-v1 --opponents classic-v1-direct \
  --seed 12000 --pairs 20 --seconds 300 --frame-skip 4 \
  --workers 4 --output /tmp/classic-v1-h2h.json
```

The benchmark uses the installed `NHL94-Genesis-v0` integration's default 2P
save. Each seed plays both team assignments against the ablation: 40 trials
total, each lasting 300 first-period game-clock seconds. These are individual
periods, not complete three-period games. Goals decide wins; shaped training
rewards are ignored. Ties remain draws. No score, player attribute, or gameplay
rule is changed. The clock and ROM RNG are initialized after each reset.

Gym's seed alone does not randomize this emulator save. The benchmark explicitly
sets the Genesis RNG at `0xFFD066`. It also reads actual controller slots and
repairs the star-derived control/possession inputs equally for all controllers;
the installed away-team star mapping is incomplete. Physical coordinates and
goal ends remain unchanged when presenting the away team's view. Correct
16-bit clock and shot counters avoid the integration's composite-word fields.
These corrections are confined to the benchmark; installed integration files,
controller tactics, and historical saved-model observations are unchanged.

JSON reports include every score, shot count, one-timer attempt/goal count,
V1 setup count, completion flag, action-trajectory
hash, initial-state hash, source hashes, package versions, and decision timing.
An incomplete period fails the run rather than counting a partial score.
Timing includes the agent adapter but excludes emulator stepping and state
decoding; it depends on the machine and current load.

## Development and limitations

Early tuning used seeds 0–7. The first fresh-seed validation, seeds 1000–1019,
beat V3 but failed to beat V2: 13 wins, 11 draws, 16 losses against V2. That
report is preserved in [classic-v4-initial.json](benchmarks/classic-v4-initial.json).
The larger subsequent development set used seeds 0–15. Aiming at middle height
instead of always high improved the V2 result to 17 wins, 12 draws, 3 losses
over 32 trials. The final validation uses a new seed range, 2000–2019.
The same final shooting rule went 11–18–3 against V3 in the 32-trial development
set. The controller was frozen before running the final validation.

The historical scripted comparisons use the Pittsburgh/Ottawa full-team save,
first-period play, filtered controls, and four-frame action repeat. The newer
CPU reports above add Montreal/Quebec and an explicit intent-control sample.
None establishes strength on other rosters, later periods, different frame
skips, or the 1-on-1/2-on-2 variants. The smaller variants have execution smoke
checks, not measured superiority claims.

```bash
python -m unittest discover -s tests -v
python -m pylint nhl94_ai tests
python -m tests.integration.classic_v1
python -m tests.integration.classic_defense
python -m tests.integration.play_v1
python tests/integration/one_timers.py
python tests/integration/cpu_benchmark.py
python tests/integration/smoke.py
```

The V1 integration check verifies valid actions in all three variants and the
full-team intent wrapper, live goalie motion in full-team play, plus
reproducibility for equal ROM seeds and distinct
trajectories for different seeds. The original five regression traces replay
their frozen inputs, independently of the removed agents, to verify unchanged
observations, rewards, and timing. The play replay checks deterministic defensive
trajectories rather than claiming the intentional tactical change preserves the
old V4 match hash.
The one-timer replay captures a successful setup from a live seeded match, then
replays it through both filtered buttons and the intent processor. Both must
produce a ROM-counted one-timer. Emulator snapshots remain in memory.

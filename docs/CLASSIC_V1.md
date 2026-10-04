# Classic V1

V1 retains the former V4 finishing/one-timer machinery, adds target-first reactive
defense and progression-oriented offense, starting in `nhl94_ai/agents/classic_v1.py`.
The original V1–V3 implementations
have been removed. Use `classic-v1` (or `classic`); old versioned command names
are not retained.

Historical results below and the archived benchmark files keep their original
V4/V2/V3 labels, sources, and hashes. They are not new V1 benchmark runs, and
comparisons against the removed controllers cannot be rerun from this tree.

Offense uses accuracy-aware finishing and observed pass outcomes:

1. Pass to a nearby skater when the goalie owns the puck.
2. Shoot across the goalie at middle height from close range, moving closer for
   less accurate shooters, unless continuing would enter the goalie's space.
   Brake or move aside before a collision-prone swing; finish earlier when the
   approach is unsafe but the shot's coasting clearance is safe. Generate a fresh
   C press and hold aim through the swing. Both finishing paths require the puck
   and its estimated coasting release point to remain in front of the goal line.
3. Otherwise, take a safe one-timer pass to a suitable shooter, using the
   projected contact position. Same-side passes must improve shooting value.
4. Prefer a safe advancement pass outside finishing range, unless already on
   a verified breakaway. In the attacking zone, pass to a stronger shooting
   position or make a bounded cut only when it improves an opportunity over
   continuing straight; otherwise carry toward the slot only after checking
   contact and interception clearance. An unsafe default route triggers a
   bounded lateral escape, brake or retreat instead.
5. Without possession, choose a defensive destination, then a skater to execute it.

Possession comes from the engine's puck-owner slot. An opponent taking possession
ends the shot follow-through. A different friendly carrier, including a live
goalie slot in reduced variants, interrupts it on the next emulator frame and
is evaluated immediately. Aim stays stable until a new recorded shot and
loss of shooter possession confirm release, checking the shooter slot when
available. A loose owner alone is insufficient. After release, the remaining
six-decision follow-through can steer clear of the goalie instead of continuing
to coast into them.
Shot left/right refers to the physical goal mouth, even when
attacking the lower net. There is no inheritance from other agents and no model asset.
Shot follow-through and legacy setup movement use agent decisions, normally every
four emulator frames. One-timer execution/retry, advancement-pass and feint
deadlines use emulator frames;
offensive planning retains the configured decision interval.
Defense and its cooldowns run every emulator frame.

Optional read-only goalie motion observations from the experiment remain
available. V1 uses goalie position for aim and motion for collision avoidance. Legacy velocity
fields and neural input arrays remain unchanged.

## Offensive correctness: staged Mighty Ducks/Sabres measurement

Initial corrections to the seven reported control-flow/safety problems were measured in
cumulative checkpoints against the built-in CPU. Each checkpoint uses the same
20 ROM seeds, `20261004..20261023`, with the AI controlling Buffalo at home and
Anaheim away in `SabresVsMightyDucks.ManualGoalie.Start`. The other team is the
actual CPU, not an idle second human. Goalkeeper assistance and cross-crease
finishing are off; controls are `FILTERED`, offensive decisions occur every four
emulator frames, and each trial completes a 300-game-clock-second first period.
There are **480 completed periods across 12 checkpoints**. Starting save hashes,
physical teams, lineups and initial effective accuracy match across checkpoints.

The [comparison report](benchmarks/classic-v1-offense-fixes-ducks-sabres.json)
links every raw report and ordered source patch. Patches were replayed from the
baseline and checked against each checkpoint's recorded source fingerprints.
Baseline source is commit `2c5c6d3f1a42dadb09cbdf728f057051b737e159`.
Apply the patches in order to reproduce intermediate implementations; final
source fingerprints identify the historical checkpoint, not subsequent follow-up
changes described below.

Scores below are aggregate **goals for-against over 20 periods per AI side**.
The last column is the combined goal-difference change against the immediately
preceding checkpoint, not an isolated ablation.

| Checkpoint | Sabres, home | Mighty Ducks, away | Combined GD delta |
| --- | --- | --- | --- |
| 00: unchanged baseline | 40-4 | 24-19 | -- |
| 01: native one-timer release/stoppage completion | 40-7 | 29-18 | +3 |
| 02: new friendly carrier interrupts old follow-through | 40-7 | 29-17 | +1 |
| 03: frame deadlines, B sampling and cadence-aware C eligibility | 41-5 | 22-14 | -1 |
| 04: select only worthwhile safe passes | 42-6 | 23-15 | 0 |
| 05: normal/early shot release geometry | 43-5 | 22-14 | +2 |
| 06: independent bounded skater projections | 37-6 | 21-14 | -8 |
| 07: contact/interception-checked default carrying | 37-3 | 16-8 | +4 |
| 08: prefer shooting/progress among safe escapes | 38-8 | 17-12 | -7 |
| 09: preserve the live puck offset in carry forecasts | 39-6 | 19-6 | +11 |
| 10: finish even after the original passer recovers | 39-6 | 19-6 | 0 |
| 11: observe timeout on the exact frame boundary | 40-7 | 16-5 | -2 |

The projection correction exposed a coupled forecast error: hypothetical carries
had relocated the puck to the skater's center, inventing pass lanes that did not
exist with the real stick offset. Forecasts now preserve the observed puck offset
while translating the carrier. This remains an approximation when sprites/facing
change, not a full future `GetHot` simulation. The recovered-passer hardening
produced identical action hashes in all 40 measured periods; its extra regression
case validates the lifecycle edge without claiming a performance gain.

| AI side | Baseline W/D/L | Final W/D/L | Baseline goals | Final goals |
| --- | --- | --- | --- | --- |
| Sabres, home | 18/2/0 | 16/3/1 | 40-4 | 40-7 |
| Mighty Ducks, away | 9/7/4 | 11/8/1 | 24-19 | 16-5 |
| Combined | 27/9/4 | 27/11/2 | 64-23 | 56-12 |

**This is mixed evidence, not an across-the-board offensive improvement.**
Anaheim conceded substantially less and lost fewer periods; Buffalo regressed,
and combined scoring fell by eight goals. Combined goal difference improved by
three, or 0.075 per period. The paired 95% seed-cluster bootstrap interval is
`[-0.475, 0.650]` per period and includes zero. Resampling uses 10,000 draws,
seed 94, and keeps both AI sides together within each ROM-seed cluster.
Refinements used the same development seeds; this is not held-out confirmation,
full-game win-rate evidence, or proof that every individual change is stronger.
The negative intermediate results remain archived rather than being omitted.

Functional coverage separately verifies native completion/cancellation,
immediate friendly-possession handoff, exact frame deadlines at intervals
1/4/10, second-choice pass eligibility, both finishing branches and attacking
ends, board uncertainty, coherent puck attachment, and safe/best-effort carrying.
Native one-timer replays capture a successful setup **at each tested cadence**:
changing receiver activation timing changes its movement, so a pass successful
at interval four is not guaranteed to succeed unchanged at interval ten.
The setup-cut check now uses a coherent, isolated RAM initializer and verifies
native fixed-point pose change, actual requested recipient, attempt-counter
increase and next-frame completion in both action formats; it does not depend
on a particular full-period policy trajectory retaining a successful cut.

After reconstructing historical checkpoint 11, reproduce its CPU measurement:

```bash
nhl94 benchmark-cpu --agent classic-v1 \
  --matchups sabres-ducks-manual ducks-sabres-manual \
  --goalie-policy off --trials 20 --seed 20261004 --seconds 300 \
  --frame-skip 4 --action-type FILTERED --workers 4 \
  --output docs/benchmarks/classic-v1-offense-11-exact-frame-deadline-ducks-sabres.json
```

### Follow-up: Mighty Ducks / All-Star Campbell

The same pre-fix baseline and controller after the original fixes were compared on the
previously untuned `MightyDucksVsAllStarCampbell.ManualGoalie.Start` matchup.
Anaheim is home and All-Star Campbell is away; each version plays 20 completed
first periods per AI side against the actual CPU. Settings and seeds match the
offensive-correctness study above: `20261004..20261023`, 300 game-clock seconds,
four-frame offensive decisions, `FILTERED`, and both goalie assistance and
cross-crease finishing off. All **80 periods** complete. Starting save hashes,
teams, lineups and effective accuracy match between baseline/current, and
source fingerprints match historical checkpoint 11.

The [baseline report](benchmarks/classic-v1-offense-baseline-ducks-campbell.json),
[current report](benchmarks/classic-v1-offense-final-ducks-campbell.json), and
[paired comparison](benchmarks/classic-v1-offense-fixes-ducks-campbell.json)
preserve every trial and source fingerprint. This follow-up compares the entire
set of fixes; it does not repeat the individual intermediate checkpoints.

| AI side | Baseline W/D/L | Current W/D/L | Baseline goals | Current goals |
| --- | --- | --- | --- | --- |
| Mighty Ducks, home | 13/4/3 | 11/7/2 | 30-10 | 26-8 |
| All-Star Campbell, away | 15/5/0 | 18/1/1 | 40-8 | 52-7 |
| Combined | 28/9/3 | 29/8/3 | 70-18 | 78-15 |

The larger gain is when Classic controls Campbell: 12 more goals and one fewer
conceded. Anaheim scores four fewer and concedes two fewer, with fewer wins and
one fewer loss; its goal difference declines by two. Combined goal difference
improves by 11, or 0.275 per period, but the paired seed-cluster bootstrap 95%
interval is `[-0.275, 0.825]` and includes zero. Resampling uses the same 10,000
draws and seed 94, retaining both AI sides within each seed cluster. No gameplay
was changed or tuned on these results.

This is new-matchup evidence, not a universal strength claim or a full-game win
rate. Comparing Anaheim's absolute results with the Buffalo matchup also changes
home/away status and saved initial ratings. The historical 100-period Campbell
measurement below additionally enabled selective goalie assistance and is not
the same protocol as this goalie-off comparison.

```bash
nhl94 benchmark-cpu --agent classic-v1 \
  --matchups ducks-campbell-manual campbell-ducks-manual \
  --goalie-policy off --trials 20 --seed 20261004 --seconds 300 \
  --frame-skip 4 --action-type FILTERED --workers 4 \
  --output docs/benchmarks/classic-v1-offense-final-ducks-campbell.json
```

### Additional lifecycle, release and cue hardening

The follow-up review exposed remaining defects rather than invalidating the
historical scores above. Recovery of an unsuccessful pass is now distinguished
from initial retained possession in **both** one-timer and ordinary pass waits.
One-timer outcomes reconcile exactly, including defensive cancellation and
period termination. Normal finishing bounds native animation timing and the
old/glide sprite stick position that can remain latched during windup. The cue
gate uses earliest body/stick contact instead of closest approach. Negative-role
off-ice players are excluded consistently without erasing fallen on-ice bodies.

The [follow-up comparison](benchmarks/classic-v1-offense-followup-comparison.json)
links raw reports, verified incremental runtime patches and paired bootstrap
estimates. Its baseline is **historical checkpoint 11**, not the original pre-fix
controller: reconstruct it from the original baseline commit and its eleven
ordered patches, then apply the five follow-up patches in order. Source hashes
were checked after every replay and against the final working package.

Five cumulative checkpoints each complete 20 periods per AI side across both
matchups: **400 new interval-4 periods**, with 80 archived baseline periods reused.
An additional **160 interval-8 periods** compare the release-guard checkpoint
against the cue-only checkpoint, isolating that cue change. Both experiments use
the same `20261004..20261023` seeds, full 300-game-clock-second first periods,
`FILTERED`, real CPU opponents, four workers, and goalie/cross-crease assistance
off. Saves, teams, lineups and initial effective accuracy match within each
paired comparison. These are reused development seeds, not held-out confirmation.

Scores are aggregate goals for-against over 20 periods per AI side.

| Interval-4 checkpoint | Sabres, home | Ducks vs Sabres, away | Ducks vs Campbell, home | Campbell, away |
| --- | --- | --- | --- | --- |
| Archived baseline | 40-7 | 16-5 | 26-8 | 52-7 |
| 01: one-timer recovery and complete outcome accounting | 42-6 | 18-5 | 26-9 | 49-7 |
| 02: native shot-release envelope | 42-6 | 18-5 | 26-9 | 48-7 |
| 03: earliest-contact cue eligibility | 42-6 | 18-5 | 26-9 | 49-5 |
| 04: inactive-player exclusions | 42-6 | 18-5 | 26-9 | 49-5 |
| 05: ordinary-pass recovery and immediate reevaluation | 37-7 | 18-5 | 28-9 | 50-5 |

| AI side | Baseline W/D/L | Final W/D/L | Baseline goals | Final goals |
| --- | --- | --- | --- | --- |
| Sabres, home | 16/3/1 | 16/3/1 | 40-7 | 37-7 |
| Mighty Ducks vs Sabres, away | 11/8/1 | 12/7/1 | 16-5 | 18-5 |
| Mighty Ducks vs Campbell, home | 11/7/2 | 11/7/2 | 26-8 | 28-9 |
| All-Star Campbell, away | 18/1/1 | 19/1/0 | 52-7 | 50-5 |
| Combined | 56/19/5 | 58/18/4 | 134-27 | 133-26 |

**There is no combined interval-4 goal-difference gain.** The change is zero
per period, with a paired seed-cluster bootstrap 95% interval
`[-0.1125, 0.1125]`. Buffalo scoring declines; Anaheim scoring improves in both
matchups; Campbell concedes less but also scores less. Ordinary-pass telemetry
records nine actual `recovered-by-passer` cancellations across the final 80
periods. One-timer starts/endings balance in every completed follow-up report.
The inactive stage encounters zero negative-role skater frames and reproduces
all 80 preceding action hashes exactly, so it is not credited with a gain in
these penalties-off runs.

At interval 8, before/after **the cue change only**:

| AI side | Before goals | After goals | Before W/D/L | After W/D/L |
| --- | --- | --- | --- | --- |
| Sabres, home | 30-7 | 25-5 | 13/5/2 | 12/6/2 |
| Ducks vs Sabres, away | 14-12 | 13-13 | 9/6/5 | 8/6/6 |
| Ducks vs Campbell, home | 20-19 | 15-14 | 7/5/8 | 5/9/6 |
| Campbell, away | 32-8 | 45-8 | 14/4/2 | 16/3/1 |

Combined interval-8 goal difference improves by eight, or 0.1 per period;
the combined 95% interval `[-0.15, 0.35]` still includes zero. Most of the gain
comes from Campbell, while both Buffalo and Anaheim-away regress. Neither
comparison establishes a universal strength improvement. Resampling uses 10,000
draws, seed 94, retaining all four AI sides within each matched ROM-seed cluster.

`python -m tests.integration.offensive_followup` reproduces actual pre-release
puck positions at/behind the goal line under the old four-frame C schedule,
then verifies guarded finishing at both attacking ends and in both action
formats. Native sprite latching matters: checking only the eventual shot sprite
misses the older glide hotspot that pulls the puck forward during windup.
The same module demonstrates a short pass that can be one-timed at interval 4
but must be rejected at interval 8, and a completed native advancement pass past
a genuinely off-ice, negative-role actor. Longer native one-timers remain
covered at intervals 1/4/8/10. The setup-cut fixture now uses a longer,
cadence-valid receiving position and still verifies an actual executed cut and
receiver-attributed release in both action formats.

The final interval-4 command is:

```bash
nhl94 benchmark-cpu --agent classic-v1 \
  --matchups sabres-ducks-manual ducks-sabres-manual \
    ducks-campbell-manual campbell-ducks-manual \
  --goalie-policy off --trials 20 --seed 20261004 --seconds 300 \
  --frame-skip 4 --action-type FILTERED --workers 4 \
  --output docs/benchmarks/classic-v1-offense-followup-05-ordinary-recovery.json
```

## Opt-in held-C cross-crease finishing

```bash
nhl94 play --agent classic-v1 --env NHL94-Genesis-v0 \
  --state PenguinsVsSenators.start --cross-crease
```

`--cross-crease` is **off by default**. It is also available to headless
evaluation, collection, DAgger teachers and CPU benchmarks. Full-team
`FILTERED` and `HOCKEY_INTENT_DPAD` controls are supported; reduced variants,
learned-only evaluation and self-play reject the option. Shootout states retain
ordinary finishing instead of assuming the normal game's pre-release goalie
save branch. No public action IDs, ordered observations or saved models change.

`agents/cross_crease.py` separates evaluation from execution. From an
offensive-zone wing it plans the approach and crossing, then compares the
complete maneuver with immediate shooting, a currently available safe
one-timer, positional passing/cutting and continued carry. Values are
uncalibrated opportunity scores, **not goal probabilities**. A crossing must
beat the best alternative by more than eight points; an already worthwhile
crossing is preferred to additional setup movement for a higher estimated score.
The comparison precedes Classic's normal finishing/one-timer priority.

Candidates require known ownership/control, signed motion, shot power,
handedness and animation feedback. Entry is within attacking depth **88-244**
and absolute X **8-120**. Neither existing lateral speed nor a nearly horizontal
velocity is an entry requirement. Up to 160 ordinary-skating frames can create
the approach, turn and lateral momentum. Candidates include direct crossings
and wing-ingress/crossing waypoints, projected with the shared velocity-aware
steering and skating/stop helpers. Only the carrier must remain within the
rink: an irrelevant skater's extrapolation through the boards does not veto it.

Net-front traffic is not a blanket rejection. Swept body paths, imminent
collisions, stationary occupants, net routing and defender shot lanes remain
physical guards. Other actors' motion is extrapolated for at most 12 frames;
later possible pursuit is a bounded score penalty, not an optimistic boosted-
arrival veto. Pressure considers both carrier and carried puck. The potential
goalie response uses a named windup-angle scenario rather than demanding that
the current goalie already leave the eventual shooting lane clear. Its
anticipated body sweep is checked separately. These forecasts cannot establish
that a future ROM save will occur.

Arming is based on the live reachable puck opening, not an instantaneous
velocity cutoff. Power below 20 uses a shorter, 32-frame projected release
window instead of the high-power 38-frame window. Both ends of a bounded
release-hotspot interval contribute to the opportunity score. Future CPU
decisions, turn timing and save geometry remain uncertain.

Execution is `approach -> fresh C press -> hold/coast -> C release -> confirmed
shot -> exit`. It runs every emulator frame while other offensive planning
keeps its configured interval. **D-pad input during ShotMode is aim, not skating
acceleration**; the crossing uses established inertia. C is held at most 24
frames, with preparation and the swing/shot outcome bounded by a 240-frame
sequence deadline. Preparation validates its proposed path, then checks fresh
12-frame movement sweeps while following the current waypoint. Every 12
approach frames it can explicitly replace the route with a freshly evaluated
one; each change is recorded, and replanning cannot deepen the crossing lane.
Failure to forecast a replacement alone does not invalidate a physically clear
existing route. Actual obstruction, ownership/control loss or a missed window
does. It can arm earlier when live momentum supports an immediately worthwhile
crossing against the selection-time alternatives; the executed plan and actual
charge position/frame are recorded separately. These alternatives are not a
per-frame reranking of all ordinary actions.

The native animation can force an earlier release. Save/dive animation,
wrong-way lateral motion and an observed four-unit advance toward the charging
skater during actual windup are separately identified commitment evidence.
An advance is not counted as an accepted save animation or guaranteed opening.
Early C release can jump directly to a two-frame native release instead of
waiting for a full 14-frame swing; the early-opening check uses that shorter
window and requires the body and puck to have crossed. The selected
world-coordinate aim stays stable through release.

The crossing projection and native-windup monitor use the ROM's combined
16-unit player-body contact radius plus four units of uncertainty. Before an
accepted windup, the live hold monitor retains the ordinary 24-unit shooting
buffer. These maneuver-specific guards
does not change the ordinary 32-unit movement or 24-unit shooting guards.
An imminent obstruction, missing motion or an advancing animation forces C release;
it does not turn a charged shot into a pass or silently redirect its aim.
An invalidated setup is discarded. Turnover/control loss releases buttons
before resuming ordinary defense. Manual goalie takeover cannot interrupt an
active crossing.

Diagnostics expose alternatives, candidate rejection reasons, the exact route
and current waypoint, replans, charge position, phase, commitment kind and
release reason. A submitted C press, accepted windup, fresh
shooter-attributed shot and goal are recorded separately. Contacts require new
native impact feedback, not a stale positive word or visual proximity.
CPU reports include per-attempt events and cross-crease counters, plus
`goalie_contact_impulses` in general offense metrics. The latter counts fresh
controlled-player collision impulses, not unique collision episodes.

```bash
python -m tests.integration.cross_crease

# Matched CPU ablation: repeat this command with --cross-crease.
nhl94 benchmark-cpu --trials 3 --seed 48300 --seconds 300 --workers 2 \
  --output cross-crease-off.json

# Benchmark-only label: the same Classic controller, with one-timers retained.
nhl94 benchmark --agent classic-v1-cross-crease --opponents classic-v1 \
  --pairs 3 --seed 49000 --seconds 300 --workers 2
```

The isolated ROM checks cover both attacking ends, both crossing directions,
both action formats, stationary wide-wing starts at X +/-100 and attacking
depth 180, a deeper (-100, 160) approach and goal, live net-front pursuit,
safe missed-window exit, and identical active near-net sequences at decision intervals
1/4/10. They verify actual held-C travel, pre-release commitment, shooter-counted
shots and absence of native goalie contact. In the coherent timing fixture,
tap/low-power-held/high-power-held shots release at 6/32/38 frames for both
handedness values; holding for 60 frames does not postpone the high-power shot
beyond frame 38. These are fixture measurements, not universal timing constants
or evidence of improved match strength.

### Historical near-net selection measurement

The [matched CPU report](benchmarks/classic-v1-cross-crease-cpu.json) retains
an initial pilot on seeds 48000-48002 and a separate frozen evaluation on
48100-48102. Each cohort has nine five-minute first periods per policy:
three seeds each for Pittsburgh/Ottawa, Ottawa/Pittsburgh and Quebec/Montreal.
The initial pilot selected no crossings; subsequent development added bounded
momentum preparation, shorter low-power windows and evaluation/rejection counts.

The frozen evaluation also selected **zero attempts**. Its 5,187 controlled-
carrier decisions contained 4,529 outside the entry window, 619 without a safe
crossing and 39 ineligible animation/shot states. All nine enabled action hashes
equalled their matched disabled hashes. Both policies scored 13 goals and
conceded five; one-timers, turnovers and goalie-contact impulses were unchanged.
The initial pilot likewise had identical actions and scores, 12 goals for and
three against per policy.

These results belong to the superseded near-net evaluator, not the wing-origin
planner, and do **not** establish improved normal-match scoring.

### Wing-origin CPU measurement

The [wing-origin report](benchmarks/classic-v1-cross-crease-wing-cpu.json)
preserves a separate frozen comparison on unused seeds **48300-48302**, again
nine five-minute periods per policy across the same three matchups. Sources,
lineups and initial-state hashes match between policies; all nine action hashes
now differ. Reproduce the CPU command above with `--seed 48300`, once without
and once with `--cross-crease`.

The enabled policy selected **71 plans, eight actual C attempts, eight accepted
windups, eight attributed shots and one cross-crease goal**. Five attempts
observed windup advance and one a pre-release save animation; none of the
attempt events recorded goalie contact. Real approaches include
(106, 219) -> (52, 193) over 38 frames and
(111, -102) -> (36, -177) over 91 frames. Fifty plans were invalidated by fresh
route guards, three missed their crossing window, eight lost possession, and
one stopped with play; one plan remained active at period end.

Overall enabled scoring was **12 for / six against**, versus **nine for / two
against** disabled. General goalie-contact impulses rose from 28 to 66 and
zone turnovers from 41 to 48, despite no contact on attributed attempts.
The trajectories also change subsequent ordinary play; these are not causal
per-attempt contact estimates. The small comparison demonstrates
ordinary-game use and an attributed goal, **not a match-strength improvement**.
The tactic remains experimental and off by default.
Keep the tactic experimental and off by default rather than treating the
isolated goal examples as a match-strength result.

## Opt-in manual goalie AI

```bash
nhl94 play --agent classic-v1 --env NHL94-Genesis-v0 \
  --mode model_vs_game --state SabresVsMightyDucks.ManualGoalie.Start \
  --side home --goalie-policy selective --max_playback_speed 1.0
```

This controls Buffalo against the Mighty Ducks CPU. Use `--side away` to
control Anaheim against Buffalo. No ROM, integration file or save is modified.
The supplied save must be installed in Retro's NHL94 data directory.

Add `--seed 12000` to vary repeated playback episodes reproducibly. The first
episode writes ROM seed 12000, the next 12001, and so on (wrapping at uint32).
The write happens after loading the save and before task setup and its first
emulator frame, including automatic episode resets. Without `--seed`, playback
retains the save's RNG state and deterministic replay behavior. Only randomness
changes: starting positions, rosters, goalie settings and the PostPlay cutoff
remain unchanged. These are repeated saved starts, not successive game periods.

`--goalie-policy off` is the default and retains the existing controller.
`selective` keeps useful skater defense and takes the goalie only for a
reachable projected threat without a useful controlled skater contest/block.
`always` is a defensive-zone takeover ablation, not continuous goalie selection
during attacks. Both enabled policies require full-team Classic, `FILTERED`,
`PostPlay`, no self-play, and Manual goalie settings for the physical team
**and** its joystick index. Automatic-goalie saves fail explicitly.
CPU and human opponents are supported; human play additionally requires the
usual two-controller save. Reduced-player games and intent/learned policies
are not supported by this option.

`agents/goalie.py` chooses a **cyan square** target before generating movement.
Green skater-defense and amber offense targets are unchanged. The diagnostic
shows the target, actual/desired slot, B countdown, reach/crossing estimates,
animation, CPU fallback and request/result counters. Inactive goalie diagnostics
do not hide a live skater target.

Released-puck prediction uses full signed motion, friction, height and the
crease/goal planes; it stops at uncertain wall contacts. A geometrically
reachable incoming receiver moves the angle target before reception. Carrier
release deadlines remain estimates, not promises about an attacker's next move.
Selective takeover budgets 24 frames for selection plus goalie travel and a
six-frame response margin. It keeps close skater contests and existing shooting
lane blocks. It does not interrupt a pending skater pass, one-timer, shot,
switch or body check.

The controller runs every emulator frame. It releases old buttons, holds B
until the goalie slot **and joystick flag** confirm selection, releases B, and
then sends goalie input. A new short B press/release requests a skater; only an
actual skater slot confirms return. Handoff attempts time out after 60 frames
with a warning and bounded backoff. Movement uses goalie-specific acceleration
and strong neutral braking, rather than skater turning/boost estimates. Targets
remain near the crease; pulled/unavailable goalies are not takeover candidates.
Animation locks are respected. Offscreen CPU assistance receives neutral input
and its catches are not counted as manual catches.

Fresh C requests the ROM's save selection. Rare low-shot emergencies can request
A plus direction for a dive. Request counters are separate from confirmed save/
dive animation counters; neither is a count of prevented goals. On possession,
the AI considers safety-scored B outlets with the correct goalie pass base/parity,
waits for a fresh pass counter and observed reception, and protects that flight
from skater switching or boosting. If no safe outlet opens, a bounded hold
ends in an A clear. **A clears; it does not freeze.** Neutral manual holding is
not assumed to run the CPU cover timer.

The installed Retro `FILTERED` preset omits A. Opted-in environments therefore
use the backend's unfiltered binary buttons to deliver dives/clears, retaining
the same ordered 12-button public interface (24 for human matches). Classic
emits exclusive A/B/C edges; the default backend/filter and all saved-model
schemas remain unchanged.

Focused native verification:

```bash
python -m tests.integration.manual_goalie
```

This checks actual B selection/return, D-pad movement/braking, C save contact,
A dive, low-speed catch, safe outlet/reception and A clear, then exercises the
actual watched home/away command. Contact fixtures write only the test emulator
after selection has been obtained by B; production handoffs never force RAM.

Matched first-period ablations can be run with:

```bash
nhl94 benchmark-cpu --agent classic-v1 \
  --matchups sabres-ducks-manual ducks-sabres-manual \
  --goalie-policy selective --trials 3 --seed 9100 --seconds 300 \
  --workers 2 --output manual-goalie-selective.json
```

Repeat with `off` and `always`. These manual matchups are opt-in; the existing
default three-matchup cohort is unchanged.

The [manual-goalie report](benchmarks/classic-v1-manual-goalie.json) records
three matched seeds (9100-9102), each with Buffalo and Anaheim controlled:
**six five-minute first periods per policy**, not full games.

| Policy | Goals for / against | Confirmed takeovers | Save / dive animations |
| --- | --- | --- | --- |
| `off` (CPU goalie) | 8 / 2 | 0 | Not manually attributed |
| `selective` | 8 / 0 | 35 | 12 / 1 |
| `always` | 6 / 4 | 51 | 28 / 4 |

Selective recorded ten controlled catches and eight requested-recipient outlet
receptions. All 35 requested takeovers confirmed. Always cancelled three of its
54 requests when friendly possession returned during the hold. Neither enabled
policy had a handoff timeout. Save/dive animations are not prevented-goal counts;
CPU fallback and interrupted/expired outlets remain separately recorded.

This is a small **development cohort**, not a 50-period held-out comparison.
Selective's concession reduction is encouraging, but does not establish reliable
superiority over the CPU goalie. More frequent manual actions were not better:
always-manual conceded more and scored less here. The report preserves source
hashes, action hashes, starting-state hashes, lineups and per-seed outcomes.

### Buffalo/Anaheim: 50 CPU periods per side

The [100-period report](benchmarks/classic-v1-ducks-sabres-selective-100-periods.json)
uses selective goalie control, `FILTERED` buttons and offense frame skip 4.
Both matchups use the same 50 ROM seeds, 20261002-20261051, from
`SabresVsMightyDucks.ManualGoalie.Start`. All periods reach the five-minute
first-period clock cutoff; wins/draws/losses compare scores at that cutoff.
Buffalo remains the home team and Anaheim the away team. Starting lineups and
ratings are fixed, with Grant Fuhr and Guy Hebert in goal.

Wins/draws/losses are from Classic's perspective. Paired statistic columns
show **Classic / CPU** totals; one-timers are ROM-counted attempts, not just
requested passes.

| Classic team | Periods | Wins | Draws | Losses | Goals | Shots | One-timers | One-timer goals |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Anaheim vs Buffalo CPU | 50 | 23 | 19 | 8 | 51 / 26 | 357 / 217 | 103 / 69 | 31 / 6 |
| Buffalo vs Anaheim CPU | 50 | 39 | 7 | 4 | 105 / 16 | 415 / 141 | 121 / 49 | 67 / 8 |
| Combined | 100 | 62 | 26 | 12 | 156 / 42 | 772 / 358 | 224 / 118 | 98 / 14 |

Classic averages 7.72 shots and 2.24 one-timers per period, versus the CPU's
3.58 shots and 1.18 one-timers. It outperforms the built-in opponent in both
directions of this fixed matchup with selective goalie control. This is not
a full-game win rate or evidence of superiority across all teams, lineups,
goalie policies or game settings.

```bash
nhl94 benchmark-cpu --agent classic-v1 \
  --matchups ducks-sabres-manual sabres-ducks-manual \
  --goalie-policy selective --trials 50 --seed 20261002 --seconds 300 \
  --frame-skip 4 --action-type FILTERED --workers 4 \
  --output docs/benchmarks/classic-v1-ducks-sabres-selective-100-periods.json
```

### Anaheim/Campbell: 50 CPU periods per side

The [100-period Campbell report](benchmarks/classic-v1-ducks-campbell-selective-100-periods.json)
repeats the same protocol and ROM seeds, 20261002-20261051, with
`MightyDucksVsAllStarCampbell.ManualGoalie.Start`. Anaheim is home and
Campbell is away; goalies are Guy Hebert and Ed Belfour. The single-controller
save is used for both sides, with live RAM controller transfer for away trials.
All 100 five-minute first periods complete. Agent/gameplay source hashes match
the Buffalo/Anaheim report; only the CPU benchmark's opt-in matchup list changed.

Wins/draws/losses are from Classic's perspective. Statistic pairs show
**Classic / CPU** totals, including ROM-counted one-timer attempts.

| Classic team | Periods | Wins | Draws | Losses | Goals | Shots | One-timers | One-timer goals |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Anaheim vs Campbell CPU | 50 | 19 | 22 | 9 | 47 / 33 | 359 / 240 | 112 / 62 | 27 / 16 |
| Campbell vs Anaheim CPU | 50 | 41 | 9 | 0 | 122 / 15 | 426 / 138 | 141 / 46 | 88 / 7 |
| Combined | 100 | 60 | 31 | 9 | 169 / 48 | 785 / 378 | 253 / 108 | 115 / 23 |

Classic again wins more than it loses in both directions, with no losses when
controlling Campbell. The Ducks' outcome is tighter than against Buffalo:
19/22/9 wins/draws/losses and goals 47/33, versus 23/19/8 and goals 51/26.
Its shot totals are similar, 359 versus 357, while opposing shots rise from
217 to 240 and opposing one-timer goals from 6 to 16.

This compares two fixed saved setups, not opponent strength in isolation:
the Ducks are home here but away against Buffalo, and their saved effective
ratings differ. Post-load ROM seeding does not regenerate initial ratings or
saved hot/cold tables. Neither report establishes full-game win rates or
performance across arbitrary lineups and settings.

```bash
nhl94 benchmark-cpu --agent classic-v1 \
  --matchups ducks-campbell-manual campbell-ducks-manual \
  --goalie-policy selective --trials 50 --seed 20261002 --seconds 300 \
  --frame-skip 4 --action-type FILTERED --workers 4 \
  --output docs/benchmarks/classic-v1-ducks-campbell-selective-100-periods.json
```

## Target-first defense

`agents/defense.py` keeps target selection separate from player selection:

| Mode | Target and safety rule |
| --- | --- |
| `protect-lane` | Stay between the predicted carrier and our goal; fill an uncovered goal-mouth lane rather than duplicate a teammate's block. |
| `contain-boards` | Close the inside escape from the boards while remaining goal-side; do not leave an open slot receiver without another defender. |
| `recover-safe` | Reach a predicted low puck with time to settle possession before an opponent can contest it. |
| `intercept-pass` | Reach an intermediate trajectory point before reception, including switch delay and a safety margin. |
| `deny-reception` | Cover an incoming receiver's shooting lane; also anticipate an uncovered receiver when teammates already cover the carrier. |
| `deny-goalie-outlet` | An opposing goalie-held puck is not loose: cover a likely outlet receiver instead of pursuing the crease. No poke is requested against the goalie. |
| `goalie-avoid` | Classic's movement guard redirects a collision-prone route away from the opposing goalie; the green target shows that immediate escape. |

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

### Teammate shooting-lane coverage

Coverage uses the existing center and two inset goal-mouth targets, not a
teammate's presence somewhere in the middle of the rink. Each eligible skater
must already lie between the threat and that goal target, within an eight-unit
body radius. Full X/Y velocity predicts whether the skater will remain in the
path when a modeled shot reaches them. The bounded estimate checks shot speeds
of 3–6 rink units/frame and a four-frame release window, including any predicted
pass delay; predictions beyond 32 frames do not count as dependable cover.
Missing motion, locked/inactive players, goalies and players behind the shooter
or goal do not justify delegating a lane.

When only some carrier lanes are covered, the green target moves toward an
uncovered goal-mouth ray, favoring the side farther from the goalie. When all
three are covered, the controller considers nearby slot receivers, predicts
their motion and prioritizes the earliest uncovered pass-plus-shot threat.
It only redirects if the same teammates also cover the carrier's current and
projected shooting lanes through the possible pass delay. If no dependable
alternative exists, it retains goal-side support rather than inventing a safe
assignment.

The acting skater and a pending switch destination cannot supply the coverage
that supposedly frees them to leave. Supporting skaters relied upon by the
plan are reserved and excluded from target-player selection. Coverage is
recomputed each frame, so drift, lost eligibility or a new threat releases old
reservations immediately. Loose-puck chase safety and board containment use
the same lane geometry; covering a carrier does not imply covering a receiver.

This is a conservative low-shot model, **not guaranteed shot blocking**.
Three sampled rays are not the entire goal mouth; airborne shots, stick
animation, collisions, changes in CPU steering and shot speeds outside the
modeled range can defeat the prediction. Neural tactical-feature calculations
and the learned target controller are unchanged.

### Skater selection and execution

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
boost/poke/body-check requests do not interrupt pending switch input. No direct player-control
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

### Deliberate C body checks

Defense now prefers a fresh `check-request` over a poke when a close, verified
opponent skater carrier presents a favorable collision. This deliberate branch
does not target goalies, loose pucks or arbitrary nearby opponents; ordinary
skating boosts can still cause incidental contact. The checker must be
goal-side, within 30 rink units, facing within roughly 26 degrees of the carrier,
unlocked and actually joystick-controlled, with known motion/weight/facing
feedback and at least 1024 energy. Pending switches and held B/C prohibit a
new check.

The estimate adds C's energy-dependent impulse in the **current facing**
direction, not the newly requested D-pad direction. It predicts first contact
within eight frames using the ROM's 16-unit combined body radius and estimates
normal impact from relative full-word velocities. It requires impact at least
20 and at least four points above the weight threshold. Existing accumulated
impact and random checking-rating successes are deliberately not relied on.
The burst estimate conservatively subtracts the 204-energy cost even when the
ROM's line-change configuration bypasses that drain.

**A simple "heavier checker wins" rule is wrong for this controller.** Classic
uses a joystick, so the Genesis weight bug applies:

```python
threshold = ((240 - checker.weight + carrier.weight) & 0xFF) >> 1
```

These are stored weights (eight times the ROM rating), and smaller thresholds
are easier. A weight-4 checker against weight 8 gives threshold 8; the reverse
gives 104. This differs from an autonomous CPU check's base of 120.
See the [weight arithmetic](nhl94%20Deep%20dive.md#85-the-weight-bug-precisely).
The CPU's slot-defense `check4check` assignment does not grant its decision
bonuses to a joystick-driven agent.

Checks reject receding/glancing contact, unsafe net/board routes, loss of
goal-side position and predicted collisions with another skater or goalie.
They also reject leaving a separately assigned threat or an otherwise
uncovered receiver lane currently blocked by the acting skater.
For the predicted contact time plus two frames (at most ten), the agent releases
buttons without switching away or braking against its burst. A pass, recovery,
changed control or timeout cancels this follow-through immediately.
C checks share the skating boost's 36-frame cooldown and suppress a new poke
for 16 frames; unfavorable checks retain the existing B-poke fallback.

This is a conservative opportunity estimate, **not a guaranteed knockdown or
recovery**. Actual steering changes, timing, accumulated impact, high-stick
stagger exceptions and penalty logic still matter. The agent does not predict
charging/roughing probabilities or treat a check request as a successful hit.

### Debugging and execution

Normal `nhl94 play --agent classic-v1` shows the **green tactical target square on
the ice**, plus the same square in the rink diagram. Cyan identifies the desired
defender; the overlay also shows the actual defender, waypoint, projected puck
path, reason, arrival estimates, receiver and missing feedback. Ideal/obtainable
skaters, the predicted B selection, switch status and the last observed result
explain why it switches or retains control. It uses the existing pre-step camera
projection and clears on resets.
The rink overlay draws predicted covered lanes in green, uncovered lanes in
purple and the selected shooting ray in cyan. Diagnostics include `lanes`
(origin, goal, threat slot, blockers and release delay), `shot_goal` and
`reserved_slots`; the green square remains the skating destination.
The body-check line shows the current eligibility/rejection status and estimated
impact versus weight threshold. `check` diagnostics also include the carrier
slot, predicted contact time/point when available; `pending_check` identifies
the bounded follow-through. `check-request` describes submitted C, not a
confirmed collision.

`AgentOutput.diagnostics["classic_defense"]` exposes the same information without
a display. Evaluation includes the last defensive decision per episode; CPU
benchmarks count tactical frames, switch/poke/check/boost requests and controlled
recoveries. Requests are not counted as successful switches or checks.
Play, scripted evaluation, collection, DAgger teacher observations and raw
benchmarks all use `predict_frame`. The learned target-position controller,
neural observation ordering, tasks/rewards and bundled model assets are unchanged.
`python -m tests.integration.classic_defense` exercises a useful but non-ideal
switch through the real ROM in both action formats, checks selection feedback
in all three variants, and compares per-frame/four-frame caller behavior.
It also exercises teammate-aware retargeting without stealing the covering
skater, then injects drift in an isolated ROM scenario to verify that live
velocity feedback invalidates that coverage.
Its isolated body-check scenario uses a controlled 4-versus-8 weight matchup:
one fresh C produces the burst animation, contact with the intended carrier,
a ROM-recorded body check and possession loss in both action formats.
Matched neutral input produces no recorded body check. No controlled recovery
was observed within that 16-frame scenario, so disruption and recovery remain
distinct outcomes. This fixture changes only its own emulator RAM, not benchmark
rosters or saved assets.

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

### Lane-coverage follow-up: same 50 Ottawa periods

The teammate-aware coverage change was compared against the switching follow-up
using exactly the same seeds, saves, rosters, controller side and timing above.
All 50 periods completed. **The functional coverage fix did not improve the
aggregate defensive result in this sample:**

| Metric | Before | Lane-aware |
| --- | --- | --- |
| Goals conceded | 27 | 34 |
| Goals conceded per period | 0.54 | 0.68 |
| Shutout periods | 29/50 | 26/50 |
| Shots conceded | 217 | 229 |
| Pittsburgh one-timer goals | 6 | 13 |
| Ottawa goals scored | 35 | 43 |
| Period wins / draws / losses | 19 / 17 / 14 | 19 / 15 / 16 |

Compared seed by seed, 12 periods conceded fewer goals, 20 the same and 18 more.
The new concession distribution was 26 zero-goal, 15 one-goal, eight two-goal
and one three-goal periods. The change correctly handles the covered/uncovered
lane scenarios, but those checks must not be presented as proof of stronger
overall defense. These are fixed-roster first-period team outcomes, not
full-game win rates or guaranteed future performance.

The [lane-aware raw report](benchmarks/classic-v1-lane-coverage-senators-penguins-50-periods.json)
preserves the per-period scores and source hashes. To reproduce it, use the
previous command with
`--output docs/benchmarks/classic-v1-lane-coverage-senators-penguins-50-periods.json`.
The before report retains its original source hashes and must not be overwritten.

### Body-check follow-up: same 50 Ottawa periods

Deliberate C checks were compared with the lane-aware controller using the same
50 five-minute first periods, seeds 20260930–20260979, rosters, saves and timing.
All periods completed at zero clock, and the report's source hashes match the
measured implementation. Only `agents/defense.py` and `agents/motion.py` changed
among the gameplay sources recorded by the benchmark.

| Metric | Lane-aware before | With deliberate checks |
| --- | --- | --- |
| Goals conceded | 34 | 32 |
| Goals conceded per period | 0.68 | 0.64 |
| Shutout periods | 26/50 | 26/50 |
| Shots conceded | 229 | 213 |
| Pittsburgh one-timers / one-timer goals | 70 / 13 | 69 / 13 |
| Ottawa goals scored | 43 | 42 |
| Period wins / draws / losses | 19 / 15 / 16 | 20 / 17 / 13 |
| Controlled defensive recoveries | 481 | 493 |
| Deliberate C requests | 0 | 72 |

Compared seed by seed, eight periods conceded fewer goals, 35 the same and
seven more. The new concession distribution was 26 zero-goal, 16 one-goal and
eight two-goal periods. This is a **small improvement in this development
sample**, not evidence of guaranteed defense or a statistically established
strength gain. Concessions remain above the earlier switching-only run's 27;
the teammate-coverage regression is not erased. One-timer goals and shutouts
did not improve.

The 72 C requests are attempts, not 72 confirmed body checks. Likewise, the
additional controlled recoveries cannot all be attributed to checks; the
benchmark records whole-period behavior, not causal check-to-recovery chains.
No parameters were retuned against these results.

The [body-check raw report](benchmarks/classic-v1-body-checks-senators-penguins-50-periods.json)
preserves every period and source hash. Both previous reports remain unchanged.
Reproduce with:

```bash
nhl94 benchmark-cpu --agent classic-v1 --matchups senators-penguins \
  --trials 50 --seed 20260930 --seconds 300 --frame-skip 4 \
  --action-type FILTERED --workers 4 \
  --output docs/benchmarks/classic-v1-body-checks-senators-penguins-50-periods.json
```

## One-timers

V1 preserves its close-range direct shot as the first choice. Otherwise, it
considers a teammate across the slot: the passer must be beyond attack-relative
Y=100, the receiver beyond Y=175–190 depending on accuracy and below Y=245,
and the pass must be 30–150 units long with less than 80 units of vertical
separation. Receivers must be within 70 units of the middle, on the opposite
side of the centerline, with more than 30 units of lateral separation.
Live candidates additionally use the moving-pass evaluator described below,
including release timing, recipient selection and reception pressure. Safe
candidates are ranked rather than accepted in roster order. With absent live
passing telemetry, the historical 14-unit static-segment check remains an
explicitly unverified compatibility fallback. Falling receivers are excluded.

With absent live passing telemetry, a carrier between Y=140 and 215 can
move toward the opposite side of a potential receiver for at most eight
decisions. The receiver must already be near shooting range. A 48-decision
cooldown prevents repeated sideways movement; a turnover or receiver knockdown
cancels the setup. This creates a passing angle without waiting indefinitely.
Normal ROM play uses the bounded, motion-checked setup cuts below.

Live effective shot accuracy is read from each skater's RAM record. Direct-shot
depth is `218 + max(0, 15 - accuracy) * 0.4`; the minimum one-timer receiving
depth is `175 + max(0, 15 - accuracy)`, evaluated at predicted stick contact in
live play, not the receiver's current position. An unknown attribute falls back to 15;
a real zero remains zero. These are heuristic position thresholds, not scoring
probabilities. The additional fields do not change neural observation ordering
or shapes. Goalie position determines normal-shot aim; goalie ratings
are not used to assign a scoring probability.

V1 aims the pass and holds B for at least four emulator frames, then releases B.
While the puck is loose it
generates fresh C presses to activate the ROM's one-timer routine. The ROM
chooses the actual pass recipient from the direction and handles one-timer aim.
V1 stops pressing C when the intended receiver enters the one-timer animation.
It ends as soon as a fresh native one-timer attempt counter advances and
`shot_player` identifies the requested receiver, without waiting for a shot on goal. These
read-only counters are `0xFFCA2A`/`0xFFCD8E`, exposed as each team's
`one_timer_attempts`; ordinary `shots` can update later or never for an off-target
release. A fresh, receiver-attributed recorded shot is a compatibility fallback,
not a persistent shooter flag or C press. Release and stoppage feedback are
observed every emulator frame, interrupting held actions before the next normal
offensive decision. A stopped play clears the sequence and submits neutral input.

Interception, clean reception, receiver knockdown, wrong recipient and bounded
timeout also cancel the commitment. The live deadline is the rounded predicted
flight plus release allowance plus 20 emulator frames; missing live flight
telemetry retains a 72-frame deadline. The retry cooldown is 96 emulator frames.
Timeout observation uses the current processing frame, not the previous frame's
clock, so the bound is exact even between offensive decisions. Confirmed native
release wins over timeout and ends the sequence even if the original passer has
already recovered the rebound.
If a pass has actually been observed loose and the original passer regains it
without that release evidence, V1 instead ends the commitment as
`recovered-by-passer` and reevaluates immediately. Ordinary advancement/position
passes use the same flight-versus-windup distinction and interrupt cached neutral
waiting actions on recovery.
Every one-timer start has one recorded ending in a completed CPU report,
including interception and period end. `one_timer_accounting` verifies this
invariant; native attempt/goal totals are separate from controller outcomes.
Fresh pass-counter feedback confirms the actual recipient once; a wrong
recipient cancels the intended one-timer instead of pressing C at the wrong
play. Both pass types require B release before another request.
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

## Progressive offense

`agents/offense.py` plans progression and cuts; `agents/passing.py` estimates
pass selection, flight and reception. Defense retains its previous tactics;
shared `agents/motion.py` also supplies coasting and bounded offensive forecasts.

| Mode | Rule |
| --- | --- |
| `advance-pass` | Prefer a safe forward pass with at least 20 units of gain that bypasses a defender, gains at least 50 units, or crosses the attacking blue line. |
| `position-pass` | In the attacking zone, require a modeled shooting-value improvement, not merely a deeper receiver. |
| `carry-breakaway` | No goal-side skater in the nearby corridor, no modeled short-horizon interception and a contact-clear carrying route; execution shares the ordinary carrying safeguards. |
| `carry` / `carry-escape` | Keep a verified safe slot route; otherwise compare bounded escapes/braking/retreats. Rank safe alternatives by modeled shooting opportunity and actual projected progress, not unnecessary retreat once safety is established. |
| `feint` | A left/right cut must improve the modeled shot or receiving opportunity over both the current position and continuing straight. |
| `one-timer-setup` | A bounded cut must open a safe predicted one-timer without reducing the modeled opportunity; finish or abandon the cut rather than immediately replacing it with an ordinary pass. |
| `pass-release` / `pass-flight` | Release buttons and follow observed ownership/recipient feedback; an advancement pass must not trigger a one-timer C. |

### Pass safety and uncertainty

The requested D-pad direction is a recipient-selection hint, not a player ID.
The predictor follows ROM `vtoa`'s 2:1 sectors, direction/distance scoring,
adjacent-sector penalty and later-slot tie breaking. A nearer teammate can win
the same direction; such an ambiguous intended pass is rejected. The intent
adapter uses authoritative control when available, so a stale star does not
change which roster entry the pass intent addresses.

Optional skater `passing` feedback reads object `+0x6E`. The normal speed
estimate uses `160 + 2 * passing`, including the odd-rating increment, and
converts from the ROM's time base. It is **not** a pass-accuracy percentage.
Live ROM playback reads the actual sprite stick offset through `GetHot`'s table,
X/Y flips and horizontal-view transform. The launch model follows `passto`'s
signed word arithmetic, integer square root and eighth-second lead quantization,
not an ideal continuous interception at a fixed nominal speed. It predicts the
resulting puck path with friction and rejects launches that cannot reach the
moving stick within 14 units. The ordinary reception-speed gate uses the vector's
estimated speed at contact, not just the nominal passing rating.

Flight is limited to 48 frames. Live geometry uses a two-frame release estimate,
shared by launch, offside and collision timing. The no-sprite-feedback compatibility
path retains its explicitly approximate facing hotspot and four-frame allowance.
`geometry` and `release_frames` identify which path was used in diagnostics.
Sprite/assignment changes between the decision and release, receiver steering,
deflections and animation contact timing remain uncertain; this is not a complete
ROM simulator.

Every candidate checks swept relative puck/body motion between flight samples,
moving opponents and their live stick offsets along the flight, optimistic opponent
arrival times including plausible boosts, goalies, friendly body obstruction,
net crossings and pressure at reception. Friendly sticks can obstruct a pass even
when their bodies do not; opponent stick reach is included in arrival and
uncertainty checks. Ordinary reception includes four
frames to settle; a one-timer does not. At least three frames of interception/
reception margin are required. For an ordinary reception, the receiver's live
stick-handling byte must support the modeled puck speed: the raw velocity
threshold is `13000 + 350 * live_stick`, with no additional energy multiplier.
Fatigue still affects skating/reachability estimates; stealing uses a separate
energy-scaled ROM path. A one-timer bypasses that normal catching threshold,
matching the ROM's dedicated stick-contact
branch. Its projected contact must lie within the shooting window, leave
at least four flight frames and have its **first possible body/stick contact**
strictly after the first available C cue
at the configured offensive decision interval. That cue follows the four-frame
B hold and estimated release allowance; a ten-frame decision interval is not
treated as if it could cue at frame four.
Closest approach can occur after an ordinary catch, so closest-approach timing
alone is not sufficient cue margin. Diagnostics expose `first_contact_frame`.
Missing necessary feedback rejects the
new pass rather than declaring it safe. Offside-enabled blue-line crossings
also reject predicted teammates ahead of the puck; unknown rule telemetry
does not justify claiming such a crossing legal.

Eight fixed, seeded common movement scenarios perturb short-horizon opponent
positions within an acceleration-based envelope. At least seven must remain
clear **after** the hard interception gates. This preserves reproducibility
without using the live ROM RNG as clairvoyance. The displayed `robustness`
is a scenario fraction, **not a calibrated completion probability**.
CPU steering, integer contact timing, deflections and random stick challenges
can still defeat an accepted pass.

The planner filters safe passes by the advancement or shooting-improvement rule
before selecting the highest-ranked remaining candidate. An ineligible first
choice cannot hide a worthwhile second choice. Ranking favors forward gain,
defenders bypassed, receiving space and modeled shooting quality. A close central
finisher with valid release geometry keeps possession instead
of passing simply for extra depth. Successful reception, another receiver,
opponent possession and bounded timeout finish the advancement sequence
separately. Ordinary pass requests have a 24-frame cooldown; one-timer requests
use 96 frames. No gameplay/control RAM is written by the agent.

One-timers rank receiving-shot value and safety margin rather than forward gain.
They are not limited to current cross-center geometry: an incoming receiver can
reach the slot during flight, and a same-side one-timer is allowed when its
modeled shooting value exceeds the carrier's by more than 12. Recipient prediction,
interception margins and uncertainty gates still apply. Uncontrolled teammates
continue to follow the ROM's support behavior.

### Purposeful cuts and diagnostics

Cuts use actual momentum, agility/energy-dependent acceleration, the controller's
eight-way movement and a conservative six-frame-per-sector turning budget.
They cannot instantly reverse velocity. A cut has an 18-frame maximum and a
72-frame start-to-start cooldown, rejects predicted body collisions and
uncertain wall/net motion, and abandons the commitment when pressure worsens
or a close finishing opportunity takes priority. The ROM check exposed a
turn-before-acceleration delay; a simple instantaneous-heading projection was
not sufficient. These remain motion estimates, not exact CPU-response predictions.

The carrier's actual target-directed or braking path is validated before
constructing a future state. Its unused straight-line extrapolation cannot veto
a valid brake, and braking paths themselves must stay within the playable area.
Other skaters reaching conservative board/net bounds no longer reject the whole
forecast. Their projected centers are bounded and the displacement introduces
`projection_uncertainty`, which expands collision/shot/pass envelopes and
optimistic reachability. An uncertain forecast receiver is rejected individually
rather than invalidating unrelated shooting opportunities. Goalies retain their
separate motion treatment. This bounds uncertainty; it does not predict an exact
board rebound. The field is forecast-only and never enters neural arrays.
The puck keeps its live offset from the carrier instead of being recentered,
so projected passing lanes start from the observed attachment geometry.

Passing and feinting never assume that uncontrolled teammates obey new
destinations: their observed motion and existing ROM support behavior determine
the options. C while carrying is a shot, so these cuts do not use it as a boost.

Live `one-timer-setup` cuts share the 18-frame limit, 72-frame cooldown, collision
guards and momentum model. They are considered explicitly before committing to
an ordinary positional pass when a cut can open a stronger immediate-shot lane.
The no-one-timer benchmark ablation does not create these setups.

Live cuts resolve goalie avoidance **before** projecting their passing/shooting
opportunity. When the original diagonal cut is redirected, the planner also
considers pure lateral alternatives. It scores the resolved target, not the
discarded proposal, and shares the same braking projection with goalie clearance.
Execution uses that exact validated target without a second tactical rewrite.
If fresh feedback would require a different route, the old cut/opportunity is
cancelled rather than silently redirected. A rerouted escape is not treated as
an unchanged straight-line alternative.

Safe `one-timer-setup` plans take priority over the goalie guard's earlier normal
shot, while ordinary safe close-range finishing retains its priority. A valid
one-timer pass is no longer hidden by a close-range shot whose coasting path would
hit the goalie. Collision buffers, interception gates, timing limits and the
no-one-timer ablation remain unchanged.

Playback shows an **amber offensive target**, candidate pass rays and the
chosen receiver, flight/margin/robustness and last observed advancement-pass
outcome. The green defensive square is unchanged. `classic_offense` diagnostics
are available through the agent, frame-skip traces and evaluation metadata;
reset/turnover clears stale annotations. Action IDs, normalized observations,
tasks, saved neural models and learned target-controller behavior are unchanged.
Optional goalie full-word motion now uses the actual goalie slots in reduced
variants, rather than assuming full-team slots 5/11.

Ordinary carrying now validates the actual routed, goalie-adjusted waypoint at
4/8/12/18 frames against friendly/opposing body contact and at least three frames
of optimistic interception margin. Safe default routes are preserved. If one
fails, safe lateral, braking and retreat options are ranked by their projected
shooting value and progress. When no safe option exists, the controller chooses
a best-effort bounded escape and reports `carry_safe=False`; unavoidable wall/net
contact requests braking without claiming safety. Missing live motion remains
the explicit legacy compatibility path with `carry_safe=None`.
`carry_clearance`, `carry_pressure`, `carry_progress` and `carry_shot_value`
explain the selection. CPU reports additionally count controller-observed
one-timer endings and safe/default/escape/least-risk carrying decisions.
Ordinary-pass endings are additionally recorded in `ordinary_pass_metrics`.
Negative native roles mean off ice: those players are not projected into blockers,
but an unavailable or fallen player with an on-ice role remains an obstacle.

`python -m tests.integration.classic_offense` confirms an actual ROM-selected
advanced recipient, reception and control transfer in both action formats,
without defensive switching or one-timer C. It also verifies that a purposeful
cut changes actual carrier direction. Those scenarios demonstrate execution,
not improved match strength.

### Opposing-goalie clearance

Goalie clearance is part of live cut planning. `goalie-avoid` applies to carrier
movement, legacy setup cuts, launched
ordinary-pass follow-through, shot follow-through after confirmed release,
loose-puck recovery, and defensive skating near the opposing goalie. It suppresses
boost while escaping; pass aim, one-timer cues, shot windup aim and pending
skater-switch/check sequences are not redirected.

The guard compares momentum-aware routes and swept relative motion against the
goalie's current velocity over 32 frames. It also checks the current coasting
course, so an optimistic turning estimate cannot dismiss an approaching collision.
Candidates include braking opposite the current facing, lateral escapes and
retreats, projected inside playable bounds and outside net-crossing routes.
The normal forward-skater opposite-heading stop path uses the ROM's `stopna`
150-raw-unit per-axis deceleration estimate, not an instant reversal or a
goalie-style stop. Routes are replanned from fresh feedback.

The 32-unit movement buffer and 24-unit shot-coasting buffer are conservative
planning choices, not fixed ROM body radii or Penguins-specific calibration.
Normal finishing range is retained on clear approaches. When carrying closer
would be unsafe, a central shooter beyond depth 175 can finish earlier after
checking coasting clearance through the swing. A higher-speed approach can
instead require braking first. The current puck and estimated release after the
configured C hold plus two native release frames must both remain in front of
the goal line. Behind-the-net carrying uses net-aware waypoints back into
position. A zero conservative `shot_value` alone is not a legality test: legal
very-close shots beyond its heuristic depth window remain possible.

Diagnostics expose `goalie_clearance`, `original_goalie_clearance` and
`goalie_avoidance_safe`, with an amber offensive or green defensive escape target.
If no modeled candidate clears the buffer, the guard reports a best-effort escape,
not a collision-free guarantee. Animation locks, uncertain future goalie steering
and an already-too-close starting position can still cause contact.

`tests.integration.goalie_avoidance` checks actual ROM impact/other-player contact
feedback, not merely requested avoidance directions. Both filtered and intent
controls avoid contact in the two coherent approach fixtures while still recording
a shot. These checks are not a new CPU goal-rate benchmark; the historical
reports below predate this guard.

### Exact Penguins playback and pass-geometry correction

The supplied command was reproduced through the installed parser, display,
vector environment and actual emulator, without changing its save or ROM RNG:

```bash
nhl94 play --agent classic-v1 --env NHL94-Genesis-v0 \
  --mode model_vs_game --side home --max_playback_speed 1.0
```

It resolves to `PenguinsVsSenators.start`, with Pittsburgh home against Ottawa's
real CPU. Before the geometry correction, five deliberate pass requests completed
only one one-timer, at **0:20 remaining**, and the score was 0-0. The earlier seeded
CPU samples and Montreal command did not establish what this particular default
playback would do.

The old continuous/facing-hotspot pass estimate accepted trajectories that the
ROM did not produce. Using live sprite hotspots, integer launch geometry,
consistent release/flight timing and stick-aware interceptions produced five
completed deliberate one-timers at **4:11, 2:44, 2:05, 1:22 and 0:24 remaining**.
Each fresh ROM counter increase matched an active request and its intended
receiver, not an incidental CPU teammate shot. Pittsburgh scored two, Ottawa
zero in this exact replay. These are one-period reproduction results, not a
general winning-rate claim or a promise of five attempts in every matchup.
The [watched-play report](benchmarks/classic-v1-penguins-watched-pass-geometry.json)
preserves before/after action hashes, requested-versus-recorded events, clocks
and final measured source hashes. Both runs stop at the `PostPlay` cutoff below
ten seconds, not at the end of a full three-period game.

`python -m tests.integration.penguins_play` now requires multiple attributable
one-timers, including one before the period's halfway point, through that exact
command. It cannot pass by merely requesting B/C or counting an old saved stat.
The goalie-contact and both-format setup/pass/one-timer checks remain separate.
The Montreal away first-period fixture still records a one-timer with the same
shared live geometry. Historical CPU reports below predate this correction.

### One-timer planner/goalie-guard correction

The original goalie guard could replace a planned `one-timer-setup` target
without recomputing its predicted opportunity. The correction resolves and
validates the actual route in the planner, preserves safe setups ahead of early
finishing, and checks that execution and planning targets agree.

`tests.integration.one_timers` now captures a naturally successful setup cut
against the real CPU, then replays **cut -> pass -> the requested receiver's
ROM-counted one-timer** in filtered and intent controls. It does not merely start
from an already-selected pass or accept another teammate's incidental shot.
The existing fast-pass/zero-stick checks and actual goalie-contact fixtures remain.

Twelve matched 300-second first periods, four seeds `20261100-20261103` for each
matchup, produced:

| Controlled team / CPU opponent | One-timers before -> after | Goals for | Goals against |
| --- | --- | --- | --- |
| Pittsburgh / Ottawa | 10 -> 15 | 8 -> 9 | 1 -> 1 |
| Ottawa / Pittsburgh | 7 -> 7 | 3 -> 4 | 1 -> 2 |
| Quebec / Montreal | 5 -> 5 | 4 -> 4 | 2 -> 2 |
| **Total** | **22 -> 27** | **15 -> 17** | **4 -> 5** |

One-timer goals increased from 12 to 13; periods without any decreased from four
to three. ROM totals can include autonomous teammate one-timers. This is a small
development comparison, not a statistical holdout or a guarantee of an attempt
in every period. Scoring increased in this sample, but so did concessions.
Saves, seeds, sides, starting lineups and initial-state hashes match; both reports
archive the relevant controller sources and all measured source hashes.

Reports: [frozen before](benchmarks/classic-v1-one-timer-guard-before-periods.json)
and [corrected after](benchmarks/classic-v1-one-timer-guard-after-periods.json).
Earlier measurements below describe their own historical revisions.

### One-timer recovery measurement

The initial progressive offense omitted live setup skating, rejected receivers
based on their current rather than projected shooting position, and incorrectly
applied normal catch-speed limits to one-timers. The recovery fixes these gates,
allows stronger same-side receiving shots, and sweeps contact between flight
samples rather than allowing a body to slip between sampled endpoints.

Scripted watched play now starts from the wrapper's fresh reset RAM feedback,
including automatic episode resets. Previously it submitted four blind neutral
frames before initializing the agent, so a raw benchmark and the watched save
could immediately diverge. Learned playback retains its existing timing.
The exact watched Montreal command completes a ROM-counted one-timer in its
default saved first period; this bounded run ended Quebec 1, Montreal 1.

Eight five-minute first periods per matchup, seeds `20261060` through `20261067`,
were matched against a frozen pre-fix controller:

| Matchup / controlled team | One-timers, before -> after | Goals scored | Goals conceded |
| --- | --- | --- | --- |
| Pittsburgh vs Ottawa / Pittsburgh | 8 -> 15 | 8 -> 11 | 2 -> 4 |
| Pittsburgh vs Ottawa / Ottawa | 0 -> 9 | 8 -> 8 | 6 -> 5 |
| Montreal vs Quebec / Quebec | 4 -> 16 | 4 -> 11 | 5 -> 9 |
| Total, 24 periods | 12 -> 40 | 20 -> 30 | 13 -> 18 |

One-timer goals increased from 6 to 17. Periods without any recorded one-timer
fell from 14 to 5, not zero. Scoring improved in this sample, but concessions
also increased; more one-timers are not proof of better defense or guaranteed
wins. These seeds are separate from the original four-seed development cohort,
but were evaluated during candidate confirmation, not an untouched statistical
holdout.

The [before](benchmarks/classic-v1-one-timer-recovery-before-fresh-periods.json)
report preserves the three frozen source snapshots and hashes.
The [after](benchmarks/classic-v1-one-timer-recovery-after-fresh-periods.json)
report records the measured current source hashes. Matching seeds, lineups,
saves, sides and initial state hashes were verified; the frozen baseline also
reproduced a previous action hash, goals, shots, one-timers and frame count.
ROM counters distinguish completed one-timers from requested B/C inputs.
`python -m tests.integration.one_timers` verifies both action formats, including
maximum passing speed with zero receiver stick handling.
`python -m tests.integration.away_play --one-timer-period` checks the full watched
Montreal command, actual joystick routing and the completed one-timer counter.

### Offense measurement: results and regression

The final implementation was compared against the body-check controller on
the same 50 Ottawa-versus-Pittsburgh five-minute first periods:

| Metric | Before | Progressive offense |
| --- | --- | --- |
| Ottawa goals scored | 42 | 28 |
| Ottawa shots | 106 | 82 |
| Goals conceded | 32 | 33 |
| Shutout periods | 26 | 24 |
| Period wins / draws / losses | 20 / 17 / 13 | 13 / 20 / 17 |
| Ottawa one-timers / one-timer goals | 53 / 19 | 26 / 11 |

**The requested progression/feint behavior works, but scoring strength regressed
in this sample.** There were 852 advancement requests and 168 positional-pass
requests. Advancement outcomes included 658 ordinary receptions and 139
interceptions; a request is not a completion. Compared seed by seed, nine
periods scored more, 22 the same and 19 fewer goals. This must not be advertised
as stronger offense merely because it passes more.

After gameplay was frozen, a separate comparison used seeds 20261030–20261037
for eight periods in each supported matchup. The old offense was loaded from
its verified Git snapshot, not restored into the worktree. Before using it,
seed 20260930 reproduced the prior body-check report's exact action hash,
score, shots and frame count. Both controllers used common read-only telemetry
and outcome instrumentation; no new historical agent name was registered.

| Matchup, candidate first | Goals scored before / after | Goals conceded before / after | Skater-possession zone entries before / after | Mean entry frames before / after |
| --- | --- | --- | --- | --- |
| Pittsburgh / Ottawa | 7 / 5 | 4 / 1 | 65 / 62 | 203.4 / 176.4 |
| Ottawa / Pittsburgh | 7 / 3 | 6 / 3 | 70 / 63 | 196.6 / 214.0 |
| Quebec / Montreal | 6 / 7 | 5 / 2 | 66 / 62 | 186.4 / 173.6 |
| Total, 24 periods | 20 / 15 | 15 / 6 | 201 / 187 | 195.4 / 188.2 |

Entry time improved for two rosters, but entries decreased on all three and
Ottawa's entry time worsened. The near-entry threat estimate also did not
improve consistently. Fewer goals conceded on these 24 periods does not
isolate an improvement in defense: possession and offensive play change the
situations the unchanged defender encounters. Eight periods per matchup is
small; this is not a full-game win rate or statistical strength guarantee.
The only post-freeze gameplay correction rejected a net-crossing pass edge case
that had raised a route-search exception; no scoring parameters were retuned
against these held-out results.

An entry means skater possession outside Y=88 followed by skater possession
inside, including a loose-puck pass, before an opponent takes possession.
Receiving an opponent turnover already inside the zone is not a new entry.
Entry frames are emulator frames; the threat diagnostic counts opponent
skaters estimated to reach a point 16 units ahead within 16 frames, not all
defenders remaining goal-side. Turnovers exclude observed recorded-shot
follow-ups; `possession_losses` separately includes them. The earlier
development report's `turnovers` counted all possession losses.

The initial development comparison (seeds 20261010–20261017) is preserved:
the first candidate scored 15 versus 21 baseline goals over 24 periods.
Subsequent changes addressed close-finisher passing, facing delay, feedback
and net-route correctness, not a claim of successful strength tuning.

Raw evidence:
[50-period final offense](benchmarks/classic-v1-offense-senators-penguins-50-periods.json),
[frozen held-out baseline](benchmarks/classic-v1-offense-before-heldout-periods.json),
[held-out final offense](benchmarks/classic-v1-offense-after-heldout-periods.json),
and [initial development baseline](benchmarks/classic-v1-offense-before-fresh-periods.json).
The [first candidate report](benchmarks/classic-v1-offense-first-candidate-fresh-periods.json)
retains that development candidate's original source hashes.
The original body-check and lane-coverage reports remain untouched. Reproduce
the final 50-period measurement with the body-check command above, changing
the output to `docs/benchmarks/classic-v1-offense-senators-penguins-50-periods.json`.

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
They now additionally track advancement and positional passes by purpose;
CPU reports include the zone-entry and possession-loss diagnostics above.
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
python -m tests.integration.penguins_play
python -m tests.integration.goalie_avoidance
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

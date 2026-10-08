# Classic V1

V1 retains the former V4 finishing/one-timer machinery, adds target-first reactive
defense and progression-oriented offense, starting in `nhl94_ai/agents/classic_v1.py`.
The original V1–V3 implementations
have been removed. Use `classic-v1` (or `classic`); old versioned command names
are not retained.

Historical results below and the archived benchmark files keep their original
V4/V2/V3 labels, sources, and hashes. They are not new V1 benchmark runs, and
comparisons against the removed controllers cannot be rerun from this tree.

## Classic refinements (2026-10-08)

Five deterministic changes were implemented against `05ab06e`. **Evaluated
goalie outlets are enabled by default. The other four remain experimental:**
native checks establish particular mechanics, while the gameplay comparisons
below do not justify promoting the combined policy. No learned model is used.

| Change | Behavior | Enable |
| --- | --- | --- |
| Goalie outlets | Share the manual controller's pass evaluator; require eligible receivers, ROM recipient agreement and reception safety; track actual launch/receiver/outcome; require a fresh B edge. Hold for automatic cover when no outlet qualifies. | Default |
| Pass timing | Project puck attachment and all candidate receivers at the second held-B input, accounting for native slot update order. Reuse that launch/contact forecast through reception and flag changing recipient predictions. | `--classic-refinements pass-timing` |
| Carry and feint motion | Share `carry_path`, actual input cadence, fractional facing and swept body clearance across ordinary carry estimates, initial/continuing cuts and breakaways. | `--classic-refinements carry-motion` |
| Finishing | Compare eligible normal finishes and one-timers with a shared release-geometry score; choose ordinary shot aim and C hold using motion, shot power/energy and goalie pose. Release C inside the tactical input block when needed. | `--classic-refinements finishing` |
| Interception deadlines | Keep cheap candidate ranking, then simulate the best two with the live steering rule, switching/check delays and boosts. Require controlled arrival and select the verified defender. | `--classic-refinements interceptions` |

Several refinements can follow the same flag. They work with the default
possession policy or the separately enabled `--offense-lookahead`; enabling a
refinement does not select lookahead. They are supported for full-team Classic
with `FILTERED` or `HOCKEY_INTENT_DPAD`. Manual goalie control still requires
`FILTERED`. Benchmarks record the selected refinements, and the default replay
gate rejects experimental references.

The recipient forecast holds current CPU steering and sprite offsets across
the short release window. It does not predict new CPU decisions or guarantee
exact ROM selection in every state. Two reproduced native recipient mismatches
are fixed; a separate 97-case artificial moving-teammate probe reduced
mismatches from 21 to 12, with all nine changed predictions correct in that
sample. Those remaining errors are unresolved, and this is not a gameplay
error-rate estimate. One-frame decisions now retain B through direction
selection instead of cancelling the request early.

The carry regression test catches a teammate collision inside the next
four-frame input block that the old sampled clearance misses. The defense test
rejects a 56-frame interception admitted by the cheap arrival estimate. These
are model-level counterexamples. Finishing scores are explicit geometry
heuristics, not scoring probabilities or calibrated possession values; goalie
motion is only projected over a short window. Interception simulation does not
model every collision, CPU action during a switch, or future steering replan.

### Refinement measurements

The [comparison](benchmarks/classic-v1-refinements-comparison.json) and
[compressed evidence archive](benchmarks/classic-v1-refinements-evidence.json.gz)
retain report source hashes, runtime versions, physical fixtures, seeds, applied
input hashes, event metrics and reproducible patches against `05ab06e`.
All score comparisons use completed 300-clock-second first periods with
four-frame offense, `FILTERED`, CPU goalies and other experimental tactics off.

The initial cumulative pilot used four seeds (`20266001..20266004`) on each
standard Buffalo/Anaheim assignment:

| Cumulative implementation | GF-GA, eight periods |
| --- | ---: |
| Frozen baseline | 16-1 |
| Evaluated outlets | 15-0 |
| Add release-time pass prediction | 12-2 |
| Add native carry/cut prediction | 14-2 |
| Add finishing comparison | 9-3 |
| Add simulated interception deadlines | 4-5 |

The separate 40-period validation cohort (`20266101..20266120`, both standard
assignments) scored **53-17 before and 58-7 with evaluated outlets**. The paired
mean goal-difference change was +0.375 per period, with a seed-clustered 95%
bootstrap interval of **[-0.025, 0.8]**: favorable evidence, not established
superiority. Outlets plus pass/carry prediction scored 62-19. A narrower
four-frame collision guard scored 65-14, tying outlet-only goal difference
while conceding more; its eight-period pilot scored 10-3. That guard was not
retained as an additional default rule. All tested variants, including failed
experiments, are preserved in the archive.

These seeds became development evidence when used to choose defaults. Fixed
rosters and first periods do not establish full-game or human-opponent strength.
The final current-source default reproduces all 40 outlet-only action hashes.
On three additional roster fixtures, using ten new seeds per fixture
(`20266201..20266210`), it scores **54-9 versus 45-19** before. The paired
goal-difference change is +0.633 per period, with a seed-clustered 95% interval
of [0.133, 1.133]. This is a small fixed-fixture comparison, not universal
strength evidence.

| Additional roster fixture | Baseline GF-GA | Final default GF-GA |
| --- | ---: | ---: |
| Pittsburgh vs Ottawa | 25-6 | 26-1 |
| Ottawa vs Pittsburgh | 8-9 | 11-4 |
| Quebec vs Montreal | 12-4 | 17-4 |

Independent eight-period pilots at the final source revision isolate each
refinement against the outlet-only default. They reuse the pilot seeds and are
development measurements:

| Enabled refinement | GF-GA |
| --- | ---: |
| None (final default) | 15-0 |
| Pass timing only | 12-2 |
| Carry motion only | 14-5 |
| Finishing only | 9-2 |
| Interception deadlines only | 10-1 |

The [current-source native replay gate](benchmarks/classic-v1-refinements-gate.json)
matches the full away period, including all 9,491 native frames and applied
inputs. The archive contains 356 completed period records, including repeated
seeds and parity reruns; these are not 356 independent samples.

Reproduce the final standard reference and its native cadence gate with:

```bash
python -m nhl94_ai benchmark-cpu --agent classic-v1 \
  --matchups sabres-ducks-manual ducks-sabres-manual \
  --goalie-policy off --trials 20 --seed 20266101 --seconds 300 \
  --frame-skip 4 --action-type FILTERED --workers 4 \
  --output /tmp/classic-refinements-standard.json
python -m nhl94_ai.evaluation.carry_replay_gate \
  --benchmark /tmp/classic-refinements-standard.json \
  --output /tmp/classic-refinements-gate.json
```

For an individual refinement pilot, add its `--classic-refinements` value to
the benchmark command and use `--trials 4 --seed 20266001`. The additional
roster cohort uses `--matchups penguins-senators senators-penguins
nordiques-canadiens --trials 10 --seed 20266201`.

Validation: 775 unit/discovered tests pass with one existing expected failure;
Pylint reports 10.00/10. Explicit ROM suites `tests.integration.classic_offense`,
`classic_carry`, `offensive_followup`, `manual_goalie` and the new
`classic_refinements` pass. The new native checks cover both recipient
counterexamples and exact two-frame C holds through native shot release in both
button and intent schemas.

## Default offense restoration (2026-10-08)

Ordinary `classic-v1` again uses the immediate carry/pass priorities, goalie
avoidance model and ordinary-reception cadence from `5109c57`. The default
possession policy lives in `agents/possession.py`; it shares passing geometry,
pass/one-timer lifecycle, release guards and defensive control with the current
controller. The decision inspector and native predictors remain available.

`--offense-lookahead` explicitly selects the native carry/receiver-continuation
policy introduced in `117efb1`, including its native goalie-avoidance projection
and next-frame reception interrupts. `--uncertain-carry` and `--chance-creation`
also select those semantics, preserving their experimental behavior. All three
options are off by default. Cross-crease and deke finishers remain separately
opt-in and operate on the selected offense policy.

The restored default ranks ordinary carries using sampled body clearance,
arrival estimates, positional value and forward progress. These estimates do
not certify a route against every possible defender input. Its diagnostics use
`carry_pressure_model=arrival-estimate`, `carry_estimated_clear` and
`carry_safe=null`; positional value is not executable finishing credit.
Experimental native forecasts retain their independent `carry_safe` certificate.

The regression was not isolated to carry ranking. An eight-period component
pilot restored ordinary shots, but its separate 40-period comparison did not
restore match strength. Restoring the earlier goalie-avoidance and reception
timing as well reproduced all eight earlier-version pilot action hashes.
The completed comparisons below preserve the unsuccessful partial restoration
alongside the final policy; local mechanics correctness is not a strength claim.

Manual-goalie validation also accepts an absent diagnostic assignment label.
An empty native assignment stack can otherwise abort a match even when motion,
ratings and controller ownership are available. Those required gameplay inputs
remain checked. This is separate from the offensive policy restoration.

### Restoration measurements

The [paired comparison](benchmarks/classic-v1-default-restoration-comparison.json)
verifies matching saved starts, physical teams, lineups and seeds; completed
zero-clock periods; zero inactive-skater frames; balanced one-timer accounting;
and final runtime source hashes. All **200 final periods** complete. Every
comparison uses 300-clock-second first periods, four-frame offense, `FILTERED`,
and disabled experimental tactics.

| Cohort | Paired periods | `117efb1` GF-GA | Restored GF-GA | Before W/D/L | Restored W/D/L |
| --- | ---: | ---: | ---: | --- | --- |
| Standard Ducks/Sabres, goalie off | 40 | 41-14 | 60-13 | 20/16/4 | 25/12/3 |
| Pittsburgh/Ottawa and Montreal/Quebec, goalie off | 80 | 92-48 | 131-26 | 36/27/17 | 59/18/3 |
| Buffalo/Anaheim and Anaheim/Campbell, selective goalie | 79 | 104-46 | 157-37 | 40/20/19 | 59/17/3 |

The standard cohort uses seeds `20265001..20265020`, disjoint from the
four-seed component pilots. The other cohorts reuse the original regression
seeds `20263001..20263020`. These are development comparisons, not untouched
held-out policy-selection tests. The original selective-goalie version crashed
on Campbell seed `20263018`, so that pair is excluded from the score comparison.
The restored policy completes all 80 selective trials at **160-37, 60/17/3**.
The separate [crash replay](benchmarks/classic-v1-default-restoration-goalie-crash.json)
also completes with `--offense-lookahead`, independently checking the validation
fix on the policy that previously failed.

All 120 goalie-off final action hashes match `5109c57`, recovering its previous
scores exactly. The newer selective-goalie controller is retained: against
`5109c57`, its full 80-period result changes from 141-32 to 160-37. The paired
goal-difference interval includes zero, so this does not establish superiority
to the earlier goalie policy.

| Classic fixture | Policy | Before GF-GA | Restored GF-GA |
| --- | --- | ---: | ---: |
| Pittsburgh vs Ottawa | off | 35-8 | 40-5 |
| Ottawa vs Pittsburgh | off | 14-11 | 26-3 |
| Montreal vs Quebec | off | 31-13 | 38-13 |
| Quebec vs Montreal | off | 12-16 | 27-5 |
| Buffalo vs Anaheim | selective | 28-3 | 46-5 |
| Anaheim vs Buffalo | selective | 19-13 | 35-16 |
| Anaheim vs Campbell | selective | 18-22 | 27-11 |
| Campbell vs Anaheim, 19 matched periods | selective | 39-8 | 49-5 |

Offensive production recovers: in the 80 goalie-off regression periods, recorded
shots rise **325 to 570**, ordinary `shoot` decisions **164 to 494**, and goals
excluding one-timers **13 to 43**. Escape decisions fall **34,387 to 14,767**.
Native one-timer attempts fall 223 to 197 while their goals rise 79 to 88.
Scoreless periods fall 28 to 14. Ordinary turnovers rise 582 to 593 and offensive
zone turnovers 313 to 383: the improvement does not mean every safety or
possession metric improved. The report retains those overlapping metrics,
opponent shots, goalie contacts, per-matchup outcomes and seed-clustered
bootstrap intervals.

The [component archive](benchmarks/classic-v1-default-restoration-components.json.gz)
preserves all six eight-period pilots and links exact patches against `117efb1`.
The unsuccessful [partial restoration](benchmarks/classic-v1-default-restoration-partial.json)
scores 45-20 in the 40-period standard cohort, versus 41-14 before: four extra
goals do not offset six extra concessions. This is why the complete offense
policy, including steering guards and reception cadence, was restored together.
The cumulative experiments do not assign a unique causal contribution to each
component. The original negative reports remain unchanged.

The [current-source native gate](benchmarks/classic-v1-default-restoration-gate.json)
reproduces the complete away period's 8,520 frames and applied inputs. A separate
selective dispatch gate is embedded in its raw report. These verify caller
parity, not general playing strength. Fixed saved rosters and first periods
remain limitations; these are not full-game win rates or all-team coverage.

Reproduce the standard comparison with the frozen revisions and source patches
identified in the reports, and the final policy with:

```bash
python -m nhl94_ai benchmark-cpu --agent classic-v1 \
  --matchups sabres-ducks-manual ducks-sabres-manual \
  --goalie-policy off --trials 20 --seed 20265001 --seconds 300 \
  --frame-skip 4 --action-type FILTERED --workers 4 \
  --output /tmp/classic-restoration-standard.json
python -m nhl94_ai.evaluation.carry_replay_gate \
  --benchmark /tmp/classic-restoration-standard.json \
  --output /tmp/classic-restoration-gate.json
```

The four-fixture matrices use `evaluation/fixture_regression.py` with
`--revision-root "$PWD"`, `--group off` or `--group selective`, `--trials 20`,
`--seed 20263001`, `--workers 4` and a fresh `--output` path; selective also
uses `--gate`. Their raw reports are
[CPU goalie](benchmarks/classic-v1-default-restoration-off.json) and
[selective goalie](benchmarks/classic-v1-default-restoration-selective.json).

Validation: `python -m unittest discover -s tests -v` completes 763 tests with
one existing expected failure; `python -m pylint nhl94_ai tests` passes at
10.00/10. Explicit ROM checks in `tests.integration.classic_offense`,
`tests.integration.classic_carry`, `tests.integration.offensive_followup` and
`tests.integration.manual_goalie` pass. The captured live retreat is retained in
`tests/fixtures/classic-carry-retreat.json`; unit checks also preserve the
default/experimental reception-cadence distinction and optional assignment
feedback behavior.

### Same-seed benchmark rerun (2026-10-08)

A fresh current-source rerun completes all **200 restored-default periods** and
reproduces every archived match record exactly, including applied-action hashes,
scores, native counters and event metrics. No gameplay sources were changed.
The [rerun comparison](benchmarks/classic-v1-default-restoration-rerun-comparison.json)
records per-team and per-fixture GF-GA, W/D/L, scoreless periods, shots,
one-timers, turnovers and phase-tagged goalie contacts, plus source/runtime
fingerprints and report hashes.

| Cohort | Completed periods | Rerun GF-GA | Rerun W/D/L | Changed input periods |
| --- | ---: | ---: | --- | ---: |
| Standard Buffalo/Anaheim, goalie off | 40 | 60-13 | 25/12/3 | 0 |
| Pittsburgh/Ottawa and Montreal/Quebec, goalie off | 80 | 131-26 | 59/18/3 | 0 |
| Buffalo/Anaheim and Anaheim/Campbell, selective goalie | 80 | 160-37 | 60/17/3 | 0 |

The raw [standard](benchmarks/classic-v1-default-restoration-rerun-standard.json),
[CPU-goalie](benchmarks/classic-v1-default-restoration-rerun-off.json) and
[selective-goalie](benchmarks/classic-v1-default-restoration-rerun-selective.json)
reports retain the original seed ranges and settings above. Completed coverage,
zero remaining clocks, active skaters, balanced one-timer lifecycles and matching
saved starts, teams, lineups and effective attributes were checked. The fresh
[native replay gate](benchmarks/classic-v1-default-restoration-rerun-gate.json)
matches 8,520 away frames; the selective report's independent dispatch gate
matches 8,512 frames.

This is a same-seed reproducibility check, not new held-out strength evidence.
The weaker-policy scores were reused from the historical reports, not rerun.
Against its 79 completed selective periods, the matched restored result remains
**157-37, 59/17/3**; the full **160-37** includes the formerly crashing Campbell
seed `20263018`, which now completes. `--offense-lookahead` and all other
experimental tactics remain disabled in these rerun measurements.

### Current offensive priorities

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

## Experimental skating dekes

```bash
nhl94 play --agent classic-v1 --env NHL94-Genesis-v0 \
  --state PenguinsVsSenators.start --deke
```

`--deke` is **off by default and not recommended as a stronger policy**.
The initial matched CPU measurement regressed. This is an instrumented
prototype and scenario runner, not a promotion of new default tactics.
Full-team `FILTERED` and `HOCKEY_INTENT_DPAD` play, evaluation, collection,
DAgger teachers and CPU benchmarks share the option. Reduced variants,
learned-only evaluation and training self-play are rejected. Shootout states
retain ordinary finishing without skating dekes.
The paired benchmark exposes the same controller as `classic-v1-deke`;
one-timers remain enabled. With both `--deke` and `--cross-crease`, the highest
valued eligible maneuver wins and only one sequence starts.

`agents/deke.py` evaluates local finishing opportunities, not a breakaway
predicate. Entry requires controlled, authoritative skater possession, live
motion/animation/rating feedback, attacking depth 150-216 and absolute X at most
60. It compares mirrored bait/cut candidates with immediate shooting, carrying,
passing and available one-timers. Values are uncalibrated opportunity scores,
not goal probabilities. Nearby defenders contribute body/puck-path pressure and
an interception-risk cost; distance alone does not reject them. Swept physical
obstructions and imminent puck pressure remain hard guards.

Execution is `bait -> observed tracking -> cut -> hold/release -> exit`.
A central carrier first moves laterally; a carrier already on a wing can draw
the goalie by advancing without an unnecessary second lateral setup. A cut
requires actual skater movement and live goalie displacement or tracking
velocity. Bait lasts at most 64 emulator frames, a cut at most 96, and the
whole sequence at most 224, followed by a 72-frame retry cooldown. Active
execution runs every emulator frame, independent of the ordinary decision
interval. Unit checks cover intervals 1/4/10.

Skating retains a 32-unit goalie buffer and checks a fresh 24-frame route.
Shot coasting retains the ordinary 24-unit buffer. An already-open close shot
is not overplayed. The controller normally requests a four-frame wrist shot;
a one-frame release is considered when the longer coast is unsafe. Directional
input during ShotMode is aim, not acceleration. Release checks use both the
live puck and the X envelope of shot, glide and pre-shot sprite hotspots, with
the normal shot's physical X +/-13 corner. The additional read-only
`Player.shot_offsets_x` feedback does not enter normalized neural inputs.

Route changes, pressure, deadlines, missing feedback, possession or control loss
end the old plan explicitly; it is not silently redirected to retain the old
opportunity score. Requested C presses, accepted windups, observed releases,
fresh shooter-attributed shot counters, goals and native goalie impacts have
separate accounting. Unaccepted presses get at most two explicit fresh-edge
retries. Native post-shot unavailability does not discard a shot already in
progress, and inputs stop when control belongs to another skater.

### Scenario and iteration protocol

`tests/integration/deke.py` reuses the coherent full-team RAM initializer,
including fixed-point placement, assignments, possession and rebuilt object
ordering. The goalie and defenders run the real ROM AI, not an idle second
human controller. Eight families cover central and wing approaches, wrong-way
nearby pressure, trailing/closing defenders, an immediate blocker, an already
open shot and excessive forward speed. Four consecutive seeds cover both deke
directions and handedness values at both attacking ends.

Each policy restores the same stored emulator snapshot in the same emulator;
the runner verifies matching full exposed-RAM hashes before acting. Core-private
serialization bytes are not assumed to re-serialize deterministically. Each
opportunity has the same 288-frame maximum observation window: a deke abort
does **not** stop the trial or hide subsequent fallback behavior. Phase/rejection
timelines, outcomes, initial hashes and source fingerprints are retained.

```bash
python -m tests.integration.deke --check --seeds 4 \
  --output docs/benchmarks/classic-v1-deke-development.json
python -m tests.integration.deke --seed 72000 --seeds 4 --jitter 6 \
  --output docs/benchmarks/classic-v1-deke-heldout.json

nhl94 benchmark-cpu --trials 3 --seed 73000 --seconds 300 --workers 2 \
  --output docs/benchmarks/classic-v1-deke-cpu-baseline.json
# Repeat the CPU command with --deke and a distinct output path.
```

The initial revision's fixed native contracts verified actual tracking, a cut, accepted windup,
applied C duration, attributed release/shot and no goalie contact in the
positive fixture, in both action formats. They also verify eligibility with a
nearby moving CPU defender, neutral interruption after a RAM ownership change,
and identical baseline actions for a rejected high-speed approach. These
assertions establish execution, not improvement over the old finisher.

The original [development cohort](benchmarks/classic-v1-deke-development.json) contains
64 matched opportunities per policy, on seeds 71000-71003. Baseline/deke totals
were 12/13 goals, 79/58 shots and 11/13 native goalie-impact impulses.
There were 40 deke plans, 30 observed tracking/cut transitions, two accepted
and recorded shots, and one attributed deke goal. This small development result
is not a held-out strength claim.

The separate [held-out cohort](benchmarks/classic-v1-deke-heldout.json) uses
unused seeds 72000-72003 and position jitter up to six units, again with
64 opportunities per policy. Baseline/deke totals were 13/11 goals,
75/64 shots and 10/17 goalie-impact impulses. Its 41 plans produced 28
tracking/cut transitions, three accepted and recorded shots, and one attributed
deke goal; 36 were invalidated and one reached the fixture cutoff. The result
does not show better finishing and is not used to retune these frozen thresholds.

### Frozen CPU result: do not enable by default

The [baseline](benchmarks/classic-v1-deke-cpu-baseline.json) and
[enabled report](benchmarks/classic-v1-deke-cpu-enabled.json) each contain nine
completed five-minute first periods: three unused seeds, 73000-73002, across
Pittsburgh/Ottawa, Ottawa/Pittsburgh and Quebec/Montreal. Controls are
`FILTERED`, the offensive interval is four frames, and goalie assistance and
cross-crease finishing are off. Source fingerprints, initial saves, teams,
lineups and seeds match between policies.

| Policy | Goals for / against | Zone turnovers | Native goalie-impact impulses |
|---|---|---|---|
| Baseline | 14 / 4 | 39 | 39 |
| `--deke` | 7 / 8 | 37 | 21 |

The enabled policy selected 22 plans, observed nine tracking/cut transitions,
and completed **zero deke C attempts or shots**. Twenty-one plans were
invalidated by fresh safety checks; one remained active at period end.
This exposes a selection-to-execution gap, not a successful match tactic.
Reduced general contacts do not compensate for the scoring/concession
regression and are not a causal per-deke collision comparison. The sample is
small, but it gives no basis to enable this prototype by default.

### Consistency and native-movement refinement

The initial selector approved paths that the live controller would reject:
some target depths were below its 202-unit finishing gate, imagined cuts did
not wait for goalie tracking, a single release point replaced the full X
envelope, and planning used different clearance semantics. The refined
selector runs the **same controller** on a projected scene. It includes the
same tracking trigger, deadlines, pressure/24-frame skating guard, full
release envelope, hold/tap decision and immediate-shot exit. Candidate targets
sit beyond the shared finishing gate by the existing target-controller arrival
tolerance. It no longer teleports the goalie into a favorable bait response or
zeros its momentum.

At this historical refinement, `agents/skating.py` was deliberately deke-only.
The later [experimental carry and receiving continuations](#experimental-carry-and-receiving-continuations)
also reuse its native updates. Native grounded movement applies
signed friction (at least one raw unit), integrates the full 16.16 position,
then applies input. Turns retain fractional facing and can accelerate along the
current facing while rotating; a six-frame acceleration-free sector turn was
not faithful to the ROM. The native speed limit rejects additional acceleration
even after an over-limit impulse. Cross-crease and legacy setup-cut/goalie
projections retain their existing conservative motion helper.

```bash
python -m tests.integration.deke_motion
python -m tests.integration.deke --check --seed 74000 --seeds 1 \
  --output docs/benchmarks/classic-v1-deke-refinement-development.json
python -m tests.integration.deke --seed 75000 --seeds 1 --jitter 6 \
  --output docs/benchmarks/classic-v1-deke-refinement-heldout.json
```

Movement calibration covers 14 actual-ROM traces (both ends; lateral,
diagonal, neutral, forward and opposite inputs), sampled at 1/4/8/12/24 frames.
Full positions, velocities and fractional facing matched exactly. This
establishes grounded carrier mechanics, not a full emulator clone: projections
still approximate future puck-sprite offsets and uncontrolled skater choices.
CPU-goalie projection uses its real retained direction and decision cadence,
preserves inertia, and suspends new decisions during active/locked animations.
Nonordinary positioning branches are rejected. Changed future headings use
ROM-derived all-direction release bounds/durations rather than stale narrow
current-pose geometry; this intentionally favors false negatives over
promising an unvalidated opening. New feedback remains read-only and outside
the neural schema.

The current native finishing contract deliberately **starts after a cut** to
check accepted windup, applied C duration, attributed release/shot and no goalie
contact in both action formats. It is not an end-to-end successful deception
claim. Separate contracts verify pre-execution rejection of the old unsafe
pressured maneuver, neutral ownership interruption and identical baseline
actions for the rejected fast approach. Historical v1 reports above remain
unchanged; new scenario reports use protocol v2 and include the calibrated
helper in their source fingerprints.

Fresh refinement measurements use new seeds, not the historical development
or held-out cohorts. These are small diagnostic comparisons, not a promotion
test:

| Cohort | Opportunities/periods per policy | Baseline | Refined `--deke` |
|---|---|---|---|
| [Development](benchmarks/classic-v1-deke-refinement-development.json), seed 74000 | 16 opportunities | 2 goals, 20 shots | 2 goals, 20 shots; no plans |
| [Held-out](benchmarks/classic-v1-deke-refinement-heldout.json), seed 75000, jitter 6 | 16 opportunities | 5 goals, 19 shots | 4 goals, 19 shots; one plan, no deke C attempt |
| CPU, seed 76000 | 3 completed 300-second first periods | 5 for / 1 against | 5 for / 1 against; no plans |

All development action hashes match baseline. Fifteen of sixteen held-out
action hashes match; the remaining wing opportunity tracked/cut, then missed
its cut deadline without a shot. The
[CPU baseline](benchmarks/classic-v1-deke-refinement-cpu-baseline.json) and
[enabled](benchmarks/classic-v1-deke-refinement-cpu-enabled.json) have matching
source/initial-state fingerprints and identical action hashes in all three
games. Equality therefore means **the revised tactic declined to intervene**,
not that it became effective. The selection/execution contradictions are
removed, but the maneuver family remains too slow or conservative and the
remaining world-model uncertainty can still invalidate a predicted finish.
It stays experimental and off by default.

Learning is a possible next tactical layer, not part of this refinement.
Matched real-ROM outcomes can train a ranker over maneuver choices, then
potentially a short-horizon policy for steering and shot timing. Keep native
ownership/action attribution, bounded execution and collision checks
deterministic. Learned scores need untouched evaluation starts and CPU matches;
neither a calibrated movement model nor an isolated successful shot establishes
better match play.

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

### Isolated normal-game player/goalie probe

`nhl94_ai.evaluation.cross_crease_probe` separates the finishing technique from
full-team approach/selection. It uses the ordinary full-team ROM and one human
joystick, not shootout mode or an idle human-controlled goalie. Only the
controlled carrier and opposing CPU goalie remain active. The other ten player
objects have negative roles and are checked every frame to remain inactive and
stationary. Fixed-point positions, motion, possession, assignments and object
ordering are initialized through the existing defensive reset helpers.

The goalie profiles use neutral-hot/cold skill levels 1 and 6. Movement and
catch ratings become live bytes 5 and 30; defensive awareness becomes delay 14
and 7, rather than increasing with skill. Glove-left follows the loader's nibble
mask (5 and 14). Weight and goalie handedness are identical in both profiles.
These are controlled effective gameplay ratings, not menu overall ratings.
The carrier has fixed power 30, accuracy 20, speed/agility 20 and weight 64;
consecutive seeds alternate carrier handedness.

Four prepared starts vary attacking depth, lateral speed and starting width:
`deep`, `near`, `slow` and `wide`. `rest-near` starts stationary in the close
lane; `rest` starts farther back and requires a diagonal approach. Both ends
and crossing directions are exercised. Each strategy restores the same stored
snapshot in the same emulator, with matching initial RAM for the same goalie
profile. `stationary` is an explicit counterfactual: only carrier/puck velocity
is cleared, keeping geometry, ratings and C timing unchanged.

The controls are an immediate four-frame tap, a forced 24-frame held-C crossing,
a stationary charged shot and the current full Classic cross-crease controller.
The forced policies do not use opportunity-selection gates. After release,
inputs are neutral; subsequent ordinary fallback is not counted as finishing
success. Requested C, native windup, attributed release/shot/goal, pre-release
save animations and fresh contact impulses are recorded separately. The
observation horizon is 304 frames, with a short counter-settling interval after
goalie possession. Full timelines can be retained with `--trace`.

```bash
python -m tests.integration.cross_crease_isolated
python -m nhl94_ai.evaluation.cross_crease_probe --seeds 4 --trace \
  --output docs/benchmarks/classic-v1-cross-crease-isolated-development.json
gzip -n docs/benchmarks/classic-v1-cross-crease-isolated-development.json
python -m nhl94_ai.evaluation.cross_crease_probe --seed 82000 --seeds 4 \
  --jitter 3 --trace \
  --output docs/benchmarks/classic-v1-cross-crease-isolated-heldout.json
gzip -n docs/benchmarks/classic-v1-cross-crease-isolated-heldout.json
python -m nhl94_ai.evaluation.cross_crease_probe \
  --scenarios deep near slow wide --seeds 1 --sweep \
  --output docs/benchmarks/classic-v1-cross-crease-isolated-timing.json
```

`--sweep` evaluates 42 combinations of press delay (0/4/8), hold duration
(1/4/8/12/16/24/60) and aim relative to crossing (+/-1). Results are grouped by
timing, not treated as one policy's success rate. A 60-frame request cannot
postpone the native automatic release indefinitely.

The initial isolated measurements establish that the technique can beat a
strong goalie and is not merely a harder charged shot: all four mirrored
near-lane held crossings scored against the high profile, whereas neither taps
nor stationary charged shots scored. Classic can also build momentum and score
from rest in the close lane. The farther-back diagonal approach is a distinct
failure: it arms while drifting away from goal and does not induce the same
pre-release save response against the strong goalie.

The preserved [pre-change diagonal baseline](benchmarks/classic-v1-cross-crease-isolated-rest-before.json)
records this failure. A trial restriction to a deeper predicted release window
worsened outcomes and introduced contacts, so it was reverted rather than
loosening safety or declaring the approach fixed. This diagnostic work does not
change production cross-crease tactics or enable the experimental flag by
default. Prepared success is not evidence of match strength.

The frozen [development report](benchmarks/classic-v1-cross-crease-isolated-development.json.gz)
uses seeds 81000-81003; the
[held-out report](benchmarks/classic-v1-cross-crease-isolated-heldout.json.gz)
uses fresh seeds 82000-82003, with up to three units of position jitter and
768 raw velocity units of lateral-speed jitter. Stationary scenarios remain
stationary. Each report has 64 prepared opportunities per profile/strategy and
16 opportunities for each from-rest scenario. Gzip preserves every timeline
frame; it does not thin observations. Source fingerprints and matched starting
snapshot/RAM contracts were verified.

| Prepared starts | Goalie | Immediate tap | Held-C crossing | Stationary charged | Current Classic |
|---|---|---|---|---|---|
| Development | Low | 27/64 | 61/64 | 11/64 | 61/64 |
| Development | High | 0/64 | 54/64 | 3/64 | 50/64 |
| Held-out, jittered | Low | 26/64 | 46/64 | 8/64 | 49/64 |
| Held-out, jittered | High | 5/64 | 33/64 | 1/64 | 31/64 |

Entries are attributed goals, not button requests or shaped rewards. No
goalie-contact impulses occurred in either frozen cohort. The lateral crossing
therefore offers a real isolated advantage over both a tap and an equally
charged stationary shot, including against the strong profile, but its
sensitivity to small starting changes remains substantial.

From rest in the close lane, Classic scored 16/16 against each profile in
development and 12/16 against each in held-out starts. On the farther-back
diagonal approach it scored 14/16 against low and 0/16 against high in
development, versus 6/16 and 4/16 held out. All these goals came from actual
cross-crease attempts, not subsequent fallback. The approach/arming geometry
is still unresolved; weak-goalie success masks that deficiency.

The separate [42-setting timing sweep](benchmarks/classic-v1-cross-crease-isolated-timing.json)
used 16 mirrored prepared starts per profile, all one handedness. Immediate
arming with 24 or 60 requested hold frames and aim toward the crossing scored
16/16 against low and 14/16 against high, the best sweep results. Holding longer
than 24 did not improve that cohort. These are development measurements, not
reasons to retune on held-out starts or claim a universal release duration.

### Fixed-geometry incoming-velocity map

`nhl94_ai.evaluation.cross_crease_velocity` removes the geometry/speed confound
in the earlier prepared scenarios. Before the first input, carrier position is
exactly 35 units to the starting side of center at attacking depth 216, with
the puck ten units toward the crossing. The goalie starts at depth 250 and
twelve units to that same side. Current and previous carrier/puck positions
are fixed together; full signed velocities are independently encoded as
`round(velocity * 65536 / 17)`. Native feedback verifies the quantized values.
Heading remains lateral. Positive lateral speed means toward the crossing;
positive goalward speed means toward the attacking goal. Speeds are rink
units per emulator frame, not legacy signed velocity high bytes.

Each fixture restores one warmed snapshot before injecting each velocity pair.
Policies restore the same cell snapshot in the same emulator, with identical
exposed RAM per goalie profile. Both physical ends, crossing directions and
carrier handedness are covered. First-C position and velocity are recorded:
Classic can skate before arming, so its initial velocity is not automatically
its charge velocity. Forced held-C starts immediately, requests 24 C frames
and aims toward the crossing. In the clean replicated cells, native release
is observed at frame 37; directions during accepted windup do not add skating
acceleration.

```bash
python -m tests.integration.cross_crease_velocity
python -m nhl94_ai.evaluation.cross_crease_velocity --seeds 2 --workers 2 \
  --trace --output docs/benchmarks/classic-v1-cross-crease-velocity-development.json.gz
python -m nhl94_ai.evaluation.cross_crease_velocity --seed 84000 --seeds 2 \
  --workers 2 --trace \
  --output docs/benchmarks/classic-v1-cross-crease-velocity-heldout.json.gz
python -m nhl94_ai.evaluation.cross_crease_velocity --seed 85000 --seeds 16 \
  --lateral 1 1.25 1.5 2 2.5 3 --goalward 0 0.25 0.5 --policies held \
  --workers 2 --trace \
  --output docs/benchmarks/classic-v1-cross-crease-velocity-replication.json.gz
```

The frozen [development](benchmarks/classic-v1-cross-crease-velocity-development.json.gz)
and [fresh-seed](benchmarks/classic-v1-cross-crease-velocity-heldout.json.gz)
maps each contain 2,160 trials across 45 velocity pairs, two goalie profiles
and tap/held/Classic policies. The first eight-trial cells were seed-sensitive:
strong-goalie held crossings at lateral 1.5 and zero goalward drift scored
8/8 in development but 4/8 on the fresh seeds. Accordingly, a separate
[16-seed replication](benchmarks/classic-v1-cross-crease-velocity-replication.json.gz)
tests 18 candidate pairs with 64 opportunities per profile/pair, using seeds
85000-85015. The three reports total 6,624 trials and preserve all 1,023,360
observed frames, source hashes and initial snapshot/RAM hashes.

Replication results below are **attributed goals without native goalie-contact
impulses**, out of 64 opportunities. An asterisk marks a cell containing any
contact trials; those trials are not counted as safe goals.

| Initial lateral speed | Low, goalward 0 | High, goalward 0 | High, goalward +0.25 | High, goalward +0.5 |
|---|---|---|---|---|
| 1.0 | 20/64 | 18/64 | 2/64* | 0/64* |
| 1.25 | 49/64 | 50/64 | 2/64* | 0/64* |
| 1.5 | 51/64 | 51/64 | 0/64* | 0/64* |
| 2.0 | 53/64 | 54/64 | 53/64 | 26/64* |
| 2.5 | 50/64 | 49/64 | 50/64 | 51/64 |
| 3.0 | 33/64 | 32/64 | 38/64 | 19/64 |

At this geometry, roughly 1.25-2.5 incoming lateral speed with no goalward
drift is a useful observed band, not a guaranteed goal or universal minimum.
Increasing speed indefinitely does not improve finishing: at 3.0, the body
releases near crossing X +48 and the puck near +60, rather than near the net's
center. With lateral 1.25, the body releases just short of center while the
puck is already across it; requiring body-center crossing would discard
successful stick/puck geometry.

Goalward drift changes the safe band substantially. At lateral 1.5 and
goalward +0.25, all 64 strong-goalie trials make contact and none score safely.
At lateral 2.0, goalward +0.25 instead scores 53/64 without contact. Increasing
that drift to +0.5 produces 32/64 contact trials, while 2.5 lateral clears
those contacts. High incoming impulses are a diagnostic, not a promise that
ordinary skating can sustain every tested speed.

The map also isolates a **selection mismatch**, not merely a holding failure.
At lateral 2.0 and goalward +0.25, Classic takes an ordinary four-frame C shot,
with no cross-crease attempt/event, in all sixteen development/fresh-seed
strong-goalie trials; none score. The matched forced crossing scores 12/16,
and the larger replication scores 53/64 without contact. These fallback taps
must not be counted as failed executions of a selected crossing. The planner's
forecast/selection therefore remains a separate fix target. This measurement
does not change production tactics, enable the flag by default, establish
defender tolerance or demonstrate improved match strength.

### Distance and intervening-player comparisons

`nhl94_ai.evaluation.cross_crease_distance` tests distance independently of
incoming lateral momentum. `--distances` means the **initial longitudinal
body-to-goalie gap**, not Euclidean distance or distance to the goal line.
The goalie starts at depth 250, the carrier at `250 - distance`, with fixed
crossing width 35 and zero goalward drift by default. Thus gap 34 corresponds
to roughly 41 units of body-center distance. Both ends, crossing directions,
handedness and controlled goalie profiles remain paired. Actual body/puck
separations are retained at initialization, first C and observed release;
Classic's later arming distance is not confused with its starting distance.

Forced shots aim at middle height by default. `--heights high` adds world UP
during shot aim, including when attacking down the rink; low uses world DOWN.
These are shot-height choices, not goalward acceleration during windup.
Classic is measured once per fixture under height `policy`, with its own
unchanged aim. `--goalward`, `--press-frame`, `--hold-frames` and `--aim` allow
separate controlled experiments without retuning production tactics.

The single intervening-player fixtures are:

| Traffic kind | Initial placement | Movement/decisions |
|---|---|---|
| `none` | No extra active skater | Original isolated carrier/CPU goalie |
| `friendly-initial`, `opponent-initial` | Midpoint of initial puck and goalie | Collidable, autonomous AI disabled |
| `opponent-release` | Crossing X +20, halfway in longitudinal depth | Collidable blocker in the prospective release lane |
| `pursuit-initial` | Same initial puck-line placement | Ordinary native nearest-puck pursuit |
| `friendly-between`, `opponent-between` | Midpoint of carrier and goalie bodies | Fits the close lane with valid body clearance |
| `pursuit-between` | Same body-midpoint placement | Ordinary native nearest-puck pursuit |

Each nonempty fixture has nine inactive actors rather than ten. The active
traffic skater has role 3, fixed neutral synthetic ratings and energy 4096.
Idle actors use the verified assignment-zero return instead of a second idle
joystick. Reassignment is suppressed without teleporting them or resetting
collision impulses. Pursuers keep their native assignment, decisions,
animations and physics; they can choose to guard rather than always advance.
Body spacing is checked jointly with carrier and goalie before measurement.
An actor already touching another body is not a valid starting screen.

Native `ltplayer` feedback distinguishes actual touches from mere proximity.
Touches/catches after a goalie touch are classified as rebounds, not direct
shot blocks; same-frame ordering uncertainty is retained explicitly.
Body-versus-stick shot deflections remain grouped. Steals, recorded opponent
checks, knockdowns, goalie contacts and ambiguous launch/whiff cases are
separate. A goalie save requires an actual touch without a goal, not just an
animation. After the original carrier first loses possession, all its inputs
stay neutral even if it recovers the puck; another attempt cannot inflate the
trial. Contact-free goals exclude carrier/goalie and traffic body impacts.

```bash
python -m tests.integration.cross_crease_distance
python -m nhl94_ai.evaluation.cross_crease_distance --seeds 4 --workers 2 \
  --trace --output docs/benchmarks/classic-v1-cross-crease-distance-development.json.gz
python -m nhl94_ai.evaluation.cross_crease_distance --seed 87000 --seeds 4 \
  --workers 2 --trace \
  --output docs/benchmarks/classic-v1-cross-crease-distance-heldout.json.gz
python -m nhl94_ai.evaluation.cross_crease_distance --distances 50 90 130 \
  --lateral 1.5 2 --traffic none friendly-initial opponent-initial \
  opponent-release pursuit-initial --heights middle high --seed 88000 \
  --seeds 2 --workers 2 --trace \
  --output docs/benchmarks/classic-v1-cross-crease-traffic-development.json.gz
# Repeat with seed 89000 and the traffic-heldout output name.
python -m nhl94_ai.evaluation.cross_crease_distance --distances 34 \
  --lateral 1.25 1.5 --traffic none friendly-between opponent-between \
  pursuit-between --heights middle high --seed 90000 --seeds 8 --workers 2 \
  --trace --output docs/benchmarks/classic-v1-cross-crease-traffic-close.json.gz
```

The [distance development](benchmarks/classic-v1-cross-crease-distance-development.json.gz)
and [fresh-seed distance](benchmarks/classic-v1-cross-crease-distance-heldout.json.gz)
reports each contain 2,688 trials. Pooling their seeds 86000-86003 and
87000-87003 gives 32 opportunities per distance/speed/profile/policy.
For forced middle-height held-C against the strong goalie:

| Initial longitudinal gap | Lateral 1.25: contact-free goals | Lateral 2.0: contact-free goals |
|---|---|---|
| 20 | 0/32 | 0/32 |
| 34 | 24/32 | 28/32 |
| 50 | 20/32 | 3/32 |
| 70 | 16/32 | 3/32 |
| 90 | 7/32 | 2/32 |
| 120 | 4/32 | 1/32 |
| 150 | 0/32 | 0/32 |

Too close is not automatically better: gap 20 produces frequent goalie
contact. At gap 34, every forced held trial observes a pre-release save
animation; at gaps 50 and beyond, none do. This is consistent with the ROM's
near-net windup trigger, but does not prove that trigger is the only cause.
The useful lateral band changes with distance: faster 2.0 is effective in
the close lane but much worse than 1.25 farther out. At gap 90, lateral 1.0
still scores 15/32 against high. The unchanged template's zero at gap 150 is
not proof that every possible distant crossing is impossible: forward
momentum, timing, starting width and aim are additional dimensions.

The [traffic development](benchmarks/classic-v1-cross-crease-traffic-development.json.gz)
and [fresh-seed traffic](benchmarks/classic-v1-cross-crease-traffic-heldout.json.gz)
reports each contain 2,400 trials, covering gaps 50/90/130, two speeds, five
traffic kinds, middle/high forced aim and Classic. Initial-line idle players
often have no effect because the lateral crossing leaves them outside the
actual release trajectory. Moving the blocker into that trajectory creates
real shot interceptions. Native pursuit adds steals and checks, not merely a
static blocked-lane flag. The two reports record 89 steal events, 14 recorded
opponent-check events and 135 direct block/deflection touches; 695 later
traffic touches are instead rebounds after goalie contact. Elevating the
shot is not an automatic cure for traffic or distance.

The [close-range replication](benchmarks/classic-v1-cross-crease-traffic-close.json.gz)
contains another 2,560 trials using fresh seeds 90000-90007. At gap 34,
body-midpoint traffic is at crossing X -23.5 and depth 233, with more than
twenty units of initial clearance from both bodies. For lateral 1.5,
middle-height shots and the high goalie:

| Traffic | Forced held-C contact-free goals | Current Classic contact-free goals |
|---|---|---|
| None | 26/32 | 25/32 |
| Standing friendly skater between | 26/32 | 0/32 |
| Standing opponent between | 26/32 | 0/32 |
| Native CPU pursuer between | 15/32 | 0/32 |

The standing actors do not touch the puck or either body in these forced
trials: their presence between the initial bodies does **not** block the
later crossing shot. No visual-screening benefit is established. Pursuit is
different: nine of the 32 lateral-1.5 held trials have body contact, four have
steals and one a direct shot intervention. There are 21 attributed goals,
but only 15 without contact. At lateral 1.25, the corresponding contact-free
results are 24/32 without traffic, 24/32 with either standing actor and
10/32 with the pursuer.

Classic selects crossings in all 32 no-traffic fixtures but chooses ordinary
shots in all 32 fixtures with either standing actor, scoring zero against
high. This is a reproducible traffic-selection mismatch, not evidence that
the physical technique requires an empty breakaway. Removing all safety
guards is not justified: the genuine pursuer substantially changes risk.

Across the five reports, 12,736 trials retain all 2,106,880 observed frames.
Measurement source/ROM hashes and matched snapshot/RAM identities are
preserved. The first four reports were reclassified after inspecting the
recorded contact sequence: `sources` still identifies their original
emulation, and `analysis_sources` identifies the updated classifier.
Trajectory/action hashes and native goal outcomes were verified unchanged.
`--reanalyze <reports...>` applies that idempotent classification without
replaying emulator frames. Historical isolation/velocity reports are not
regenerated. This work still does not change production tactics, model
inputs/checkpoints or the default experimental flags, and one synthetic
traffic actor is not evidence of full-team match strength.

### Full-team Ducks/Sabres on/off benchmark

The usual `SabresVsMightyDucks.ManualGoalie.Start` benchmark tests the current
policy unchanged, rather than implementing the isolated traffic-selection
fix first. The [crossing-off report](benchmarks/classic-v1-cross-crease-ducks-sabres-off.json)
and [crossing-on report](benchmarks/classic-v1-cross-crease-ducks-sabres-on.json)
each contain 40 completed 300-game-clock-second first periods: twenty seeds
`20261004-20261023`, with Classic controlling Buffalo home and Anaheim away.
Controls are `FILTERED`, offensive interval four, goalie assistance off and
skating dekes off. Every skater remains active; the other team is the ordinary
ROM CPU. The only policy difference is `--cross-crease`.

Runtime fingerprints, saved starting-state hashes, teams, starting lineups
and effective starting accuracy match across every paired fixture. Saved
hot/cold ratings are not regenerated by post-load RNG seeding. All eighty
periods reach clock zero. Goals and W/D/L below are from Classic's perspective,
with twenty periods per side/condition:

| Classic side | Crossing off GF-GA | Crossing on GF-GA | Off W/D/L | On W/D/L |
|---|---|---|---|---|
| Sabres home vs Ducks | 37-7 | 36-6 | 16/3/1 | 15/4/1 |
| Ducks away vs Sabres | 18-5 | 20-14 | 12/7/1 | 9/6/5 |
| Combined | 55-12 | 56-20 | 28/10/2 | 24/10/6 |

**This does not establish stronger match play.** Buffalo's goal difference
is unchanged; Anaheim scores two more but concedes nine more. Combined goal
difference falls by seven. The [paired comparison](benchmarks/classic-v1-cross-crease-ducks-sabres-comparison.json)
uses 10,000 ROM-seed-cluster bootstrap draws, seed 94, retaining both AI sides.
The 95% interval for total goal-difference change over forty paired periods
is **-28 to +13**. Concessions increase by eight, interval **+2 to +14**.
These are established development seeds, not held-out confirmation, and
period W/D/L is not a full-game win rate.

| Classic side | Selected plans | C attempts | Native windups | Recorded shots | Attributed crossing goals | Goalie contacts |
|---|---|---|---|---|---|---|
| Sabres home | 204 | 28 | 28 | 25 | 6 | 5 |
| Ducks away | 144 | 8 | 8 | 5 | 2 | 1 |
| Combined | 348 | 36 | 36 | 30 | 8 | 6 |

Eight crossing goals are not eight extra team goals: total scoring increases
by only one. Of the 348 selected plans, 263 end with the approach invalidated
before C, and another seventeen miss their crossing window. One plan remains
open at the final period cutoff; it is not counted as a goal or diagnosed
abort. Candidate-rejection counters overlap and are not separate shot
attempts. Individual completed events verify the C/windup/shot/goal/contact
funnel. Every paired period changes its action trace when crossing is enabled.
The measured concession increase does not identify the mechanism of every
conceded goal, but the current planner is not justified as a default strength
improvement. Experimental flags and production tactics remain unchanged.

```bash
python -m nhl94_ai benchmark-cpu --agent classic-v1 \
  --matchups sabres-ducks-manual ducks-sabres-manual --goalie-policy off \
  --trials 20 --seed 20261004 --seconds 300 --frame-skip 4 \
  --action-type FILTERED --workers 2 \
  --output docs/benchmarks/classic-v1-cross-crease-ducks-sabres-off.json
python -m nhl94_ai benchmark-cpu --agent classic-v1 \
  --matchups sabres-ducks-manual ducks-sabres-manual --goalie-policy off \
  --trials 20 --seed 20261004 --seconds 300 --frame-skip 4 \
  --action-type FILTERED --workers 2 --cross-crease \
  --output docs/benchmarks/classic-v1-cross-crease-ducks-sabres-on.json
```

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
goalie's live contact depth; it stops at uncertain wall contacts. Incoming
receiver prediction uses friction and first modeled body/stick contact, not
closest approach. A fresh opposing pass counter can start anticipation while
the passer still holds the puck. A reachable receiver before the goalie plane
changes the angle target instead of triggering a save on the pass itself.
Stale receiver targets at the shooter do not suppress real released shots.
These are modeled receptions, not guarantees that a CPU receiver will catch
or one-time the puck.

Selective takeover initially budgets 24 frames for selection plus goalie travel
and a six-frame response margin. During B hold it uses the live countdown and
observed decrement spacing, not elapsed time subtracted from a fixed estimate.
It keeps close carrier contests, reachable moving-pass interceptions and useful
shooting-lane blocks from the predicted shot origin. Carrier release deadlines
remain estimates, not promises about an attacker's next move. It does not
interrupt a pending skater pass, one-timer, shot, switch or body check.

The controller runs every emulator frame. It releases old buttons, holds B
until the goalie slot **and joystick flag** confirm selection, releases B, and
then sends goalie input. A new short B press/release requests a skater; only an
actual skater slot confirms return. Handoff attempts time out after 60 frames
with a warning and bounded backoff. Movement uses goalie-specific acceleration
and strong neutral braking, rather than skater turning/boost estimates. Targets
remain near the crease; pulled/unavailable goalies are not takeover candidates.
Animation locks are respected. Offscreen CPU assistance receives neutral input
and its catches are not counted as manual catches.

Fresh C requests the ROM's save selection, which quarters normal-save X/Y
momentum. For a low/body-height shot, the controller finishes reachable lateral
alignment before committing C when the horizontal gap exceeds eight units.
Unreachable or high shots retain emergency saves; rare low-shot emergencies
can request A plus direction for a dive. Unconfirmed or cancelled requests
release their long cooldown so they cannot suppress the next real save.
Request counters are separate from confirmed save/dive animation counters;
neither is a count of prevented goals. Diagnostics distinguish holding the
target, save commitment/recovery, B handoff waiting and offscreen CPU fallback.
Neutral input during a locked save animation remains deliberate: D-pad input
cannot undo that ROM lock. On possession,
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

**Actuation and fairness:** Classic reads decoded RAM, so it is not a pixel-only
or human-information-limited agent. Its gameplay policy returns the ordered
binary button array; it does not write player/puck positions, velocities,
ratings, possession or scores. This goalie change adds no production RAM writes.
Existing environment save restoration, reset/seeding and away-controller routing
repair are unchanged. The routing repair corrects the ROM's joystick reassignment
bug; it is an environment-level RAM write, not a goalie tactic. Native fixtures
may arrange a test emulator before execution or restore identical snapshots.
The natural replay check separately verifies that every goalie policy call
leaves all 65,536 gameplay RAM bytes unchanged.

Focused native verification:

```bash
python -m tests.integration.manual_goalie
```

This checks actual B selection/return, D-pad movement/braking, C save contact,
A dive, low-speed catch, safe outlet/reception and A clear, then exercises the
actual watched home/away command. Contact fixtures write only the test emulator
after selection has been obtained by B; production handoffs never force RAM.

For an archived pre-fix package, the bounded natural replay check runs:

```bash
python -m tests.integration.goalie_timing \
  --baseline-root /path/to/archived-package --output /tmp/goalie-replays.json
```

It captures natural CPU opportunities without setting up a shot, then restores
each identical snapshot for both policies. It checks actual pre-contact lateral
movement, native one-timer counters with shooter attribution, respected animation
locks/CPU fallback, binary button shape and policy-call RAM immutability.
These short goalie-only continuations are not full Classic match benchmarks
and do not imply that every one-timer can be prevented.

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

### Goalie timing follow-up: matched CPU comparison

The [pre-fix selective report](benchmarks/classic-v1-goalie-timing-baseline-selective.json)
and [corrected selective report](benchmarks/classic-v1-goalie-timing-fixed-selective.json)
each contain **80 completed five-minute first periods**, 20 seeds per AI side,
**20261201-20261220**. Four independent emulator workers use `FILTERED`, offensive
interval four and cross-crease off. Match settings, saved-state hashes and lineups
match; `agents/goalie.py` is the only differing fingerprinted runtime source.
The baseline includes the preserved goalie edits present before this timing work,
not simply the goalie implementation in `5109c57`.

Scores and W/D/L are from Classic's perspective:

| Classic side | Pre-fix GF-GA | Corrected GF-GA | Pre-fix W/D/L | Corrected W/D/L | Opponent one-timer goals, before/after |
| --- | ---: | ---: | ---: | ---: | ---: |
| Sabres home vs Ducks | 25-2 | 36-4 | 16/4/0 | 17/2/1 | 0/2 |
| Ducks away vs Sabres | 27-10 | 28-5 | 10/7/3 | 14/6/0 | 7/3 |
| Ducks home vs Campbell | 16-13 | 19-15 | 5/10/5 | 8/8/4 | 6/5 |
| Campbell away vs Ducks | 47-8 | 44-7 | 18/2/0 | 17/2/1 | 3/3 |
| Combined | 115-33 | 127-31 | 49/23/8 | 56/18/6 | 16/13 |

The Ducks/Sabres result improves, but this is **not a universal defensive gain**:
Sabres concessions rise 2 to 4, Ducks/Campbell concessions rise 13 to 15, and
Campbell's goal difference falls 39 to 37. Opponent shots rise 284 to 300;
controlled catches fall 131 to 123. Fewer save animations, 175 to 163, are not
proof of more prevented goals. Both selective versions have zero handoff timeouts.

The [comparison](benchmarks/classic-v1-goalie-timing-comparison.json) uses 10,000
paired ROM-seed-cluster bootstrap draws, retaining all four fixture sides.
Combined goal-difference change is +14, with a 95% interval **-7 to +36** over
80 periods. Concession change is -2, interval **-14 to +10**; opposing one-timer
goal change is -3, interval **-10 to +5**. All include no improvement. The
corrected benchmark scores were not used to tune the controller, but four of
these seeds were inspected for natural replays, so this is a development
comparison rather than an untouched held-out strength result.

A separate [80-period CPU-goalie reference](benchmarks/classic-v1-goalie-timing-baseline-off.json)
records GF-GA/WDL of 33-9/12/5/3 for Sabres, 17-14/8/8/4 for Ducks vs Sabres,
17-16/6/6/8 for Ducks vs Campbell and 48-5/18/2/0 for Campbell.
The [disabled-policy smoke](benchmarks/classic-v1-goalie-timing-fixed-off-smoke.json)
repeats seed 20261201 on all four sides: every complete match record equals the
corresponding CPU-goalie baseline, including per-frame action hashes.
Default `--goalie-policy off` is unchanged.

The [native replays](benchmarks/classic-v1-goalie-timing-native.json) contain
three natural reachable alignment opportunities and four attributed native
one-timer releases. Alignment improves in actual pre-contact movement while
policy calls leave RAM unchanged. Both policies still concede the four captured
one-timers and enter save/dive recovery during their flight. These cases expose
remaining limits; they are not successful-save examples or proof that earlier
pass anticipation would prevent those goals. Locked animations remain respected.
Existing native B handoff, C contact, A dive, catch, outlet, return and watched
home/away checks also pass. All 544 unit tests pass with one existing expected
failure; full pylint passes.

The [baseline runtime patch](benchmarks/classic-v1-goalie-timing-baseline-runtime.patch)
and [corrected runtime patch](benchmarks/classic-v1-goalie-timing-fixed-runtime.patch)
reconstruct fingerprinted runtime sources from `5109c57`. All reported periods
are first-period cutoffs, not full games. Save ratings are fixed; post-load RNG
seeding does not regenerate hot/cold tables. Manual goalie remains opt-in.

```bash
nhl94 benchmark-cpu --agent classic-v1 \
  --matchups sabres-ducks-manual ducks-sabres-manual \
    ducks-campbell-manual campbell-ducks-manual \
  --goalie-policy selective --trials 20 --seed 20261201 --seconds 300 \
  --frame-skip 4 --action-type FILTERED --workers 4 \
  --output docs/benchmarks/classic-v1-goalie-timing-fixed-selective.json
```

### Pre-pass one-timer component experiment

The [pre-pass experiment](benchmarks/classic-v1-goalie-prepass-experiment.json)
revisits the four failed one-timers from the native report, beginning **180
emulator frames before the last accepted pass**. Emulator RAM/state, decoded
state history and the complete Classic model are restored together. This is
not the earlier short goalie-only continuation with idle skaters. An initial
faceoff/stoppage is replayed normally rather than mistaken for the attack's end.
The corrected-policy branch starts from the same archived baseline history;
it is not a fresh corrected-policy match from reset.

All four archived full-model continuations reproduce exactly, including their
per-frame buttons and trajectories, and duplicate controls are identical.
Seven variants per case plus duplicate baseline controls make **32 bounded
continuations**. Every policy call preserves all 65,536 gameplay RAM bytes.
Production policy code, ratings and controller-routing repairs are unchanged.
These are selected failures, not a random sample or a match-strength benchmark.

Each component variant inherits the pre-fix controller. Receiver anticipation
changes only the positioning target/reason, retaining the old crossing,
deadline, save and takeover rules. Alignment-only substitutes the current
reachable-alignment gate, retaining old targets/takeover rules. No-save
commitment suppresses new C/A save requests but keeps movement, catches and
outlets; this is diagnostic, not a proposed policy. Earlier takeover uses the
existing `always` policy. The CPU variant declines takeovers and returns by B
when safe, respecting animation recovery and automatic goalie possession.

Numbers below are **goals against within each replay window**, not period scores:

| Archived failure | Baseline | Corrected | Anticipation only | No save commitment | Earlier takeover | CPU goalie |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Ducks vs Sabres, seed 20261204 | 1 | 0 | 0 | 1 | 1 | 0 |
| Ducks vs Campbell, seed 20261211 | 1 | 1 | 1 | 0 | 1 | 0 |
| Campbell vs Ducks, seed 20261214 | 1 | 1 | 1 | 1 | 1 | 0 |
| Ducks vs Campbell, seed 20261203 | 1 | 1 | 1 | 1 | 0 | 0 |

Alignment-only changes no buttons in these four cases and concedes all four.
Replays stop at the first stoppage after live play begins or 64 frames after
the archived goal. A zero without a confirmed contact is only no goal at that
cutoff, not a claimed save.

**Demonstrated save-commitment failure:** Ducks/Campbell seed 20261211 has the
same native recipient release at frame 329 and identical puck history through
that release. At frame 343, the baseline goalie is at `(-6, -238)`, moving left
at 1.53 units per physics update. C starts a save and the next observation
reduces that speed to 0.38. The movement-only intervention keeps LEFT, reaching
1.58 instead. The puck reverses away from the net at frame 349 and goalie
ownership confirms the catch at frame 358, without a goal. The alignment gate
rejects this opportunity because modeled reach 7.02 plus its one-frame margin
exceeds crossing time 7.26. This establishes that committing C was harmful in
this reproduced state; it does not establish a generally safe replacement gate.
ROM physics-update timing can also diverge after changing input, so this is a
real button-level counterfactual, not a frozen scripted puck trajectory.

**Anticipation sensitivity:** Ducks/Sabres seed 20261204 changes its first input
at frame 6781, before the final pass at 6814. The anticipation-only and corrected
branches have no goal within the window, but puck height/physics timing first
differs at 6800, before that final pass. The same requested recipient does
record a one-timer; this is not an identical released-shot rescue.

**Remaining positional limits:** Campbell/Ducks seed 20261214 is about 29 units
from its modeled crossing X at release, with crossing time 12.23 and modeled
target reach 15.17. Removing C still concedes. Ducks/Campbell seed 20261203
releases only four frames after the pass; its goalie is at `(31, -262)` while
the modeled target is near `(3.23, -254)`, crossing in 12.32 versus estimated
reach 19.23. Removing the dive also concedes. These identify position/reach
shortfalls, not proven fixes or a complete collision-radius diagnosis.

**Takeover is not a pure goalie comparison:** earlier takeover changes no
buttons in the first three recorded windows. It avoids the fourth goal, but
changes the puck trace at 4656 and the original one-timer is not reproduced.
The CPU-goalie branch has no goals in any window, but all four pre-release
puck traces differ from the baseline; two no longer reproduce the requested
recipient's one-timer. Giving skater defense back to Classic changes the CPU
attack, so this does not prove that CPU-controlled Belfour alone is superior.

The experiment passes the full 551-test suite with one existing expected
failure and full pylint. No new production heuristics were added. To replay:

```bash
python -m tests.integration.goalie_prepass \
  --baseline-root /path/to/archived-package \
  --output docs/benchmarks/classic-v1-goalie-prepass-experiment.json
```

Use the earlier baseline runtime patch to reconstruct the archived package;
the harness verifies its source fingerprints before running.

## Carrier-cutoff follow-up: neutral-zone pursuit

**Historical experiment, reverted.** `agents/defense.py` has been restored
byte-for-byte from `5109c57`; the earlier offensive fixes remain intact.
The carrier-cutoff and depth-120 retreat behavior below are not active.
Read-only benchmark telemetry and the raw reports/source patches are retained.
The experimental native fixtures and unit tests are archived separately so they
can be reconstructed without leaving experimental behavior in the default agent.

**Restoration verification:** a fresh
[80-period rerun](benchmarks/classic-v1-carrier-cutoff-restored.json), using four
isolated worker processes and the same seeds/settings, reproduces **all 80
baseline match records exactly**, including per-frame action hashes, goals,
shots, recoveries, lengths, starting lineups and save hashes. Every summary
field also matches. This is an exact replay check, not a new strength estimate.
Concurrent manual-goalie edits were preserved; they are the sole runtime-source
difference in the frozen rerun, and that controller is disabled by goalie policy
off. The restored tree passes 531 unit tests with one existing expected failure
and the Classic-defense ROM checks. Full pylint separately reports `R0916` at
`agents/goalie.py:268` in that unrelated edit; restoration did not modify it.

The carrier-cutoff controller was compared with committed controller
`5109c57fa493249c82efe3088268bd3e4acbdde3` on **160 completed CPU first periods**:
20 matched seeds per AI side per version, using both sides of Mighty Ducks–Sabres
and Mighty Ducks–All-Star Campbell. Seeds **20261101–20261120** were held out from
the preceding offensive study; the controller was not tuned to these scores.
Each period uses a 300-second clock, `FILTERED` actions, decision interval four,
goalie policy off and cross-crease off.

| AI side | Baseline GF–GA | Experimental GF–GA | Baseline W/D/L | Experimental W/D/L |
| --- | ---: | ---: | ---: | ---: |
| Sabres, home | 31–5 | 52–5 | 14/5/1 | 17/1/2 |
| Ducks vs Sabres, away | 26–6 | 31–14 | 15/3/2 | 10/7/3 |
| Ducks vs Campbell, home | 29–14 | 27–19 | 12/2/6 | 8/8/4 |
| Campbell, away | 40–7 | 44–5 | 17/2/1 | 18/1/1 |
| Combined | 126–32 | 154–43 | 58/12/10 | 53/17/10 |

Combined goal difference increases by 17, or **+0.2125 per period**, but its
paired seed-cluster bootstrap 95% interval is **[-0.2875, 0.7125]**, including zero.
Buffalo supplies most of the gain; both Ducks fixtures regress in goal difference.
Period wins fall by five, draws rise by five and losses are unchanged. These
are first-period outcomes, not full-game win rates or an across-the-board
strength improvement.

The more direct positioning measurement is the selected defender's time behind
an opponent carrying the puck in neutral ice:

| AI side | Baseline behind-carrier frames | Cutoff behind-carrier frames |
| --- | ---: | ---: |
| Sabres, home | 31.29% | 30.03% |
| Ducks vs Sabres, away | 33.51% | 26.79% |
| Ducks vs Campbell, home | 28.92% | 23.00% |
| Campbell, away | 29.44% | 20.74% |
| Combined | 30.69% | 25.16% |

The combined change is **-5.53 percentage points**, with paired bootstrap 95%
interval **[-9.10, -1.74]**. It conditions on live opponent skater possession and
a valid selected defender; altered opportunity mix can affect the fraction.
Goal-side positioning is not itself a confirmed block. The experimental controller recorded
3,706 neutral-zone cutoff-plan frames and 7,250 retreat-lane frames. Neutral poke
requests increase from 137 to 165, but close-range frames fall from 15,591 to
13,712 and direct controlled recovery from a carried opponent puck falls from
five to three. Total defensive controlled recoveries, including loose-puck
recoveries, fall **816 to 774**. Thus less trailing is not proof of better
puck-winning.

The independent ROM regression uses an approaching, human-controlled carrier
and isolated effective ratings. At both defending ends, the original target
makes no carrier contact within 64 frames; the corrected route makes attributed
goal-side contact and releases possession at frames **20 and 18**, before the
blue line. Both action schemas produce identical traces. Separate cases preserve
an existing contact and verify that an unreachable pursuit selects the fixed
depth-120 lane without claiming an interception.

Frozen source manifests, save hashes and initial lineup/rating identity were
verified for every matched pair. Both CPU versions use the same read-only
carrier-defense instrumentation; four paired 20-second diagnostics confirm that
enabling it leaves action hashes, scores, shots, lengths and recoveries unchanged.
Only `agents/defense.py` differs behaviorally between the official versions.
An earlier changed-controller run was stopped before completion to fix premature
cutoff cancellation at the blue line; it contributes no official comparison
periods. The experimental snapshot passed 541 unit tests with one existing expected
failure, full pylint and the explicit carrier-cutoff/Classic-defense ROM checks.

The experiment used an 8–32-frame arrival-time search, an eight-unit goal-side
cutoff and a four-frame arrival margin. It retained its point and absolute
deadline, cancelled invalidated crossings with a bounded retry, and allowed a
reachable commitment to finish across the blue line. When a trailing defender
had no reachable cutoff, the new fallback recovered a depth-120 inner lane.
These combined tactical changes reduced trailing but did not establish better
puck-winning or Ducks results; neither is retained as an untested partial revert.

The [paired comparison](benchmarks/classic-v1-carrier-cutoff-comparison.json)
links raw [baseline](benchmarks/classic-v1-carrier-cutoff-baseline.json),
[final](benchmarks/classic-v1-carrier-cutoff-final.json) and
[native fixture traces](benchmarks/classic-v1-carrier-cutoff-native.json).
Reconstruct the baseline from commit `5109c57` plus the shared
[telemetry patch](benchmarks/classic-v1-carrier-cutoff-telemetry.patch), then apply
the [behavior patch](benchmarks/classic-v1-carrier-cutoff-behavior.patch) for the
experimental controller. The archived
[test/fixture patch](benchmarks/classic-v1-carrier-cutoff-tests.patch) restores
the experimental tests on the same `5109c57` reconstruction. Patch replay
reproduces both runtime fingerprint manifests and the original test sources.
The comparison's original final-worktree verification describes the experimental
snapshot before restoration, not the current default.
Run from the reconstructed package directory, not a different checkout:

```bash
nhl94 benchmark-cpu --agent classic-v1 \
  --matchups sabres-ducks-manual ducks-sabres-manual \
    ducks-campbell-manual campbell-ducks-manual \
  --goalie-policy off --trials 20 --seed 20261101 --seconds 300 \
  --frame-skip 4 --action-type FILTERED --workers 4 \
  --output carrier-cutoff-report.json
```

To replay the historical native fixtures, also apply the test/fixture patch in
that reconstructed checkout, then run `python -m tests.integration.carrier_cutoffs`.
That module is deliberately absent from the restored default test tree.

ROM seeding changes play after save loading, not the saved hot/cold tables.
Fixed rosters, home/away differences and bounded constant-velocity forecasts
remain limitations. The shared telemetry is observational, not causal tackle
attribution; the isolated ROM fixtures are not a guarantee against real CPU
turns, acceleration or collisions.

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

CPU reports expose `carrier_defense_metrics` by rink zone:
frame counts describe positioning/plans; fresh button requests and direct
controlled recoveries are separate events. `goal_side_regained` measures a
geometric transition while the same skater and carrier remain selected, not a
confirmed shot-lane block or tackle.

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
| `advance-pass` | Require at least 20 units of safe gain plus a defender bypass, progress beyond the retained carry, a blue-line entry the carry cannot make, or a stronger shooting continuation. |
| `position-pass` | In the attacking zone, require immediate or post-carry shooting value more than 12 points above the modeled keep-puck opportunity, not merely a deeper receiver. |
| `carry-breakaway` | No goal-side skater in the nearby corridor, no modeled short-horizon interception and a contact-clear carrying route; execution shares the ordinary carrying safeguards. |
| `carry` / `carry-escape` | Keep a verified safe slot route; otherwise compare bounded escapes/braking/retreats. Rank safe alternatives by modeled shooting opportunity and actual projected progress, not unnecessary retreat once safety is established. |
| `carry-opportunity` | Execute the exact safe short-carry target credited with more than 10 points of shooting improvement when no finishing/pass/setup takes priority. |
| `feint` | A left/right cut must improve the modeled shot or receiving opportunity over both the current position and continuing straight. |
| `one-timer-setup` | A bounded cut must open a safe predicted one-timer without reducing the modeled opportunity; finish or abandon the cut rather than immediately replacing it with an ordinary pass. |
| `pass-release` / `pass-flight` | Release buttons and follow observed ownership/recipient feedback; an advancement pass must not trigger a one-timer C. |

### Opt-in action-conditioned chance creation

`--chance-creation` adds an experimental `create-chance` choice to full-team
Classic `FILTERED` and `HOCKEY_INTENT_DPAD` play. It is **off by default**.
It does not enable dekes, cross-crease or uncertified carrying, change neural
inputs, or install a separate learned policy.

The selector compares the ordinary slot route with two 26-unit cuts, braking
and holding near the current location. An alternative must model a stronger
ordinary-pass or one-timer shooting window than the immediate opportunities
and the ordinary route, after a **0.3-point/frame** delay cost and a
**10-point** improvement margin. These are opportunity heuristics, not goal
or completion probabilities. Existing close-shot, live one-timer, breakaway
and active setup priorities remain.

`agents/responses.py` models one stable-assignment skating scenario conditional
on each candidate carrier path. Offensive defensemen follow their native
puck-side point target; wings and centers retain their observed support
destinations and avoid the carrier through `EvadePC`. An observed opposing
`assnearest` skater uses its observed containment branch; committed direct
pursuit has different timing/contact rules and remains explicitly unsupported.
Steering retains
the live countdown, twelve-tick refresh, velocity compensation and eight-sector
direction quantization, with native grounded inertia, turning and stopping.
The scenario includes the nearest skater's slot-speed allowance.

This is **not a complete CPU/goalie simulator or a safety certificate**.
Future random support destinations, nearest-skater reassignment, checking,
backward-skating transitions, collisions and sprite hotspots are not predicted.
Unsupported actors retain the existing bounded velocity projection and are
listed separately; a support-zone transition withholds that actor's response
credit. The scenario does not inspect future ROM RNG. Live assignment, steering,
timers and support targets are optional, read-only `Player` feedback outside
the saved neural schema.

Every proposed carry still needs the original control-reach **`carry_safe`**
certificate, even with `--uncertain-carry`. A swept body-contact check on the
response scenario can additionally reject it; a favorable CPU response cannot
override a failed certificate. Predicted passing windows only select movement:
the actual receiver, lane and one-timer cue are reevaluated from live RAM before
any B/C request. Execution follows the exact evaluated target.
Known ordinary-pass and one-timer retry deadlines also withhold windows whose
predicted release decision would be too early.

A setup lasts at most **18 emulator frames**, not 18 frames renewed at every
replan. Subsequent forecasts shrink to its remaining time; expiry imposes a
72-frame retry delay and interrupts cached input on the exact emulator frame,
including at intervals 4 and 10. Friendly reception or lost possession cancels the old
setup. Diagnostics expose `chance_candidates`, the comparison baseline,
predicted receiver/purpose and `predicted-not-observed` window status.

```bash
nhl94 play --agent classic-v1 --env NHL94-Genesis-v0 \
  --state SabresVsMightyDucks.ManualGoalie.Start --chance-creation

python -m nhl94_ai.evaluation.chance_creation_probe \
  --output /path/to/chance-response.json
```

The explicit native probe covers stable center, wing, defenseman and opposing
nearest-skater motion at both rink ends and input intervals 1/4/10. Each pair
restores the same full emulator state and verifies initial RAM, active ownership,
assignment and stationary inactive actors. Its position tolerance concerns
these fixtures only, not arbitrary full-team response accuracy or playing
strength. Counterattack/defensive-cover scoring and deliberate EA-special
execution are separate, unimplemented tactics.

The [native development report](benchmarks/classic-v1-chance-creation-native.json)
contains **288 traces / 144 identical-state route pairs**, including different
carrier widths and observed puck offsets near the avoidance boundary. Maximum
18-frame endpoint error is **3.442 rink units**, within the predeclared four-unit
fixture tolerance. This does **not** mean every action-conditioned response is
predicted correctly: at a half-unit route-difference threshold, the ROM changes
30 pairs, the model changes twelve, and **18 real response signals are missed**.
There are no spurious signals at that threshold; maximum error in the paired
route difference is 0.736 units. Those negative boundary cases remain in the
report. These are development geometries, not held-out accuracy evidence.
Both action schemas use real `predict_frame` dispatch and input processing.
All traces interrupt the injected setup immediately after its eighteenth
native input frame, including cadences 4 and 10. Injected movement/deadline
choices diagnose execution; they are not ordinary-policy decisions or goals.

#### Matched CPU measurement (2026-10-07)

The corrected controller, including exact-frame setup interruption and known
pass retry availability, completed **80 first periods**: 20 seeds per AI side
per policy, `20262701..20262720`, against the actual built-in CPU in
`SabresVsMightyDucks.ManualGoalie.Start`. Both policies use 300 clock seconds,
four-frame offense and `FILTERED`; goalie assistance, dekes, cross-crease and
uncertified carrying are off. Only the candidate enables `--chance-creation`.
Starting saves, physical teams, lineups and initial accuracy match; all actors
remain active, sources/versions match and one-timer lifecycles reconcile.

The [paired report](benchmarks/classic-v1-chance-creation-comparison.json) links
the [default](benchmarks/classic-v1-chance-creation-baseline.json) and
[candidate](benchmarks/classic-v1-chance-creation-candidate.json) raw reports.
The [current default reference](benchmarks/classic-v1-chance-creation-reference.json)
and [native cadence gate](benchmarks/classic-v1-chance-creation-cadence.json)
preserve complete away-prefix action parity; both reference periods also match
the corresponding full-baseline frame counts and action hashes.

| AI side | Default W/D/L | Candidate W/D/L | Default GF-GA | Candidate GF-GA |
| --- | --- | --- | --- | --- |
| Sabres, home | 14/4/2 | 15/4/1 | 30-5 | 32-5 |
| Mighty Ducks, away | 9/8/3 | 9/8/3 | 16-5 | 17-6 |
| Combined | 23/12/5 | 24/12/4 | 46-10 | 49-11 |

**This is a small, sparse positive signal, not a default-policy promotion.**
Combined goal difference increases by two, or **0.05 per period**; its paired
95% ROM-seed-cluster bootstrap interval is **[0.00, 0.15]**, including zero.
Resampling uses 10,000 draws, seed 94, keeping both AI sides together.
The nonnegative interval does **not** establish an absence of downside:
resampling this sparse observed distribution cannot reveal harmful situations
that were never sampled.
Only four `create-chance` decisions occur, and only **three of 40 paired
periods** change their applied-action digest. The entire goal-difference gain
comes from one Sabres seed, `20262703`, which changes from 0-1 to 2-1.
Another changed Sabres period remains 0-0; the changed Ducks period moves from
0-0 to 1-1. A changed decision label is not necessarily changed input.

| Combined measure | Default | Candidate |
| --- | --- | --- |
| Scoreless periods | 15 | 13 |
| Native one-timer attempts / goals | 102 / 38 | 106 / 40 |
| Recorded shots / opponent shots | 169 / 129 | 180 / 128 |
| Ordinary turnovers / zone turnovers | 274 / 155 | 274 / 153 |
| Controlled-player goalie-contact impulses | 126 | 138 |

More shots and one-timers do not establish better decisions. Contact impulses
also increase; the reports retain phase-tagged context rather than attributing
every contact to an offensive cut. After the initial input change, the whole
period diverges, so these totals do not isolate goals caused by a particular
setup. This is a development comparison, not held-out confirmation, full-game
win-rate evidence, or a manual-goalie-opponent result. The model's missed native
response signals and the very low tactic exposure remain reasons to keep the
feature experimental.

```bash
nhl94 benchmark-cpu --agent classic-v1 \
  --matchups sabres-ducks-manual ducks-sabres-manual \
  --trials 20 --seed 20262701 --seconds 300 --frame-skip 4 \
  --action-type FILTERED --goalie-policy off --workers 4 \
  --output docs/benchmarks/classic-v1-chance-creation-baseline.json
# Repeat with --chance-creation and a distinct candidate output.
```

#### Native passing-window investigation

The [window replay report](benchmarks/classic-v1-chance-creation-windows.json)
replays all four selected benchmark decisions and eight fresh, candidate-driven
periods: both sides for seeds `20262801..20262804`. The four archived periods
reproduce their **complete production frame counts and applied-action hashes**,
not just their capture states. Every branch restores full emulator/RNG state
and checks identical starting RAM; histories include the actual native
observation interrupts, clocks, button edges and cached input.

The natural policy replaces **all four setups after four frames**. None follows
the held route for the entire 18-frame forecast. Endpoint probes therefore test
the forecast **conditional on holding its evaluated route**; they do not pretend
that the natural policy executed it. Natural continuations run until a native
attack ending or an explicit 240-frame timeout.

| Selected fixture | Predicted receiver/purpose | Held-route endpoint | Natural continuation |
| --- | --- | --- | --- |
| Home `20262703`, frame 461 | 2 / position | Fails: live selector and native pass choose 1 | Pass targets 2 at +15; no requested reception or shot. Friendly 3 recovers at +158; timeout at +240. |
| Home `20262712`, frame 6157 | 1 / one-timer | Native requested one-timer works, **also on the ordinary route** | Later pass targets 1 at +31; intercepted, ordinary turnover at +169. No shot. |
| Home `20262718`, frame 8553 | 1 / position | Requested reception works; ordinary-route diagnostic also works | Receiver 1 makes a native one-timer at +23, saved. **Identical inputs and outcome with chance creation disabled.** |
| Away `20262710`, frame 3505 | 8 / position | Fails: live selector and native pass choose 7 | Actual pass targets 8 at +18; intercepted, ordinary turnover at +108. No shot. |

No selected natural continuation scores within its measured horizon. In the
first case, disabling chance creation from the **same candidate history**
instead produces a later slot-4 shot, saved with an opposing rebound recovery.
That is a local comparison, not a replay of the whole baseline period or proof
that the original two-goal period difference has been causally explained.
The ordinary-route control in `20262718` is itself uncertified and remains
a diagnostic, not an admitted policy alternative.

Across the twelve full prefixes, **1,642 candidate evaluations / 7,386 route
rows** explain rare selection:

| First blocking requirement / control | Route rows |
| --- | ---: |
| Uncertified carry | 7,045 |
| No supported response | 66 |
| No positive passing window | 197 |
| Insufficient gain over baseline plus margin | 19 |
| Above margin | 6 |
| Ordinary-route comparison control | 53 |

Thus **95.4%** of evaluated routes fail carry admission. Recomputed rejection
details are `uncertified` (5,348), projected body contact (1,057), unbounded route
(603) and projected goalie contact (37). The six above-margin rows are
alternatives, not six distinct selected decisions. The fresh periods select
no `create-chance` action; all eight lack the above-margin sampling stratum.
Higher-priority actions and absent evaluations are counted separately in
`outer_gates`, not folded into route rejections.

Fresh sampling captures **20 first-in-stratum decisions**, at most four per
period, without looking at probe outcomes. A native usable window means a fresh
pass selects the requested receiver and either an ordinary reception settles
four frames in `|x| <= 71`, attack depth `(150,245)`, or a requested native
one-timer is accepted with useful geometry, a recorded shot or goal. This
bounded executability definition is **not** calibrated shooting quality and
does not supersede the live planner's conservative geometry/contact gates.

| Certified-carry receiver/purpose probes | Correct opening | False opening | Missed opening | Correct negative | Withheld/unavailable |
| --- | ---: | ---: | ---: | ---: | ---: |
| Four archived selected states | 3 | 5 | 14 | 58 | 0 |
| Twenty fresh stratified states | 0 | 2 | 21 | 145 | 112 |

The fresh certified-route endpoints contain **23 available native windows**:
21 missed by issued forecasts and two in withheld scenarios. Of the 21 misses,
six pass the live endpoint's pass-evaluation gates; the others remain rejected
live for one-timer geometry (seven), recipient mismatch (four) or unreachable
reception (four). The discrepancy therefore cannot be assigned solely to
teammate-response forecasting. Five of the seven certified-route false
openings across both cohorts have a live recipient mismatch confirmed by the
ROM. In the other two, the live selector predicts 2 but the native pass selects
1, retaining a separate recipient/release-timing counterexample.

Rejected-carry diagnostics also expose **45 available native windows** in the
fresh samples, including 35 missed by issued extrapolated forecasts. One
rejected route loses its carrier before the endpoint and receives no confusion
label. Executable passes after forced rejected movement do **not** invalidate
the carry certificate or justify weakening it.

These are overlapping receiver/purpose probes on stratified states, not
independent samples or a global match-play error rate. Abstentions, unavailable
retries/control, lost carriers and missing strata remain visible. The evidence
does not establish consistent **new** chances from the selected movement.
Chance creation remains off by default; no tactical weights, live safety
gates or finishing priorities were changed for this investigation.

```bash
python -m nhl94_ai.evaluation.chance_creation_replay \
  --fresh-seed 20262801 --fresh-seeds 4 --samples-per-period 4 \
  --horizon 240 --pass-horizon 120 --workers 4 \
  --output docs/benchmarks/classic-v1-chance-creation-windows.json
```

### Experimental carry and receiving continuations

These changes now require `--offense-lookahead`, `--uncertain-carry` or
`--chance-creation`; they are no longer active in ordinary `classic-v1` after
the default restoration above. Neither `--cross-crease` nor `--deke` is required.
Close ordinary shots and available one-timers retain their finishing
priority. No model assets, input layouts, action IDs or CLI flags change.

`agents/carry.py` projects the native grounded update while holding the same
ordinary steering buttons as the live controller at its configured decision
interval. It retains momentum and fractional facing rather than assuming an
instant turn. Agility, stored weight, speed and energy affect the trajectory,
not an arbitrary permission to ignore opponents. The carrier must stay in the
rink and swept body separation must exceed the existing **16-unit** physical
buffer plus uncertainty. On-ice fallen actors remain physical obstacles.

Defender reach is modeled separately for body contact and body/stick access to
the exposed puck. A control-reach envelope covers arbitrary grounded steering
and braking around native friction/inertia trajectories. Initial facing bursts
and arbitrary later bursts are included, using the verified **24-tick** burst
animation-lock lower bound; they are possible inputs, not predictions of CPU
choices. Full-energy override feedback is decoded in `GameState`; unknown
feedback uses the larger burst allowance. Currently fallen/locked players are
not assumed unavailable for the entire horizon.

The envelope supplies a **lower bound** on contact time. Failure of two pursuit
rollouts is no longer treated as proof that a defender cannot arrive sooner.
Pressure is checked every frame, not just at four sample times. Missing physics
feedback explicitly uses optimistic reach allowances. The existing three-frame
safety margin remains. Ordinary goalie clearance uses the same native trajectory
and held-input cadence in planning and live steering, preserving the **32-unit**
guard; legacy cuts and experimental finishers retain their older model. This
does not estimate the chance of surviving a check or winning a stick challenge.
These bounds cover the modeled skating/control choices, not arbitrary new ROM
collision impulses, attribute changes or exact future goalie/CPU decisions.

**Opt-in `--uncertain-carry` separates certification from action selection.**
It is **off by default**: the full experiment regressed for the Ducks. Within
`--offense-lookahead`, omitting this flag retains certificate-only selection.
The restored default instead uses the immediate possession policy. A geometrically clear
route that lacks a reachability certificate can remain `carry_viable`, while
`carry_safe` stays false. Native directional, coasting and initial-facing-burst
scenarios provide a separate modeled-contact assessment; unsuccessful samples
are never a proof of safety. Body/goalie contact, wall/net escape, missing risk
feedback and modeled interception at or before the next replan remain ineligible.
Later sampled contact no longer becomes viable merely because a replan occurs
first. The held-input prefix must lead to a checked native escape, or an eligible
shot must release before contact. Escape candidates include coasting, braking
and 12/24-unit adjustments, with swept body, goalie and sampled defender checks.
Native position integration precedes new braking/acceleration: a frame-four
decision cannot instantly cancel momentum that creates frame-five contact.
These are sampled-model feasibility checks, not new safety certificates.

For a verified exit, opportunity and progress are credited at the actual replan
state rather than the unsafe 18-frame endpoint. Other bodies advance by that
same elapsed time. Exit mode, frame, target and any release frame remain
explicit diagnostics. Any sampled contact during the coasting shot windup
withholds finishing credit.

For other uncertified routes, let `H = forecast_frames + 3` and `D` be the
decision interval. The risk index is at least `D/H`; a sampled contact at frame
`F` raises it to `max(D/H, (H-F)/H)`, capped at one. This is an explicit
heuristic temporal-exposure index, **not a loss probability**. It discounts
opportunity by `1-index` and independently subtracts `index * 0.3 * H`.
The cost reuses the existing frame-delay coefficient; zero shooting/positional
value therefore still pays risk instead of letting forward progress break
all zero-value ties. Samples are cached
per decision using the defender's actual position, momentum, fractional facing,
ratings and energy. The native carrier trajectory and certificate are unchanged.
Fallback comparisons prioritize body/goalie separation before hypothetical
defender timing when every alternative is ineligible.
The escape candidates now include **12- and 24-unit** lateral corrections,
with/without forward movement, before retaining the larger 48-unit lateral,
braking and retreat options. A close defender is not itself a rejection.

For each physically safe ordinary pass, the receiver is advanced to the
modeled **first body/stick contact**, not the later closest-approach point.
The receiver's body is not teleported onto the puck. A bounded **18-frame**
continuation compares the normal slot target, straight movement, two small
inside/outside adjustments and braking. The current carrier is evaluated with
the same short-carry helper. Other bodies are projected consistently, and
defender reach starts from the live pre-pass state: flight time is available
reaction time, not a free pause for the opponents.

The pass keeps its original progression/bypass/reception terms and adds
`0.6 * max(0, continuation_value - immediate_shot_value)`. Continuation value
is the greater of **0.6 times positional value** and executable finishing value.
With `--uncertain-carry`, that opportunity is additionally discounted by its
separate carry-risk index and pays the independent risk cost above; finishing
value also reflects any uncertified windup/release exposure.
This separates a useful skating destination from a shot the controller will
actually initiate. Finishing credit requires the shared live shot conditions,
body/puck safety through the native coasting windup and release, and a release
in front of the net. With one-timers enabled, speculative early shots get no
finishing credit because a one-timer or its setup can take live priority.
Positional value pays **0.3 points per skating frame** for delay; finishing
value also pays for windup/release frames. These weights,
like the existing shot heuristic, are not calibrated goal probabilities.
Positional eligibility uses the stronger immediate/continuation opportunity
and compares it with retaining possession **before** ranking candidates.
Advancement also compares progress with the retained carry. Hypothetical cuts
retain their immediate pass/shot evaluation rather than recursively adding
another receiving carry to the search.

This is planning, not a committed pass/carry macro. An actual reception,
including a different friendly receiver, interrupts cached input on the next
emulator frame and replans from observed possession. Future CPU decisions,
contact timing, sprite hotspots and puck lag remain uncertain; a projected
shooting opportunity is not a guaranteed shot or goal.
When the flight estimate exists but modeled first contact is ambiguous, the
continuation receives no credit and reports `no-modeled-reception-contact`.
The original immediate-pass evaluation remains available; debug shows
unmeasured carry value as `--`, not a fabricated zero.

The debug world-rink labels identify teammate **slots**, not jersey numbers:
`P` is the composite ordinary-pass score; `Pos` and `carry` are positional
heuristics; `Finish` is executable continuation-shot value; `OT` is the
one-timer shot score. A positive `carry` with zero `Finish` explicitly means
better positioning, not a shot within the forecast. `*` marks the chosen option, amber
marks its receiver, and rejected passes show **`--` plus the reason**, never a
fake zero. Safe candidates that fail the worthwhile-action filter are marked
as insufficient tactical gain. The keep-puck opportunity value is displayed
separately so it is not mistaken for the composite pass score. Labels avoid
existing text/each other where space permits and share the away-slot-aware
debug/live overlay.
Uncertified continuations explicitly show that status and their risk index;
neither a positive score nor `carry_viable` is displayed as certified safety.

**Debug inspection controls:** ordinary Classic playback now starts with
teammate scores enabled and the extra lane, velocity, orientation, distance
and planner overlays disabled. Keys **1-7** toggle the existing layers,
**8** toggles AI planner geometry/details, and **9** toggles teammate scores.
Shortcut help is printed to the console at startup; overlay/control changes
and pause/resume status are also reported there, not permanently on the canvas.

The debug window has a **1920x1080 logical layout** with the action inspector
on the left and an enlarged **1040x780** game image beside it, preserving the
existing 4:3 display aspect. A **164x300** mini rink sits below the game, with
compact team statistics and evaluation context alongside it. Mini-rink
overlays use smaller markers and compact score labels; detailed reasons and
score kinds stay in the inspector. The
window scales to fit the desktop and can be resized without changing game
coordinates. **F2** saves the full-resolution logical canvas.

The inspector lists carrying/setup, ordinary passes and one-timers to each live
skater position/slot, finishing, defense and goalie actions. **Green** identifies
the current tactical plan and execution; the lifecycle phase and actual applied
input are shown separately. A shot/pass request is not proof of ROM acceptance.
Mouse wheel over the panel or **PgUp/PgDn** scrolls the list; hovering a row
exposes its full reason or additional score kinds and probability scope.
Current rejections and disabled/unavailable actions remain visible.

The **Score / Kind** columns use existing evidence only: pass, keep-puck,
position, carry, finish or window heuristics are not interchangeable units.
Unknown scores are `--`. **P eff/raw** reserves separate effective/raw policy
probabilities for future neural diagnostic producers; Classic leaves it blank.
Policy P means action selection, not pass completion or goal probability, and
each producer must name its network/head or conditional scope. Old numerical
scores turn gray and retain their original evaluation age; they never restore
an old green selection. Agents without tactical diagnostics show a catalogue
without inventing probabilities from their button or coordinate outputs.

This first iteration is **read-only**. Goalie policy is shown as
Off/Selective/Always status, not a live toggle. Disabled experimental tactics
stay disabled; standalone slapshot and behind-net/wraparound choices are marked
unsupported, not silently added to Classic. No tactic priorities, safety gates,
native input timing, saved-model layouts or public action IDs change.

Press **Space** or **P** to pause/resume playback. This freezes both the
emulator and AI decision/state-machine clocks; it is not the ROM Start/Pause
button. Overlay toggles, **F2** screenshots and **Esc** remain responsive while
paused. Resuming restores normal playback pacing without a catch-up burst.
Enter retains its existing in-game Start button mapping.

The last genuine teammate evaluation stays visible through defense, shot/pass
waits and selective goalie control. Its original carrier and age in emulator
frames are shown beneath the game image. **Gray values / LAST evaluation**
mean historical reasoning, not newly computed advice at the players' current
positions. A fresh evaluation replaces the snapshot, including fresh rejections
shown as `--`; game resets clear it. This avoids both disappearing useful scores
and pretending that an old scoring opportunity is still live.
`tests.unit.test_debug_playback` covers this lifecycle and pause ordering.
With locally installed ROMs, run `python -m tests.integration.debug_playback`
under a display (or SDL's dummy drivers): it exercises the seeded
Sabres/Ducks away replay with selective goalie control, verifies unchanged RAM,
policy clocks and score data while paused, and then resumes.

**Playback performance (seed 12000, away/selective goalie):** the lag on Ducks
possession was reproduced in a bounded 1,800-native-frame prefix using the
command's save, side, policy and four-frame offensive cadence. Heavy fallback
carry scans repeatedly evaluated the same conservative defender envelopes;
the display also rerendered unchanged inspector text and allocated scaled
surfaces each frame. The optimization batches identical contact checks, shares
physics-keyed projections within a decision, caches rendered text and reuses
scaling surfaces. It does not change tactical gates, priorities or input cadence.

The [unpaced report](benchmarks/classic-v1-debug-performance-unpaced.json)
records **51.8 -> 119.3 native FPS** of measured processing capacity. During
the 434 AI-skater-possession frames, mean prediction cost falls
**24.4 -> 4.7 ms** and its 95th percentile **122.5 -> 20.7 ms**.
Mean rendering cost falls **9.4 -> 3.9 ms** across the whole prefix.
The [production-paced report](benchmarks/classic-v1-debug-performance-paced.json)
records **46.3 -> 60.0 FPS** at the requested 1x cap. Both reports embed their
original baseline summary and verify every native input, ownership, decision
label and controller clock plus initial/final RAM against the original prefix.

These measurements use **SDL's dummy video driver**: they exercise actual
rendering/scaling but cannot establish desktop-compositor/vsync/GPU performance
on every machine. Some individual frames still exceed the 16.67 ms budget;
60 FPS aggregate throughput does not mean perfectly uniform presentation.
The seeded prefix is not a global worst-case bound or a strength comparison.
Reproduce bounded timing with the locally installed ROM and configured Python:

```bash
python -m nhl94_ai.evaluation.playback_profile --frames 1800 --paced \
  --output /path/to/session/playback-timing.json
# Omit --paced for processing capacity; --profile writes instrumented cProfile
# data, whose timing must not be compared directly with unprofiled wall time.
```

`tests.unit.test_carry_continuations` covers skating-sensitive receiving value,
first-contact body/momentum/facing preservation, safe-pass gating, reactions
during flight, body-versus-puck reach, keeping possession, and actual-receiver
handoff. `python -m tests.integration.classic_carry` checks twelve native carry
traces at both ends, low/high skating profiles and 1/4/10-frame cadence with
zero position/velocity/facing error. Its wide-wing fixture selects a pass for
the receiver's carry rather than an immediate shot, observes native reception
in both `FILTERED` and `HOCKEY_INTENT_DPAD`, verifies immediate replanning,
then follows the actual inward carry to a fresh receiver-attributed ordinary
shot. The original nearby-idle-defender fixture recorded reception at frame 5
and a shot at frame 66, but did not validate active pressure. The corrected
positive fixture has a defender moving away initially: reception occurs at
frame 5 and its ordinary shot at frame 138. Its **47.9 positional value** gets
**28.74 continuation credit and zero finishing credit**, rather than pretending
the 18-frame endpoint is shot-ready.

The same receiving start is also checked with real CPU pursuit, retaining
possession through the 18-frame continuation in both action formats.
Sixteen additional native pressure starts cover head-on, flank, trailing and
remote defenders, both attacking ends and low/high skating profiles. Six routes
are certified in that set, with no observed false-safe route; two other starts
lose possession. CPU response is verified from movement, turning or a checking
animation, rather than requiring every valid defensive response to translate
the player. Run `python -m tests.integration.classic_carry --output <report.json>`
to persist the timings, pressure outcomes and source fingerprints.
These local contracts do not establish stronger full-team match play.

#### Predictor repair ablation (2026-10-06)

The [paired report](benchmarks/classic-v1-predictor-repair-comparison.json)
preserves a frozen pre-repair baseline and cumulative reach, goalie and
finishing repairs. Each stage runs four identical development seeds per AI
side in the usual Sabres/Ducks save: eight 300-clock-second first periods,
four-frame offense, `FILTERED`, goalie policy off, dekes/cross-crease off.
Baseline action digests match the corresponding archived default-offense
trials. Initial-state/lineup hashes, completed periods, inactive-player counts,
runtime source fingerprints and one-timer lifecycle accounting are checked.

| Cumulative stage | Agent goals for-against | W/D/L |
| --- | --- | --- |
| Frozen pre-repair default | 6-2 | 4/3/1 |
| Conservative defender reach | 7-3 | 3/5/0 |
| Plus native ordinary goalie clearance | 5-2 | 3/3/2 |
| Plus positional/executable finishing separation | 8-1 | 4/3/1 |

The final paired goal-difference change is **+3**, with a seed-clustered
95% interval of **[-2, +7]**. Four reused seeds and cumulative, order-dependent
changes are a diagnostic comparison, not evidence of a general strength gain
or an explanation of the earlier full-benchmark regression. The original
40-period reports below remain unchanged.

The comparison links three sequential runtime patches, each applicable to its
recorded predecessor runtime. Reapplying them to the frozen baseline is checked
against all final benchmark source hashes; repository HEAD alone is not that
dirty-tree baseline. Native evidence is saved in the
[pressure/receiving report](benchmarks/classic-v1-predictor-repair-native.json).

#### Full held-out predictor-repair benchmark (2026-10-06)

The full follow-up compares the frozen
[pre-repair policy](benchmarks/classic-v1-predictor-repair-heldout-baseline.json)
with the
[corrected policy](benchmarks/classic-v1-predictor-repair-heldout-final.json).
Both run the usual Ducks/Sabres protocol on **20 fresh seeds per AI side**:
`20262004..20262023`, 300-clock-second first periods, four-frame offense,
`FILTERED`, goalie policy off, dekes/cross-crease off and four workers.
These ROM seeds are disjoint from the development set. All **80 periods**
complete.

The [paired comparison](benchmarks/classic-v1-predictor-repair-heldout-comparison.json)
checks identical initial saves, physical teams and starting lineups for every
seed/side pair, unchanged measured policy fingerprints, zero inactive-skater
frames and balanced one-timer lifecycle accounting.

| AI side, 20 periods each | Pre-repair GF-GA | Corrected GF-GA | Pre-repair W/D/L | Corrected W/D/L |
| --- | --- | --- | --- | --- |
| Sabres home | 25-3 | 28-3 | 16/4/0 | 12/7/1 |
| Ducks away | 27-11 | 26-11 | 11/6/3 | 11/5/4 |
| Combined, 40 periods per policy | 52-14 | 54-14 | 27/10/3 | 23/12/5 |

The paired goal-difference change is **+2**, with a seed-clustered 95% interval
of **[-20, +26]** (10,000 bootstrap draws, seed 94; both AI sides remain in
the same ROM-seed cluster). Seventeen seed/side pairs improve, seventeen
worsen and six are unchanged. Wins fall by four; losses increase by two.
The extra goals do not establish a stronger policy.

| Observed offensive context, combined | Pre-repair | Corrected |
| --- | --- | --- |
| Controlled zone entries | 315 | 295 |
| Turnovers excluding recorded-shot follow-through | 298 | 302 |
| Offensive-zone turnovers | 189 | 175 |
| Possession losses, including shot follow-through | 468 | 409 |
| Offensive-zone possession losses | 359 | 280 |
| Controlled-skater goalie-contact impulse events | 152 | 196 |

The loss metrics overlap and are not additive. Impulse events are the existing
read-only contact telemetry, not proof that a particular goal came from a
collision. Fewer zone losses coexist with fewer controlled entries and more
goalie contacts. The full usual benchmark therefore **does not demonstrate
an overall gameplay improvement**, despite the corrected local prediction and
scoring contracts. This result is for fixed saved rosters/ratings and first
periods with goalie assistance off, not full-game or selective-goalie win rates.
No policy tuning or gameplay edits were made during this measurement.

#### Uncertified-carry experiment (2026-10-06)

`--uncertain-carry` is an **off-by-default experimental option**, not a
replacement for the corrected default policy. Add it to an existing Classic
play/evaluation command to inspect the experiment. It does not enable dekes,
cross-crease or manual goalies, change neural fields, or alter live shot triggers.

The [full comparison](benchmarks/classic-v1-uncertain-carry-comparison.json)
measures the original prototype, before the risk-cost/exit fixes below. It
contains 20 matched seeds per AI side (`20262004..20262023`), 300-clock-second
first periods, four-frame offense, `FILTERED`, goalie assistance off and both
optional finishers off. All **80 periods** complete. These previously measured
seeds are a matched development comparison, not a new held-out evaluation.

| AI side | Corrected default GF-GA | Prototype GF-GA | Default W/D/L | Prototype W/D/L |
| --- | --- | --- | --- | --- |
| Sabres home | 28-3 | 35-5 | 12/7/1 | 15/5/0 |
| Ducks away | 26-11 | 13-15 | 11/5/4 | 8/6/6 |
| Combined, 40 periods per policy | 54-14 | 48-20 | 23/12/5 | 23/11/6 |

Carry-opportunity decisions rise **8 to 106** and recorded shots **175 to 196**.
Fallback selections with no viable alternative fall from **83.6% to 51.4%**;
another 43.3% of prototype fallback selections are explicitly uncertified but
viable. This does **not** increase certification: 94.8% of prototype fallback
selections still lack a certificate. Ordinary turnovers rise **302 to 317**,
offensive-zone turnovers **175 to 206**, and zone possession losses **280 to
336**. The paired goal-difference delta is **-12**, with a seed-clustered 95%
interval of **[-38, +12]**. Restored attacking choices did not produce an overall
improvement, and the Ducks regress sharply; the user selected opt-in deployment.

Native pressure fixtures restore six of the ten uncertified routes as viable;
all six retain possession over their checked horizon, and none is relabeled
safe. The wing sequence receives at frame 5, carries inward, then makes a
verified return pass before the original carrier records a shot at frame 107
in both input formats. An active CPU pursuer also preserves the bounded
18-frame receiving continuation. These observations do not calibrate a general
loss probability.

**Contact attribution changes the earlier diagnosis.** Read-only replay of the
historical policies preserves every action digest:

| Contact context | Pre-repair policy | Corrected default | Uncertified prototype |
| --- | --- | --- | --- |
| Total impulse events | 152 | 196 | 160 |
| Defensive phase | 119 | 168 | 131 |
| Offensive phase | 33 | 28 | 29 |
| Controlled skater carrying before the event | 19 | 20 | 19 |

The earlier **+44** contacts comprise **+49 defensive-phase and -5
offensive-phase** events; only one additional event has the controlled skater
carrying beforehand. Aggregate contacts therefore did not demonstrate that
offensive fallback ranking caused the increase. Phase/input histories locate
the events but do not establish a causal mechanism. No defensive goalie
avoidance behavior was modified for this experiment.

The full reports record the measured prototype before its public opt-in gate
was added; those hashes are not rewritten. Native 60-clock-second parity runs
on both sides verified that the gate-only **default** reproduced the prior
policy and its opt-in mode reproduced the measured prototype. Later experimental
changes are measured separately below. Contact tracing
remains active in CPU benchmarks in either mode and does not change the
historical scalar metric. See the
[native report](benchmarks/classic-v1-uncertain-carry-native.json) and the
[historical contact replay](benchmarks/classic-v1-uncertain-carry-original-contacts.json).

#### Explicit risk costs and achievable exits (2026-10-07)

Two reproduced defects are fixed without enabling the experiment by default.
At zero opportunity value, a frame-five-contact route with risk `16/21`
previously beat a certified alternative on progress alone. The independent risk
cost now preserves the certified choice. A route threatened just after the
four-frame replan must also retain a checked native escape or shot release;
the prefix/suffix preserves momentum, precise position and input cadence.
The unit regressions cover frame-five integration latency, escape continuity
and advancing other bodies by the actual four-frame replan time.

**Superseded replay evidence:** subsequent cadence verification found that the
standalone replay called `ClassicAIV1Model.predict_game_state` directly.
Assigning its `frame_skip` attribute did not install `ScriptedAgent`'s dispatch,
so it made one decision per emulator frame rather than using production
`predict_frame(..., frame_skip=4)`. Its controller clocks also advanced
incorrectly. The following artifact/table are retained as historical diagnostics,
**not evidence about the usual four-frame policy**. The full CPU benchmarks
use the correct wrapper and are unaffected.

The [historical same-state replay](benchmarks/classic-v1-threat-exits-ducks-replay.json)
searched six usual seeds, `20262004..20262009`, under its default-driven prefix:
`SabresVsMightyDucks.ManualGoalie.Start`, away control, full teams, `FILTERED`,
and a 300-clock-second period. Each search was bounded at
6,000 emulator frames and requires a free controlled puck carrier at attacking
depth at least 140, with no pending pass, one-timer or shot follow-through.
Four first default/original-experiment input divergences qualify; seeds
20262006 and 20262007 have none within the search.

At each divergence, branches share the full emulator/RNG snapshot and verified
initial RAM hash. Reference and revised controllers inherit the actual default
temporal/action history, not fresh episode state. Each branch runs 180 frames
against the ordinary responding CPU, comparing default, original/revised
experiments and forced carry/escape/pass alternatives.

| Seed | Divergence frame | Default first opponent possession | Original experiment | Revised experiment | Notable observation |
| --- | --- | --- | --- | --- | --- |
| 20262004 | 2233 | 129 | None within 180 | None within 180 | Forced carry/pass lose at 76/48; escape has no opponent possession |
| 20262005 | 4229 | None within 180 | 110 | 144 | Default records one native one-timer attempt; neither experiment does |
| 20262008 | 1110 | 66 | 67 | 67 | Forced carry has no opponent possession; escape/pass lose at 72/147 |
| 20262009 | 1523 | 72 | 108 | None within 180 | Revised has six modeled one-timer-option frames versus default's two |

"None" means no observed opponent ownership in the branch horizon, not
continuous controlled possession. All cases start with **zero modeled
one-timer options**, and all branches record **zero goals and zero shots**.
No modeled-safe ordinary pass exists in these four captured states: the pass
branches are explicitly forced despite rejected evaluation. Fresh native pass
counters/targets and actual ownership after loose-puck flight distinguish
planned recipients from completed receptions.

The earlier interpretation of 20262005 as a lost **production** one-timer
pathway, and 20262009 as improved production retention, is withdrawn because
of that cadence mismatch. These outcomes neither explain the eleven-goal
decline nor measure the normal policy. The subsequent full comparison below
does use production cadence; the historical 48-20 remains the original
prototype's separate measurement.

The [native report](benchmarks/classic-v1-threat-exits-native.json) retains exact
grounded movement traces across both rink ends and 1/4/10-frame input cadence,
verified receiver/return-pass shots in both control formats, active CPU receiving
pressure and 16 pressure starts. A separate
[default-parity report](benchmarks/classic-v1-threat-exits-default-parity.json)
checks identical applied-action digests on both sides over 60-clock-second
periods against the frozen certificate-only reference. `--uncertain-carry`
remains **off by default**, with no shooting-weight or defensive-AI tuning.

To repeat the bounded possession investigation with a trusted local copy of
the original experimental runtime:

```bash
python -m nhl94_ai.evaluation.carry_replay \
  --reference-runtime /path/to/frozen-runtime \
  --seed 20262004 --seeds 6 --search-frames 6000 --horizon 180 --min-depth 140 \
  --output docs/benchmarks/classic-v1-threat-exits-ducks-replay.json
```

The runtime contains its `nhl94_ai/agents/{carry,offense,classic_v1}.py` files;
shared supporting modules come from the current package. Reported reference
and current source hashes identify exactly what was loaded. Raw ROM snapshots
remain in memory, not in committed model or ROM assets.

#### Full risk-cost/exit revision benchmark (2026-10-07)

The [paired comparison](benchmarks/classic-v1-threat-exits-full-comparison.json)
reruns [default](benchmarks/classic-v1-threat-exits-full-baseline.json) and the
[revised experiment](benchmarks/classic-v1-threat-exits-full-final.json) on the
usual 20 matched seeds per side (`20262004..20262023`): **80 completed
300-clock-second first periods**, `FILTERED`, four-frame offense, four workers,
goalie assistance off, dekes/cross-crease off. No policy changes or weight
tuning occur during measurement. These already measured development seeds
are not a new held-out evaluation.

| AI side, 20 periods per policy | Default GF-GA | Revised GF-GA | Default W/D/L | Revised W/D/L |
| --- | --- | --- | --- | --- |
| Sabres home | 28-3 | 34-5 | 12/7/1 | 17/2/1 |
| Ducks away | 26-11 | 23-8 | 11/5/4 | 11/6/3 |
| Combined | 54-14 | 57-13 | 23/12/5 | 28/8/4 |

The revision gains five wins on this matched set, all for the Sabres. The
Ducks retain the same wins and goal difference, with one fewer loss but three
fewer goals. The paired total goal-difference change is **+4**, with a
seed-clustered 95% interval **[-19, +28]** (10,000 bootstrap draws, seed 94;
both AI sides stay in their ROM-seed cluster). This is favorable measured
performance, concentrated in Buffalo, not an established gain across new
seeds/rosters or full games.

| Combined offensive context | Default | Revised |
| --- | --- | --- |
| Recorded shots | 175 | 206 |
| Opponent shots | 109 | 128 |
| Native one-timer attempts | 111 | 137 |
| One-timer goals | 43 | 48 |
| Carry-opportunity decisions | 8 | 84 |
| Turnovers excluding recorded-shot follow-through | 302 | 274 |
| Offensive-zone turnovers | 175 | 180 |
| Offensive-zone possession losses including follow-through | 280 | 329 |

More attempts coexist with more opponent shots and slightly more zone
turnovers. The full-period turnover metric excludes **recorded-shot**
follow-through, not every unrecorded release; it differs from the more
specific native shot tracking in outcome-ended replays.

| Ducks context | Default | Original prototype | Revised |
| --- | --- | --- | --- |
| Recorded shots | 65 | 70 | 96 |
| Native one-timer attempts | 45 | 35 | 63 |
| One-timer goals | 22 | 11 | 19 |
| Offensive-zone turnovers | 77 | 109 | 86 |
| Opponent shots | 74 | 95 | 76 |

Relative to the original prototype, the Ducks recover eight of its eleven
lost one-timer goals and substantially reduce the zone-turnover increase.
They still score fewer one-timer goals than default despite more attempts:
19/63 versus 22/45. Those descriptive conversions do not identify which
passes, receptions or finishes account for the remaining deficit. The
original prototype's 48-20 and this revision's **57-13** are separate
measurements; neither report's historical source hashes are rewritten.

All 40 current default action digests match their historical reference,
extending the short parity check to complete periods on both sides. Every
paired starting save, physical team and lineup matches; source fingerprints
are unchanged and identical between the two runs, inactive-skater frames are
zero, and native one-timer lifecycles balance. The public deployment remains
**off by default**; this measurement does not enable `--uncertain-carry`.

#### Outcome-ended replay protocol

The corrected driver explicitly calls `predict_frame(..., frame_skip=4)` for
every native step, including search prefixes and all policy branches. First
forced choices advance the same decision/defense clocks and encoded button
edges; subsequent input follows the native held-button cadence. Sampling
observes actual decision-tick changes, including reception-triggered replans,
rather than assuming the pre-call input cache establishes a decision boundary.

The [prefix parity report](benchmarks/classic-v1-threat-exits-replay-prefix-parity.json)
verifies a complete away default period for seed 20262004: **8,448 frames**,
clock zero and the exact applied-action SHA256 of the full CPU benchmark.
The earlier incorrect-cadence pilot and four-case report are superseded.

The packaged fail-closed gate repeats that proof rather than trusting an
existing `passed` marker:

```bash
python -m nhl94_ai.evaluation.carry_replay_gate \
  --benchmark docs/benchmarks/classic-v1-threat-exits-full-baseline.json \
  --output /path/to/session/replay-gate.json
```

It checks a current-source, completed standard default reference, then native
clock/frame/action parity for one away seed and unchanged source hashes after
the run. It needs no frozen experimental runtime. A changed policy/helper makes
that historical reference stale; generate a new default reference instead of
editing old source hashes. Success output cannot overwrite the reference or
measured code. This is a caller/provenance gate, not a strength result.

`--outcome-ended --horizon 900` replaces the three-second observation cutoff
with a bounded attack outcome: goal, confirmed turnover, stoppage, resolved
shot or an explicit timeout. Shots are followed beyond release. Native
one-timer counters/shooter attribution and normal-shot animation, ownership
release and puck impulse establish shot evidence; C input or stale
`shot_player` alone does not. Recorded shots remain a separate counter.

Four consecutive frames of stable opponent skater ownership outside shot
follow-through confirm an ordinary turnover. Shot catches/recoveries are
reported separately. Native `ltplayer` identifies goalie touches and
blocks/deflections; loose rebounds remain pending, allowing a later goal.
A puck passing the net plane without a goal has a 24-frame settling allowance.
Unresolved shots at a stoppage/timeout remain labeled unresolved, not saves,
goals or successful attacks.

Possession accounting separates controlled and autonomous friendly skaters,
both goalies, opponents and loose pucks. Controlled/team offensive-zone time,
completed native pass receptions, newly opened one-timer windows, native
attempts, observed releases and recorded shots distinguish useful attack
creation from merely avoiding opponent ownership. Elapsed frames and
possession fractions accompany variable-duration branches.

Sampling criteria are declared before outcomes: `general` takes the first
free on-puck input divergence per seed, while `one-timer` and `safe-pass`
also admit opportunity-control states when the policies agree. Viable
one-timers would otherwise be excluded precisely because all three policies
prioritize them identically. A one-timer-context divergence also qualifies
within 32 frames of a viable option or with a current setup decision.
Each stratum takes its first qualifying state; overlaps share one snapshot,
and missing strata/search lengths are explicit. Forced alternatives remain
probes, not tuned production policies.

The [corrected native replay](benchmarks/classic-v1-threat-exits-extended-ducks-replay.json)
and [derived summary](benchmarks/classic-v1-threat-exits-extended-ducks-summary.json)
cover all 20 usual seeds, `20262004..20262023`, with a 12,000-frame search cap
and a 900-frame maximum branch. All three strata qualify for every seed:
**51 unique captures**, comprising **22 input divergences and 29 opportunity
controls**. Overlaps account for the difference from 60 stratum assignments.
Thirty-three captured states have a modeled-safe ordinary pass; the other
18 pass branches remain explicitly forced despite rejected evaluation.

| Policy, 51 same-state branches each | Goals | Recorded shots | Native one-timer attempts | Confirmed ordinary turnovers |
| --- | --- | --- | --- | --- |
| Default | 10 | 38 | 30 | 13 |
| Original prototype | 6 | 32 | 27 | 14 |
| Revised experiment | 6 | 32 | 24 | 14 |

The **20 established one-timer controls** produce identical outcomes in all
three policies, including **six goals, 17 recorded shots and 17 native
attempts**. The **20 general divergence captures**, however, yield default
**three goals / 15 shots / 10 one-timer attempts** versus revised **zero /
eight / five**. This locates a remaining problem in preparation before an
established opportunity, rather than demonstrating that the revision refuses
the same already-available one-timer.

For example, seed **20262014**, capture frame **506**, starts with no modeled
one-timer option at `(-120, -142)`. Default creates a one-timer release at
branch frame 86 and scores at **103**. The revision instead reaches a confirmed
ordinary turnover at **51**, without an attempt. Seed 20262009 has another
default goal after 173 frames where the revision creates no observed release.
These are actual scored counterfactual pathways, not merely preserved attempts;
they still do not allocate the full-period three-goal Ducks deficit.

All 153 production-policy branches reach a meaningful ending before timeout.
Only one forced-escape probe reaches its 900-frame cap. Goal, save/catch,
rebound, block/deflection and net-plane outcomes are recorded separately.
Normal release tracking requires explicit animation/ownership/impulse evidence;
an unobserved release is not proven absent. Variable branch durations and
diagnostic selection prevent treating these counts as unbiased match win
rates. The local Ducks captures favor default, whereas the full-period revision
improves the combined matched results through Buffalo; those observations
are complementary, not interchangeable.

To reproduce this corrected investigation:

```bash
python -m nhl94_ai.evaluation.carry_replay \
  --reference-runtime /path/to/frozen-runtime \
  --seed 20262004 --seeds 20 --search-frames 12000 --horizon 900 \
  --outcome-ended --strata general one-timer safe-pass --workers 2 \
  --output docs/benchmarks/classic-v1-threat-exits-extended-ducks-replay.json
```

#### Historical default-policy Ducks/Sabres measurement

The [original default-offense report](benchmarks/classic-v1-default-offense-lookahead.json) and
[paired comparison](benchmarks/classic-v1-default-offense-lookahead-comparison.json)
compare these default changes with the archived
[cross-crease-off baseline](benchmarks/classic-v1-cross-crease-ducks-sabres-off.json).
The [runtime-only patch](benchmarks/classic-v1-default-offense-lookahead-runtime.patch)
records the incremental policy change against that exact measured baseline,
not against repository HEAD, and is independently reversible from its recorded
post-change runtime. Display changes are recorded separately by the comparison
fingerprint; later debug-only updates do not rewrite these historical sources.
Every baseline runtime fingerprint matched the working implementation before
this change. The comparison verifies the recorded post-change fingerprints and every
matched save hash, physical team and starting lineup. All **40 new periods**
complete, with no inactive skaters and balanced one-timer lifecycle accounting.

Both versions use `SabresVsMightyDucks.ManualGoalie.Start`,
`20261004..20261023`, 20 first periods per AI side, 300 game-clock seconds,
`FILTERED`, four-frame offensive decisions and goalie assistance off.
Cross-crease and skating dekes are **off in both**. The only incidental setting
differences are report destination and worker count (two before, four after).
These are reused development seeds, not held-out confirmation.

| AI side | Before goals for-against | After goals for-against | Before W/D/L | After W/D/L |
| --- | --- | --- | --- | --- |
| Sabres, home | 37-7 | 35-8 | 16/3/1 | 14/4/2 |
| Ducks, away | 18-5 | 14-11 | 12/7/1 | 5/10/5 |
| Combined | 55-12 | 49-19 | 28/10/2 | 19/14/7 |

**The observed match-play result regresses; this is not a demonstrated strength
improvement.** Combined scoring falls by six, concessions increase by seven,
and goal difference falls by thirteen. A paired 10,000-draw ROM-seed-cluster
bootstrap (seed 94, keeping both AI sides together) gives aggregate-change
95% intervals of `[-28, +16]` for goals for, `[-4, +17]` for goals against,
and `[-39, +13]` for goal difference. These wide intervals do not establish a
universal strength verdict. The local receive/carry/shot demonstration and
more expressive diagnostics must not be substituted for improved match results.
This is a combined change, not a component-by-component ablation.

```bash
nhl94 benchmark-cpu --agent classic-v1 \
  --matchups sabres-ducks-manual ducks-sabres-manual \
  --goalie-policy off --trials 20 --seed 20261004 --seconds 300 \
  --frame-skip 4 --action-type FILTERED --workers 4 \
  --output docs/benchmarks/classic-v1-default-offense-lookahead.json
```

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

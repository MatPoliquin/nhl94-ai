# Agents refactoring and cleanup plan

Date: 2026-10-21

Status: planning only. Implementation and native baseline benchmarks have not
started. The review's unit run passed 1,293 tests with one expected failure;
that result is not a substitute for the baseline required below.

## Objective

Refactor `nhl94_ai/agents` to make configuration, tactical selection, action
execution, platform mechanics, and diagnostics easier to change independently.
Fix the demonstrated behavior issues in isolated changes, then preserve those
corrected behaviors through structural cleanup.

Scope includes the agent folder and the callers, tests, platform helpers, and
benchmark source manifests that must change with its interfaces. Existing
uncommitted work must be preserved and included in the recorded baseline.

## Mandatory benchmark gate: Ducks and Sabres, before and after

**Run and archive the baseline before changing implementation, tests, model
artifacts, configuration, or benchmark code for this work.** Writing this plan
does not satisfy that gate. If the ROM, save state, or environment is unavailable,
resolve that prerequisite before starting implementation.

Use the existing `nhl94 benchmark-cpu` runner. Both matchups use
`SabresVsMightyDucks.ManualGoalie.Start` in `NHL94-Genesis-v0`:

| Matchup | Agent controls | Built-in CPU controls |
| --- | --- | --- |
| `sabres-ducks-manual` | Sabres, home/controller 1 | Ducks |
| `ducks-sabres-manual` | Ducks, away/controller 2 | Sabres |

The matchup name identifies the agent's team first. Report each team separately;
an aggregate improvement must not conceal a regression on the other team.
These are first-period trials, not full-game win rates.

### Fixed comparison settings

| Setting | Required value |
| --- | --- |
| Agent | `classic-v1` |
| Seeds | 8000 through 8019, the same 20 seeds for each team |
| Period duration | 300 game-clock seconds |
| Action schema | `FILTERED` |
| Goalie policy | `selective` |
| Goalie positioning | `deep-csave` |
| Strict control interval | 0, native-frame observations/input reactions |
| Tactical decision intervals | Separate profiles at 1 and 4 frames |
| Workers | 1, kept fixed for comparable elapsed-time measurements |
| Optional experiments | Off in the core benchmark; add affected profiles separately |

One complete suite contains 80 periods: two teams × 20 seeds × two tactical
cadences. Run it before the first change, after each implementation stage, and
on the final candidate. A stage may contain smaller commits, but must pass its
gate before proceeding to the next stage. Reuse the preceding accepted report
as the next stage's reference only when its code, options, and assets are
unchanged. Keep the original baseline permanently.

The CPU runner does not apply playback's `classic_bot.json` defaults. Explicit
flags below pin the benchmark behavior; one-frame and four-frame tactics are
different profiles, not interchangeable measurements.

### Baseline preparation

- [ ] Record the commit, full working-tree diff, relevant untracked source files,
  Python/dependency versions, machine details, and exact commands.
- [ ] Archive a restorable copy of the baseline source, including existing dirty
  files. A commit ID alone does not identify this working tree.
- [ ] Record ROM and save-state hashes and hashes of any optional checkpoints.
  Supplement the runner's source manifest with all agent and platform sources.
- [ ] Verify the installed package resolves to this checkout and the imported ROM
  and named save state are available.
- [ ] Run the unit suite and preserve its output and expected-failure list.
- [ ] Run the complete baseline suite below before editing code.
- [ ] Repeat each profile on seeds 8000 and 8001 for both teams to establish
  deterministic action hashes and outcomes before relying on exact comparison.
- [ ] Confirm every period completed, the lineups match the intended teams, and
  the opposing team is CPU-controlled.

Run commands from the repository root. The Python path below is the environment
used for the review; substitute the project's environment on another machine.
Use a new output root if this plan is executed again.

```bash
AGENTS_REFACTOR_PY=/home/mat/vretro/bin/python
AGENTS_REFACTOR_RUN=tmp/agents-refactor/2026-10-21

run_agents_benchmarks() {
  local stage="$1"
  local destination="$AGENTS_REFACTOR_RUN/$stage"
  local cadence
  if [ -e "$destination" ]; then
    echo "Refusing to overwrite benchmark evidence: $destination" >&2
    return 1
  fi
  mkdir -p "$destination" || return 1
  for cadence in 1 4; do
    /usr/bin/time -p "$AGENTS_REFACTOR_PY" -m nhl94_ai benchmark-cpu \
      --agent classic-v1 \
      --matchups sabres-ducks-manual ducks-sabres-manual \
      --trials 20 --seed 8000 --seconds 300 \
      --frame-skip "$cadence" --classic-control-interval 0 \
      --action-type FILTERED \
      --goalie-policy selective --goalie-positioning deep-csave \
      --workers 1 \
      --output "$destination/native-${cadence}.json" \
      > "$destination/native-${cadence}.stdout.log" \
      2> "$destination/native-${cadence}.stderr-time.log" || return 1
  done
}

run_agents_benchmarks before
```

Run the same function after each stage with a new label, such as `after-stage-1`,
and finally `after-final`. For baseline reproducibility checks, use the same
benchmark arguments with `--trials 2`, keeping seed 8000, and separate output
files. Never overwrite the original reports or change source files during a run.

### Comparison and acceptance

Join individual results by matchup, seed, and profile. Verify identical protocol,
ROM/save state, lineup, dependency versions, and effective settings before
comparing results. Source hashes are expected to change during implementation;
benchmark semantics and assets must remain fixed.

For each team and cadence, report completed periods, wins/draws/losses, goals for
and against, goal differential, shots for and against, pass outcomes, one-timer
attempts/goals, available goalie/defense metrics, action hashes, and frame counts.
Record elapsed wall time separately; these reports do not establish per-decision
latency. Keep machine load comparable and investigate material slowdowns.

| Change type | Acceptance requirement |
| --- | --- |
| Structural refactor or cleanup | Identical action hashes, frames, scores, shots, and deterministic gameplay outcomes for every paired core trial and every affected optional profile. Explain any diagnostic-only representation changes. |
| Intentional behavior fix | A focused reproduction passes; document why the first action divergence is expected; compare all paired results and affected native replays. No unexplained regression is acceptable. |
| Incomplete or inconsistent run | Fail the gate. Preserve the evidence, resolve the cause, and rerun under a new output label. Never drop failed seeds or count partial scores. |

For behavior fixes, use zero reduction in per-team aggregate goal differential
and no increase in per-team aggregate goals conceded as the initial acceptance
target for each cadence. If either target fails, investigate the changed seeds
before proceeding. Any accepted tradeoff must be stated explicitly with evidence;
it cannot be hidden by pooling Ducks and Sabres. These finite deterministic
fixtures are regression gates, not proof of general playing strength.

Maintain this comparison table as reports become available:

| Stage | Cadence | Agent team | Completed | W/D/L | GF/GA | Goal-difference delta | Action-hash matches | Report | Gate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Before | 1 | Sabres | Pending | | | | | | Pending |
| Before | 1 | Ducks | Pending | | | | | | Pending |
| Before | 4 | Sabres | Pending | | | | | | Pending |
| Before | 4 | Ducks | Pending | | | | | | Pending |

Add four rows for every subsequent stage and the final candidate. Compare final
results to both the original baseline and the most recent accepted stage.

## Stage 1: fix demonstrated behavior issues in isolated changes

### Persistent learned-agent adapters

Source: [multi_model.py](../nhl94_ai/agents/multi_model.py),
[base.py](../nhl94_ai/agents/base.py).

- [ ] Construct one `LearnedAgent` adapter per loaded model and retain it between
  predictions. Reset recurrent state only at the appropriate episode boundary.
- [ ] Preserve independent state when switching between models and maintain
  existing scripted-agent, probability-display, and away-team behavior.
- [ ] Add a stateful fake-policy regression showing that the second call receives
  the first call's recurrent state and `episode_start=False`, and that reset
  starts a new episode.
- [ ] Verify both the compatibility entry points and the `Agent.act()` path.

The review reproduced recurrent-state loss through a new adapter on every call.
The core scripted benchmark should remain exactly unchanged by this fix.

### Retention features include observable goalies

Source: [retention_features.py](../nhl94_ai/agents/retention_features.py),
[defense.py](../nhl94_ai/agents/defense.py).

- [ ] Separate on-ice/observable-body predicates from controllable-skater
  eligibility. Select the predicate deliberately at each affected call site.
- [ ] Include goalies in retention features and verify nearest-actor ordering,
  presence flags, goalie flags, unavailable actors, and both attacking directions.
- [ ] Version changed feature semantics. Preserve an explicit legacy encoder or
  reject incompatible artifacts; do not silently apply new semantics to old weights.
- [ ] Retrain/revalidate affected retention artifacts before declaring the new
  feature path ready. Record artifact hashes and evaluation evidence.

The review reproduced a goalie one rink unit from the carrier being excluded by
`eligible()`. This fix needs a retention-enabled before/after profile at cadence
4 on both Ducks and Sabres, in addition to the core suite. Capture that optional
baseline before changing the encoder; a core run with retention disabled cannot
validate it. Keep old and newly trained artifacts separately identifiable.

### Coherent forecast coordinates

Source: [offense.py](../nhl94_ai/agents/offense.py),
[skating.py](../nhl94_ai/agents/skating.py).

- [ ] Repair the legacy `projected_state()` path so precise and integer/display
  coordinates describe the same projected position under the chosen model.
- [ ] Audit puck coordinates/velocity and related forecast builders for the same
  invariant, without changing their modeling assumptions incidentally.
- [ ] Add a regression that composes projection with a native movement step and
  verifies no jump toward the old position and no mutation of the source state.
- [ ] Exercise relevant feint, carry, pass-continuation, and finishing replays.

The coordinate inconsistency was reproduced directly. Its match impact has not
yet been established, so explain any observed gameplay divergence with replay
evidence rather than assuming a strength improvement.

Gate: keep these fixes independently reviewable. Run relevant regressions and
the required Ducks/Sabres suite after the stage; also benchmark affected optional
features before and after their individual behavior changes.

## Stage 2: normalize options and separate loading from episode state

Source: [classic_v1.py](../nhl94_ai/agents/classic_v1.py),
[registry.py](../nhl94_ai/agents/registry.py),
[classic_config.py](../nhl94_ai/agents/classic_config.py),
[base.py](../nhl94_ai/agents/base.py),
[config.py](../nhl94_ai/config.py).

- [ ] Introduce immutable normalized Classic options. Preserve distinctions
  between unspecified values, explicit overrides, and disabled features.
- [ ] Establish one compatibility-validation entry point used by CLI,
  benchmarks, and direct callers, with explicit platform restrictions.
- [ ] Extract offense and goalie construction into small factories.
- [ ] Separate immutable loaded artifacts from per-episode execution state.
  Reset clocks, commitments, recurrent state, and episode metrics without
  reloading neural checkpoints or rereading JSON resources.
- [ ] Read/hash/parse retention artifacts once before schema dispatch.
- [ ] Check that sharing immutable resources never shares mutable state between
  opponents, teammate controllers, or episodes.
- [ ] Preserve the ordinary scripted import path's avoidance of neural imports.

Gate: option/default/compatibility tests, reset/isolation tests, and exact core
benchmark parity. Add goalie-off and affected goalie/model profiles on both
teams before their construction/reset paths change.

## Stage 3: separate tactical proposals from action execution

Source: [classic_v1.py](../nhl94_ai/agents/classic_v1.py),
[lifecycle.py](../nhl94_ai/agents/lifecycle.py),
[scheduling.py](../nhl94_ai/agents/scheduling.py),
[learned_offense.py](../nhl94_ai/agents/learned_offense.py).

- [ ] Introduce typed proposals for carries, passes, one-timers, shots, and
  maneuvers, containing target, selected option, and decision evidence.
- [ ] Extract proposal selection from `_attack()` and route proposals through
  shared execution entry points that start the existing lifecycles.
- [ ] Reuse those entry points for learned offense commands while retaining
  teacher labeling, action masks, and command-consumption behavior.
- [ ] Keep observed release/reception distinct from an input request.
- [ ] Preserve exclusive input-owner priority, B/C press/release edges, handoff
  frames, retry history, and frame-versus-decision deadlines.
- [ ] Make cancellation/stoppage operations explicit instead of having callers
  clear another controller's internal fields individually.

Gate: lifecycle, reception, one-timer, shot-hold, control-rate, and learned-offense
regressions; native lifecycle replay comparisons; exact Ducks/Sabres core parity.
Capture and compare affected experiment profiles before changing their executors.

## Stage 4: consolidate forecasting and correct dependency direction

Source: `agents/motion.py`, `skating.py`, `passing.py`, `offense.py`,
`possession_value.py`, `responses.py`, teammate forecasts, and `game/platforms`.

- [ ] Extract shared actor predicates and forecast cloning/update primitives.
- [ ] Move console arithmetic and physical primitives beneath the agent layer.
  Platform backends must not depend on tactical controller modules.
- [ ] Keep tactical scoring and admission decisions in agents.
- [ ] Preserve separate conservative, sampled, coasting, and CPU-response models
  and their uncertainty labels while sharing state-update mechanics.
- [ ] Update benchmark/replay source-fingerprint manifests for every moved file
  and shared dependency, preserving benchmark semantics and report contracts.
- [ ] Keep compatibility re-exports where existing callers need them, then
  migrate internal imports to the owning module.

Gate: platform mechanics, projection invariants, no-source-mutation checks,
carry/pass replay comparisons, native Genesis and SNES checks, and exact core
benchmark parity. A source-path-only fingerprint change is expected; an action
change is not.

## Stage 5: consolidate offense workflows and decision evidence

Source: `offense.py`, `possession.py`, `possession_value.py`,
`possession_ablation.py`, `receiver_selection.py`, `decisions.py`, and `base.py`.

- [ ] Share the duplicated feedback checks, close-finish/breakaway handling,
  cut continuation, pass/feint sequence, and fallback candidate generation.
- [ ] Retain explicit hooks for admission, scoring, and continuation models.
  Preserve candidate ordering, tie-breaking, and incumbent fallback.
- [ ] Produce typed decision evidence with the selected proposal and serialize
  it to current UI/evaluation dictionaries at the boundary.
- [ ] Reduce decision-name parsing and introspection of controller internals.
- [ ] Define the supported `act`, frame execution, and compatibility inference
  contracts; migrate callers before removing wrappers or broad `__getattr__`.
- [ ] Keep diagnostics observational: building a snapshot must not evaluate new
  alternatives, change controller state, or inspect fresh state on held frames.

Gate: offense selection, ablation, receiver, inspector, overlay, and held-frame
tests; exact core parity; before/after parity for each affected optional policy.

## Stage 6: remove dead code and organize modules

- [ ] Remove verified unused legacy helpers/members in `multi_model.py`, including
  `test`, `target_xy`, `DistToPos()`, and `SelectRandomTarget()`, after checking
  supported external interfaces and compatibility requirements.
- [ ] Remove write-only teammate coordinator history fields and update tests that
  still assign them; preserve `SharedReceiving`'s actual history ownership.
- [ ] Remove unused imports and duplicate button-encoding helpers.
- [ ] Preserve intentional exports such as `passing.rom_direction` until their
  consumers have migrated.
- [ ] Preserve movement/action ID ordering and checkpoint contracts when sharing
  direction tables; different tables currently have different orders.
- [ ] Group stable modules by Classic tactics, goalie control, teammate control,
  and experiments only after their dependencies are clear. Keep public import
  shims as needed and update documentation/source manifests together.
- [ ] Run focused unused-code lint without blanket removal of compatibility code.

Gate: exact gameplay parity, package/import contracts, lint, wheel build, and
relevant teammate checks on both Ducks and Sabres.

## Additional validation matrix

Before touching a supported path, record its baseline as well as its post-change
result. Optional experiments should be tested individually under their existing
compatibility rules. Record exact flags and artifact hashes in the run manifest.

| Affected area | Additional required coverage |
| --- | --- |
| Goalie construction/lifecycle | Automatic goalies off, original manual positioning, and each affected learned/planner controller; both Genesis teams |
| Timing/configuration | Native tactics at 1/4 frames, supported strict control intervals, and playback config overrides |
| Retention | Retention-enabled cadence-4 profile for both teams, legacy/new schema checks, artifact revalidation |
| Learned adapters/offense | Stateful fake-policy tests, episode resets, model switching, masks, teacher labels, available checkpoint smoke checks |
| Shared mechanics | Genesis and SNES platform tests plus native smoke/replay checks; SNES Ducks and Sabres sides |
| Teammate coordination | Home Sabres and away Ducks, two and three bots, coordinated/independent strategies, pressure/support options when affected |
| Diagnostics/API | Decision inspector, held snapshots, overlays, recording consumers, action shapes, package/import compatibility |

Use existing native integration runners for the relevant change, such as
`tests.integration.classic_v1`, `tests.integration.cpu_benchmark`,
`tests.integration.classic_config`, `tests.integration.manual_goalie`, and the
domain-specific lifecycle/offense/teammate runners. Inspect each runner's
arguments and dependencies before invoking it. Existing frozen fixtures must not
be regenerated just to make a refactor pass.

General checks at each stage:

```bash
/home/mat/vretro/bin/python -m unittest discover -s tests -v
/home/mat/vretro/bin/python -m pylint nhl94_ai tests
```

Build the wheel after module moves and at final validation:

```bash
/home/mat/vretro/bin/python -m build --wheel
```

## Final completion criteria

- [ ] Baseline and final benchmark reports exist for both Ducks and Sabres at
  both required cadences, with all periods complete.
- [ ] Every intermediate stage has a recorded gate result; the final four-row
  team/cadence comparison includes links to raw reports and paired differences.
- [ ] Each behavior fix has focused regression evidence and explained gameplay
  changes; each structural stage preserves its accepted reference's gameplay.
- [ ] All affected optional, teammate, learned, and SNES paths have before/after
  evidence, with any unavailable required prerequisite resolved before completion.
- [ ] Existing test IDs/assertions remain intact unless a deliberate behavior
  change requires a documented adjustment. Expected failures are accounted for.
- [ ] Unit tests, required native checks, lint, and packaging checks pass.
- [ ] Model semantics, action IDs, reset behavior, public imports, and benchmark
  fingerprints remain compatible or have explicit migration/version handling.
- [ ] Reports and generated artifacts stay under the ignored run directory;
  source documentation contains the plan and final evidence references.
- [ ] Unrelated pre-existing working-tree changes are preserved.

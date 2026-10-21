# Agents refactoring and cleanup plan

Date: 2026-10-21

Status: all seven stages and final validation are complete at `112bfa3`.
The final Genesis suite completes all 80 periods with exact original gameplay;
all 176 optional periods match their accepted references. Teammate, lifecycle,
learned, packaging, and basic SNES checks pass. The final unit suite passes 1,348
tests with the original expected failure. Retention feature correction, retraining,
and further SNES validation remain deferred. The final acceptance and readability
reviews below link the complete evidence, including resolved failed gates.

## Objective

Make the agent code easier to read, trace, and change. The main opportunities are
to shorten `ClassicAIV1Model`, centralize option handling, remove duplicated
offense logic, and make ownership of execution state explicit. File moves alone
do not meet this objective, and a lower line count is not the acceptance test.

Scope includes the agent folder and callers, tests, platform helpers, and source
manifests affected by its interfaces. Structural work must preserve behavior;
the bounded correctness fixes below require separate commits and evidence.
Retention feature changes and retraining are deferred to a separate workstream.
Do not introduce a general strategy framework, plugin registry, or event bus.

The current starting commit is `ec71dfa`; record the actual revision and any
subsequent working-tree changes when taking the baseline. Preserve unrelated work.

### What must become simpler

- [x] Initialization reads normalized options and constructs components; option
  compatibility rules have one authoritative implementation. Intentional defaults
  that differ between playback, benchmarks, and direct callers remain explicit.
- [x] `_attack()` exposes tactical priority in a readable sequence, with bounded
  helpers for execution. Understanding a decision must not require following a
  chain of thin wrappers across many files.
- [x] Each commitment, cooldown, button edge, and reset operation has a named
  owner. Callers use explicit operations instead of editing another controller's
  private fields.
- [x] Shared offense steps have one implementation, with small, explicit policy
  differences. Avoid a single function with an expanding collection of switches.
- [x] Physical primitives sit below tactics in the dependency graph. Diagnostics
  use explicit evidence instead of reconstructing intent from private state.
- [x] Compare the largest functions, duplicated decision branches, dependency
  cycles, and cross-controller state writes before and after. Use these as review
  evidence, not arbitrary line-count or file-count targets.
- [x] Walk through one observation, tactical choice, button output, cancellation,
  and episode reset in the final code. A reviewer should be able to identify the
  decision order and state owners without reverse-engineering hidden side effects.

Some complexity is intrinsic to frame timing and hockey tactics. Tests and small
data types may add lines; the production decision paths should become shorter
and require fewer unrelated details to understand. No speed or playing-strength
improvement is assumed.

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
| Workers | 15 (all 16 available cores minus one), per the user's updated instruction; original baseline used 1 |
| Optional experiments | Off in the core benchmark; add affected profiles separately |

The user subsequently prioritized Genesis, limited SNES to a basic execution
smoke, and requested all available CPU cores minus one. These instructions
supersede the original single-worker and comprehensive SNES validation rules.
Use 15 workers for subsequent native matrices. Check parallel reproducibility
against the frozen original source; do not compare single-worker and parallel
elapsed times as performance evidence. Keep the gameplay acceptance rules intact.

One complete suite contains 80 periods: two teams × 20 seeds × two tactical
cadences. Keep the original baseline permanently. Use this schedule:

| Gate | Required runs |
| --- | --- |
| Before any implementation change | Full 80-period suite, then an 8-period repeat on seeds 8000–8001 to establish reproducibility; capture affected replay/optional baselines too |
| Small structural checkpoint | 8 periods: both teams × seeds 8000–8001 × cadences 1/4, plus affected focused tests/replays |
| Behavior fix or major execution/state transition change | Full relevant matrix: all 80 core periods when Classic execution can change, plus each affected optional profile on both teams |
| Final candidate | Full 80-period suite and affected optional profiles, compared with the original baseline and accepted behavior-fix references |

A stage may contain smaller, independently reversible commits. Do not continue
past a failed gate. A small checkpoint is a screen, not evidence of parity across
all 20 seeds. Reuse a report only if its source, configuration, assets, and
required coverage match the revision being evaluated; a completed final-stage
full suite can satisfy the final gate under that condition.

The CPU runner does not apply playback's `classic_bot.json` defaults. Explicit
flags below pin the benchmark behavior; one-frame and four-frame tactics are
different profiles, not interchangeable measurements.

### Baseline preparation

- [x] Record the commit, full working-tree diff, relevant untracked source files,
  Python/dependency versions, machine details, and exact commands.
- [x] Archive a restorable copy of the baseline source. If the tree is dirty,
  include those files; a commit ID alone then does not identify the baseline.
- [x] Record ROM and save-state hashes and hashes of any optional checkpoints.
  Supplement the runner's source manifest with all agent and platform sources.
- [x] Verify the installed package resolves to this checkout and the imported ROM
  and named save state are available.
- [x] Run the unit suite and preserve its output and expected-failure list.
- [x] Run the complete baseline suite below before editing code.
- [x] Repeat each profile on seeds 8000 and 8001 for both teams to establish
  deterministic action hashes and outcomes before relying on exact comparison.
- [x] Confirm every period completed, the lineups match the intended teams, and
  the opposing team is CPU-controlled.
- [x] Archive native lifecycle traces using the existing replay runner before
  changing execution/state ownership. Its fixture set complements the core CPU
  suite; it does not substitute for the required core seeds and settings.

Run commands from the repository root. The Python path below is the environment
used for the review; substitute the project's environment on another machine.
Use a new output root if this plan is executed again.

```bash
AGENTS_REFACTOR_PY=/home/mat/vretro/bin/python
AGENTS_REFACTOR_RUN=tmp/agents-refactor/2026-10-21

run_agents_benchmarks() {
  local stage="$1"
  local tier="${2:-full}"
  local trials
  case "$tier" in
    full) trials=20 ;;
    check) trials=2 ;;
    *) echo "Expected benchmark tier: full or check" >&2; return 1 ;;
  esac
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
      --trials "$trials" --seed 8000 --seconds 300 \
      --frame-skip "$cadence" --classic-control-interval 0 \
      --action-type FILTERED \
      --goalie-policy selective --goalie-positioning deep-csave \
      --workers 15 \
      --output "$destination/native-${cadence}.json" \
      > "$destination/native-${cadence}.stdout.log" \
      2> "$destination/native-${cadence}.stderr-time.log" || return 1
  done
}

run_agents_benchmarks before full
run_agents_benchmarks before-repeat check
```

Use a fresh label for each checkpoint, for example
`run_agents_benchmarks after-options check`, and finish with
`run_agents_benchmarks after-final full`. Never overwrite reports or change source
files during a run. Commands in this plan are instructions, not completed runs.

The existing lifecycle runner can capture its full default fixture set:

```bash
"$AGENTS_REFACTOR_PY" nhl94_ai/evaluation/lifecycle_replay.py \
  --root "$PWD" --workers 15 \
  --output "$AGENTS_REFACTOR_RUN/before-lifecycle"
```

Use the same case set for subsequent comparisons. The runner supports
`--compare REFERENCE/report.json CANDIDATE/report.json --output comparison.json`.
It compares decisions/targets as well as actions, so deliberate diagnostic schema
changes need the explicit handling below.

### Comparison tooling and trace evidence

The existing [fixture comparator](../nhl94_ai/evaluation/fixture_regression_compare.py)
assumes a different seed set, goalie profiles, and cadence. It cannot validate
this matrix unchanged. The CPU reports contain action hashes, not frame traces.
Do not claim to locate a first divergence from a hash alone.

After archiving the untouched baseline, and before agent changes, add a small
report comparison utility. This is planned tooling, not an existing command.
Keep it independent of match execution and pin its version for every comparison.
It must:

1. Read both cadence reports and their run manifests. Validate protocol,
   effective gameplay settings, dependency versions, ROM/save/checkpoint hashes,
   lineup and side. Permit expected source/output-path differences; never ignore
   arbitrary settings or gameplay fields to obtain a match.
2. Index `matches` by matchup, seed, and effective profile. Reject duplicate,
   missing, unexpected, failed, or incomplete trials. In checkpoint mode,
   explicitly select seeds 8000–8001 from the full reference and require exactly
   that set in the candidate. Full mode requires all 80 periods.
3. Compare `actions_sha256`, frame counts, and deterministic gameplay fields
   separately from timing and diagnostics. Derive agent/opponent score indices
   from controller side: report arrays are home/away, not agent/opponent.
4. Emit paired differences and per-team/per-cadence summaries in machine-readable
   and readable form. Exact mode fails on any gameplay mismatch. Behavior-change
   mode reports differences for review; it cannot automatically declare a fix good.
5. Test missing/duplicate seeds, wrong profiles/assets, incomplete periods,
   away-team score attribution, exact matches, and intentional gameplay changes.

Preserve the existing lifecycle trace contract during structural work wherever
possible. If diagnostics intentionally change, compare the applied action stream
and gameplay outcomes independently and test the diagnostic mapping separately;
do not discard a whole trace mismatch as cosmetic.

For a core CPU hash mismatch, replay the exact affected seed/profile on both the
frozen reference and candidate with identical trace instrumentation. First verify
the instrumented action hashes reproduce their saved uninstrumented reports.
Then inspect the first differing applied action and its input state. Keep this
instrumentation fixed across revisions and outside measured timing. Existing
lifecycle fixtures with different settings cannot establish that divergence.

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
| Structural refactor or cleanup | Identical action hashes, frames, scores, shots, and deterministic gameplay outcomes for every trial required by that gate. Explain and test diagnostic-only representation changes. Final validation covers the full matrix. |
| Intentional behavior fix | A focused reproduction passes; document why the first action divergence is expected; compare all paired results and affected native replays. No unexplained regression is acceptable. |
| Incomplete or inconsistent run | Fail the gate. Preserve the evidence, resolve the cause, and rerun under a new output label. Never drop failed seeds or count partial scores. |

For behavior fixes, a passing regression establishes the targeted correction;
goals and wins provide impact evidence, not a correctness oracle. Set this
escalation rule before inspecting candidate results: a reduction in aggregate
goal differential or increase in goals conceded for either team/cadence requires
review of the changed seeds and a matched extension on seeds 8020–8039 for both
teams at the affected cadence/profile, on both reference and candidate. Any crash,
incomplete period, or invariant failure blocks acceptance immediately. Investigate
other unexplained divergences even when scores improve.

Record the evidence and reasoning for accepting or rejecting a behavior tradeoff;
do not pool away a team regression or choose new thresholds after seeing results.
The extension is additional evidence, not proof of general playing strength.
Exact structural parity remains mandatory regardless of score improvements.

Maintain this comparison table as reports become available:

| Stage/tier | Cadence | Agent team | Completed/required | W/D/L | GF/GA | Goal-difference delta | Action-hash matches | Report | Gate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Before/full | 1 | Sabres | 20/20 | 12/4/4 | 22/5 | Reference | Reference | [native-1.json](../tmp/agents-refactor/2026-10-21/before/native-1.json) | Core/repeat passed |
| Before/full | 1 | Ducks | 20/20 | 12/6/2 | 28/8 | Reference | Reference | [native-1.json](../tmp/agents-refactor/2026-10-21/before/native-1.json) | Core/repeat passed |
| Before/full | 4 | Sabres | 20/20 | 15/4/1 | 42/4 | Reference | Reference | [native-4.json](../tmp/agents-refactor/2026-10-21/before/native-4.json) | Core/repeat passed |
| Before/full | 4 | Ducks | 20/20 | 11/7/2 | 23/4 | Reference | Reference | [native-4.json](../tmp/agents-refactor/2026-10-21/before/native-4.json) | Core/repeat passed |
| Before-repeat/check | 1 | Sabres | 2/2 | 1/0/1 | 6/1 | 0 (paired seeds) | 2/2 | [native-1.json](../tmp/agents-refactor/2026-10-21/before-repeat/native-1.json) | Exact |
| Before-repeat/check | 1 | Ducks | 2/2 | 1/1/0 | 3/1 | 0 (paired seeds) | 2/2 | [native-1.json](../tmp/agents-refactor/2026-10-21/before-repeat/native-1.json) | Exact |
| Before-repeat/check | 4 | Sabres | 2/2 | 1/1/0 | 3/1 | 0 (paired seeds) | 2/2 | [native-4.json](../tmp/agents-refactor/2026-10-21/before-repeat/native-4.json) | Exact |
| Before-repeat/check | 4 | Ducks | 2/2 | 0/1/1 | 0/1 | 0 (paired seeds) | 2/2 | [native-4.json](../tmp/agents-refactor/2026-10-21/before-repeat/native-4.json) | Exact |
| Stage 1a/check | 1 | Sabres | 2/2 | 1/0/1 | 6/1 | 0 | 2/2 | [native-1.json](../tmp/agents-refactor/2026-10-21/after-learned/native-1.json) | Exact |
| Stage 1a/check | 1 | Ducks | 2/2 | 1/1/0 | 3/1 | 0 | 2/2 | [native-1.json](../tmp/agents-refactor/2026-10-21/after-learned/native-1.json) | Exact |
| Stage 1a/check | 4 | Sabres | 2/2 | 1/1/0 | 3/1 | 0 | 2/2 | [native-4.json](../tmp/agents-refactor/2026-10-21/after-learned/native-4.json) | Exact |
| Stage 1a/check | 4 | Ducks | 2/2 | 0/1/1 | 0/1 | 0 | 2/2 | [native-4.json](../tmp/agents-refactor/2026-10-21/after-learned/native-4.json) | Exact |
| Stage 1b/full | 1 | Sabres | 20/20 | 12/4/4 | 22/5 | 0 | 20/20 | [native-1.json](../tmp/agents-refactor/2026-10-21/after-forecast-parallel/native-1.json) | Accepted; core exact |
| Stage 1b/full | 1 | Ducks | 20/20 | 12/6/2 | 28/8 | 0 | 20/20 | [native-1.json](../tmp/agents-refactor/2026-10-21/after-forecast-parallel/native-1.json) | Accepted; core exact |
| Stage 1b/full | 4 | Sabres | 20/20 | 15/4/1 | 42/4 | 0 | 20/20 | [native-4.json](../tmp/agents-refactor/2026-10-21/after-forecast-parallel/native-4.json) | Accepted; core exact |
| Stage 1b/full | 4 | Ducks | 20/20 | 11/7/2 | 23/4 | 0 | 20/20 | [native-4.json](../tmp/agents-refactor/2026-10-21/after-forecast-parallel/native-4.json) | Accepted; core exact |
| Stage 2/check | 1 | Sabres | 2/2 | 1/0/1 | 6/1 | 0 | 2/2 | [native-1.json](../tmp/agents-refactor/2026-10-21/after-options/native-1.json) | Exact |
| Stage 2/check | 1 | Ducks | 2/2 | 1/1/0 | 3/1 | 0 | 2/2 | [native-1.json](../tmp/agents-refactor/2026-10-21/after-options/native-1.json) | Exact |
| Stage 2/check | 4 | Sabres | 2/2 | 1/1/0 | 3/1 | 0 | 2/2 | [native-4.json](../tmp/agents-refactor/2026-10-21/after-options/native-4.json) | Exact |
| Stage 2/check | 4 | Ducks | 2/2 | 0/1/1 | 0/1 | 0 | 2/2 | [native-4.json](../tmp/agents-refactor/2026-10-21/after-options/native-4.json) | Exact |
| Stage 3a/check | 1 | Sabres | 2/2 | 1/0/1 | 6/1 | 0 | 2/2 | [native-1.json](../tmp/agents-refactor/2026-10-21/after-history/native-1.json) | Exact |
| Stage 3a/check | 1 | Ducks | 2/2 | 1/1/0 | 3/1 | 0 | 2/2 | [native-1.json](../tmp/agents-refactor/2026-10-21/after-history/native-1.json) | Exact |
| Stage 3a/check | 4 | Sabres | 2/2 | 1/1/0 | 3/1 | 0 | 2/2 | [native-4.json](../tmp/agents-refactor/2026-10-21/after-history/native-4.json) | Exact |
| Stage 3a/check | 4 | Ducks | 2/2 | 0/1/1 | 0/1 | 0 | 2/2 | [native-4.json](../tmp/agents-refactor/2026-10-21/after-history/native-4.json) | Exact |
| Stage 3b/full | 1 | Sabres | 20/20 | 12/4/4 | 22/5 | 0 | 20/20 | [native-1.json](../tmp/agents-refactor/2026-10-21/after-resources/native-1.json) | Exact |
| Stage 3b/full | 1 | Ducks | 20/20 | 12/6/2 | 28/8 | 0 | 20/20 | [native-1.json](../tmp/agents-refactor/2026-10-21/after-resources/native-1.json) | Exact |
| Stage 3b/full | 4 | Sabres | 20/20 | 15/4/1 | 42/4 | 0 | 20/20 | [native-4.json](../tmp/agents-refactor/2026-10-21/after-resources/native-4.json) | Exact |
| Stage 3b/full | 4 | Ducks | 20/20 | 11/7/2 | 23/4 | 0 | 20/20 | [native-4.json](../tmp/agents-refactor/2026-10-21/after-resources/native-4.json) | Exact |
| Stage 4/full | 1 | Sabres | 20/20 | 12/4/4 | 22/5 | 0 | 20/20 | [native-1.json](../tmp/agents-refactor/2026-10-21/after-execution/native-1.json) | Exact |
| Stage 4/full | 1 | Ducks | 20/20 | 12/6/2 | 28/8 | 0 | 20/20 | [native-1.json](../tmp/agents-refactor/2026-10-21/after-execution/native-1.json) | Exact |
| Stage 4/full | 4 | Sabres | 20/20 | 15/4/1 | 42/4 | 0 | 20/20 | [native-4.json](../tmp/agents-refactor/2026-10-21/after-execution/native-4.json) | Exact |
| Stage 4/full | 4 | Ducks | 20/20 | 11/7/2 | 23/4 | 0 | 20/20 | [native-4.json](../tmp/agents-refactor/2026-10-21/after-execution/native-4.json) | Exact |
| Stage 5/check | 1 | Sabres | 2/2 | 1/0/1 | 6/1 | 0 | 2/2 | [native-1.json](../tmp/agents-refactor/2026-10-21/after-mechanics/native-1.json) | Exact |
| Stage 5/check | 1 | Ducks | 2/2 | 1/1/0 | 3/1 | 0 | 2/2 | [native-1.json](../tmp/agents-refactor/2026-10-21/after-mechanics/native-1.json) | Exact |
| Stage 5/check | 4 | Sabres | 2/2 | 1/1/0 | 3/1 | 0 | 2/2 | [native-4.json](../tmp/agents-refactor/2026-10-21/after-mechanics/native-4.json) | Exact |
| Stage 5/check | 4 | Ducks | 2/2 | 0/1/1 | 0/1 | 0 | 2/2 | [native-4.json](../tmp/agents-refactor/2026-10-21/after-mechanics/native-4.json) | Exact |
| Stage 6/check | 1 | Sabres | 2/2 | 1/0/1 | 6/1 | 0 | 2/2 | [native-1.json](../tmp/agents-refactor/2026-10-21/after-workflows/native-1.json) | Exact |
| Stage 6/check | 1 | Ducks | 2/2 | 1/1/0 | 3/1 | 0 | 2/2 | [native-1.json](../tmp/agents-refactor/2026-10-21/after-workflows/native-1.json) | Exact |
| Stage 6/check | 4 | Sabres | 2/2 | 1/1/0 | 3/1 | 0 | 2/2 | [native-4.json](../tmp/agents-refactor/2026-10-21/after-workflows/native-4.json) | Exact |
| Stage 6/check | 4 | Ducks | 2/2 | 0/1/1 | 0/1 | 0 | 2/2 | [native-4.json](../tmp/agents-refactor/2026-10-21/after-workflows/native-4.json) | Exact |
| Stage 7 + final/full | 1 | Sabres | 20/20 | 12/4/4 | 22/5 | 0 | 20/20 | [native-1.json](../tmp/agents-refactor/2026-10-21/after-final/native-1.json) | Exact |
| Stage 7 + final/full | 1 | Ducks | 20/20 | 12/6/2 | 28/8 | 0 | 20/20 | [native-1.json](../tmp/agents-refactor/2026-10-21/after-final/native-1.json) | Exact |
| Stage 7 + final/full | 4 | Sabres | 20/20 | 15/4/1 | 42/4 | 0 | 20/20 | [native-4.json](../tmp/agents-refactor/2026-10-21/after-final/native-4.json) | Exact |
| Stage 7 + final/full | 4 | Ducks | 20/20 | 11/7/2 | 23/4 | 0 | 20/20 | [native-4.json](../tmp/agents-refactor/2026-10-21/after-final/native-4.json) | Exact |

Add four rows for each checkpoint and the final candidate, marking 2 versus 20
required seeds per row. Compare final results with both the original baseline and
the latest full accepted reference; separate intentional fixes from structural
changes. Capture a full reference after any accepted fix affecting the core suite.
Refresh affected replay/optional references after that fix as well, retaining the
original evidence. Subsequent structural stages compare against those accepted
behaviors; they must not silently undo an earlier correction.

## Stage 1: bounded correctness fixes, separate from simplification

These patches are independently reviewable. Do not combine them with extraction
or directory moves; neither promises better match results.

### Persistent learned-agent adapters

Source: [multi_model.py](../nhl94_ai/agents/multi_model.py),
[base.py](../nhl94_ai/agents/base.py),
[play.py](../nhl94_ai/evaluation/play.py).

- [x] Construct one `LearnedAgent` adapter per loaded model and retain it between
  predictions. Define reset, model replacement, and switching semantics explicitly.
  Preserve independent state for independent controller streams.
- [x] Update the actual playback episode-boundary path: it currently resets
  models only for scripted play. Learned adapters must receive episode resets,
  including automatic environment resets, without sending stale state into a
  new episode. Retain scripted, probability-display, and away-team behavior.
- [x] Add a stateful fake-policy regression showing that the second call receives
  the first call's recurrent state and `episode_start=False`, and that reset
  starts a new episode.
- [x] Exercise two episodes through the playback caller, including auto-reset,
  switching/replacement, and all supported compatibility and `Agent.act()` paths.

The review reproduced state loss with a stateful fake policy. Current supported
PPO/GRU policies process observation windows without carrying recurrent state
across calls, so degradation of an existing checkpoint has not been demonstrated.
Validate the adapter contract without overstating its playing-strength impact.
Use focused learned playback/checkpoint coverage and the 8-period structural
screen; the Classic core results must stay exactly unchanged.

### Coherent forecast coordinates

Source: [offense.py](../nhl94_ai/agents/offense.py),
[skating.py](../nhl94_ai/agents/skating.py).

- [x] Repair the legacy `projected_state()` path so precise and integer/display
  coordinates describe the same projected position under the chosen model.
- [x] Audit puck coordinates/velocity and related forecast builders for the same
  invariant, without changing their modeling assumptions incidentally.
- [x] Add a regression that composes projection with a native movement step and
  verifies no jump toward the old position and no mutation of the source state.
- [x] Exercise relevant feint, carry, pass-continuation, and finishing replays.

The coordinate inconsistency was reproduced directly. Its match impact has not
yet been established, so explain any observed gameplay divergence with replay
evidence rather than assuming a strength improvement.

Gate for the forecast fix: focused regression, full Ducks/Sabres core suite, and
affected optional before/after profiles. Establish an accepted full reference
before subsequent structural changes. Preserve unrelated forecast semantics.

## Stage 2: normalize options and simplify construction

Source: [classic_v1.py](../nhl94_ai/agents/classic_v1.py),
[registry.py](../nhl94_ai/agents/registry.py),
[classic_config.py](../nhl94_ai/agents/classic_config.py),
[base.py](../nhl94_ai/agents/base.py),
[config.py](../nhl94_ai/config.py).

- [x] Introduce immutable normalized Classic options. Preserve distinctions
  between unspecified values, explicit overrides, and disabled features.
- [x] Establish one compatibility-validation entry point used by CLI,
  benchmarks, and direct callers, with explicit platform restrictions. Preserve
  intentional caller-specific defaults and the order of explicit overrides.
- [x] Extract offense and goalie construction into small factories.
- [x] Keep this change separate from resource caching, reset changes, and file
  reorganization. Factories should remove repeated construction decisions, not
  introduce a general registration framework.
- [x] Preserve the ordinary scripted import path's avoidance of neural imports.

Gate: option/default/compatibility tests, direct and legacy caller timing tests,
and exact 8-period core parity. Capture and compare affected goalie/model profiles
on both teams before changing their construction paths.

## Stage 3: define state ownership, then simplify reset and resource loading

Source: [lifecycle.py](../nhl94_ai/agents/lifecycle.py),
[scheduling.py](../nhl94_ai/agents/scheduling.py),
[offense.py](../nhl94_ai/agents/offense.py),
[lifecycle_history.py](../nhl94_ai/evaluation/lifecycle_history.py),
[chance_creation_replay.py](../nhl94_ai/evaluation/chance_creation_replay.py).

First make state ownership and replay transfer explicit, in a separate commit:

- [x] Inventory clocks, button edges, commitments, retry/cooldown history,
  diagnostics, and loaded resources. Record each owner and its reset, cancel,
  continue, and replay-fork behavior.
- [x] Characterize state changes during proposal selection, including proposals
  later overridden. `_feint()` updates target/deadline/retry state; chance creation
  updates owner/deadline state and later cleanup depends on the final decision.
  These effects are existing behavior, not automatically safe to remove.
- [x] Replace replay consumers' direct `__dict__` copying with a narrow, explicit
  history export/import or fork contract before changing controller state layout.
  Preserve the target implementation when transferring reference history, isolate
  mutable state, and keep existing replay/archive data readable where supported.
  Avoid building a universal serialization framework for this need.
- [x] Test overwritten proposals, cancellation, held frames, episode reset, and
  independent replay branches against the existing behavior.

Then change resource loading and reset in an independently reversible commit:

- [x] Separate immutable loaded artifacts from per-episode execution state.
  Reset clocks, commitments, recurrent state, and episode metrics without
  reloading neural checkpoints or rereading JSON resources.
- [x] Read/hash/parse retention artifacts once before schema dispatch.
- [x] Check that sharing immutable resources never shares mutable state between
  opponents, teammate controllers, or episodes.
- [x] Preserve existing retention encoders, schema dispatch, and model outputs.
  Changes to file loading must not silently correct feature semantics.

Gate: state-transfer/reset/isolation tests and existing lifecycle replay parity.
Use the full core matrix for state transition/reset changes, plus both-team
before/after coverage for affected model and goalie paths. A purely mechanical
loading extraction can use the smaller checkpoint when execution is unchanged.

## Stage 4: separate tactical selection from action execution incrementally

Source: [classic_v1.py](../nhl94_ai/agents/classic_v1.py),
[lifecycle.py](../nhl94_ai/agents/lifecycle.py),
[scheduling.py](../nhl94_ai/agents/scheduling.py),
[learned_offense.py](../nhl94_ai/agents/learned_offense.py).

- [x] Start from the Stage 3 ownership and side-effect inventory. Selection is
  currently stateful: do not assume only the winning proposal may update state.
- [x] Extract one tactic at a time from `_attack()`, keeping priority visible in
  the orchestrator and preserving the order of evaluation and state updates.
  Use small typed proposals where they clarify target, selected option, and
  evidence; reuse existing types where sufficient.
- [x] Route selected actions through shared entry points that start the existing
  lifecycles. Preserve side effects of overridden/abandoned proposals until a
  separately justified behavior change explicitly removes them.
- [x] Reuse those entry points for learned offense commands while retaining
  teacher labeling, action masks, and command-consumption behavior.
- [x] Keep observed release/reception distinct from an input request.
- [x] Preserve exclusive input-owner priority, B/C press/release edges, handoff
  frames, retry history, and frame-versus-decision deadlines.
- [x] Make cancellation/stoppage operations explicit instead of having callers
  clear another controller's internal fields individually.

Gate: lifecycle, reception, one-timer, shot-hold, control-rate, and learned-offense
regressions; native lifecycle/history replay parity. Use small checkpoints for
individual bounded extractions and the full Ducks/Sabres core suite before
accepting the completed executor change. Capture and compare affected experiment
profiles before changing their executors.

## Stage 5: consolidate forecasting and correct dependency direction

Source: `agents/motion.py`, `skating.py`, `passing.py`, `offense.py`,
`possession_value.py`, `responses.py`, teammate forecasts, and `game/platforms`.

- [x] Extract shared actor predicates and forecast cloning/update primitives.
  Name controllable-skater and observable-body predicates distinctly, but retain
  each caller's existing semantics, including the legacy retention encoder.
- [x] Move console arithmetic and physical primitives beneath the agent layer.
  Platform backends must not depend on tactical controller modules.
- [x] Keep tactical scoring and admission decisions in agents.
- [x] Preserve separate conservative, sampled, coasting, and CPU-response models
  and their uncertainty labels while sharing state-update mechanics.
- [x] Update benchmark/replay source-fingerprint manifests for every moved file
  and shared dependency, preserving benchmark semantics and report contracts.
- [x] Keep compatibility re-exports where existing callers need them, then
  migrate internal imports to the owning module.

Gate: platform mechanics, projection invariants, no-source-mutation checks,
carry/pass replay comparisons, native Genesis checks, one basic SNES run, and exact core
parity under the benchmark schedule. A source-path-only fingerprint change is
expected; an action change is not.

## Stage 6: consolidate offense workflows and decision evidence

Source: `offense.py`, `possession.py`, `possession_value.py`,
`possession_ablation.py`, `receiver_selection.py`, `decisions.py`, and `base.py`.

- [x] Share the duplicated feedback checks, close-finish/breakaway handling,
  cut continuation, pass/feint sequence, and fallback candidate generation.
- [x] Retain explicit hooks for admission, scoring, and continuation models.
  Preserve candidate ordering, tie-breaking, incumbent fallback, and stateful
  selection effects. Share only workflows with the same underlying semantics.
- [x] Produce typed decision evidence with the selected proposal and serialize
  it to current UI/evaluation dictionaries at the boundary.
- [x] Reduce decision-name parsing and introspection of controller internals.
- [x] Define the supported `act`, frame execution, and compatibility inference
  contracts; migrate callers before removing wrappers or broad `__getattr__`.
- [x] Keep diagnostics observational: building a snapshot must not evaluate new
  alternatives, change controller state, or inspect fresh state on held frames.

Gate: offense selection, ablation, receiver, inspector, overlay, and held-frame
tests; exact core parity under the benchmark schedule; before/after parity for
each affected optional policy. Review whether tracing a decision is actually
easier; undo abstractions that add navigation without removing duplication.

## Stage 7: remove dead code; reorganize files only where useful

- [x] Remove verified unused legacy helpers/members in `multi_model.py`, including
  `test`, `target_xy`, `DistToPos()`, and `SelectRandomTarget()`, after checking
  supported external interfaces and compatibility requirements.
- [x] Remove write-only teammate coordinator history fields and update tests that
  still assign them; preserve `SharedReceiving`'s actual history ownership.
- [x] Remove unused imports and duplicate button-encoding helpers.
- [x] Preserve intentional exports such as `passing.rom_direction` until their
  consumers have migrated.
- [x] Preserve movement/action ID ordering and checkpoint contracts when sharing
  direction tables; different tables currently have different orders.
- [x] Evaluate grouping stable modules by Classic tactics, goalie control,
  teammate control, and experiments; retain the existing layout because the new
  ownership boundaries already improve navigation (justification below).
  Directory reorganization is optional and must be a separate commit from logic,
  configuration, or reset changes. Keep public import shims as needed and update
  documentation/source manifests together.
- [x] Run focused unused-code lint without blanket removal of compatibility code.

Gate: exact gameplay parity, package/import contracts, lint, wheel build, and
relevant teammate checks on both Ducks and Sabres.

## Deferred: corrected retention features and artifact migration

The review reproduced an observable goalie being excluded by the current
controllable-skater predicate in
[retention_features.py](../nhl94_ai/agents/retention_features.py). Correcting that
changes the meaning of inputs to existing weights, so it is not structural cleanup.

[retention.py](../nhl94_ai/training/retention.py) and
[retention_ranking.py](../nhl94_ai/training/retention_ranking.py) currently expose
unavailable training entry points, consistent with [ARTIFACTS.md](ARTIFACTS.md).
Restoring training is outside this plan and must not block the refactor.

If this separate workstream is later undertaken, it must capture retention-enabled
cadence-4 baselines on both teams, introduce versioned feature semantics, retain
an explicit legacy encoder or reject incompatible artifacts, test actor ordering
and presence/goalie flags in both directions, and retrain/revalidate new artifacts.
Record artifact hashes and paired impact evidence. Do not feed corrected features
to old weights or claim the corrected path is ready without that work.

## Additional validation matrix

Before touching a supported path, record its baseline as well as its post-change
result. Optional experiments should be tested individually under their existing
compatibility rules. Record exact flags and artifact hashes in the run manifest.

| Affected area | Additional required coverage |
| --- | --- |
| Goalie construction/lifecycle | Goalie policy `off`, original manual positioning, and each affected learned/planner controller; both Genesis teams |
| Timing/configuration | Native tactics at 1/4 frames, supported strict control intervals, direct versus legacy callers, playback overrides, and `HOCKEY_INTENT_DPAD` paths |
| Retention loading/structure | Existing retention-enabled cadence-4 profiles for both teams, unchanged feature/model semantics, schema dispatch and reset isolation; corrected features/retraining remain deferred |
| Learned adapters/offense | Stateful fake-policy tests, two playback episodes including auto-reset, model switching/replacement, masks, teacher labels, available checkpoint smoke checks |
| State ownership/history | Overridden proposal side effects, cancellation, reset, held frames, lifecycle/chance replay history transfer and branch isolation |
| Shared mechanics | Genesis platform and native replay checks; SNES basic execution smoke only, with further SNES validation deferred by user instruction |
| Teammate coordination | Home Sabres and away Ducks, two and three bots, coordinated/independent strategies, pressure/support options when affected |
| Diagnostics/API | Decision inspector, held snapshots, overlays, recording consumers, action shapes, package/import compatibility |

Use existing native integration runners for the relevant change, such as
`tests.integration.classic_v1`, `tests.integration.cpu_benchmark`,
`tests.integration.classic_config`, `tests.integration.manual_goalie`, and the
domain-specific lifecycle/offense/teammate runners. Inspect each runner's
arguments and dependencies before invoking it. Existing frozen fixtures must not
be regenerated just to make a refactor pass.

Run focused checks for each small change and the full suite/lint at accepted
stage boundaries. Do not repeat successful runs on unchanged source without a
specific reason:

```bash
/home/mat/vretro/bin/python -m unittest discover -s tests -v
/home/mat/vretro/bin/python -m pylint nhl94_ai tests -j 15
```

Build the wheel after module moves and at final validation:

```bash
/home/mat/vretro/bin/python -m build --wheel
```

## Final completion criteria

- [x] Baseline and final benchmark reports exist for both Ducks and Sabres at
  both required cadences, with all periods complete.
- [x] Every intermediate stage has a recorded gate result; the final four-row
  team/cadence comparison includes links to raw reports and paired differences.
  Coverage labels distinguish 8-period screens from full 80-period suites.
- [x] Each behavior fix has focused regression evidence and explained gameplay
  changes; each structural stage preserves its accepted reference's gameplay.
- [x] All affected optional, teammate, learned, and SNES paths have before/after
  evidence, with any unavailable required prerequisite resolved before completion.
- [x] Existing test IDs/assertions remain intact unless a deliberate behavior
  change requires a documented adjustment. Expected failures are accounted for.
- [x] Unit tests, required native checks, lint, and packaging checks pass.
- [x] Model semantics, action IDs, reset behavior, public imports, and benchmark
  fingerprints remain compatible or have explicit migration/version handling.
- [x] The readability criteria at the start of this plan are met, with a short
  before/after walkthrough showing fewer responsibilities and clearer state
  ownership. File moves and extra abstraction do not count as simplification.
- [x] Existing retention model semantics are preserved. Deferred training and
  feature correction are recorded separately, not left as hidden completion gates.
- [x] Reports and generated artifacts stay under the ignored run directory;
  source documentation contains the plan and final evidence references.
- [x] Unrelated pre-existing working-tree changes are preserved.

## Execution record

### Baseline preparation (2026-10-21)

Evidence root: [`tmp/agents-refactor/2026-10-21`](../tmp/agents-refactor/2026-10-21/).
The starting revision is `ec71dfa`; the only pre-existing working-tree change was
this plan. Its complete original diff is preserved in
[`baseline-working-tree.patch`](../tmp/agents-refactor/2026-10-21/baseline-working-tree.patch).
No untracked source files were present. The source archive includes the dirty
plan and all 517 tracked files; its extracted copy is available for matched
reference runs.

- [`baseline-manifest.json`](../tmp/agents-refactor/2026-10-21/baseline-manifest.json)
  records the full revision, source hashes (including all agent/platform files),
  Python 3.12.3 and dependency versions, machine/load details, package import path,
  and Genesis ROM/save-state hashes.
- [`baseline-commands.txt`](../tmp/agents-refactor/2026-10-21/baseline-commands.txt)
  records the exact core/repeat/lifecycle invocation and unit command. Core runs
  use the plan's settings without overrides and one worker.
- [`baseline-source.tar.gz`](../tmp/agents-refactor/2026-10-21/baseline-source.tar.gz)
  is the restorable source; `baseline-source/` is its extracted copy.
- [`before-unit.log`](../tmp/agents-refactor/2026-10-21/before-unit.log): 1,293
  tests passed in 23.194 seconds, with the existing expected failure
  `unit.test_environment_contracts.EnvironmentContracts.test_frame_skip_stops_on_truncation`.
- [`baseline-optional-assets.json`](../tmp/agents-refactor/2026-10-21/baseline-optional-assets.json)
  records hashes of existing tree/ranker retention artifacts and available goalie
  and offense checkpoints, copied into `baseline-assets/`. No artifacts were
  trained or changed. Additional optional-profile evidence remains pending.
- [`before-structure.json`](../tmp/agents-refactor/2026-10-21/before-structure.json)
  records function sizes, agent imports, and component-assignment sites. The
  largest agent function is `_attack()` (203 lines); Classic construction spans
  155 lines. The mechanical assignment inventory contains 74 sites and requires
  semantic review rather than treating every assignment as an ownership defect.

The core suite completed all 80 periods before any implementation, test,
configuration, or benchmark-tooling change. Cadence 1 took 956.20 wall seconds
(938.47 user, 19.40 system); cadence 4 took 641.24 wall seconds (623.24 user,
19.59 system). Each team/cadence has exactly seeds 8000–8019, complete periods,
correct controller sides, and the Sabres-home/Ducks-away starting lineup. CPU
ownership is asserted by the existing runner before each period. The repeat completed all eight periods (93.99 seconds at cadence 1;
60.47 seconds at cadence 4). Direct comparison of every match field, source hash,
dependency version, and effective setting was exact; see
[`initial-comparison.json`](../tmp/agents-refactor/2026-10-21/before-repeat/initial-comparison.json).
The tested comparator also verified these reports:
[`validated-comparison.json`](../tmp/agents-refactor/2026-10-21/before-repeat/validated-comparison.json),
with [readable paired summaries](../tmp/agents-refactor/2026-10-21/before-repeat/validated-comparison.txt).
All eight action hashes and every deterministic match field agree. The complete
17-case lifecycle baseline is archived in
[`before-lifecycle/report.json`](../tmp/agents-refactor/2026-10-21/before-lifecycle/report.json)
with compressed native traces and unchanged-source verification. Additional
optional-path baselines remain required before changes to those paths; the
frozen original source is retained for those measurements.

### Comparison tooling

Added [`agents_refactor_compare.py`](../nhl94_ai/evaluation/agents_refactor_compare.py),
independent of execution, after the untouched baseline capture. It validates
manifests/assets/dependencies, effective profiles, exact seed coverage, trial
completion, controller sides, lineups, and lifecycle accounting. It compares all
match fields, reports diagnostic counts separately, attributes home/away scores
to the agent's side, and emits JSON plus readable summaries. Behavior mode never
automatically accepts a change. Every comparison records the comparator SHA-256.
Nine focused tests pass; focused pylint is 10.00/10. An initial lint failure
(test resource cleanup and loop-lambda bindings) was fixed before proceeding.

The learned-adapter loss is independently reproduced in
[`before-learned-adapter.json`](../tmp/agents-refactor/2026-10-21/before-learned-adapter.json):
the second call again receives no recurrent state and `episode_start=True`.
This establishes the adapter defect, not checkpoint degradation.

### Stage 1a candidate: persistent learned streams

Each model slot retains one adapter. Switching slots resumes that slot's history;
replacing a model starts a new stream; replacing the loaded model list removes
omitted slots. Episode reset clears all streams, including inactive slots, while
independent systems/slots retain independent histories. Existing Agent objects
continue to own their `act()` and `reset()` implementation.

Learned playback now resets adapters on explicit and vector auto-reset boundaries
and stops applying the ended episode's held action. Terminal RAM feedback is not
passed into the next episode. Learned-versus-learned playback also retains two
adapters and preserves its historical sampled prediction default. Scripted reset,
away views, probability display, and target-position playback contracts remain.

Validation: [`learned-unit-final.log`](../tmp/agents-refactor/2026-10-21/learned-unit-final.log)
passes 1,310 tests with the same expected failure. Eight new stream regressions
exercise compatibility prediction, `Agent.act()`, switching, replacement, independent
streams, and two episodes through both playback loops with explicit/automatic reset.
Available offense and goalie artifacts produce identical deterministic actions and
probabilities across two resets on the frozen original and candidate:
[`learned-checkpoint-comparison.json`](../tmp/agents-refactor/2026-10-21/learned-checkpoint-comparison.json).
Those are experiment artifacts, not general button-playback checkpoints; actual
playback recurrence is validated by stateful fake policies. The checkpoint probe
was run against the frozen source after the candidate edit; no artifact or inference
implementation changed. The earlier adapter reproduction was captured before editing.
The core 8-period screen passed exactly against the original baseline:
[`after-learned/comparison.json`](../tmp/agents-refactor/2026-10-21/after-learned/comparison.json),
[readable summary](../tmp/agents-refactor/2026-10-21/after-learned/comparison.txt).
This is an 8-period screen, not a new full reference. The behavior patch is
committed separately from comparison tooling and lint-only annotations.

Lint gate deviation: the first full lint run exposed seven pre-existing warnings
in six unchanged files (three compound guards, two platform interface sizes, and
two unimplemented legacy Genesis shot methods). A separate annotation-only commit
preserves their ASTs exactly and scopes suppression to the affected statement/class;
[`learned-lint-resolved.log`](../tmp/agents-refactor/2026-10-21/learned-lint-resolved.log)
now passes at 10.00/10. The temporary Genesis annotation is to be removed when
Stage 5 moves those physical primitives. The initial comparator lint failures and
a missing probability array in a new test mock were also resolved before advancing.

Additional baseline coverage is pinned in [`optional-profiles.json`](../tmp/agents-refactor/2026-10-21/optional-profiles.json): 44 individual profiles, both teams, seeds 8000–8001, 300 game-clock seconds, one worker. These cover offense experiments, both existing retention schemas, fixed shot hold, goalie construction variants/planners/checkpoint, strict control intervals, and intents. Runs use the frozen original source before forecast/factory changes; their manifests record effective commands and artifact hashes. The fixed-hold JSON is a deterministic control artifact, not a trained model.


Native divergence tooling is pinned in
[`agents_refactor_trace.py`](../nhl94_ai/evaluation/agents_refactor_trace.py).
It captures actual emulator inputs and their pre-decision state, keeps diagnostic
changes separate, and refuses a trace unless every saved uninstrumented match
field and the independently hashed applied stream reproduce. Four focused tests
and lint pass; an actual native reproduction check remains pending before use.

The comparator also supports the predeclared `extension` tier: exactly seeds
8020–8039 on both teams. It rejects using the original seed range as extension
evidence and reports goal-difference/conceded-goal escalation separately for each
team/profile. Eleven comparator tests pass, including an away-team regression
that cannot be concealed by an improvement on the home team; focused lint passes.
See [`comparison-extension-unit.log`](../tmp/agents-refactor/2026-10-21/comparison-extension-unit.log).
The original comparison outputs retain their original comparator fingerprints.

### Stage 1b reproduction and forecast audit

[`before-projection.json`](../tmp/agents-refactor/2026-10-21/before-projection.json)
reproduces a carrier advancing from y=-130 to y=-108.59 while `precise_y` remains
-130. The next native movement step returns to y=-129, a 20.41-pixel jump.
Seven regression tests now fail on the unfixed code for the expected coordinate
invariant, including receiver-origin, reception/carry puck, CPU-response puck,
crossing forecasts, and deke-preview puck updates. Their initial output is retained
in [`before-projection-tests.log`](../tmp/agents-refactor/2026-10-21/before-projection-tests.log).
The tests assert source isolation and existing velocity assumptions. Native
skating, possession-value's future builder, and teammate support's translated
precise puck coordinates already preserve their chosen position model.

### State-ownership inventory for the structural stages

The mechanical inventory is
[`before-state-writes.json`](../tmp/agents-refactor/2026-10-21/before-state-writes.json).
The following responsibilities and existing cancellation semantics are the
reference for Stage 3; implementation and replay-transfer validation remain pending.

| Owner | Episode history | Cancellation/continuation contract |
| --- | --- | --- |
| `ActionScheduler` | Native frames, tactical decisions, elapsed/remaining cadence, held action, defense transition | Interrupt clears remaining hold; tactical deadlines use decisions and pause during exclusive actions |
| `ButtonState` | Last applied B/C edges | New press requires a released edge; held frames retain the cached action |
| `ShotLifecycle` | Hold/follow-through deadlines, shooter/side, shot count, crossing/rebound state | Cancel ends commitment and release evidence; recovery count remains episode history |
| `OneTimerLifecycle` | Request, release/receiver evidence, starts/outcomes, retry frame | Finishing records an outcome and clears pending; retry history survives; candidate geometry is tactical evidence |
| `PassLifecycle` | Request, observed flight/reception, correction, last request/outcome, metrics | Cancel discards pending without fabricating an outcome; deadline precedes reception in observation priority |
| `SetupState` | Receiver, decision deadline, retry decision | End setup without erasing retry history |
| `OffenseController` | Pass/feint/chance retry clocks, feint target/mode/deadline, chance owner/deadline, diagnostics/risk cache | Cancel clears active feint/chance/pass and caches, retaining retry clocks; starting a pass clears feint/chance commitments |
| `DefenseController` | Frame clock, assignment, switch/check requests and retries, local button feedback | Idle clears requests/selection/button feedback, retaining retry clocks |
| Goalie controller | Native phase/clock, takeover/save/outlet requests, handoff/retry timing, observed feedback, metrics | Stoppage/return use existing neutral handoff frames; episode reset clears all history |
| Crossing/deke controllers | Maneuver, phase/deadlines, retry clock, event history/metrics | End records outcome and starts the established cooldown; no fabricated release |
| Classic orchestrator | Input owner, final decision/target/preferences, recovery switch, pass-request evidence, component diagnostics | Episode reset resets all execution history while retaining configured artifacts |

Selection is intentionally stateful: `_feint()` commits target, mode, deadline,
and retry time before later Classic priorities can override its proposal.
Chance creation likewise sets owner/deadline and cleans up based on the final
decision. Replay forks must isolate these mutable histories while retaining the
target implementation and excluding instrumentation hooks and loaded resources.

The SNES ROM/save prerequisite is available under stable-retro's **experimental**
integration (`Integrations.ALL`, matching the existing native factory). A lookup
using stable-only defaults initially missed it; no installation or asset change
was needed. [`baseline-snes-assets.json`](../tmp/agents-refactor/2026-10-21/baseline-snes-assets.json)
records both hashes. Additional native coverage will use existing SNES recording
and playback functions (both teams/controllers, cadences 1/4, plus both CPU-playback
sides) and teammate runners (home/away, two/three bots, independent/coordinated,
pressure-off/support options). Teammate probes use complete 60-game-second periods
with seed 8000; they are additional focused coverage, not the 300-second core
benchmark. Commands and complete results will remain under the evidence root.

Additional baseline gate findings: all 44 optional profiles completed 176 periods,
but strict protocol validation rejected `goalie-classic` and `goalie-always`.
The existing CPU runner recorded explicit `classic` positioning in settings but
omitted that value when dispatching each trial, causing automatic `deep-csave`
selection. This does not affect the core suite's explicit `deep-csave` profile.
The invalid reports are preserved and are not accepted as classic-goalie baselines.
The other 42 profiles (168 periods) validate completely in
[`before-optional/validation-42.json`](../tmp/agents-refactor/2026-10-21/before-optional/validation-42.json).

A separate runner correctness patch always forwards the resolved positioning.
A regression first failed for both selective and always policies, then passed;
70 focused tests and lint pass after updating mocked dispatch signatures.
The two affected profiles will be recaptured under a fresh label on a frozen
copy of the original source with only this runner fix. The remaining original
agent sources and artifacts are unchanged. The core eight-period screen will
also verify that this correction preserves the pinned core protocol.

The extra native driver initially lacked a multiprocessing main guard: playback's
forkserver child reexecuted output-directory creation. The failed run and its two
completed SNES periods remain under `before-native`; they are partial evidence.
The corrected external driver reran the entire matrix under
[`before-native-v2/report.json`](../tmp/agents-refactor/2026-10-21/before-native-v2/report.json):
two complete SNES periods (9,366 and 7,957 frames), both CPU playback sides, and
all 16 complete teammate periods pass. No agent source or native fixture changed.

The two corrected classic-goalie baselines now pass all protocol checks (eight
complete periods):
[`before-classic-goalie-corrected/validation.json`](../tmp/agents-refactor/2026-10-21/before-classic-goalie-corrected/validation.json).
Together with the 42 validated original profiles, all 44 have valid before-change
coverage. The original optional matrix took 3,156.69 wall seconds. The invalid
classic-goalie results remain excluded explicitly, with replacement evidence;
no seed was dropped. A byte comparison confirms the corrected frozen tree differs
from the original only in CPU dispatch, not in any agent/platform/fixture source.

The first native trace check rejected Python tuples in raw contact events versus
lists in saved JSON. The tool now compares the complete native result using the
same JSON representation as the saved report, with tests proving that changed
coordinates and changed applied hashes still fail. Six trace tests and lint pass.
The fresh
[`before-trace-ducks-8000-4-v2/report.json`](../tmp/agents-refactor/2026-10-21/before-trace-ducks-8000-4-v2/report.json)
reproduces every saved uninstrumented match field and independently verifies the
actual applied-action hash and frame count. The failed first trace is retained;
subsequent divergence comparisons must use this corrected instrumentation version.

The isolated benchmark-dispatch correction passes the complete eight-period core
screen exactly against the original:
[`after-benchmark-fix/comparison.json`](../tmp/agents-refactor/2026-10-21/after-benchmark-fix/comparison.json).
All action hashes, frames, gameplay fields, and diagnostic counts match.

### Stage 1b candidate validation

The coordinate patch adds precise-position updates at the existing projection
assignments in offense, passing, CPU response, crossing, and deke forecasts.
It preserves display positions, velocities, uncertainty, and each forecasting
model's chosen trajectory. It introduces no shared abstraction or tactical change.
The receiver's effective pre-release origin is also synchronized before reuse.
The reproduced projection now composes with a 1.59-pixel native step instead of
the 20.41-pixel backward jump:
[`after-projection.json`](../tmp/agents-refactor/2026-10-21/after-projection.json).

All seven new regressions pass; the full suite passes 1,326 tests with the original
expected failure (`EnvironmentContracts.test_frame_skip_stops_on_truncation`):
[`forecast-unit.log`](../tmp/agents-refactor/2026-10-21/forecast-unit.log).
Full lint passes at 10.00/10:
[`forecast-lint.log`](../tmp/agents-refactor/2026-10-21/forecast-lint.log).
The full 80-period Genesis core, 17 affected optional forecast profiles, and
complete lifecycle matrix will establish the candidate's impact before acceptance.
Stage 2 has not started.

### User-directed validation scope and parallel execution

The user requested Genesis first and only a basic SNES run; repeated SNES periods,
both-side replay comparisons, and broader SNES validation are deferred. The
already captured SNES baseline remains archived. Optional checkpoint coverage is
limited to affected paths. For the forecast correction, the 17 profiles are
lookahead, uncertain carry, chance creation, crossing, deke, possession value,
all five possession ablations, all three receiver selectors, and pass-timing,
carry-motion, and finishing refinements. Their distinct forecast consumers call
the corrected builders. Unchanged goalie construction, strict cadence, retention
artifacts, and reception/rebound/shot-hold execution need no repeated optional
matrix at this checkpoint; common offense remains covered by core/lifecycle runs.

The user then requested all cores minus one. This machine exposes 16 available
CPUs, so subsequent benchmark and replay matrices use 15 workers. The running
single-worker forecast gate was interrupted after cadence 1 completed: all 40
result rows are identical to the original, including every gameplay and diagnostic
field. Its partial cadence-4 run is excluded, with interruption recorded in
`after-forecast/run-manifest.json`. New parallel runs use fresh output labels.
The frozen original gets a full parallel reproducibility run before the candidate
is compared. The comparator requires explicit worker counts for each side and
marks timing incomparable when they differ; it continues to reject every other
settings mismatch. This changes scheduling, not the playing-strength gates.

### Stage 1b accepted reference

The parallel baseline reproduces all 80 original periods exactly in 146.16 wall
seconds across the two profiles. The forecast candidate also preserves all 80
core periods exactly:
[`core comparison`](../tmp/agents-refactor/2026-10-21/after-forecast-parallel/comparison-original.json).
All 68 affected optional periods complete. Sixteen profiles (64 periods) remain
exact; only pass timing changes. Sabres GF/GA changes from 5/2 to 4/0, and Ducks
from 3/1 to 4/1: each team's goal differential increases by one, with no conceded
goal increase. The prespecified extension rule is not triggered. Full paired
metrics remain in the
[`optional comparison`](../tmp/agents-refactor/2026-10-21/after-forecast-optional-parallel/comparison-original.json).

All four changed trials were traced on both frozen original and candidate with
identical instrumentation. Each of the eight traces reproduces every saved
uninstrumented result field, action hash, and frame count. First applied action
changes occur at Sabres seed 8000/frame 2895 (carry → feint), Sabres 8001/frame 305
(feint → carry), Ducks 8000/frame 5690 (feint → carry), and Ducks 8001/frame 1631
(carry → feint). Native inputs are identical at each first divergence. The changed
legacy projection feeds `pass_launch_view`, which uses precise coordinates in a
native movement step; synchronizing those coordinates changes forecast pass
opportunities. Ducks 8000 first changes proposal diagnostics at frame 5240 while
the same one-timer wins: retained feint history explains the later input change.
That proposal side effect must remain intact during structural extraction.

Sixteen lifecycle fixtures remain exact. The combined-refinement fixture changes
carry to feint at frame 6758, through the same corrected forecast path; its goals
and period length remain unchanged. Review evidence and trace links are indexed
in [`forecast-acceptance.json`](../tmp/agents-refactor/2026-10-21/forecast-acceptance.json).
The basic SNES FILTERED smoke passes 300 environment steps:
[`snes-basic-forecast.log`](../tmp/agents-refactor/2026-10-21/snes-basic-forecast.log).
Further SNES validation remains deferred. These runs establish the accepted
forecast behavior for subsequent structural stages; original evidence remains
permanent. No playing-strength improvement is claimed.

### Stage 2 implementation, implementation and gate evidence

`ClassicOptions` captures immutable normalized configuration while retaining
unspecified goalie positioning and decision cadence separately from explicit
classic positioning, one-frame timing, and disabled manual goalies. Playback still
resolves file/default overrides before validation. The common validator is used
by direct construction, environment configuration, and CPU benchmarking. Legacy
direct callers intentionally permit lookahead, uncertain carry, and refinements
against supplied states despite a reduced-team CLI default; application validation
retains its existing full-team restriction. Two existing tests caught this
boundary during implementation and pass after preserving it explicitly.

Offense and goalie factories remove construction decisions from the controller;
its initializer shrinks from 155 to 40 lines. Ordinary construction remains
Torch-free. Reset and resource loading are unchanged. Source-fingerprint lists
include both extracted modules, including archive-aware replay loading.
Five new option contracts cover immutability, defaults/overrides, legacy callers,
shared rejection before loading, and equivalent normalized benchmark/direct values.
The full suite passes 1,332 tests with the original expected failure, and full
15-worker lint passes at 10.00/10 (`options-unit-accepted.log`,
`options-lint-accepted.log`). The native checkpoint will cover eight core periods
and 21 affected construction/timing profiles: all goalie variants, the three
lookahead configuration modes, strict intervals, and intent cadences. Stage 3
must wait for those exact comparisons.

Stage 2 is accepted at `68f2a86`: the eight-period core screen is exact against
the accepted forecast reference, and all 84 periods across the 21 construction/
timing profiles are exact. Evidence:
[`core`](../tmp/agents-refactor/2026-10-21/after-options/comparison.json),
[`19 original profiles`](../tmp/agents-refactor/2026-10-21/after-options-optional/comparison-original.json),
[`two corrected classic-goalie profiles`](../tmp/agents-refactor/2026-10-21/after-options-optional/comparison-corrected-goalies.json).
The original single-worker optional reports remain gameplay references; their
elapsed times are not compared with the parallel runs. Before changing history
transfer, a focused native fork probe will additionally capture existing
`copy_history` and chance-replay cloning against live Genesis states.

### Stage 3a history transfer, implementation and gate evidence

The native before-change history probe captured 159 forks across four complete
Genesis periods (default/chance creation, Sabres/Ducks, seed 8000). Both existing
replay copy paths reproduce the live action/decision/target, and every complete
result matches its saved uninstrumented report:
[`before-history-forks/validation.json`](../tmp/agents-refactor/2026-10-21/before-history-forks/validation.json).
The owner/field inventory is archived as `history-field-inventory.json`.

`ClassicAIV1Model` now exports/imports explicit in-process history and forks through
that contract. Named owners declare their fields; dataclass lifecycles already
supply their field contract. Existing target executors and environment handles
remain owned by the target. Diagnostic aliases are preserved within each branch,
mutable histories are isolated between branches, and bound planners point to the
branch's offense controller. Injected decision observers are excluded. Evaluation
keeps a narrow adapter for archived layouts instead of copying object dictionaries.
The goalie outcome replay uses the same contract. No live transition or reset
logic changes in this commit; resource loading remains separate.

Six new contracts cover complete owner declarations, target implementations,
alias/branch isolation, held frames and learned command history, overridden feint
retry history, and previous-decision-dependent chance cancellation. All 1,338 unit
tests pass with the original expected failure (`history-unit.log`); lint passes at
10.00/10 (`history-lint-final.log`). The pending gate uses the eight-period screen,
the same four native fork periods, and all 17 accepted lifecycle fixtures. The full
core suite remains required for Stage 3b's reset change.

Stage 3a is accepted at `7f73c64`: the
[`eight-period screen`](../tmp/agents-refactor/2026-10-21/after-history/comparison.json),
[`159 native history forks`](../tmp/agents-refactor/2026-10-21/after-history-forks/comparison.json),
and [`all 17 lifecycle fixtures`](../tmp/agents-refactor/2026-10-21/after-history-lifecycle/comparison.json)
are exact against their accepted references.

Before changing reset/resources, an additional native probe reused one scripted
stream for two complete episodes on each team, individually covering default,
retention tree, retention ranker, fixed shot hold, and neural goalie. All 20
periods reproduce the existing uninstrumented matches exactly:
[`before-reset-streams/validation.json`](../tmp/agents-refactor/2026-10-21/before-reset-streams/validation.json).
The probe records two neural checkpoint loads per stream. Its total JSON read
counts are eight for the tree, six for the ranker, and four for shot placement;
these include benchmark fingerprint reads as well as controller loads. Stage 3b
will preserve each episode's outputs while eliminating repeated controller I/O.

### Stage 3b resource/reset implementation, implementation and gate evidence

Artifacts now belong to a frozen, per-stream resource bundle. JSON bytes supply
hashing and parsing once before retention schema dispatch. Artifact mappings and
model arrays are read-only. Neural weights are reused across episode resets and
replay forks; opponents load independent models and all command/history/metric
state remains isolated. Scripted reset still constructs a fresh controller,
removing injected observers, and reuses only its resources. Paths are pinned for
the stream; explicitly changing a path loads a new artifact. Retention feature
encoders, model outputs, and training are unchanged.

Five resource contracts and the existing suite pass: 1,343 tests with the original
expected failure (`resources-unit.log`), and full 15-worker lint is 10.00/10
(`resources-lint.log`). Ordinary scripted construction/reset remains Torch-free.
The pending gate includes the full 80 core periods, the same 20 two-episode native
reset periods, 159 history forks, and all 17 lifecycle fixtures.

Stage 3b is accepted at `5d2d0a4`: all 80 core periods, 20 reset periods,
159 history forks and 17 lifecycle fixtures match exactly. Evidence:
[`core`](../tmp/agents-refactor/2026-10-21/after-resources/comparison.json),
[`reset streams`](../tmp/agents-refactor/2026-10-21/after-reset-streams/comparison.json),
[`forks`](../tmp/agents-refactor/2026-10-21/after-resources-forks/comparison.json),
[`lifecycle`](../tmp/agents-refactor/2026-10-21/after-resources-lifecycle/comparison.json).
Each stream now loads its neural checkpoint once. JSON totals fall to three
(one controller load plus two benchmark fingerprint reads) for each tree, ranker,
and shot-placement stream. No shared mutable command or episode state was found.

Before Stage 4, `before-learned-execution.json` additionally captures 84 fixed
command/teacher histories: every command on both team perspectives at intervals
1/4, plus both teacher scopes. These complement the existing unit, checkpoint,
optional and native lifecycle baselines; they pin masks, labels, consumption,
button arrays, targets, ownership, request records and both clocks.

### Stage 4 execution extraction, implementation and gate evidence

`_attack` now states the priority in 18 lines: evaluate proposals in the original
order, finishers, one-timer, opportunistic crossing, normal shot, selected pass or
carry, then legacy setup/fallback. `AttackSelection` explicitly documents that
proposal evaluation is stateful; overridden feint/chance updates remain intact.
Selection and bounded execution helpers stay together in `classic_v1.py` to avoid
navigation through a new strategy hierarchy. Classic and learned commands share
pass, one-timer and shot starts, retaining their different follow-through budgets,
button admission, teacher labels, compact learned request records and consumption.
Goalie faceoff and held-clock advancement, setup cancellation/start, and displaced
chance cleanup now belong to their state owners. Observation and release/reception
logic is unchanged.

All 84 captured learned command/teacher histories (3,360 frames) match exactly:
[`comparison`](../tmp/agents-refactor/2026-10-21/learned-execution-comparison.json).
The full suite passes 1,343 tests with the original expected failure and lint is
10.00/10 (`execution-unit-accepted.log`, `execution-lint-accepted.log`). A first
focused command named a nonexistent test module; the full run resolved coverage.
The first full run exposed a clock-only SimpleNamespace test double: it now uses
an actual GoalieController and additionally asserts the release deadline remains
unchanged on held frames. No assertion was weakened.

The completed executor change uses the full gate directly, rather than redundant
8-period screens between adjacent mechanical extractions: 80 core periods, all
44 individually compatible optional profiles (176 periods), 159 native forks,
and 17 lifecycle fixtures. All were captured before these paths changed. Only
Genesis native coverage is expanded; SNES remains a basic final execution smoke.

Stage 4 is accepted at `ee3becf`: the full core matrix and all 176 optional
periods match exactly, including every deterministic gameplay/diagnostic field.
All 159 history forks and 17 lifecycle fixtures are also exact. Evidence:
[`core`](../tmp/agents-refactor/2026-10-21/after-execution/comparison.json),
[`forecast consumers`](../tmp/agents-refactor/2026-10-21/after-execution-optional/comparison-forecast.json),
[`remaining original profiles`](../tmp/agents-refactor/2026-10-21/after-execution-optional/comparison-original.json),
[`corrected classic goalies`](../tmp/agents-refactor/2026-10-21/after-execution-optional/comparison-corrected-goalies.json),
[`forks`](../tmp/agents-refactor/2026-10-21/after-execution-forks/comparison.json),
[`lifecycle`](../tmp/agents-refactor/2026-10-21/after-execution-lifecycle/comparison.json).

Before Stage 5, all 16 Genesis teammate fixtures (both teams, 2/3 bots,
independent/coordinated/pressure-off/support-options) also match the original
native results and complete input traces:
[`before-mechanics-teammates/comparison-canonical.json`](../tmp/agents-refactor/2026-10-21/before-mechanics-teammates/comparison-canonical.json).
The extracted Genesis-only driver initially compared in-memory tuples against
saved JSON arrays in pressure-role diagnostics. The original failed comparison is
retained; v2 compares persisted JSON consistently and validates every complete
trace/hash. No result field is ignored beyond the existing timing/source exclusions.
The v2 driver will be used for subsequent teammate checks; SNES fixtures are not
included, as instructed.

### Stage 5 mechanics/forecast extraction, implementation and gate evidence

Console motion, grounded skating, pass arithmetic/contact and shot release/speed
now live in `game/`. Tactical lane cover, shot scoring, pass admission and policy
choices remain in agents. All 40 extracted physical function bodies have identical
ASTs (actor predicate export names excepted):
[`mechanics-extraction-ast.json`](../tmp/agents-refactor/2026-10-21/mechanics-extraction-ast.json).
Backends no longer import agents; Genesis implements the existing shot contract,
removing its temporary abstract-method lint suppression. SNES application/policy
validation moved to the application option boundary with its existing restrictions.
Compatibility imports remain available, including `passing.rom_direction`.

`controllable_skater` and `observable_body` distinguish actor purposes. The legacy
retention encoder deliberately keeps the former, including its goalie exclusion.
Forecast models share actor/puck cloning and coherent position updates while
retaining their conservative, coasting, sampled and CPU-response assumptions,
uncertainty labels, engine sharing, supplied carrier aliases and puck velocity
rules. Teammate forecasts retain their distinct fractional-coordinate translation.
Hardcoded source manifests now include the shared files and platform dependencies;
archive-aware manifests tolerate old source layouts. Tests patch moved ROM
primitives at their new owning module without changing expected results.

All 1,346 tests pass with the original expected failure and full 15-worker lint is
10.00/10 (`mechanics-unit-accepted.log`, `mechanics-lint-accepted.log`). The first
run found a missing private backend-dispatch import in tactical pass admission;
restoring that explicit dependency resolved the failures before native validation.
Three new contracts pin actor distinction, forecast isolation/precision, and the
absence of platform-to-tactic imports.

This mechanical stage uses the eight-period checkpoint, five affected forecast/
replay profiles (20 periods), the native carry replay gate, 159 live history forks,
all 16 teammate fixtures and one basic SNES run. The broad optional executor matrix
just passed Stage 4; its full repeat remains required at final validation. Before
Stage 6, an observational probe will additionally hash every decision snapshot in
ten full Genesis periods (default, chance, manual goalie, strict held frames,
retention; both teams), verifying each match against the existing native evidence.

Stage 5 is accepted at `1e29af8`. The
[`core screen`](../tmp/agents-refactor/2026-10-21/after-mechanics/comparison.json),
[`20 forecast-profile periods`](../tmp/agents-refactor/2026-10-21/after-mechanics-optional/comparison.json),
[`complete native carry replay`](../tmp/agents-refactor/2026-10-21/after-mechanics/carry-replay.json),
[`159 history forks`](../tmp/agents-refactor/2026-10-21/after-mechanics-forks/comparison.json), and
[`16 teammate fixtures`](../tmp/agents-refactor/2026-10-21/after-mechanics-teammates/comparison.json)
are exact. The wheel built successfully (`mechanics-wheel.log`) and the SNES
300-step FILTERED smoke passed (`mechanics-snes-basic.log`). Further SNES validation
remains deferred. The pre-Stage-6 snapshot probe captured 86,578 snapshots across
ten full periods; all ten instrumented matches reproduce the saved uninstrumented
results exactly (`before-snapshots/validation.json`).

### Stage 6 shared workflows/evidence, implementation and gate evidence

Immediate and lookahead offense now share feedback initialization, close-finish
checks, breakaway plan construction, worthwhile-pass annotation, the ordered
cut/pass/feint sequence, and fallback targets. Each policy still visibly controls
when filtering/forecasting happens, its scoring/admission rule, opportunity value,
extra fallback candidates and retained-carry/chance fallback. In particular,
lookahead still clears risk first and filters before close-finish/breakaway;
immediate offense still performs those checks before filtering. No generic
strategy framework or growing switch list was introduced.

`OffensePlan` names the selected proposal while preserving the existing tuple
interface and dictionary evidence. Execution producers record `SelectedDecision`,
and pass/one-timer lifecycles return explicit action IDs. Snapshot construction
reads this evidence rather than parsing live phase names and pending commitments.
The narrow legacy adapter remains for old archives and explicitly injected legacy
diagnostics. A regression makes private commitment access and legacy parsing fail
while a current producer's snapshot still succeeds. Evidence participates in the
explicit history contract; held snapshots keep their existing observation boundary.
`ScriptedAgent` documents its act/frame/compatibility contract. Its forwarding
adapter is retained because existing playback/evaluation callers still use it;
removing it would expand the public API migration without simplifying this path.

Focused offense/history tests pass (104), focused inspector/learned tests pass
(43), and the full suite passes 1,347 tests with the original expected failure
(`workflows-unit.log`). Lint is 10.00/10 (`workflows-lint.log`), and the source
manifest contracts pass after adding the evidence dependency. The pending gate
checks all 86,578 captured snapshots first, then the core screen, all 32 offensive/
timing profiles (128 periods), and 159 native history forks. Goalie-specific
construction profiles already passed Stage 4 and remain in the final full matrix.

Stage 6's first snapshot gate failed and halted before the core/optional matrix.
All ten gameplay results remained exact, but eight reactive streams had snapshot
hash differences; both strict-hold streams were exact. Comparing current produced
evidence against the legacy observer on each live frame located the first
mismatches at Sabres frame 174 and Ducks frame 179 (`snapshot-first-differences.json`).
Native defensive handoff clears defensive diagnostics while retaining the tactical
phase; the new producer had retained the previous defensive execution ID/mode.
It now explicitly releases that evidence at the same existing handoff boundary.
A two-frame regression pins the neutral input, retained plan, cleared execution ID
and offense mode. All 1,348 tests and lint pass (`workflows-unit-corrected.log`,
`workflows-lint-corrected.log`). Failed reports remain intact; the corrected
snapshot run uses a new `after-workflows-snapshots-v2` output before any stage advances.

Stage 6 is accepted at `e6e32c0`: all 86,578 snapshots match exactly after the
handoff correction, as do eight core periods, 128 periods across the 32 affected
profiles, and 159 native forks. Evidence:
[`snapshots`](../tmp/agents-refactor/2026-10-21/after-workflows-snapshots-v2/comparison.json),
[`core`](../tmp/agents-refactor/2026-10-21/after-workflows/comparison.json),
[`optional`](../tmp/agents-refactor/2026-10-21/after-workflows-optional/comparison.json),
[`forks`](../tmp/agents-refactor/2026-10-21/after-workflows-forks/comparison.json).

### Stage 7 cleanup and final gate

Repository-wide reference checks confirm the four listed multi-model members and
the coordinator's duplicated pass-owner/count fields are unused. They are removed;
`GotoTarget`, observation targets, and SharedReceiving's real history remain.
One test's assignment to the removed fields is deleted; its assertions remain.
A shared `apply_pad` replaces six copies of additive direction encoding without
changing direction tables, action IDs, rotations, or existing button presses.
Unused tactical imports are removed; physical and registry compatibility exports
are explicit and retained. Focused lint distinguishes these intentional exports
from dead dependencies. Replay source manifests include the shared helper.

No optional directory reshuffle is performed: the new physical, option/resource,
and history owners already provide useful boundaries, while moving established
tactical modules would add compatibility shims without shortening execution paths.
The final full 80-period core suite and all 44 optional profiles will also serve
as this stage's gameplay gate, followed by both-team teammate and learned checks,
a basic SNES run, packaging, and the final readability audit.

### Final acceptance (Stages 7 and final validation)

Stage 7 is accepted at `112bfa3`. The complete result index is
[`final-acceptance.json`](../tmp/agents-refactor/2026-10-21/final-acceptance.json);
exact commands and their order are retained in
[`final_gate.py`](../tmp/agents-refactor/2026-10-21/final_gate.py) and each run manifest.
The final full matrix satisfies both the cleanup gate and final gate, avoiding an
unnecessary additional eight-period screen. All native runners use 15 workers;
optional profiles run in bounded batches of three four-trial profiles, never more
than 15 active trials. All original baselines and failed attempts are preserved.

- All **80 core periods** reproduce the original action hashes, frames, scores,
  shots, pass outcomes, one-timer results, goalie/defense metrics and diagnostics:
  [`original comparison`](../tmp/agents-refactor/2026-10-21/after-final/comparison-original.json),
  [`accepted reference comparison`](../tmp/agents-refactor/2026-10-21/after-final/comparison-accepted.json).
  Every period completed with the correct lineup, controlled side and CPU opponent.
- All **44 optional profiles / 176 periods** match the accepted structural reference:
  [`all optional comparisons`](../tmp/agents-refactor/2026-10-21/after-final-optional/optional-comparison-accepted.json).
  Against the original reference, 41 profiles / 164 periods are exact
  ([`comparison`](../tmp/agents-refactor/2026-10-21/after-final-optional/optional-comparison-original-exact.json));
  the two explicit-classic goalie profiles match the corrected runner baseline
  ([`eight periods`](../tmp/agents-refactor/2026-10-21/after-final-optional/optional-comparison-corrected-goalies.json)).
  Only the four previously accepted pass-timing trials differ from the original
  ([`behavior comparison`](../tmp/agents-refactor/2026-10-21/after-final-optional/optional-comparison-original-fix.json)).
  They exactly match Stage 1b, retain its explained first divergences and outcome
  deltas, and trigger no extension rule. This is not a new behavior acceptance.
- All **16 Genesis teammate fixtures** and their full input traces are exact on
  both teams, including two/three controllers, independent/coordinated behavior,
  pressure disabled and support options:
  [`comparison`](../tmp/agents-refactor/2026-10-21/after-final-teammates/comparison.json).
- All **17 lifecycle fixtures** reproduce complete traces and deterministic results:
  [`comparison`](../tmp/agents-refactor/2026-10-21/after-final-lifecycle/comparison.json).
  Stage 6 additionally established exact parity for 86,578 decision snapshots and
  159 native history forks before the final mechanical cleanup.
- All **84 learned command/teacher histories / 3,360 frames** match their baseline
  ([`comparison`](../tmp/agents-refactor/2026-10-21/final-learned-comparison.json)).
  Both existing neural artifacts reproduce baseline actions and probabilities
  across two episode resets
  ([`checkpoint comparison`](../tmp/agents-refactor/2026-10-21/final-checkpoint-comparison.json)).
- The full unit suite passes **1,348 tests**, with the same pre-existing expected
  failure, `unit.test_environment_contracts.EnvironmentContracts.test_frame_skip_stops_on_truncation`:
  [`unit log`](../tmp/agents-refactor/2026-10-21/final-unit-accepted.log).
  Full 15-worker lint and focused unused-import/variable lint both score 10.00/10:
  [`full lint`](../tmp/agents-refactor/2026-10-21/final-lint-accepted.log),
  [`focused lint`](../tmp/agents-refactor/2026-10-21/final-unused.log).
  Initial cleanup lint rejected redundant import aliases; scoped compatibility
  export annotations resolved it before native benchmarking. No tests were removed
  or assertions weakened; the one dead teammate-field setup assignment was removed.
- **SNES runs:** the 300-step FILTERED smoke passes
  ([`log`](../tmp/agents-refactor/2026-10-21/final-snes-basic.log)).
  Further SNES checks remain deferred as instructed.

Final packaging inspection caught a stale `training/es.py` from the pre-existing
ignored `build/` directory in the first wheel. A clean copy of tracked candidate
source was built in a new ignored directory, without deleting unrelated build
contents. The second wheel contains exactly the current 235 Python source files,
including every new module, with identical bytes. Importing the extracted wheel
from an isolated working directory verifies compatibility exports and scripted
construction/reset without loading Torch:
[`clean build`](../tmp/agents-refactor/2026-10-21/final-wheel-v2.log),
[`source/package verification`](../tmp/agents-refactor/2026-10-21/final-package-verification.json),
[`isolated smoke`](../tmp/agents-refactor/2026-10-21/final-package-smoke.log).
The original wheel and evidence remain available; only the clean wheel is accepted.

The four-row final benchmark table above reports W/D/L and goals. Additional
aggregate counters below are identical before and after; complete pass outcomes,
ordinary-pass/reception metrics, one-timer lifecycle metrics, goalie/defense
counters, hashes and individual frame counts are in the paired comparisons.

| Cadence | Agent team | Frames | Shots for/against | One-timers for/against | One-timer goals for/against |
| --- | --- | --- | --- | --- | --- |
| 1 | Sabres | 174,429 | 114/58 | 48/14 | 16/3 |
| 1 | Ducks | 180,746 | 129/79 | 49/18 | 21/1 |
| 4 | Sabres | 187,454 | 155/31 | 66/20 | 31/3 |
| 4 | Ducks | 175,964 | 121/89 | 39/26 | 16/4 |

### Runtime review

The matched 15-worker original core run took 90.30 / 55.86 seconds at cadences
1 / 4; the final run took 100.44 / 61.00 seconds. Total wall time increased from
146.16 to 161.44 seconds (**10.45%**). The original single-worker times are not a
performance reference. This increase was investigated with eight isolated native
profiles: original and final source, both teams, both cadences, seed 8000. Every
profiled match reproduced its saved uninstrumented result exactly. The profiling
pool uses 15 workers and ran after 16 of 17 lifecycle fixtures had finished, with
at most nine active trials across the two probes.

Aggregate profiled process CPU time increased only **0.28%**, and function calls
increased **0.006%**. This bounded investigation did not reproduce the full-suite
wall-time increase as corresponding CPU-work growth; machine/runtime variance is
plausible but not established as its sole cause. Retain the observed wall-time
cost as a limitation, rather than claiming a speedup or per-decision latency:
[`matched-worker timings`](../tmp/agents-refactor/2026-10-21/after-final/comparison-original-parallel.json),
[`runtime review`](../tmp/agents-refactor/2026-10-21/final-runtime-review.json),
[`paired profiles`](../tmp/agents-refactor/2026-10-21/final-runtime-profiles/comparison.json).
No gameplay regression, extra artifact loading or growing retry/history state
was found in the required gates. Further performance optimization is not needed
to establish structural parity and is not mixed into this refactor.

### Final readability and ownership review

The same AST inventory reproduces the original measurements (ties in equal-size
function ordering are immaterial). Review artifacts:
[`before`](../tmp/agents-refactor/2026-10-21/before-structure.json),
[`after`](../tmp/agents-refactor/2026-10-21/after-structure.json),
[`dependency/ownership audit`](../tmp/agents-refactor/2026-10-21/final-readability-audit.json).

| Responsibility | Before | After |
| --- | --- | --- |
| Classic initialization | 155 lines mixing validation, artifacts and state | 5-line initializer; immutable options/resources, factories, explicit 41-line episode-state setup |
| Classic attack | 203 lines mixing proposal evaluation and button/lifecycle starts | 18-line priority sequence; 71-line ordered stateful selection and bounded local execution helpers |
| Lookahead offense choice | 67 lines | 35 lines; forecast/filter order remains explicit |
| Immediate offense choice | 52 lines | 22 lines; immediate admission remains explicit |
| Shared offense work | Repeated feedback, close finish, breakaway, pass annotation, cut/pass/feint and fallback logic | Six common workflows, with policy-specific ordering/scoring visible at the caller |
| Learned execution | Duplicate pass/one-timer/shot start operations | Shared lifecycle entry points preserving masks, labels, consumption and timing |
| Direction encoding | Six copies across Classic, learned and reception paths | One additive `apply_pad`; original action tables and IDs unchanged |
| Component-assignment sites in Classic/learned offense | 74 | 48; setup/goalie field clearing removed in favor of owner operations |
| Platform-to-agent dependency edges | 9 | 0 |

The remaining 48 assignment sites are reviewed public state: scheduler/cache
bookkeeping, applied button edges, proposal option/flight data, shot timing, and
component configuration/evidence. The count is not a claim that all mutable state
should disappear. Construction factories also set explicit options. Cancellation,
goalie faceoff/held advancement, setup start/cancel, displaced chance cleanup,
artifact lifetime and history transfer now have named operations and owners.

Resolving both finite dynamic backend targets exposes a former 15-module cycle
that crossed tactics, environment and physical modules. Its remaining 11-module
cycle is confined to `game/` dispatch/ROM mechanics. The existing two-module
state/features cycle remains. This is improved dependency direction, not a claim
that the entire package is acyclic. The longest remaining agent function is
teammate `act` (158 lines); decision snapshots remain 102 lines of observable UI
formatting. Neither was split merely to reduce a number. The proposal selector's
71 lines retain real stateful tactical ordering instead of hiding it behind a
framework or a series of forwarding modules.

One frame is now traceable through these boundaries:

1. `ScriptedAgent.act(AgentInput)` receives the observation. A strict held frame
   reuses the previous action and snapshot without observing fresh state;
   reactive execution enters `ClassicAIV1Model.predict_frame` with explicit cadence.
2. `_predict_native_frame` observes action feedback and stoppage, then checks
   goalie ownership, active cross-crease/deke execution, ordinary-pass feedback
   and reception correction. `_scheduled_frame` handles one-timer/shot ownership,
   scheduler caching and tactical cadence. Frame clocks and decision clocks
   remain distinct, including their original observation order.
3. `_attack` obtains stateful proposals, then visibly applies finisher,
   one-timer, crossing-shot, ordinary-shot, selected-plan and legacy fallback
   priority. `_select_attack` preserves updates made by overridden feint/chance
   proposals. Immediate/lookahead policies share the common workflow while
   retaining their admission and forecast order.
4. `_start_pass`, `_start_one_timer_pass` and `_start_shot` start the named
   lifecycle. `apply_pad` only adds directions; `_encode` records raw B/C edges
   and produces the selected action schema. A request does not imply release or
   reception: native feedback advances those phases in their respective owners.
   Coordinated input is reconciled through `accept_applied_action`.
5. Producers record `SelectedDecision`; the snapshot serializes that evidence
   without re-running alternatives or reading commitments to guess intent.
   Historical archives/injected legacy diagnostics retain a narrow adapter.
   `cancel_actions` delegates cancellation to offense/pass, one-timer, shot,
   setup and defense owners while preserving the intended clocks/retry history.
6. `ScriptedAgent.reset` creates fresh episode/controller state through
   `new_episode`, reusing immutable loaded resources. Explicit history export,
   import and fork isolate mutable replay branches and retain their own executors.
   Learned adapters retain recurrent state within a stream and clear it on reset.

The existing `ScriptedAgent.__getattr__` and public physical re-exports remain
intentional compatibility surfaces. Their removal and optional directory moves
would add migration work without improving this execution walkthrough.
Action table definitions and legacy retention encoder function bodies also match
the original ASTs after the predicate's owning-module rename:
[`artifact contracts`](../tmp/agents-refactor/2026-10-21/final-artifact-contracts.json).
Retention goalie-feature correction and retraining remain wholly deferred.

All generated reports, profiles, source archives and wheels stay in the ignored
run directory. Implementation/behavior/tooling changes are separated in the
recorded commits. The original pre-existing plan diff remains archived and its
unrelated content is preserved in this completed plan.
No production or test source changed after the final candidate was frozen.

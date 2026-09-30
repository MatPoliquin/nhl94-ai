# Architecture

The repository is `nhl94-ai`; its importable Python package is `nhl94_ai`.
There is no `src/` directory. The root `models/` directory contains model assets;
`nhl94_ai/models/` contains Python architecture definitions.

| Package | Responsibility | Extension point |
| --- | --- | --- |
| `game` | Variants, RAM information decoding, geometry and features | `specs.GAMES`, fixture tests |
| `env` | Environment assembly, observation encoding, action timing, opponent perspective | `factory.build_single_nhl94_env`, `encoding.ObservationEncoder` |
| `tasks` | Episode setup, rewards, completion and action restrictions | `registry.register_task(name, TaskDefinition(...))` |
| `agents` | Scripted controllers and learned/multi-model adapters | `Agent.reset()` and `Agent.act(AgentInput)` |
| `models` | MLP, temporal, vision, combined and hockey networks | `factory.register_model(name, builder)` |
| `training` | PPO/ES, BC, DAgger, collection, curriculum and progress events | Services below; `curriculum.PHASE_RUNNERS` |
| `evaluation` | Headless matches, totals, playback and comparison orchestration | `runner.evaluate` |
| `ui` | Pygame views, recording and terminal progress | Consumers of training events/state |
| `cli` | Argument parsing and configuration precedence | `nhl94 --help` |

`game` imports no UI or training code. Display and export dependencies load only
when their features are requested. Environment bindings are ordinary objects;
there is no mutable singleton wrapper manager.

## Services

- `training.service.train(config)` returns a `TrainingResult`; it accepts an
  options dictionary, `RunConfig`, or resolved namespace. Headless and live modes
  share the trainer, callbacks, configuration, and environment factory.
- `evaluation.runner.evaluate(agent, EvaluationConfig(...), env=env)` returns
  rewards, lengths, termination/truncation flags, and budget exhaustion flags.
  The caller owns and closes the supplied environment. Live monitoring uses the
  same `EvaluationConfig` type with the original unlimited episode horizon.
- `nhl94 benchmark` runs paired scripted-agent periods with ROM RNG seeding,
  both team assignments, and goal-based results. See [Classic V1](CLASSIC_V1.md)
  for the benchmark protocol and measured scope.
- `nhl94 benchmark-cpu` runs seeded periods against the in-game opponent,
  including Ottawa and Montreal/Roy fixtures, with verified roster names and
  actual one-timer counters. It supports filtered buttons and hockey intents.
  Both benchmarks use `evaluation.pass_outcomes.PassOutcomes` to record deliberate
  V1 pass outcomes at emulator-frame cadence, separately from team shot totals.
- `training.collect.collect_demonstrations(agent, config)` writes a dataset and
  returns its path and metadata. Supplied agent instances use one worker and
  reset between episodes. Registered teacher names support multiple workers.
- `training.bc.train_bc(args, hyperparams)` and `training.dagger.run_dagger(...)`
  are reusable services; DAgger invokes BC directly.
- `training.curriculum.run_curriculum(path, dry_run=False)` resolves and validates
  phase settings and chains artifacts. Historical two-value unpacking remains
  supported by its result type. Phase-specific chaining and result reporting
  remain explicit in the orchestrator.

Classic V1 is registered as `ClassicAIV1` / `classic-v1` (also `classic`).
It retains the former V4 offensive decision tree and adds reactive defense in
`agents/defense.py`: select a tactical target, select a reachable skater, then
execute skating/checking. `agents/motion.py` shares arrival/puck prediction
between planning and selection. Read-only live attributes and motion are decoded
by `game/ram.py` and `game/state.py`, outside the neural input schema.
The original V1–V3 controllers and their legacy import aliases are removed;
`classic-v2`, `classic-v3`, and `classic-v4` are no longer supported names.
Historical benchmark reports retain their original version labels and hashes.
The paired benchmark now defaults to V1 versus its `classic-v1-direct`
one-timer-disabled ablation. Both variants use the same new defense.
Agent services reset the controller and pending sequences between episodes.

Play, evaluation, collection, DAgger, and inference-only guards consume the same
agent registry, so adding a scripted controller does not require extending
separate name lists in each command.

## Configuration

Precedence is parser defaults, JSON file, then explicit CLI overrides.

```json
{
  "options": {
    "env": "NHL94-Genesis-v0",
    "nn": "ResidualMlpPolicy",
    "rf": "PostPlay",
    "num_env": 1,
    "num_timesteps": 10000,
    "hyperparams": "../training/nhl94.json",
    "output_basedir": "./runs",
    "seed": 7,
    "policy_seed": 7
  }
}
```

All declared file paths, including future outputs, resolve relative to the JSON
file. CLI paths remain relative to the current working directory. `--state` is
an emulator state name, not a filesystem path. Boolean values must be JSON
booleans. Training examples are under `configs/training/`; curriculum examples
are under `configs/curricula/`.

`GameSpec` distinguishes skaters per team from `num_players`, which is the
controller count (1 or 2). `EnvironmentConfig` validates game, action and
self-play combinations before starting emulator workers. Training and evaluation
have additional validated settings. Existing flags and reward names remain valid.

Editable examples live in `configs/`. Built-in training defaults are included as
package data under `nhl94_ai/data/training/` so wheel installations need no
checkout. Keep those copies synchronized when changing defaults. Dependencies
and optional extras are declared in `pyproject.toml`.

The NHL94 training configuration uses frame skip 4 for all policies. Its
`MlpPolicy` override uses separate `[128, 128]` actor and critic networks
(114,317 parameters with the current 310 inputs and 12 binary button outputs).
Other policies retain their existing architectures. To use the smaller MLP,
start a fresh run without `--load_p1_model`: loading a checkpoint restores its
saved architecture and PPO settings rather than applying the new network size.
The opt-in DefenseZone target example uses its own larger MLP configuration,
described below; it does not change these shared defaults.

## Timing and reset

The inner environment processes actions, steps one emulator frame, updates state,
computes task rewards/completion, then encodes observations. Frame skipping wraps
that environment and repeats the action. Macro timers therefore still advance in
emulator frames; a normal policy decision spans the configured frame skip.
Learned-button gameplay retains its existing explicit four-step loop.
Classic V1 retains that offensive decision interval, but `predict_frame` observes
each emulator frame and immediately replans defense on passes/turnovers. It
releases defensive inputs on friendly possession without starting an extra
offensive decision. Offensive cooldowns remain in decisions; defensive cooldowns
advance in emulator frames, including while the team has possession.

`agents.base.configure_scripted_frames` attaches a scripted agent to
`StochasticFrameSkip` only for explicit scripted evaluation/collection.
The first submitted action and subsequent reactive substeps share one controller.
Learned actions retain normal frame skipping. DAgger's teacher observes every
substep but never overrides the learner's submitted actions. Demonstrations
record actual per-frame observations, actions, rewards and targets rather than
pretending a reactive action was held for the entire interval; see
[artifact timing](ARTIFACTS.md#reactive-scripted-demonstrations).
Raw CPU/paired benchmarks and normal Classic playback call the same per-frame
controller. The learned `TARGET_POSITION` controller and its schema are unchanged.

Classic diagnostics use `classic_defense`, separate from `target_control`, with
target, mode/reason, ideal/desired/actual slots, predicted B selection, switch
status, pending request and last observed result, arrival estimates, receiver
prediction, missing-feedback fields, and submitted button requests. The shared target drawing
helpers show its green square in normal button/intent playback, using queued
pre-step scroll values rather than the next frame's camera. Reset clears stale
annotations. A requested switch is not proof that the ROM selected that skater.
Classic compares the current skater with the ROM-obtainable candidate, including
the possibility that B retains the current skater. Actual control feedback closes
each request; failed requests back off and retry rather than permanently disabling
switching. Read-only `Player.selection_flags` comes from object byte `+0x62`,
separate from the legacy animation flags, without changing normalized inputs.

Hockey-intent ownership gates prefer the engine's owner slot over visual stars
when `EngineState.puck_owner_known` is true. Negative owners mean a loose puck;
only absent ownership telemetry falls back to the historical star flags. This
prevents stale stars from suppressing defensive switches or sustaining offensive
macros after a turnover. It does not alter the observation or action schema.

`player_vs_model` uses the normal debug display, not the older PvP display.
The AI keeps controller 1; the display samples keyboard input for controller 2
on every emulator frame. The match environment alone exposes `MultiBinary(24)`,
ordered as 12 AI buttons then 12 human buttons. Its observation wrapper validates
and splits that action before applying home-task restrictions only to the AI,
and tracks the two controllers' button state separately. The policy-loading
environment remains single-controller with its original 12-button schema.
Explicit human opponents require `FILTERED` controls and a save with home/away
controller assignments `(1, 2)`. Single-controller playback, idle-opponent
collection and self-play retain their existing routing and action schemas.
`python -m tests.integration.human_play` checks the full playback/display/ROM path
in all three variants, including key releases, actual away-skater movement and
the visible defensive target.

Learner and opponent macros have separate state. Reset reconstructs game state,
clears macro cooldowns, initializes the task, and repopulates sequence history.
Opponent perspective copies the state, exchanges teams, and applies the existing
period-dependent rotation. Legacy truncation behavior is documented separately.

Live-viewer and evaluation totals count events **after the episode's reset
boundary**, not the counters inherited from its save. The baseline comes from
the VecEnv reset info, before the first policy action; terminal-frame events
still count even when the VecEnv has already auto-reset. In particular, the
DefenseZone save contains one pre-existing home pass, which must not be added
again every episode. `Team.begin_frame` decodes the legacy shot fields through
`game.ram.decode_shot_count`: `Stats.shots` and `last_stats.shots` always contain
the actual unsigned 16-bit counters, not neighboring RAM words. Rewards and
game-state displays compare/use these counts directly. Raw-info reset reporting
and the legacy game display use the same decoder. One-timer reporting retains
its existing low-word conversion.
The raw integration fields and saved-game RAM remain unchanged for compatibility.
Shot statistics are not neural input fields; observation ordering and saved-policy
input contracts are unchanged.

The live viewer replays the **latest evaluated snapshot**, not the historical
best model. Each scheduled evaluation refreshes a separate `_latest_live`
checkpoint and advances the viewer version even for equal or lower rewards.
The replay label shows the loaded snapshot's timestep count; the best reward
and reward history remain independent. Best-model selection, evaluation cadence,
and final-model saving are unchanged. Headless sessions do not write the extra
viewer snapshot. Snapshot writes and viewer loads share a lock, so the viewer
cannot read the ZIP or metadata sidecar partway through an update.

## Target-position control

`--action_type TARGET_POSITION` is an opt-in action adapter for full-team,
single-controller PPO `MlpPolicy`. `DefenseZone` is the first supported task.
It does not change rewards, initialization, default actions, existing button models, or
the tactics of the ClassicAI agents.

The policy emits `Box(-1, 1, shape=(2,), dtype=float32)`, mapped directly to the
task's permitted world-rink area. For `DefenseZone`, the bounds are
`x = -120..120`, `y = -270..-88`: **our defensive zone**, including the blue-line
boundary, for the supported first-period home-team save.

```text
world_x = 120 * action_x
world_y = -179 + 91 * action_y
```

Thus `(0, 0)` means `(0, -179)`, not center ice. Every policy output, including
exploration at the action-space extremes, already lies in the defensive zone;
out-of-zone choices are not merely collapsed onto the blue line by a clamp.
Targets are absolute positions in the same world coordinate frame as the entity
observations, not offsets from the selected player. Unbounded tasks retain the
original `(120, 270)` scaling.
The target training example uses a tanh-squashed Gaussian; older ordinary
Gaussian checkpoints retain SB3's clipping behavior when loaded. The adapter
rejects malformed, nonfinite, or out-of-range actions rather than interpreting
them as buttons. It projects destinations into conservative rounded rink bounds
and out of the net collision areas. A small visibility graph supplies temporary
waypoints around the nets; it never selects an alternative tactical destination.

`env.target_control.TargetPositionController` produces ordinary filtered
controller buttons. It estimates arrival time from distance and momentum,
retains a player unless switching offers a meaningful advantage, and uses brief
B presses rather than holding B for goalie control. **B cannot address a chosen
skater:** the ROM selects near the projected puck. The controller requests a
switch only when that choice appears useful, confirms the actual controller
slot on subsequent frames, and limits retries for the same desired skater.
Unavailable/fallen skaters and goalies are not steered.

Skating uses velocity feedback to slow near the destination. Boosts require
distance and facing alignment and have release frames/cooldowns. Pokes are local
requests near the puck, not permission to chase it. Possession suppresses the
defensive button sequences, preventing them from becoming passes or shots.
Other players remain under the ROM's AI. No gameplay RAM writes, direct player
selection, or teleporting are used by this controller.

The policy revises its target every configured `frame_skip` (four frames in the
example); the controller runs every emulator frame. A new target takes effect
on the next decision without waiting for arrival or resetting cooldowns.
Training and headless evaluation repeat targets through the existing frame-skip
wrapper. Playback and the live viewer use `FrameRepeatAgent` to retain that same
decision interval while rendering each frame. Target mode stops action repetition
on either termination or truncation; legacy modes retain their existing timing.

### Observations, diagnostics, and extension

Target mode retains the existing ordered entity fields, then appends these 12
controller fields (322 inputs with the NHL94 configuration):

```text
target_valid, target_x, target_y,
desired_slot_0, desired_slot_1, desired_slot_2, desired_slot_3, desired_slot_4,
switch_cooldown, boost_cooldown, poke_cooldown, switch_attempts
```

Target coordinates are normalized within the task's bounds; the desired skater is one-hot; timers and attempts
are normalized by their controller limits. Reset zeros this block. The block
exposes the previous requested target and the controller's retained selection
and timing state. Only this mode registers additional read-only RAM fields for
full signed velocities, eligibility, facing, and actual controller selection.
Its control and possession features use authoritative RAM, not stale stars.
Legacy action modes keep their existing observation length, ordering, and decoding.

Negative ROM controller slots mean **no player is currently selected**. Target
mode represents these as `Team.control == -1` / `actual_slot == -1`, with no
controlled-player flags and zero undefined controlled-relative features. It
releases all buttons while waiting for the ROM to restore a selection; it does
not substitute the goalie, retain a stale skater, or index the roster negatively.
The requested destination and cooldown state persist, and normal control resumes
once a real slot returns. Reward and terminal processing continue during gaps.
The display labels this state explicitly and target error is `None`, not a
fabricated zero-distance success. Out-of-team positive slots still fail.
This selection-gap handling does not change the input layout or controller
settings. Separately, changing the target area's coordinate mapping changes
the action contract, as recorded in checkpoint metadata.

Every target-mode reset/step provides `info["target_control"]`: requested target,
projected destination, routing waypoint, desired/actual/acting slot, emitted
buttons after task restrictions, execution state, target error, observed control
changes, and switch attempts. `acting_slot` identifies the player controlled
before the frame; `actual_slot` is the selection after it, which may automatically
change on a teammate's recovery. A switch/poke/boost label describes a **request**,
not a claim that the ROM executed it.
`selection_available` distinguishes a selected skater/goalie from a selection gap.
`bounds` reports the task's target rectangle, or `None` for unrestricted targets.

The live and debug displays share target annotations: green square for the policy
target, cyan marker for a projected destination, purple route, green controlled
skater, and cyan desired skater. The same hollow green square is overlaid on the
actual game image, tracking the camera and display scaling. Off-screen targets
are clipped, not moved to a misleading on-screen position; the full-rink diagram
still shows their location. No stale target is shown on an automatic reset.

Target-only read aliases `target_scroll_x/y` expose signed `Hscroll`
(`0xFFBD1E`) and `Vscroll` (`0xFFBD20`). The observation wrapper samples their
queued values **before** stepping the emulator to derive the camera for the
displayed frame: `camera_x = -64 - Hscroll`, `camera_y = -Vscroll`.
The returned info exposes these as `target_camera_x/y`. Reading current
`Hpos/Vpos` instead can put the marker ahead of the rendered ice during scrolling.
In the supported vertical view with the
standard `(0, 0, 256, 224)` crop, an ice position maps to native pixels as
`(128 + x - camera_x, 112 - y + camera_y)`. This follows `find3d` at zero height after
subtracting the Genesis sprite bias of 128. Camera fields are display metadata,
not additional neural inputs or controller settings.
The displays do not mislabel coordinates as button probabilities.
No display imports are needed for headless operation.

A future task can opt into the same mechanics using
`TaskDefinition(..., target_control="defense", target_bounds=(x_min, x_max, y_min, y_max))`.
Bounds belong to the task, not the controller's button profile.
`target_bounds=None` preserves the unrestricted whole-rink mapping.
`target_position` and `normalize_position` provide the shared forward/inverse
conversion for diagnostics and scripted target providers. Keep reward/setup separate;
a task requiring different possession/button semantics needs a deliberately
implemented controller profile rather than silently inheriting defensive actions.
The initial mode rejects self-play, human overrides, reduced-team variants,
other policy architectures, ES, ClassicAI/BC/DAgger, mixed-model playback,
comparison recording, and export to the separate runtime.

### Running target policies

Start a fresh model, not an existing button checkpoint:

```bash
nhl94 train --config configs/training/defense-target.json --live

nhl94 play --mode model_vs_game --env NHL94-Genesis-v0 \
  --state PenguinsVsSenators.DefenseZone --rf DefenseZone --nn MlpPolicy \
  --action_type TARGET_POSITION --hyperparams configs/training/defense-target-hyperparams.json \
  --model_1 /path/to/target-checkpoint.zip

nhl94 evaluate --env NHL94-Genesis-v0 --state PenguinsVsSenators.DefenseZone \
  --rf DefenseZone --action_type TARGET_POSITION \
  --hyperparams configs/training/defense-target-hyperparams.json \
  --model /path/to/target-checkpoint.zip --episodes 100 --seed 0
```

The example uses `defense-target-hyperparams.json` with separate
`[256, 256]` actor and critic networks: 297,733 parameters for the current
322 inputs and two target outputs, versus 116,101 with `[128, 128]`.
It opts into `squash_output: true` and `log_std_init: -0.6931471805599453`
(initial standard deviation 0.5) through the existing `MlpPolicy` model builder.
`models.mlp.SquashedMlpPolicy` retains the same MLP heads and learns two global
log-standard-deviation parameters, as ordinary PPO does:

```text
training action = tanh(mean(observation) + exp(log_std) * Gaussian_noise)
deterministic action = tanh(mean(observation))
```

Squashing happens after sampling. These normalized actions go through the
existing world-coordinate mapping; training, evaluation and playback use the
same transform. SB3's squashed Gaussian supplies the corrected log-probabilities.
The bounded entropy is estimated with fresh reparameterized samples using
`H(Gaussian) + E[log(1 - tanh(raw)^2)]`, evaluated in numerically stable form.
This avoids both rewarding unbounded latent spread and SB3's fallback entropy
estimate using old rollout actions. The entropy coefficient remains 0.01;
the standard deviation remains learned, not fixed or capped.
`train/std` reports the **pre-squash** standard deviation, not the spread in
world coordinates. Large means or variance can still saturate tanh; bounded
actions alone are not evidence of successful learning.

Other PPO hyperparameters, frame skip and ordered input fields match
`nhl94.json`. Keep those settings synchronized when updating the examples.
Shared and packaged defaults remain unchanged. `--nnsize` does not override an
explicit `net_arch`. These settings apply to fresh models; existing target
checkpoints retain their saved architectures and distributions and remain compatible.
Use a separate output directory for target runs. Target checkpoint metadata includes
the coordinate/feedback convention, target bounds, ordered controller inputs, profile, settings,
and version; incompatible or unverified target checkpoints fail explicitly.
An older full-rink target checkpoint is not interchangeable with bounded
DefenseZone actions even though both have two outputs and 322 inputs.
Changing controller mechanics after training requires versioning the contract.
See [Artifacts](ARTIFACTS.md).

Run `python -m tests.integration.target_control` for actual-ROM fixed-target
tracking and paired passive/scripted-target defense episodes. Its isolated
tracking task marks other skaters inactive only within that probe, and requires
eight skating/net-routing cases to settle within six rink units for 30 consecutive final
frames. The paired episodes use the unchanged full-team DefenseZone starts;
`--model /path/to/target-checkpoint.zip` adds a trained target policy on identical
seeds. Outcomes distinguish controlled-skater and teammate recoveries from
shot failures, goals and timeouts. Goalie catches are counted separately;
possession alone is nonterminal, but a newly recorded shot ends the episode.
The passive baseline can already earn a high team-recovery reward; execution
checks passing is not evidence that a neural policy has learned useful defense.
The ROM check also exercises 128 PPO training decisions with the squashed target
example, checks initial boundary concentration, saves/reloads the temporary model,
and compares its training/playback traces. This is a bounded plumbing check,
not a test of learned defensive quality; supplied `--model` checkpoints are not trained.
The check also injects an unselected-controller transition, verifies finite
observations and neutral controls, and follows the ordinary episode boundary.
Boundary and randomized actions are checked through the real wrapper to ensure
both requested targets and projected destinations stay inside the defensive zone.
The camera check compares projected world positions with visible faceoff-dot
pixels across scrolling game frames, without requiring display dependencies.

## DefenseZone reward

`DefenseZone` ends on friendly skater possession (+1), a newly recorded opponent
shot (-1), or an opponent goal (-1). Goalie possession alone gives no recovery bonus and does not end the
episode, including while the goalie holds the puck. A subsequent skater reception
counts as recovery; a loose puck after release does not. The existing clock cutoff
still applies while the goalie holds the puck.
An opponent shot or goal takes priority over possession in the same frame. Possession
uses the engine's puck-owner slot; older integrations without that field retain
the star-based possession fallback. A touch or a loose puck with stale stars is
not a recovery when engine ownership is available.

Shots are detected from a positive per-frame change in the game's shot counter,
not a shot-button press or every attempted shot.
The game-state decoder removes the unrelated upper word from legacy shot fields;
the reward and termination functions compare decoded counts directly.
Unchanged or decreased counts do not count as a shot.
Saved shot counts are the reset baseline, not new events.
The first new shot stops frame repetition immediately and returns exactly -1,
including on a skater recovery, goalie catch or timeout frame. A simultaneous
shot and goal also returns exactly -1, not -2; multiple counter increments in
one frame still represent a single terminal failure.

These are the only reward events. All other frames give zero reward:
there is no shot-lane shaping, survival bonus, puck-progress bonus, or reward
for friendly shots.
The `time < 200` clock cutoff is unchanged; timeout without recovery, a shot or conceding
adds no terminal bonus.

`SelfPlayDefenseFinetune` delegates to the same reward and completion rules.
The `--rf DefenseZone` flag, save-state names, action and observation schemas are
unchanged. Existing models still load, but their training objective has changed.
Carrying the recovered puck toward the attack zone is no longer part of this
task; a separate breakout policy/task is not implemented here.

### Defensive starting positions

For full-team `NHL94-Genesis-v0`, both defense tasks use one jointly sampled
formation, without stages or a snapshot collection. The carrier varies across
all five opposing skaters, from central to either wing (`x = -95..95`,
`y = -180..-125` in the first period). A primary defender starts 28-48 units
toward the home net, with up to 22 units of lateral displacement. A second
defender covers the slot; other skaters cover receivers, backcheck, or provide
attacking support and point options. Both goalies start in their creases.
The complete formation is resampled if any pair is less than 20 units apart;
failure to find a valid formation raises an error instead of accepting overlap.

The existing controller selection is retained. About 75% of starts place that
skater in the primary-defender role; the others require repositioning or
switching. No switching or steering is automated in the default button-action modes.

Reset writes full fixed-point current/previous positions and full-word velocities,
not just the legacy observation bytes. It clears inherited motion, contact
cooldowns and transient animations, rebuilds normal CPU role assignments, and
establishes opponent possession without a pending shot/pass. The carrier starts
with its normal awareness-based decision delay (at least four ticks), rather
than passing or shooting on the reset frame. The ROM advances the held puck to
the stick when the observation wrapper performs its usual reset frame.
The environment RNG drives both formation sampling and the ROM RNG seed.
These writes are local to the emulator instance; no Retro integration files or
model assets are changed.

Assignment scratch ends at `+0x49`: reset preserves the wall-collision radii at
`+0x4A/+0x4C` and clears the wall-contact vector at `+0x4E/+0x50` separately.
After all placements, it rebuilds `Ylist`, `OOlist` and `OOlistpos` for all 16
objects, including the untouched nets and shadow. Stale cached coordinates can
otherwise trigger collisions between distant players and amplify their velocity.
Cached goal-line crossings are invalidated for the stationary puck, and its
normal loose-puck timeout is restored.

Each team retains one `assnearest` assignment. The initial opposing carrier has
`asspuckc -> assnearest -> ordinary role`, so after a pass or rebound the ROM
transfers puck-handling AI to the next owner. Omitting that middle assignment
leaves receivers in support-player AI; they can evade their own carried puck
and spin instead of making carrier decisions.

Use the existing `PenguinsVsSenators.DefenseZone` save. Initialization requires
live first-period five-on-five play, a home skater selected, opponent skater
possession, and more than 200 clock seconds remaining; incompatible saves fail
explicitly. Lineups, ratings, scores, clock, rewards and input/output schemas
are retained. The reduced-player variants keep their previous initializers.
The geometry is a varied, defensible training distribution, not a guarantee
against an early goal or a model of every match situation.

Run the optional ROM-backed checks with
`python -m tests.integration.defense_starts` and
`python -m tests.integration.defense_physics`. The latter follows both passive
button-mode and randomized target-mode episodes beyond reset, checking motion
and CPU ownership handoffs rather than treating plausible initial geometry as
proof of correct subsequent behavior.

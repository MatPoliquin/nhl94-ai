# Artifacts

New checkpoints use SB3 `.zip` files with `.zip.json` metadata. Demonstration
shards use `.npz` plus `.npz.json`. `format_version: 1` records:

- resolved run options and hyperparameters;
- environment seed and optional PPO initialization seed (`null` preserves the
  old unseeded policy initialization);
- code revision, dirty-worktree status, Python-source SHA-256 fingerprint, dependency
  versions, and Python version;
- game, ordered observation groups, observation schema version, normalization,
  action type, sequence length and frame skip.

`artifacts.load_policy` and dataset loading compare known schemas when an expected
configuration is supplied. Equal tensor lengths do not make different field
orders compatible. Neural architectures may differ while sharing an input schema.

Named model inputs from `model_input.json` are resolved into the saved
hyperparameters, including their ordered groups and schema version; a registry
name alone is not the schema. The compact `pvg` input additionally records
`observation_layout: player-goalie-v1` and `normalization:
player-centered-rink-v1`. Schema version 4 has 52 inputs, including C-pressed
status and a hold-duration counter, and is incompatible with the prior 50-input
schema version 3 and 310-input PvG model. Use `--model_input pvg-v3` for the
50-input contract or `--model_input default` for the earlier 310-input contract;
start fresh to train with C feedback. The `default` and `legacy` registry entries
preserve the prior version-2 and version-1 field ordering respectively.

Older artifacts remain usable. Missing or old unversioned metadata is explicitly
`compatibility: unknown`; the loader does not infer an observation contract from
tensor dimensions. `require_metadata=True` rejects such checkpoints. Resumed runs
record the parent checkpoint's compatibility status. Existing checkpoint imports
such as `models.ResidualMlpPolicy` resolve through compatibility aliases installed
only during loading. Bundled model binaries have not been changed.

## Reactive scripted demonstrations

Classic V1 now reacts defensively each emulator frame while retaining the
configured offensive decision interval. Collection records each actual substep,
including button releases, instead of attaching one action to four different
frames. DAgger likewise labels each learner-visited substep; the teacher does
not take control of the learner's trajectory.

These shards retain the existing array fields and shapes. `steps` now indexes
emulator-frame samples within each episode. Their metadata explicitly records
`sample_interval_frames: 1`, `schema.frame_skip: 1`, and
`teacher_decision_interval` (normally 4). The requested `max_steps` still limits
outer decisions, so a four-frame run can collect up to four times that many rows.
Per-frame rewards/dones belong to the corresponding sample.

Dataset loading accepts these labelled substeps for BC at the recorded teacher
interval, while still validating field ordering and action/observation schemas.
It does not relax checkpoint frame-skip compatibility. Training a clone that
acts only every four frames cannot reproduce every reactive button edge; use
`--no_frame_skip` consistently in collection/BC/DAgger for a one-frame clone.
Existing datasets and saved policies keep their original timing contracts.

Progressive Classic offense uses the same action/observation schemas and
configured offensive decision interval. Optional live passing/rule feedback is
not encoded into neural inputs. `AgentOutput.diagnostics` and recorded substeps
add `classic_offense` alongside `classic_defense`; this is diagnostic metadata,
not another training-array field or a target-policy schema revision.
Old demonstrations remain compatible, but retain the old teacher's offensive
behavior rather than becoming labels for the new strategy.

### Experimental possession-policy artifacts

The separate `nhl94 offense` workflow uses `.zip.offense.json` sidecars with
ordered observations/actions, native execution cadence and a ZIP hash. Sidecars
record the algorithm (`PPO` or `DecisionPPO`) so reloading a decision model keeps
its variable-duration rollout buffer and timeout discounting. Older sidecars
without an algorithm field load as ordinary PPO.

PPO training also records `training_contract`: full period or finite attack,
reward definition, observation-clock meaning, four-frame gamma units, and
whether lambda/timesteps count decisions or fixed ticks. The attack task keeps
the same tensor ordering but uses remaining attack seconds in the clock field.
Evaluation requires an explicit `--allow-task-transfer` when moving between
task contracts. Initializing a new training objective from existing policy
weights is an intentional transfer; `--reset-value` resets its value output.
Native frames and active decisions are reported separately from SB3 timesteps.

Attack banks reuse trusted local emulator/controller snapshots with hash
sidecars; all snapshots and both team assignments from a source game stay in
one data partition. Snapshot files remain local, and benchmark reports identify
their source games and hashes for reproducibility.

The carry-only variant retains the 19 action IDs and original feature ordering,
but masks IDs 9 through 18. `training_contract.control_scope` is `carry` or
`offense` (missing means the historical full-offense scope). Loading into a
different scope requires `--allow-control-transfer`; task/clock transfer still
requires `--allow-task-transfer` for evaluation. Opportunity checkpoints include
the versioned reward and its weights, native-frame time cost, four-frame discount
unit and decision-based GAE trace. Native reward events are discounted inside
each policy interval before variable-duration transitions are combined.

Opportunity reports count only counter deltas after snapshot restoration. A
requested pass is an admitted setup; a fresh native one-timer counter plus the
intended shooter confirms execution. These are separate from recorded shots and
goals. Report `active_decisions` (eligible policy ticks) and `executed_commands`
(ticks that actually used external movement) separately because Classic may
consume an eligible tick by choosing a finish.

New offense saves include the policy's actual `parameters` count. PPO saves
record `initialization` (`random` or `checkpoint`), `parent_timesteps`, resolved
CLI options and the training contract. `--scratch` prohibits both `--model` and
`--data`; it initializes the complete actor, critic and shared extractor anew.
`--hidden-dim`, `--features-dim` and `--head-dim` configure fresh networks without
changing the observation/action schema or the architecture of loaded models.

`--checkpoint-every` is independent of historical-opponent snapshots. Its
immutable `step-*.zip` files include step zero and the final model. Intermediate
saves occur at the next rollout start, after the previous PPO update; SB3's
rollout-end hook precedes that update and would label stale weights incorrectly.
Sidecars retain the task/control contract, eligible decisions, executed movement
commands, native frames and completed optimizer epochs (`optimizer_updates`,
SB3's `_n_updates`). `checkpoints.json` publishes paths only after ZIP and sidecar
completion. The final `.training.json` also lists these checkpoints, while the
separate `*-logs/progress.csv` contains optimizer diagnostics.

## Supervised retention artifacts

The retention experiment exports a plain JSON decision tree with schema
`native-carry-retention-tree-v1`. It records ordered physical/action features,
four-frame decision cadence, 24-frame label horizon, tree nodes, minimum
baseline risk and risk-gap thresholds, source-game partitions, dataset hashes,
source fingerprints and dependency versions. The adjacent `.training.json`
records all fixed-grid development comparisons and the selected configuration.
Inference checks feature order, finite inputs and valid tree indices. Export is
checked against scikit-learn on all train/development rows. These artifacts are
separate from all neural-policy ZIP schemas.

Pairwise movement ranking uses the separate schema
`native-carry-retention-ranker-v1`. It stores a scalar base score, ordered boosted
trees, the same feature schema and a minimum within-situation score margin.
Higher scores rank movements as preferable; these scores are not loss
probabilities and cannot use the classifier's risk thresholds. Inference uses
float32 inputs and strict `<` split comparisons, matching XGBoost. All training
and development feature rows are checked against the original predictor before
an artifact is saved. JSON inference needs NumPy only; `.[ranking]` adds the
CPU XGBoost training dependency. The existing `--retention-model` option loads
either schema and preserves pass/shot priority and native reaction timing.

Ranking labels compare alternative movements from the same snapshot. Censored
branches contribute no label; fully censored situations contribute no game
weight. Situations with identical outcomes have no unequal-label training pairs,
but remain in policy evaluation so harmful or unnecessary overrides count.
The adjacent training report records the fixed model/margin grid, training and
development pair-order accuracy, actual policy outcomes and export parity.

Retention datasets keep every alternative from a snapshot and both controlled
sides of a physical save/ROM seed in one partition. Future branch outcomes are
labels only; filtering and movement admission use current-state features. A
pass or shot ending carrying exposure does not certify that the release led to
a good attack.

## Live-training snapshots

Live-display runs keep a separate `_latest_live.zip` and `.zip.json` sidecar,
overwritten after every scheduled evaluation regardless of its score. The
viewer loads this snapshot and labels its training timestep count.
The existing `_best_live.zip` still changes only for a strictly better evaluation,
and the final checkpoint remains the completed training model. Headless runs
retain their existing best/final outputs without writing a viewer snapshot.

## Target-position policies

`TARGET_POSITION` checkpoints add `schema.target_controller`: coordinate and
authoritative-feedback conventions, the ordered appended observation fields,
controller profile, version, and fixed settings. Their two-float action space
and augmented observations are not interchangeable with button policies.
Loading for target-mode training/evaluation requires verified metadata; a legacy
checkpoint cannot be silently reinterpreted as a target policy. The current
controller contract is also checked when loading a target checkpoint without an
expected environment configuration. Ordinary legacy button-model loading is
unchanged.

The target example's `squash_output` and `log_std_init` settings are recorded in
the resolved hyperparameters. Its SB3 ZIP stores the `SquashedMlpPolicy` class,
network weights and learned log standard deviations. Reloading restores the
distribution for training, evaluation and live playback; it does not reinterpret
old clipped-Gaussian weights as a squashed policy. The normalized action and
observation schemas are unchanged, so both distributions use the same target
controller contract. The saved policy is authoritative when loading a checkpoint;
editing JSON does not convert a loaded model or reset its learned noise.

Task-bounded target policies additionally record `coordinates:
task-bounded-world-xy-v1` and `bounds: [x_min, x_max, y_min, y_max]`.
DefenseZone uses `[-120, 120, -270, -88]`. Its full `[-1, 1]` output range maps
into our defensive zone. The old `absolute-world-xy-v1` whole-rink contract
remains identifiable, but loading it into bounded DefenseZone fails explicitly:
equal tensor shapes do not make different coordinate mappings compatible.

The recorded `frame_skip` is the policy decision interval. Live/playback advance
the emulator one frame at a time but retain policy actions for that recorded,
validated interval. Controller timers are measured in emulator frames.

## Experimental manual-goalie artifacts

`nhl94 goalie train` writes an independent SB3 `DecisionPPO` ZIP plus
`MODEL.zip.goalie.json`. The sidecar records schema version, token dimensions and action order,
SHA256, parameter count, actual training decisions/native frames, decision
interval, initialization seed, architecture, reward/horizon, source-game split,
runtime source hashes and checkpoint history. Loading validates the checksum,
schema, policy class and any explicitly requested interval. These artifacts
are separate from legacy skater/carry checkpoints. Deployment uses the Python
goalie controller; no separate-runtime export contract is provided. New training starts from random weights and
requires a fresh output/checkpoint directory.

The 292-float schema is fourteen twenty-field entity/context tokens followed
by twelve action-mask bits. Actions are neutral, eight directional pads, save,
left dive and right dive in the defended-net-at-negative-Y frame. Schema
version 2 includes the correctly decoded native one-timer-shot
flag in context field 16. Version-1 experimental goalie files are rejected;
legacy skater-policy inputs and artifacts retain their original contract.
Model files alone do not encode the native handoff/animation/outlet controller. Deployment
requires `--goalie-model MODEL.zip` inside enabled full-team Classic manual
goalie play. Copy the ZIP and its sidecar together.

`python -m nhl94_ai.evaluation.goalie` collects locally generated
`native-selected-manual-goalie-starts` banks. A bank lists checksum-protected
`.pkl.gz` snapshots and their `.json` metadata: complete emulator/RNG bytes,
controller/GameState history, initial RAM digest, physical source game, side
and native frame. Pickle snapshots are trusted local artifacts, not a public
dataset interchange format. Restores verify checksums and initial RAM.
Splitting individual snapshots from the same physical game across train/test
is rejected during evaluation unless explicitly allowed for a training audit.

## Export to the separate runtime

```bash
nhl94 export --src checkpoint.zip --dest export/policy --samples demos/shard.npz
```

Install `.[export]` first. This produces `.onnx` (opset 17), `.pt` (TorchScript),
and a JSON sidecar for each after numerical verification passes. The export
accepts Box observations with batch size 1 and float32 input named `input`.
Outputs are deterministic `actions`, `values`, and `log_prob`, in that order.
Image inputs use the model's channel order. SB3 environment clipping/rescaling is
excluded. Dict observations are rejected explicitly.

Verification compares all three outputs with the Python policy at `rtol=1e-4`,
`atol=1e-5`, using zeros, ones, a seeded input, and up to 32 supplied observations.
Without `--samples`, only the three synthetic probes are used. The sidecar records
input shape, output names, source metadata, tolerances, and tested sample count.
Legacy source schemas remain marked unknown in exports too. Runtime code belongs to `retro-ai-runtime`.

Target-policy export is rejected before writing files: coordinates require the
matching scripted controller and its state/observation contract, which the
separate runtime does not yet implement.

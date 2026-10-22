# Artifacts

This guide describes supported artifact loading, playback, evaluation, and export.
Training, fitting, demonstration collection, BC, and DAgger workflows are
unavailable pending the ML redesign; ES support is removed. Historical training
metadata remains readable but does not imply that those workflows can run.
See the [README](../README.md#learned-policies) for existing-policy playback and
the [evaluation guide](../nhl94_ai/evaluation/README.md) for evaluation commands.

| Artifact | Files | Current use |
| --- | --- | --- |
| General SB3 policy | `MODEL.zip` and `MODEL.zip.json` | Compatible policy playback, evaluation, and button-policy export |
| Demonstration shard | `SHARD.npz` and `SHARD.npz.json` | Dataset loading and export verification samples |
| Offense policy | `MODEL.zip` and `MODEL.zip.offense.json` | Offense inference, playback, and evaluation |
| Goalie policy | `MODEL.zip` and `MODEL.zip.goalie.json` | Goalie evaluation and enabled Classic manual-goalie play |
| Retention model | JSON tree or ranker; historical `.training.json` report | Optional Classic movement filtering |
| Policy export | `.onnx` and `.pt`, each with a `.json` sidecar | Verified deterministic inference in a separate runtime |

Keep model files with their matching sidecars. These formats have distinct
loaders and contracts; their metadata is not interchangeable.

## Checkpoint metadata and compatibility

General policy and demonstration metadata use `format_version: 1`, recording:

- resolved options and hyperparameters, environment seed, and optional policy seed;
- code revision, dirty-worktree status, Python-source SHA-256 fingerprint,
  dependency versions, and Python version;
- game, ordered observation fields, schema version, normalization, action type,
  sequence length, and frame skip.

`artifacts.load_policy` and `datasets.load_demo_arrays` compare known schemas
when an expected configuration is supplied. Equal tensor lengths do not make
field orders compatible. Different neural architectures may share an input
schema. Unsupported metadata format versions fail explicitly.

Missing or unversioned metadata is marked `compatibility: unknown`; loaders do
not infer a schema from tensor dimensions. Legacy button policies can still load,
but the caller must supply the correct configuration. `require_metadata=True`
rejects unknown checkpoint schemas. Historical imports such as
`models.ResidualMlpPolicy` resolve through aliases installed during loading.

The retired `DartPolicy`, `CustomMlpPolicy`, `HockeyMultiHeadPolicy`,
`AttentionMLPPolicy`, `CnnTransformerPolicy`, and `ViTPolicy` implementations and
their historical aliases are removed. Checkpoints requiring those classes need
a repository revision from before their removal.

Named inputs from `model_input.json` resolve into saved ordered groups and schema
versions; a registry name alone is not the schema. The compact `pvg` layout records
`observation_layout: player-goalie-v1` and `normalization: player-centered-rink-v1`.
Its version-4 schema has 52 inputs, including C-pressed status and hold duration.
It is incompatible with the 50-input `pvg-v3` and earlier 310-input PvG contract.
Use the input configuration matching the checkpoint: `pvg-v3` for the 50-input
contract, or `default` for the earlier 310-input PvG contract. `default` and
`legacy` retain version-2 and version-1 field ordering respectively.

Built-in task rewards currently return zero as compatibility stubs. Evaluation
marks them unavailable and uses `null` for reward metrics. Historical reward
values and definitions in saved artifacts retain their original meaning.

## Reactive scripted demonstrations

Historical reactive Classic demonstrations contain one sample per emulator frame,
including button releases. Their `steps` field indexes native-frame samples within
an episode; rewards and dones belong to those samples. Metadata records
`sample_interval_frames: 1`, `schema.frame_skip: 1`, and
`teacher_decision_interval` (normally 4). Historical collection budgets counted
outer decisions, so a four-frame run could produce up to four times that many rows.
DAgger labels described learner-visited substeps, without teacher takeover.

The dataset loader accepts these per-frame samples against an expected teacher
interval when the metadata agrees, while still checking other schema fields.
This exception does not relax checkpoint frame-skip compatibility. A policy acting
every four frames cannot reproduce every recorded native-frame button edge.
Collection and imitation commands remain unavailable.

`classic_offense` and `classic_defense` diagnostics describe controller decisions;
they are not additional neural input fields. Existing datasets preserve the
behavior of the teacher that produced them, even when Classic's tactics change.

## Experimental possession-policy artifacts

`nhl94 offense` loads `.zip.offense.json` sidecars with ordered observations/actions,
execution cadence, and a ZIP hash. The `algorithm` field selects `PPO` or the
legacy `DecisionPPO` compatibility class; an omitted field defaults to PPO.
Prediction remains supported; rollout collection and learning are removed.

Historical `training_contract` fields describe the task, reward, observation
clock, discount units, and whether counters measured decisions or fixed ticks.
Finite-attack tasks use remaining attack seconds in the clock field while keeping
the tensor ordering. Evaluation requires `--allow-task-transfer` to cross task
contracts. Native frames and active decisions are distinct from SB3 timesteps.

The carry-only variant retains the 19 action IDs and feature order but masks
IDs 9 through 18. `training_contract.control_scope` is `carry` or `offense`;
missing scope means historical full offense. Evaluation requires
`--allow-control-transfer` to cross scopes, independently of task transfer.
Opportunity checkpoints may also record reward weights, native-frame time costs,
four-frame discount units, and decision-based GAE details from historical training.

Attack banks contain trusted local emulator/controller snapshots with hash
sidecars. All snapshots and both team assignments from a source game belong in
one data partition. Reports identify source games and hashes. Outcome counters
start after snapshot restoration: requested passes, native one-timer attempts,
shots, and goals are separate events. `active_decisions` counts eligible policy
ticks; `executed_commands` counts ticks that used external movement, since Classic
can take priority to finish. See the [evaluation guide](../nhl94_ai/evaluation/README.md)
for experiment protocols.

Sidecars may retain `parameters`, `initialization`, `parent_timesteps`, and CLI
options. Checkpoint manifests and `.training.json` reports describe historical
runs; they are not instructions for producing new checkpoints.

## Supervised retention artifacts

`--retention-model` loads either JSON schema below, preserving Classic pass/shot
priority and native reactions. Both require four-frame offensive decisions,
a 24-frame label horizon, and matching ordered physical/action features.

| Schema | Stored model | Decision threshold |
| --- | --- | --- |
| `native-carry-retention-tree-v1` | Shallow decision tree with leaf risks | Minimum baseline risk and risk improvement |
| `native-carry-retention-ranker-v1` | Base score and ordered boosted trees | Minimum within-situation score margin |

Higher ranker scores indicate preferred movements, not loss probabilities.
Classifier risk thresholds cannot be applied to ranker scores. Inference validates
feature order, finite inputs, tree structure, and thresholds. It uses float32
inputs with `<=` splits for the classifier and strict `<` splits for the ranker.
JSON inference needs NumPy; scikit-learn/XGBoost fitting workflows are retired.

Historical artifacts and adjacent training reports record source-game partitions,
dataset/source hashes, dependency versions, fixed model/threshold comparisons,
outcomes, and export parity. Ranking labels compared alternative movements from
the same snapshot; censored branches supplied no label. Every alternative and
both controlled sides of a physical save/ROM seed belong in the same partition.
Future outcomes are labels only; inference uses current-state features. Ending
carrying exposure with a pass or shot does not establish a successful attack.

## Historical live-training snapshots

Existing `*_latest_live.zip` snapshots remain readable by the optional viewer.
Snapshot-producing callbacks are removed; no new live-training sessions are
available. The filename does not replace the checkpoint's compatibility metadata.

## Target-position policies

`TARGET_POSITION` checkpoints add `schema.target_controller`: coordinate and
feedback conventions, ordered appended observations, controller profile, version,
and settings. Their two-float actions and augmented observations are not
interchangeable with button policies. Target-mode loading requires verified
metadata; the current controller contract is checked even without an explicit
expected environment configuration.

A saved `SquashedMlpPolicy` includes its weights and learned log standard
deviations; `squash_output` and `log_std_init` are recorded in hyperparameters.
Loading for evaluation/playback restores that saved distribution. Editing JSON
does not convert clipped-Gaussian weights into a squashed policy or reset noise.
Both distributions use the same normalized target-controller schema.

Task-bounded policies record `coordinates: task-bounded-world-xy-v1` and
`bounds: [x_min, x_max, y_min, y_max]`. DefenseZone uses
`[-120, 120, -270, -88]`, mapping the full `[-1, 1]` action range into that zone.
The older `absolute-world-xy-v1` whole-rink contract is identifiable but rejected
for bounded DefenseZone, despite equal tensor shapes.

Recorded `frame_skip` is the policy decision interval. Playback advances one
emulator frame at a time while holding actions for that validated interval;
controller timers count emulator frames.

## Experimental manual-goalie artifacts

Goalie policies use an independent `DecisionPPO` ZIP and `MODEL.zip.goalie.json`.
Sidecars record schema, token/action dimensions, checksum, decision interval,
architecture, and historical training provenance. Loading validates schema,
checksum, policy class, and any explicitly requested interval.

The 292-float input contains fourteen twenty-field entity/context tokens and
twelve action-mask bits. Actions are neutral, eight directions, save, and left/right
dives, with the defended net at negative Y. Schema version 2 includes the native
one-timer-shot flag in context field 16; version-1 goalie artifacts are rejected.

The checkpoint does not encode native handoff, animation, or outlet control.
Use `--goalie-model MODEL.zip` inside enabled full-team Classic manual-goalie play,
keeping the ZIP and sidecar together. Deployment uses the Python goalie controller;
there is no separate-runtime export contract for these policies.

`python -m nhl94_ai.evaluation.goalie` still collects evaluation start banks with
schema `native-selected-manual-goalie-starts`. Checksum-protected `.pkl.gz`
snapshots and JSON metadata preserve emulator/RNG bytes, controller/GameState
history, RAM digest, physical source game, side, and native frame. These are
trusted local snapshots rather than a public dataset interchange format.
Restores verify checksums and initial RAM. Goalie policy evaluation rejects starts
from recorded training sources unless `--allow-training-starts` is explicit.

## Export to the separate runtime

```bash
nhl94 export --src checkpoint.zip --dest export/policy --samples demos/shard.npz
```

Install `.[export]` first. The command writes ONNX (opset 17) and TorchScript
files, verifies their outputs, then writes a JSON sidecar for each. Failed
verification can leave model files without validated sidecars.

Export supports compatible button policies with Box observations, batch size 1,
and float32 input named `input`. Outputs are deterministic `actions`, `values`,
and `log_prob`, in that order. Images use the policy's channel order. SB3
clipping/rescaling is excluded. Dict observations are unsupported; target-policy
export is rejected before writing files because the separate runtime lacks its
scripted controller contract.

Verification compares all outputs with the Python policy at `rtol=1e-4`,
`atol=1e-5`, using zeros, ones, a seeded input, and up to 32 supplied observations.
`--samples` accepts `.npy` observations or an existing `.npz` demonstration shard;
without it only the three synthetic probes run. Sidecars record input shape,
output names, source metadata, tolerances, and sample count. Unknown legacy schemas
remain unknown after export. Runtime code belongs to `retro-ai-runtime`.

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
Legacy source schemas remain marked unknown in exports too. Runtime code changes
belong to `retro-ai-runtime` and were not made here.

Target-policy export is rejected before writing files: coordinates require the
matching scripted controller and its state/observation contract, which the
separate runtime does not yet implement.

# Refactor results

Implemented in `nhl94-ai` on 2026-09-26. Changes are in the working tree;
no Git history was imported and no commit was created. The sibling emulator,
original scripts repository, and C++ runtime were not modified.

## Delivered

- Root-level installable `nhl94_ai/` package and `nhl94` CLI, with no `src/`.
- Separate state/features, observation encoding, actions, perspective, tasks,
  agents, neural architectures, training, evaluation, and optional display code.
- Named task/model/agent registries and stateless environment bindings.
- One environment assembly path and shared RL trainer for ordinary/live sessions.
- Direct typed configuration calls for curriculum and DAgger/BC execution.
- Agent reset/action interface, scripted/learned/multi-model adapters, headless
  evaluation and reusable demonstration collection.
- Versioned checkpoint/dataset metadata and numerical ONNX/TorchScript checks.
- Packaged defaults, explicit package discovery, dependency extras, CI and docs.
- Unused generic wrappers, other-game documentation, obsolete runtime notice,
  temporary forwarding scripts, duplicate configuration directories, and
  requirements aliases removed. The package CLI and `configs/` are the supported
  entry points; dependencies are declared in `pyproject.toml`.

## Verification

| Check | Result |
| --- | --- |
| ROM-free unit suite | 19 tests: 18 passed, 1 expected failure |
| Frozen gameplay traces | All five match the pre-refactor observations, actions, rewards and completion flags |
| NHL94 variants | 1-on-1, 2-on-2 and full-team exercised |
| PPO workflow | 16-step ResidualMlpPolicy train/save/reload passed |
| Imitation workflow | Two short demo episodes and one BC epoch passed |
| DAgger | One collection/retraining round passed |
| Curriculum | Bundled curriculum dry-run and a collect → BC chain passed |
| Installed headless training | 16-step run with final evaluation and saved model passed |
| Common agent service | ClassicAIV3 collection through `collect_demonstrations` passed |
| Historical checkpoint | Custom ResidualMlpPolicy saved before migration loaded and predicted from checkout and wheel |
| Export | ONNX and TorchScript matched all three Python outputs on 35 inputs, including 32 recorded observations |
| Wheel | Built, installed into a separate environment, and imported from `/tmp`; packaged defaults and every command help entry passed |
| Command migration | Original forwarding scripts were checked before removal; integration workflows now invoke the package CLI from outside the checkout |
| Optional dependencies | Headless training/curriculum/evaluation/export imports did not load pygame, timm, ONNX or ONNX Runtime |
| UI | Live/comparison modules imported and live display object constructed with dummy SDL drivers |
| Lint | Pylint passed using `pyproject.toml`, including undefined-name checks |
| Bundled models | All four model-file SHA-256 hashes match the pre-refactor copies |

The wheel check reused installed dependency distributions while loading project
code exclusively from the separately installed wheel. It was not a fresh download
of every dependency. CI is configured for package/unit/lint checks; the remote CI
job itself has not been run during this local refactor. Full interactive display
behavior and every optional architecture were not exhaustively tested.

## Compatibility decisions

Use `nhl94` or `python -m nhl94_ai` instead of the removed forwarding scripts.
Configuration examples live under `configs/training/` and `configs/curricula/`.
The old path fallback was removed; packaged defaults remain available when no
file is supplied. Install dependencies through `pyproject.toml` extras.

Checkpoint import aliases are retained inside `nhl94_ai/compat.py`; they load
historical custom policies without restoring the old script directory. The
standalone Evolution Strategies runner remains available through
`python -m nhl94_ai.training.rl --alg es`.

Legacy scripted-controller episode timing and input/reward calculations are
preserved. New resettable agent services make their own episode boundaries
explicit. Live evaluation keeps its previous full-episode horizon; bounded
headless matches use the same evaluation settings type with an explicit step
budget.

The plan's checks are completed at this migration's scope, not a claim that every
historical mode is bug-free. Inherited truncation, PvP, and custom self-play
limitations are recorded in [KNOWN_ISSUES.md](KNOWN_ISSUES.md) for separate behavior
changes. Model binaries remain untouched.

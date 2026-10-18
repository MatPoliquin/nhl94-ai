# NHL94 AI contributor instructions

- Put Python implementation in the root `nhl94_ai/` package; do not introduce a
  Python `src/` layout. Keep the independent TypeScript/WASM browser app in `web/`.
- For Python workflows, use installed `nhl94` commands or `python -m nhl94_ai`;
  implementation belongs in the package. Keep Python configuration examples
  under `configs/`.
- Preserve CLI flags, task names, state names, input field ordering and saved-model
  compatibility. Keep behavioral/tactical changes separate from structural work.
- `models/` holds binary assets. Do not regenerate them during code refactors.
- `stable-retro` and `retro-ai-runtime` are separate repositories; change them only
  when explicitly included in the task.
- Declare Python dependencies and lint settings in `pyproject.toml`. Install
  `.[dev]` for Python development and the relevant optional extras when needed.
  Browser dependencies and scripts belong in `web/package.json`.
- For Python changes, run `python -m unittest discover -s tests -v` and
  `python -m pylint nhl94_ai tests`. ROM checks in `tests/integration/` run explicitly.
- Keep UI and export imports optional. Follow `docs/ARCHITECTURE.md` for extension
  points and `docs/ARTIFACTS.md` for checkpoint/dataset/export contracts.
- For Python training rewards, randomized resets or RAM-backed behavior, start with
  `.github/skills/nhl94-task-development/SKILL.md` for the source map, known ROM
  pitfalls and focused validation workflow.
- For browser changes, follow
  [web instructions](instructions/web.instructions.md). For the WASM bridge,
  RAM/controller contracts, Classic porting, replays or browser networking, use
  [browser arena development](skills/nhl94-browser-arena-development/SKILL.md).

- Keep plans, benchmark results, run records, and development history outside the
  repository. Public docs describe current usage and contracts. Use explicit
  external paths for measurement inputs and outputs.

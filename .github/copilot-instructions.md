# NHL94 AI contributor instructions

- Put implementation in the root `nhl94_ai/` package; do not introduce `src/`.
- Use installed `nhl94` commands or `python -m nhl94_ai`; implementation belongs
  in the package. Keep configuration examples under `configs/`.
- Preserve CLI flags, task names, state names, input field ordering and saved-model
  compatibility. Keep behavioral/tactical changes separate from structural work.
- `models/` holds binary assets. Do not regenerate them during code refactors.
- `stable-retro` and `retro-ai-runtime` are separate repositories; change them only
  when explicitly included in the task.
- Declare dependencies and lint settings in `pyproject.toml`. Install `.[dev]`
  for development and the relevant optional extras when needed.
- Run `python -m unittest discover -s tests -v` and
  `python -m pylint nhl94_ai tests`. ROM checks in `tests/integration/` run explicitly.
- Keep UI and export imports optional. Follow `docs/ARCHITECTURE.md` for extension
  points and `docs/ARTIFACTS.md` for checkpoint/dataset/export contracts.
- For training rewards, randomized resets or RAM-backed behavior, start with
  `.github/skills/nhl94-task-development/SKILL.md` for the source map, known ROM
  pitfalls and focused validation workflow.

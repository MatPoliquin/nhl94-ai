"""Retired training entrypoints; machine learning is being redesigned."""
from nhl94_ai.training.unavailable import build_parser, main, unavailable

run = unavailable
fit = unavailable

from nhl94_ai.evaluation.shot_direction_model import read_cases, selected_index, evaluate


if __name__ == '__main__':
    raise SystemExit(main())

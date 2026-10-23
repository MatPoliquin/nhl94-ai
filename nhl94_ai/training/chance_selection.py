"""Retired training entrypoints; machine learning is being redesigned."""
from nhl94_ai.training.unavailable import build_parser, main, unavailable

run = unavailable
fit = unavailable

from nhl94_ai.evaluation.chance_selection_model import cases, select, reward, evaluate


if __name__ == '__main__':
    raise SystemExit(main())

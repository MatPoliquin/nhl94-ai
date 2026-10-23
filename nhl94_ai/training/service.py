"""Retired training entrypoints; machine learning is being redesigned."""
from nhl94_ai.training.unavailable import build_parser, main, unavailable

run = unavailable
train = unavailable

from nhl94_ai.training.events import LiveTrainingResult as TrainingResult


if __name__ == '__main__':
    raise SystemExit(main())

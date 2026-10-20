"""Retired training entrypoints; machine learning is being redesigned."""
from nhl94_ai.training.unavailable import build_parser, main, unavailable

run = unavailable
train_bc = unavailable


if __name__ == '__main__':
    raise SystemExit(main())

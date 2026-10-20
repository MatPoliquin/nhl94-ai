"""Retired training entrypoints; machine learning is being redesigned."""
from nhl94_ai.training.unavailable import main, unavailable

run = unavailable

from nhl94_ai.run_options import build_parser, parse_cmdline


class ModelTrainer:
    def __init__(self, *args, **kwargs):
        unavailable()


if __name__ == '__main__':
    raise SystemExit(main())

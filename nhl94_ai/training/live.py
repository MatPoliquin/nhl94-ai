"""Retired training entrypoints; machine learning is being redesigned."""
from nhl94_ai.training.unavailable import main, unavailable

run = unavailable
run_training_session = unavailable

from nhl94_ai.run_options import build_parser, parse_cmdline, prepare_args
from nhl94_ai.evaluation.session import run_evaluation_session


if __name__ == '__main__':
    raise SystemExit(main())

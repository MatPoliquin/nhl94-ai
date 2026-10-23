"""Retired training entrypoints; machine learning is being redesigned."""
from nhl94_ai.training.unavailable import build_parser, main, unavailable

run = unavailable
run_dagger = unavailable
collect_dagger_round = unavailable
run_bc_retrain = unavailable


if __name__ == '__main__':
    raise SystemExit(main())

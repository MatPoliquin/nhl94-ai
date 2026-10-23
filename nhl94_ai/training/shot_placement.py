"""Retired training entrypoints; machine learning is being redesigned."""
from nhl94_ai.training.unavailable import build_parser, main as unavailable_main, unavailable

run = unavailable
fit_logistic = unavailable

from nhl94_ai.evaluation.shot_placement_model import read_cases, evaluate
from nhl94_ai.models.shot_scoring import estimate


def main(argv=None):
    import sys
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == 'evaluate':
        from nhl94_ai.evaluation.shot_placement_model import main as evaluate_main
        return evaluate_main(argv[1:])
    return unavailable_main(argv)


if __name__ == '__main__':
    raise SystemExit(main())

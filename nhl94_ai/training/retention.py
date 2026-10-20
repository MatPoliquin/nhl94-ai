"""Retired training entrypoints; machine learning is being redesigned."""
from nhl94_ai.training.unavailable import build_parser, main as unavailable_main, unavailable

run = unavailable
train = unavailable

from nhl94_ai.evaluation.retention_model import load_rows, cluster_interval, score_policy, predictive_metrics, evaluate


def main(argv=None):
    import sys
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == 'evaluate':
        from nhl94_ai.evaluation.retention_model import main as evaluate_main
        return evaluate_main(argv[1:])
    return unavailable_main(argv)


if __name__ == '__main__':
    raise SystemExit(main())

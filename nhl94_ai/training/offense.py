"""Compatibility entrypoint; only existing-model evaluation remains available."""
from nhl94_ai.evaluation.offense_policy import build_parser, run, evaluate
from nhl94_ai.training.unavailable import unavailable

collect = unavailable
clone = unavailable
train = unavailable


if __name__ == '__main__':
    import sys
    from nhl94_ai.training.unavailable import main
    if len(sys.argv) > 1 and sys.argv[1] in ('train', 'bc', 'collect'):
        raise SystemExit(main())
    run(build_parser().parse_args())

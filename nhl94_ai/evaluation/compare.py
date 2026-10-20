"""The train-then-compare workflow is retired pending the ML redesign."""
from nhl94_ai.training.unavailable import main, unavailable

train_model = unavailable


if __name__ == '__main__':
    raise SystemExit(main())

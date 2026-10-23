"""Explicit boundary for retired machine-learning workflows."""
import argparse
import sys

MESSAGE = "Training is unavailable while the machine-learning workflows are being redesigned. Existing non-ES model playback and evaluation remain available."


class TrainingUnavailableError(NotImplementedError):
    """A retired workflow was requested."""


def unavailable(*args, **kwargs):
    raise TrainingUnavailableError(MESSAGE)


def build_parser():
    return argparse.ArgumentParser(description=MESSAGE)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    if '-h' in argv or '--help' in argv:
        build_parser().print_help()
        return 0
    print(MESSAGE, file=sys.stderr)
    return 2

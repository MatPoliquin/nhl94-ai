"""Application services shared by installed commands and Python callers."""
from nhl94_ai.config import RunConfig, TrainingConfig, namespace_from_mapping
from nhl94_ai.training.events import LiveTrainingResult as TrainingResult


def train(config, *, status_reporter=None) -> TrainingResult:
    from nhl94_ai.training.live import build_parser, prepare_args, run_training_session
    if isinstance(config, (dict, RunConfig)):
        options = config.options if isinstance(config, RunConfig) else config
        config = namespace_from_mapping(build_parser(), {'headless': True, **options})
    TrainingConfig.from_args(config)
    prepare_args(config)
    return run_training_session(config, status_reporter=status_reporter)

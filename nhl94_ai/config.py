import copy
import json
import os
from dataclasses import dataclass, field
from importlib.resources import files
from types import SimpleNamespace
from typing import Any, Dict, Optional, Tuple


def default_config_path(name='default'):
    if name not in ('default', 'nhl94'):
        raise ValueError(f'Unknown built-in training configuration: {name}')
    return str(files('nhl94_ai').joinpath(f'data/training/{name}.json'))


@dataclass(frozen=True)
class EnvironmentConfig:
    env: str = 'NHL941on1-Genesis-v0'
    num_players: int = 1
    nn: str = 'MlpPolicy'
    action_type: str = 'FILTERED'
    selfplay: bool = False
    seq_len: int = 16

    def validate(self):
        from nhl94_ai.game.specs import get_game
        get_game(self.env)
        if self.num_players not in (1, 2):
            raise ValueError('num_players is the controller count and must be 1 or 2')
        if self.seq_len < 1:
            raise ValueError('seq_len must be positive')
        if self.action_type not in ('FILTERED', 'DISCRETE', 'MULTI_DISCRETE', 'HOCKEY_INTENT_DPAD', 'TARGET_POSITION'):
            raise ValueError(f'Unsupported action type: {self.action_type}')
        if self.action_type == 'HOCKEY_INTENT_DPAD' and (self.env != 'NHL94-Genesis-v0' or self.num_players != 1 or self.selfplay):
            raise ValueError('HOCKEY_INTENT_DPAD requires full-team NHL94 with one learner controller')
        if self.action_type == 'TARGET_POSITION' and (
                self.env != 'NHL94-Genesis-v0' or self.num_players != 1 or self.selfplay or self.nn != 'MlpPolicy'):
            raise ValueError('TARGET_POSITION requires full-team NHL94, MlpPolicy and one controller without self-play')
        if self.selfplay and self.action_type == 'DISCRETE':
            raise ValueError('Self-play does not support DISCRETE actions')
        if self.selfplay and self.nn in ('CnnPolicy', 'CustomCnnPolicy', 'ImpalaCnnPolicy', 'CnnTransformerPolicy', 'ViTPolicy'):
            raise ValueError('Self-play requires structured observations')
        return self

    @classmethod
    def from_args(cls, args):
        values = {name: getattr(args, name, definition.default) for name, definition in cls.__dataclass_fields__.items()}
        values['action_type'] = values['action_type'].upper()
        config = cls(**values).validate()
        if getattr(args, "alg", "ppo2") not in ("ppo2", "es"):
            raise ValueError("alg must be ppo2 or es")
        from nhl94_ai.models.factory import MODEL_BUILDERS
        from nhl94_ai.agents.registry import CONTROLLERS
        if config.nn not in MODEL_BUILDERS and config.nn not in CONTROLLERS:
            raise ValueError(f'Unknown policy or agent: {config.nn}')
        goalie_policy = getattr(args, 'goalie_policy', 'off')
        if goalie_policy not in ('off', 'selective', 'always'):
            raise ValueError('goalie_policy must be off, selective or always')
        goalie_supported = (config.nn in CONTROLLERS and config.env == 'NHL94-Genesis-v0'
                            and config.action_type == 'FILTERED' and not config.selfplay
                            and getattr(args, 'rf', 'PostPlay') == 'PostPlay')
        if goalie_policy != 'off' and not goalie_supported:
            raise ValueError('Manual goalie AI requires full-team Classic FILTERED PostPlay without self-play')
        if getattr(args, 'mode', None) == 'player_vs_model' and (
                config.action_type != 'FILTERED' or config.selfplay):
            raise ValueError('player_vs_model requires FILTERED buttons without self-play')
        if config.action_type == 'TARGET_POSITION':
            from nhl94_ai.tasks.registry import get_task
            from nhl94_ai.env.target_control import validate_target_bounds
            if getattr(args, 'alg', 'ppo2') != 'ppo2':
                raise ValueError('TARGET_POSITION currently supports PPO only')
            task = get_task(getattr(args, 'rf', ''))
            if task.target_control != 'defense':
                raise ValueError('This task has not opted into TARGET_POSITION defensive control')
            validate_target_bounds(task.target_bounds)
            if getattr(args, 'mode', 'model_vs_game') != 'model_vs_game' or getattr(args, 'model_2', ''):
                raise ValueError('TARGET_POSITION supports one model versus the game, not human or mixed-model modes')
        return config


@dataclass
class RunConfig:
    """Resolved application options; environment constraints are validated once."""
    options: Dict[str, Any] = field(default_factory=dict)

    def namespace(self):
        args = SimpleNamespace(**copy.deepcopy(self.options))
        EnvironmentConfig.from_args(args)
        return args


def namespace_from_mapping(parser, values):
    """Apply typed values to parser defaults without reparsing synthetic CLI text."""
    import argparse
    actions = {}
    for action in parser._actions:
        if action.dest != 'help':
            actions.setdefault(action.dest, action)
    unknown = set(values) - set(actions)
    if unknown:
        raise ValueError(f'Unknown options: {", ".join(sorted(unknown))}')
    result = {name: copy.deepcopy(action.default) for name, action in actions.items()}
    for name, value in values.items():
        action = actions[name]
        if isinstance(action, (argparse._StoreTrueAction, argparse._StoreFalseAction)) and value is not None and not isinstance(value, bool):
            raise TypeError(f'{name} must be a JSON boolean')
        if value is not None and action.type and action.nargs not in ('+', '*'):
            value = action.type(value)
        if action.choices and value is not None and value not in action.choices:
            raise ValueError(f'{name} must be one of {action.choices}')
        result[name] = value
    for name, action in actions.items():
        if action.required and result.get(name) is None:
            raise ValueError(f'Missing required option: {name}')
    return SimpleNamespace(**result)


def resolve_config_path(path: Optional[str], base_dir: Optional[str] = None) -> Optional[str]:
    if not path:
        return None

    expanded = os.path.expanduser(path)
    if os.path.isabs(expanded):
        return expanded

    if base_dir:
        candidate = os.path.abspath(os.path.join(base_dir, expanded))
        if os.path.isfile(candidate):
            return candidate

    return os.path.abspath(expanded)


def load_hyperparams(path: Optional[str], *, required: bool = True, base_dir: Optional[str] = None) -> Dict[str, Any]:
    if not path:
        if required:
            raise ValueError("Hyperparameters path is required but was not provided.")
        return {}

    resolved = resolve_config_path(path, base_dir=base_dir)
    if resolved is None:
        if required:
            raise ValueError("Hyperparameters path is required but was not provided.")
        return {}

    if not os.path.isfile(resolved):
        raise FileNotFoundError(f"Hyperparameters file not found: {resolved}")

    with open(resolved, "r", encoding="utf-8") as handle:
        hyperparams = json.load(handle)

    if not isinstance(hyperparams, dict):
        raise TypeError(f"Hyperparameters file must contain a JSON object: {resolved}")

    return hyperparams


def _deep_merge_dicts(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = _deep_merge_dicts(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def resolve_hyperparams_for_model(
    hyperparams: Optional[Dict[str, Any]],
    nn_type: Optional[str],
) -> Dict[str, Any]:
    if not hyperparams:
        return {}

    if "common" not in hyperparams and "model_overrides" not in hyperparams:
        return copy.deepcopy(hyperparams)

    common = hyperparams.get("common", {})
    if common is None:
        common = {}
    if not isinstance(common, dict):
        raise TypeError("hyperparams['common'] must be a JSON object when provided")

    model_overrides = hyperparams.get("model_overrides", {})
    if model_overrides is None:
        model_overrides = {}
    if not isinstance(model_overrides, dict):
        raise TypeError("hyperparams['model_overrides'] must be a JSON object when provided")

    model_override = model_overrides.get(nn_type, {}) if nn_type else {}
    if model_override is None:
        model_override = {}
    if not isinstance(model_override, dict):
        raise TypeError(f"hyperparams['model_overrides']['{nn_type}'] must be a JSON object when provided")

    resolved = _deep_merge_dicts(common, model_override)
    if "model_input" in hyperparams:
        resolved["model_input"] = copy.deepcopy(hyperparams["model_input"])
    return resolved


def load_json_dict(path: Optional[str], *, required: bool = True, base_dir: Optional[str] = None, label: str = "JSON") -> Dict[str, Any]:
    if not path:
        if required:
            raise ValueError(f"{label} path is required but was not provided.")
        return {}

    resolved = resolve_config_path(path, base_dir=base_dir)
    if resolved is None:
        if required:
            raise ValueError(f"{label} path is required but was not provided.")
        return {}

    if not os.path.isfile(resolved):
        raise FileNotFoundError(f"{label} file not found: {resolved}")

    with open(resolved, "r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if not isinstance(payload, dict):
        raise TypeError(f"{label} file must contain a JSON object: {resolved}")

    return payload


def load_curriculum(path: str, *, base_dir: Optional[str] = None) -> Dict[str, Any]:
    resolved = resolve_config_path(path, base_dir=base_dir)
    if resolved is None:
        raise ValueError("Curriculum path is required but was not provided.")

    curriculum = load_json_dict(resolved, label="Curriculum")

    phases = curriculum.get("phases")
    if not isinstance(phases, list) or not phases:
        raise ValueError(f"Curriculum must define a non-empty 'phases' list: {resolved}")

    common = curriculum.get("common", {})
    if common is None:
        common = {}
    if not isinstance(common, dict):
        raise TypeError(f"Curriculum 'common' must be a JSON object: {resolved}")

    normalized_phases = []
    for index, phase in enumerate(phases, start=1):
        if not isinstance(phase, dict):
            raise TypeError(f"Curriculum phase #{index} must be a JSON object: {resolved}")
        normalized_phases.append(phase)

    curriculum["common"] = common
    curriculum["phases"] = normalized_phases
    curriculum["_resolved_path"] = resolved
    curriculum["_base_dir"] = os.path.dirname(resolved)

    return curriculum


def resolve_clip_reward(args, hyperparams: Optional[Dict[str, Any]]) -> bool:
    if hasattr(args, "clip_reward") and getattr(args, "clip_reward") is not None:
        return bool(getattr(args, "clip_reward"))

    if hyperparams and "clip_reward" in hyperparams:
        return bool(hyperparams.get("clip_reward"))

    return True


def resolve_sticky_action_settings(
    requested_enabled: bool,
    hyperparams: Optional[Dict[str, Any]],
    *,
    default_prob: float = 0.25,
) -> Tuple[bool, float]:
    if not requested_enabled:
        return False, -1.0

    enabled = True
    probability = default_prob

    if hyperparams and "sticky_actions" in hyperparams:
        enabled = bool(hyperparams.get("sticky_actions"))

    if hyperparams and "sticky_action_prob" in hyperparams:
        probability = float(hyperparams.get("sticky_action_prob"))

    if not enabled:
        return False, -1.0

    probability = max(0.0, min(1.0, probability))
    return True, probability


PATH_OPTIONS = frozenset({
    'hyperparams', 'output_basedir', 'output', 'output_dir', 'output_model',
    'model', 'model_1', 'model_2', 'load_model', 'load_p1_model', 'load_p2_model',
    'load_opponent_model', 'src', 'dest', 'video_path', 'datasets', 'base_datasets',
})


def resolve_declared_paths(options, base_dir):
    """Resolve even not-yet-created outputs relative to the declaring file."""
    from pathlib import Path
    def resolve(value):
        if not value:
            return value
        if isinstance(value, list):
            return [resolve(item) for item in value]
        path = Path(value).expanduser()
        return str(path if path.is_absolute() else Path(base_dir) / path)
    return {key: resolve(value) if key in PATH_OPTIONS else value for key, value in options.items()}


@dataclass(frozen=True)
class TrainingConfig:
    num_env: int = 1
    num_timesteps: int = 6_000_000
    seed: int = 0

    @classmethod
    def from_args(cls, args):
        result = cls(**{key: getattr(args, key, field.default) for key, field in cls.__dataclass_fields__.items()})
        if result.num_env < 1 or result.num_timesteps < 1:
            raise ValueError('num_env and num_timesteps must be positive')
        EnvironmentConfig.from_args(args)
        if getattr(args, 'nn', '').startswith('Classic'):
            raise ValueError('Scripted agents are inference-only; select a neural policy for training')
        return result


@dataclass(frozen=True)
class EvaluationConfig:
    episodes: int = 5
    max_steps: Optional[int] = 4500
    seed: int = 0
    deterministic: bool = True

    def __post_init__(self):
        if self.episodes < 1 or (self.max_steps is not None and self.max_steps < 1):
            raise ValueError('Evaluation episodes and max_steps must be positive')

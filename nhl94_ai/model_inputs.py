"""Reusable named observation definitions, independent of tasks and policies."""
import copy
import json
from functools import lru_cache
from importlib.resources import files
from pathlib import Path


def default_model_input_path():
    return str(files('nhl94_ai').joinpath('data/model_input.json'))


def _read_registry(path):
    registry = json.loads(Path(path).expanduser().read_text(encoding='utf-8'))
    if not isinstance(registry, dict):
        raise TypeError('Model input registry must be a JSON object.')
    if 'default' not in registry:
        raise ValueError('Model input registry requires a default variant.')
    for name, definition in registry.items():
        if not isinstance(definition, dict) or not isinstance(definition.get('groups'), dict):
            raise TypeError(f'Model input variant {name!r} requires a groups object.')
        version = definition.get('schema_version')
        if isinstance(version, bool) or not isinstance(version, int) or version < 1:
            raise ValueError(f'Model input variant {name!r} requires a positive schema_version.')
    return registry


@lru_cache(maxsize=1)
def _packaged_registry():
    return _read_registry(default_model_input_path())


def load_model_input(name='default', path=None):
    registry = _read_registry(path) if path else _packaged_registry()
    if name not in registry:
        raise ValueError(f'Unknown model input variant {name!r}; available: {", ".join(registry)}')
    definition = copy.deepcopy(registry[name])
    definition['variant'] = name
    definition['registry_path'] = str(Path(path or default_model_input_path()).expanduser().resolve())
    return definition


def resolve_model_input(args, hyperparams, *, task=None):
    """CLI selection overrides a task default; old inline definitions still work."""
    params = copy.deepcopy(hyperparams or {})
    configured = params.get('model_input')
    if configured is not None and not isinstance(configured, (str, dict)):
        raise TypeError('model_input must be a variant name or an inline JSON object.')
    selected = getattr(args, 'model_input', None)
    path = getattr(args, 'model_input_config', None)
    if selected is None:
        if isinstance(configured, dict) and 'variant' not in configured:
            return params
        selected = getattr(task, 'model_input_variant', None)
        if selected is None:
            selected = configured.get('variant', 'default') if isinstance(configured, dict) else (
                configured if isinstance(configured, str) else 'default'
            )
    if not isinstance(selected, str) or not selected:
        raise TypeError('model_input selection must be a nonempty variant name.')
    same_registry = not path or (
        isinstance(configured, dict)
        and configured.get('registry_path') == str(Path(path).expanduser().resolve())
    )
    if isinstance(configured, dict) and configured.get('variant') == selected and same_registry:
        params['model_input'] = copy.deepcopy(configured)
    else:
        params['model_input'] = load_model_input(selected, path)
    return params


def add_model_input_arguments(parser):
    parser.add_argument('--model_input', default=None,
                        help='Named observation variant; overrides the task default')
    parser.add_argument('--model_input_config', default=None,
                        help='JSON registry of named observation variants')
    return parser

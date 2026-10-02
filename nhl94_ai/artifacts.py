"""Versioned model and dataset contracts. Older files remain explicitly unknown."""
from dataclasses import asdict, dataclass, is_dataclass
import importlib.metadata
import hashlib
import json
from pathlib import Path
import subprocess
import sys

FORMAT_VERSION = 1


def _json_value(value):
    if is_dataclass(value):
        return _json_value(asdict(value))
    if isinstance(value, dict):
        return {str(k): _json_value(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(v) for v in value]
    if hasattr(value, 'tolist'):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return value


def run_metadata(args, hyperparams=None):
    options = dict(vars(args)) if hasattr(args, '__dict__') else dict(args)
    params = hyperparams if hyperparams is not None else options.get('hyperparams_dict', {})
    from nhl94_ai.env.encoding import _normalize_model_input_config
    schema = {
        'game': options.get('env'), 'policy': options.get('nn'),
        'observation_fields': _normalize_model_input_config(params.get('model_input')),
        'observation_schema_version': (params.get('model_input') or {}).get('schema_version', 1),
        'action_type': options.get('action_type', 'FILTERED'),
        'seq_len': options.get('seq_len', 16), 'frame_skip': 1 if options.get('no_frame_skip', False) else params.get('frame_skip', 4),
        'normalization': 'nhl94-game-constants-v1',
    }
    if schema['action_type'].upper() == 'TARGET_POSITION':
        from nhl94_ai.env.target_control import target_schema
        from nhl94_ai.tasks.registry import get_task
        schema['action_type'] = 'TARGET_POSITION'
        task = get_task(options['rf'])
        schema['target_controller'] = target_schema(task.target_control, task.target_bounds)
    versions = {}
    for package in ('nhl94-ai', 'stable-retro', 'stable-baselines3', 'torch', 'numpy'):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    digest = hashlib.sha256()
    package = Path(__file__).resolve().parent
    for source in sorted(package.rglob("*.py")):
        digest.update(str(source.relative_to(package)).encode("utf-8"))
        digest.update(source.read_bytes())
    revision = None
    dirty = None
    root = Path(__file__).resolve().parents[1]
    if (root / '.git').exists():
        result = subprocess.run(['git', '-C', str(root), 'rev-parse', 'HEAD'], capture_output=True, text=True, check=False)
        revision = result.stdout.strip() or None
        dirty = bool(subprocess.run(['git', '-C', str(root), 'status', '--porcelain'], capture_output=True, text=True, check=False).stdout)
    return _json_value({'format_version': FORMAT_VERSION, 'schema': schema, 'config': options,
                        'hyperparams': params, 'seed': options.get('seed', 0), 'policy_seed': options.get('policy_seed'),
                        'source_sha256': digest.hexdigest(), 'code_revision': revision, 'code_dirty': dirty, 'versions': versions, 'python': sys.version.split()[0]})


def read_metadata(path):
    sidecar = Path(str(path) + '.json')
    if not sidecar.exists():
        return {'compatibility': 'unknown'}
    data = json.loads(sidecar.read_text(encoding='utf-8'))
    if 'format_version' not in data:
        return {'compatibility': 'unknown', 'legacy_metadata': data}
    if data.get('format_version') != FORMAT_VERSION:
        raise ValueError(f'Unsupported artifact metadata version: {sidecar}')
    return data


def validate_schema(actual, expected):
    if actual.get('compatibility') == 'unknown':
        raise ValueError('Artifact schema is unknown; supply a verified legacy configuration')
    for key, value in expected['schema'].items():
        if key == 'policy':
            continue  # Different architectures may share observations and actions.
        if actual['schema'].get(key) != value:
            raise ValueError(f'Incompatible artifact {key}: {actual["schema"].get(key)!r} != {value!r}')


def save_checkpoint(model, path, args=None, hyperparams=None):
    model.save(path)
    artifact = Path(str(path) if str(path).endswith('.zip') else str(path) + '.zip')
    metadata = run_metadata(args, hyperparams) if args is not None else getattr(model, '_nhl94_metadata', None)
    if metadata is not None:
        Path(str(artifact) + '.json').write_text(json.dumps(metadata, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    return str(artifact)


def load_policy(path, *, expected=None, require_metadata=False, **kwargs):
    from nhl94_ai.compat import install_checkpoint_aliases
    from stable_baselines3 import PPO
    install_checkpoint_aliases()
    artifact = Path(path).expanduser()
    if not artifact.exists() and artifact.suffix != '.zip':
        artifact = Path(str(artifact) + '.zip')
    metadata = read_metadata(artifact)
    target_requested = expected is not None and expected['schema'].get('action_type') == 'TARGET_POSITION'
    if target_requested and metadata.get('compatibility') == 'unknown':
        raise ValueError('TARGET_POSITION requires checkpoint metadata; legacy weights cannot be reinterpreted as targets')
    if metadata.get('schema', {}).get('action_type') == 'TARGET_POSITION':
        from nhl94_ai.env.target_control import target_schema
        controller = metadata['schema'].get('target_controller')
        if not controller or controller != target_schema(controller.get('profile'), controller.get('bounds')):
            raise ValueError('Incompatible target controller settings or observation schema')
    if require_metadata and metadata.get('compatibility') == 'unknown':
        raise ValueError(f'No schema metadata for legacy checkpoint: {artifact}')
    if expected is not None and metadata.get('compatibility') != 'unknown':
        validate_schema(metadata, expected)
    model = PPO.load(str(artifact), **kwargs)
    model._nhl94_metadata = metadata
    return model


def ensure_zip_path(path: str) -> str:
    """Append the SB3 .zip suffix if the path doesn't already include it."""

    return path if path.endswith(".zip") else f"{path}.zip"


@dataclass(frozen=True)
class Dataset:
    path: str
    metadata: dict


@dataclass
class CurriculumResult:
    final_model_path: str
    phases: list

    def __iter__(self):
        # Historical callers unpacked the two-tuple.
        return iter((self.final_model_path, self.phases))

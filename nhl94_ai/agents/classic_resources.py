"""Read-only artifacts retained by one Classic stream across episode resets."""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any


class FrozenDict(dict):
    """JSON-compatible immutable mappings; nested sequences are stored as tuples."""
    def _immutable(self, *_args, **_kwargs):
        raise TypeError('Loaded artifacts are immutable')

    __setitem__ = __delitem__ = clear = pop = popitem = setdefault = update = __ior__ = _immutable

    def __deepcopy__(self, memo):
        return self

    def __reduce__(self):
        return type(self), (dict(self),)


def _freeze(value):
    if isinstance(value, dict):
        return FrozenDict({key: _freeze(child) for key, child in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(child) for child in value)
    return value


@dataclass(frozen=True)
class JsonArtifact:
    path: str | None
    sha256: str | None
    data: FrozenDict

    def __deepcopy__(self, memo):
        return self


def load_json_artifact(path=None, *, artifact=None):
    """One read supplies hashing, parsing, and version/schema dispatch."""
    source = Path(path) if path else None
    raw = source.read_bytes() if source is not None else None
    data = artifact if artifact is not None else json.loads(raw)
    return JsonArtifact(str(source.resolve()) if source is not None else None,
                        hashlib.sha256(raw).hexdigest() if raw is not None else None, _freeze(data))


@dataclass(frozen=True)
class GoalieArtifact:
    path: str
    requested_interval: int | None
    model: Any


@dataclass(frozen=True)
class ClassicResources:
    retention: JsonArtifact | None = None
    shot: JsonArtifact | None = None
    goalie: GoalieArtifact | None = None

    def copy_memo(self):
        # Inference weights are shared only within this stream's replay forks.
        # Commands, observations, cooldowns and metrics belong to fresh controllers.
        return {id(self): self, **({id(self.goalie.model): self.goalie.model} if self.goalie else {})}

    def __deepcopy__(self, memo):
        memo.update(self.copy_memo())
        return self


def _json_resource(path, previous):
    if not path:
        return None
    if previous is not None and previous.path == str(Path(path).resolve()):
        return previous
    return load_json_artifact(path)


def load_resources(options, previous=None):
    """Reuse pinned resources; an explicitly changed path creates a new bundle."""
    previous = previous or ClassicResources()
    goalie = None
    if options.goalie_model:
        path = str(Path(options.goalie_model).resolve())
        old = previous.goalie
        if old is not None and (old.path, old.requested_interval) == (path, options.goalie_decision_interval):
            goalie = old
        else:
            import torch
            from nhl94_ai.models.goalie import load_model
            torch.set_num_threads(1)
            goalie = GoalieArtifact(path, options.goalie_decision_interval,
                                    load_model(options.goalie_model, interval=options.goalie_decision_interval))
    return ClassicResources(_json_resource(options.retention_model, previous.retention),
                            _json_resource(options.shot_placement, previous.shot), goalie)

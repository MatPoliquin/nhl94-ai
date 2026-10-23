"""Per-call Classic parameters for isolated optimization experiments.

Ordinary gameplay uses DEFAULT. The context belongs to the calling controller,
so a tuned agent cannot change its opponent's settings. ROM attributes, physics,
button timings and action-lifecycle checks remain fixed.
"""
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass, fields
import math


@dataclass(frozen=True)
class ClassicTuning:
    normal_shot_depth: float = 218
    shot_accuracy_depth_scale: float = 0.4
    normal_shot_half_width: float = 48
    carry_lane_offset: float = 35
    carry_target_depth: float = 235
    position_pass_min_improvement: float = 12
    advance_pass_min_progress: float = 20
    pass_progress_weight: float = 0.4
    pass_bypassed_defender_weight: float = 16
    pass_receiver_shot_weight: float = 0.6
    defensive_gap_scale: float = 0.18
    defender_switch_margin: float = 4


DEFAULT = ClassicTuning()
BOUNDS = (
    (205, 230), (0.1, 0.8), (32, 60), (20, 50), (225, 250),
    (4, 24), (10, 35), (0.2, 0.8), (8, 28), (0.3, 1.2),
    (0.10, 0.30), (1, 10),
)
NAMES = tuple(field.name for field in fields(ClassicTuning))
_CURRENT = ContextVar('classic_tuning', default=DEFAULT)


def current():
    return _CURRENT.get()


def validate(params):
    if not isinstance(params, ClassicTuning):
        raise TypeError('Expected immutable ClassicTuning parameters')
    for name, (low, high) in zip(NAMES, BOUNDS):
        value = getattr(params, name)
        if isinstance(value, bool) or not math.isfinite(value) or not low <= value <= high:
            raise ValueError(f'{name} must be finite and within [{low}, {high}]')
    if params.normal_shot_depth + 15 * params.shot_accuracy_depth_scale + 2 > params.carry_target_depth:
        raise ValueError('Carry target must exceed the worst-accuracy shooting threshold by two units')
    return params


def decode(vector):
    if len(vector) != len(NAMES):
        raise ValueError(f'Expected {len(NAMES)} normalized coordinates')
    if any(not math.isfinite(float(value)) or not 0 <= value <= 1 for value in vector):
        raise ValueError('Normalized coordinates must be finite and within [0, 1]')
    return validate(ClassicTuning(**{
        name: float(low + value * (high - low))
        for name, value, (low, high) in zip(NAMES, vector, BOUNDS)
    }))


def encode(params=DEFAULT):
    validate(params)
    return [(getattr(params, name) - low) / (high - low)
            for name, (low, high) in zip(NAMES, BOUNDS)]


def load_params(mapping):
    if set(mapping) != set(NAMES):
        raise ValueError('Parameter artifact must contain exactly the declared tuning fields')
    return validate(ClassicTuning(**mapping))


@contextmanager
def scope(params):
    """Validate at load time rather than every emulator frame."""
    token = _CURRENT.set(params)
    try:
        yield
    finally:
        _CURRENT.reset(token)


def specification():
    return [{'name': name, 'default': asdict(DEFAULT)[name], 'bounds': list(bounds)}
            for name, bounds in zip(NAMES, BOUNDS)]

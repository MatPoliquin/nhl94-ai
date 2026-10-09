"""Explicit history export for archived Classic replay implementations.

Compatibility is confined to evaluation: the live controller has no aliases for
its former private fields. Old snapshots still need their original field layout
when a current trajectory is forked through their policy.
"""
from copy import deepcopy


LEGACY_FIELDS = {
    '_tick': 'scheduler.decisions', '_frame_remaining': 'scheduler.remaining',
    '_frame_action': 'scheduler.action', '_decision_interval': 'scheduler.interval',
    '_defense_elapsed': 'scheduler.elapsed', '_was_defending': 'scheduler.was_defending',
    '_b_down': 'buttons.b_down', '_c_down': 'buttons.c_down',
    '_switch_at': 'recovery_switch_at_decision',
    '_shot_until': 'shot.until_decision', '_shot_hold_until': 'shot.hold_until_frame',
    '_shots_before': 'shot.shots_before', '_shot_slot': 'shot.slot', '_shot_side': 'shot.side',
    '_setup_until': 'setup.until_decision', '_setup_at': 'setup.retry_at_decision', '_setup_slot': 'setup.slot',
    '_pass_at': 'one_timer.retry_at_frame', 'one_timer_metrics': 'one_timer.metrics',
    'one_timer_starts': 'one_timer.starts', '_one_timer_option': 'one_timer.option',
    '_one_timer_flight': 'one_timer.flight_frames',
}
ONE_TIMER_FIELDS = {
    '_one_timer_started': 'started', '_one_timer_passes_before': 'passes_before',
    '_one_timer_attempts_before': 'attempts_before', '_one_timer_shots_before': 'shots_before',
    '_one_timer_actual': 'actual_receiver', '_one_timer_launched': 'launched',
}


def _attribute(value, path):
    for part in path.split('.'):
        value = getattr(value, part)
    return value


def copy_history(history, policy):
    """Preserve target policy implementations while copying source action state."""
    values = deepcopy(vars(history))
    offense_values = values.pop('offense').__dict__
    offense = policy.offense
    if hasattr(policy, 'scheduler'):
        for name in ('scheduler', 'buttons', 'shot', 'one_timer', 'setup'):
            getattr(policy, name).__dict__.update(vars(values.pop(name)))
        offense.pass_action.__dict__.update(vars(offense_values.pop('pass_action')))
    else:
        for legacy, path in LEGACY_FIELDS.items():
            values[legacy] = deepcopy(_attribute(history, path))
        pending = history.one_timer.pending
        values['_one_timer'] = (pending.passer, pending.receiver, pending.deadline) if pending else None
        if pending is not None:
            values.update({old: deepcopy(getattr(pending, field)) for old, field in ONE_TIMER_FIELDS.items()})
        for name in ('scheduler', 'buttons', 'shot', 'one_timer', 'setup', 'recovery_switch_at_decision'):
            values.pop(name, None)
        # Defense may have been idle when a replay constructed this history.
        values['defense'].frames = history.scheduler.frames
        passing = offense_values.pop('pass_action')
        offense_values.update(pending=passing.pending.snapshot() if passing.pending else None,
                              last_pass=passing.last_pass, last_request=passing.last_request)
    policy.__dict__.update(values)
    offense.__dict__.update(offense_values)
    policy.offense = offense


def cache_input(model, action, remaining):
    if hasattr(model, 'scheduler'):
        model.scheduler.cache(action, remaining)
    else:
        model._frame_action = action
        model._frame_remaining = remaining

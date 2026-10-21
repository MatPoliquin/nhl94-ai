"""Explicit history export for archived Classic replay implementations.

Compatibility is confined to evaluation: the live controller has no aliases for
its former private fields. Old snapshots still need their original field layout
when a current trajectory is forked through their policy.
"""


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
    """Use the live history contract, adapting only the archived field layout."""
    from nhl94_ai.agents.history import CONTROLLER_FIELDS, COMPONENT_FIELDS, export_history, import_history, fork_history
    if hasattr(policy, 'scheduler'):
        return import_history(policy, export_history(history))
    source = fork_history(history)
    for name in CONTROLLER_FIELDS:
        if hasattr(source, name):
            setattr(policy, name, getattr(source, name))
    for name in COMPONENT_FIELDS:
        if '.' not in name and name not in ('scheduler', 'buttons', 'shot', 'one_timer', 'setup', 'offense'):
            setattr(policy, name, getattr(source, name))
    for legacy, path in LEGACY_FIELDS.items():
        setattr(policy, legacy, _attribute(source, path))
    pending = source.one_timer.pending
    policy._one_timer = (pending.passer, pending.receiver, pending.deadline) if pending else None
    if pending is not None:
        for old, field in ONE_TIMER_FIELDS.items():
            setattr(policy, old, getattr(pending, field))
    policy.defense.frames = source.scheduler.frames
    for name in COMPONENT_FIELDS['offense']:
        setattr(policy.offense, name, getattr(source.offense, name))
    passing = source.offense.pass_action
    policy.offense.pending = passing.pending.snapshot() if passing.pending else None
    policy.offense.last_pass, policy.offense.last_request = passing.last_pass, passing.last_request
    return policy


def cache_input(model, action, remaining):
    if hasattr(model, 'scheduler'):
        model.scheduler.cache(action, remaining)
    else:
        model._frame_action = action
        model._frame_remaining = remaining

"""Classic's in-process replay history, with an explicit set of state owners.

This is not a serialized checkpoint format. Component implementations belong to
the receiving policy; clocks, commitments, diagnostics and runtime settings come
from the captured history. Environment handles and injected observers never do.
"""
from copy import deepcopy
from dataclasses import dataclass, fields, is_dataclass


CONTROLLER_FIELDS = (
    'args', 'options', 'resources', 'platform', 'refinements', '_one_timers', '_intents', '_size',
    'recovery_switch_at_decision', 'input_owner', '_last_decision', '_last_target',
    '_last_action_preferences', '_last_pass_request', 'carry_metrics',
    'defense_diagnostics', 'offense_diagnostics', 'goalie_diagnostics',
    # The learned-offense controller adds command/teacher state to Classic.
    'control_scope', 'mode', 'interval', 'command', 'teacher_command',
    'teacher_allowed', 'command_used', 'last_buttons',
)

# Dataclass lifecycles already declare their complete field contract. Other
# owners list their replay state here, including their existing runtime settings.
COMPONENT_FIELDS = {
    'scheduler': (), 'buttons': (), 'shot': (), 'one_timer': (), 'setup': (),
    'offense': (
        'one_timers', 'decision_interval', 'allow_uncertified', 'chance_creation',
        'chance_owner', 'chance_until', 'chance_at', 'one_timer_at', 'pass_at',
        'feint_until', 'feint_at', 'feint_target', 'feint_mode', 'diagnostics',
        'risk_cache', 'pass_timing', 'carry_motion', 'rank_pass_risk'),
    'offense.pass_action': (),
    'defense': (
        'verify_interceptions', 'frames', 'plan', 'desired_slot', 'switch_at', 'poke_at',
        'boost_at', 'switch_attempts', 'pending_switch', 'pending_check', 'last_switch_result',
        'last_selection', 'b_down', 'c_down', 'last_owner', 'diagnostics'),
    'goalie': (
        'policy', 'platform', 'pass_timing', 'save_timing', 'frames', 'phase', 'started',
        'retry_at', 'save_at', 'release_until', 'possession_since', 'outlet', 'pending_save',
        'pass_windup', 'incoming_pass', 'last_anim', 'last_effective', 'last_event', 'last_owner',
        'last_passes', 'hold_sample', 'hold_stride', 'hold_tick', 'metrics', 'diagnostics',
        'command', 'commands_used', 'cached', 'interval', 'next_choice', 'model',
        'planning', 'planner_times_ns'),
    'cross_crease': (
        'aim_side', 'committed', 'diagnostics', 'event', 'events', 'metrics', 'phase',
        'plan', 'pressed', 'release_at', 'retry_at', 'started', 'waypoint'),
    'deke': ('diagnostics', 'event', 'events', 'metrics', 'phase', 'phase_at', 'plan',
             'pressed', 'retry_at', 'started'),
    'possession_value': ('diagnostics', 'interval', 'metrics', 'one_timer_at', 'one_timers',
                         'pass_timing', 'refine_finishing'),
    'possession_ablation': ('continuations', 'diagnostics', 'interval', 'metrics', 'one_timer_at',
                           'one_timers', 'pass_timing', 'profile', 'refine_finishing', 'relaxed_risk'),
    'receiver_selector': ('diagnostics', 'expand_pool', 'forced_slot', 'interval', 'metrics',
                          'one_timer_at', 'one_timers', 'pass_timing', 'profile', 'refine_finishing'),
    'crossing_finish': ('diagnostics', 'events', 'metrics'),
    'crossing_finish.probe': ('attempts', 'diagnostics', 'direction', 'opportunity', 'pad',
                              'phase', 'prepare_release', 'pressed', 'release_estimate', 'select_release',
                              'settings', 'target', 'target_frame', 'traffic_aware', 'visited', 'windup'),
    'shot_placement': ('bias', 'diagnostics', 'fixed_hold', 'mean', 'metrics', 'min_gain',
                       'model', 'names', 'scale', 'sha256', 'weights'),
    'retention_filter': ('artifact', 'base_score', 'metrics', 'minimum_gap', 'minimum_margin',
                        'minimum_risk', 'names', 'path', 'sha256', 'tree', 'trees'),
}


@dataclass
class ComponentHistory:
    implementation: type
    values: dict
    binds_offense: bool = False


@dataclass
class ClassicHistory:
    implementation: type
    controller: dict
    components: dict[str, ComponentHistory | None]


def _owner(policy, path):
    for name in path.split('.'):
        policy = getattr(policy, name, None)
        if policy is None:
            break
    return policy


def export_history(policy):
    """Capture one isolated graph, preserving aliases between diagnostic payloads."""
    components = {}
    for path, names in COMPONENT_FIELDS.items():
        owner = _owner(policy, path)
        if owner is None:
            components[path] = None
            continue
        names = tuple(field.name for field in fields(owner)) if is_dataclass(owner) else names
        values = {name: getattr(owner, name) for name in names if hasattr(owner, name)}
        components[path] = ComponentHistory(type(owner), values, hasattr(owner, 'offense'))
    values = {name: getattr(policy, name) for name in CONTROLLER_FIELDS if hasattr(policy, name)}
    resources = getattr(policy, 'resources', None)
    return deepcopy(ClassicHistory(type(policy), values, components), resources.copy_memo() if resources else None)


def import_history(policy, history):
    """Retain existing executor objects; isolate the history of each replay branch."""
    resources = history.controller.get('resources')
    history = deepcopy(history, resources.copy_memo() if resources else None)
    for name, value in history.controller.items():
        setattr(policy, name, value)
    for path, captured in history.components.items():
        parent_path, _, name = path.rpartition('.')
        parent = _owner(policy, parent_path) if parent_path else policy
        if parent is None:
            continue
        if captured is None:
            setattr(parent, name, None)
            continue
        owner = getattr(parent, name, None)
        if owner is None:
            owner = captured.implementation.__new__(captured.implementation)
            setattr(parent, name, owner)
        for field, value in captured.values.items():
            setattr(owner, field, value)
        if captured.binds_offense:
            owner.offense = policy.offense
    return policy


def fork_history(policy):
    """Fork the policy implementation and history, excluding patched observers."""
    branch = type(policy).__new__(type(policy))
    branch.env = policy.env
    return import_history(branch, export_history(policy))

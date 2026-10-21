"""Immutable Classic settings and the shared compatibility boundary.

Playback resolves its config-file overrides before entering here. Direct callers
retain their legacy timing and automatic-goalie defaults; benchmarks supply their
own explicit settings. Loading models and resetting execution are separate jobs.
"""
from dataclasses import dataclass, fields, replace


@dataclass(frozen=True)
class ClassicOptions:
    env: str = 'NHL94-Genesis-v0'
    action_type: str = 'FILTERED'
    selfplay: bool = False
    one_timers: bool = True
    classic_decision_interval: int | None = None
    classic_control_interval: int = 0
    player_ratings_consideration: bool = True
    classic_refinements: frozenset[str] = frozenset()
    offense_lookahead: bool = False
    uncertain_carry: bool = False
    chance_creation: bool = False
    cross_crease: bool = False
    deke: bool = False
    possession_value: bool = False
    possession_ablation: str | None = None
    receiver_selection: str | None = None
    shot_placement: str | None = None
    reception_control: bool = False
    one_timer_execution: str | None = None
    rebound_recovery: bool = False
    opportunistic_cross_crease: bool = False
    retention_model: str | None = None
    goalie_policy: str = 'off'
    goalie_positioning: str | None = None
    goalie_save_timing: str = 'classic'
    goalie_model: str | None = None
    goalie_decision_interval: int | None = None
    goalie_planner: str | None = None

    @classmethod
    def from_args(cls, args):
        values = {field.name: getattr(args, field.name, field.default) for field in fields(cls)}
        values['action_type'] = values['action_type'].upper()
        values['classic_refinements'] = frozenset(values['classic_refinements'])
        return cls(**values)

    @property
    def effective_goalie_positioning(self):
        return resolve_goalie_positioning(self)


def resolve_goalie_positioning(args):
    """Preserve explicit overrides, including classic and disabled manual control."""
    explicit = getattr(args, 'goalie_positioning', None)
    if explicit is not None:
        return explicit
    if getattr(args, 'env', '') == 'NHL94-Snes-v0':
        return 'classic'
    replaced = (getattr(args, 'goalie_model', None) or getattr(args, 'goalie_planner', None)
                or getattr(args, 'goalie_decision_interval', None) not in (None, 1)
                or getattr(args, 'goalie_save_timing', 'classic') != 'classic')
    return ('deep-csave' if getattr(args, 'goalie_policy', 'off') in ('selective', 'always')
            and not replaced else 'classic')


OFFENSIVE_EXPERIMENTS = (
    ('opportunistic_cross_crease', 'Opportunistic crossing'), ('rebound_recovery', 'Rebound recovery'),
    ('one_timer_execution', 'One-timer execution'), ('reception_control', 'Reception control'),
    ('shot_placement', 'Shot placement'), ('receiver_selection', 'Receiver selection'),
    ('possession_ablation', 'Possession ablation'), ('possession_value', 'Possession-value'),
    ('cross_crease', 'Cross-crease'), ('deke', 'Deke'), ('uncertain_carry', 'Uncertain-carry'),
    ('chance_creation', 'Chance-creation'), ('offense_lookahead', 'Offense-lookahead'),
    ('classic_refinements', 'Classic refinements'),
)


def _offense_compatibility(options, classic, *, application):
    from nhl94_ai.agents.lifecycle import validate_one_timer_execution, validate_rebound_recovery
    from nhl94_ai.agents.crossing_finish import validate_crossing_finish
    from nhl94_ai.agents.receiver_selection import INCOMPATIBLE as RECEIVER_INCOMPATIBLE
    from nhl94_ai.agents.shot_placement import INCOMPATIBLE as SHOT_INCOMPATIBLE
    from nhl94_ai.agents.receiving import INCOMPATIBLE as RECEPTION_INCOMPATIBLE
    validate_one_timer_execution(options)
    validate_rebound_recovery(options)
    validate_crossing_finish(options)
    full_team_controls = classic and options.env == 'NHL94-Genesis-v0' and options.action_type in ('FILTERED', 'HOCKEY_INTENT_DPAD')
    for flag, label in OFFENSIVE_EXPERIMENTS:
        # Legacy direct callers construct these policies against supplied states,
        # independently of the CLI's selected environment (including its default).
        if not application and flag in ('offense_lookahead', 'uncertain_carry', 'classic_refinements'):
            continue
        if getattr(options, flag) and (not full_team_controls or application and options.selfplay):
            raise ValueError(f'{label} AI requires full-team Classic buttons/intents without self-play')
    conflicts = (
        ('possession_value', ('offense_lookahead', 'uncertain_carry', 'chance_creation'),
         'Possession-value replaces offense-lookahead, uncertain-carry and chance-creation selection'),
        ('possession_ablation', ('possession_value', 'offense_lookahead', 'uncertain_carry',
                                'chance_creation', 'cross_crease', 'deke', 'classic_refinements'),
         'Possession ablations isolate legacy proposals; disable other offensive experiments/refinements'),
        ('receiver_selection', RECEIVER_INCOMPATIBLE,
         'Receiver selection isolates legacy passes; disable other experiments/refinements'),
        ('shot_placement', SHOT_INCOMPATIBLE,
         'Shot placement isolates normal-shot execution; disable other offensive experiments'),
        ('reception_control', RECEPTION_INCOMPATIBLE,
         'Reception control isolates ordinary-pass execution; disable other offensive experiments'),
        ('retention_model', ('possession_value', 'possession_ablation', 'receiver_selection', 'shot_placement',
                             'reception_control', 'one_timer_execution', 'rebound_recovery', 'classic_refinements',
                             'offense_lookahead', 'uncertain_carry', 'chance_creation', 'cross_crease', 'deke'),
         'Retention filtering cannot combine with other offensive experiments'),
    )
    for flag, incompatible, message in conflicts:
        if getattr(options, flag) and any(getattr(options, name) for name in incompatible):
            raise ValueError(message)
    if options.retention_model and (options.action_type != 'FILTERED' or options.classic_control_interval):
        raise ValueError('Retention filtering requires FILTERED and native Classic reactions')


def _goalie_compatibility(options):
    policy, interval = options.goalie_policy, options.goalie_decision_interval
    positioning = options.effective_goalie_positioning
    if policy not in ('off', 'selective', 'always'):
        raise ValueError('goalie_policy must be off, selective or always')
    if policy != 'off' and options.action_type != 'FILTERED':
        raise ValueError('Manual goalie AI requires FILTERED buttons')
    if positioning not in ('classic', 'deep', 'deep-csave'):
        raise ValueError('goalie_positioning must be classic, deep or deep-csave')
    if positioning in ('deep', 'deep-csave') and (policy == 'off' or options.goalie_model
            or interval not in (None, 1) or options.goalie_planner):
        raise ValueError('Deep goalie positioning requires manual goalies and no other goalie controller')
    if positioning == 'deep-csave' and options.goalie_save_timing != 'classic':
        raise ValueError('Deep C-save positioning already includes its own save timing')
    if (options.goalie_model or interval is not None or options.goalie_save_timing != 'classic') and policy == 'off':
        raise ValueError('Goalie model/timing options require --goalie-policy selective or always')
    if interval is not None and interval < 1:
        raise ValueError('Goalie decision interval must be positive')
    if options.goalie_model and options.goalie_save_timing != 'classic':
        raise ValueError('Learned goalie and scripted contact timing are separate controllers')
    another_controller = options.goalie_model or interval not in (None, 1) or options.goalie_save_timing != 'classic'
    if options.goalie_planner and (policy == 'off' or another_controller or options.classic_control_interval):
        raise ValueError('Goalie planning requires manual goalies, native reactions and no other goalie experiment')
    if options.env == 'NHL94-Snes-v0' and (positioning != 'classic' or another_controller or options.goalie_planner):
        raise ValueError('SNES manual goalies use shared classic positioning with native SNES saves')


def validate_classic_options(args, *, application=None, benchmark=False, classic=True, scripted_policy=True):
    """Normalize once and validate CLI, benchmark, and direct construction alike."""
    from nhl94_ai.agents.control_rate import control_interval
    options = ClassicOptions.from_args(args)
    if application is not None:
        from nhl94_ai.game.specs import get_game
        options = replace(options, env=application.env, action_type=application.action_type,
                          selfplay=application.selfplay, goalie_policy=options.goalie_policy or 'off')
        supported = (scripted_policy and get_game(options.env).skaters_per_team == 5
                     and options.action_type == 'FILTERED' and not options.selfplay
                     and getattr(args, 'rf', 'PostPlay') == 'PostPlay')
        if options.goalie_policy != 'off' and not supported:
            raise ValueError('Manual goalie AI requires full-team Classic FILTERED PostPlay without self-play')
        if (options.goalie_model or options.goalie_decision_interval is not None
                or options.goalie_save_timing != 'classic') and not supported:
            raise ValueError('Goalie model/timing options require enabled full-team Classic manual goalie control')
    control_interval(options)
    _offense_compatibility(options, classic, application=application is not None)
    _goalie_compatibility(options)
    if benchmark and options.retention_model and getattr(args, 'frame_skip', 4) != 4:
        raise ValueError('Retention filtering requires four-frame offense and native reactions')
    return options


def validate_platform_application(args, *, platform, classic):
    if platform != 'snes':
        return
    policy = getattr(args, 'nn', '')
    human = getattr(args, 'mode', None) == 'player_vs_game'
    if (not classic and not (human and policy == 'MlpPolicy')) or getattr(args, 'rf', 'PostPlay') not in ('', 'PostPlay'):
        raise ValueError('SNES currently supports Classic PostPlay; RAM-writing training tasks and learned policies require a SNES contract')
    if any(getattr(args, name, None) for name in ('model', 'model_1', 'model_2', 'load_p1_model', 'load_p2_model', 'retention_model')):
        raise ValueError('Genesis learned-policy artifacts cannot be loaded as SNES policies')
    model_input = getattr(args, 'model_input', 'default')
    if isinstance(model_input, str) and model_input.startswith('pvg'):
        raise ValueError('The compact PvG observation layout requires Genesis ROM attributes')
    if (getattr(args, 'action_type', 'FILTERED').upper() != 'FILTERED'
            or getattr(args, 'selfplay', False)):
        raise ValueError('SNES Classic requires FILTERED controls without self-play')
    mode = getattr(args, 'mode', 'model_vs_game')
    if mode not in ('model_vs_game', 'player_vs_game', 'player_vs_model', 'model_vs_model'):
        raise ValueError('Unsupported SNES playback mode')
    players = getattr(args, 'num_players', 1)
    if players != (2 if mode in ('player_vs_model', 'model_vs_model') else 1):
        raise ValueError('SNES playback requires one controller versus CPU or two for multiplayer')
    if getattr(args, 'mode', None) is not None and getattr(args, 'seed', None) is not None:
        raise ValueError('SNES ROM RNG seeding is not mapped; omit --seed to use the saved RNG')

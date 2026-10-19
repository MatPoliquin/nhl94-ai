"""Player-facing Classic settings and a ratings-neutral decision view."""
from copy import copy
import json
from pathlib import Path
import re

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / 'configs' / 'classic_bot.json'
DEFAULTS = {'decision_hz': 60, 'ai_goalie': True, 'player_ratings_consideration': True}
DECISION_RATES = (60, 15, 10, 6)
_JSON_TOKENS = re.compile(r'"(?:\\.|[^"\\])*"|//[^\r\n]*|/\*[\s\S]*?\*/')


def load_classic_config(path=None):
    """Allow JSON comments, but reject typos and values with the wrong type."""
    source = Path(path).expanduser() if path is not None else DEFAULT_CONFIG_PATH
    if path is None and not source.is_file():
        # Installed wheels need not have the checkout's editable configs folder.
        return dict(DEFAULTS)
    text = source.read_text(encoding='utf-8')
    text = _JSON_TOKENS.sub(
        lambda match: match[0] if match[0].startswith('"') else
        ''.join('\n' if char == '\n' else ' ' for char in match[0]), text)
    try:
        values = json.loads(text)
    except json.JSONDecodeError as error:
        raise ValueError(f'Invalid Classic config {source}: {error}') from error
    if not isinstance(values, dict):
        raise ValueError('Classic config must contain a JSON object')
    unknown = values.keys() - DEFAULTS.keys()
    if unknown:
        raise ValueError('Unknown Classic config setting(s): ' + ', '.join(sorted(unknown)))
    config = DEFAULTS | values
    if (isinstance(config['decision_hz'], bool) or not isinstance(config['decision_hz'], int)
            or config['decision_hz'] not in DECISION_RATES):
        raise ValueError('decision_hz must be one of 60, 15, 10, 6')
    for name in ('ai_goalie', 'player_ratings_consideration'):
        if not isinstance(config[name], bool):
            raise ValueError(f'{name} must be true or false')
    return config


def apply_play_config(args):
    """Resolve the file once, before playback validates or constructs agents."""
    from nhl94_ai.agents.registry import CONTROLLERS
    if getattr(args, '_classic_config_applied', False):
        return
    classic = (args.nn in CONTROLLERS and args.mode != 'player_vs_game'
               and not getattr(args, 'model_1', '') and not getattr(args, 'model_2', ''))
    if classic:
        config = load_classic_config(getattr(args, 'classic_config', None))
        if getattr(args, 'classic_control_interval', None) is None:
            # 60 Hz retains native reactions and the existing tactical scheduler.
            args.classic_control_interval = 0 if config['decision_hz'] == 60 else 60 // config['decision_hz']
        if not config['ai_goalie'] and getattr(args, 'goalie_policy', None) is None:
            args.goalie_policy = 'off'
        args.player_ratings_consideration = config['player_ratings_consideration']
    elif getattr(args, 'classic_config', None) is not None:
        raise ValueError('--classic-config requires Classic playback without learned models')
    if getattr(args, 'classic_control_interval', None) is None:
        args.classic_control_interval = 0
    args._classic_config_applied = True


# Live Genesis ratings use 0..30; weight uses a different scale. Handedness,
# energy, positions, velocities and animation feedback remain observable.
NEUTRAL_RATINGS = dict.fromkeys((
    'speed', 'agility', 'shot_power', 'shot_accuracy', 'passing', 'stick',
    'puck_control', 'checking', 'endurance', 'glove_right', 'stick_left', 'stick_right',
    'shot_bias'), 15) | {'weight': 64, 'glove_left': 7, 'aggression': 7,
                       'offensive_delay': 7, 'defensive_delay': 7}


def _neutral_player(player):
    result = copy(player)
    for name, value in NEUTRAL_RATINGS.items():
        if getattr(player, name, None) is not None:
            setattr(result, name, value)
    if getattr(player, 'input_state', None) is not None:
        result.input_state = {name: NEUTRAL_RATINGS.get(name, value)
                              for name, value in player.input_state.items()}
    return result


def decision_state(state, consider_ratings=True):
    """Neutralize both teams for every tactic without mutating emulator state."""
    if consider_ratings:
        return state
    result = copy(state)
    for name in ('team1', 'team2'):
        original = getattr(state, name)
        team = copy(original)
        team.players = [_neutral_player(player) for player in original.players]
        team.goalie = _neutral_player(original.goalie)
        team.nz_players = [_neutral_player(player) for player in original.nz_players]
        team.nz_goalie = _neutral_player(original.nz_goalie)
        team.goalie_stats = copy(original.goalie_stats)
        team.nz_goalie_stats = copy(original.nz_goalie_stats)
        for field in vars(team.goalie_stats):
            setattr(team.goalie_stats, field, NEUTRAL_RATINGS[field])
            maximum = 120 if field == 'weight' else 15 if field == 'glove_left' else 30
            setattr(team.nz_goalie_stats, field, NEUTRAL_RATINGS[field] / maximum)
        setattr(result, name, team)
    return result

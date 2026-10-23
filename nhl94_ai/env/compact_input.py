"""Controlled-player/defending-goalie observations in player-centered rink axes."""
import math

from nhl94_ai.game.constants import GameConsts
from nhl94_ai.game.ram import GOALIE_INPUT_ATTRIBUTES, SKATER_INPUT_ATTRIBUTES
from nhl94_ai.model_inputs import load_model_input


_PLAYER_FIELDS = {
    'is_controlled', 'vx', 'vy', 'facing_x', 'facing_y', 'has_puck', 'is_falling', 'anim', 'anim_frame',
    'c_pressed', 'c_frames_held',
    *SKATER_INPUT_ATTRIBUTES,
}
_GOALIE_FIELDS = {
    'is_present', 'x', 'y', 'vx', 'vy', 'facing_x', 'facing_y', 'has_puck', 'is_pad_stack',
    'is_dive', 'anim', 'anim_frame', *GOALIE_INPUT_ATTRIBUTES,
}
_NET_FIELDS = {'left_x', 'left_y', 'right_x', 'right_y', 'depth'}
_FIELDS = {'player': _PLAYER_FIELDS, 'goalie': _GOALIE_FIELDS, 'net': _NET_FIELDS}


def normalize_groups(config):
    if config.get('normalization') != 'player-centered-rink-v1':
        raise ValueError('player-goalie-v1 requires player-centered-rink-v1 normalization.')
    supplied = config.get('groups')
    if not isinstance(supplied, dict):
        raise TypeError('Compact model input requires a groups object.')
    unknown = set(supplied) - set(_FIELDS)
    if unknown:
        raise ValueError(f'Unsupported compact model input groups: {", ".join(sorted(unknown))}')
    defaults = load_model_input('pvg')['groups']
    groups = {}
    for name, allowed in _FIELDS.items():
        fields = supplied.get(name, defaults[name])
        if fields is None or fields is False:
            fields = []
        elif fields is True:
            fields = defaults[name]
        if not isinstance(fields, list) or any(not isinstance(field, str) for field in fields):
            raise TypeError(f'Compact model input group {name!r} must contain field names.')
        if set(fields) - allowed or len(fields) != len(set(fields)):
            raise ValueError(f'Unsupported or duplicate compact model input fields in {name!r}: {fields}')
        groups[name] = list(fields)
    return groups


def _controlled_player(team):
    if team.defense_control is None:
        player = team.get_controlled_player()
        return player if player is not team.goalie else None
    index = team.defense_control % 6
    return team.players[index] if team.defense_control >= 0 and index < team.num_players else None


def _live_value(entity, field):
    if not entity.input_state_known or entity.input_state is None or field not in entity.input_state:
        raise ValueError(f'Compact model input requires live RAM feedback for {field!r}.')
    return entity.input_state[field]


def _entity_value(entity, field, attributes):
    if entity is None:
        return 0.0
    if field in attributes:
        return _live_value(entity, field) / attributes[field][1]
    if field in ('vx', 'vy'):
        return _live_value(entity, field) / 32768
    if field in ('facing_x', 'facing_y'):
        angle = _live_value(entity, 'facing') * math.pi / 4
        return math.sin(angle) if field == 'facing_x' else math.cos(angle)
    if field == 'has_puck':
        return float(_live_value(entity, field))
    if field == 'anim':
        return entity.anim / 65535
    if field == 'anim_frame':
        return entity.anim_frame / 32
    if field in ('is_controlled', 'is_present'):
        return 1.0
    return float(getattr(entity, field))


def encode(state, groups):
    player = _controlled_player(state.team1)
    if player is not None and player.input_state_known and player.input_state is None:
        player = None
    goalie = state.team2.goalie
    if not goalie.input_state_known:
        raise ValueError('Compact model input requires live goalie RAM feedback.')
    if goalie.input_state is None:
        goalie = None
    x_scale, y_scale = 2 * GameConsts.MAX_PLAYER_X, 2 * GameConsts.MAX_PLAYER_Y
    values = []
    for field in groups['player']:
        if field == 'c_pressed':
            values.append(float(state.c_pressed))
        elif field == 'c_frames_held':
            values.append(min(state.c_frames_held / state.SLAPSHOT_HOLD_FRAMES, 1.0))
        else:
            values.append(_entity_value(player, field, SKATER_INPUT_ATTRIBUTES))
    for field in groups['goalie']:
        if field in ('x', 'y'):
            values.append(0.0 if player is None or goalie is None else (
                (goalie.x - player.x) / x_scale if field == 'x' else (goalie.y - player.y) / y_scale
            ))
        elif field in ('vx', 'vy'):
            values.append(0.0 if player is None or goalie is None else (
                (_live_value(goalie, field) - _live_value(player, field)) / 65536
            ))
        else:
            values.append(_entity_value(goalie, field, GOALIE_INPUT_ATTRIBUTES))
    net = state.team2.net
    net_values = {} if player is None else {
        'left_x': (net.left - player.x) / x_scale, 'left_y': (net.y - player.y) / y_scale,
        'right_x': (net.right - player.x) / x_scale, 'right_y': (net.y - player.y) / y_scale,
        'depth': net.depth / y_scale,
    }
    values.extend(net_values.get(field, 0.0) for field in groups['net'])
    if any(not math.isfinite(value) for value in values):
        raise ValueError('Compact model input contains nonfinite state values.')
    return tuple(max(-1.0, min(1.0, float(value))) for value in values)

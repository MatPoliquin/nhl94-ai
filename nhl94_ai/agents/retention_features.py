"""Shared current-state/action features and admission rules for retention probes."""
import numpy as np

# Legacy artifact semantics intentionally exclude goalies from this feature.
from nhl94_ai.game.actors import controlled_slot, controllable_skater
from nhl94_ai.game.motion import velocity
from nhl94_ai.game.skating import grounded_step
from nhl94_ai.game.constants import GameConsts as Buttons
from nhl94_ai.game.geometry import apply_pad


INTERVAL = 4
PADS = ((0, 0), (0, 1), (1, 1), (1, 0), (1, -1),
        (0, -1), (-1, -1), (-1, 0), (-1, 1))
ACTION_NAMES = ('coast', 'up', 'up-right', 'right', 'down-right', 'down',
                'down-left', 'left', 'up-left')
DIRECTIONS = (Buttons.INPUT_LEFT, Buttons.INPUT_RIGHT, Buttons.INPUT_UP, Buttons.INPUT_DOWN)


def attack_sign(state):
    return 1 if state.team2.net.y > state.team1.net.y else -1


def movement_action(command, sign):
    if not 0 <= command < len(PADS) or sign not in (-1, 1):
        raise ValueError('Expected one of nine movement commands and an attack sign')
    action = np.zeros(12, dtype=np.int8)
    dx, dy = PADS[command]
    return apply_pad(action, (dx * sign, dy * sign))


def movement_id(action, sign):
    pad = (sign * int(action[Buttons.INPUT_RIGHT] - action[Buttons.INPUT_LEFT]),
           sign * int(action[Buttons.INPUT_UP] - action[Buttons.INPUT_DOWN]))
    return PADS.index(pad)


def available(model, state):
    owner = state.engine.puck_owner
    player = state.team1.get_player_by_scnum(owner)
    return bool(player is not None and owner == controlled_slot(state.team1)
                and owner != state.team1.goalie_scnum() and state.team1.owns_scnum(owner)
                and controllable_skater(player) and player.y * attack_sign(state) >= 88
                and not state.engine.clock_stopped and model.scheduler.remaining == 0
                and not model.scheduler.was_defending and model.one_timer.pending is None
                and model.offense.pass_action.pending is None
                and not model.shot.active(model.scheduler.decisions)
                and (model.goalie is None or model.goalie.phase == 'skater'))


def ordinary_movement(model, action):
    return bool(model.input_owner == 'offense' and model.one_timer.pending is None
                and model.offense.pass_action.pending is None
                and not model.shot.active(model.scheduler.decisions)
                and not any(value for i, value in enumerate(action) if i not in DIRECTIONS))


def features(state, command):
    """Current physical measurements only; no replay outcome or future state."""
    sign, puck = attack_sign(state), state.puck
    player = state.team1.get_player_by_scnum(state.engine.puck_owner)
    vx, vy = velocity(player)
    px, py = velocity(puck)
    phase = player.facing_phase if player.facing_phase is not None else player.orientation
    angle = phase * np.pi / 4
    dx, dy = PADS[command]
    row = dict(x=sign * player.x, y=sign * player.y, vx=sign * vx, vy=sign * vy,
               facing_x=float(sign * np.sin(angle)), facing_y=float(sign * np.cos(angle)),
               puck_dx=sign * (puck.x - player.x), puck_dy=sign * (puck.y - player.y),
               puck_dvx=sign * (px - vx), puck_dvy=sign * (py - vy), puck_height=puck.height,
               pad_x=dx, pad_y=dy)
    for name in ('speed', 'agility', 'weight', 'stick', 'handedness', 'energy', 'movement_bonus'):
        value = getattr(player, name, None)
        if value is None:
            raise ValueError('Retention features require native carrier attributes: ' + name)
        row[name] = float(value)
    future = player
    for _ in range(INTERVAL):
        future = grounded_step(future, (dx * sign, dy * sign))
    row.update(projected_dx=sign * (future.x - player.x), projected_dy=sign * (future.y - player.y),
               projected_vx=sign * velocity(future)[0], projected_vy=sign * velocity(future)[1])
    for team_name, team, count in (('opponent', state.team2, 6), ('teammate', state.team1, 2)):
        others = sorted((p for p in (*team.players, team.goalie) if p is not player and controllable_skater(p)),
                        key=lambda p: (p.x - player.x)**2 + (p.y - player.y)**2)
        for index in range(count):
            other = others[index] if index < len(others) else None
            values = (0.0,) * 9
            if other is not None:
                ox, oy = velocity(other)
                values = (1.0, sign * (other.x - player.x), sign * (other.y - player.y),
                          sign * (ox - vx), sign * (oy - vy), float(other.weight or 0),
                          float(other.stick or 0), float(other.is_falling), float(other.role == 0))
            for field, value in zip(('present', 'dx', 'dy', 'dvx', 'dvy', 'weight', 'stick', 'falling', 'goalie'), values):
                row[f'{team_name}_{index}_{field}'] = float(value)
    if not all(np.isfinite(value) for value in row.values()):
        raise ValueError('Non-finite retention feature')
    return row



def admissible(rows, baseline):
    """Available-at-decision progress guard, independent of observed labels."""
    base = rows[baseline]['features']
    return [i for i, row in enumerate(rows)
            if i == baseline or (row['features']['projected_dy'] >= base['projected_dy'] - 0.25
                                 and row['features']['pad_y'] >= min(0, base['pad_y']))]

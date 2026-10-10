"""Evaluation-only nine-way normal-shot aim, with Classic's trigger and hold.

Pad y is a screen input (positive = UP), not skating toward the attacking net.
The native shot table assigns UP/high, neutral/middle, DOWN/low at either end.
Horizontal follow-through keeps Classic's original side; only aiming changes.
"""
from collections import Counter
from copy import deepcopy
import math

import numpy as np

from nhl94_ai.agents.classic_v1 import ClassicAIV1Model
from nhl94_ai.agents.motion import velocity
from nhl94_ai.agents.shot_placement import legacy_action, shot_features
from nhl94_ai.game.constants import GameConsts as Buttons
from nhl94_ai.training.shot_placement import estimate


PADS = tuple((x, y) for y in (0, 1, -1) for x in (-1, 0, 1))
MODEL_VERSION = 'native-shot-direction-v1'


def legacy_pad(state):
    return legacy_action(state, 4)[0], 0


def aim_buttons(action, pad):
    if tuple(pad) not in PADS:
        raise ValueError('Expected one of nine directional inputs')
    action[Buttons.INPUT_UP:Buttons.INPUT_C] = 0
    x, y = pad
    if x:
        action[Buttons.INPUT_LEFT if x < 0 else Buttons.INPUT_RIGHT] = 1
    if y:
        action[Buttons.INPUT_UP if y > 0 else Buttons.INPUT_DOWN] = 1


def direction_features(state, pad):
    x, y = pad
    values = shot_features(state, x, 4)
    if values is None:
        return None
    player = state.team1.get_player_by_scnum(state.engine.puck_owner)
    goalie = state.team2.goalie
    hand = float(player.handedness or 0)
    values.update(height=y, high=float(y == 1), low=float(y == -1),
                  centered=float(x == 0), hand=hand, aim_hand=x*hand,
                  animation_known=float(player.selection_flags is not None),
                  animation_locked=float(bool((player.selection_flags or 0) & 0x20)),
                  goalie_falling=float(bool(goalie.is_falling)),
                  goalie_animation_timer=float(goalie.animation_timer or 0)/30,
                  goalie_dive=float(bool(goalie.is_dive)),
                  puck_height=float(state.puck.height or 0)/12)
    for key in ('net_distance_y', 'power', 'carrier_vy', 'carrier_vx', 'goalie_depth',
                'goalie_vx', 'goalie_vy', 'goalie_down', 'goalie_dive', 'goalie_falling',
                'accuracy', 'facing_x', 'facing_y', 'lane_clearance'):
        values['height_' + key] = y * values[key]
    return values if all(math.isfinite(value) for value in values.values()) else None


class DirectionSelector:
    """Portable fixed controls or regularized logistic ranking over nine inputs."""
    def __init__(self, model):
        if model.get('version') != MODEL_VERSION:
            raise ValueError('Unsupported directional shot model')
        self.model = model
        self.metrics, self.diagnostics = Counter(), {}
        if model['kind'] == 'fixed-height':
            if (not isinstance(model['height'], int) or isinstance(model['height'], bool)
                    or model['height'] not in (-1, 0, 1)):
                raise ValueError('Fixed height must be -1, 0 or 1')
        elif model['kind'] == 'logistic':
            if model.get('scope', 'nine') not in ('nine', 'height-only'):
                raise ValueError('Unknown directional action scope')
            n = len(model['features'])
            if not n or n != len(set(model['features'])):
                raise ValueError('Directional model features must be nonempty and unique')
            if (not 0 <= model['min_gain'] <= 1 or not math.isfinite(model['bias'])
                    or any(len(model[key]) != n or not np.isfinite(model[key]).all()
                           for key in ('mean', 'scale', 'weights'))
                    or min(model['scale']) <= 0):
                raise ValueError('Invalid directional model parameters')
        else:
            raise ValueError('Unknown directional shot model kind')

    def choose(self, state):
        old = legacy_pad(state)
        if self.model['kind'] == 'fixed-height':
            chosen = old[0], self.model['height']
        else:
            pads = [old, *(pad for pad in PADS if pad != old
                          and (self.model.get('scope', 'nine') == 'nine' or pad[0] == old[0]))]
            features = [direction_features(state, pad) for pad in pads]
            if any(row is None for row in features):
                self.metrics['missing-feedback'] += 1
                return old
            scores = [estimate(self.model, row) for row in features]
            best = max(range(len(scores)), key=scores.__getitem__)
            chosen = pads[best] if scores[best] > scores[0] + self.model['min_gain'] else old
        self.metrics['shots'] += 1
        self.metrics['overrides'] += chosen != old
        self.metrics['height-overrides'] += chosen[1] != 0
        self.metrics['horizontal-overrides'] += chosen[0] != old[0]
        self.metrics[f'pad:{chosen[0]},{chosen[1]}'] += 1
        self.diagnostics = {'baseline': old, 'selected': chosen}
        return chosen


class DirectionalShotModel(ClassicAIV1Model):  # pylint: disable=abstract-method
    def __init__(self, args=None, env=None):
        super().__init__(args, env)
        self.direction_selector = None
        self.directional_aim = None

    def _observe_actions(self, state, elapsed, *, native=False):
        if (self.directional_aim is not None
                and (state.engine.puck_owner != self.shot.slot or state.engine.clock_stopped)):
            self.directional_aim = None
        return super()._observe_actions(state, elapsed, native=native)

    def _attack(self, state, action, player, actual, attack):
        result, intent = super()._attack(state, action, player, actual, attack)
        if (self._last_decision == 'shoot' and result[Buttons.INPUT_C]
                and self.shot.hold_until_frame == self.scheduler.frames + self.scheduler.interval):
            pad = self.direction_selector.choose(state) if self.direction_selector is not None else legacy_pad(state)
            self.directional_aim = (self.shot.slot, self.shot.hold_until_frame, pad)
            aim_buttons(result, pad)
        return result, intent

    def _continue_shot(self, state, action, player):
        result, intent = super()._continue_shot(state, action, player)
        aim = self.directional_aim
        if aim is not None and aim[:2] == (self.shot.slot, self.shot.hold_until_frame):
            aim_buttons(result, aim[2])
        return result, intent


def placed(history, pad):
    """Change only this committed shot; subsequent decisions use original Classic."""
    model = deepcopy(history)
    model.__class__ = DirectionalShotModel
    model.direction_selector = None
    model.directional_aim = (model.shot.slot, model.shot.hold_until_frame, tuple(pad))
    action = model.scheduler.action.copy()
    aim_buttons(action[0], pad)
    model.scheduler.action = action
    return model

"""Learned placement at an already-selected normal shot; no tactical arbitration.

Scores estimate primary-shot goals under the recorded native rollout protocol.
They are not calibrated game-wide xG. The model is a small regularized logistic
regression stored as checked, portable JSON rather than executable pickles.
"""
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from nhl94_ai.agents.carry import carry_path
from nhl94_ai.agents.defense import eligible, on_ice
from nhl94_ai.agents.finishing import _score, shot_speed
from nhl94_ai.agents.motion import facing, velocity
from nhl94_ai.agents.offense import normal_shot_release_frames, shot_release_in_front
from nhl94_ai.agents.passing import pressure_margin
from nhl94_ai.agents.possession_value import shot_quality


MODEL_VERSION = 'native-shot-logistic-v1'
CONTROL_VERSION = 'native-shot-fixed-hold-v1'
HOLDS = (1, 4, 8, 12)
INCOMPATIBLE = ('reception_control', 'possession_value', 'possession_ablation', 'receiver_selection', 'offense_lookahead',
                'uncertain_carry', 'chance_creation', 'cross_crease', 'deke', 'classic_refinements')


def legacy_action(state, interval):
    return (-1 if state.team2.goalie.x > 0 else 1), interval


def shot_actions(state, interval):
    player = state.team1.get_player_by_scnum(state.engine.puck_owner)
    baseline = legacy_action(state, interval)
    # Always retain the admitted legacy shot, even when a forecast is incomplete.
    actions = [baseline]
    for hold in sorted({*HOLDS, interval}):
        if shot_release_in_front(state, player, hold):
            actions.extend((side, hold) for side in (-1, 0, 1) if (side, hold) != baseline)
    return actions


def shot_features(state, side, hold):
    player = state.team1.get_player_by_scnum(state.engine.puck_owner)
    if player is None or not eligible(player) or any(getattr(player, name) is None for name in (
            'motion_x', 'motion_y', 'speed', 'agility', 'weight', 'energy', 'shot_power', 'shot_accuracy')):
        return None
    sign = 1 if state.team2.net.y > state.team1.net.y else -1
    goalie = state.team2.goalie
    if goalie.motion_x is None or goalie.motion_y is None:
        return None
    release = normal_shot_release_frames(player, hold)
    path = carry_path(player, None, release)
    if path is None:
        return None
    point = state.puck.x + path[-1].x - player.x, state.puck.y + path[-1].y - player.y
    speed = shot_speed(player, hold)
    clearance, _ = _score(state, player, point, side, release, speed)
    vx, vy = velocity(player)
    gx, gy = velocity(goalie)
    fx, fy = facing(player)
    distance = (state.team2.net.y - state.puck.y) * sign / 100
    x, goal_x = state.puck.x / 80, goalie.x / 50
    h = hold / 12
    values = {
        'puck_x': x, 'net_distance_y': distance, 'puck_x_squared': x*x,
        'distance_squared': distance*distance, 'carrier_vx': vx/4, 'carrier_vy': vy*sign/4,
        'goalie_x': goal_x, 'goalie_depth': (state.team2.net.y-goalie.y)*sign/40,
        'goalie_vx': gx/4, 'goalie_vy': gy*sign/4,
        'facing_x': fx, 'facing_y': fy*sign,
        'accuracy': player.shot_accuracy/30, 'power': player.shot_power/30, 'energy': player.energy/4096,
        'pressure': max(-30, min(30, pressure_margin(state.team2, point, release)))/30,
        'goalie_down': float(goalie.live_anim in (0x250, 0x2A2)),
        'nearest_defender': min((math.dist((player.x, player.y), (p.x, p.y))
                                 for p in state.team2.players if on_ice(p)), default=150)/150,
        'aim': side, 'neutral': float(side == 0), 'hold': h, 'hold_squared': h*h,
        'aim_carrier_x': side*x, 'aim_goalie_x': side*goal_x, 'aim_goalie_vx': side*gx/4,
        'aim_carrier_vx': side*vx/4, 'aim_facing_x': side*fx,
        'hold_carrier_vy': h*vy*sign/4, 'hold_goalie_vx': h*gx/4, 'hold_distance': h*distance,
        'aim_hold': side*h, 'aim_hold_goalie_x': side*h*goal_x,
        'release_x': point[0]/80, 'release_distance': (state.team2.net.y-point[1])*sign/100,
        'release_frames': release/20, 'shot_speed': speed/10,
        'lane_clearance': max(-40, min(40, clearance))/40,
        'geometry_quality': shot_quality(state, player, point, side, release, speed)/100,
    }
    return values if all(math.isfinite(value) for value in values.values()) else None


def extend_hold(model, hold):
    """The default seven-decision follow-through must cover a longer C hold."""
    model.shot.extended_hold = hold > model.scheduler.interval
    if model.shot.extended_hold:
        model.shot.until_decision = model.scheduler.decisions + max(7, math.ceil(hold/model.scheduler.interval) + 3)


class ShotPlacement:
    def __init__(self, path):
        data = Path(path).read_bytes()
        self.sha256 = hashlib.sha256(data).hexdigest()
        self.model = json.loads(data)
        self.metrics, self.diagnostics = Counter(), {}
        self.fixed_hold = None
        if self.model.get('version') == CONTROL_VERSION:
            self.fixed_hold = self.model['hold']
            if not isinstance(self.fixed_hold, int) or isinstance(self.fixed_hold, bool) or self.fixed_hold not in HOLDS:
                raise ValueError('Fixed shot control requires a supported integer hold')
            return
        if self.model.get('version') != MODEL_VERSION:
            raise ValueError('Unsupported shot-placement model version')
        self.names = self.model['features']
        if not self.names or len(set(self.names)) != len(self.names):
            raise ValueError('Shot-placement features must be unique')
        self.mean = np.asarray(self.model['mean'], dtype=float)
        self.scale = np.asarray(self.model['scale'], dtype=float)
        self.weights = np.asarray(self.model['weights'], dtype=float)
        self.bias = float(self.model['bias'])
        self.min_gain = float(self.model['min_gain'])
        if (any(array.shape != (len(self.names),) for array in (self.mean, self.scale, self.weights))
                or not all(np.isfinite(array).all() for array in (self.mean, self.scale, self.weights))
                or not (self.scale > 0).all() or not math.isfinite(self.bias)
                or not 0 <= self.min_gain <= 1):
            raise ValueError('Invalid shot-placement model parameters')

    def estimate(self, features):
        vector = np.asarray([features[name] for name in self.names])
        logit = float(((vector-self.mean)/self.scale) @ self.weights + self.bias)
        return 1 / (1 + math.exp(-max(-40, min(40, logit))))

    def choose(self, state, interval):
        baseline = legacy_action(state, interval)
        choices = shot_actions(state, interval)
        if self.fixed_hold is not None:
            candidate = (baseline[0], self.fixed_hold)
            selected = candidate if candidate in choices else baseline
            self.metrics['shots'] += 1
            self.metrics['overrides'] += selected != baseline
            self.metrics['hold-overrides'] += selected != baseline
            self.diagnostics = {'status': 'fixed-hold-control', 'model': CONTROL_VERSION,
                                'sha256': self.sha256, 'selected': selected, 'baseline': baseline}
            return selected
        rows = []
        for side, hold in choices:
            features = shot_features(state, side, hold)
            if features is None:
                self.metrics['missing-feedback'] += 1
                self.diagnostics = {'status': 'missing-feedback-use-legacy'}
                return baseline
            rows.append({'side': side, 'hold': hold, 'estimate': self.estimate(features)})
        old = rows[0]
        best = max(rows, key=lambda row: row['estimate'])
        selected = best if best['estimate'] > old['estimate'] + self.min_gain else old
        self.metrics['shots'] += 1
        self.metrics['overrides'] += selected != old
        self.metrics['aim-overrides'] += selected['side'] != old['side']
        self.metrics['hold-overrides'] += selected['hold'] != old['hold']
        self.diagnostics = {'status': 'selected', 'model': MODEL_VERSION, 'sha256': self.sha256,
                            'candidates': rows, 'selected': selected, 'baseline': old,
                            'scope': 'primary-shot goal estimate; not calibrated game-wide xG'}
        return selected['side'], selected['hold']

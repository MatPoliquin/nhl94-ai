"""Immediate possession policy; long-horizon carry certificates stay experimental."""
import math

from nhl94_ai.agents.classic_tuning import current as tuning

from nhl94_ai.agents.offense import (
    CARRY_SAMPLES, FEINT_FRAMES, OffenseController, _carry_motion, carry_clearance,
    goalie_avoidance,
)
from nhl94_ai.agents.passing import pressure_margin, shot_value
from nhl94_ai.env.target_control import project_target, route_waypoint


class PossessionOffenseController(OffenseController):
    """Rank immediate opportunities without requiring a worst-case carry certificate.

    Arrival estimates guide decisions, but are not proof against arbitrary future
    defender inputs. Report that distinction instead of calling the route safe.
    The experimental policies continue to use the native continuation model.
    """

    @property
    def uses_lookahead(self):
        return self.allow_uncertified or self.chance_creation

    def _estimated_carry(self, state, player, target):
        target = route_waypoint((player.x, player.y), project_target(target))
        escape = goalie_avoidance(state, player, target)
        details = {}
        if escape is not None:
            target, details = escape
            target = route_waypoint((player.x, player.y), target)
        clearance = carry_clearance(state, player, target, decision_interval=self.movement_interval)
        motions = [_carry_motion(player, target, frames, decision_interval=self.movement_interval)
                   for frames in CARRY_SAMPLES]
        bounded = all(motion is not None for motion in motions)
        margin = min(pressure_margin(state.team2, motion[0], frames)
                     for motion, frames in zip(motions, CARRY_SAMPLES)) if bounded else -math.inf
        clear = clearance > 0 and margin >= 3 and details.get('goalie_avoidance_safe', True)
        sign = 1 if state.team2.net.y > state.team1.net.y else -1
        return target, {
            **details, 'carry_safe': None, 'carry_estimated_clear': clear, 'carry_bounded': bounded,
            'carry_clearance': clearance, 'carry_pressure': margin,
            'carry_progress': (motions[-1][0][1] - player.y) * sign if bounded else -math.inf,
            'carry_position_value': shot_value(state, player, motions[-1][0], FEINT_FRAMES) if clear else 0,
            'carry_shot_value': 0.0, 'carry_pressure_model': 'arrival-estimate',
            'carry_status': 'estimated-clear' if clear else 'fallback',
            'carry_movement_model': 'native-held-input' if self.carry_motion else 'sampled-legacy',
        }

    def fallback_carry(self, state, player):
        if self.allow_uncertified or self.chance_creation:
            return super().fallback_carry(state, player)
        preferred = self.carry_target(state, player)
        if (player.motion_x is None or player.motion_y is None
                or not state.engine.puck_owner_known):
            return route_waypoint((player.x, player.y), preferred), {
                'mode': 'carry', 'carry_safe': None,
                'reason': 'missing feedback; retain legacy carry/setup',
            }
        first = self._estimated_carry(state, player, preferred)
        if first[1]['carry_estimated_clear']:
            return first[0], {
                **first[1], 'mode': 'goalie-avoid' if 'goalie_clearance' in first[1] else 'carry',
                'reason': 'model-based forward route; defender safety uncertified',
            }
        sign = 1 if state.team2.net.y > state.team1.net.y else -1
        targets = self._fallback_targets(player, sign)
        brake = targets[0]
        options = [first, *(self._estimated_carry(state, player, target) for target in targets)]
        bounded = [option for option in options if option[1]['carry_bounded']]
        if not bounded:
            return route_waypoint((player.x, player.y), project_target(brake)), {
                'mode': 'carry-escape', 'carry_safe': None, 'carry_bounded': False,
                'carry_estimated_clear': False, 'carry_pressure_model': 'arrival-estimate',
                'reason': 'no bounded carry; brake before unavoidable wall or net contact',
            }
        point, details = max(bounded, key=lambda option: (
            option[1]['carry_estimated_clear'], option[1]['carry_position_value'],
            option[1]['carry_progress'] if option[1]['carry_estimated_clear'] else min(option[1]['carry_pressure'], 12),
            min(option[1]['carry_pressure'], 12) if option[1]['carry_estimated_clear'] else min(option[1]['carry_clearance'], 16),
            min(option[1]['carry_clearance'], 16), -math.dist(option[0], preferred)))
        mode = ('goalie-avoid' if 'goalie_clearance' in first[1] else
                'carry' if point == first[0] else 'carry-escape')
        return point, {
            **details, 'mode': mode,
            'reason': 'model-based escape; defender safety uncertified' if details['carry_estimated_clear']
            else 'no estimated-clear carry; maximize contact and interception clearance',
        }

    @staticmethod
    def _immediate_gain(option, player, sign, purpose, current):
        if purpose == 'advance':
            return option.forward_gain >= tuning().advance_pass_min_progress and (
                option.bypassed > 0 or option.forward_gain >= 50
                or player.y * sign < 88 <= option.point[1] * sign)
        return option.shot_value > current + tuning().position_pass_min_improvement

    def choose(self, state, frame):
        if self.allow_uncertified or self.chance_creation:
            return super().choose(state, frame)
        player = self._begin_choice(state, frame)
        if player is None:
            return None
        sign = 1 if state.team2.net.y > state.team1.net.y else -1
        purpose = 'advance' if player.y * sign < 140 else 'position'
        choices, self.diagnostics['candidates'] = self.passes(state, purpose)
        if self._close_to_finish(state, player, sign):
            return None
        if self.breakaway(state, player):
            return self._breakaway_plan(state, player, 'no reachable skater interception in the arrival estimate')
        current = shot_value(state, player)
        choices = [option for option in choices if self._immediate_gain(option, player, sign, purpose, current)]
        self._mark_worthwhile(choices)
        opportunity = max(current, choices[0].shot_value + min(choices[0].margin, 12) if choices else 0)
        selected = self._pass_or_cut(state, player, frame, choices, purpose, opportunity)
        if selected is not None:
            return selected
        self.diagnostics['status'] = 'no-better-safe-option'
        return None

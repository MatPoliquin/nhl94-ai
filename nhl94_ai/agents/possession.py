"""Immediate possession policy; long-horizon carry certificates stay experimental."""
import math

from nhl94_ai.agents.defense import eligible
from nhl94_ai.agents.motion import facing
from nhl94_ai.agents.offense import (
    CARRY_SAMPLES, FEINT_FRAMES, OffenseController, _carry_motion, carry_clearance,
    goalie_avoidance, shot_release_in_front,
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

    @staticmethod
    def _estimated_carry(state, player, target):
        target = route_waypoint((player.x, player.y), project_target(target))
        escape = goalie_avoidance(state, player, target)
        details = {}
        if escape is not None:
            target, details = escape
            target = route_waypoint((player.x, player.y), target)
        clearance = carry_clearance(state, player, target)
        motions = [_carry_motion(player, target, frames) for frames in CARRY_SAMPLES]
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
        fx, fy = facing(player)
        brake = player.x - fx * 48, player.y - fy * 48
        targets = [brake, (player.x, player.y)]
        targets.extend((player.x + dx, player.y + sign * dy)
                       for dx, dy in ((-48, 0), (48, 0), (0, -48), (-48, -24), (48, -24),
                                      (-48, 24), (48, 24)))
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
            return option.forward_gain >= 20 and (
                option.bypassed > 0 or option.forward_gain >= 50
                or player.y * sign < 88 <= option.point[1] * sign)
        return option.shot_value > current + 12

    def choose(self, state, frame):
        if self.allow_uncertified or self.chance_creation:
            return super().choose(state, frame)
        player = state.team1.get_player_by_scnum(state.engine.puck_owner)
        self.diagnostics = {'candidates': [], 'scenario_samples': 8,
                            'evaluation_frame': frame, 'evaluation_carrier': state.engine.puck_owner}
        if (player is None or not eligible(player) or player.motion_x is None
                or player.motion_y is None or not state.engine.puck_owner_known):
            self.diagnostics['status'] = 'missing-feedback'
            return None
        sign = 1 if state.team2.net.y > state.team1.net.y else -1
        purpose = 'advance' if player.y * sign < 140 else 'position'
        choices, self.diagnostics['candidates'] = self.passes(state, purpose)
        finish_depth = 198 + max(0, 15 - (player.shot_accuracy if player.shot_accuracy is not None else 15)) * 0.4
        if (player.y * sign > finish_depth and abs(player.x) < 48
                and shot_release_in_front(state, player, self.decision_interval)
                and pressure_margin(state.team2, (player.x, player.y), 8) >= 3):
            self.feint_target = None
            self.diagnostics['status'] = 'close-to-finish'
            return None
        if self.breakaway(state, player):
            self.feint_target = None
            return self._plan(state, 'carry-breakaway', self.carry_target(state, player),
                              'no reachable skater interception in the arrival estimate')
        current = shot_value(state, player)
        choices = [option for option in choices if self._immediate_gain(option, player, sign, purpose, current)]
        worthwhile_slots = {option.slot for option in choices}
        for candidate in self.diagnostics['candidates']:
            if candidate['status'] == 'safe':
                candidate['worthwhile'] = candidate['slot'] in worthwhile_slots
        opportunity = max(current, choices[0].shot_value + min(choices[0].margin, 12) if choices else 0)
        if self.feint_mode == 'one-timer-setup':
            continuation = self._continue_cut(state, player, frame)
            if continuation:
                return continuation
        if choices and frame >= self.pass_at:
            best = choices[0]
            if purpose == 'position':
                setup = self._feint(state, player, frame, opportunity, one_timer_only=True)
                if setup:
                    return setup
            return self._plan(state, 'advance-pass' if purpose == 'advance' else 'position-pass',
                              best.point, 'advance beyond defenders with reception space', best)
        continuation = self._continue_cut(state, player, frame)
        if continuation:
            return continuation
        cut = self._feint(state, player, frame, opportunity)
        if cut:
            return cut
        self.diagnostics['status'] = 'no-better-safe-option'
        return None

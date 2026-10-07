"""Read-only carrier-defense telemetry, not causal tackle attribution."""
from collections import Counter
import math

from nhl94_ai.agents.defense import controlled_slot, eligible
from nhl94_ai.game.constants import GameConsts


class CarrierDefenseMetrics:
    def __init__(self):
        self.zones = {}
        self.previous = None
        self.recovery_context = None

    def observe(self, state, diagnostics):
        self.recovery_context = None
        owner, actual = state.engine.puck_owner, controlled_slot(state.team1)
        carrier = next((player for index, player in enumerate(state.team2.players)
                        if state.team2.skater_scnum_base() + index == owner and eligible(player)), None)
        player = next((player for index, player in enumerate(state.team1.players)
                       if state.team1.skater_scnum_base() + index == actual and eligible(player)), None)
        if not diagnostics or carrier is None or player is None:
            self.previous = None
            return
        sign = 1 if state.team1.net.y > 0 else -1
        depth = carrier.y * sign
        zone = ('neutral' if abs(depth) < GameConsts.SLOT_BAND_MIN_Y else
                'defensive' if depth >= GameConsts.SLOT_BAND_MIN_Y else 'attacking')
        counts = self.zones.setdefault(zone, Counter())
        goal_side = player.y * sign >= depth
        counts['frames'] += 1
        counts['goal_side_frames' if goal_side else 'behind_carrier_frames'] += 1
        counts['close_frames'] += math.dist((player.x, player.y), (carrier.x, carrier.y)) <= 30
        counts['cutoff_frames'] += diagnostics['decision'] == 'intercept-carrier'
        counts['retreat_frames'] += diagnostics['decision'] == 'retreat-lane'
        counts['poke_requests'] += diagnostics['mode'] == 'poke-request'
        counts['check_requests'] += diagnostics['mode'] == 'check-request'
        if self.previous == (owner, actual, False) and goal_side:
            counts['goal_side_regained'] += 1
        self.previous = owner, actual, goal_side
        self.recovery_context = zone, owner, actual

    def after_step(self, owner):
        if self.recovery_context is not None:
            zone, before, actual = self.recovery_context
            if before != owner == actual:
                self.zones[zone]['direct_controlled_recoveries'] += 1
        self.recovery_context = None

    def summary(self):
        return {zone: dict(counts) for zone, counts in self.zones.items()}


class GoalieDefenseMetrics:
    """Control context immediately before a goal, not causal save attribution."""

    def __init__(self, score, one_timer_goals):
        self.score, self.one_timer_goals = score, one_timer_goals
        self.phases = Counter()
        self.input_modes = Counter()
        self.context = None
        self.events = []

    def observe(self, state, controller, buttons, frame):
        goalie = state.team1.goalie
        selected = controlled_slot(state.team1) == state.team1.defense_goalie
        effective = (controller is not None and selected and bool(goalie.selection_flags & 8)
                     and not goalie.live_state_flags & 4)
        phase = controller.phase if controller is not None else 'cpu'
        self.phases[phase] += 1
        mode = ('moving' if any(buttons[4:8]) else
                'C-only' if buttons[GameConsts.INPUT_C] else
                'B-only' if buttons[GameConsts.INPUT_B] else 'neutral')
        self.input_modes[mode] += 1
        self.context = {
            'frame': frame, 'phase': phase, 'effective_manual': effective,
            'selected_slot': controlled_slot(state.team1),
            'cpu_fallback': bool(goalie.live_state_flags & 4),
            'goalie_position': [goalie.x, goalie.y],
            'goalie_velocity': [goalie.motion_x, goalie.motion_y],
            'input_mode': mode,
        }

    def after_step(self, score, one_timer_goals):
        for name, value, previous in (('score', score, self.score),
                                      ('one_timer_goals', one_timer_goals, self.one_timer_goals)):
            if value > previous:
                self.events.append({**self.context, 'counter': name, 'delta': value - previous})
        self.score, self.one_timer_goals = score, one_timer_goals

    def summary(self):
        return {'phase_frames': dict(self.phases), 'input_frames': dict(self.input_modes),
                'goal_events': self.events}

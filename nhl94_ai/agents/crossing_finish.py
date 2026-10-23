"""Use the isolated crossing shot when ordinary offense has already made room."""
from collections import Counter

from nhl94_ai.agents.cross_crease_roaming import RoamingCrossCrease
from nhl94_ai.game.actors import controlled_slot, controllable_skater
from nhl94_ai.game.motion import velocity


def validate_crossing_finish(args):
    enabled = getattr(args, 'opportunistic_cross_crease', False)
    unsupported = (
            getattr(args, 'action_type', 'FILTERED').upper() != 'FILTERED'
            or getattr(args, 'env', 'NHL94-Genesis-v0') != 'NHL94-Genesis-v0'
            or getattr(args, 'selfplay', False)
            or getattr(args, 'classic_control_interval', 0))
    incompatible = any(getattr(args, name, False) for name in (
                'cross_crease', 'possession_value', 'possession_ablation', 'shot_placement',
                'retention_model', 'one_timer_execution', 'rebound_recovery'))
    if enabled and (unsupported or incompatible):
        raise ValueError('Opportunistic crossing requires full-team Classic FILTERED native reactions; '
                         'disable cross-crease setup, alternative possession/shot models and lifecycle experiments')
    return enabled


class CrossingFinishSelector:
    """Select only an immediate finish; never own the approach or override a pass."""

    def __init__(self):
        self.probe = RoamingCrossCrease()
        self.metrics = Counter()
        self.events = []
        self.diagnostics = {}

    def choose(self, state, frame):
        self.metrics['evaluations'] += 1
        self.diagnostics = {}
        player = state.team1.get_player_by_scnum(state.engine.puck_owner)
        reason = None
        carrier_ready = (player is not None and player is not state.team1.goalie and controllable_skater(player)
                         and state.engine.puck_owner == controlled_slot(state.team1))
        if (state.numPlayers != 5 or state.engine.clock_stopped or state.is_shootout_active
                or not carrier_ready or state.is_shooting):
            reason = 'ineligible'
        elif any(getattr(player, name) is None for name in (
                'motion_x', 'motion_y', 'shot_power', 'speed', 'agility', 'weight', 'energy')):
            reason = 'missing-feedback'
        elif player.selection_flags is not None and player.selection_flags & 0x20:
            reason = 'animation-locked'
        else:
            self.probe.release_estimate = {}
            direction = self.probe._opening(state, player)
            self.diagnostics.update(self.probe.opportunity)
            if direction is None:
                reason = 'no-opening'
            elif not self.probe.release_ready(state, player, direction):
                reason = 'release-not-ready'
            self.diagnostics.update(self.probe.release_estimate)
        if reason is not None:
            self.diagnostics['status'] = reason
            self.metrics[reason] += 1
            return None
        self.diagnostics.update(status='ready', direction=direction)
        self.metrics['selected'] += 1
        self.events.append({
            'frame': frame, 'slot': state.engine.puck_owner, 'player': (player.x, player.y),
            'puck': (state.puck.x, state.puck.y), 'velocity': velocity(player),
            'goalie': (state.team2.goalie.x, state.team2.goalie.y), **self.diagnostics,
        })
        return direction, self.probe.settings.hold_frames

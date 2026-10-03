"""Observe controlled entries and turnovers without affecting gameplay."""
from nhl94_ai.agents.defense import controlled_slot
from nhl94_ai.agents.motion import arrival_time
from nhl94_ai.game.constants import GameConsts


class OffenseMetrics:
    def __init__(self):
        self.rush_start = None
        self.in_zone = False
        self.entries = []
        self.turnovers = 0
        self.zone_turnovers = 0
        self.possession_losses = self.zone_possession_losses = 0
        self.last_shots = None
        self.after_shot = False
        self.had_possession = False
        self.last_goalie_impact = None
        self.goalie_contact_impulses = 0

    def observe(self, frame, state):
        goalie = state.team2.goalie
        impact = goalie.contact_impact
        if impact is not None:
            if (self.last_goalie_impact is not None and impact > self.last_goalie_impact
                    and goalie.contact_player == controlled_slot(state.team1)):
                self.goalie_contact_impulses += 1
            self.last_goalie_impact = impact
        shots = state.team1.stats.shots
        if shots is not None and self.last_shots is not None and shots > self.last_shots:
            self.after_shot = True
        self.last_shots = shots
        owner = state.engine.puck_owner
        player = state.team1.get_player_by_scnum(owner)
        skater = player is not None and player is not state.team1.goalie
        if owner < 0:
            return
        if not skater:
            if self.had_possession and state.team2.owns_scnum(owner):
                self.possession_losses += 1
                self.zone_possession_losses += self.in_zone
                self.turnovers += not self.after_shot
                self.zone_turnovers += self.in_zone and not self.after_shot
            self.rush_start, self.in_zone, self.had_possession = None, False, False
            self.after_shot = False
            return
        self.after_shot = False
        sign = 1 if state.team2.net.y > state.team1.net.y else -1
        inside = player.y * sign >= GameConsts.ATACKZONE_POS_Y
        if not inside and self.rush_start is None:
            self.rush_start = frame
        if inside and not self.in_zone and self.rush_start is not None:
            point = player.x, player.y + sign * 16
            threats = sum(arrival_time(opponent, point, optimistic=True) <= 16
                          for opponent in state.team2.players
                          if opponent.role is None or opponent.role > 0)
            self.entries.append({'frame': frame, 'elapsed_frames': frame - self.rush_start,
                                 'owner': owner, 'controlled': owner == controlled_slot(state.team1),
                                 'threatening_defenders': threats})
            self.rush_start = None
        self.in_zone, self.had_possession = inside, True

    def summary(self):
        return {
            'entries': len(self.entries),
            'controlled_entries': sum(event['controlled'] for event in self.entries),
            'entry_frames_total': sum(event['elapsed_frames'] for event in self.entries),
            'entry_threats_total': sum(event['threatening_defenders'] for event in self.entries),
            'turnovers': self.turnovers, 'zone_turnovers': self.zone_turnovers,
            'possession_losses': self.possession_losses, 'zone_possession_losses': self.zone_possession_losses,
            'goalie_contact_impulses': self.goalie_contact_impulses,
        }

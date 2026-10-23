"""Explicit coordinator context; a team_vs_game CLI mode is not ownership."""
from dataclasses import dataclass


@dataclass(frozen=True)
class TeamControlContext:
    slots: tuple
    known: bool
    support_target: tuple
    cpu_support: bool = True
    goalie_ownership: bool = False

    def cpu_carrier(self, state):
        owner = state.engine.puck_owner
        return (self.cpu_support and self.known and state.engine.puck_owner_known
                and state.team1.owns_scnum(owner) and owner != state.team1.goalie_scnum()
                and owner not in self.slots)

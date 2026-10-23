"""Actor identity and availability; observability does not imply skater control."""


def controlled_slot(team):
    return team.controlled_scnum() if team.defense_control is None else team.defense_control

def owns_puck(team, owner):
    goalie = team.goalie_scnum() if team.defense_goalie is None else team.defense_goalie
    return team.skater_scnum_base() <= owner < team.skater_scnum_base() + len(team.players) or owner == goalie

def controllable_skater(player):
    locked = player.is_falling if player.selection_flags is None else player.selection_flags & 0x20
    return not locked and (player.role is None or player.role > 0) and not player.unavailable & 4

def observable_body(player):
    return player.role is None or player.role >= 0

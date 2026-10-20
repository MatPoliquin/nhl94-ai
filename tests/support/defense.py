"""Fresh defensive, goalie, and teammate-pressure scenarios."""
from nhl94_ai.game.state import NHL94GameState


def defense_state():
    state = NHL94GameState(5)
    state.team1.net.y, state.team2.net.y = -264, 264
    state.team1.control = state.team2.control = 1
    state.team1.defense_control, state.team2.defense_control = 0, 6
    state.engine.puck_owner = 6
    for team, y in ((state.team1, -215), (state.team2, 70)):
        for index, player in enumerate(team.players):
            player.x, player.y = -90 + index * 40, y
            player.motion_x = player.motion_y = 0
            player.speed = player.agility = player.stick = 20
            player.weight, player.energy, player.checking = 64, 4096, 20
            player.role, player.facing = 4, 0
    state.team1.players[0].x = 0
    state.team2.players[0].x, state.team2.players[0].y = 0, -150
    state.puck.x, state.puck.y = 0, -150
    state.puck.motion_x = state.puck.motion_y = 0
    state.puck.height, state.puck.motion_z = 0, 0
    return state


def goalie_state(selected=False):
    state = defense_state()
    state.team1.defense_goalie, state.team2.defense_goalie = 5, 11
    state.engine.goalie_modes, state.engine.goalie_hold_counts = (0, 0), (17, 17)
    state.engine.controller_teams = (1, 0)
    state.engine.input_block, state.engine.clock_stopped = 0, False
    state.engine.camera = (0, -200)
    goalie = state.team1.goalie
    goalie.x, goalie.y = 0, -246
    goalie.motion_x = goalie.motion_y = 0
    goalie.role, goalie.facing, goalie.assignment = 0, 0, 14
    goalie.speed, goalie.agility, goalie.weight, goalie.energy = 20, 20, 64, 4096
    goalie.stick, goalie.passing = 8, 6
    goalie.live_anim, goalie.live_state_flags, goalie.selection_flags = 2, 0, 0x80
    state.puck.x, state.puck.y = 10, -120
    state.puck.motion_x, state.puck.motion_y = 0, -1.3
    state.team1.players[0].x, state.team1.players[0].y = -110, 40
    if selected:
        state.team1.defense_control = 5
        goalie.selection_flags |= 8
    return state


def pressure_state():
    state = defense_state()
    state.time = 300
    state.engine.clock_stopped = False
    state.engine.puck_owner_known = True
    state.team1.goalie.x, state.team1.goalie.y = 0, -264
    state.team2.goalie.x, state.team2.goalie.y = 0, 264
    for team in (state.team1, state.team2):
        for player in team.players:
            player.role = -1
            player.selection_flags = 8
    for i, (x, y, facing) in enumerate(((-24, -24, 1), (24, 24, 5), (0, -72, 0))):
        player = state.team1.players[i]
        player.x, player.y, player.facing, player.role = x, y, facing, 1
    carrier = state.team2.players[0]
    carrier.x = carrier.y = state.puck.x = state.puck.y = 0
    carrier.role, carrier.facing, carrier.selection_flags = 1, 4, 0x40
    return state

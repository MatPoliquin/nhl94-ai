"""Fresh offensive scenarios and controller histories shared by unit tests."""
import json
from pathlib import Path
from types import SimpleNamespace

from nhl94_ai.agents.classic_v1 import ClassicAIV1Model
from nhl94_ai.agents.lifecycle import PassState
from nhl94_ai.game.state import NHL94GameState


def offense_state():
    state = NHL94GameState(5)
    state.team1.net.y, state.team2.net.y = -264, 264
    state.team1.control, state.team1.defense_control = 1, 0
    state.engine.puck_owner, state.engine.puck_owner_known = 0, True
    state.engine.offsides_enabled = False
    state.team1.pass_attempts = 0
    for team in (state.team1, state.team2):
        for player in (*team.players, team.goalie):
            player.motion_x = player.motion_y = 0
            player.speed, player.agility, player.weight = 20, 20, 64
            player.passing, player.energy, player.stick = 20, 4096, 20
            player.burst_full_energy = False
            player.role, player.facing, player.selection_flags = 4, 0, 0
            player.shot_accuracy = 20
        team.goalie.role = 0
    for player, point in zip(state.team1.players, ((0, -130), (-80, -30), (100, -210), (110, -240), (70, -220))):
        player.x, player.y = point
    for player, point in zip(state.team2.players, ((40, -75), (110, 80), (110, 140), (90, 180), (100, 220))):
        player.x, player.y = point
    state.team1.players[0].selection_flags = 8
    state.team1.goalie.y, state.team2.goalie.y = -250, 250
    state.puck.x, state.puck.y = 0, -120
    state.puck.motion_x = state.puck.motion_y = 0
    state.puck.height = 0
    return state


def live_one_timer_state(*, away=False):
    state = offense_state()
    passer, receiver = state.team1.players[:2]
    passer.x, passer.y, passer.facing = -50, 180, 1
    receiver.x, receiver.y, receiver.motion_y = 35, 160, 0.7
    state.puck.x, state.puck.y = -50, 190
    for other in (*state.team1.players[2:], *state.team2.players):
        other.x, other.y = 100, -180
    if away:
        for team in (state.team1, state.team2):
            team.controller = 3 - team.controller
            team.defense_control = team.skater_scnum_base()
            team.net.y = -team.net.y
            for player in (*team.players, team.goalie):
                player.x, player.y = -player.x, -player.y
                player.motion_x, player.motion_y = -player.motion_x, -player.motion_y
                player.facing = (player.facing + 4) % 8
        state.engine.puck_owner = 6
        state.puck.x, state.puck.y = -state.puck.x, -state.puck.y
    return state


def crossing_state(*, direction=1, attack=1):
    state = offense_state()
    player = state.team1.players[0]
    player.x, player.y = -direction * 35, attack * 212
    player.motion_x, player.motion_y = direction * 1.65, 0
    player.shot_power, player.handedness = 30, 0
    player.live_anim, player.live_anim_frame, player.animation_timer = 0, 0, 0
    player.facing = 2 if direction > 0 else 6
    for other in (*state.team1.players[1:], *state.team2.players):
        other.x, other.y = 110, -attack * 180
    state.team1.net.y, state.team2.net.y = -attack * 264, attack * 264
    state.team1.goalie.y = -attack * 250
    state.team2.goalie.x, state.team2.goalie.y = -direction * 12, attack * 250
    state.team2.goalie.live_anim = 2
    state.puck.x, state.puck.y = player.x + direction * 10, player.y
    state.engine.shot_player = -1
    return state


def deke_state():
    state = crossing_state()
    player = state.team1.players[0]
    player.x, player.y, player.motion_x, player.motion_y, player.facing = 0, 180, 0, 0.2, 0
    state.puck.x, state.puck.y = 0, 188
    state.team2.goalie.x = 0
    for team in (state.team1, state.team2):
        for other in (*team.players, team.goalie):
            other.precise_x, other.precise_y = float(other.x), float(other.y)
            other.facing_phase, other.sprite_flipped_x = float(other.facing), False
    player.shot_projection_offsets_x, player.shot_projection_offsets_y = (-24, 24), (-24, 24)
    player.shot_offsets_x, player.shot_offsets_y = (-6, 6), (12, 20)
    player.shot_durations = player.shot_projection_durations = (4,) * 8
    goalie = state.team2.goalie
    goalie.decision_timer, goalie.decision_interval, goalie.steering = 0, 2, 8
    goalie.cpu_tracking_active = True
    return state


def chain_state():
    state = offense_state()
    for player in (*state.team1.players[3:], *state.team2.players):
        player.x, player.y = 100, -180
    for player, point in zip(state.team1.players, ((-80, 180), (70, 150), (-30, 220))):
        player.x, player.y = point
    state.puck.x, state.puck.y = -80, 190
    state.team2.goalie.x = 30
    return state


def finish_state():
    state = offense_state()
    for player in (*state.team1.players[1:], *state.team2.players):
        player.x, player.y = 100, -150
    state.team1.players[0].x, state.team1.players[0].y = 0, 225
    state.team1.players[0].motion_y = 1
    state.puck.x, state.puck.y = 0, 230
    state.team2.goalie.x = 40
    return state


def response_state():
    state = offense_state()
    carrier, receiver = state.team1.players[:2]
    carrier.x, carrier.y, carrier.facing = -50, 160, 0
    receiver.x, receiver.y, receiver.facing = 0, 190, 0
    state.puck.x, state.puck.y = -50, 170
    receiver.assignment, receiver.cpu_target = 6, (0, 170)
    receiver.steering, receiver.steering_timer = 8, 0
    receiver.cpu_response_available, receiver.support_zone = True, 16
    for team in (state.team1, state.team2):
        for player in team.players:
            if player not in (carrier, receiver):
                player.role = -1
    return state


def wing_reception_state(*, carrier_x=80, receiver_x=105):
    state = offense_state()
    passer, receiver = state.team1.players[:2]
    passer.x, passer.y, passer.facing = carrier_x, 145, 0
    state.puck.x, state.puck.y = carrier_x, 155
    receiver.x, receiver.y, receiver.facing = receiver_x, 180, 6
    receiver.motion_x, receiver.motion_y = -1.5, 0
    for team in (state.team1, state.team2):
        for player in team.players:
            if player is not passer and player is not receiver:
                player.role = -1
    defender = state.team2.players[0]
    defender.role, defender.x, defender.y, defender.motion_x = 4, 0, 230, -2
    state.team2.goalie.x = -15
    return state


def shooting_state():
    state = NHL94GameState(5)
    state.team1.control = state.team2.control = 1
    state.team1.net.y, state.team2.net.y = -264, 264
    state.team1.players[0].x, state.team1.players[0].y = 30, 225
    state.team2.players[0].x, state.team2.players[0].y = -30, -225
    state.team2.goalie.x = 10
    state.team1.goalie.x = -10
    state.engine.puck_owner = 0
    return state


def approach_state(*, away=False):
    state = offense_state()
    player = state.team1.players[0]
    player.x, player.y, player.motion_y = 0, 180, 1.6
    state.team2.goalie.x, state.team2.goalie.y = 0, 228
    state.puck.x, state.puck.y = 0, 190
    for other in (*state.team1.players[1:], *state.team2.players):
        other.x, other.y = 110, -180
    if away:
        state.team1.controller, state.team2.controller = 2, 1
        state.team1.defense_control = state.engine.puck_owner = 6
        for team in (state.team1, state.team2):
            team.net.y *= -1
            for other in (*team.players, team.goalie):
                other.y *= -1
                other.motion_y *= -1
                other.facing = (other.facing + 4) % 8
        state.puck.y *= -1
    return state


def cross_slot_state():
    state = NHL94GameState(5)
    state.team1.net.y, state.team2.net.y = -264, 264
    state.team1.control = 1
    state.engine.puck_owner = 0
    state.team1.player_haspuck = True
    for player in state.team1.players:
        player.x, player.y = 80, -180
    for player in state.team2.players:
        player.x, player.y = 90, 0
    state.team1.players[0].x, state.team1.players[0].y = -50, 190
    state.team1.players[0].orientation = 1  # UP+RIGHT, toward this receiver.
    state.team1.players[1].x, state.team1.players[1].y = 35, 210
    state.puck.x, state.puck.y = -48, 190
    return state


def reception_state():
    fixture = json.loads((Path(__file__).parents[1] / 'fixtures/classic-missed-reception.json').read_text())
    state = offense_state()
    for name, value in fixture['receiver_feedback'].items():
        setattr(state.team1.players[0], name, value)
    for name, value in fixture['puck_feedback'].items():
        setattr(state.puck, name, value)
    state.team1.defense_control = 3
    state.engine.puck_owner = -247
    return state


def waiting_model(schema='FILTERED'):
    model = ClassicAIV1Model(SimpleNamespace(action_type=schema))
    model.scheduler.frames = 5
    model.offense.pass_action.pending = PassState(**{'passer': 3, 'receiver': 0, 'actual_receiver': 0,
                             'launched': True, 'flight_observed': True, 'frame': 0,
                             'deadline': 50, 'purpose': 'advance-pass', 'point': (-94, -136),
                             'passes_before': 0})
    return model

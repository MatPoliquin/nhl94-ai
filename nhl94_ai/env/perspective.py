import copy
import numpy as np
import gymnasium as gym
from nhl94_ai.game.constants import GameConsts

HOCKEY_PASS_RELEASE_FRAMES = 5
HOCKEY_ONE_TIMER_SHOT_DELAY_FRAMES = 1
HOCKEY_ONE_TIMER_SHOT_FRAMES = 12
HOCKEY_SLAPSHOT_HOLD_FRAMES = 18
HOCKEY_POKE_DISTANCE = 18.0
HOCKEY_CATCH_MAX_LOOKAHEAD_FRAMES = 36
HOCKEY_CATCH_PLAYER_SPEED_ESTIMATE = 7.2
HOCKEY_CATCH_BURST_DISTANCE = 22.0


class OpponentPerspective:
    """Extracted behavior with an explicit environment-state context."""

    def __init__(self, context):
        self.context = context

    @property
    def game_state(self):
        return self.context.game_state

    @property
    def opponent_action_state(self):
        return self.context.opponent_action_state

    def _mirror_env_action(self, ac):
        mirrored = np.asarray(ac, dtype=np.int8).copy()
        if len(mirrored) == GameConsts.INPUT_MAX:
            up = mirrored[GameConsts.INPUT_UP]
            down = mirrored[GameConsts.INPUT_DOWN]
            left = mirrored[GameConsts.INPUT_LEFT]
            right = mirrored[GameConsts.INPUT_RIGHT]
            mirrored[GameConsts.INPUT_UP] = down
            mirrored[GameConsts.INPUT_DOWN] = up
            mirrored[GameConsts.INPUT_LEFT] = right
            mirrored[GameConsts.INPUT_RIGHT] = left
            return mirrored

        if len(mirrored) == 3:
            if mirrored[0] == 1:
                mirrored[0] = 2
            elif mirrored[0] == 2:
                mirrored[0] = 1

            if mirrored[1] == 1:
                mirrored[1] = 2
            elif mirrored[1] == 2:
                mirrored[1] = 1
            return mirrored

        raise ValueError(f"Unsupported action format for mirroring: {ac}")

    def _mirror_gamestate_action(self, gamestate_action):
        mirrored = list(gamestate_action)
        mirrored[0], mirrored[1] = mirrored[1], mirrored[0]
        mirrored[2], mirrored[3] = mirrored[3], mirrored[2]
        return mirrored

    def _should_flip_zones_for_opponent(self):
        period = int(getattr(self.game_state, 'period', 1) or 1)
        return (period % 2) == 0

    def _rotate_player_180(self, player):
        for attr in (
            'x', 'y', 'vx', 'vy',
            'rel_puck_x', 'rel_puck_y', 'rel_puck_vx', 'rel_puck_vy',
            'rel_controlled_x', 'rel_controlled_y', 'rel_controlled_vx', 'rel_controlled_vy',
            'ori_x', 'ori_y',
        ):
            if hasattr(player, attr):
                setattr(player, attr, -getattr(player, attr))

        if hasattr(player, 'orientation'):
            player.orientation = (player.orientation + 4) % 8
        if player.input_state is not None:
            player.input_state['vx'] = -player.input_state['vx']
            player.input_state['vy'] = -player.input_state['vy']
            player.input_state['facing'] = (player.input_state['facing'] + 4) % 8

    def _rotate_team_180(self, team):
        for player in list(team.players) + [team.goalie]:
            self._rotate_player_180(player)

        team.stats.fullstar_x = -team.stats.fullstar_x
        team.stats.fullstar_y = -team.stats.fullstar_y
        team.stats.emptystar_x = -team.stats.emptystar_x
        team.stats.emptystar_y = -team.stats.emptystar_y

    def _recompute_team_relationships(self, team, puck):
        team.update_controlled_relationships()
        for player in [*team.players, team.goalie]:
            player.rel_puck_x = puck.x - player.x
            player.rel_puck_y = puck.y - player.y
            player.rel_puck_vx = puck.vx - player.vx
            player.rel_puck_vy = puck.vy - player.vy
            player.dist_to_puck = GameConsts.Distance(
                (player.x, player.y),
                (puck.x, puck.y),
            )

    def _refresh_derived_state(self, mirrored_state):
        self._recompute_team_relationships(mirrored_state.team1, mirrored_state.puck)
        self._recompute_team_relationships(mirrored_state.team2, mirrored_state.puck)
        mirrored_state.update_nets()
        mirrored_state._update_passing_lanes()
        mirrored_state._update_opponent_controlled_distances()
        mirrored_state.team1.Normalize()
        mirrored_state.team2.Normalize()
        mirrored_state.nz_puck.x = mirrored_state.puck.x / GameConsts.MAX_PUCK_X
        mirrored_state.nz_puck.y = mirrored_state.puck.y / GameConsts.MAX_PUCK_Y
        mirrored_state.nz_puck.vx = mirrored_state.puck.vx / GameConsts.MAX_VEL_XY
        mirrored_state.nz_puck.vy = mirrored_state.puck.vy / GameConsts.MAX_VEL_XY

    def _build_opponent_view_state(self):
        mirrored_state = copy.deepcopy(self.game_state)
        mirrored_state.Flip()

        mirrored_state.team1.controller = 1
        mirrored_state.team1.ram_var_prefix = 'p1_'
        mirrored_state.team1.ram_var_goalie_prefix = 'g1_'
        mirrored_state.team2.controller = 2
        mirrored_state.team2.ram_var_prefix = 'p2_'
        mirrored_state.team2.ram_var_goalie_prefix = 'g2_'

        if self._should_flip_zones_for_opponent():
            self._rotate_team_180(mirrored_state.team1)
            self._rotate_team_180(mirrored_state.team2)
            self._rotate_player_180(mirrored_state.puck)
            mirrored_state.action = self._mirror_gamestate_action(self.opponent_action_state['last_gamestate_action'])
        else:
            mirrored_state.action = list(self.opponent_action_state['last_gamestate_action'])
        mirrored_state.slapshot_frames_held = self.opponent_action_state['slapshot_frames']
        mirrored_state.c_pressed = bool(mirrored_state.action[5])
        mirrored_state.c_frames_held = self.opponent_action_state['slapshot_frames']

        self._refresh_derived_state(mirrored_state)
        return mirrored_state

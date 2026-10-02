"""Possession, shot and goal rewards, and terminal contracts without an emulator."""
import json
from pathlib import Path
import unittest

from nhl94_ai.game.state import NHL94GameState
from nhl94_ai.tasks.defensezone import isdone_defensezone, rf_defensezone


def defensive_state(skaters=5):
    state = NHL94GameState(skaters)
    state.time = state.last_time = 300
    state.update_nets()
    for team in (state.team1, state.team2):
        for player in team.players:
            player.x, player.y = 100, 0
        team.goalie.x, team.goalie.y = 100, team.net.y
    state.team2.players[0].x, state.team2.players[0].y = 0, -180
    state.puck.x, state.puck.y = 0, -180
    state.engine.puck_owner = 6
    return state


class DefenseZoneContracts(unittest.TestCase):
    def test_recovery_ends_immediately_for_any_skater(self):
        for skaters in (1, 2, 5):
            for owner in range(skaters):
                for puck_y in (-230, 0, 150):
                    with self.subTest(skaters=skaters, owner=owner, puck_y=puck_y):
                        state = defensive_state(skaters)
                        state.engine.puck_owner = owner
                        state.puck.y = puck_y
                        self.assertEqual(rf_defensezone(state), 1.0)
                        self.assertIs(isdone_defensezone(state), True)

    def test_conceding_overrides_possession_and_shots(self):
        state = defensive_state()
        self.assertEqual(rf_defensezone(state), 0.0)
        state.engine.puck_owner = 0
        state.team2.stats.shots += 1
        state.team2.stats.score += 1
        self.assertEqual(rf_defensezone(state), -1.0)
        self.assertIs(isdone_defensezone(state), True)
        state.EndFrame()
        state.engine.puck_owner = -256
        self.assertEqual(rf_defensezone(state), 0.0)
        self.assertIs(isdone_defensezone(state), False)

    def test_known_ownership_overrides_stale_stars(self):
        for owner in (-256, 5, 6, 11):
            with self.subTest(owner=owner):
                state = defensive_state()
                state.team1.player_haspuck = state.team1.goalie_haspuck = True
                state.engine.puck_owner = owner
                self.assertEqual(rf_defensezone(state), 0.0)
                self.assertIs(isdone_defensezone(state), False)

    def test_missing_owner_retains_legacy_possession_detection(self):
        for goalie in (False, True):
            with self.subTest(goalie=goalie):
                state = defensive_state(1)
                state.engine.puck_owner = -1
                state.team1.goalie_haspuck = goalie
                state.team1.player_haspuck = not goalie
                player = state.team1.goalie if goalie else state.team1.players[0]
                state.team1.stats.fullstar_x = player.x
                state.team1.stats.fullstar_y = player.y
                self.assertEqual(rf_defensezone(state), 0.0 if goalie else 1.0)
                self.assertIs(isdone_defensezone(state), not goalie)

    def test_missing_owner_without_possession_does_not_reward_default_stars(self):
        state = NHL94GameState(1)
        state.time = 300
        state.update_nets()
        self.assertEqual(rf_defensezone(state), 0.0)
        self.assertIs(isdone_defensezone(state), False)

    def test_no_possession_reward_for_loose_puck_or_opponent(self):
        for owner in (-256, *range(6, 12)):
            with self.subTest(owner=owner):
                state = defensive_state()
                state.engine.puck_owner = owner
                self.assertEqual(rf_defensezone(state), 0.0)
                self.assertIs(isdone_defensezone(state), False)

    def test_no_carry_friendly_shot_or_bodycheck_rewards(self):
        state = defensive_state()
        state.engine.puck_owner = -256
        for puck_y in (-230, 0, 150):
            state.puck.y = puck_y
            state.team1.stats.shots += 1
            state.team1.stats.bodychecks += 1
            state.team2.stats.bodychecks += 1
            state.team1.stats.score += 1
            self.assertEqual(rf_defensezone(state), 0.0)
            self.assertIs(isdone_defensezone(state), False)

    def test_timeout_preserves_clock_boundary_without_success_reward(self):
        state = defensive_state()
        state.time = 200
        self.assertIs(isdone_defensezone(state), False)
        self.assertEqual(rf_defensezone(state), 0.0)
        state.time = 199
        self.assertIs(isdone_defensezone(state), True)
        self.assertEqual(rf_defensezone(state), 0.0)
        state.engine.puck_owner = 0
        self.assertEqual(rf_defensezone(state), 1.0)

    def test_positions_do_not_add_shaping_rewards(self):
        for shooter in ((0, -180), (90, -180), (0, -270), (0, 0)):
            for defender in ((100, 0), (0, -222), (20, -250)):
                with self.subTest(shooter=shooter, defender=defender):
                    state = defensive_state()
                    state.team2.players[0].x, state.team2.players[0].y = shooter
                    state.team1.players[0].x, state.team1.players[0].y = defender
                    state.team1.goalie.x, state.team1.goalie.y = 0, -252
                    state.team1.goalie.is_pad_stack = 1.0
                    self.assertEqual(rf_defensezone(state), 0.0)
                    self.assertFalse(isdone_defensezone(state))

    def test_no_events_over_time_give_zero_reward(self):
        state = defensive_state()
        for _ in range(6000):
            self.assertEqual(rf_defensezone(state), 0.0)
            state.EndFrame()

    def test_decoded_episode_stops_on_first_shot(self):
        fixture = Path(__file__).resolve().parents[1] / 'fixtures/NHL94-Genesis-v0.json'
        info = json.loads(fixture.read_text(encoding='utf-8'))
        for team in ('p1', 'p2'):
            for slot in ('', '_2', '_3', '_4', '_5'):
                info[f'{team}{slot}_x'] = 100
                info[f'{team}{slot}_y'] = 0
        info.update(puck_owner=6, puck_x=0, puck_y=-180,
                    p2_x=0, p2_y=-180, g1_x=100, g1_y=-264)
        state = NHL94GameState(5)
        state.BeginFrame(info, [0] * 6)
        state.EndFrame()
        total = 0.0
        for frame in range(10):
            if frame == 2:
                info['p2_shots'] += 1
            state.BeginFrame(info, [0] * 6)
            total += rf_defensezone(state)
            done = isdone_defensezone(state)
            state.EndFrame()
            if done:
                break
        self.assertEqual(frame, 2)
        self.assertEqual(total, -1.0)


class DefenseZoneGoalieContracts(unittest.TestCase):
    def test_goalie_catch_hold_and_release_do_not_reward_or_end_episode(self):
        for skaters in (1, 2, 5):
            with self.subTest(skaters=skaters):
                state = defensive_state(skaters)
                state.engine.puck_owner = 5
                self.assertEqual(rf_defensezone(state), 0.0)
                self.assertFalse(isdone_defensezone(state))
                state.EndFrame()
                for _ in range(60):
                    self.assertEqual(rf_defensezone(state), 0.0)
                    self.assertFalse(isdone_defensezone(state))
                    state.EndFrame()
                state.engine.puck_owner = -256
                self.assertEqual(rf_defensezone(state), 0.0)
                self.assertFalse(isdone_defensezone(state))
                state.engine.puck_owner = 0
                self.assertEqual(rf_defensezone(state), 1.0)
                self.assertTrue(isdone_defensezone(state))

    def test_goalie_possession_does_not_hide_shots_goals_or_timeout(self):
        for outcome in ('shot', 'goal', 'timeout'):
            with self.subTest(outcome=outcome):
                state = defensive_state()
                state.engine.puck_owner = 5
                if outcome == 'shot':
                    state.team2.stats.shots += 1
                elif outcome == 'goal':
                    state.team2.stats.score += 1
                else:
                    state.time = 199
                self.assertEqual(rf_defensezone(state), 0.0 if outcome == 'timeout' else -1.0)
                self.assertTrue(isdone_defensezone(state))


class DefenseZoneShotContracts(unittest.TestCase):
    def test_shot_counter_increase_is_one_terminal_failure(self):
        for skaters in (1, 2, 5):
            with self.subTest(skaters=skaters):
                state = defensive_state(skaters)
                state.engine.puck_owner = -256
                state.team2.stats.shots = 7
                state.EndFrame()
                self.assertEqual(rf_defensezone(state), 0.0)
                self.assertFalse(isdone_defensezone(state))
                for added in (1, 3):
                    state.team2.stats.shots += added
                    self.assertEqual(rf_defensezone(state), -1.0)
                    self.assertTrue(isdone_defensezone(state))
                    state.EndFrame()
                    self.assertEqual(rf_defensezone(state), 0.0)
                    self.assertFalse(isdone_defensezone(state))

    def test_shot_penalty_uses_counter_not_shooting_flags(self):
        state = defensive_state()
        state.engine.puck_owner = -256
        state.engine.shot_taken = state.engine.shot_mode_active = 1.0
        self.assertEqual(rf_defensezone(state), 0.0)
        self.assertFalse(isdone_defensezone(state))
        state.team2.stats.shots += 1
        self.assertEqual(rf_defensezone(state), -1.0)
        self.assertTrue(isdone_defensezone(state))

    def test_shot_counter_decrease_has_no_reward_or_penalty(self):
        state = defensive_state()
        state.engine.puck_owner = -256
        state.team2.last_stats.shots = 7
        state.team2.stats.shots = 0
        self.assertEqual(rf_defensezone(state), 0.0)
        self.assertFalse(isdone_defensezone(state))
        state.EndFrame()
        state.team2.stats.shots = 1
        self.assertEqual(rf_defensezone(state), -1.0)
        self.assertTrue(isdone_defensezone(state))

    def test_shot_failure_overrides_recovery_save_or_timeout(self):
        for owner in (0, 5, -256):
            with self.subTest(owner=owner):
                state = defensive_state()
                state.engine.puck_owner = owner
                state.team2.stats.shots += 1
                if owner < 0:
                    state.time = 199
                self.assertEqual(rf_defensezone(state), -1.0)
                self.assertTrue(isdone_defensezone(state))


if __name__ == '__main__':
    unittest.main()

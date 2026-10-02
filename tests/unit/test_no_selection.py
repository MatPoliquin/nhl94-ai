"""A real controller sentinel is not a goalie, skater index, or fatal error."""
import unittest

import numpy as np

from nhl94_ai.env.wrappers import StochasticFrameSkip
from tests.unit.test_target_control import control_frame, target_env


class NoSelectionContracts(unittest.TestCase):
    def test_absent_selection_has_no_player_and_zero_relative_features(self):
        env = target_env()
        self.addCleanup(env.close)
        env.reset(seed=0)
        env.env.info.update(p1_control_slot=-1, shot_player=-1)
        observation, _, done, _, info = env.step([0.1, -0.6])
        state = env.game_state
        self.assertFalse(done)
        self.assertEqual(observation.shape, (322,))
        self.assertTrue(np.all(np.isfinite(observation)))
        self.assertEqual(state.team1.control, -1)
        self.assertIsNone(state.team1.get_controlled_player())
        self.assertEqual(state.team1.controlled_scnum(), -1)
        self.assertEqual(state.engine.controlled_is_shooter, 0.0)
        for player in [*state.team1.nz_players, state.team1.nz_goalie]:
            self.assertEqual(player.is_controlled, 0)
            for field in ('rel_controlled_x', 'rel_controlled_y', 'rel_controlled_vx',
                          'rel_controlled_vy', 'dist_to_controlled'):
                self.assertEqual(getattr(player, field), 0)
        for player in [*state.team2.players, state.team2.goalie]:
            self.assertEqual(player.dist_to_controlled_opp, 0)
        for field in ('rel_controlled_left', 'rel_controlled_right', 'rel_controlled_y'):
            self.assertEqual(getattr(state.team1.net, field), 0)
        self.assertEqual(info['target_control']['actual_slot'], -1)
        self.assertFalse(info['target_control']['selection_available'])
        self.assertIsNone(info['target_control']['distance'])
        view = env._build_opponent_view_state()
        self.assertEqual(view.team2.control, -1)
        self.assertIsNone(view.team2.get_controlled_player())
        self.assertTrue(all(player.dist_to_controlled == 0 for player in view.team2.players))

    def test_no_eligible_player_waits_without_movement_or_button_spam(self):
        env = target_env()
        self.addCleanup(env.close)
        env.reset(seed=0)
        env.env.info['p1_control_slot'] = -1
        for slot in range(5):
            env.env.info[f'target_unavailable_{slot}'] = 4
        env.step([0.1, -0.6])
        env = StochasticFrameSkip(env, 4, -1)
        for _ in range(3):
            observation, _, _, _, info = env.step([0.5, -0.5])
            diagnostics = info['target_control']
            self.assertEqual(diagnostics['mode'], 'waiting-for-selection')
            self.assertEqual(diagnostics['acting_slot'], -1)
            self.assertFalse(any(diagnostics['buttons']))
            self.assertTrue(np.all(np.isfinite(observation)))

    def test_controller_waits_and_resumes_only_when_selection_returns(self):
        env, state, info = control_frame()
        self.addCleanup(env.close)
        controller = env.target_controller
        info['p1_control_slot'] = -1
        controller.observe(state, info)
        self.assertFalse(controller._eligible(state, info, -1))
        self.assertFalse(controller._eligible(state, info, 5))
        action = controller.step([0.25, 0], state, info)
        np.testing.assert_array_equal(action, np.zeros(12))
        self.assertEqual(controller.mode, 'waiting-for-selection')
        self.assertEqual(controller.actual_slot, -1)
        state.action[4] = 1
        self.assertFalse(np.any(controller.step([0.25, 0], state, info)))
        info['p1_control_slot'] = 1
        controller.observe(state, info)
        state.action[4] = 0
        self.assertTrue(controller.step([0.25, 0], state, info)[7])
        self.assertEqual(controller.acting_slot, 1)

    def test_control_features_restore_after_no_selection(self):
        env = target_env()
        self.addCleanup(env.close)
        env.reset(seed=0)
        for slot in (-1, 3, -256, 5, -255, 0):
            env.env.info['p1_control_slot'] = slot
            observation, _, _, _, info = env.step([0.1, -0.6])
            flags = [player.is_controlled for player in env.game_state.team1.nz_players]
            flags.append(env.game_state.team1.nz_goalie.is_controlled)
            self.assertEqual(flags, [float(index == slot) for index in range(6)])
            self.assertEqual(info['target_control']['actual_slot'], max(-1, slot))
            self.assertTrue(np.all(np.isfinite(observation)))

    def test_other_invalid_slots_still_fail_explicitly(self):
        env, state, info = control_frame()
        self.addCleanup(env.close)
        for slot in (6, 7, 32767):
            with self.subTest(slot=slot):
                env.env.info['p1_control_slot'] = slot
                with self.assertRaisesRegex(ValueError, 'Invalid controller slot'):
                    env.step([0, 0])
                info['p1_control_slot'] = slot
                with self.assertRaisesRegex(ValueError, 'unselected'):
                    env.target_controller.observe(state, info)

    def test_no_selection_does_not_hide_terminal_events(self):
        for outcome, reward in (('goal', -1), ('shot', -1), ('recovery', 1), ('timeout', 0),
                                ('shot-recovery', -1), ('shot-save', -1)):
            with self.subTest(outcome=outcome):
                inner = target_env()
                env = StochasticFrameSkip(inner, 4, -1)
                self.addCleanup(env.close)
                env.reset(seed=0)
                info = inner.env.info
                info.update(p1_control_slot=-1, puck_owner=-256)
                if outcome.startswith('shot'):
                    info['p2_shots'] += 1
                if outcome == 'goal':
                    info['p2_score'] += 1
                elif outcome in ('recovery', 'shot-recovery'):
                    info['puck_owner'] = 0
                elif outcome == 'shot-save':
                    info['puck_owner'] = 5
                elif outcome == 'timeout':
                    info['time'] = 199
                observation, actual_reward, done, _, info = env.step([0.1, -0.6])
                self.assertTrue(done)
                self.assertAlmostEqual(actual_reward, reward)
                self.assertEqual(info['target_control']['frame'], 1)
                self.assertTrue(np.all(np.isfinite(observation)))


if __name__ == '__main__':
    unittest.main()

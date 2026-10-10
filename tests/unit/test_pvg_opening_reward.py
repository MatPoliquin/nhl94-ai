"""One-time pre-release shaping, actual ownership, geometry and frame timing."""
import copy
from types import SimpleNamespace
import unittest

import numpy as np

from nhl94_ai.env.observation import NHL94Observation2PEnv
from nhl94_ai.env.wrappers import StochasticFrameSkip
from nhl94_ai.game.state import NHL94GameState
from nhl94_ai.tasks.pvg import isdone_pvg, rf_pvg
from tests.unit.test_environment_contracts import FixtureEnv
from tests.unit.test_pvg import ShootoutFixtureEnv


def opening_info():
    info = copy.deepcopy(FixtureEnv().info)
    for team in ('p1', 'p2'):
        for slot in ('', '_2', '_3', '_4', '_5'):
            info[f'{team}{slot}_x'], info[f'{team}{slot}_y'] = -240, 0
    info.update(
        ba_ps_flags=0x0400, sflags=0, sflags2=0, p1_control_slot=0, puck_owner=0,
        p1_x=0, p1_y=190, p1_emptystar_x=0, p1_emptystar_y=190,
        puck_x=0, puck_y=210, g2_x=0, g2_y=247, g2_vel_x=0, g2_vel_y=0,
        goalie_chk_body=14,
    )
    return info


class PvGOpeningRewardTests(unittest.TestCase):
    def setUp(self):
        self.info = opening_info()
        self.state = NHL94GameState(5)
        self.step()

    def step(self, action=None):
        self.state.BeginFrame(self.info, action or [0] * 6)
        reward = rf_pvg(self.state)
        done = isdone_pvg(self.state)
        self.state.EndFrame()
        return reward, done

    def test_opening_rewards_once_and_does_not_end_the_attempt(self):
        self.info['g2_x'] = -70
        self.assertEqual(self.step(), (0.05, False))
        for _ in range(10):
            self.assertEqual(self.step(), (0.0, False))
        self.info['g2_x'] = 0
        self.assertEqual(self.step(), (0.0, False))
        self.info['g2_x'] = -70
        self.assertEqual(self.step(), (0.0, False))

    def test_an_opening_in_the_reset_frame_is_not_earned(self):
        self.info['g2_x'] = -70
        self.state = NHL94GameState(5)
        self.assertEqual(self.step(), (0.0, False))
        self.info['g2_x'] = 0
        self.step()
        self.info['g2_x'] = -70
        self.assertEqual(self.step(), (0.0, False))

    def test_holding_c_and_windup_remain_eligible_before_release(self):
        self.info.update(g2_x=-70, sflags=0x0800, ba_ps_flags=0x2400)
        self.assertEqual(self.step([0, 0, 0, 0, 0, 1]), (0.05, False))
        self.assertIs(self.state.is_shooting, True)
        self.assertIs(self.state.has_released_shot, False)

    def test_release_frame_and_regained_possession_cannot_earn_opening_bonus(self):
        self.info.update(g2_x=-70, sflags2=0x1000)
        self.assertEqual(self.step(), (0.0, False))
        self.info.update(g2_x=0, sflags2=0, puck_owner=-256)
        self.step()
        self.info.update(g2_x=-70, puck_owner=0)
        self.assertEqual(self.step(), (0.0, False))

    def test_only_the_controlled_carrier_in_shooting_range_is_eligible(self):
        for values in (
            {'puck_owner': -256}, {'puck_owner': 11}, {'p1_control_slot': 1},
            {'p1_y': 163}, {'p1_y': 264}, {'p1_y': 270}, {'p1_x': 100, 'p1_y': 190},
        ):
            with self.subTest(values=values):
                info = opening_info()
                state = NHL94GameState(5)
                state.BeginFrame(info, [0] * 6)
                state.EndFrame()
                info.update(g2_x=-70, **values)
                state.BeginFrame(info, [0] * 6)
                self.assertEqual(rf_pvg(state), 0.0)

    def test_range_boundary_is_inclusive(self):
        self.info.update(g2_x=-70, p1_y=164)
        self.assertEqual(self.step(), (0.05, False))

    def test_goal_overrides_same_frame_bonus_and_failure_has_no_penalty(self):
        self.state.created_pre_shot_opening = True
        self.state.team1.stats.score += 1
        self.assertEqual(rf_pvg(self.state), 1.0)
        self.state.EndFrame()
        self.state.is_shootout_active = False
        self.assertEqual(rf_pvg(self.state), 0.0)

    def test_bonus_is_cleared_when_the_frame_ends(self):
        self.info['g2_x'] = -70
        self.state.BeginFrame(self.info, [0] * 6)
        self.assertEqual(rf_pvg(self.state), 0.05)
        self.state.EndFrame()
        self.assertEqual(rf_pvg(self.state), 0.0)

    def test_shootout_completion_cannot_create_a_bonus(self):
        self.info.update(g2_x=-70, ba_ps_flags=0)
        self.assertEqual(self.step(), (0.0, True))


class PvGOpeningFrameSkipTests(unittest.TestCase):
    def test_reward_is_charged_once_across_repetition_and_rearms_only_after_reset(self):
        for skip in (1, 4, 10):
            with self.subTest(frame_skip=skip):
                args = SimpleNamespace(env='NHL94-Genesis-v0', nn='MlpPolicy',
                                       action_type='FILTERED', hyperparams_dict={})
                inner = NHL94Observation2PEnv(ShootoutFixtureEnv(), args, 1, 'PvG')
                env = StochasticFrameSkip(inner, skip, -1)
                self.addCleanup(env.close)
                inner.env.info.update(opening_info())
                for _ in range(2):
                    _, _ = env.reset(seed=0)
                    inner.env.info['g2_x'] = -70
                    _, reward, done, _, _ = env.step(np.zeros(12, dtype=np.int8))
                    self.assertEqual(reward, 0.05)
                    self.assertFalse(done)
                    _, reward, done, _, _ = env.step(np.zeros(12, dtype=np.int8))
                    self.assertEqual(reward, 0.0)
                    self.assertFalse(done)
                    inner.env.info['g2_x'] = 0


if __name__ == '__main__':
    unittest.main()

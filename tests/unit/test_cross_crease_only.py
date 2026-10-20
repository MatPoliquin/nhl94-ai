"""The specialist retains possession control and does not require the old shot certificate."""
from dataclasses import replace
import unittest

import numpy as np

from nhl94_ai.agents.cross_crease_only import CrossCreaseOnly, CrossingSettings
from nhl94_ai.evaluation.cross_crease_only import summary
from nhl94_ai.game.constants import GameConsts as Buttons
from tests.support.offense import crossing_state


class CrossCreaseOnlyTests(unittest.TestCase):
    def test_shoots_during_the_cut_without_body_center_crossing_or_a_save_animation(self):
        for direction in (-1, 1):
            for attack in (-1, 1):
                state = crossing_state(direction=direction, attack=attack)
                player = state.team1.players[0]
                player.x, player.y = -direction * 25, attack * 226
                state.puck.x, state.puck.y = -direction * 8, player.y
                state.team2.goalie.live_anim = 0
                model = CrossCreaseOnly()
                model.phase, model.direction = 'cross', direction
                action = model.step(state, 0)
                self.assertTrue(action[Buttons.INPUT_C])
                self.assertTrue(action[Buttons.INPUT_RIGHT if direction > 0 else Buttons.INPUT_LEFT])
                self.assertEqual(len(model.attempts), 1)
                self.assertLess(player.x * direction, 0)

    def test_shot_releases_c_and_stoppage_releases_all_buttons(self):
        state = crossing_state()
        model = CrossCreaseOnly(CrossingSettings(hold_frames=12))
        model.phase, model.direction, model.pressed = 'shot', 1, 0
        state.is_shooting = True
        for frame in range(20):
            action = model.step(state, frame)
            self.assertEqual(bool(action[Buttons.INPUT_C]), frame < 12)
            self.assertFalse(action[Buttons.INPUT_B])
        state.engine.clock_stopped = True
        self.assertFalse(model.step(state, 21).any())

    def test_loss_of_possession_never_turns_a_shot_into_a_check_or_pass(self):
        state = crossing_state()
        model = CrossCreaseOnly()
        model.phase, model.direction, model.pressed = 'shot', 1, 0
        state.engine.puck_owner = -1
        action = model.step(state, 4)
        self.assertFalse(action[Buttons.INPUT_C])
        self.assertFalse(action[Buttons.INPUT_B])

    def test_does_not_shoot_with_an_uncontrolled_carrier(self):
        state = crossing_state()
        state.team1.defense_control = 2
        model = CrossCreaseOnly()
        model.phase, model.direction = 'cross', 1
        self.assertFalse(model.step(state, 0).any())

    def test_movement_forecasts_leave_the_live_state_untouched(self):
        state = crossing_state()
        player = state.team1.players[0]
        before = dict(vars(player))
        model = CrossCreaseOnly()
        action = model.step(state, 0)
        self.assertEqual(vars(player), before)
        self.assertEqual(action.shape, (12,))
        self.assertTrue(np.isin(action, (0, 1)).all())
        self.assertFalse(action[Buttons.INPUT_B])

    def test_invalid_experiments_fail_before_emulation(self):
        for field, value in (('goal_gap', float('nan')), ('width', 0), ('release_x', 100),
                             ('hold_frames', 0), ('hold_frames', 1.5), ('depth_weight', float('inf'))):
            with self.subTest(field=field), self.assertRaises(ValueError):
                replace(CrossingSettings(), **{field: value})

    def test_contact_does_not_disqualify_an_actual_goal(self):
        row = {'goal': True, 'shots': 1, 'releases': [], 'presses': [{}],
               'goalie_contacts': 2, 'outcome': 'goal'}
        result = summary([row])
        self.assertEqual(result['goals'], 1)
        self.assertEqual(result['goalie_contacts'], 2)
        self.assertEqual(result['with_release'], 0)


if __name__ == '__main__':
    unittest.main()

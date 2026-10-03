"""Registration, shot edges, perspective, and reset contracts for Classic V1."""
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from nhl94_ai.agents.base import AgentInput
from nhl94_ai.agents.classic_v1 import ClassicAIV1Model
from nhl94_ai.agents.registry import ALIASES, CONTROLLERS, create_scripted
from nhl94_ai.env.intents import HOCKEY_INTENT_DPAD_ACTION_SPACE
from nhl94_ai.evaluation.benchmark import away_view
from nhl94_ai.game.constants import GameConsts as Buttons
from nhl94_ai.game.state import NHL94GameState


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


class ClassicV1Contracts(unittest.TestCase):
    def test_only_the_retained_controller_is_registered(self):
        self.assertEqual(set(CONTROLLERS), {'ClassicAIV1'})
        self.assertEqual(ALIASES, {'classic': 'ClassicAIV1', 'classic-v1': 'ClassicAIV1'})
        for name in (*ALIASES, *CONTROLLERS):
            with self.subTest(name=name):
                agent = create_scripted(name, SimpleNamespace(action_type='FILTERED'))
                self.assertIs(type(agent.controller), ClassicAIV1Model)

    def test_retired_controller_names_are_rejected(self):
        for name in ('ClassicAI', 'ClassicAIV2', 'ClassicAIV3', 'ClassicAIV4',
                     'classic-v2', 'classic-v3', 'classic-v4'):
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, 'Unknown scripted agent'):
                create_scripted(name, SimpleNamespace(action_type='FILTERED'))

    def test_controller_remains_inference_only(self):
        model = ClassicAIV1Model()
        for operation in (model.learn, model.save):
            with self.subTest(operation=operation.__name__), self.assertRaisesRegex(NotImplementedError, 'ClassicAIV1'):
                operation()

    def test_scripted_command_defaults_use_v1(self):
        from nhl94_ai.evaluation import benchmark, cpu_benchmark, runner
        from nhl94_ai.training import collect, dagger
        for module in (benchmark, cpu_benchmark, runner, collect, dagger):
            with self.subTest(command=module.__name__):
                parser = module.build_parser()
                self.assertEqual(parser.get_default('agent'), 'classic-v1')
                choices = next(action.choices for action in parser._actions if action.dest == 'agent')
                expected = {*ALIASES, 'classic-v1-direct', 'classic-v1-cross-crease'} if module is benchmark else set(ALIASES)
                self.assertEqual(set(choices), expected)

    def test_play_resolves_both_aliases_to_v1(self):
        from nhl94_ai.cli import main
        for name in ALIASES:
            with self.subTest(name=name), patch('nhl94_ai.evaluation.play.run') as run:
                self.assertEqual(main(['play', '--agent', name]), 0)
                self.assertEqual(run.call_args.args[0].nn, 'ClassicAIV1')

    def test_less_accurate_shooter_carries_closer_before_shooting(self):
        state = shooting_state()
        state.team1.players[0].y = 220
        for accuracy, decision in ((30, 'shoot'), (0, 'carry'), (None, 'shoot')):
            with self.subTest(accuracy=accuracy):
                state.team1.players[0].shot_accuracy = accuracy
                model = ClassicAIV1Model()
                model.predict_game_state(state)
                self.assertEqual(model._last_decision, decision)
        state.team1.players[0].y = 240
        state.team1.players[0].shot_accuracy = 0
        self.assertEqual(ClassicAIV1Model().predict_game_state(state)[0, Buttons.INPUT_C], 1)

    def test_shot_is_pressed_then_released_with_stable_aim(self):
        state, model = shooting_state(), ClassicAIV1Model()
        press = model.predict_game_state(state)[0]
        self.assertEqual(press[Buttons.INPUT_C], 1)
        self.assertEqual(press[Buttons.INPUT_UP], 0)
        self.assertEqual(press[Buttons.INPUT_LEFT], 1)
        # A moving goalie must not cause aim to oscillate during the windup.
        state.team2.goalie.x = -10
        for _ in range(6):
            release = model.predict_game_state(state)[0]
            self.assertEqual(release[Buttons.INPUT_C], 0)
            np.testing.assert_array_equal(release[4:8], press[4:8])

    def test_shot_aim_stays_in_world_coordinates_at_the_lower_net(self):
        state = shooting_state()
        state.engine.puck_owner = 6
        action = ClassicAIV1Model().predict_game_state(away_view(state))[0]
        self.assertEqual(action[Buttons.INPUT_C], 1)
        self.assertEqual(action[Buttons.INPUT_UP], 0)
        self.assertEqual(action[Buttons.INPUT_DOWN], 0)
        self.assertEqual(action[Buttons.INPUT_RIGHT], 1)

    def test_carrier_skates_towards_either_attacking_end(self):
        for away in (False, True):
            state = shooting_state()
            state.team1.players[0].y = state.team2.players[0].y = 0
            state.team1.goalie.y, state.team2.goalie.y = state.team1.net.y, state.team2.net.y
            state.engine.puck_owner = 6 if away else 0
            view = away_view(state) if away else state
            action = ClassicAIV1Model().predict_game_state(view)[0]
            self.assertEqual(action[Buttons.INPUT_DOWN if away else Buttons.INPUT_UP], 1)
            self.assertEqual(action[Buttons.INPUT_C], 0)

    def test_burst_is_released_before_starting_a_new_shot(self):
        model, state = ClassicAIV1Model(), shooting_state()
        state.engine.puck_owner = -256
        state.puck.x, state.puck.y = 30, 225
        # A defensive boost may have been submitted just before possession.
        model._c_down = True
        state.engine.puck_owner = 0
        self.assertEqual(model.predict_game_state(state)[0, Buttons.INPUT_C], 0)
        self.assertEqual(model.predict_game_state(state)[0, Buttons.INPUT_C], 1)

    def test_opponent_possession_cancels_follow_through(self):
        model, state = ClassicAIV1Model(), shooting_state()
        model.predict_game_state(state)
        state.engine.puck_owner = 6
        model.predict_game_state(state)
        self.assertNotEqual(model._last_decision, 'shot-follow-through')

    def test_stale_stars_do_not_trigger_a_shot(self):
        state = shooting_state()
        state.team1.player_haspuck = True
        for owner in (-256, 1, 6):
            state.engine.puck_owner = owner
            model = ClassicAIV1Model()
            model.predict_game_state(state)
            self.assertNotEqual(model._last_decision, 'shoot')

    def test_actions_are_valid_and_reset_is_deterministic(self):
        for schema in ('FILTERED', 'HOCKEY_INTENT_DPAD'):
            agent = create_scripted('classic-v1', SimpleNamespace(action_type=schema))
            inputs = AgentInput(shooting_state())
            first = agent.act(inputs).action
            for _ in range(20):
                action = agent.act(inputs).action
                self.assertEqual(action.dtype, np.int8)
                if schema == 'FILTERED':
                    self.assertEqual(action.shape, (12,))
                    self.assertTrue(np.all((action == 0) | (action == 1)))
                    self.assertFalse(action[4] and action[5] or action[6] and action[7])
                    self.assertFalse(action[0] and action[8])
                else:
                    self.assertTrue(np.all(action >= 0))
                    self.assertTrue(np.all(action < HOCKEY_INTENT_DPAD_ACTION_SPACE))
            agent.reset()
            np.testing.assert_array_equal(agent.act(inputs).action, first)
            np.testing.assert_array_equal(agent.get_action_preferences()[0], first)

    def test_goalie_intent_selects_the_intended_receiver(self):
        state = shooting_state()
        state.engine.puck_owner = 5
        state.team1.control = 0
        state.team1.goalie.y = -250
        state.team1.players[2].y = -210
        model = ClassicAIV1Model(SimpleNamespace(action_type='HOCKEY_INTENT_DPAD'))
        model.predict_game_state(state)
        action = model.predict_game_state(state)[0]
        self.assertEqual(action[0], 9)  # PASS_TEAMMATE_3


if __name__ == '__main__':
    unittest.main()

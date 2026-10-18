"""Crossing finishes must coexist with Classic's passes and frame scheduler."""
from copy import deepcopy
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from nhl94_ai.agents.classic_v1 import ClassicAIV1Model
from nhl94_ai.agents.crossing_finish import CrossingFinishSelector
from nhl94_ai.agents.lifecycle import PassState
from nhl94_ai.evaluation.cpu_benchmark import MATCHUPS, build_parser, run
from nhl94_ai.game.constants import GameConsts as Buttons
from tests.unit.test_classic_offense import offense_state, live_one_timer_state
from tests.unit.test_cross_crease import crossing_state


def ready_state(direction=1, attack=1, away=False):
    state = crossing_state(direction=direction, attack=attack)
    player = state.team1.players[0]
    player.x, player.y = -direction * 25, attack * 226
    state.puck.x, state.puck.y = -direction * 8, player.y
    if away:
        state.team1.controller, state.team2.controller = 2, 1
        state.team1.defense_control = state.engine.puck_owner = 6
    return state


def hybrid():
    return ClassicAIV1Model(SimpleNamespace(opportunistic_cross_crease=True))


class CrossingFinishContracts(unittest.TestCase):
    def test_finish_works_in_both_directions_ends_and_controller_slots(self):
        for direction in (-1, 1):
            for attack in (-1, 1):
                for away in (False, True):
                    model = hybrid()
                    action = model.predict_frame(ready_state(direction, attack, away))[0]
                    self.assertEqual(model._last_decision, 'crossing-shoot')
                    self.assertTrue(action[Buttons.INPUT_C])
                    self.assertEqual(model.shot.side, direction)
                    self.assertEqual(model.shot.slot, 6 if away else 0)

    def test_hold_is_twelve_native_frames_across_decision_intervals(self):
        for interval in (1, 4, 8, 16):
            with self.subTest(interval=interval):
                model, state = hybrid(), ready_state()
                actions = [model.predict_frame(state, interval)[0] for _ in range(24)]
                self.assertEqual([int(a[Buttons.INPUT_C]) for a in actions], [1] * 12 + [0] * 12)
                self.assertTrue(all(a[Buttons.INPUT_RIGHT] for a in actions))
                self.assertEqual(model.crossing_finish.metrics['selected'], 1)

    def test_crowded_or_unready_positions_keep_original_actions(self):
        crowded = ready_state()
        crowded.team2.players[0].x, crowded.team2.players[0].y = 0, 218
        states = [crowded, offense_state()]
        stationary = ready_state()
        stationary.team1.players[0].motion_x = 0
        states.append(stationary)
        for state in states:
            baseline, model = ClassicAIV1Model(), hybrid()
            for _ in range(24):
                np.testing.assert_array_equal(baseline.predict_frame(state), model.predict_frame(state))
                self.assertEqual(baseline._last_decision, model._last_decision)
            self.assertEqual(model.crossing_finish.metrics['selected'], 0)

    def test_pending_pass_and_one_timer_keep_their_inputs(self):
        for kind in ('pass', 'one-timer'):
            state = ready_state()
            baseline, model = ClassicAIV1Model(), hybrid()
            for agent in (baseline, model):
                if kind == 'pass':
                    agent.offense.pass_action.pending = PassState(0, 1, 0, 60, point=(110, -180))
                else:
                    agent.one_timer.start(state, 1, 0)
            for _ in range(12):
                np.testing.assert_array_equal(baseline.predict_frame(state), model.predict_frame(state))
            self.assertEqual(model.crossing_finish.metrics['evaluations'], 0)

    def test_live_one_timer_choice_keeps_priority(self):
        state = live_one_timer_state()
        baseline, model = ClassicAIV1Model(), hybrid()
        np.testing.assert_array_equal(baseline.predict_frame(state), model.predict_frame(state))
        self.assertEqual(model._last_decision, 'one-timer-pass')
        self.assertEqual(model.crossing_finish.metrics['evaluations'], 0)

    def test_public_play_flag_and_environment_validation(self):
        from nhl94_ai.cli import main
        from nhl94_ai.config import EnvironmentConfig
        with patch('nhl94_ai.evaluation.play.run') as play:
            main(['play', '--agent', 'classic-v1', '--env', 'NHL94-Genesis-v0',
                  '--opportunistic-cross-crease'])
        args = play.call_args.args[0]
        self.assertTrue(args.opportunistic_cross_crease)
        self.assertEqual(EnvironmentConfig.from_args(args).nn, 'ClassicAIV1')
        args.selfplay = True
        with self.assertRaises(ValueError):
            EnvironmentConfig.from_args(args)

    def test_handoff_releases_c_immediately(self):
        for change in ('stopped', 'teammate', 'opponent', 'control', 'falling'):
            with self.subTest(change=change):
                model, state = hybrid(), ready_state()
                model.predict_frame(state)
                if change == 'stopped':
                    state.engine.clock_stopped = True
                elif change in ('teammate', 'opponent'):
                    state.engine.puck_owner = 1 if change == 'teammate' else 6
                elif change == 'control':
                    state.team1.defense_control = 1
                else:
                    state.team1.players[0].is_falling = True
                action = model.predict_frame(state)[0]
                self.assertFalse(action.any())
                self.assertFalse(model.shot.active(model.scheduler.decisions))

    def test_ineligible_and_missing_feedback_do_not_start_a_finish(self):
        for field, value in (('motion_x', None), ('shot_power', None), ('selection_flags', 0x20)):
            state = ready_state()
            setattr(state.team1.players[0], field, value)
            before = deepcopy(state.puck.__dict__)
            self.assertIsNone(CrossingFinishSelector().choose(state, 0))
            self.assertEqual(state.puck.__dict__, before)

    def test_benchmark_flag_reaches_trials(self):
        args = build_parser().parse_args(['--opportunistic-cross-crease', '--trials', '1'])
        fixtures = []
        def match(fixture):
            fixtures.append(fixture)
            return dict(matchup=fixture[1], side=MATCHUPS[fixture[1]][1], completed=True,
                        goals=[0, 0], one_timers=[0, 0], one_timer_goals=[0, 0], decisions={},
                        crossing_finish_metrics={'selected': 2})
        with patch('nhl94_ai.evaluation.cpu_benchmark.cpu_match', side_effect=match), patch('builtins.print'):
            report = run(args)
        self.assertTrue(all(len(f) == 25 and f[24] is True for f in fixtures))
        self.assertTrue(all(r['crossing_finish_metrics']['selected'] == 2 for r in report['summary'].values()))

    def test_unsupported_combinations_fail_explicitly(self):
        for extra in ({'action_type': 'HOCKEY_INTENT_DPAD'}, {'classic_control_interval': 4},
                      {'selfplay': True}, {'cross_crease': True}, {'possession_value': True},
                      {'shot_placement': 'missing.json'}, {'retention_model': 'missing.json'}):
            with self.assertRaises(ValueError):
                ClassicAIV1Model(SimpleNamespace(opportunistic_cross_crease=True, **extra))


if __name__ == '__main__':
    unittest.main()

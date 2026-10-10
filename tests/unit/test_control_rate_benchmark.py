"""Rate limits must survive real agent entrypoints, substeps and resets."""
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from nhl94_ai.agents.base import AgentInput, configure_scripted_frames
from nhl94_ai.agents.control_rate import ControlRateLimit
from nhl94_ai.agents.registry import create_scripted
from nhl94_ai.config import EnvironmentConfig
from nhl94_ai.env.wrappers import StochasticFrameSkip
from nhl94_ai.evaluation.control_rate_benchmark import portable
from tests.unit.test_classic_defense import defense_state
from tests.unit.test_environment_contracts import wrapped


def make_agent(interval=0):
    agent = create_scripted('classic-v1', SimpleNamespace(
        action_type='FILTERED', classic_control_interval=interval))
    agent.frame_skip = 4
    return agent


class ControlRateContracts(unittest.TestCase):
    def test_custom_intervals_hold_all_inputs_and_do_not_inspect_intermediate_states(self):
        for interval in (1, 2, 3, 4, 10, 20, 37):
            with self.subTest(interval=interval):
                agent, state = make_agent(interval), defense_state()
                with patch.object(agent.controller, 'predict_frame', wraps=agent.controller.predict_frame) as predict:
                    for frame in range(interval * 2 + 1):
                        # An opaque object cannot be read as a game state.
                        observation = state if frame % interval == 0 else object()
                        output = agent.act(AgentInput(observation))
                        if frame % interval == 0:
                            held, snapshot = output.action.copy(), output.decision
                        else:
                            np.testing.assert_array_equal(output.action, held)
                            self.assertIs(output.decision, snapshot)
                        output.action[:] = -1  # Must not corrupt the cached input.
                        self.assertEqual(agent.scheduler.frames, frame + 1)
                    self.assertEqual(predict.call_count, 3)
                self.assertEqual(agent.control_rate.observations, 3)
                self.assertEqual(agent.defense.frames, interval * 2 + 1)

    def test_hold_preserves_pending_actions_and_advances_goalie_clock(self):
        agent = make_agent(10)
        first = agent.predict_game_state(defense_state()).copy()
        model = agent.controller
        model.goalie = SimpleNamespace(frames=1, release_until=2)
        model.defense.pending_check = {'until': 3}
        model.defense.pending_switch = {'frame': 1}
        buttons = (model.buttons.b_down, model.buttons.c_down)
        decisions = model.scheduler.decisions
        for _ in range(9):
            np.testing.assert_array_equal(agent.predict_game_state(object()), first)
        self.assertEqual((model.scheduler.frames, model.defense.frames, model.goalie.frames), (10, 10, 10))
        self.assertEqual(model.scheduler.decisions, decisions)
        self.assertEqual(model.scheduler.remaining, 0)
        self.assertEqual(model.defense.pending_check, {'until': 3})
        self.assertEqual(model.defense.pending_switch, {'frame': 1})
        self.assertEqual((model.buttons.b_down, model.buttons.c_down), buttons)

    def test_interval_one_matches_reactive_controller_and_reset_discards_held_input(self):
        actual, reference = make_agent(1), make_agent()
        state = defense_state()
        for _ in range(12):
            np.testing.assert_array_equal(actual.predict_game_state(state), reference.predict_game_state(state))
        self.assertEqual(actual.scheduler.decisions, reference.scheduler.decisions)
        self.assertEqual(actual.scheduler.frames, reference.scheduler.frames)
        self.assertIsNone(reference.control_rate)
        slow = make_agent(20)
        slow.predict_game_state(state)
        previous = slow.controller
        slow.reset()
        self.assertIsNot(slow.controller, previous)
        self.assertIs(slow.control_rate.controller, slow.controller)
        self.assertIsNone(slow.last_output)
        self.assertEqual((slow.control_rate.remaining, slow.scheduler.frames), (0, 0))
        np.testing.assert_array_equal(slow.predict_game_state(state), make_agent(20).predict_game_state(state))

    def test_substeps_keep_cadence_across_nondivisible_outer_frame_skip(self):
        inner = wrapped()
        env = StochasticFrameSkip(inner, 4, -1)
        self.addCleanup(env.close)
        agent = make_agent(10)
        configure_scripted_frames(agent, env, record=True)
        observation, _ = env.reset(seed=7)
        inner.game_state = defense_state()
        actual = []
        with patch.object(agent.controller, 'predict_frame', wraps=agent.controller.predict_frame) as predict:
            for _ in range(6):
                action = agent.act(AgentInput(inner.game_state, observation)).action
                observation, _, _, _, info = env.step(action)
                actual.extend(frame[1] for frame in info['scripted_frames'])
            self.assertEqual(predict.call_count, 3)  # Native frames 0, 10, 20.
        self.assertEqual(agent.scheduler.frames, 24)
        for frame, action in enumerate(actual):
            np.testing.assert_array_equal(action, actual[frame // 10 * 10])

    def test_strict_direct_calls_use_native_time_without_explicit_frame_configuration(self):
        agent = make_agent(4)
        agent.frame_skip = None
        agent.predict_game_state(defense_state())
        for _ in range(3):
            agent.predict_game_state(object())
        self.assertEqual(agent.scheduler.frames, 4)
        self.assertEqual(agent.control_rate.observations, 1)

    def test_rejects_bad_rate_and_action_macros(self):
        for interval in (-1, 1.5, True, None):
            with self.assertRaises(ValueError):
                make_agent(interval)
        with self.assertRaises(ValueError):
            ControlRateLimit(make_agent().controller, 0)
        with self.assertRaisesRegex(ValueError, 'FILTERED'):
            create_scripted('classic-v1', SimpleNamespace(
                action_type='HOCKEY_INTENT_DPAD', classic_control_interval=4))
        with self.assertRaisesRegex(ValueError, 'Classic agent'):
            EnvironmentConfig.from_args(SimpleNamespace(classic_control_interval=10))
        from nhl94_ai.evaluation.play import NHL94Player, parse_cmdline
        args = parse_cmdline(['--nn', 'ClassicAIV1', '--mode', 'model_vs_game',
                             '--classic-control-interval', '10', '--model_2', 'second.zip'])
        with self.assertRaisesRegex(ValueError, 'one Classic agent'):
            NHL94Player(args, None, need_display=False)

    def test_cli_option_is_shared_by_play_evaluation_and_cpu_benchmark(self):
        from nhl94_ai.cli import main
        for command, module in (('play', 'play'), ('evaluate', 'runner'), ('benchmark-cpu', 'cpu_benchmark')):
            with patch(f'nhl94_ai.evaluation.{module}.run') as run:
                main([command, '--agent', 'classic-v1', '--classic-control-interval', '20'])
            self.assertEqual(run.call_args.args[0].classic_control_interval, 20)

    def test_cpu_benchmark_passes_custom_interval_as_named_option(self):
        from nhl94_ai.evaluation.cpu_benchmark import build_parser, run
        args = build_parser().parse_args(['--classic-control-interval', '10', '--trials', '1'])
        def match(fixture, *, classic_control_interval):
            self.assertEqual(classic_control_interval, 10)
            return dict(matchup=fixture[1], side=1, completed=True, goals=[0, 0],
                        one_timers=[0, 0], one_timer_goals=[0, 0], decisions={})
        with patch('nhl94_ai.evaluation.cpu_benchmark.cpu_match', side_effect=match), patch('builtins.print'):
            report = run(args)
        self.assertEqual(report['settings']['classic_control_interval'], 10)
        self.assertIn('nhl94_ai/agents/control_rate.py', report['sources'])

    def test_nonfinite_diagnostics_are_portable(self):
        self.assertEqual(portable({'values': (float('inf'), float('-inf'), float('nan'), 1)}),
                         {'values': ['inf', '-inf', 'nan', 1]})

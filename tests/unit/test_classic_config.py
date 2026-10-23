"""Player config parsing, override precedence, and global ratings isolation."""
from copy import deepcopy
from contextlib import ExitStack
import json
from pathlib import Path
import pickle
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from nhl94_ai.agents.classic_config import (
    DEFAULTS, NEUTRAL_RATINGS, decision_state, load_classic_config,
)
from nhl94_ai.agents.classic_v1 import ClassicAIV1Model
from nhl94_ai.agents.registry import create_scripted
from nhl94_ai.evaluation.play import parse_cmdline
from tests.support.defense import goalie_state


class ClassicConfigTests(unittest.TestCase):
    def setUp(self):
        stack = ExitStack()
        self.addCleanup(stack.close)
        directory = stack.enter_context(tempfile.TemporaryDirectory())
        self.path = Path(directory) / 'classic.json'

    def write(self, values):
        self.path.write_text(json.dumps(values), encoding='utf-8')
        return str(self.path)

    def play(self, *overrides):
        return parse_cmdline(['--nn', 'ClassicAIV1', '--env', 'NHL94-Genesis-v0',
                             '--mode', 'model_vs_game', '--classic-config', str(self.path),
                             *overrides])

    def test_comments_defaults_and_invalid_values(self):
        self.path.write_text('''{
          // Both teams use neutral ratings.
          "player_ratings_consideration": false /* no gameplay changes */
        }''', encoding='utf-8')
        self.assertEqual(load_classic_config(self.path), DEFAULTS | {'player_ratings_consideration': False})
        for values in ([1], {'unknown': True}, {'decision_hz': True}, {'decision_hz': 60.0},
                       {'decision_hz': 12}, {'ai_goalie': 'false'},
                       {'player_ratings_consideration': 0}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                load_classic_config(self.write(values))
        self.path.write_text('{"ai_goalie": true,}', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'Invalid Classic config'):
            load_classic_config(self.path)

    def test_missing_explicit_file_fails_but_missing_checkout_uses_defaults(self):
        with self.assertRaises(FileNotFoundError):
            load_classic_config(self.path)
        with patch('nhl94_ai.agents.classic_config.DEFAULT_CONFIG_PATH', self.path):
            self.assertEqual(load_classic_config(), DEFAULTS)

    def test_default_checkout_file_is_loaded_automatically(self):
        self.write({'decision_hz': 10, 'ai_goalie': False, 'player_ratings_consideration': False})
        with patch('nhl94_ai.agents.classic_config.DEFAULT_CONFIG_PATH', self.path):
            args = parse_cmdline(['--nn', 'ClassicAIV1', '--mode', 'model_vs_game',
                                 '--env', 'NHL94-Genesis-v0'])
        self.assertEqual(args.classic_control_interval, 6)
        self.assertEqual(args.goalie_policy, 'off')
        self.assertFalse(args.player_ratings_consideration)

    def test_all_rates_goalie_defaults_and_cli_overrides(self):
        for hz, interval in ((60, 0), (15, 4), (10, 6), (6, 10)):
            with self.subTest(hz=hz):
                self.write({'decision_hz': hz})
                args = self.play()
                self.assertEqual(args.classic_control_interval, interval)
                self.assertEqual(args.goalie_policy, 'selective')
                self.assertIn('ManualGoalie', args.state)
                agent = create_scripted('classic', args)
                self.assertIsNotNone(agent.goalie)
                self.assertEqual(agent.control_interval, interval)
        self.write({'decision_hz': 6, 'ai_goalie': False})
        args = self.play('--classic-control-interval', '0', '--goalie-policy', 'always')
        self.assertEqual(args.classic_control_interval, 0)
        self.assertFalse(hasattr(args, 'classic_decision_interval'))
        self.assertEqual(args.goalie_policy, 'always')
        args = self.play()
        self.assertIsNone(create_scripted('classic', args).goalie)
        self.write({'ai_goalie': True})
        self.assertEqual(self.play('--goalie-policy', 'off').goalie_policy, 'off')

    def test_installed_cli_alias_resolves_config_before_run(self):
        from nhl94_ai.cli import main
        self.write({'decision_hz': 15, 'ai_goalie': False})
        with patch('nhl94_ai.evaluation.play.run') as run:
            main(['play', '--agent', 'classic', '--env', 'NHL94-Genesis-v0',
                  '--classic-config', str(self.path)])
        args = run.call_args.args[0]
        self.assertEqual(args.classic_control_interval, 4)
        self.assertEqual(args.goalie_policy, 'off')

    def test_60hz_config_runs_tactics_on_every_native_frame(self):
        from tests.support.offense import shooting_state
        self.write({'decision_hz': 60, 'ai_goalie': False})
        args = self.play()
        for configured in (False, True):
            with self.subTest(configured=configured):
                agent = create_scripted('classic', args)
                if configured:
                    agent.frame_skip = 4  # Playback's outer dispatch interval.
                for frame in range(24):
                    agent.predict_game_state(shooting_state())
                    self.assertEqual(agent.scheduler.decisions, frame + 1)
                    self.assertEqual(agent.scheduler.frames, frame + 1)
                self.assertEqual(agent.scheduler.interval, 1)

    def test_lower_config_rates_keep_observations_and_tactics_at_the_selected_rate(self):
        from tests.support.offense import shooting_state
        for hz, repeat in ((15, 4), (10, 6), (6, 10)):
            with self.subTest(hz=hz):
                self.write({'decision_hz': hz, 'ai_goalie': False})
                agent = create_scripted('classic', self.play())
                agent.frame_skip = 4
                for frame in range(30):
                    # Held frames must not even read a fresh game state.
                    state = shooting_state() if frame % repeat == 0 else object()
                    agent.predict_game_state(state)
                    self.assertEqual(agent.scheduler.decisions, frame // repeat + 1)
                    self.assertEqual(agent.scheduler.frames, frame + 1)

    def test_relative_path_in_general_options_file(self):
        from nhl94_ai.cli import main
        self.write({'decision_hz': 10})
        options = self.path.with_name('options.json')
        options.write_text(json.dumps({'agent': 'classic', 'env': 'NHL94-Genesis-v0',
                                       'classic_config': self.path.name}), encoding='utf-8')
        with patch('nhl94_ai.evaluation.play.run') as run:
            main(['play', '--config', str(options)])
        self.assertEqual(run.call_args.args[0].classic_control_interval, 6)

    def test_non_classic_play_does_not_read_bot_defaults(self):
        with patch('nhl94_ai.agents.classic_config.load_classic_config') as load:
            args = parse_cmdline(['--mode', 'player_vs_game'])
        load.assert_not_called()
        self.assertEqual(args.classic_control_interval, 0)
        with self.assertRaisesRegex(ValueError, 'Classic playback'):
            parse_cmdline(['--mode', 'player_vs_game', '--classic-config', str(self.path)])


class RatingsTests(unittest.TestCase):
    def test_teammate_coordination_uses_neutral_ratings_before_selecting_roles(self):
        from nhl94_ai.agents.teammates import TeammateCoordinator
        from tests.support.teammates import team_args, team_info
        coordinator = TeammateCoordinator(team_args(player_ratings_consideration=False))
        info = team_info(defense_0_speed=30, defense_0_agility=28,
                         defense_6_speed=2, defense_6_agility=4)
        before = deepcopy(info)
        with patch.object(coordinator, '_choose_primary', wraps=coordinator._choose_primary) as choose:
            coordinator.act(info)
        state = choose.call_args.args[0]
        self.assertEqual(state.team1.players[0].speed, 15)
        self.assertEqual(state.team2.players[0].speed, 15)
        self.assertEqual(info, before)

    def test_neutral_view_covers_both_teams_and_preserves_live_state(self):
        state = goalie_state(selected=True)
        state.team1.players[0].input_state = {'speed': 30, 'weight': 120, 'handedness': 1, 'vx': 777}
        before = pickle.dumps(state)
        self.assertIs(decision_state(state), state)
        neutral = decision_state(state, False)
        for original, team in ((state.team1, neutral.team1), (state.team2, neutral.team2)):
            for source, player in zip([*original.players, original.goalie], [*team.players, team.goalie]):
                for name, value in NEUTRAL_RATINGS.items():
                    if getattr(source, name, None) is not None:
                        self.assertEqual(getattr(player, name), value)
                for name in ('x', 'y', 'motion_x', 'motion_y', 'energy', 'handedness', 'role'):
                    self.assertEqual(getattr(player, name), getattr(source, name))
            self.assertEqual(team.goalie_stats.speed, 15)
            self.assertEqual(team.nz_goalie_stats.speed, 0.5)
        self.assertEqual(neutral.team1.players[0].input_state,
                         {'speed': 15, 'weight': 64, 'handedness': 1, 'vx': 777})
        self.assertEqual(pickle.dumps(state), before)

    def test_ratings_are_neutral_before_goalie_and_tactical_entrypoints(self):
        state = goalie_state(selected=True)
        model = ClassicAIV1Model(SimpleNamespace(player_ratings_consideration=False))
        for entrypoint, downstream in (('predict_frame', '_predict_native_frame'),
                                      ('predict_game_state', '_goalie_frame')):
            with self.subTest(entrypoint=entrypoint), patch.object(
                    model, downstream, return_value=np.zeros((1, 12))) as decide:
                getattr(model, entrypoint)(state)
                observed = decide.call_args.args[0]
                self.assertEqual(observed.team1.goalie.agility, 15)
                self.assertEqual(observed.team2.players[0].speed, 15)
        self.assertEqual(state.team1.goalie.agility, 20)

    def test_different_ratings_produce_identical_manual_goalie_inputs_when_disabled(self):
        low = goalie_state(selected=True)
        high = deepcopy(low)
        for state, value in ((low, 2), (high, 28)):
            for team in (state.team1, state.team2):
                for player in [*team.players, team.goalie]:
                    for name in NEUTRAL_RATINGS:
                        if getattr(player, name, None) is not None:
                            setattr(player, name, value)
        args = SimpleNamespace(goalie_policy='always', goalie_positioning='classic',
                               player_ratings_consideration=False, classic_control_interval=6)
        agents = [create_scripted('classic', args) for _ in range(2)]
        for _ in range(18):
            np.testing.assert_array_equal(agents[0].predict_game_state(low),
                                          agents[1].predict_game_state(high))
        self.assertEqual(agents[0].control_rate.observations, 3)
        self.assertEqual(agents[0].goalie.frames, 18)


if __name__ == '__main__':
    unittest.main()

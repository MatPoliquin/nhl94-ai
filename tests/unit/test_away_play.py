"""Away Classic playback keeps the opposing team under the real CPU."""
from copy import deepcopy
import unittest
from unittest.mock import Mock, patch

import numpy as np

from nhl94_ai.agents.base import AgentOutput
from nhl94_ai.agents.multi_model import NHL94AISystem
from nhl94_ai.cli import main
from nhl94_ai.env.observation import NHL94Observation2PEnv
from nhl94_ai.evaluation.play import NHL94Player, parse_cmdline
from nhl94_ai.game.ram import restore_away_control, select_cpu_side
from tests.unit.test_environment_contracts import FixtureEnv


def away_args(*flags):
    return parse_cmdline(['--mode=model_vs_game', '--nn=ClassicAIV1',
                          '--env=NHL94-Genesis-v0', '--rf=PostPlay', '--side=away', *flags])


class AwayFixture(FixtureEnv):
    def __init__(self):
        super().__init__()
        self.info.update(defense_team1=1, defense_team2=0, defense_control1=2,
                         defense_control2=-1, defense_2_flags=0x88, defense_8_flags=0x40)
        self.saved_info = deepcopy(self.info)

    def reset(self, **kwargs):
        self.info = deepcopy(self.saved_info)
        result = super().reset(**kwargs)
        self.data.lookup_all = lambda: dict(self.info)
        self.data.set_value = self.set_value
        self.data.update_ram = Mock()
        return result


class AwayPlayContracts(unittest.TestCase):
    def test_scripted_play_uses_fresh_reset_feedback_without_blind_initial_frames(self):
        player = NHL94Player.__new__(NHL94Player)
        player.args, player.need_display = away_args(), False
        fresh, following = [{'reset': 'first'}], [{'reset': 'next'}]
        vector = Mock(reset_infos=fresh)
        vector.reset.return_value = np.zeros((1, 2))
        player.display_env = vector
        player.ai_sys = Mock(models=[Mock()])
        player.ai_sys.predict.return_value = np.zeros((1, 12), dtype=np.int8)
        player._reset_playback_timer = Mock()
        player._throttle_display_frame = Mock()
        class Finished(Exception):
            """Stop once the second episode receives its initial action."""
        def step(_actions):
            if vector.reset_infos is following:
                raise Finished
            vector.reset_infos = following
            return np.zeros((1, 2)), np.zeros(1), np.ones(1, dtype=bool), [{'terminal': True}]
        vector.step.side_effect = step
        with self.assertRaises(Finished):
            player.play_player_or_game_mode(continuous=True, need_reset=False)
        self.assertIs(player.ai_sys.predict.call_args_list[0].kwargs['info'], fresh)
        self.assertIs(player.ai_sys.predict.call_args_list[1].kwargs['info'], following)
        player.ai_sys.models[0].reset.assert_called_once()

    def test_full_installed_command_selects_classic_away_against_cpu(self):
        with patch('nhl94_ai.evaluation.play.run') as run:
            main(['play', '--agent', 'classic-v1', '--env', 'NHL94-Genesis-v0',
                  '--state', 'CanadiensVsNordiques.start', '--side', 'away',
                  '--max_playback_speed', '1.0'])
        args = run.call_args.args[0]
        self.assertEqual((args.nn, args.mode, args.side, args.num_players),
                         ('ClassicAIV1', 'model_vs_game', 'away', 1))
        self.assertEqual(args.state, 'CanadiensVsNordiques.start')
        self.assertEqual(args.max_playback_speed, 1.0)
        with patch('nhl94_ai.evaluation.play.run') as run:
            main(['play', '--agent', 'classic-v1'])
        self.assertEqual(run.call_args.args[0].side, 'home')
        self.assertEqual(run.call_args.args[0].mode, 'model_vs_game')

    def test_unsupported_away_modes_fail_before_creating_emulators(self):
        for flags in (['--mode=player_vs_model'], ['--mode=player_vs_game'],
                      ['--mode=model_vs_model'], ['--nn=MlpPolicy'],
                      ['--env=NHL942on2-Genesis-v0'], ['--rf=DefenseZone'],
                      ['--action_type=HOCKEY_INTENT_DPAD'], ['--action_type=TARGET_POSITION'],
                      ['--model_2=other.zip']):
            with self.subTest(flags=flags), patch('nhl94_ai.evaluation.play.init_env') as init:
                with self.assertRaisesRegex(ValueError, '--side away requires'):
                    NHL94Player(away_args(*flags), None, need_display=False)
                init.assert_not_called()

    def test_transfer_releases_home_and_preserves_unrelated_player_flags(self):
        data = Mock()
        info = dict(defense_team1=1, defense_team2=0, defense_control1=2,
                    defense_2_flags=0x88, defense_8_flags=0x40)
        select_cpu_side(data, info, 2)
        self.assertEqual(dict(call.args for call in data.set_value.call_args_list),
                         {'defense_team1': 2, 'defense_control1': 8,
                          'defense_2_flags': 0x82, 'defense_8_flags': 0x48})
        data.reset_mock()
        select_cpu_side(data, info, 1)
        data.set_value.assert_not_called()
        for changed in (dict(info, defense_team2=2), dict(info, defense_team1=2),
                        dict(info, defense_control1=-1), dict(info, defense_control1=5)):
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                select_cpu_side(data, changed, 2)
            data.set_value.assert_not_called()
        with self.assertRaises(ValueError):
            select_cpu_side(data, info, 3)
        data.set_value.assert_not_called()

    def test_reset_and_repeat_reset_transfer_before_first_emulator_step(self):
        args = away_args()
        env = NHL94Observation2PEnv(AwayFixture(), args, 1, 'PostPlay')
        self.addCleanup(env.close)
        for _ in range(2):
            with patch.object(env.env, 'step', wraps=env.env.step) as step:
                observation, info = env.reset()
            self.assertEqual((info['defense_team1'], info['defense_team2']), (2, 0))
            self.assertEqual(info['defense_control1'], 8)
            self.assertEqual(env.game_state.team2.defense_control, 8)
            self.assertIsNone(env.game_state.team1.defense_control)
            self.assertEqual(np.asarray(observation).shape, (env.NUM_PARAMS,))
            self.assertEqual(env.game_state.team2.control, 3)
            self.assertEqual(env.game_state.team1.control, -1)
            self.assertEqual(step.call_args.args[0].shape, (12,))
            env.step(np.zeros(12, dtype=np.int8))
            self.assertEqual(env.env.info['defense_team2'], 0)

    def test_cross_team_rom_reassignment_restores_actual_away_control(self):
        env = NHL94Observation2PEnv(AwayFixture(), away_args(), 1, 'PostPlay')
        self.addCleanup(env.close)
        env.reset()
        env.env.info.update(defense_control1=3, defense_3_flags=0xA8, defense_9_flags=0x40,
                            defense_8_flags=0x40)
        with self.assertWarnsRegex(RuntimeWarning, 'restored away slot 9'):
            _, _, _, _, info = env.step(np.zeros(12, dtype=np.int8))
        self.assertEqual(info['away_control_restored_from'], 3)
        self.assertEqual(env.env.info['defense_control1'], 9)
        self.assertEqual(env.env.info['defense_3_flags'], 0xA2)
        self.assertEqual(env.env.info['defense_9_flags'], 0x48)
        env.env.data.update_ram.assert_called_once()
        self.assertEqual(env.game_state.team1.controlled_scnum(), -1)
        self.assertEqual(env.game_state.team2.controlled_scnum(), 9)

    def test_away_reassignment_guard_never_silently_accepts_invalid_feedback(self):
        info = dict(defense_team1=2, defense_team2=0, defense_control1=3,
                    defense_3_flags=0xA8, defense_9_flags=0x40)
        for changed in (dict(info, defense_team1=1), dict(info, defense_team2=1),
                        dict(info, defense_control1=5), dict(info, defense_control1=12),
                        dict(info, defense_3_flags=0xA0)):
            data = Mock()
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                restore_away_control(data, changed)
            data.set_value.assert_not_called()
        for slot in (-256, 6, 11):
            data = Mock()
            unchanged = dict(info, defense_control1=slot)
            self.assertIs(restore_away_control(data, unchanged), unchanged)
            data.set_value.assert_not_called()

    def test_vector_auto_reset_reapplies_away_assignment(self):
        from stable_baselines3.common.vec_env import DummyVecEnv
        inner = NHL94Observation2PEnv(AwayFixture(), away_args(), 1, 'PostPlay')
        inner.env.truncate = True
        env = DummyVecEnv([lambda: inner])
        self.addCleanup(env.close)
        env.reset()
        _, _, done, info = env.step([np.zeros(12, dtype=np.int8)])
        self.assertTrue(done[0])
        self.assertEqual(info[0]['defense_team1'], 2)
        self.assertEqual(env.reset_infos[0]['defense_team1'], 2)
        self.assertEqual(env.reset_infos[0]['defense_team2'], 0)
        self.assertEqual(inner.game_state.team2.defense_control, 8)

    def test_agent_gets_away_team_without_mutating_world_or_mirroring_buttons(self):
        env = NHL94Observation2PEnv(AwayFixture(), away_args(), 1, 'PostPlay')
        self.addCleanup(env.close)
        observation, info = env.reset()
        system = NHL94AISystem(away_args(), env, None)
        action = np.arange(12, dtype=np.int8) % 2
        agent = Mock()
        agent.act.return_value = AgentOutput(action, {'side': 'away'})
        system.models[1] = agent
        system.model_in_use = system.num_models = 1
        for _ in range(2):
            with patch('nhl94_ai.agents.multi_model.get_model_probabilities', return_value=[[0] * 12]):
                result = system.predict(observation, [info], True)
            view = agent.act.call_args.args[0].game_state
            self.assertEqual(view.team1.controller, 2)
            self.assertEqual(view.team1.defense_control, 8)
            self.assertEqual(view.team1.control, 3)
            self.assertIs(view.team1, system.game_state.team2)
            self.assertIs(view.team2, system.game_state.team1)
            self.assertEqual(system.game_state.team1.controller, 1)
            self.assertEqual(view.team1.players[0].x, info['p2_x'])
            self.assertEqual(view.team1.net.y, system.game_state.team2.net.y)
            np.testing.assert_array_equal(result[0], action)
        self.assertEqual(system.last_diagnostics, {'side': 'away'})

    def test_failed_transfer_does_not_claim_a_cpu_opponent(self):
        env = NHL94Observation2PEnv(AwayFixture(), away_args(), 1, 'PostPlay')
        self.addCleanup(env.close)
        with patch('nhl94_ai.env.observation.select_cpu_side'):
            with self.assertRaisesRegex(ValueError, 'home team under CPU control'):
                env.reset()


if __name__ == '__main__':
    unittest.main()

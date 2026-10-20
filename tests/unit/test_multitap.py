"""Three-bot configuration, independent state, routing, and wrapper contracts."""
from collections import defaultdict
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import gymnasium as gym
import numpy as np
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from nhl94_ai.agents.teammates import TeammateCoordinator
from nhl94_ai.cli import main, parse_configured
from nhl94_ai.config import EnvironmentConfig, resolve_play_controllers
from nhl94_ai.env.actions import filter_menu_actions
from nhl94_ai.env.observation import NHL94Observation2PEnv
from nhl94_ai.evaluation.play import build_parser, parse_cmdline, NHL94Player
from nhl94_ai.evaluation.teammates import replay_trace
from nhl94_ai.game.multitap import controller_snapshot, require_multitap
from nhl94_ai.game.ram import configure_teammates, teammate_controller_info
from tests.unit.test_teammates import TeammateFixture, team_args, team_info, FixtureData


def three_info(human=False, **values):
    info = team_info(defense_team3=1, defense_team4=2 if human else 0,
                     defense_control3=2, defense_control4=6 if human else -1)
    for slot in range(12):
        info[f'defense_{slot}_flags'] = (8 if slot in (0, 1, 2) or human and slot == 6 else 0) | (64 if slot >= 6 else 0)
    info.update(values)
    return info


def three_args(human=False, **values):
    return team_args(team_size=3, num_players=4 if human else 3,
                     mode='team_vs_player' if human else 'team_vs_game', **values)


class MultitapFixture(TeammateFixture):
    def reset(self, **kwargs):
        super().reset(**kwargs)
        for port in (3, 4):
            for kind, base, dtype in (('team', 0xFFC328, '>u2'), ('control', 0xFFC320, '>i2')):
                self.data.set_variable(f'defense_{kind}{port}', {'address': base + (port - 1) * 2, 'type': dtype})
        for slot in range(12):
            self.data.set_variable(f'defense_{slot}_flags',
                                   {'address': 0xFFB04A + slot * 128 + 0x62, 'type': '|u1'})
        return self.render(), self.data.lookup_all()


def wrapper(human=False, side='home'):
    base = MultitapFixture()
    base.action_space = gym.spaces.MultiBinary(48 if human else 36)
    return NHL94Observation2PEnv(base, three_args(human, side=side), 4 if human else 3, 'PostPlay')


class MultitapConfigTests(unittest.TestCase):
    def test_both_entrypoints_resolve_omitted_count_and_check_explicit_count(self):
        for mode, count in (('team_vs_game', 3), ('team_vs_player', 4)):
            flags = ['--mode', mode, '--team-size', '3', '--env', 'NHL94-Genesis-v0']
            legacy = parse_cmdline([*flags, '--nn', 'ClassicAIV1'])
            with patch('nhl94_ai.evaluation.play.run') as run:
                main(['play', *flags, '--agent', 'classic-v1'])
            self.assertEqual((legacy.num_players, run.call_args.args[0].num_players), (count, count))
            self.assertEqual(parse_cmdline([*flags, '--num_players', str(count)]).num_players, count)
            with self.assertRaisesRegex(ValueError, 'omit --num_players'):
                parse_cmdline([*flags, '--num_players', '2'])
            with patch('nhl94_ai.evaluation.play.run'), self.assertRaisesRegex(ValueError, 'omit --num_players'):
                main(['play', *flags, '--num_players', '2'])

    def test_json_counts_and_unrelated_options(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'config.json'
            options = dict(mode='team_vs_player', team_size=3)
            for explicit in (None, 4, 2):
                path.write_text(json.dumps({**options, **({'num_players': explicit} if explicit else {})}))
                args = parse_configured(build_parser(), ['--config', str(path)])
                if explicit == 2:
                    with self.assertRaises(ValueError):
                        resolve_play_controllers(args)
                else:
                    self.assertEqual(resolve_play_controllers(args).num_players, 4)
        for flag in ('--team-size=2', '--team-strategy=independent'):
            with self.assertRaisesRegex(ValueError, 'require team_vs'):
                parse_cmdline(['--mode=model_vs_game', flag])
        for options in (dict(nn='MlpPolicy'), dict(goalie_policy='selective'), dict(side='invalid'),
                        dict(env='NHL94-Snes-v0'), dict(selfplay=True)):
            with self.subTest(options=options), self.assertRaises(ValueError):
                EnvironmentConfig.from_args(three_args(**options))

    def test_older_dependency_and_headless_human_fail_early(self):
        with patch('nhl94_ai.game.multitap.retro.RetroEmulator', object), self.assertRaisesRegex(RuntimeError, 'stable-retro'):
            require_multitap()
        with self.assertRaisesRegex(ValueError, 'display'):
            NHL94Player(three_args(True), None, need_display=False)


class MultitapStateTests(unittest.TestCase):
    def test_unique_initial_assignments_and_unused_controller(self):
        for human in (False, True):
            data = FixtureData(team_info())
            selected = configure_teammates(data, team_size=3, human=human)
            self.assertEqual(len(set(selected)), 3)
            self.assertEqual([data.memory.extract(0xFFC328 + p * 2, '>u2') for p in range(4)],
                             [1, 1, 1, 2 if human else 0])
            self.assertEqual(data.memory.extract(0xFFC326, '>i2'), 6 if human else -1)

    def test_transient_native_alias_is_unavailable_and_recovers_without_writes(self):
        info = three_info(True, defense_team2=2, defense_control2=6)
        corrected = controller_snapshot(info, 3, True)
        self.assertTrue(corrected['teammate_transient'])
        self.assertEqual([corrected[f'defense_control{i}'] for i in range(1, 5)], [-1] * 4)
        self.assertEqual(info['defense_control2'], 6)
        sentinel = three_info(defense_control1=-254, defense_0_flags=0)
        self.assertFalse(controller_snapshot(sentinel, 3)['teammate_transient'])
        stable = three_info(True)
        for controller in (1, 2, 3):
            view = teammate_controller_info(stable, controller, team_size=3, human=True)
            self.assertEqual(view['p1_control_slot'], controller - 1)
            self.assertEqual(view['p2_control_slot'], 6)
        with self.assertRaises(ValueError):
            controller_snapshot(three_info(defense_control3=12), 3)

    def test_independent_histories_and_distinct_support_targets(self):
        for strategy in ('independent', 'coordinated'):
            agent = TeammateCoordinator(three_args(), strategy=strategy)
            first = agent.act(three_info())
            self.assertEqual(first.action.shape, (36,))
            self.assertEqual(len({id(state) for state in agent.states}), 3)
            self.assertEqual(len({id(bot) for bot in agent.controllers}), 3)
            rows = first.diagnostics['teammates']['controllers']
            self.assertEqual([row['slot'] for row in rows], [0, 1, 2])
            if strategy == 'coordinated':
                self.assertNotEqual(rows[1]['target'], rows[2]['target'])
                self.assertEqual(rows[2]['role'], 'deep-coverage')
            agent.act(three_info(puck_owner=2))
            self.assertEqual(agent.primary, 2)
            self.assertEqual([bot.scheduler.frames for bot in agent.controllers], [2, 2, 2])

    def test_deep_support_ignores_a_previous_outlet_coast_command(self):
        agent = TeammateCoordinator(three_args(teammate_support_options=True))
        agent.support[2].coast = True
        result = agent.act(three_info())
        self.assertEqual(result.diagnostics['teammates']['controllers'][2]['role'], 'deep-coverage')
        self.assertTrue(result.action[24:32].any())

    def test_third_controller_can_send_and_receive_shared_passes(self):
        agent = TeammateCoordinator(three_args())
        agent.act(three_info())
        for sender, receiver in ((1, 2), (3, 1)):
            request = dict(passer=sender - 1, receiver=receiver, point=(0, 0), frame=10,
                           deadline=60, purpose='pass')
            agent.receiving.pending = None
            agent.receiving.propose(request, sender, [0, 1, 2], agent.states[0], 10)
            self.assertEqual(agent.receiving.pending.passer_controller, sender)
            self.assertEqual(agent.receiving.pending.receiver_controller, receiver + 1)

    def test_unavailable_selection_reassigns_primary_and_persistent_alias_fails(self):
        agent = TeammateCoordinator(three_args())
        info = three_info(defense_control1=-254, defense_0_flags=0, puck_owner=-1)
        agent.act(info)
        self.assertIn(agent.primary, (1, 2))
        for _ in range(120):
            action = agent.act(three_info(defense_control1=2)).action
            self.assertFalse(action.any())
        with self.assertRaisesRegex(RuntimeError, 'ownership failed'):
            agent.act(three_info(defense_control1=2))

    def test_filter_rejects_bad_shapes_and_values(self):
        data = FixtureData(team_info())
        for width in (12, 24, 36, 48):
            result = filter_menu_actions(data, np.ones(width))
            self.assertEqual(result.shape, (width,))
            self.assertTrue(all(result[3::12]))
        for bad in (np.zeros(35), np.ones((3, 12)), np.full(36, np.nan), np.full(48, 2)):
            with self.assertRaises(ValueError):
                filter_menu_actions(data, bad)

    def test_trace_layout_is_rejected_before_emulator_creation(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'bad.jsonl'
            for header in (dict(schema='nhl94-teammate-inputs-v2', strategy='independent', team_size=4),
                           dict(schema='nhl94-teammate-inputs-v1', strategy='independent', team_size=3)):
                path.write_text(json.dumps(header) + '\n')
                with patch('nhl94_ai.evaluation.teammates.make_match_env') as make, self.assertRaises(ValueError):
                    replay_trace(path)
                make.assert_not_called()


class MultitapWrapperTests(unittest.TestCase):
    def test_monitor_vecenv_batching_scalar_reward_and_auto_reset(self):
        for human in (False, True):
            inner = wrapper(human)
            vector = DummyVecEnv([lambda inner=inner: Monitor(inner)])
            self.addCleanup(vector.close)
            with patch('nhl94_ai.game.multitap.prepare_multitap_save'):
                obs = vector.reset()
                self.assertEqual(vector.action_space.shape, (48 if human else 36,))
                self.assertEqual(obs.shape, (1, inner.NUM_PARAMS))
                action = np.zeros((1, 48 if human else 36), dtype=np.int8)
                action[0, 7::12] = 1
                _, reward, done, _ = vector.step(action)
                self.assertEqual(reward.shape, (1,))
                self.assertFalse(done[0])
                np.testing.assert_array_equal(inner.env.last_action, action[0])
                for history in inner.teammate_action_states:
                    self.assertTrue(history['last_env_action'][7])
                inner.teammate_started_periods.add(3)
                inner.env.data.memory.assign(0xFFC466, '>u2', 2)
                inner.env.data.memory.assign(0xFFC468, '>u2', 0)
                _, reward, done, info = vector.step(action)
                self.assertTrue(done[0])
                self.assertIn('terminal_observation', info[0])
                self.assertTrue(all(not state['last_env_action'].any() for state in inner.teammate_action_states))

    def test_display_merges_only_last_slot_and_releases(self):
        import pygame
        from nhl94_ai.ui.debug import NHL94DebugDisplay
        with patch.dict(os.environ, {'SDL_VIDEODRIVER': 'dummy', 'SDL_AUDIODRIVER': 'dummy'}), \
                patch('pygame.joystick.get_count', return_value=0):
            pygame.init()
            self.addCleanup(pygame.quit)
            inner = wrapper(True, side='away')
            vector = DummyVecEnv([lambda inner=inner: Monitor(inner)])
            self.addCleanup(vector.close)
            with patch('nhl94_ai.game.multitap.prepare_multitap_save'):
                vector.reset()
            display = NHL94DebugDisplay(vector, three_args(True, side='away'), 0, 'ClassicAIV1', [])
            bots = np.zeros(36, dtype=np.int8)
            bots[6::12] = 1
            for pressed in ({pygame.K_RIGHT: True}, {}):
                with patch('pygame.key.get_pressed', return_value=defaultdict(bool, pressed)):
                    display.step([bots])
                np.testing.assert_array_equal(inner.env.last_action[:36], bots)
                self.assertEqual(inner.env.last_action[43], int(bool(pressed)))
                self.assertEqual(inner.env.last_action[36:].sum(), int(bool(pressed)))
            pad = [0] * 12
            pad[8] = 1
            with patch.object(display.gamepad, 'read', return_value=pad), \
                    patch('pygame.key.get_pressed', return_value=defaultdict(bool)):
                display.step([bots])
            np.testing.assert_array_equal(inner.env.last_action[:36], bots)
            np.testing.assert_array_equal(inner.env.last_action[36:], pad)
            pygame.event.post(pygame.event.Event(pygame.WINDOWFOCUSLOST))
            display.process_events()
            with patch('pygame.key.get_pressed', return_value=defaultdict(bool, {pygame.K_RIGHT: True})):
                display.step([bots])
            self.assertFalse(inner.env.last_action[36:].any())
            np.testing.assert_array_equal(inner.env.last_action[:36], bots)

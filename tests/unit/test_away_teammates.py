"""Both physical teams share the same bot-first input and replay contracts."""
from itertools import product
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import gymnasium as gym
import numpy as np

from nhl94_ai.agents.teammates import TeammateCoordinator
from nhl94_ai.agents.teammate_pressure import SharedPressure
from nhl94_ai.config import EnvironmentConfig
from nhl94_ai.env.observation import NHL94Observation2PEnv
from nhl94_ai.evaluation.play import NHL94Player, parse_cmdline
from nhl94_ai.evaluation.teammates import play_match, replay_trace, trace_layout
from nhl94_ai.game.multitap import controller_snapshot
from nhl94_ai.game.ram import configure_teammates, teammate_controller_info
from tests.support.defense import pressure_state
from tests.support.teammates import MultitapFixture, team_args


def layout(side, size, human):
    base = MultitapFixture()
    base.reset()
    selected = configure_teammates(base.data, side, team_size=size, human=human)
    return base, selected, base.data.lookup_all()


class AwayTeammateTests(unittest.TestCase):
    def test_sandwich_roles_follow_physical_away_slots_and_changed_ends(self):
        for side, size, period in product((1, 2), (2, 3), (1, 2, 3)):
            state = pressure_state()
            state.team1.controller, state.team2.controller = side, 3 - side
            state.engine.puck_owner = (2 - side) * 6
            state.period = period
            state.update_nets()
            if state.team1.net.y > 0:
                for team in (state.team1, state.team2):
                    for player in (*team.players, team.goalie):
                        player.y *= -1
                        player.facing = (4 - player.facing) % 8 if player.facing is not None else None
            slots = [(side - 1) * 6 + i for i in range(size)]
            planner = SharedPressure()
            first = planner.choose(state, slots, 1)
            self.assertEqual([a.plan.mode for a in first], ['contain', 'close'] + (['cover'] if size == 3 else []))
            self.assertGreater(first[0].plan.target[1] * state.team1.net.y, 0)
            self.assertLess(first[1].plan.target[1] * state.team1.net.y, 0)
            self.assertLessEqual(sum(a.contact for a in first), 1)
            switched = slots[::-1]
            self.assertIsNone(planner.choose(state, switched, 2))
            reassigned = planner.choose(state, switched, 14)
            self.assertEqual([a.plan.mode for a in reassigned], [a.plan.mode for a in first][::-1])

    def test_all_reset_layouts_have_distinct_skaters_and_only_selected_joysticks(self):
        for side, size, human in product((1, 2), (2, 3), (False, True)):
            with self.subTest(side=side, size=size, human=human):
                base, selected, info = layout(side, size, human)
                self.addCleanup(base.close)
                self.assertEqual(selected, tuple((side - 1) * 6 + i for i in (2, 0, 1)[:size]))
                teams = [side] * size + ([3 - side] if human else []) + [0] * (4 - size - human)
                self.assertEqual([info[f'defense_team{i}'] for i in range(1, 5)], teams)
                slots = list(selected) + ([(2 - side) * 6] if human else [])
                self.assertEqual([info[f'defense_control{i}'] for i in range(1, 5)],
                                 slots + [-1] * (4 - len(slots)))
                for slot in range(12):
                    flags = base.data.memory.extract(0xFFB04A + slot * 128 + 0x62, '|u1')
                    self.assertEqual(bool(flags & 8), slot in slots)
                    if slot not in slots:
                        self.assertTrue(flags & 2)
                self.assertEqual((info['goalie_mode_1'], info['goalie_mode_2']), (1, 1))

    def test_cli_and_player_accept_every_layout(self):
        for side, size, human in product(('home', 'away'), (2, 3), (False, True)):
            args = parse_cmdline(['--mode', 'team_vs_player' if human else 'team_vs_game',
                                  '--team-size', str(size), '--side', side,
                                  '--env', 'NHL94-Genesis-v0', '--nn', 'ClassicAIV1', '--rf', 'PostPlay'])
            EnvironmentConfig.from_args(args)
            with patch.object(NHL94Player, 'init_teammate_mode') as init:
                NHL94Player(args, None)
                init.assert_called_once()
            self.assertEqual(args.num_players, size + human)

    def test_wrapper_routes_inputs_and_physical_display_without_gameplay_repairs(self):
        for side, size, human in product(('home', 'away'), (2, 3), (False, True)):
            args = team_args(side=side, team_size=size, num_players=size + human,
                             mode='team_vs_player' if human else 'team_vs_game')
            base = MultitapFixture()
            base.action_space = gym.spaces.MultiBinary(12 * (size + human))
            env = NHL94Observation2PEnv(base, args, size + human, 'PostPlay')
            self.addCleanup(env.close)
            with patch('nhl94_ai.game.multitap.prepare_multitap_save'), \
                    patch('nhl94_ai.env.observation.restore_away_control',
                          side_effect=AssertionError('single-controller repair')):
                _, info = env.reset()
                physical = 1 if side == 'home' else 2
                self.assertEqual(info[f'p{physical}_control_slot'], (physical - 1) * 6 + 2)
                self.assertEqual(info[f'p{3-physical}_control_slot'], (2 - physical) * 6 if human else -1)
                buttons = np.zeros(12 * (size + human), dtype=np.int8)
                buttons[7::12] = 1
                with patch.object(base.data.memory, 'assign', side_effect=AssertionError('gameplay RAM write')):
                    env.step(buttons)
                np.testing.assert_array_equal(base.last_action, buttons)
                self.assertEqual(env.game_state.team1.controller, 1)
                self.assertEqual(env.game_state.team2.controller, 2)

    def test_away_alias_is_neutral_and_stable_cross_team_selection_fails(self):
        base, _, info = layout(2, 3, True)
        self.addCleanup(base.close)
        alias = dict(info, defense_team2=1, defense_control2=0)
        corrected = controller_snapshot(alias, 3, True, 2)
        self.assertTrue(corrected['teammate_transient'])
        agent = TeammateCoordinator(team_args(side='away', team_size=3, mode='team_vs_player'))
        self.assertFalse(agent.act(alias).action.any())
        self.assertEqual(alias['defense_control2'], 0)
        stable = teammate_controller_info(info, 3, 2, team_size=3, human=True)
        self.assertEqual((stable['p2_control_slot'], stable['p1_control_slot']), (7, 0))
        bad = dict(info, defense_control3=1)
        bad['defense_7_flags'] &= ~8
        bad['defense_1_flags'] |= 8
        with self.assertRaisesRegex(ValueError, 'Cross-team'):
            controller_snapshot(bad, 3, True, 2)

    def test_highlight_restore_alias_waits_in_menu_but_live_timeout_is_retained(self):
        base, _, info = layout(2, 3, False)
        self.addCleanup(base.close)
        agent = TeammateCoordinator(team_args(side='away', team_size=3))
        alias = dict(info, defense_control1=11, sflags=0x0100, period=1)
        for _ in range(180):
            action = agent.act(alias).action
            self.assertFalse(np.delete(action, 3).any())
        self.assertEqual(agent.transient_frames, 0)
        alias['sflags'] = 0
        for _ in range(120):
            self.assertFalse(agent.act(alias).action.any())
        with self.assertRaisesRegex(RuntimeError, 'ownership failed'):
            agent.act(alias)

    def test_replay_validates_side_and_physical_ownership_before_creation(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'trace.jsonl'
            header = dict(schema='nhl94-teammate-inputs-v2', strategy='coordinated',
                          side='away', **trace_layout(3, True, 2))
            self.assertEqual(header['teams'], [2, 2, 2, 1])
            for change in ({'side': 'invalid'}, {'teams': [1, 1, 1, 2]}, {'opponent': 'invalid'}):
                path.write_text(json.dumps({**header, **change}) + '\n')
                with patch('nhl94_ai.evaluation.teammates.make_match_env') as make:
                    with self.assertRaises(ValueError):
                        replay_trace(path)
                    make.assert_not_called()
        for size, human in ((3, None), (2, lambda *_: np.zeros(12))):
            with self.assertRaisesRegex(ValueError, 'teammate strategy'):
                play_match(strategy='single', side='away', team_size=size, human_input=human)


if __name__ == '__main__':
    unittest.main()

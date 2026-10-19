"""Same-team controller identity, coordinated roles and native input histories."""
from copy import deepcopy
import importlib.util
import os
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import gymnasium as gym
import numpy as np

from nhl94_ai.agents.teammates import TeammateCoordinator
from nhl94_ai.agents.lifecycle import PassState
from nhl94_ai.agents.teammate_receiving import SharedReceiving
from nhl94_ai.agents.teammate_support import SupportPlanner
from nhl94_ai.cli import main
from nhl94_ai.config import EnvironmentConfig
from nhl94_ai.env.observation import NHL94Observation2PEnv
from nhl94_ai.evaluation.teammate_metrics import ReceptionOutcomes
from nhl94_ai.evaluation.play import parse_cmdline
from nhl94_ai.game.ram import configure_teammates, teammate_controller_info
from nhl94_ai.game.state import NHL94GameState
from tests.ram_fixture import FixtureMemory
from tests.unit.test_environment_contracts import FixtureEnv


def team_args(**values):
    return SimpleNamespace(**dict({'mode': 'team_vs_game', 'env': 'NHL94-Genesis-v0',
                                   'nn': 'ClassicAIV1', 'num_players': 2, 'rf': 'PostPlay',
                                   'action_type': 'FILTERED', 'goalie_policy': 'off',
                                   'hyperparams_dict': {'frame_skip': 4}}, **values))


def team_info(**values):
    info = FixtureEnv().info
    info.update(defense_team1=1, defense_team2=1, defense_control1=0, defense_control2=1,
                teammate_clock=274, goalie_mode_1=1, goalie_mode_2=1,
                goalie_clock_flags=0, goalie_input_block=0, puck_owner=0,
                p1_pass_attempts=0, pass_target=1)
    info.update(values)
    return info


class FixtureData:
    def __init__(self, info):
        self.memory = FixtureMemory(info)
        for name, address, kind in (
                ('defense_team1', 0xFFC328, '>u2'), ('defense_team2', 0xFFC32A, '>u2'),
                ('defense_control1', 0xFFC320, '>i2'), ('defense_control2', 0xFFC322, '>i2'),
                ('teammate_clock', 0xFFC468, '>u2'), ('goalie_mode_1', 0xFFD05A, '>u2'),
                ('goalie_mode_2', 0xFFD05C, '>u2')):
            self.memory.fields[name] = address, kind
            if name in info:
                self.memory.assign(address, kind, info[name])

    def set_variable(self, name, spec):
        self.memory.fields[name] = spec['address'], spec['type']

    @staticmethod
    def filter_action(buttons):
        return buttons & ~sum(1 << index for index in (1, 2, 3, 9, 10, 11))

    def update_ram(self):
        for name, (address, kind) in self.memory.fields.items():
            self.memory.info[name] = self.memory.extract(address, kind)

    def lookup_all(self):
        self.update_ram()
        return deepcopy(self.memory.info)


class TeammateFixture(FixtureEnv):
    def __init__(self):
        super().__init__()
        self.initial = team_info(defense_team2=0, defense_control1=2, defense_control2=-1)
        self.info = deepcopy(self.initial)
        self.data = FixtureData(self.info)
        self.action_space = gym.spaces.MultiBinary(24)
        self.render_mode = 'rgb_array'

    def reset(self, **kwargs):
        gym.Env.reset(self, seed=kwargs.get('seed'))
        self.info = deepcopy(self.initial)
        self.data = FixtureData(self.info)
        self.steps = 0
        return self.render(), self.data.lookup_all()

    def step(self, action):
        self.steps += 1
        self.last_action = np.asarray(action).copy()
        return self.render(), 0.0, False, False, self.data.lookup_all()


class TeammateSetupTests(unittest.TestCase):
    def test_cli_uses_two_controllers_and_automatic_goalies(self):
        with patch('nhl94_ai.evaluation.play.run') as run:
            main(['play', '--agent', 'classic-v1', '--env', 'NHL94-Genesis-v0', '--mode', 'team_vs_game'])
        self.assertEqual(run.call_args.args[0].num_players, 2)
        args = parse_cmdline(['--env=NHL94-Genesis-v0', '--nn=ClassicAIV1', '--mode=team_vs_game'])
        self.assertEqual((args.num_players, args.goalie_policy), (2, 'off'))

    def test_unsupported_modes_fail_before_emulator_creation(self):
        EnvironmentConfig.from_args(team_args())
        for changes in ({'env': 'NHL942on2-Genesis-v0'}, {'env': 'NHL94-Snes-v0'},
                        {'side': 'away'}, {'nn': 'MlpPolicy'}, {'goalie_policy': 'selective'},
                        {'num_players': 1}, {'selfplay': True}, {'action_type': 'MULTI_DISCRETE'},
                        {'deke': True}, {'model_1': 'old.zip'}, {'classic_control_interval': 4}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                EnvironmentConfig.from_args(team_args(**changes))

    def test_reset_matches_native_two_teammate_assignment_and_releases_cpu(self):
        env = TeammateFixture()
        data = env.data
        first, second = configure_teammates(data)
        self.assertEqual((first, second), (2, 0))
        self.assertEqual((data.memory.extract(0xFFD04A, '>u2'),
                          data.memory.extract(0xFFC328, '>u2'),
                          data.memory.extract(0xFFC32A, '>u2')), (3, 1, 1))
        for slot in range(12):
            self.assertEqual(bool(data.memory.extract(0xFFB04A + slot * 0x80 + 0x62, '|u1') & 8),
                             slot in (first, second))
        self.assertEqual(data.memory.extract(0xFFC32C, '>u2'), 0)
        self.assertEqual(data.memory.extract(0xFFD05A, '>u2'), 1)

    def test_partial_or_shootout_save_is_rejected_before_any_writes(self):
        for address, value in ((0xFFD048, 4), (0xFFB04A + 0x34, -1), (0xFFC320, -1)):
            data = TeammateFixture().data
            data.memory.assign(address, '>i2', value)
            original = bytes(data.memory.buffer)
            with self.subTest(address=address), self.assertRaises(ValueError):
                configure_teammates(data)
            self.assertEqual(bytes(data.memory.buffer), original)

    def test_two_controller_views_preserve_identity_and_source_info(self):
        info = team_info(defense_control1=1, defense_control2=4)
        original = deepcopy(info)
        states = []
        for controller in (1, 2):
            state = NHL94GameState(5)
            state.BeginFrame(teammate_controller_info(info, controller), [0] * 6)
            states.append(state)
        self.assertEqual([s.team1.controlled_scnum() for s in states], [1, 4])
        self.assertEqual([s.team1.defense_control for s in states], [1, 4])
        self.assertEqual([s.team2.control for s in states], [-1, -1])
        self.assertEqual(info, original)
        self.assertIsNot(states[0].team1.players[0], states[1].team1.players[0])

    def test_invalid_assignments_fail_and_negative_selection_is_preserved(self):
        view = teammate_controller_info(team_info(defense_control1=-256))
        self.assertEqual(view['p1_control_slot'], -256)
        for changes in ({'defense_team2': 2}, {'defense_control1': 6},
                        {'defense_control2': 0}, {'goalie_mode_1': 0}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                teammate_controller_info(team_info(**changes))

    def test_away_evaluation_assigns_both_teammates_to_visitor_slots(self):
        data = TeammateFixture().data
        self.assertEqual(configure_teammates(data, side=2), (8, 6))
        info = data.lookup_all()
        self.assertEqual((info['defense_team1'], info['defense_team2']), (2, 2))
        self.assertEqual(teammate_controller_info(info, 1, side=2)['p2_control_slot'], 8)
        self.assertEqual(teammate_controller_info(info, 2, side=2)['p2_control_slot'], 6)
        self.assertEqual(teammate_controller_info(info, 2, side=2)['p1_control_slot'], -1)

    def test_away_coordinator_uses_physical_visitor_identity_without_mirroring_inputs(self):
        ai = TeammateCoordinator(team_args(side='away'))
        info = team_info(defense_team1=2, defense_team2=2, defense_control1=6,
                         defense_control2=7, puck_owner=6)
        with patch('nhl94_ai.agents.teammates.ClassicAIV1Model.predict_frame',
                   return_value=(np.zeros((1, 12), dtype=np.int8), None)):
            result = ai.act(info)
        self.assertEqual([s.team1.controller for s in ai.states], [2, 2])
        self.assertEqual([s.team1.controlled_scnum() for s in ai.states], [6, 7])
        self.assertEqual([r['role'] for r in result.diagnostics['teammates']['controllers']],
                         ['carrier', 'passing-option'])


class TeammateEnvironmentTests(unittest.TestCase):
    def environment(self):
        env = NHL94Observation2PEnv(TeammateFixture(), team_args(), 2, 'PostPlay')
        self.addCleanup(env.close)
        env.reset()
        return env

    def test_independent_buttons_and_reset_histories(self):
        env = self.environment()
        action = np.zeros(24, dtype=np.int8)
        action[[0, 7, 12+6, 12+8]] = 1
        expected = action.copy()
        env.step(action)
        np.testing.assert_array_equal(env.env.last_action, expected)
        np.testing.assert_array_equal(action, expected)
        np.testing.assert_array_equal(env.learner_action_state['last_env_action'], expected[:12])
        np.testing.assert_array_equal(env.opponent_action_state['last_env_action'], expected[12:])
        env.reset()
        self.assertFalse(env.learner_action_state['last_env_action'].any())
        self.assertFalse(env.opponent_action_state['last_env_action'].any())

    def test_bad_actions_do_not_advance_emulator(self):
        env = self.environment()
        before = env.env.steps
        for action in (np.zeros(12), np.zeros((2, 12)), np.full(24, 2), np.full(24, np.nan)):
            with self.subTest(shape=action.shape), self.assertRaisesRegex(ValueError, '24 binary buttons'):
                env.step(action)
            self.assertEqual(env.env.steps, before)

    def test_start_reaches_native_menu_without_enabling_other_unfiltered_buttons(self):
        env = self.environment()
        action = np.zeros(24, dtype=np.int8)
        action[[1, 2, 3, 12+3]] = 1
        env.step(action)
        self.assertEqual(env.env.last_action[3], 1)
        self.assertEqual(env.env.last_action[12+3], 1)
        self.assertFalse(env.env.last_action[[1, 2]].any())

    def test_match_continues_between_periods_and_finishes_regulation(self):
        env = self.environment()
        for period, expected in ((0, False), (1, False), (2, True)):
            env.env.data.memory.assign(0xFFC466, '>u2', period)
            if period:
                # PeriodOver increments the period before ResetClock. That
                # transient zero is not the end of the new period.
                _, _, terminated, _, _ = env.step(np.zeros(24, dtype=np.int8))
                self.assertFalse(terminated)
            env.env.data.memory.assign(0xFFC468, '>u2', 300)
            env.step(np.zeros(24, dtype=np.int8))
            env.env.data.memory.assign(0xFFC468, '>u2', 0)
            _, _, terminated, _, info = env.step(np.zeros(24, dtype=np.int8))
            self.assertEqual(terminated, expected)
            self.assertEqual(info['time'], 0)
            self.assertEqual(env.game_state.period, period + 1)


class TeammateCoordinationTests(unittest.TestCase):
    def test_new_pass_reserves_receiving_input_on_the_release_frame_for_either_carrier(self):
        for carrier in (0, 1):
            ai = TeammateCoordinator(team_args())
            def request(_state, executor=ai.controllers[carrier], sender=carrier, **_kwargs):
                executor.offense.pass_action.pending = PassState(sender, 1 - sender, 1, 50)
                action = np.zeros((1, 12), dtype=np.int8)
                action[0, 0] = 1
                return action
            with patch.object(ai.controllers[carrier], 'predict_frame', side_effect=request):
                result = ai.act(team_info(puck_owner=carrier))
            receiver = 1 - carrier
            self.assertEqual(result.diagnostics['teammates']['controllers'][receiver]['role'], 'receive')
            self.assertFalse(result.action[receiver * 12:(receiver + 1) * 12].any())
            self.assertEqual(ai.pending_pass['receiver_controller'], receiver + 1)

    def test_role_handoff_keeps_valid_pass_and_observes_both_controllers(self):
        ai = TeammateCoordinator(team_args())
        ai.act(team_info())
        executors = list(ai.controllers)
        request = PassState(0, 2, ai.frames, ai.frames + 45, passes_before=0)
        ai.controllers[0].offense.pass_action.pending = request
        ai.primary, ai.primary_until = 1, 100
        ai.act(team_info(puck_owner=-1, pass_target=2, p1_pass_attempts=1))
        self.assertEqual(ai.controllers, executors)
        self.assertIs(ai.controllers[0].offense.pass_action.pending, request)
        self.assertTrue(request.launched)
        self.assertEqual([c.scheduler.frames for c in ai.controllers], [2, 2])
        ai.act(team_info(puck_owner=7, pass_target=2, p1_pass_attempts=1))
        self.assertIsNone(ai.controllers[0].offense.pass_action.pending)
        self.assertEqual(ai.controllers[0].offense.pass_action.last_pass['outcome'], 'intercepted')

    def test_support_takeover_uses_applied_edges_and_fresh_decision(self):
        ai = TeammateCoordinator(team_args())
        for _ in range(5):
            ai.act(team_info())
        support = ai.controllers[1]
        self.assertEqual(support.scheduler.frames, 5)
        self.assertEqual(support.scheduler.decisions, 0)
        self.assertFalse(support.buttons.b_down or support.buttons.c_down)
        result = ai.act(team_info(puck_owner=1))
        self.assertIs(ai.controllers[1], support)
        self.assertGreater(support.scheduler.decisions, 0)
        self.assertEqual(support.buttons.b_down, bool(result.action[12]))
        self.assertEqual(support.buttons.c_down, bool(result.action[20]))

    def test_shot_follow_through_survives_primary_role_change(self):
        ai = TeammateCoordinator(team_args())
        ai.act(team_info())
        executor = ai.controllers[0]
        executor.shot.start(ai.states[0], executor.scheduler.decisions, executor.scheduler.frames, 4)
        ai.primary, ai.primary_until = 1, 100
        ai.act(team_info(puck_owner=-1))
        self.assertIs(ai.controllers[0], executor)
        self.assertTrue(executor.shot.active(executor.scheduler.decisions))

    def test_carrier_and_support_share_the_frame_and_have_separate_state(self):
        ai = TeammateCoordinator(team_args())
        info = team_info()
        original = deepcopy(info)
        with patch('nhl94_ai.agents.teammates.ClassicAIV1Model.predict_frame',
                   return_value=(np.zeros((1, 12), dtype=np.int8), None)) as classic:
            result = ai.act(info)
        self.assertEqual(classic.call_count, 1)
        self.assertEqual(result.action.shape, (24,))
        self.assertEqual([row['role'] for row in result.diagnostics['teammates']['controllers']],
                         ['carrier', 'passing-option'])
        self.assertEqual(info, original)
        self.assertIsNot(ai.controllers[0], ai.controllers[1])
        self.assertEqual([s.team1.controlled_scnum() for s in ai.states], [0, 1])

    def test_no_selection_stays_neutral_and_reset_clears_pending_execution(self):
        ai = TeammateCoordinator(team_args())
        result = ai.act(team_info(defense_control1=-256, defense_control2=-1, puck_owner=-1))
        self.assertFalse(result.action.any())
        ai.pending_pass = {'source': 0, 'receiver': 1, 'until': 100}
        ai.previous[:] = 1
        ai.reset()
        self.assertEqual(ai.frames, 0)
        self.assertIsNone(ai.pending_pass)
        self.assertFalse(ai.previous.any())

    def test_negative_faceoff_selections_remain_neutral_until_native_assignment(self):
        ai = TeammateCoordinator(team_args())
        info = team_info(defense_control1=-256, defense_control2=-1, goalie_clock_flags=1)
        self.assertFalse(ai.act(info).action.any())
        self.assertFalse(any(controller.buttons.b_down for controller in ai.controllers))
        self.assertTrue(ai.act({**info, 'defense_control1': 0}).action[0])

    def test_support_respects_both_attacking_directions_and_offside_line(self):
        ai = TeammateCoordinator(team_args())
        for period, sign in ((0, 1), (1, -1)):
            state = NHL94GameState(5)
            state.BeginFrame(teammate_controller_info(team_info(period=period)), [0] * 6)
            state.team1.players[0].y = 50 * sign
            state.engine.offsides_enabled = True
            target, role = ai._support_target(state)
            self.assertEqual(role, 'passing-option')
            self.assertLess(target[1] * sign, 88)

    def test_support_options_are_reachable_spaced_and_recover_goal_side(self):
        state = NHL94GameState(5)
        state.BeginFrame(teammate_controller_info(team_info()), [0] * 6)
        carrier, support = state.team1.players[:2]
        carrier.x, carrier.y = 40, 50
        support.x, support.y = -45, 60
        state.puck.x, state.puck.y = 40, 50
        planner = SupportPlanner()
        target, role = planner.choose(state, 1, 1, 0)
        self.assertEqual(role, 'passing-option')
        self.assertLess(target[1], 88)
        for candidate in planner.diagnostics['candidates']:
            self.assertLessEqual(candidate['arrival'], 48)
            self.assertGreaterEqual(np.linalg.norm(np.subtract(candidate['point'], (40, 50))), 30)
        state.engine.puck_owner = 7
        target, role = planner.choose(state, 1, 2, 0)
        self.assertEqual(role, 'coverage')
        self.assertLess(target[1], state.puck.y)
        self.assertGreaterEqual(np.linalg.norm(np.subtract(target, (state.puck.x, state.puck.y))), 30)
        self.assertLessEqual(abs(target[0]), 70)
        planner.choose(state, 1, 3, None)  # Both pursuit candidates may be animation-locked.

    def test_reception_metrics_follow_receiver_shots_and_exclude_other_shooters(self):
        for shooter in (1, 2):
            state = NHL94GameState(5)
            state.BeginFrame(teammate_controller_info(team_info()), [0] * 6)
            metrics = ReceptionOutcomes()
            metrics.observe(1, state, [0, 1])
            state.engine.puck_owner, state.engine.pass_target = -1, 1
            state.team1.pass_attempts = 1
            metrics.observe(2, state, [0, 1])
            state.engine.puck_owner = 1
            metrics.observe(3, state, [0, 1])
            state.engine.puck_owner, state.engine.shot_player = -1, shooter
            state.team1.stats.shots += 1
            metrics.observe(4, state, [0, 1])
            metrics.finish(5, 'period-ended')
            summary = metrics.summary()
            self.assertEqual(summary['controlled_pair_receptions'], 1)
            self.assertEqual(summary['reception_to_shots'], int(shooter == 1))

    def test_reception_metrics_wait_for_delayed_shot_counter_and_track_long_carries(self):
        state = NHL94GameState(5)
        state.BeginFrame(teammate_controller_info(team_info()), [0] * 6)
        metrics = ReceptionOutcomes()
        metrics.observe(1, state, [0, 1])
        state.engine.puck_owner, state.engine.pass_target = -1, 1
        state.team1.pass_attempts = 1
        metrics.observe(2, state, [0, 1])
        state.engine.puck_owner = 1
        metrics.observe(3, state, [0, 1])
        metrics.observe(150, state, [0, 1])
        state.engine.puck_owner, state.engine.shot_player = -1, 1
        metrics.observe(151, state, [0, 1])
        self.assertIsNotNone(metrics.pending)
        state.team1.stats.shots += 1
        metrics.observe(155, state, [0, 1])
        metrics.finish(160, 'period-ended')
        self.assertEqual(metrics.summary()['controlled_pair_reception_to_shots'], 1)

    def test_native_normal_release_is_attributed_before_recorded_shot_or_turnover(self):
        state = NHL94GameState(5)
        state.BeginFrame(teammate_controller_info(team_info()), [0] * 6)
        metrics = ReceptionOutcomes()
        metrics.observe(1, state, [0, 1])
        state.engine.puck_owner, state.engine.pass_target = -1, 1
        state.team1.pass_attempts = 1
        metrics.observe(2, state, [0, 1])
        state.engine.puck_owner = 1
        shooter = state.team1.players[1]
        shooter.live_anim, shooter.live_anim_frame = 0x7FC, 24
        state.puck.motion_x = state.puck.motion_y = 0
        metrics.observe(3, state, [0, 1])
        shooter.live_anim_frame = 28
        state.engine.puck_owner, state.engine.shot_player = -1, 1
        state.puck.motion_y = 4
        metrics.observe(4, state, [0, 1])
        state.engine.puck_owner = 7
        metrics.observe(5, state, [0, 1])
        self.assertEqual(metrics.summary()['controlled_pair_reception_to_shots'], 1)
        self.assertEqual(metrics.events[-1]['ending'], 'shot-recovery')
        self.assertEqual(metrics.events[-1]['shot_evidence'], 'native-animation-release-and-impulse')

    def test_receiver_selection_loss_and_expiry_end_the_shared_commitment(self):
        for reason in ('receiver-selection-lost', 'unresolved-release'):
            state = NHL94GameState(5)
            state.BeginFrame(teammate_controller_info(team_info()), [0] * 6)
            shared = SharedReceiving()
            shared.observe(state, [0, 1], 1)
            shared.propose({'passer': 0, 'receiver': 1, 'frame': 1, 'deadline': 9,
                            'purpose': 'one-timer'}, 1, [0, 1], state, 1)
            shared.observe(state, [0, 2] if reason == 'receiver-selection-lost' else [0, 1],
                           2 if reason == 'receiver-selection-lost' else 9)
            self.assertIsNone(shared.pending)
            self.assertEqual(shared.events[-1]['reason'], reason)

    def test_ordinary_native_pass_reserves_receiver_without_speculative_c_edges(self):
        ai = TeammateCoordinator(team_args())
        with patch('nhl94_ai.agents.teammates.ClassicAIV1Model.predict_frame',
                   return_value=(np.zeros((1, 12), dtype=np.int8), None)):
            ai.act(team_info(p1_2_y=210, puck_y=205, p1_2_x=30, puck_x=30))
            info = team_info(puck_owner=-1, p1_pass_attempts=1, p1_2_y=210,
                             puck_y=205, p1_2_x=30, puck_x=30)
            result = ai.act(info)
            self.assertEqual(result.diagnostics['teammates']['controllers'][1]['role'], 'receive')
            self.assertEqual(result.action[12+8], 0)
            result = ai.act(info)
            self.assertEqual(result.action[12+8], 0)
            ai.act({**info, 'puck_owner': 1})
            self.assertEqual(ai.pending_pass['phase'], 'received')

    def test_shared_cue_requires_confirmed_release_and_fires_once_at_both_ends(self):
        for period in (0, 1):
            state = NHL94GameState(5)
            state.BeginFrame(teammate_controller_info(team_info(period=period)), [0] * 6)
            sign = 1 if state.team2.net.y > state.team1.net.y else -1
            receiver = state.team1.players[1]
            receiver.x, receiver.y = 30, 210 * sign
            shared = SharedReceiving()
            shared.observe(state, [0, 1], 1)
            shared.propose({'passer': 0, 'receiver': 1, 'point': (30, 210 * sign),
                            'frame': 1, 'deadline': 61, 'purpose': 'one-timer'}, 1, [0, 1], state, 1)
            before = shared.receiver_action(state, 2, 1, np.zeros(12))[0]
            self.assertFalse(before[8])
            state.engine.puck_owner, state.engine.pass_target = -1, 1
            state.team1.pass_attempts = 1
            shared.observe(state, [0, 1], 2)
            with patch('nhl94_ai.agents.teammate_receiving.one_timer_reception',
                       return_value={'gap': -3, 'frames': 8}), \
                 patch('nhl94_ai.agents.teammate_receiving.shot_value', return_value=30):
                first = shared.receiver_action(state, 2, 2, np.zeros(12))[0]
                second = shared.receiver_action(state, 2, 3, first)[0]
                third = shared.receiver_action(state, 2, 4, second)[0]
            self.assertEqual(first[8], 1)
            self.assertFalse(second[8] or third[8])
            self.assertFalse(first[0] or second[0] or third[0])

    def test_shared_intention_cancels_on_interception_without_fabricating_release(self):
        state = NHL94GameState(5)
        state.BeginFrame(teammate_controller_info(team_info()), [0] * 6)
        shared = SharedReceiving()
        shared.observe(state, [0, 1], 1)
        shared.propose({'passer': 0, 'receiver': 1, 'frame': 1, 'deadline': 61,
                        'purpose': 'one-timer'}, 1, [0, 1], state, 1)
        state.engine.puck_owner = 7
        shared.observe(state, [0, 1], 2)
        self.assertIsNone(shared.pending)
        self.assertEqual(shared.events[-1]['reason'], 'lost-before-release')
        self.assertFalse(any(e['phase'] == 'released' for e in shared.events))

    def test_a_switch_without_native_pass_counter_change_is_not_a_pass(self):
        ai = TeammateCoordinator(team_args())
        ai.last_owner, ai.last_pass_count = -1, 0
        ai.previous[0, 0] = 1
        ai.act(team_info(puck_owner=-1))
        self.assertIsNone(ai.pending_pass)

    def test_intermission_advances_only_controller_one_and_releases_start(self):
        ai = TeammateCoordinator(team_args())
        info = team_info(period=1, teammate_clock=300, sflags=0x0100)
        first = ai.act(info).action
        second = ai.act(info).action
        self.assertEqual(first[3], 1)
        self.assertFalse(first[12:].any())
        self.assertFalse(second.any())

    def test_highlight_teams_are_not_played_or_counted_as_our_controllers(self):
        ai = TeammateCoordinator(team_args())
        info = team_info(defense_team1=0, defense_team2=0, goalie_clock_flags=0x10,
                         period=2, teammate_clock=154)
        with patch('nhl94_ai.agents.teammates.ClassicAIV1Model.predict_frame') as classic:
            result = ai.act(info)
            classic.assert_not_called()
        self.assertEqual(result.action[3], 1)
        self.assertEqual(result.action.sum(), 1)
        self.assertTrue(all(row['slot'] == -1 for row in result.diagnostics['teammates']['controllers']))
        self.assertIsNone(ai.pending_pass)


@unittest.skipUnless(importlib.util.find_spec('pygame'), 'optional display extra is not installed')
class TeammateDisplayTests(unittest.TestCase):
    def test_two_controller_actions_and_role_diagnostics_reach_display(self):
        from stable_baselines3.common.vec_env import DummyVecEnv
        from nhl94_ai.ui.debug import NHL94DebugDisplay
        with patch.dict(os.environ, {'SDL_VIDEODRIVER': 'dummy', 'SDL_AUDIODRIVER': 'dummy'}):
            inner = NHL94Observation2PEnv(TeammateFixture(), team_args(), 2, 'PostPlay')
            vector = DummyVecEnv([lambda: inner])
            display = NHL94DebugDisplay(vector, team_args(), 0, 'ClassicAIV1', [])
            try:
                vector.reset()
                ai = TeammateCoordinator(team_args())
                action = ai.act(vector.reset_infos[0]).action
                display.set_ai_sys_info(ai)
                with patch.object(display, 'draw_frame'), patch.object(display, 'process_events'):
                    display.step([action])
                np.testing.assert_array_equal(inner.env.last_action, action)
                self.assertEqual(len(display.teammates['controllers']), 2)
            finally:
                display.close()


if __name__ == '__main__':
    unittest.main()

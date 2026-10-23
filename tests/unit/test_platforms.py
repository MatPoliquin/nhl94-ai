"""Console decoding and control boundaries without importing a ROM."""
from copy import deepcopy
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from nhl94_ai.agents.classic_v1 import ClassicAIV1Model
from nhl94_ai.agents.motion import skating, velocity
from nhl94_ai.agents.goalie import GoalieController, goalie_arrival
from nhl94_ai.agents.skating import grounded_step
from nhl94_ai.config import EnvironmentConfig
from nhl94_ai.game.platforms import get_backend
from nhl94_ai.game.platforms.base import HockeyCommand
from nhl94_ai.game.platforms.snes import PLAYER_PREFIXES
from nhl94_ai.game.specs import get_game
from nhl94_ai.game.state import NHL94GameState


def snes_info(owner=-2, selected=2):
    info = dict(time=300, period=0, puck_x=0, puck_y=0, puck_z=-1,
                puck_vel_x=0, puck_vel_y=0, puck_live_vel_x=8192, puck_live_vel_y=-4096,
                puck_owner=owner, shot_player=3)
    for side in (1, 2):
        for field in ('score', 'shots', 'bodychecks', 'attackzone', 'faceoffwon', 'passing', 'onetimer'):
            info[f'p{side}_{field}'] = 0
    for slot, prefix in enumerate(PLAYER_PREFIXES):
        info.update({prefix + 'x': (slot % 6 - 2) * 20,
                     prefix + 'y': -100 if slot < 6 else 100,
                     prefix + 'vel_x': 8, prefix + 'vel_y': 0,
                     prefix + 'live_vel_x': 8192, prefix + 'live_vel_y': 0,
                     prefix + 'ori': 0, prefix + 'anim': 0xE0FE,
                     prefix + 'anim_frame': 0,
                     prefix + 'state_flags': 8 if slot == selected else 0})
    return info


def snes_args(**kwargs):
    values = dict(env='NHL94-Snes-v0', nn='ClassicAIV1', rf='PostPlay', num_players=1,
                  action_type='FILTERED', mode='model_vs_game', goalie_policy='off', seed=None)
    return SimpleNamespace(**{**values, **kwargs})


def snes_controls(info, teams=(1, 0)):
    """Native joystick tables, including the three unused multitap entries."""
    for joystick, team in enumerate((*teams, 0, 0, 0)):
        info[f'snes_controller_assignment_{joystick}'] = team
        info[f'snes_controller_team_{joystick}'] = (team - 1) * 2
        info[f'snes_controller_player_{joystick}'] = ((team - 1) * 6 + 2) * 2 if team else -1
    info.update(snes_goalie_manual_0=1, snes_goalie_manual_1=1,
                snes_input_block=0, snes_game_flags=0)
    return info


class ConsoleDecodingTests(unittest.TestCase):
    def test_owner_offsets_and_shooter_ids_have_distinct_semantics(self):
        backend = get_backend('snes')
        for native, logical in ((-2, -1), (-1, -1), (0, 0), (2, 1), (10, 5), (12, 6), (22, 11)):
            with self.subTest(native=native):
                decoded = backend.decode_info(snes_info(owner=native))
                self.assertEqual(decoded['puck_owner'], logical)
                self.assertEqual(decoded['native_puck_owner'], native)
                self.assertEqual(decoded['shot_player'], 3)
        for native in (1, 23, 24):
            with self.assertRaisesRegex(ValueError, 'byte offset'):
                backend.decode_info(snes_info(owner=native))

    def test_decoding_does_not_modify_native_info_or_decode_twice(self):
        backend = get_backend('snes')
        original = snes_info(owner=2)
        saved = deepcopy(original)
        decoded = backend.decode_info(original)
        self.assertEqual(original, saved)
        self.assertIs(backend.decode_info(decoded), decoded)

    def test_selection_is_independent_of_ownership_and_animation(self):
        state = NHL94GameState(5, platform=get_backend('snes'))
        state.BeginFrame(snes_info(owner=2, selected=3), [0] * 6)
        self.assertEqual(state.team1.controlled_scnum(), 3)
        self.assertEqual(state.team1.get_possession_player(), state.team1.players[1])
        self.assertEqual(state.team1.players[1].has_puck, 1)
        selected = state.team1.players[3]
        self.assertFalse(selected.has_puck)
        self.assertFalse(selected.is_one_timer)
        self.assertFalse(selected.is_pad_stack)
        self.assertEqual(selected.native_state_flags, 8)
        self.assertEqual(selected.facing, 0)
        self.assertAlmostEqual(selected.ori_x, 0)
        self.assertAlmostEqual(selected.ori_y, 1)

    def test_loose_puck_and_no_selection_never_invent_a_carrier(self):
        state = NHL94GameState(5, platform=get_backend('snes'))
        state.BeginFrame(snes_info(owner=2, selected=1), [0] * 6)
        state.BeginFrame(snes_info(owner=-1, selected=-1), [0] * 6)
        self.assertEqual(state.team1.control, -1)
        self.assertFalse(state.team1.player_haspuck)
        self.assertFalse(any(player.has_puck for player in state.team1.players))
        state.BeginFrame(snes_info(owner=None, selected=-1), [0] * 6)
        self.assertFalse(state.engine.puck_owner_known)

    def test_goalie_identity_and_period_end_changes(self):
        state = NHL94GameState(5, platform=get_backend('snes'))
        info = snes_info(owner=10, selected=5)
        state.BeginFrame(info, [0] * 6)
        self.assertTrue(state.team1.goalie_haspuck)
        self.assertEqual(state.team1.get_controlled_player(), state.team1.goalie)
        self.assertEqual(state.team1.net.y, -264)
        info['period'] = 1
        state.BeginFrame(info, [0] * 6)
        self.assertEqual(state.period, 2)
        self.assertEqual(state.team1.net.y, 264)

    def test_full_signed_velocity_uses_snes_units(self):
        state = NHL94GameState(5, platform=get_backend('snes'))
        state.BeginFrame(snes_info(), [0] * 6)
        self.assertEqual(velocity(state.team1.players[0]), (2, 0))
        self.assertEqual(velocity(state.puck), (2, -1))
        self.assertEqual(state.puck.height, 0)
        self.assertIsNone(state.team1.players[0].speed)
        self.assertIsNone(state.team1.players[0].weight)
        self.assertGreater(skating(state.team1.players[0])[0], 0)
        future = grounded_step(state.team1.players[0], (0, 1))
        self.assertGreater(future.x, state.team1.players[0].x)
        self.assertEqual(future.platform, 'snes')

    def test_rom_fixed_point_motion_and_physics_cadence_decode(self):
        info = snes_info()
        info.update(p1_3_x=-20, p1_3_x_frac=32768,
                    p1_3_y=100, p1_3_y_frac=49152,
                    p1_3_live_vel_x=-1000, p1_3_live_vel_y=250,
                    puck_x=3, puck_x_frac=49152, puck_y=-4, puck_y_frac=16384,
                    puck_z=2, puck_z_frac=32768, puck_live_vel_z=-128,
                    p1_3_native_facing_phase=-16384, p1_3_motion_weight=48,
                    p1_3_motion_agility=15, p1_3_speed_rating=9,
                    p1_3_turning_state=0, p1_3_carrier_state=4,
                    p1_3_motion_flags=8, p1_3_input_lock_x=0, p1_3_input_lock_y=0,
                    snes_frame_clock=10, snes_physics_clock=9, snes_frame_step=2)
        state = NHL94GameState(5, platform=get_backend('snes'))
        state.BeginFrame(info, [0] * 6)
        player = state.team1.players[2]
        self.assertEqual(player.position_frac_x, 32768)
        self.assertEqual(player.position_frac_y, 49152)
        self.assertAlmostEqual(player.precise_x, -19.5)
        self.assertAlmostEqual(player.precise_y, 100.75)
        self.assertEqual(player.raw_vel_x, -1000)
        self.assertEqual(player.raw_vel_y, 250)
        self.assertEqual(player.physics_frame_debt, 1)
        self.assertEqual(player.physics_step_frames, 2)
        self.assertEqual(player.native_facing_phase, -16384)
        self.assertEqual(player.facing_phase, -0.25)
        self.assertEqual(player.motion_weight, 48)
        self.assertEqual(player.motion_agility, 15)
        self.assertEqual(player.speed_rating, 9)
        self.assertEqual(player.turning_state, 0)
        self.assertEqual(player.carrier_state, 4)
        self.assertEqual(player.motion_flags, 8)
        self.assertEqual(player.input_lock_x, 0)
        self.assertEqual(player.input_lock_y, 0)
        self.assertEqual(state.engine.physics_frame_debt, 1)
        self.assertAlmostEqual(state.puck.precise_x, 3.75)
        self.assertAlmostEqual(state.puck.precise_y, -3.75)
        self.assertAlmostEqual(state.puck.height, 2.5)
        self.assertAlmostEqual(state.puck.motion_z, -128 / 4096)

    def test_profiles_and_rink_geometry_do_not_leak_between_instances(self):
        genesis = NHL94GameState(5)
        snes = NHL94GameState(5, platform=get_backend('snes'))
        self.assertEqual(genesis.team1.net.depth, 10)
        self.assertEqual(snes.team1.net.depth, 8)
        self.assertEqual(genesis.rink.puck_y, 270)
        self.assertEqual(snes.rink.puck_y, 300)

    def test_multiple_selected_players_require_an_explicit_controller_map(self):
        info = snes_info()
        info['p1_state_flags'] = 8
        with self.assertRaisesRegex(ValueError, 'multiple selected'):
            get_backend('snes').decode_info(info)
        # The native player->joystick map resolves the transient double flag.
        info['snes_player_controller_2'] = 0
        result = get_backend('snes').decode_info(info)
        self.assertEqual(result['p1_control_slot'], 2)


class ConsoleControlTests(unittest.TestCase):
    def test_hockey_commands_map_to_each_consoles_shot_and_clear_buttons(self):
        action = np.zeros(12, dtype=np.int8)
        action[8] = 1
        action[1] = 1
        command = HockeyCommand.from_buttons(action)
        for platform, expected in (('genesis', {'C', 'A'}), ('snes', {'A', 'Y'})):
            backend = get_backend(platform)
            native = command.encode(backend.buttons, backend.control_mapping)
            self.assertEqual({name for name, pressed in zip(backend.buttons, native) if pressed}, expected)

    def test_genesis_preserves_every_existing_button(self):
        for index in range(12):
            action = np.eye(12, dtype=np.int8)[index]
            np.testing.assert_array_equal(get_backend().encode_action(action), action)

    def test_both_consoles_use_the_same_classic_controller(self):
        state = NHL94GameState(5, platform=get_backend('snes'))
        state.BeginFrame(snes_info(), [0] * 6)
        controller = ClassicAIV1Model(args=snes_args())
        action = controller.predict_frame(state)[0]
        self.assertEqual(action.shape, (12,))
        self.assertEqual(controller._last_decision, 'faceoff')

    def test_watched_playback_uses_the_selected_console_state(self):
        from nhl94_ai.agents.multi_model import NHL94AISystem
        system = NHL94AISystem(snes_args(), None, None)
        system.game_state.BeginFrame(snes_info(owner=2), [0] * 6)
        self.assertEqual(system.game_state.platform.name, 'snes')
        self.assertEqual(system.game_state.engine.puck_owner, 1)
        self.assertEqual(velocity(system.game_state.puck), (2, -1))

    def test_faceoff_cancels_actions_committed_before_the_stoppage(self):
        state = NHL94GameState(5, platform=get_backend('snes'))
        state.BeginFrame(snes_info(), [0] * 6)
        controller = ClassicAIV1Model(args=snes_args())
        controller.shot.until_decision = 100
        controller.offense.pass_action.pending = object()
        controller.setup.until_decision = 50
        controller.predict_frame(state)
        self.assertIsNone(controller.offense.pass_action.pending)
        self.assertEqual(controller.shot.until_decision, 0)
        self.assertEqual(controller.setup.until_decision, 0)

    def test_snes_normalization_and_native_flags_are_platform_specific(self):
        from nhl94_ai.env.encoding import NET_FIELD_GETTERS
        from nhl94_ai.tasks.registry import resolve_task_model_input
        state = NHL94GameState(5, platform=get_backend('snes'))
        info = snes_info(owner=-1)
        info.update(puck_y=300, sflags=0xFFFF, sflags2=0xFFFF, ba_ps_flags=0xFFFF)
        state.BeginFrame(info, [0] * 6)
        self.assertEqual(state.nz_puck.y, 1)
        self.assertFalse(state.is_shootout_active)
        self.assertFalse(state.engine.shot_mode_active)
        self.assertAlmostEqual(NET_FIELD_GETTERS['rel_puck_y'](state, None), -36 / 300)
        params = resolve_task_model_input(snes_args(), {})
        self.assertEqual(params['model_input']['normalization'], 'nhl94-snes-game-constants-v1')

    def test_snes_telemetry_registration_never_uses_genesis_ram(self):
        variables = {}
        data = SimpleNamespace(set_variable=lambda name, definition: variables.update({name: definition}))
        get_backend('snes').register(SimpleNamespace(data=data), 5)
        self.assertEqual(variables['snes_controller_assignment_1'],
                         {'address': 0x7E1C86, 'type': '<i2'})
        self.assertEqual(variables['snes_controller_player_1'],
                         {'address': 0x7E19EE, 'type': '<i2'})
        self.assertEqual(variables['snes_goalie_manual_1'],
                         {'address': 0x7E1C92, 'type': '<i2'})
        self.assertEqual(variables['p1_3_x_frac'],
                         {'address': 0x7E0D7B, 'type': '<u2'})
        self.assertEqual(variables['p1_3_y_frac'],
                         {'address': 0x7E0DDD, 'type': '<u2'})
        self.assertEqual(variables['p1_3_native_facing_phase'],
                         {'address': 0x7E1187, 'type': '<i2'})
        self.assertEqual(variables['p1_3_motion_weight'],
                         {'address': 0x7E12C7, 'type': '<i2'})
        self.assertEqual(variables['p1_3_motion_agility'],
                         {'address': 0x7E12E7, 'type': '<i2'})
        self.assertEqual(variables['p1_3_speed_rating'],
                         {'address': 0x7E1307, 'type': '<i2'})
        self.assertEqual(variables['p1_3_turning_state'],
                         {'address': 0x7E1527, 'type': '<i2'})
        self.assertEqual(variables['snes_motion_puck_owner'],
                         {'address': 0x7E0AD9, 'type': '<i2'})
        self.assertEqual(variables['snes_motion_mode'],
                         {'address': 0x7E0D07, 'type': '<i2'})
        self.assertEqual(variables['puck_x_frac'],
                         {'address': 0x7E0D93, 'type': '<u2'})
        self.assertEqual(variables['puck_z_frac'],
                         {'address': 0x7E0DFD, 'type': '<u2'})
        self.assertEqual(variables['puck_live_vel_z'],
                         {'address': 0x7E0E01, 'type': '<i2'})
        self.assertEqual(variables['snes_frame_clock'],
                         {'address': 0x7E0083, 'type': '<u2'})
        self.assertEqual(variables['snes_physics_clock'],
                         {'address': 0x7E0D01, 'type': '<u2'})
        self.assertEqual(variables['snes_frame_step'],
                         {'address': 0x7E0ADB, 'type': '<u2'})
        self.assertTrue(all(0x7E0000 <= item['address'] < 0x800000 and item['type'] in ('<i2', '<u2')
                            for item in variables.values()))

    def test_snes_only_terminates_at_the_end_of_the_countdown(self):
        backend = get_backend('snes')
        self.assertFalse(backend.episode_done(SimpleNamespace(time=9), lambda state: True))
        self.assertTrue(backend.episode_done(SimpleNamespace(time=0), lambda state: False))

    def test_unsupported_modes_fail_before_starting_emulation(self):
        EnvironmentConfig.from_args(snes_args())
        for options in (dict(rf='DefenseZone'), dict(action_type='MULTI_DISCRETE'),
                        dict(num_players=2), dict(seed=42),
                        dict(model='/tmp/genesis-policy.zip'), dict(model_input='pvg'),
                        dict(nn='CnnPolicy', agent='classic-v1')):
            with self.subTest(options=options), self.assertRaises(ValueError):
                EnvironmentConfig.from_args(snes_args(**options))
        self.assertEqual(get_game('NHL94-Snes-v0').default_state, 'DucksVsSabres.ManualGoalie.Start')


class SnesMultiplayerTests(unittest.TestCase):
    def test_reset_assigns_both_directions_and_releases_the_cpu_team(self):
        backend = get_backend('snes')
        for side, players, teams in (('home', 2, (1, 2)), ('away', 1, (2, 0)), ('away', 2, (2, 1))):
            with self.subTest(side=side, players=players):
                info = snes_controls(snes_info())
                data = SimpleNamespace(lookup_all=lambda info=info: dict(info), set_value=info.__setitem__)
                backend.configure_controllers(SimpleNamespace(data=data), side, players)
                decoded = backend.decode_info(info)
                self.assertEqual(tuple(decoded[f'defense_team{i}'] for i in (1, 2)), teams)
                for team in (1, 2):
                    self.assertEqual(info[f'snes_team_controller_count_{team - 1}'], teams.count(team))
                for joystick, team in enumerate(teams):
                    if team:
                        slot = (team - 1) * 6 + 2
                        self.assertEqual(decoded[f'p{team}_control_slot'], slot)
                        self.assertEqual(info[f'snes_player_controller_{slot}'], joystick * 2)
                        for field in ('first', 'last', 'next'):
                            self.assertEqual(info[f'snes_{field}_controller_{team - 1}'], joystick * 2)
                    else:
                        other = 3 - teams[0]
                        self.assertEqual(decoded[f'p{other}_control_slot'], -1)
                self.assertEqual(info['time'], 300)
                self.assertEqual(info['puck_owner'], -2)

    def test_home_single_controller_reset_preserves_the_save(self):
        info = snes_controls(snes_info())
        saved = deepcopy(info)
        data = SimpleNamespace(lookup_all=lambda: dict(info), set_value=info.__setitem__)
        get_backend('snes').configure_controllers(SimpleNamespace(data=data), 'home', 1)
        self.assertEqual(info, saved)

    def test_supported_modes_and_platform_specific_goalie_defaults(self):
        from nhl94_ai.evaluation.play import parse_cmdline
        from nhl94_ai.agents.registry import resolve_goalie_positioning
        for mode in ('model_vs_game', 'player_vs_model', 'model_vs_model'):
            for side in ('home', 'away'):
                args = parse_cmdline(['--env=NHL94-Snes-v0', '--nn=ClassicAIV1', '--rf=PostPlay',
                                      '--mode=' + mode, '--side=' + side])
                EnvironmentConfig.from_args(args)
                self.assertEqual(args.num_players, 1 if mode == 'model_vs_game' else 2)
                self.assertEqual(args.goalie_policy, 'selective')
                self.assertEqual(resolve_goalie_positioning(args), 'classic')
                self.assertEqual(args.state, 'DucksVsSabres.ManualGoalie.Start')
        from nhl94_ai.cli import main
        with patch('nhl94_ai.evaluation.play.run') as run:
            main(['play', '--agent=classic-v1', '--env=NHL94-Snes-v0', '--mode=model_vs_model'])
        self.assertEqual(run.call_args.args[0].num_players, 2)
        for kwargs in (dict(goalie_positioning='deep-csave'), dict(goalie_model='model.zip'),
                       dict(goalie_planner='physics'), dict(goalie_decision_interval=4)):
            with self.subTest(kwargs=kwargs), self.assertRaisesRegex(ValueError, 'SNES manual goalies'):
                EnvironmentConfig.from_args(snes_args(goalie_policy='selective', **kwargs))

    def test_scripted_match_uses_independent_native_frame_controllers(self):
        from nhl94_ai.evaluation.play import NHL94Player, parse_cmdline
        args = parse_cmdline(['--env=NHL94-Snes-v0', '--nn=ClassicAIV1', '--rf=PostPlay',
                              '--mode=model_vs_model', '--goalie-policy=off'])
        args.hyperparams_dict = {}
        player = NHL94Player.__new__(NHL94Player)
        player.args, player.need_display, player.logger, player.scripted_match = args, False, None, True
        with patch('nhl94_ai.evaluation.play.init_play_env'):
            player.init_scripted_match()
        first, second = player.ai_sys.models[1], player.opponent_ai_sys.models[1]
        self.assertIsNot(first.controller, second.controller)
        info = get_backend('snes').decode_info(snes_controls(snes_info(), (1, 2)))
        action = player._ai_actions(np.zeros((1, 310)), [info])
        self.assertEqual(action.shape, (24,))
        self.assertEqual((first.controller.scheduler.frames, second.controller.scheduler.frames), (1, 1))
        first.reset()
        self.assertEqual(second.controller.scheduler.frames, 1)

    def test_headless_human_match_keeps_the_second_controller_neutral(self):
        from nhl94_ai.evaluation.play import NHL94Player
        player = NHL94Player.__new__(NHL94Player)
        player.args, player.need_display = snes_args(mode='player_vs_model', deterministic=True), False
        action = np.eye(12, dtype=np.int8)[7]
        player.ai_sys = SimpleNamespace(predict=lambda *args, **kwargs: [action])
        combined = player._ai_actions(None, None)
        np.testing.assert_array_equal(combined[:12], action)
        self.assertFalse(combined[12:].any())
        self.assertEqual(combined.shape, (24,))


class SnesManualGoalieTests(unittest.TestCase):
    def state(self, selected=2, owner=12):
        info = snes_controls(snes_info(owner=owner, selected=selected))
        info.update(puck_x=0, puck_y=-140, g1_x=0, g1_y=-246,
                    g1_live_vel_x=0, g1_live_vel_y=0)
        state = NHL94GameState(5, platform=get_backend('snes'))
        state.BeginFrame(info, [0] * 6)
        return state

    def test_takeover_uses_x_and_confirms_selection_before_positioning(self):
        state = self.state()
        controller = GoalieController('always', platform=state.platform)
        self.assertFalse(controller.step(state).any())
        self.assertFalse(controller.step(state).any())
        action = controller.step(state)
        self.assertEqual(np.flatnonzero(action).tolist(), [9])
        selected = self.state(selected=5)
        self.assertFalse(controller.step(selected).any())
        self.assertEqual(controller.metrics['takeovers'], 1)
        self.assertEqual(controller.phase, 'release-goalie')
        self.assertIsNone(selected.team1.goalie.weight)
        self.assertTrue(np.isfinite(goalie_arrival(selected.team1.goalie, (5, -250))))

    def test_ignored_x_edge_is_retried_without_a_genesis_b_hold(self):
        state = self.state()
        controller = GoalieController('always', platform=state.platform)
        actions = [controller.step(state) for _ in range(15)]
        self.assertTrue(any(action[9] for action in actions[2:6]))
        self.assertFalse(any(action[9] for action in actions[6:10]))
        self.assertTrue(any(action[9] for action in actions[10:14]))
        self.assertFalse(any(action[0] for action in actions))

    def test_save_feedback_uses_native_snes_animation_pointers(self):
        state = self.state(selected=5)
        state.team1.goalie.live_anim = 0xE2AA
        controller = GoalieController('always', platform=state.platform)
        controller.pending_save = 'save', 0
        controller.step(state)
        self.assertEqual(controller.metrics['save_animations'], 1)
        self.assertIsNone(controller.pending_save)

    def test_clear_feedback_uses_release_animation_when_no_pass_counter_increments(self):
        state = self.state(selected=5, owner=-1)
        state.team1.goalie.live_anim = 0xE34E
        state.team1.pass_attempts = 5
        controller = GoalieController('always', platform=state.platform)
        controller.frames = 2
        controller.outlet = dict(receiver=None, passes=5, started=0, deadline=60, launched=False)
        controller._outlet_feedback(state)
        self.assertEqual(controller.metrics['clears_launched'], 1)
        self.assertEqual(state.team1.pass_attempts, 5)

    def test_return_to_skater_uses_b_and_waits_for_native_feedback(self):
        state = self.state(selected=5, owner=2)
        controller = GoalieController('always', platform=state.platform)
        self.assertEqual(np.flatnonzero(controller.step(state)).tolist(), [0])
        self.assertEqual(controller.phase, 'request-skater')
        controller.step(self.state(selected=1, owner=2))
        self.assertEqual(controller.metrics['returns'], 1)
        self.assertEqual(controller.phase, 'neutral-handoff')

    def test_manual_mode_validation_uses_physical_team_for_away_controller(self):
        state = self.state()
        state.Flip()
        state.engine.controller_teams = (2, 0)
        state.engine.goalie_modes = (1, 0)
        self.assertEqual(GoalieController.validate(state), 0)
        state.engine.goalie_modes = (0, 1)
        with self.assertRaisesRegex(ValueError, 'Manual goalie settings'):
            GoalieController.validate(state)


if __name__ == '__main__':
    unittest.main()

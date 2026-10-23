"""Explicit ROM checks: python -m tests.integration.teammates."""
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from nhl94_ai.env.factory import make_retro
from nhl94_ai.evaluation.teammates import DEFAULT_STATE, play_match, prepare_match, replay_trace
from nhl94_ai.agents.teammates import TeammateCoordinator
from nhl94_ai.game.ram import configure_teammates, register_teammate_state
from nhl94_ai.game.multitap import prepare_multitap_save
from nhl94_ai.game.motion import VELOCITY_SCALE
from nhl94_ai.agents.receiving import one_timer_reception, reception_trajectory
from nhl94_ai.tasks.cross_crease_setup import OBJECT_BASE, OBJECT_STRIDE
from nhl94_ai.tasks.defense_setup import (
    init_full_team_defense, _reset_player, _place_object, _rebuild_object_order,
)


def selected_pair_fixture(side, *, one_timer=False, receiver_facing=None, team_size=2,
                          pair=(0, 1), lifecycle=None, details=False, receiver_motion=(0, 0),
                          coast_after_release=False, frames=200):
    """Coherent reset only; all defenders respond natively throughout gameplay."""
    env = make_retro(game='NHL94-Genesis-v0', state='PenguinsVsSenators.DefenseZone',
                     num_players=team_size, allow_menu_buttons=True)
    try:
        env.reset(seed=94)
        register_teammate_state(env)
        init_full_team_defense(env)
        if team_size == 3:
            prepare_multitap_save(env)
        selected = configure_teammates(env.data, side, team_size=team_size)
        carrier, receiver = (selected[i] for i in pair)
        memory, sign = env.data.memory, 1 if side == 1 else -1
        start, finish = ((-50, 190), (35, 210)) if one_timer else ((0, -100), (-70, -25))
        friends = [slot for slot in range((side - 1) * 6, (side - 1) * 6 + 5)
                   if slot not in (carrier, receiver)]
        opponents = list(range((2 - side) * 6, (2 - side) * 6 + 5))
        positions = {carrier: start, receiver: finish,
                     **dict(zip(friends, ((-100, -120), (100, -150), (0, -200)))),
                     **dict(zip(opponents, ((0, 100), (85, 100), (-85, 120), (110, 200), (-110, 210))))}
        for slot, point in positions.items():
            role = memory.extract(OBJECT_BASE + slot * OBJECT_STRIDE + 0x34, '>i2')
            assignment = 1 if role in (1, 2) else 6 if role == 4 else 4
            temporary = (0x10, 0x11) if slot == carrier else (0x11,) if slot == opponents[0] else ()
            facing = 0 if side == 1 else 4
            if slot == receiver and receiver_facing is not None:
                facing = receiver_facing
            _reset_player(memory, slot, (point[0], point[1] * sign), facing,
                          assignment, slot in selected, temporary)
            memory.assign(OBJECT_BASE + slot * OBJECT_STRIDE + 0x71, '|u1', 30)
        for slot, point in ((5, (0, -250)), (11, (0, 250))):
            _reset_player(memory, slot, point, 0 if slot == 5 else 4, 0x0E, False)
        for address, value in ((0xFFB7AA, carrier), (0xFFBEDA, carrier), (0xFFBEE0, -1), (0xFFBED8, -1)):
            memory.assign(address, '>i2', value)
        memory.assign(OBJECT_BASE + carrier * OBJECT_STRIDE + 0x6E, '|u1', 20)
        for offset, component in zip((0x28, 0x2A), receiver_motion):
            memory.assign(OBJECT_BASE + receiver * OBJECT_STRIDE + offset, '>i2',
                          round(component / VELOCITY_SCALE))
        _place_object(memory, 14, (start[0], (start[1] + 8) * sign))
        _rebuild_object_order(memory)
        memory.assign(0xFFC2EC, '>u2', memory.extract(0xFFC2EC, '>u2') & ~0x0C00)
        memory.assign(0xFFC2EE, '>u2', memory.extract(0xFFC2EE, '>u2') & ~0x1000)
        env.data.update_ram()
        info = env.data.lookup_all()
        agent = TeammateCoordinator(SimpleNamespace(
            env='NHL94-Genesis-v0', action_type='FILTERED', goalie_policy='off',
            side='home' if side == 1 else 'away', hyperparams_dict={'frame_skip': 4}, one_timers=one_timer,
            team_size=team_size, teammate_transient_receiving=lifecycle))
        events, actions, defender_positions = [], [], []
        trace, forecast, released = [], None, False
        for _ in range(frames):
            info = agent.controllers[0].platform.frame_info(env, info)
            result = agent.act(info)
            events.extend(result.diagnostics['teammates']['events'])
            released |= any(event['phase'] == 'released' for event in result.diagnostics['teammates']['events'])
            if coast_after_release and released:
                result.action[:] = 0
                agent.previous[:] = 0
                for controller in agent.controllers:
                    controller.accept_applied_action(np.zeros(12, dtype=np.int8))
            if details:
                state = agent.states[0]
                player = state.team1.get_player_by_scnum(receiver)
                if released and forecast is None:
                    forecast = {'frame': agent.frames, 'path': list(reception_trajectory(state, player, 16)),
                                'contact': one_timer_reception(state, player)}
                trace.append({'frame': agent.frames, 'owner': state.engine.puck_owner,
                              'receiver': receiver, 'shooter': state.engine.shot_player,
                              'attempts': state.team1.one_timer_attempts, 'armed': bool(player.is_one_timer),
                              'position': [player.precise_x, player.precise_y],
                              'puck': [state.puck.x, state.puck.y],
                              'puck_precise': [state.puck.precise_x, state.puck.precise_y],
                              'puck_motion': [state.puck.motion_x, state.puck.motion_y],
                              'puck_height': state.puck.height, 'puck_friction': state.puck.friction,
                              'animation': [player.live_anim, player.live_anim_frame],
                              'stick': [player.stick_x, player.stick_y],
                              'one_timer_offsets': player.one_timer_offsets,
                              'coast_intervention': bool(coast_after_release and released),
                              'clearance': min(np.hypot(player.x-other.x, player.y-other.y)
                                               for team in (state.team1, state.team2)
                                               for other in (*team.players, team.goalie) if other is not player),
                              'slots': [row['slot'] for row in result.diagnostics['teammates']['controllers']],
                              'pads': result.action.tolist(),
                              'commitment': result.diagnostics['teammates']['pending_pass'],
                              'events': result.diagnostics['teammates']['events']})
            actions.append(result.action.copy())
            defender = agent.states[0].team2.get_player_by_scnum(opponents[0])
            defender_positions.append((defender.x, defender.y))
            *_, info = env.step(result.action)
        return {'trace': trace, 'forecast': forecast} if details else (events, actions, defender_positions)
    finally:
        env.close()


class NativeTeammateTests(unittest.TestCase):
    def test_selected_pair_receptions_and_one_timers_with_active_defenders_at_both_ends(self):
        for side in (1, 2):
            for one_timer in (False, True):
                with self.subTest(side=side, one_timer=one_timer):
                    events, actions, defenders = selected_pair_fixture(side, one_timer=one_timer,
                                                                       receiver_facing=4 if one_timer else None)
                    self.assertTrue(any(event['phase'] == ('shot' if one_timer else 'received') for event in events), events)
                    self.assertGreater(np.linalg.norm(np.subtract(defenders[-1], defenders[0])), 1)
                    if one_timer:
                        shots = [event for event in events if event['phase'] == 'shot']
                        self.assertTrue(all(event['commitment']['cue'] is not None for event in shots))
                        cues = [event for event in events if event['phase'] == 'one-timer-cue']
                        self.assertEqual(len(cues), len({event['commitment']['started'] for event in cues}))
                    else:
                        release = next(event['frame'] for event in events if event['phase'] == 'released')
                        received = next(event['frame'] for event in events if event['phase'] == 'received')
                        self.assertFalse(any(action[20] for action in actions[release:received]))

    def test_invalid_shot_pose_keeps_pass_ordinary_and_tracks_native_interception(self):
        events, actions, _ = selected_pair_fixture(1, one_timer=True, receiver_facing=0)
        self.assertTrue(any(event['phase'] == 'released' for event in events))
        self.assertFalse(any(event['phase'] == 'one-timer-cue' for event in events))
        self.assertTrue(any(event.get('reason') == 'intercepted' for event in events))
        self.assertFalse(any(action[20] for action in actions[:30]))

    def test_each_controller_independently_moves_its_selected_skater(self):
        env = make_retro(game='NHL94-Genesis-v0', state=DEFAULT_STATE, num_players=2)
        try:
            info = prepare_match(env, seed=94, seconds=300)
            for frame in range(1500):
                action = np.zeros(24, dtype=np.int8)
                action[0] = frame % 8 < 4
                _, _, _, _, info = env.step(action)
                if 6 <= info['puck_owner'] < 11 and info['goalie_clock_flags'] & 1 == 0:
                    break
            else:
                self.fail('Could not reach live CPU possession for the movement check')
            self.assertEqual((info['defense_team1'], info['defense_team2']), (1, 1))
            self.assertEqual((info['goalie_mode_1'], info['goalie_mode_2']), (1, 1))
            slots = [info['defense_control1'], info['defense_control2']]
            self.assertNotEqual(*slots)
            self.assertTrue(all(0 <= slot < 5 for slot in slots))
            snapshot = env.em.get_state()
            for controller, slot in enumerate(slots):
                positions = []
                for direction in (6, 7):
                    env.em.set_state(snapshot)
                    env.data.update_ram()
                    action = np.zeros(24, dtype=np.int8)
                    action[controller * 12 + direction] = 1
                    for _ in range(40):
                        env.step(action)
                    memory = env.data.memory
                    positions.append(tuple(memory.extract(0xFFB04A + slot * 0x80 + offset, '>i2')
                                           for offset in (0, 0x14)))
                self.assertGreater(np.linalg.norm(np.subtract(*positions)), 2, (controller, positions))
        finally:
            env.close()

    def test_recorded_inputs_replay_to_identical_ram_and_seeded_actions_repeat(self):
        with TemporaryDirectory(prefix='nhl94-teammates-') as directory:
            trace = str(Path(directory) / 'inputs.jsonl')
            first = play_match(seed=94, seconds=30, trace=trace)
            replay = replay_trace(trace)
            second = play_match(seed=94, seconds=30)
            self.assertTrue(first['completed'])
            for key in ('frames', 'actions_sha256', 'final_ram_sha256'):
                self.assertEqual(first[key], replay[key], key)
                self.assertEqual(first[key], second[key], key)
            self.assertEqual(first['initial_ram_sha256'], second['initial_ram_sha256'])
            self.assertTrue(all(first['selected_frames']))
            self.assertTrue(any('passing-option' in role for role in first['roles']))
            self.assertTrue(any('coverage' in role for role in first['roles']))

    def test_support_options_trace_is_observational_on_both_sides(self):
        for side in ('home', 'away'):
            with self.subTest(side=side), TemporaryDirectory(prefix='nhl94-support-') as directory:
                trace = str(Path(directory) / 'inputs.jsonl')
                first = play_match(seed=2026101800, seconds=30, side=side,
                                   support_options=True, trace=trace)
                second = play_match(seed=2026101800, seconds=30, side=side, support_options=True)
                replay = replay_trace(trace)
                self.assertTrue(first['completed'])
                for key in ('frames', 'actions_sha256', 'final_ram_sha256'):
                    self.assertEqual(first[key], second[key], key)
                    self.assertEqual(first[key], replay[key], key)

    def test_all_three_regulation_periods_complete(self):
        result = play_match(seed=94, periods=3)
        self.assertTrue(result['completed'], result)
        self.assertEqual([row['period'] for row in result['periods']], [1, 2, 3])
        self.assertTrue(any('intermission' in role for role in result['roles']))

    def test_short_baselines_complete_against_cpu(self):
        for strategy in ('single', 'independent'):
            with self.subTest(strategy=strategy):
                result = play_match(seed=94, seconds=30, strategy=strategy)
                self.assertTrue(result['completed'], result)

    def test_ducks_away_matches_and_replay_keep_sabres_under_cpu_control(self):
        with TemporaryDirectory(prefix='nhl94-ducks-') as directory:
            trace = str(Path(directory) / 'away-inputs.jsonl')
            result = play_match(seed=94, seconds=30, side='away', trace=trace)
            replay = replay_trace(trace)
            self.assertTrue(result['completed'], result)
            self.assertEqual(result['teams'], [2, 0])
            self.assertEqual(result['side'], 'away')
            self.assertEqual(result['final_ram_sha256'], replay['final_ram_sha256'])
            self.assertTrue(any('passing-option' in role for role in result['roles']))
            baseline = play_match(seed=94, seconds=30, side='away', strategy='single')
            self.assertTrue(baseline['completed'])
            self.assertEqual(result['lineup'], baseline['lineup'])
            self.assertEqual(result['initial_ratings'], baseline['initial_ratings'])

    def test_real_playback_display_accepts_both_controllers(self):
        from nhl94_ai.evaluation.play import NHL94Player, parse_cmdline
        args = parse_cmdline(['--env=NHL94-Genesis-v0', '--nn=ClassicAIV1',
                              '--rf=PostPlay', '--mode=team_vs_game'])
        args.hyperparams_dict = {'frame_skip': 4}
        with patch.dict(os.environ, {'SDL_VIDEODRIVER': 'dummy', 'SDL_AUDIODRIVER': 'dummy'}):
            player = NHL94Player(args, None)
            try:
                state = player.display_env.reset()
                info = player._scripted_reset_info()
                for _ in range(12):
                    action = player._ai_actions(state, info)
                    self.assertEqual(action.shape, (24,))
                    player.display_env.set_ai_sys_info(player.ai_sys)
                    state, _, _, info = player.display_env.step([action])
                self.assertEqual(len(player.display_env.teammates['controllers']), 2)
            finally:
                player.close()


if __name__ == '__main__':
    from tests.integration.multitap import NativeMultitapTests  # pylint: disable=unused-import
    unittest.main()

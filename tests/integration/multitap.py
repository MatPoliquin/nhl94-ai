"""Explicit native EA 4-Way Play gate and three-bot acceptance checks."""
import hashlib
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from nhl94_ai.agents.teammates import TeammateCoordinator
from nhl94_ai.config import load_hyperparams, resolve_hyperparams_for_model
from nhl94_ai.env.factory import make_retro
from nhl94_ai.evaluation.play import NHL94Player, parse_cmdline
from nhl94_ai.evaluation.teammates import DEFAULT_STATE, play_match, prepare_match, replay_trace
from nhl94_ai.game.multitap import prepare_multitap_save, controller_snapshot
from nhl94_ai.game.ram import configure_teammates
from tests.unit.test_multitap import three_args


def scripted_human(frame, _info):
    action = np.zeros(12, dtype=np.int8)
    action[6 if frame // 180 % 2 else 7] = 1
    action[0], action[8] = frame % 60 < 4, frame % 90 == 45
    return action


class NativeMultitapTests(unittest.TestCase):
    def test_ports_buttons_movement_and_snapshot_replay(self):
        for human in (False, True):
            env = make_retro(game='NHL94-Genesis-v0', state=DEFAULT_STATE,
                             num_players=4 if human else 3, allow_menu_buttons=True)
            try:
                prepare_match(env, seed=94, seconds=300, team_size=3, human=human)
                memory = env.data.memory
                snapshot = env.em.get_state()
                count = 4 if human else 3
                for port in range(count):
                    for button, native in ((7, 8), (0, 16), (8, 32)):
                        self.assertTrue(env.em.set_state(snapshot))
                        action = np.zeros(count * 12, dtype=np.int8)
                        action[port * 12 + button] = 1
                        for _ in range(3):
                            env.step(action)
                        channels = [memory.extract(0xFFBEF6 + p, '|u1') for p in range(4)]
                        self.assertEqual(channels, [255 ^ native if p == port else 255 for p in range(4)])
                env.em.set_state(snapshot)
                movement = False
                for frame in range(2000):
                    action = np.zeros(count * 12, dtype=np.int8)
                    action[0] = frame % 8 < 4
                    env.step(action)
                    if frame < 200 or frame % 40:
                        continue
                    snapshot = env.em.get_state()
                    slots = [memory.extract(0xFFC320 + p * 2, '>i2') for p in range(count)]
                    if len(slots) != len(set(slots)) or any(slot not in (0, 1, 2, 3, 4, 6, 7, 8, 9, 10) for slot in slots):
                        continue
                    distances = []
                    for port, slot in enumerate(slots):
                        positions = []
                        for direction in (6, 7):
                            env.em.set_state(snapshot)
                            action = np.zeros(count * 12, dtype=np.int8)
                            action[port * 12 + direction] = 1
                            for _ in range(40):
                                env.step(action)
                            positions.append([memory.extract(0xFFB04A + slot * 128 + offset, '>i2') for offset in (0, 0x14)])
                        distances.append(np.linalg.norm(np.subtract(*positions)))
                    env.em.set_state(snapshot)
                    if min(distances) > 2:
                        movement = True
                        break
                self.assertTrue(movement, 'No common live snapshot demonstrated all independent controllers')
                hashes = []
                for _ in range(2):
                    env.em.set_state(snapshot)
                    for frame in range(120):
                        action = np.tile(scripted_human(frame, {}), count)
                        env.step(action)
                    hashes.append(hashlib.sha256(env.get_ram().tobytes()).hexdigest())
                self.assertEqual(hashes[0], hashes[1])
                # Reset reapplies polling setup; gameplay snapshot restore never does.
                env.reset()
                prepare_multitap_save(env)
            finally:
                env.close()

    def test_independent_coordinated_and_human_traces_repeat(self):
        with TemporaryDirectory(dir='/home/mat/OUTPUT', prefix='nhl94-multitap-') as directory:
            for strategy, human, size in (('independent', False, 3), ('coordinated', False, 3),
                                          ('coordinated', True, 3), ('coordinated', True, 2)):
                trace = str(Path(directory) / f'{strategy}-{human}-{size}.jsonl')
                kwargs = dict(team_size=size, strategy=strategy, seed=94, seconds=30,
                              human_input=scripted_human if human else None)
                first = play_match(**kwargs, trace=trace)
                replay, second = replay_trace(trace), play_match(**kwargs)
                self.assertTrue(first['completed'])
                self.assertTrue(all(count > 100 for count in first['selected_frames']))
                for key in ('frames', 'actions_sha256', 'final_ram_sha256'):
                    self.assertEqual(first[key], replay[key], key)
                    self.assertEqual(first[key], second[key], key)

    def test_native_penalty_and_unavailable_skaters_preserve_assignments(self):
        env = make_retro(game='NHL94-Genesis-v0', state=DEFAULT_STATE,
                         num_players=3, allow_menu_buttons=True)
        try:
            env.reset()
            prepare_multitap_save(env)
            configure_teammates(env.data, team_size=3)
            memory = env.data.memory
            # Reset-only fixture: enable penalties without offsides, using the
            # original saved RNG and a fixed starting controller permutation.
            memory.assign(0xFFD056, '>u2', 2)
            for port in range(3):
                memory.assign(0xFFC320 + port * 2, '>i2', port)
            penalty_seen = unavailable_seen = False
            unsettled = 0
            for frame in range(10000):
                action = np.zeros(36, dtype=np.int8)
                for port in range(3):
                    action[12 * port + 4 + (frame // 60 + port) % 4] = 1
                    action[12 * port] = frame % 16 < 4
                    action[12 * port + 8] = frame % 12 < 6
                *_, info = env.step(action)
                penalty_seen |= bool(memory.extract(0xFFC3EA, '|u1'))
                unavailable_seen |= any(info[f'defense_{slot}_unavailable'] & 4 for slot in (0, 1, 2, 3, 4))
                view = controller_snapshot(info, 3)
                unsettled = unsettled + 1 if view['teammate_transient'] else 0
                self.assertLess(unsettled, 120)
                if not view['teammate_transient']:
                    self.assertEqual([view[f'defense_team{i}'] for i in range(1, 5)], [1, 1, 1, 0])
            self.assertTrue(penalty_seen and unavailable_seen)
        finally:
            env.close()

    def test_three_regulation_periods_with_cpu_and_human(self):
        for human in (False, True):
            match = play_match(team_size=3, periods=3, seed=94,
                               human_input=scripted_human if human else None)
            self.assertTrue(match['completed'], match)
            self.assertEqual([row['period'] for row in match['periods']], [1, 2, 3])
            self.assertTrue(all(count > 1000 for count in match['selected_frames']))
            self.assertGreater(match['pass_attempts'], 0)
            self.assertGreater(sum(match['shots']), 0)
            output = Path('/home/mat/OUTPUT/three-bots')
            output.mkdir(parents=True, exist_ok=True)
            (output / f'acceptance-{"human" if human else "cpu"}.json').write_text(json.dumps(match, indent=2) + '\n')

    def test_real_display_merge_pause_restart_and_shutdown(self):
        import pygame
        with patch.dict(os.environ, {'SDL_VIDEODRIVER': 'dummy', 'SDL_AUDIODRIVER': 'dummy'}):
            args = parse_cmdline(['--mode=team_vs_player', '--team-size=3', '--nn=ClassicAIV1',
                                  '--env=NHL94-Genesis-v0', '--rf=PostPlay', '--seed=94'])
            args.hyperparams_dict = resolve_hyperparams_for_model(load_hyperparams(args.hyperparams), args.nn)
            player = NHL94Player(args, None)
            try:
                display = player.display_env
                display.env.reset()
                info = display.env.reset_infos[0]
                agent = TeammateCoordinator(three_args(True))
                for frame in range(240):
                    if frame == 120:
                        display.env.reset()
                        info = display.env.reset_infos[0]
                        agent.reset()
                    action = agent.act(info).action
                    expected = scripted_human(frame, info) if frame < 200 else np.zeros(12, dtype=np.int8)
                    with patch.object(display, 'get_human_input', return_value=expected):
                        _, reward, _, infos = display.step([action])
                    self.assertEqual(reward.shape, (1,))
                    histories = display.env.get_attr('teammate_action_states')[0]
                    np.testing.assert_array_equal(histories[-1]['last_env_action'], expected)
                    np.testing.assert_array_equal(np.concatenate([history['last_env_action'] for history in histories[:3]]), action)
                    info = infos[0]
                display.paused = True
                before = display.playback_frames
                pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_SPACE))
                display.wait_until_running()
                self.assertEqual(display.playback_frames, before)
                self.assertFalse(display.paused)
            finally:
                player.close()


if __name__ == '__main__':
    unittest.main()

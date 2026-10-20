"""Bounded native layout, physical input, old-format replay and away-game gates."""
import json
from itertools import product
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from nhl94_ai.agents.teammates import TeammateCoordinator
from nhl94_ai.evaluation.teammates import DEFAULT_STATE, make_match_env, prepare_match, play_match, replay_trace
from nhl94_ai.game.multitap import controller_snapshot
from tests.integration.multitap import scripted_human
from tests.unit.test_teammates import team_args


class AwayNativeTests(unittest.TestCase):
    def test_all_layouts_complete_and_replay(self):
        with TemporaryDirectory() as directory:
            for side, size, human in product(('home', 'away'), (2, 3), (False, True)):
                with self.subTest(side=side, size=size, human=human):
                    trace = Path(directory) / f'{side}-{size}-{human}.jsonl'
                    match = play_match(side=side, team_size=size, seconds=15, trace=trace,
                                       human_input=scripted_human if human else None)
                    self.assertTrue(match['completed'])
                    self.assertTrue(all(count > 100 for count in match['selected_frames']))
                    # Older home traces omitted side; v1/v2 retain that meaning.
                    if side == 'home':
                        rows = trace.read_text().splitlines()
                        header = json.loads(rows[0])
                        del header['side']
                        trace.write_text('\n'.join([json.dumps(header), *rows[1:]]) + '\n')
                    replay = replay_trace(trace)
                    for key in ('frames', 'actions_sha256', 'final_ram_sha256'):
                        self.assertEqual(match[key], replay[key], key)

    def test_away_ports_move_their_selected_team_and_snapshot_replays(self):
        # Reuse the physical input gate for all three/four-channel away ports.
        from tests.integration import multitap
        original = multitap.prepare_match
        with patch.object(multitap, 'prepare_match',
                          side_effect=lambda *a, **kw: original(*a, **kw, side=2)):
            multitap.NativeMultitapTests('test_ports_buttons_movement_and_snapshot_replay'
                               ).test_ports_buttons_movement_and_snapshot_replay()

    def test_away_real_display_human_merge_pause_and_reset(self):
        from tests.integration import multitap
        parse, args = multitap.parse_cmdline, multitap.three_args
        with patch.object(multitap, 'parse_cmdline', side_effect=lambda flags: parse([*flags, '--side=away'])), \
                patch.object(multitap, 'three_args', side_effect=lambda human: args(human, side='away')):
            multitap.NativeMultitapTests('test_real_display_merge_pause_restart_and_shutdown'
                                        ).test_real_display_merge_pause_restart_and_shutdown()

    def test_away_cpu_control_switches_and_live_roles_follow_native_players(self):
        for size in (2, 3):
            env = make_match_env(DEFAULT_STATE, team_size=size)
            try:
                info = prepare_match(env, seed=95, seconds=30, team_size=size, side=2)
                agent = TeammateCoordinator(team_args(side='away', team_size=size))
                selections, roles = set(), set()
                cpu_positions = set()
                for _ in range(2400):
                    result = agent.act(info)
                    rows = result.diagnostics['teammates']['controllers']
                    selections.add(tuple(row['slot'] for row in rows))
                    roles.update(row['role'] for row in rows)
                    view = controller_snapshot(info, size, side=2) if size == 3 else info
                    if not view.get('teammate_transient'):
                        self.assertEqual([info[f'defense_team{i}'] for i in range(1, 5)],
                                         [2] * size + [0] * (4 - size))
                        self.assertTrue(all(row['slot'] < 0 or 6 <= row['slot'] < 12 for row in rows))
                        self.assertFalse(any(info[f'defense_{slot}_flags'] & 8 for slot in range(5)))
                    cpu_positions.add((info['p1_x'], info['p1_y']))
                    _, _, _, _, info = env.step(result.action)
                self.assertGreater(len(selections), 2)
                self.assertGreater(len(cpu_positions), 20)
                self.assertIn('retrieve', roles)
                self.assertEqual(agent.states[0].team1.controller, 2)
            finally:
                env.close()

    def test_away_three_bot_regulation_periods_highlights_and_replay(self):
        from nhl94_ai.evaluation import teammates
        output = Path('tmp/away-teammates')
        output.mkdir(parents=True, exist_ok=True)
        trace = output / 'away-regulation-highlights.jsonl'
        filter_inputs = teammates.filter_menu_actions
        highlight_seen = False

        def watch_first_highlight(data, action):
            nonlocal highlight_seen
            highlight_seen |= bool(data.lookup_all()['goalie_clock_flags'] & 0x10)
            action = filter_inputs(data, action)
            if not highlight_seen:
                # Normal playback skips the intermission preview. Let the ROM
                # enter one naturally, using only buttons, to exercise restore.
                action[3::12] = 0
            return action

        with patch.object(teammates, 'filter_menu_actions', side_effect=watch_first_highlight):
            match = play_match(side='away', team_size=3, periods=3, seed=94, trace=trace)
        (output / 'away-regulation-highlights.json').write_text(json.dumps(match, indent=2) + '\n')
        self.assertTrue(match['completed'])
        self.assertEqual([row['period'] for row in match['periods']], [1, 2, 3])
        self.assertGreater(match['roles'].get('p1:highlight', 0), 0)
        self.assertGreater(match['pass_attempts'], 0)
        self.assertGreater(sum(match['shots']), 0)
        replay = replay_trace(trace)
        for key in ('frames', 'actions_sha256', 'final_ram_sha256'):
            self.assertEqual(match[key], replay[key], key)


if __name__ == '__main__':
    unittest.main()

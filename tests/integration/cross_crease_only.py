"""Ordinary-ROM evidence for a specialist that must skate into its own chances."""
from collections import defaultdict

from nhl94_ai.evaluation.cross_crease_only import build_parser, run
from nhl94_ai.game.constants import GameConsts as Buttons
from nhl94_ai.tasks.cross_crease_setup import goalie_attributes


def native_contracts():
    report = run(build_parser().parse_args([
        '--seeds', '1', '--sides', '1', '--starts', 'close', 'deep', '--trace',
    ]))
    paired = defaultdict(list)
    for row in report['trials']:
        paired[row['seed'], row['side'], row['direction'], row['start']].append(row)
        assert row['initial']['velocity'] == (0, 0), row['initial']
        assert row['initial']['goalie_cpu'] and not row['initial']['shootout']
        assert row['initial']['goalie_attributes'] == goalie_attributes(6)
        assert len(row['initial']['inactive']) == 10
        if row['policy'] == 'only':
            assert row['presses'] and row['presses'][0]['frame'] > 0
            assert all(Buttons.INPUT_B not in frame['action'] for frame in row['timeline'])
    for rows in paired.values():
        assert len(rows) == 2 and rows[0]['snapshot_sha256'] == rows[1]['snapshot_sha256']
        assert rows[0]['initial'] == rows[1]['initial']
    assert report['summary']['only']['goals'] > 0, report['summary']
    print('PASS: matched from-rest starts, active strong CPU goalie, ordinary skating and actual goals; no passes')


if __name__ == '__main__':
    native_contracts()

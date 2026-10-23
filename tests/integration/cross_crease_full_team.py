"""Verify native full-roster branches, control assignment, and repeatable restores."""
from collections import defaultdict

from nhl94_ai.evaluation.cross_crease_full_team import build_parser, run
from nhl94_ai.game.constants import GameConsts as Buttons


def validate_report(report):
    paired = defaultdict(list)
    for row in report['trials']:
        paired[row['id']].append(row)
        assert row['full_lineup_observations'] == row['frames'] + 1, row['id']
        assert tuple(row['initial']['goalie_modes']) == (1, 1)
        assert len(row['initial']['positions']) == 12
        assert min(row['initial']['roles']) >= 0
        # Native support skaters can hold position. Full roles and in-rink
        # coordinates establish the roster; neither team may be entirely parked.
        if row['frames'] > 60:
            assert sum(row['actor_travel'][:5]) > 0
            assert sum(row['actor_travel'][6:11]) > 0
        if row['policy'] != 'classic':
            assert all(set(frame['action']) <= {Buttons.INPUT_UP, Buttons.INPUT_DOWN,
                                               Buttons.INPUT_LEFT, Buttons.INPUT_RIGHT, Buttons.INPUT_C}
                       for frame in row['timeline'])
    assert paired, 'No full-team attacking possessions collected.'
    for rows in paired.values():
        assert {row['policy'] for row in rows} == set(report['summary'])
        for row in rows[1:]:
            assert rows[0]['snapshot_sha256'] == row['snapshot_sha256']
            assert rows[0]['ram_sha256'] == row['ram_sha256']
            assert rows[0]['initial'] == row['initial']
    return paired


def native_contracts():
    args = build_parser().parse_args([
        '--matchups', 'penguins-senators', 'senators-penguins',
        '--seeds', '1', '--samples', '1', '--collect-frames', '6000', '--trace',
        '--policies', 'roam', 'roam-v1', 'only', 'classic',
    ])
    first, second = run(args), run(args)
    assert {row['side'] for row in first['trials']} == {1, 2}
    assert len(validate_report(first)) == 2
    validate_report(second)
    assert [(r['id'], r['policy'], r['actions_sha256'], r['outcome']) for r in first['trials']] == [
        (r['id'], r['policy'], r['actions_sha256'], r['outcome']) for r in second['trials']]
    print('PASS: all twelve native players, automatic goalies, identical branch starts, both sides, repeatable inputs')


if __name__ == '__main__':
    native_contracts()

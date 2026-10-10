"""Explicit ordinary-ROM isolation, ratings and matched crossing contracts."""
from collections import defaultdict

from nhl94_ai.evaluation.cross_crease_probe import build_parser, run


def native_contracts():
    report = run(build_parser().parse_args(['--scenarios', 'near', '--seeds', '1']))
    groups = defaultdict(list)
    for row in report['trials']:
        if row['policy'] != 'stationary':
            groups[row['side'], row['direction'], row['seed'], row['goalie_rating']].append(row)
        assert row['initial']['goalie_cpu'] and not row['initial']['shootout']
        assert len(row['initial']['inactive']) == 10
        assert row['windup_frame'] is not None and row['release_frame'] is not None
        assert row['goalie_contact_impulses'] == 0
    for rows in groups.values():
        assert len({row['initial_ram_sha256'] for row in rows}) == 1
        assert len({row['initial_snapshot_sha256'] for row in rows}) == 1
    high = report['summary']['high']
    assert high['held']['goals'] == 4 and high['tap']['goals'] == 0, high
    assert high['stationary']['goals'] == 0, high
    assert high['classic']['goals'] == 4, high
    print('PASS: other ten actors remain inactive/stationary, live weak/strong ratings are verified, '
          'and four native high-goalie crossings score from matched starts without contact')
    rest = run(build_parser().parse_args(['--scenarios', 'rest-near', '--seeds', '1', '--policies', 'classic']))
    for policies in rest['summary'].values():
        assert policies['classic']['goals'] == 4 and policies['classic']['goalie_contact_impulses'] == 0, policies
    print('PASS: Classic builds lateral momentum from rest in the close lane and scores against both goalies')


if __name__ == '__main__':
    native_contracts()

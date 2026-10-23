"""Explicit native distance, idle collision and genuine CPU pursuit contracts."""
from collections import defaultdict

from nhl94_ai.evaluation.cross_crease_distance import build_parser, run


def check_pairs(report):
    pairs = defaultdict(set)
    for row in report['trials']:
        key = row['side'], row['direction'], row['seed'], row['distance'], row['lateral'], row['traffic'], row['goalie_rating']
        pairs[key].add((row['initial_snapshot_sha256'], row['initial_ram_sha256']))
        assert row['separations']['initial']['longitudinal'] == row['distance']
        assert row['frames'] == len(row['timeline'])
        assert row['initial']['goalie_cpu'] and not row['initial']['shootout']
        assert len(row['initial']['inactive']) == (10 if row['traffic'] == 'none' else 9)
        if row['release_frame'] is not None:
            assert row['separations']['release'] is not None
    assert all(len(hashes) == 1 for hashes in pairs.values()), pairs
    assert {row['handedness'] for row in report['trials']} == {0, 1}


def native_contracts():
    parser = build_parser()
    clean = run(parser.parse_args([
        '--distances', '34', '90', '--lateral', '1.5', '--policies', 'held',
        '--goalies', 'high', '--seed', '83000', '--seeds', '2', '--trace',
    ]))
    check_pairs(clean)
    near = next(cell for cell in clean['grid'] if cell['distance'] == 34)
    assert near['results']['high']['held']['middle']['contact_free_goals'] == 8
    print('PASS: close/far starts, exact momentum, native charge/release distance and old near-lane goals')
    traffic = run(parser.parse_args([
        '--distances', '50', '90', '--lateral', '1.5', '--policies', 'tap', 'held',
        '--traffic', 'none', 'friendly-initial', 'opponent-initial', 'opponent-release', 'pursuit-initial',
        '--heights', 'middle', 'high', '--goalies', 'high', '--seeds', '2', '--trace',
    ]))
    check_pairs(traffic)
    pursuers = [row for row in traffic['trials'] if row['traffic'] == 'pursuit-initial']
    assert any(row['maximum_traffic_displacement'] > 0 for row in pursuers)
    for row in pursuers:
        decisions = {(frame['traffic']['assignment'], frame['traffic']['decision_timer'],
                      frame['traffic']['animation']) for frame in row['timeline']}
        assert len(decisions) > 1, (row['side'], row['direction'], row['seed'], row['distance'], decisions)
    assert any(event['kind'] == 'shot-block-or-deflection'
               for row in traffic['trials'] for event in row['traffic_events'])
    print('PASS: collidable friendly/opponent idle actors, live CPU pursuit, elevated shots and native interventions')
    between = run(parser.parse_args([
        '--distances', '34', '--lateral', '1.5', '--policies', 'held',
        '--traffic', 'none', 'friendly-between', 'opponent-between', 'pursuit-between',
        '--goalies', 'high', '--seed', '83000', '--seeds', '2', '--trace',
    ]))
    check_pairs(between)
    assert all(row['traffic_fixture']['position'][0] == -row['direction'] * 23.5
               for row in between['trials'] if row['traffic'] != 'none')
    print('PASS: an intervening actor fits the close lane with valid initial body clearance')


if __name__ == '__main__':
    native_contracts()

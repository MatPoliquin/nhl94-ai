"""Explicit fixed-geometry and independent-axis native crossing contracts."""
from collections import defaultdict

from nhl94_ai.evaluation.cross_crease_velocity import build_parser, run


def native_contracts():
    args = build_parser().parse_args([
        '--lateral', '1.5', '2', '--goalward', '0', '0.5',
        '--seeds', '2', '--policies', 'tap', 'held', '--goalies', 'high',
    ])
    report = run(args)
    geometry = defaultdict(set)
    for row in report['trials']:
        initial = row['initial']
        geometry[row['side'], row['direction']].add(
            (tuple(initial['player']), tuple(initial['puck']), tuple(initial['goalie'])))
        assert row['first_c'] == 0
        assert tuple(initial['player']) == (-row['direction'] * 35, (1 if row['side'] == 1 else -1) * 216)
        assert initial['player_velocity'] == row['charge_geometry']['player_velocity']
    assert all(len(poses) == 1 for poses in geometry.values()), geometry
    assert {row['handedness'] for row in report['trials']} == {0, 1}
    for cell in report['grid']:
        if cell['goalward'] == 0:
            held = cell['results']['high']['held']
            assert held['safe_goals'] == held['attempts'] == 8, cell
        if cell['goalward'] == 0.5 and cell['lateral'] == 1.5:
            assert cell['results']['high']['held']['contact_trials'] > 0, cell
    print('PASS: exact independently encoded velocities, constant geometry, mirrored ends, '
          'matched policy RAM and unsafe goalward-drift contacts are measured by the ROM')


if __name__ == '__main__':
    native_contracts()

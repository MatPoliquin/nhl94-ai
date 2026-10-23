"""Native pair regressions and fixed contact-model calibration; no strength claim."""
import argparse
import json
import math
from pathlib import Path

from tests.integration.teammates import selected_pair_fixture


def pair_matrix(output):
    results = []
    for size in (2, 3):
        for side in (1, 2):
            for sender in range(size):
                for receiver in range(size):
                    if sender == receiver:
                        continue
                    for lifecycle in (None, False, True):
                        case = selected_pair_fixture(side, one_timer=True, receiver_facing=4,
                            team_size=size, pair=(sender, receiver), lifecycle=lifecycle, details=True)
                        trace = case['trace']
                        shots = [row for before, row in zip(trace, trace[1:])
                                 if row['attempts'] > before['attempts']]
                        expected = not (size == 3 and side == 1 and receiver == 2)
                        assert bool(shots) == expected, (size, side, sender, receiver, lifecycle)
                        assert all(row['shooter'] == row['receiver'] for row in shots)
                        for row in trace:
                            for event in row['events']:
                                if event['phase'] == 'one-timer-cue':
                                    controller = event['commitment']['receiver_controller'] - 1
                                    assert row['pads'][controller * 12 + 8]
                                    assert row['slots'][controller] == row['receiver']
                        events = [event for row in trace for event in row['events']]
                        if lifecycle is not False and expected:
                            assert any(event['phase'] == 'one-timer-armed' for event in events)
                            assert any(event['phase'] == 'shot' for event in events)
                        option = 'default' if lifecycle is None else str(int(lifecycle))
                        name = f'pair-{size}-{side}-{sender+1}-{receiver+1}-{option}'
                        (output / (name + '.json')).write_text(json.dumps(case) + '\n')
                        results.append({'name': name, 'size': size, 'side': side, 'sender': sender+1,
                                        'receiver': receiver+1, 'lifecycle': lifecycle,
                                        'native_one_timers': len(shots), 'expected_shot': expected})
    return results


def calibration(output):
    results = []
    # Fixed before outcomes: stationary valid/invalid pose and four motion directions.
    cases = ((0, 0, 4), (.5, 0, 4), (-.5, 0, 4), (0, .5, 4), (0, -.5, 4), (0, 0, 0))
    for size in (2, 3):
        for side in (1, 2):
            for receiver in (0, size - 1):
                sender = 1 if receiver == 0 else 0
                for index, (vx, vy, facing) in enumerate(cases):
                    name = f'contact-{size}-{side}-{receiver+1}-{index}'
                    case = selected_pair_fixture(side, one_timer=True, receiver_facing=facing,
                        team_size=size, pair=(sender, receiver), lifecycle=True, details=True,
                        receiver_motion=(vx, vy if side == 1 else -vy), coast_after_release=True, frames=60)
                    (output / (name + '.json')).write_text(json.dumps(case) + '\n')
                    prediction = case['forecast']
                    result = {'name': name, 'size': size, 'side': side, 'receiver': receiver+1,
                              'initial_motion': [vx, vy], 'initial_facing': facing, 'horizons': []}
                    if prediction is None:
                        results.append(result | {'supported': False, 'reason': 'no-native-release', 'passed': False})
                        continue
                    rows = {row['frame']: row for row in case['trace']}
                    unsupported = None
                    for horizon, puck, _, position in prediction['path']:
                        actual = rows[prediction['frame'] + horizon]
                        if actual['owner'] >= 0 or actual['clearance'] < 24:
                            unsupported = 'native-contact-or-possession'
                            break
                        result['horizons'].append({'horizon': horizon,
                            'receiver_error': math.dist(position, actual['position']),
                            'puck_error': math.dist(puck, actual['puck']),
                            'predicted_receiver': position, 'native_receiver': actual['position'],
                            'predicted_puck': puck, 'native_puck': actual['puck']})
                    supported = len(result['horizons']) == 16
                    passed = supported and all(max(row['receiver_error'], row['puck_error']) <= 1
                                               for row in result['horizons'])
                    results.append(result | {'supported': supported, 'reason': unsupported,
                                             'passed': passed, 'contact': prediction['contact']})
    assert len(results) == 48
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    pairs = pair_matrix(args.output)
    print('PASS: 48 default/explicit-option pair fixtures, native shooter/counter and applied-C assertions', flush=True)
    (args.output / 'pairs.json').write_text(json.dumps(pairs, indent=2) + '\n')
    windows = calibration(args.output)
    report = {'windows': windows, 'passed': all(row['passed'] for row in windows),
              'selection': '48 predeclared cases; both sides; 2/3 pads; first/last receiver; six motion/pose cases',
              'domain': '16 coasting frames before native possession/contact; no outcome-based exclusions',
              'max_error': max((max(h['receiver_error'], h['puck_error']) for w in windows for h in w['horizons']), default=None)}
    (args.output / 'calibration.json').write_text(json.dumps(report, indent=2) + '\n')
    print('Calibration:', sum(row['passed'] for row in windows), '/48; gate', report['passed'],
          '; max error', report['max_error'], flush=True)


if __name__ == '__main__':
    main()

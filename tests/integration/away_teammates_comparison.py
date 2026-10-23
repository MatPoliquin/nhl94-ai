"""The plan's fixed 24-run comparison; run once after the native layout gates."""
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from itertools import product
import json
from pathlib import Path

from nhl94_ai.evaluation.teammates import _play_job, source_hash


def run():
    output = Path('tmp/away-teammates')
    output.mkdir(parents=True, exist_ok=True)
    frozen = source_hash()
    jobs = [dict(side=side, team_size=size, seed=seed, pressure=pressure, seconds=120)
            for side, size, seed, pressure in product(('home', 'away'), (2, 3), (94, 95, 96), (True, False))]
    matches = []
    with ProcessPoolExecutor(max_workers=3) as pool:
        for match in pool.map(_play_job, jobs):
            if match['source_sha256'] != frozen or source_hash() != frozen:
                raise RuntimeError('Source changed during the comparison')
            matches.append(match)
            name = f"{match['side']}-{match['team_size']}-{match['seed']}-{int(match['pressure'])}"
            (output / f'{name}.json').write_text(json.dumps(match, indent=2) + '\n')
            print(f"{name}: complete={match['completed']} score={match['score']} shots={match['shots']}", flush=True)
    paired = []
    for index in range(0, len(matches), 2):
        on, off = matches[index:index + 2]
        checks = {key: on[key] == off[key] for key in
                  ('initial_ram_sha256', 'initial_ratings', 'lineup', 'rom_sha256', 'initial_state_sha256')}
        paired.append(dict(side=on['side'], team_size=on['team_size'], seed=on['seed'], checks=checks))
        if not all(checks.values()):
            raise AssertionError(f'Unpaired starting conditions: {paired[-1]}')
    groups = defaultdict(list)
    for match in matches:
        groups[match['side'], match['team_size'], match['pressure']].append(match)
    summary = []
    unsafe = {'unavailable-or-missing-feedback', 'too-close-to-own-goal',
              'exposed-lane', 'approach-unreachable', 'approach-obstructed'}
    for (side, size, pressure), rows in groups.items():
        counts, exits, health = Counter(), Counter(), Counter()
        for row in rows:
            counts.update(row['pressure_outcomes']['counts'])
            exits.update(event['reason'] for event in row['pressure_outcomes']['events'])
            for key in ('transient_frames', 'transient_trap_exits'):
                health[key] += row['controller_health'][key]
        summary.append(dict(
            side=side, team='Sabres' if side == 'home' else 'Ducks', team_size=size, pressure=pressure,
            completed=sum(row['completed'] for row in rows), runs=len(rows),
            goals=[sum(row['score'][i] for row in rows) for i in range(2)],
            shots=[sum(row['shots'][i] for row in rows) for i in range(2)],
            attempts=counts['attempts'], recoveries=counts['team_recoveries_within_60_frames'],
            unsafe_exits=sum(exits[reason] for reason in unsafe), exits=dict(exits),
            **health, max_transient_streak=max(row['controller_health']['max_transient_streak'] for row in rows)))
    report = dict(schema='nhl94-away-teammates-comparison-v1', seeds=[94, 95, 96], seconds=120,
                  source_sha256=frozen, score_order=['bots', 'cpu'], paired=paired, summary=summary,
                  unsafe_exit_reasons=sorted(unsafe),
                  limitation='Three seeds per layout; descriptive observations, not evidence of improved play.')
    (output / 'comparison.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    run()

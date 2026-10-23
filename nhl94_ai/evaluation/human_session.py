"""Play and record independent first periods against Classic at native speed."""
import argparse
from pathlib import Path
from types import SimpleNamespace

from nhl94_ai.agents.classic_config import apply_play_config
from nhl94_ai.agents.registry import resolve_goalie_positioning
from nhl94_ai.evaluation.external import FIXTURES, play_match
from nhl94_ai.evaluation.human_recording import HumanRecording, write_json


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, help='New session directory; existing paths are never overwritten')
    parser.add_argument('--periods', type=int, default=10, help='Number of independent first periods (default: 10)')
    parser.add_argument('--seconds', type=int, default=300, help='Game-clock seconds per period (1..300)')
    parser.add_argument('--seed', type=int, default=94, help='First ROM seed; incremented each period')
    parser.add_argument('--fixture', choices=tuple(FIXTURES), default='sabres-ducks')
    parser.add_argument('--side', choices=('home', 'away'), default='away', help='Your side (default: away)')
    parser.add_argument('--classic-config', help='Classic JSON settings, as in ordinary play')
    parser.add_argument('--goalie-positioning', choices=('classic', 'deep', 'deep-csave'))
    parser.add_argument('--snapshot-interval', type=int, default=600, help='Native frames between seek snapshots')
    parser.add_argument('--no-gamepad', action='store_true')
    return parser


def classic_options(args):
    options = SimpleNamespace(nn='ClassicAIV1', mode='player_vs_model', action_type='FILTERED',
                              classic_config=args.classic_config, classic_control_interval=None,
                              goalie_policy=None, goalie_positioning=args.goalie_positioning)
    apply_play_config(options)
    if options.goalie_policy is None:
        options.goalie_policy = 'selective'
    options.goalie_positioning = resolve_goalie_positioning(options)
    return vars(options)


def summarize(rows):
    completed = [row for row in rows if row['completed']]
    margins = [row['goals'][row['human_side'] - 1] - row['goals'][2 - row['human_side']]
               for row in completed]
    return {'completed_periods': len(completed), 'incomplete_periods': len(rows) - len(completed),
            'human_wins': sum(value > 0 for value in margins),
            'classic_wins': sum(value < 0 for value in margins), 'draws': margins.count(0)}


def run(args):
    if args.periods < 1 or not 1 <= args.seconds <= 300 or args.snapshot_interval < 1:
        raise ValueError('Use positive periods/snapshot-interval and 1..300 seconds')
    if not 0 <= args.seed < 2**32 or args.seed + args.periods > 2**32:
        raise ValueError('All period seeds must fit uint32')
    options = classic_options(args)
    path = Path(args.output).expanduser().resolve()
    path.mkdir(parents=True, exist_ok=False)
    rows = []
    report = {'schema': 'nhl94-human-session-v1', 'settings': vars(args),
              'classic_options': options, 'matches': rows, 'summary': summarize(rows),
              'status': 'recording'}
    write_json(path / 'session.json', report)
    try:
        for index in range(args.periods):
            name = f'period-{index + 1:03d}'
            job = {'fixture': args.fixture, 'side': 1 if args.side == 'home' else 2,
                   'seed': args.seed + index, 'seconds': args.seconds, 'periods': 1,
                   'opponent': 'classic', 'human': True, 'no_gamepad': args.no_gamepad,
                   'classic_options': options}
            recorder = HumanRecording(path / name, job, snapshot_interval=args.snapshot_interval)
            print(f'Period {index + 1}/{args.periods}, seed {job["seed"]}: {path / name}', flush=True)
            try:
                result = play_match({**job, '_recorder': recorder})
            except BaseException as error:
                result = recorder.finish(error=f'{type(error).__name__}: {error}')
                rows.append({'directory': name, **result})
                raise
            else:
                result = recorder.finish(result)
                rows.append({'directory': name, **result})
            finally:
                recorder.close()
                report['summary'] = summarize(rows)
                write_json(path / 'session.json', report)
            print(f'Home–away: {result["goals"]}; completed={result["completed"]}', flush=True)
            if result['user_quit']:
                report['status'] = 'stopped'
                break
            if not result['completed']:
                raise RuntimeError('Period ended before completion; partial result saved')
        else:
            report['status'] = 'complete'
    except KeyboardInterrupt:
        report['status'] = 'interrupted'
    except BaseException:
        report['status'] = 'error'
        raise
    finally:
        write_json(path / 'session.json', report)
    print(f'Replays and results saved in {path}', flush=True)
    return report

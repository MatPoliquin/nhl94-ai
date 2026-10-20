"""Record native SNES periods: play against Classic, or Classic on both sides.

Sessions keep every native frame's buttons, decoded observations, Classic
diagnostics, periodic emulator snapshots and per-frame RAM hashes, so SNES
ROM mechanics work can replay and inspect verified periods offline.
"""
import argparse
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from nhl94_ai.env.factory import make_retro
from nhl94_ai.evaluation.human_recording import write_json
from nhl94_ai.evaluation.snes_recording import (
    GAME, SnesRecording, scoreboard,
)
from nhl94_ai.game.platforms import get_backend
from nhl94_ai.game.state import NHL94GameState

BACKEND = get_backend('snes')


def seed_words(seed):
    """Dual LFSR state words at $0792/$0794 (ROM $808658: poly 0x1D87, draw = sum).

    Both words stay nonzero so the stream is fully seed-driven; a zero word would
    remap to the ROM's reset constants 0xA5A5/0xB4B4 on the first draw.
    """
    word1 = (seed * 0x9E37 + 1) & 0xFFFF or 1
    word2 = ((seed ^ 0x5A5A) * 0x2545 + 1) & 0xFFFF or 1
    return word1, word2


def seed_rng(env, seed):
    """Pin the ROM's RNG state at period start so a seed fully determines the period."""
    env.data.set_variable('rng_state_1', {'address': 0x7E0792, 'type': '<u2'})
    env.data.set_variable('rng_state_2', {'address': 0x7E0794, 'type': '<u2'})
    word1, word2 = seed_words(seed)
    env.data.set_value('rng_state_1', word1)
    env.data.set_value('rng_state_2', word2)
    env.data.update_ram()


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, help='New session directory; existing paths are never overwritten')
    parser.add_argument('--periods', type=int, default=10, help='Number of independent first periods (default: 10)')
    parser.add_argument('--side', choices=('home', 'away'), default='away',
                        help='Your team with --human; otherwise only labels the first Classic (default: away)')
    parser.add_argument('--human', action='store_true',
                        help='Play the selected team yourself; Classic takes the other controller')
    parser.add_argument('--classic-config', help='Classic JSON settings, as in ordinary play')
    parser.add_argument('--frame-skip', type=int, default=4, help='Classic tactical decision interval')
    parser.add_argument('--snapshot-interval', type=int, default=600, help='Native frames between seek snapshots')
    parser.add_argument('--seed', type=int,
                        help='Base RNG seed; period N uses seed + N - 1 (default: unseeded save state)')
    parser.add_argument('--no-gamepad', action='store_true')
    return parser


def classic_options(args):
    from nhl94_ai.agents.classic_config import apply_play_config
    from nhl94_ai.agents.registry import resolve_goalie_positioning
    options = SimpleNamespace(env=GAME, nn='ClassicAIV1', mode='model_vs_model', action_type='FILTERED',
                              classic_config=args.classic_config, classic_control_interval=None,
                              goalie_policy=None, goalie_positioning=None)
    apply_play_config(options)
    if options.goalie_policy is None:
        options.goalie_policy = 'selective'
    options.goalie_positioning = resolve_goalie_positioning(options)
    return vars(options)


def play_snes_period(job):
    """Play one recorded native SNES period; controllers act on each emulator frame."""
    from nhl94_ai.agents.classic_v1 import ClassicAIV1Model
    from nhl94_ai.agents.control_rate import ControlRateLimit
    from nhl94_ai.evaluation.benchmark import away_view

    human = job.get('human', False)
    side = 1 if job['side'] == 'home' else 2
    options = dict(job.get('classic_options') or {})
    goalie_policy = options.get('goalie_policy', 'selective')
    recorder = job.get('_recorder')
    if recorder is not None and recorder.settings.get('goalie_policy', 'selective') != goalie_policy:
        raise ValueError('Recorder settings and match goalie policy differ')
    frame_skip = job.get('frame_skip', 4)
    interval = job.get('classic_control_interval', options.get('classic_control_interval', 0))
    side_options = {name: dict(options, **overrides)
                    for name, overrides in (job.get('classic_options_by_side') or {}).items()}
    classic_args = SimpleNamespace(**{'env': GAME, 'action_type': 'FILTERED', **options})
    env = make_retro(game=GAME, num_players=2, goalie_policy=goalie_policy)
    viewer = None
    try:
        if tuple(env.buttons) != BACKEND.buttons:
            raise ValueError('Unexpected native SNES button ordering')
        env.reset()
        BACKEND.configure_controllers(env, 'home', 2)
        env.data.update_ram()
        if job.get('seed') is not None:
            seed_rng(env, job['seed'])
        # One neutral frame lets the ROM apply the player-select tables, as in
        # the observation wrapper's reset. It precedes the recorded timeline.
        *_, info = env.step(np.zeros(24, dtype=np.int8))
        info = BACKEND.decode_info(info)
        if (info.get('defense_team1'), info.get('defense_team2')) != (1, 2):
            raise ValueError('SNES recording requires controllers on opposing teams')
        if (info['period'], info['p1_score'], info['p2_score']) != (0, 0, 0):
            raise ValueError('SNES recording requires a scoreless first-period save')
        classics, agents = {}, {}
        for team, name in ((1, 'home'), (2, 'away')):
            if human and team == side:
                classics[name] = None
                continue
            model = ClassicAIV1Model(classic_args if name not in side_options else SimpleNamespace(
                **{'env': GAME, 'action_type': 'FILTERED', **side_options[name]}))
            classics[name] = model
            agents[team] = ControlRateLimit(model, interval) if interval else model
        if human:
            from nhl94_ai.evaluation.external import MatchViewer
            viewer = MatchViewer(env, side, 0, info, human=True,
                                 no_gamepad=job.get('no_gamepad', False),
                                 opponent_name='Classic AI')
        if recorder is not None:
            recorder.start(env, info)
        state = NHL94GameState(5, platform=BACKEND)
        frames = 0
        user_quit = False
        budget = int(info['time']) * 120 + 6000  # Also bounds stoppages with a frozen clock.
        max_frames = job.get('max_frames')
        while info['time'] > 0 and frames < budget:
            if max_frames is not None and frames >= max_frames:
                break
            if viewer is not None and not viewer.before_frame():
                user_quit = True
                break
            state.BeginFrame(info, [0] * 6)
            views = (state, away_view(state))
            pads = {}
            for team, agent in agents.items():
                pads[team] = np.asarray(
                    agent.predict_frame(views[team - 1], frame_skip=frame_skip)[0], dtype=np.int8)
            if human:
                pads[side] = BACKEND.encode_action(np.asarray(viewer.human_action(), dtype=np.int8))
            buttons = np.concatenate((pads[1], pads[2]))
            if recorder is not None:
                recorder.before_step(env, info, classics)
            *_, info = env.step(buttons)
            info = BACKEND.decode_info(info)
            if recorder is not None:
                recorder.after_step(env, buttons, info)
            frames += 1
            if (info.get('defense_team1'), info.get('defense_team2')) != (1, 2):
                raise RuntimeError('Controller assignment changed during a recorded SNES period')
            if viewer is not None:
                viewer.show(env, info)
        completed = info['time'] <= 0
        for model in classics.values():
            if model is not None:
                model.end_one_timer('period-ended' if completed else 'trial-incomplete')
        import hashlib
        return {'side': job['side'], 'human': human, 'frames': frames,
                'completed': completed, 'user_quit': user_quit, **scoreboard(info),
                'final_ram_sha256': hashlib.sha256(env.get_ram().tobytes()).hexdigest(),
                'goalie_metrics': {name: (dict(model.goalie.metrics)
                                          if model is not None and model.goalie is not None else None)
                                   for name, model in classics.items()}}
    finally:
        try:
            if viewer is not None:
                viewer.close()
        finally:
            env.close()


def summarize(rows):
    completed = [row for row in rows if row['completed']]
    summary = {'completed_periods': len(completed), 'incomplete_periods': len(rows) - len(completed)}
    if rows and rows[0].get('human_side'):
        margins = [row['goals'][row['human_side'] - 1] - row['goals'][2 - row['human_side']]
                   for row in completed]
        summary.update(human_wins=sum(value > 0 for value in margins),
                       classic_wins=sum(value < 0 for value in margins), draws=margins.count(0))
    else:
        margins = [row['goals'][0] - row['goals'][1] for row in completed]
        summary.update(home_wins=sum(value > 0 for value in margins),
                       away_wins=sum(value < 0 for value in margins), draws=margins.count(0))
    return summary


def run(args):
    if args.periods < 1 or args.snapshot_interval < 1 or args.frame_skip < 1:
        raise ValueError('Use positive periods/frame-skip/snapshot-interval')
    options = classic_options(args)
    path = Path(args.output).expanduser().resolve()
    path.mkdir(parents=True, exist_ok=False)
    rows = []
    report = {'schema': 'nhl94-snes-session-v1', 'game': GAME, 'settings': vars(args),
              'classic_options': options, 'matches': rows, 'summary': summarize(rows),
              'status': 'recording'}
    write_json(path / 'session.json', report)
    try:
        for index in range(args.periods):
            name = f'period-{index + 1:03d}'
            job = {'side': args.side, 'human': args.human, 'no_gamepad': args.no_gamepad,
                   'frame_skip': args.frame_skip, 'goalie_policy': options['goalie_policy'],
                   'classic_control_interval': options.get('classic_control_interval', 0),
                   'classic_options': options}
            if args.seed is not None:
                job['seed'] = args.seed + index
            recorder = SnesRecording(path / name, job, snapshot_interval=args.snapshot_interval)
            print(f'Period {index + 1}/{args.periods}: {path / name}', flush=True)
            try:
                result = play_snes_period({**job, '_recorder': recorder})
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

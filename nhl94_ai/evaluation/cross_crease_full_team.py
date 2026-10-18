"""Matched, short attacking possessions with all twelve native players active."""
import argparse
from collections import Counter
from dataclasses import asdict
import gzip
import hashlib
import json
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from nhl94_ai.agents.classic_v1 import ClassicAIV1Model
from nhl94_ai.agents.cross_crease_only import CrossCreaseOnly, CrossingSettings
from nhl94_ai.agents.cross_crease_roaming import RoamingCrossCrease
from nhl94_ai.agents.defense import controlled_slot
from nhl94_ai.env.factory import make_retro
from nhl94_ai.evaluation.cpu_benchmark import CPU_RAM, cpu_view, lineup, select_side
from nhl94_ai.game.constants import GameConsts as Buttons
from nhl94_ai.game.ram import pass_geometry_info, restore_away_control, validate_rom_seed
from nhl94_ai.game.state import NHL94GameState


MATCHUPS = {
    'penguins-senators': ('PenguinsVsSenators.start', 1),
    'senators-penguins': ('PenguinsVsSenators.start', 2),
    'canadiens-nordiques': ('CanadiensVsNordiques.start', 1),
    'nordiques-canadiens': ('CanadiensVsNordiques.start', 2),
}
DEPTHS = (100, 160, 210)


def classic():
    return ClassicAIV1Model(SimpleNamespace(action_type='FILTERED', cross_crease=True))


POLICIES = {'only': CrossCreaseOnly, 'roam': RoamingCrossCrease,
            'roam-v1': partial(RoamingCrossCrease, select_release=False, prepare_release=False, traffic_aware=False),
            'roam-v2': partial(RoamingCrossCrease, traffic_aware=False),
            'roam-traffic': partial(RoamingCrossCrease, traffic_aware=True),
            'roam-release-only': partial(RoamingCrossCrease, prepare_release=False, traffic_aware=False),
            'classic': classic}


def actors(view):
    return (*view.team1.players, view.team1.goalie, *view.team2.players, view.team2.goalie)


def full_lineup(view):
    """Unknown roles, benched skaters, and parked fixtures are not full teams."""
    return (len(actors(view)) == 12
            and all(p.role is not None and p.role >= 0 and abs(p.x) < 160 and abs(p.y) < 310
                    for p in actors(view)))


def read_view(env, state, info, side):
    info = pass_geometry_info(env, info)
    if (info['bench_team1'], info['bench_team2']) != (side, 0):
        raise AssertionError('Opponent must be controlled by the native CPU.')
    view = cpu_view(state, info, side)
    if view.is_shootout_active:
        raise AssertionError('Shootouts are not full-team hockey.')
    if view.engine.goalie_modes != (1, 1):
        raise AssertionError('Both goalies must use the native automatic setting.')
    return view


def advance(env, action, side):
    *_, info = env.step(action)
    if side == 2:
        info = restore_away_control(env.data, info, controller_prefix='bench', player_prefix='cpu', slots=6)
    return info


def eligible_start(view, previous, model, depth):
    owner = view.engine.puck_owner
    player = view.team1.get_player_by_scnum(owner)
    sign = 1 if view.team2.net.y > 0 else -1
    return (full_lineup(view) and not view.engine.clock_stopped and not view.is_shooting
            and player is not None and player is not view.team1.goalie and not player.is_falling
            and controlled_slot(view.team1) == owner and depth <= sign * player.y < 240
            and not previous[Buttons.INPUT_B] and not previous[Buttons.INPUT_C]
            and not model.one_timer.pending and not model.offense.pass_action.pending
            and model.cross_crease.plan is None and not model.shot.active(model.scheduler.decisions))


def collect(env, side, *, samples, budget):
    """Take at most one start per uninterrupted team possession, before seeing outcomes."""
    state, model = NHL94GameState(5), classic()
    previous = np.zeros(12, dtype=np.int8)
    info = advance(env, previous, side)
    snapshots = []
    sampled = False
    for frame in range(budget):
        view = read_view(env, state, info, side)
        owner = view.engine.puck_owner
        if view.engine.clock_stopped or view.team2.owns_scnum(owner):
            sampled = False
        depth = DEPTHS[len(snapshots) % len(DEPTHS)]
        if not sampled and eligible_start(view, previous, model, depth):
            snapshots.append({'state': bytes(env.em.get_state()), 'source_frame': frame,
                              'depth_threshold': depth,
                              'ram_sha256': hashlib.sha256(env.get_ram().tobytes()).hexdigest()})
            sampled = True
            if len(snapshots) == samples:
                break
        previous = model.predict_frame(view, 4)[0]
        info = advance(env, previous, side)
        if info['period'] != 0 or info['bench_clock'] == 0:
            break
    return snapshots, {'frames': frame + 1, 'requested': samples, 'collected': len(snapshots),
                       'clock_remaining': info['bench_clock']}


def geometry(view):
    owner = view.engine.puck_owner
    player = view.team1.get_player_by_scnum(owner)
    return {'owner': owner, 'control': controlled_slot(view.team1),
            'player': (player.precise_x, player.precise_y) if player else None,
            'velocity': (player.motion_x, player.motion_y) if player else None,
            'puck': (view.puck.x, view.puck.y), 'goalie': (view.team2.goalie.x, view.team2.goalie.y)}


def play(env, sample, side, *, policy, frames, trace=False, video=None):
    env.em.set_state(sample['state'])
    env.data.update_ram()
    if hashlib.sha256(env.get_ram().tobytes()).hexdigest() != sample['ram_sha256']:
        raise AssertionError('Snapshot did not restore exactly.')
    info, state = env.data.lookup_all(), NHL94GameState(5)
    model = POLICIES[policy]()
    score, conceded = info[f'p{side}_score'], info[f'p{3-side}_score']
    shots = info[f'bench_shots{side}']
    timeline, presses, releases, attempts = [], [], [], []
    phases, digest = Counter(), hashlib.sha256()
    previous_c, previous_shooting, shooter, carrier = False, False, None, None
    previous_release = None
    windups, lineup_frames = 0, 0
    outcome, writer = 'timeout', None
    initial, last_positions, travel = None, None, np.zeros(12)
    if video:
        import cv2
        image = env.render()
        Path(video).parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*'mp4v'), 60,
                                 (image.shape[1], image.shape[0]))
        if not writer.isOpened():
            raise RuntimeError(f'Cannot open video: {video}')
    try:
        for frame in range(frames + 1):
            view = read_view(env, state, info, side)
            owner, geo = view.engine.puck_owner, geometry(view)
            positions = np.array([(p.precise_x, p.precise_y) for p in actors(view)])
            if initial is None:
                if not full_lineup(view):
                    raise AssertionError('Start requires all twelve players on the rink.')
                initial = {**geo, 'positions': positions.tolist(), 'roles': [p.role for p in actors(view)],
                           'goalie_modes': view.engine.goalie_modes, 'net_y': view.team2.net.y}
            if last_positions is not None:
                travel += np.linalg.norm(positions - last_positions, axis=1)
            last_positions = positions
            lineup_frames += full_lineup(view)
            shooting = bool(view.is_shooting and view.team1.owns_scnum(owner))
            if shooting and not previous_shooting:
                windups += 1
                shooter = owner
            # A carried-over shot flag cannot identify a new release by itself.
            release_flag = bool(view.engine.sflags2 & 0x1000)
            fresh_release = previous_release is False and release_flag
            if (shooter is not None and view.engine.shot_player == shooter
                    and (fresh_release
                         or owner < 0 and view.engine.last_puck_player == shooter)):
                skater = view.team1.get_player_by_scnum(shooter)
                releases.append({'frame': frame, 'shooter': shooter,
                                 'shooter_position': (skater.precise_x, skater.precise_y), **geo})
                shooter = None
            if not shooting:
                shooter = None
            previous_release = release_flag
            if view.team1.stats.score > score:
                outcome = 'goal'
            elif view.team2.stats.score > conceded:
                outcome = 'conceded'
            elif not full_lineup(view):
                outcome = 'lineup-changed'
            elif owner == view.team2.goalie_scnum():
                outcome = 'goalie-possession'
            elif view.team2.owns_scnum(owner):
                outcome = 'turnover'
            elif view.engine.clock_stopped:
                outcome = 'stoppage'
            action = np.zeros(12, dtype=np.int8)
            if outcome == 'timeout' and frame < frames:
                if policy != 'classic':
                    if view.team1.owns_scnum(owner) and owner != carrier:
                        # A teammate recovery starts a new attempt, not the previous carrier's windup.
                        attempts.extend(model.attempts)
                        model, carrier = POLICIES[policy](), owner
                    action = model.step(view, frame)
                    decision = model.diagnostics.get('phase', 'idle')
                else:
                    action = model.predict_frame(view, 4)[0]
                    decision = model._last_decision
                phases[decision] += 1
                if action[Buttons.INPUT_C] and not previous_c:
                    presses.append({'frame': frame, **geo})
                digest.update(action.tobytes())
            else:
                decision = outcome
            if trace:
                timeline.append({'frame': frame, **geo, 'shooting': shooting,
                                 'release_flag': release_flag, 'shot_player': view.engine.shot_player,
                                 'positions': positions.tolist(), 'roles': [p.role for p in actors(view)],
                                 'action': np.flatnonzero(action).tolist(), 'decision': decision,
                                 'diagnostics': dict(model.diagnostics) if policy != 'classic' else {}})
            if writer is not None and frame > 0:
                image = env.render().copy()
                cv2.putText(image, f'{policy} {frame} {decision}', (4, 15),
                            cv2.FONT_HERSHEY_SIMPLEX, .4, (255, 255, 0), 1)
                writer.write(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            if outcome != 'timeout' or frame == frames:
                break
            previous_shooting, previous_c = shooting, bool(action[Buttons.INPUT_C])
            info = advance(env, action, side)
    finally:
        if writer is not None:
            writer.release()
    attempts.extend(model.attempts if policy != 'classic' else model.cross_crease.events)
    return {'policy': policy, 'outcome': outcome, 'goal': outcome == 'goal', 'frames': frame,
            'shots': info[f'bench_shots{side}'] - shots, 'windups': windups,
            'presses': presses, 'releases': releases, 'attempts': attempts, 'phases': dict(phases),
            'initial': initial, 'actor_travel': travel.tolist(), 'full_lineup_observations': lineup_frames,
            'snapshot_sha256': hashlib.sha256(sample['state']).hexdigest(),
            'ram_sha256': sample['ram_sha256'], 'actions_sha256': digest.hexdigest(), 'timeline': timeline}


def summary(rows):
    return {'trials': len(rows), 'goals': sum(r['goal'] for r in rows), 'shots': sum(r['shots'] for r in rows),
            'with_press': sum(bool(r['presses']) for r in rows),
            'with_release': sum(bool(r['releases']) for r in rows),
            'outcomes': dict(Counter(r['outcome'] for r in rows))}


def source_hashes():
    root = Path(__file__).resolve().parents[2]
    # Include every runtime Python source, including Classic's many dependencies.
    return {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted((root / 'nhl94_ai').rglob('*.py'))}


def run(args):
    validate_rom_seed(args.seed)
    if min(args.seeds, args.samples, args.collect_frames, args.frames) < 1 or args.seed + args.seeds > 2**32:
        raise ValueError('Use positive counts and uint32 seeds.')
    sources, rows, collections, rom_hashes = source_hashes(), [], [], set()
    for matchup in args.matchups:
        name, side = MATCHUPS[matchup]
        env = make_retro(game='NHL94-Genesis-v0', state=name, num_players=1, goalie_policy='off')
        try:
            for key, (address, kind) in CPU_RAM.items():
                env.data.set_variable(key, {'address': address, 'type': kind})
            rom = Path(env.pass_geometry_rom).read_bytes()
            rom_hashes.add(hashlib.sha256(rom).hexdigest())
            for seed in range(args.seed, args.seed + args.seeds):
                env.reset(seed=seed)
                env.data.update_ram()
                select_side(env.data, env.data.lookup_all(), side)
                env.data.set_value('bench_rng', seed)
                samples, collection = collect(env, side, samples=args.samples, budget=args.collect_frames)
                collection.update(matchup=matchup, seed=seed, lineup=lineup(env.data.lookup_all(), rom))
                collections.append(collection)
                for index, sample in enumerate(samples):
                    trial_id = f'{matchup}-{seed}-{index}'
                    for policy in args.policies:
                        video = Path(args.video_dir) / f'{trial_id}-{policy}.mp4' if args.video_dir else None
                        row = play(env, sample, side, policy=policy, frames=args.frames, trace=args.trace, video=video)
                        row.update(id=trial_id, matchup=matchup, seed=seed, side=side,
                                   source_frame=sample['source_frame'], depth_threshold=sample['depth_threshold'])
                        rows.append(row)
                print(json.dumps({'collection': collection, 'results': {
                    policy: summary([r for r in rows if r['matchup'] == matchup and r['seed'] == seed
                                     and r['policy'] == policy]) for policy in args.policies}}), flush=True)
        finally:
            env.close()
    if sources != source_hashes():
        raise RuntimeError('Runtime source changed during experiment.')
    report = {'protocol': 'cross-crease-full-team-v1', 'settings': vars(args), 'sources': sources,
              'rom_sha256': sorted(rom_hashes), 'controller': asdict(CrossingSettings()),
              'collections': collections, 'trials': rows,
              'summary': {p: summary([r for r in rows if r['policy'] == p]) for p in args.policies},
              'limitations': ['Short attacking possessions, not complete periods or matches.',
                              'Starts collected by Classic; both branch controllers start with fresh memory.',
                              'One start per team possession, cycling minimum attack depths 100/160/210.',
                              'Native rosters, movement, opponents and goalies; no placement or rating edits.',
                              'Specialist resets its attempt when a different teammate recovers the puck.',
                              'Opponent possession ends an episode; this does not evaluate defensive play.']}
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = (json.dumps(report, separators=(',', ':')) + '\n').encode()
        path.write_bytes(gzip.compress(data, mtime=0) if path.suffix == '.gz' else data)
    print(json.dumps(report['summary'], indent=2))
    return report


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, default=101000)
    parser.add_argument('--seeds', type=int, default=2)
    parser.add_argument('--matchups', nargs='+', choices=MATCHUPS, default=list(MATCHUPS))
    parser.add_argument('--samples', type=int, default=6)
    parser.add_argument('--collect-frames', type=int, default=18000)
    parser.add_argument('--frames', type=int, default=600)
    parser.add_argument('--policies', nargs='+', choices=POLICIES, default=['only', 'classic'])
    parser.add_argument('--trace', action='store_true')
    parser.add_argument('--video-dir')
    parser.add_argument('--output')
    return parser


if __name__ == '__main__':
    run(build_parser().parse_args())

"""From-rest ordinary-ROM experiments for the cross-crease-only controller."""
import argparse
from collections import Counter
from dataclasses import asdict
import gzip
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from nhl94_ai.agents.classic_v1 import ClassicAIV1Model
from nhl94_ai.agents.cross_crease_only import CrossCreaseOnly, CrossingSettings
from nhl94_ai.env.factory import make_retro
from nhl94_ai.evaluation.benchmark import RAM, away_view, update_state
from nhl94_ai.evaluation.cross_crease_probe import _validate_isolation, snapshot_start
from nhl94_ai.game.constants import GameConsts as Buttons
from nhl94_ai.game.ram import GOALIE_INPUT_ATTRIBUTES, restore_away_control, validate_rom_seed
from nhl94_ai.game.state import NHL94GameState
from nhl94_ai.tasks.cross_crease_setup import (
    GOALIE_LEVELS, OBJECT_BASE, OBJECT_STRIDE, place_crossing_traffic, set_crossing_velocity, set_goalie_level,
)
from nhl94_ai.tasks.defense_setup import _place_object, _rebuild_object_order


STARTS = {'close': (70, 216), 'wide': (100, 180), 'deep': (100, 120), 'middle': (22, 160)}


def play_trial(env, snapshot, slots, *, settings, policy, frames, trace=False, video=None):
    env.em.set_state(snapshot)
    env.data.update_ram()
    info = env.data.lookup_all()
    carrier, goalie, inactive = slots
    side = 1 + (carrier >= 6)
    model = (CrossCreaseOnly(settings) if policy == 'only' else
             ClassicAIV1Model(SimpleNamespace(action_type='FILTERED', cross_crease=True)))
    state = NHL94GameState(5)
    initial_score, initial_shots = info[f'p{side}_score'], info[f'bench_shots{side}']
    releases, presses, timeline = [], [], []
    windups = 0
    previous_shooting = previous_c = False
    pending_windup = False
    outcome = 'timeout'
    digest = hashlib.sha256()
    initial = None
    contacts = 0
    previous_impacts = (0, 0)
    inactive_positions = [tuple(env.data.memory.extract(OBJECT_BASE + slot * OBJECT_STRIDE + offset, '>i4')
                                for offset in (0, 0x14)) for slot in inactive]
    writer = None
    if video:
        import cv2
        image = env.render()
        Path(video).parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*'mp4v'), 60, (image.shape[1], image.shape[0]))
        if not writer.isOpened():
            raise RuntimeError(f'Cannot open video: {video}')
    try:
        for frame in range(frames + 1):
            update_state(state, info, env)
            view = state if side == 1 else away_view(state)
            _validate_isolation(env, info, carrier, goalie, inactive, inactive_positions)
            player, keeper = view.team1.get_player_by_scnum(carrier), view.team2.goalie
            owner = view.engine.puck_owner
            if owner in inactive:
                raise AssertionError('An inactive actor acquired the puck in an isolated trial.')
            geometry = {'player': (player.precise_x, player.precise_y),
                        'velocity': (player.motion_x, player.motion_y),
                        'puck': (view.puck.x, view.puck.y), 'goalie': (keeper.x, keeper.y)}
            if initial is None:
                initial = {**geometry, 'goalie_cpu': True, 'shootout': bool(view.is_shootout_active),
                           'inactive': list(inactive), 'handedness': player.handedness,
                           'goalie_attributes': {name: env.data.memory.extract(
                               OBJECT_BASE + goalie * OBJECT_STRIDE + offset, '|u1')
                               for name, (offset, *_) in GOALIE_INPUT_ATTRIBUTES.items()}}
                if (owner != carrier or view.is_shootout_active or view.team1.defense_control != carrier
                        or velocity_nonzero(geometry['velocity'])):
                    raise AssertionError('Start must be ordinary play with carrier possession.')
            shooting = bool(view.is_shooting and owner == carrier)
            windups += shooting and not previous_shooting
            pending_windup |= shooting
            if (pending_windup and view.engine.shot_player == carrier
                    and (bool(view.engine.sflags2 & 0x1000)
                         or owner < 0 and view.engine.last_puck_player == carrier)):
                releases.append({'frame': frame, **geometry})
                pending_windup = False
            impacts = keeper.contact_impact or 0, player.contact_impact or 0
            contacts += bool((keeper.contact_player == carrier and impacts[0] > previous_impacts[0])
                             or (player.contact_player == goalie and impacts[1] > previous_impacts[1]))
            previous_impacts = impacts
            if view.team1.stats.score > initial_score:
                outcome = 'goal'
            elif owner == goalie:
                outcome = 'goalie-possession'
            elif owner >= 0 and view.team2.get_player_by_scnum(owner) is not None:
                outcome = 'turnover'
            elif view.engine.clock_stopped:
                outcome = 'stoppage'
            action = (model.step(view, frame) if policy == 'only' else model.predict_frame(view, 4)[0])
            if action[Buttons.INPUT_C] and not previous_c:
                presses.append({'frame': frame, **geometry})
            digest.update(action.tobytes())
            if trace:
                timeline.append({'frame': frame, **geometry, 'owner': owner, 'shooting': shooting,
                                 'action': np.flatnonzero(action).tolist(),
                                 'decision': model.diagnostics if policy == 'only' else model._last_decision})
            if writer is not None and frame > 0:
                # The restored snapshot's cached image precedes RAM placement;
                # show only frames actually rendered by this trial's execution.
                image = env.render().copy()
                label = f"{policy}  {frame}  {'running' if outcome == 'timeout' else outcome}"
                cv2.putText(image, label, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, .4, (255, 255, 0), 1)
                writer.write(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            if outcome != 'timeout' or frame == frames:
                break
            previous_shooting, previous_c = shooting, bool(action[Buttons.INPUT_C])
            *_, info = env.step(action)
            if side == 2:
                info = restore_away_control(env.data, info, controller_prefix='bench', player_prefix='cpu', slots=6)
    finally:
        if writer is not None:
            writer.release()
    return {'policy': policy, 'outcome': outcome, 'goal': outcome == 'goal', 'frames': frame,
            'shots': view.team1.stats.shots - initial_shots, 'windups': int(windups),
            'releases': releases, 'presses': presses, 'goalie_contacts': contacts, 'initial': initial,
            'snapshot_sha256': hashlib.sha256(snapshot).hexdigest(), 'actions_sha256': digest.hexdigest(),
            'attempts': model.attempts if policy == 'only' else model.cross_crease.events,
            'timeline': timeline}


def velocity_nonzero(values):
    return any(abs(value) > 1e-12 for value in values)


def summary(rows):
    return {'trials': len(rows), 'goals': sum(row['goal'] for row in rows),
            'shots': sum(row['shots'] for row in rows),
            'with_release': sum(bool(row['releases']) for row in rows),
            'with_press': sum(bool(row['presses']) for row in rows),
            'goalie_contacts': sum(row['goalie_contacts'] for row in rows),
            'outcomes': dict(Counter(row['outcome'] for row in rows))}


def source_hashes():
    root = Path(__file__).resolve().parents[2]
    names = ('agents/cross_crease_only.py', 'evaluation/cross_crease_only.py',
             'evaluation/cross_crease_probe.py', 'evaluation/benchmark.py', 'game/genesis_match.py',
             'tasks/cross_crease_setup.py', 'tasks/defense_setup.py',
             'agents/skating.py', 'agents/motion.py', 'agents/defense.py',
             'agents/classic_v1.py', 'agents/cross_crease.py', 'agents/offense.py',
             'agents/passing.py', 'agents/goalie.py', 'env/target_control.py',
             'env/factory.py', 'game/state.py', 'game/ram.py')
    return {'nhl94_ai/' + name: hashlib.sha256((root / 'nhl94_ai' / name).read_bytes()).hexdigest()
            for name in names}


def run(args):
    validate_rom_seed(args.seed)
    if args.seeds < 1 or args.seed + args.seeds > 2**32 or args.frames < 1 or not 0 <= args.jitter <= 6:
        raise ValueError('Use a positive frame/seed count, uint32 seeds and jitter 0..6.')
    settings = CrossingSettings(args.gap, args.width, args.release_x, args.hold, args.depth_weight)
    sources = source_hashes()
    env = make_retro(game='NHL94-Genesis-v0', state='PenguinsVsSenators.DefenseZone', num_players=1)
    rom_sha256 = hashlib.sha256(Path(env.pass_geometry_rom).read_bytes()).hexdigest()
    rows = []
    try:
        for name, (address, kind) in RAM.items():
            env.data.set_variable(name, {'address': address, 'type': kind})
        for seed in range(args.seed, args.seed + args.seeds):
            for side in args.sides:
                for direction in args.directions:
                    for start in args.starts:
                        for rating in args.goalies:
                            _, slots = snapshot_start(env, scenario='rest', seed=seed, side=side,
                                                      direction=direction, handedness=seed % 2)
                            rng = np.random.default_rng(seed)
                            dx, dy = rng.uniform(-args.jitter, args.jitter, 2)
                            width, depth = STARTS[start]
                            set_crossing_velocity(env, slots[0], direction=direction, lateral=0, goalward=0,
                                                  width=max(20, width + dx), depth=depth + dy)
                            set_goalie_level(env, slots[1], GOALIE_LEVELS[rating])
                            if args.defender:
                                traffic = place_crossing_traffic(env, slots[0], slots[1],
                                                                 kind='pursuit-between', direction=direction)
                                slots = slots[0], slots[1], tuple(slot for slot in slots[2] if slot != traffic.slot)
                            # A role of -1 prevents skating but does not prevent a
                            # loose puck from touching the actor. Long recovery
                            # episodes therefore park inactive bodies off-rink.
                            for index, slot in enumerate(slots[2]):
                                _place_object(env.data.memory, slot, (512 + index * 40, 512))
                            _rebuild_object_order(env.data.memory)
                            snapshot = env.em.get_state()
                            for policy in args.policies:
                                video = (Path(args.video_dir) / f'{seed}-{side}-{direction}-{start}-{rating}-{policy}.mp4'
                                         if args.video_dir else None)
                                row = play_trial(env, snapshot, slots, settings=settings, policy=policy,
                                                 frames=args.frames, trace=args.trace, video=video)
                                row.update(seed=seed, side=side, direction=direction, start=start, goalie=rating)
                                rows.append(row)
    finally:
        env.close()
    if source_hashes() != sources:
        raise RuntimeError('Experiment source changed while the trials were running.')
    report = {'protocol': 'cross-crease-only-from-rest-v1', 'settings': vars(args), 'rom_sha256': rom_sha256,
              'controller': asdict(settings), 'summary': {policy: summary([r for r in rows if r['policy'] == policy])
                                                        for policy in args.policies},
              'by_start': {start: {policy: summary([r for r in rows if r['policy'] == policy and r['start'] == start])
                                   for policy in args.policies} for start in args.starts},
              'limitations': ['Prepared possession starts with a fixed skater rating profile.',
                              'All starts have zero velocity; normal buttons must create the approach.',
                              'Goals count regardless of contact or predefined crossing geometry.',
                              'One goalie and at most one defender; not full-team match strength.'],
              'sources': sources,
              'trials': rows}
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = (json.dumps(report, separators=(',', ':')) + '\n').encode()
        path.write_bytes(gzip.compress(data, mtime=0) if path.suffix == '.gz' else data)
    print(json.dumps({'summary': report['summary'], 'by_start': report['by_start']}, indent=2))
    return report


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, default=94000)
    parser.add_argument('--seeds', type=int, default=2)
    parser.add_argument('--sides', type=int, nargs='+', choices=(1, 2), default=[1, 2])
    parser.add_argument('--directions', type=int, nargs='+', choices=(-1, 1), default=[-1, 1])
    parser.add_argument('--starts', nargs='+', choices=STARTS, default=list(STARTS))
    parser.add_argument('--goalies', nargs='+', choices=GOALIE_LEVELS, default=['high'])
    parser.add_argument('--policies', nargs='+', choices=('only', 'classic'), default=['only', 'classic'])
    parser.add_argument('--frames', type=int, default=600)
    parser.add_argument('--gap', type=float, default=46)
    parser.add_argument('--width', type=float, default=70)
    parser.add_argument('--release-x', type=float, default=8)
    parser.add_argument('--hold', type=int, default=12)
    parser.add_argument('--depth-weight', type=float, default=3)
    parser.add_argument('--jitter', type=float, default=0)
    parser.add_argument('--defender', action='store_true')
    parser.add_argument('--trace', action='store_true')
    parser.add_argument('--video-dir')
    parser.add_argument('--output')
    return parser


if __name__ == '__main__':
    run(build_parser().parse_args())

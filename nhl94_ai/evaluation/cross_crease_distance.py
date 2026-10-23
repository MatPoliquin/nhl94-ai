"""Matched distance/momentum maps and single-player traffic in the ordinary ROM."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
import gzip
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from nhl94_ai.env.factory import make_retro
from nhl94_ai.evaluation.benchmark import RAM
from nhl94_ai.evaluation.cross_crease_probe import ShotTiming, snapshot_start, trial
from nhl94_ai.evaluation.cross_crease_velocity import _sources as velocity_sources, safe_summary
from nhl94_ai.game.constants import GameConsts as Buttons
from nhl94_ai.game.ram import validate_rom_seed
from nhl94_ai.tasks.cross_crease_setup import (
    GOALIE_LEVELS, OBJECT_BASE, OBJECT_STRIDE, TRAFFIC_KINDS, keep_traffic_idle,
    crossing_touch_player, crossing_traffic_position, place_crossing_traffic, set_crossing_velocity,
)


@dataclass(frozen=True)
class AimedShotTiming(ShotTiming):
    height: str = 'middle'

    def __post_init__(self):
        super().__post_init__()
        if self.height not in ('low', 'middle', 'high'):
            raise ValueError('Use low, middle or high human shot-height aim.')

    def action(self, frame, direction):
        action = super().action(frame, direction)
        if frame >= self.press_frame:
            if self.height == 'high':
                action[Buttons.INPUT_UP] = 1
            elif self.height == 'low':
                action[Buttons.INPUT_DOWN] = 1
        return action


def classify_traffic_events(events):
    goalie_frames = [event['frame'] for event in events if event['kind'] == 'goalie-shot-touch']
    first_goalie = min(goalie_frames) if goalie_frames else None
    classified = []
    for event in events:
        event = dict(event)
        if first_goalie is not None and event['frame'] >= first_goalie:
            if event['kind'] in ('shot-block-or-deflection', 'shot-catch'):
                event['kind'] = ('rebound-traffic-touch' if event['kind'] == 'shot-block-or-deflection'
                                 else 'rebound-catch')
                if event['frame'] == first_goalie:
                    event['same_frame_order_unknown'] = True
        classified.append(event)
    return classified


def terminal_cause(row):
    kinds = {event['kind'] for event in row['traffic_events']}
    return (
        'goal' if row['goal'] else 'steal' if 'steal' in kinds else
        'shot-catch' if 'shot-catch' in kinds else
        'goalie-save' if 'goalie-shot-touch' in kinds else
        'traffic-block-or-deflection' if 'shot-block-or-deflection' in kinds else
        'unconfirmed-launch-or-whiff' if 'unconfirmed-launch-or-whiff' in kinds else
        'loss-before-release' if 'loss-before-release' in kinds else row['outcome'])


class CrossingObservations:
    """Record native interventions without inferring blocks from proximity."""

    def __init__(self, env, carrier, goalie, traffic, trace=False):
        self.env, self.carrier, self.goalie, self.traffic = env, carrier, goalie, traffic
        self.trace = trace
        self.timeline, self.events = [], []
        self.previous_owner, self.previous_touch = carrier, carrier
        self.previous_impacts = {}
        self.previous_falling = False
        self.initial_checks = None
        self.recorded_checks = 0
        self.possession_frame = None
        self.minimum_traffic_separation = None
        self.maximum_traffic_displacement = 0.0

    def _event(self, frame, kind, **evidence):
        self.events.append({'frame': frame, 'kind': kind, **evidence})

    def __call__(self, frame, state, release):
        player = state.team1.get_player_by_scnum(self.carrier)
        owner, touch, shooter = state.engine.puck_owner, crossing_touch_player(self.env), state.engine.shot_player
        phase = 'after-release' if release is not None else 'before-release'
        actor = None if self.traffic is None else (
            state.team2 if self.traffic.opponent else state.team1).get_player_by_scnum(self.traffic.slot)
        if actor is not None:
            if actor.role is None or actor.role < 0:
                raise AssertionError('Traffic actor is no longer a live skater.')
            base = OBJECT_BASE + self.traffic.slot * OBJECT_STRIDE
            acquired = owner == self.traffic.slot or self.possession_frame is not None
            if self.env.data.memory.extract(base + 0x62, '|u1') & 8 and (self.traffic.opponent or not acquired):
                raise AssertionError('Traffic actor unexpectedly acquired a human joystick.')
            position = actor.precise_x, actor.precise_y
            separation = math.dist((player.precise_x, player.precise_y), position)
            self.minimum_traffic_separation = (separation if self.minimum_traffic_separation is None
                                               else min(self.minimum_traffic_separation, separation))
            self.maximum_traffic_displacement = max(self.maximum_traffic_displacement,
                                                   math.dist(position, self.traffic.position))
            actors = {self.carrier: player, self.goalie: state.team2.goalie, self.traffic.slot: actor}
            pairs = set()
            for slot, item in actors.items():
                impact = item.contact_impact or 0
                previous = self.previous_impacts.get(slot, 0)
                if (impact > previous and item.contact_player in actors and item.contact_player != slot
                        and self.traffic.slot in (slot, item.contact_player)):
                    pairs.add(tuple(sorted((slot, item.contact_player))))
                self.previous_impacts[slot] = impact
            for pair in sorted(pairs):
                self._event(frame, 'traffic-body-contact', actors=pair, phase=phase)
            if touch == self.traffic.slot and touch != self.previous_touch:
                kind = 'shot-block-or-deflection' if release is not None else (
                    'unresolved-launch-touch' if shooter == self.carrier else 'pre-release-traffic-touch')
                self._event(frame, kind, owner=owner, shooter=shooter)
            if owner == self.traffic.slot and owner != self.previous_owner:
                kind = 'shot-catch' if release is not None else (
                    'unresolved-launch-possession' if shooter == self.carrier else
                    'steal' if self.traffic.opponent else 'friendly-possession-before-release')
                self._event(frame, kind)
                self.possession_frame = frame if self.possession_frame is None else self.possession_frame
            if player.is_falling and not self.previous_falling and player.contact_player == self.traffic.slot:
                self._event(frame, 'traffic-knockdown', phase=phase)
            checks = state.team2.stats.bodychecks
            if self.initial_checks is None:
                self.initial_checks = checks
            fresh_checks = checks - self.initial_checks
            if fresh_checks > self.recorded_checks:
                self._event(frame, 'recorded-opponent-check', count=fresh_checks - self.recorded_checks)
                self.recorded_checks = fresh_checks
            keep_traffic_idle(self.env, self.traffic)
        if owner != self.carrier and self.previous_owner == self.carrier and release is None:
            self._event(frame, 'unconfirmed-launch-or-whiff' if shooter == self.carrier else 'loss-before-release',
                        owner=owner, last_touch=touch)
        if release is not None and touch == self.goalie and touch != self.previous_touch:
            self._event(frame, 'goalie-shot-touch', owner=owner)
        if self.trace:
            assignment = decision_timer = None
            if actor is not None:
                base = OBJECT_BASE + self.traffic.slot * OBJECT_STRIDE
                index = self.env.data.memory.extract(base + 0x36, '>u2')
                if index not in range(8):
                    raise AssertionError('Traffic feedback contains an invalid assignment index.')
                assignment = self.env.data.memory.extract(base + 0x38 + index, '|u1')
                decision_timer = self.env.data.memory.extract(base + 0x40, '|i1')
            self.timeline.append({
                'last_touch': touch, 'shooter': shooter, 'released_latch': state.has_released_shot,
                'carrier_falling': bool(player.is_falling), 'puck_height': state.puck.height,
                'puck_velocity': (state.puck.motion_x, state.puck.motion_y, state.puck.motion_z),
                'traffic': None if actor is None else {
                    'position': (actor.precise_x, actor.precise_y),
                    'velocity': (actor.motion_x, actor.motion_y), 'assignment': assignment,
                    'decision_timer': decision_timer, 'animation': actor.live_anim,
                    'falling': bool(actor.is_falling), 'owner': owner == self.traffic.slot,
                },
            })
        self.previous_owner, self.previous_touch = owner, touch
        self.previous_falling = bool(player.is_falling)
        return self.possession_frame is not None and frame - self.possession_frame >= 8

    def attach(self, row):
        row['traffic_events'] = classify_traffic_events(self.events)
        row['traffic_fixture'] = None if self.traffic is None else asdict(self.traffic)
        row['minimum_traffic_separation'] = self.minimum_traffic_separation
        row['maximum_traffic_displacement'] = self.maximum_traffic_displacement
        row['terminal_cause'] = terminal_cause(row)
        row['separations'] = {}
        sign = 1 if self.carrier < 6 else -1
        for stage, geometry in (('initial', row['initial']), ('charge', row['charge_geometry']),
                                ('release', row['release_geometry'])):
            row['separations'][stage] = None if geometry is None else {
                'body_goalie': math.dist(geometry['player'], geometry['goalie']),
                'puck_goalie': math.dist(geometry['puck'], geometry['goalie']),
                'longitudinal': sign * (geometry['goalie'][1] - geometry['player'][1]),
            }
        if self.trace:
            if len(self.timeline) != len(row['timeline']):
                raise AssertionError('Traffic telemetry does not cover every observed frame.')
            for original, additional in zip(row['timeline'], self.timeline):
                original.update(additional)


def summarize_distance(rows):
    summary = safe_summary(rows)
    events = Counter(event['kind'] for row in rows for event in row['traffic_events'])
    summary['traffic_events'] = dict(events)
    summary['terminal_causes'] = dict(Counter(row['terminal_cause'] for row in rows))
    summary['goalie_save_trials'] = sum(not row['goal'] and any(
        event['kind'] == 'goalie-shot-touch' for event in row['traffic_events']) for row in rows)
    summary['traffic_contact_trials'] = sum(any(event['kind'] == 'traffic-body-contact'
                                               for event in row['traffic_events']) for row in rows)
    summary['contact_free_goals'] = sum(row['goal'] and not row['goalie_contact_impulses']
                                      and not any(event['kind'] == 'traffic-body-contact'
                                                  for event in row['traffic_events']) for row in rows)
    for kind in ('shot-block-or-deflection', 'shot-catch', 'steal', 'traffic-knockdown',
                 'recorded-opponent-check', 'loss-before-release', 'goalie-shot-touch',
                 'unresolved-launch-touch', 'unconfirmed-launch-or-whiff',
                 'rebound-traffic-touch', 'rebound-catch'):
        summary[kind + '_trials'] = sum(any(event['kind'] == kind for event in row['traffic_events']) for row in rows)
    for stage in ('initial', 'charge', 'release'):
        observed = [row['separations'][stage] for row in rows if row['separations'][stage] is not None]
        summary[stage + '_separation'] = {
            'observations': len(observed),
            'mean_body_goalie': sum(item['body_goalie'] for item in observed) / len(observed) if observed else None,
            'mean_longitudinal': sum(item['longitudinal'] for item in observed) / len(observed) if observed else None,
        }
    return summary


def validate_options(args):
    validate_rom_seed(args.seed)
    if args.seeds < 1 or args.seed + args.seeds > 2**32 or args.workers < 1:
        raise ValueError('Use a positive uint32 seed count and positive worker count.')
    for values, low, high, name in ((args.distances, 20, 170, 'distance'), (args.lateral, 0, 3, 'lateral')):
        if not values or len(set(values)) != len(values) or any(
                not math.isfinite(value) or not low <= value <= high for value in values):
            raise ValueError(f'{name} values must be unique finite numbers in {low}..{high}.')
    for name in ('traffic', 'policies', 'goalies', 'heights'):
        values = getattr(args, name)
        if not values or len(set(values)) != len(values):
            raise ValueError(f'{name} selections must be nonempty and unique.')
    if not math.isfinite(args.width) or not 20 <= args.width <= 110:
        raise ValueError('Use a fixed initial crossing width of twenty to 110 units.')
    if not math.isfinite(args.goalward) or not -1 <= args.goalward <= 1:
        raise ValueError('Use fixed goalward momentum from minus one to one rink unit/frame.')
    AimedShotTiming(args.press_frame, args.hold_frames, args.aim)
    for distance in args.distances:
        for kind in args.traffic:
            if kind != 'none':
                crossing_traffic_position((-args.width, 0), (-12, distance),
                                          (-args.width + 10, 0), kind, 1)


def _fixture(options):
    args, side, direction, seed = options
    env = make_retro(game='NHL94-Genesis-v0', state='PenguinsVsSenators.DefenseZone', num_players=1)
    rows = []
    try:
        for name, (address, kind) in RAM.items():
            env.data.set_variable(name, {'address': address, 'type': kind})
        base, slots = snapshot_start(env, scenario='near', seed=seed, side=side,
                                     direction=direction, handedness=seed % 2)
        for distance in args.distances:
            for lateral in args.lateral:
                for kind in args.traffic:
                    env.em.set_state(base)
                    actual = set_crossing_velocity(env, slots[0], direction=direction, lateral=lateral,
                                                   goalward=args.goalward, width=args.width, depth=250 - distance)
                    traffic = place_crossing_traffic(env, slots[0], slots[1], kind=kind, direction=direction)
                    inactive = tuple(slot for slot in slots[2] if traffic is None or slot != traffic.slot)
                    snapshot = env.em.get_state()
                    for rating in args.goalies:
                        ram_hash = None
                        for policy in args.policies:
                            heights = ['policy'] if policy == 'classic' else args.heights
                            for height in heights:
                                observer = CrossingObservations(env, slots[0], slots[1], traffic, args.trace)
                                timing = AimedShotTiming(args.press_frame, args.hold_frames, args.aim,
                                                        height='middle' if height == 'policy' else height)
                                row = trial(env, snapshot, carrier=slots[0], goalie=slots[1], inactive=inactive,
                                            level=GOALIE_LEVELS[rating], direction=direction, policy=policy,
                                            timing=timing, trace=args.trace, observer=observer)
                                observer.attach(row)
                                initial = row['initial']
                                expected = (-direction * args.width, (1 if side == 1 else -1) * (250 - distance))
                                if (not np.allclose(actual, initial['player_velocity'], rtol=0, atol=1e-12)
                                        or not np.allclose(expected, initial['player'], rtol=0, atol=1 / 65536)):
                                    raise AssertionError('Distance/momentum fixture differs from the requested physical encoding.')
                                if not math.isclose(row['separations']['initial']['longitudinal'], distance, abs_tol=1 / 65536):
                                    raise AssertionError('Initial goalie longitudinal separation differs from the distance cell.')
                                if ram_hash is not None and ram_hash != row['initial_ram_sha256']:
                                    raise AssertionError('Distance policies/heights did not restore identical exposed RAM.')
                                ram_hash = row['initial_ram_sha256']
                                row.update(side=side, direction=direction, seed=seed, handedness=seed % 2,
                                           distance=distance, lateral=lateral, traffic=kind, height=height,
                                           goalie_rating=rating)
                                rows.append(row)
    finally:
        env.close()
    return rows


def _sources():
    sources = velocity_sources()
    sources['nhl94_ai/evaluation/cross_crease_distance.py'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return sources


def _rom_hash():
    from stable_retro.data import get_romfile_path
    rom = Path(get_romfile_path('NHL94-Genesis-v0')).read_bytes()
    if rom[0x15464:0x15466] != bytes.fromhex('4e75'):
        raise ValueError('This ROM does not have the verified inert assignment-zero routine.')
    if rom[0x10148:0x1014E] != bytes.fromhex('31ea0052bf7a'):
        raise ValueError('This ROM does not have the verified native last-touch slot.')
    return hashlib.sha256(rom).hexdigest()


def summarize_grid(rows, args):
    return [
        {'distance': distance, 'lateral': lateral, 'traffic': traffic,
         'results': {rating: {policy: {
             height: summarize_distance([row for row in rows if row['distance'] == distance
                                         and row['lateral'] == lateral and row['traffic'] == traffic
                                         and row['goalie_rating'] == rating and row['policy'] == policy
                                         and row['height'] == height])
             for height in (['policy'] if policy == 'classic' else args.heights)}
                             for policy in args.policies} for rating in args.goalies}}
        for distance in args.distances for lateral in args.lateral for traffic in args.traffic
    ]


def _write_report(path, report):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(report, separators=(',', ':')) + '\n').encode('utf-8')
    path.write_bytes(gzip.compress(data, mtime=0) if path.suffix == '.gz' else data)


def reanalyze(paths):
    """Reclassify recorded interventions; preserve emulation provenance and traces."""
    for name in paths:
        path = Path(name)
        data = path.read_bytes()
        report = json.loads(gzip.decompress(data) if path.suffix == '.gz' else data)
        if report.get('protocol') != 'cross-crease-distance-traffic-v1':
            raise ValueError(f'{path} is not a distance/traffic report.')
        report.setdefault('measurement_report_sha256', hashlib.sha256(data).hexdigest())
        for row in report['trials']:
            row['traffic_events'] = classify_traffic_events(row['traffic_events'])
            row['terminal_cause'] = terminal_cause(row)
        report['grid'] = summarize_grid(report['trials'], argparse.Namespace(**report['settings']))
        report['event_analysis'] = 'goalie-before-traffic-v2'
        report['analysis_sources'] = _sources()
        _write_report(path, report)
        print(f'Reclassified {len(report["trials"])} recorded trials in {path}; no emulator frames replayed.')


def run(args):
    validate_options(args)
    sources, rom_hash = _sources(), _rom_hash()
    fixtures = [(args, side, direction, seed) for side in (1, 2) for direction in (-1, 1)
                for seed in range(args.seed, args.seed + args.seeds)]
    if args.workers == 1:
        batches = [_fixture(fixture) for fixture in fixtures]
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            batches = list(pool.map(_fixture, fixtures))
    if _sources() != sources or _rom_hash() != rom_hash:
        raise RuntimeError('Sources or ROM changed during measurement; distance report is not frozen.')
    rows = [row for batch in batches for row in batch]
    grid = summarize_grid(rows, args)
    report = {
        'protocol': 'cross-crease-distance-traffic-v1', 'settings': vars(args), 'sources': sources,
        'rom_sha256': rom_hash, 'grid': grid, 'trials': rows,
        'event_analysis': 'goalie-before-traffic-v2',
        'coordinates': {'distance': 'initial longitudinal body-to-goalie gap, rink units; not Euclidean distance',
                        'lateral': 'incoming rink units/frame toward crossing', 'goalward': args.goalward,
                        'goalie': 'fixed initial depth 250; normal CPU thereafter'},
        'limitations': [
            'One shot opportunity by the original carrier; no fallback after release or possession loss.',
            'Idle traffic retains native collision/impulse physics, but assignment-zero suppresses autonomous AI.',
            'Pursuit uses native assnearest with fixed synthetic ratings, not frozen steering.',
            'Initial-lane traffic starts midway between initial puck and goalie; release-lane traffic starts at crossing X +20.',
            'Between-body traffic starts at the body midpoint, allowing the close lane without initial overlap.',
            'Classic chooses its own shot height and may reposition; compare actual charge/release separation.',
            'Native post-release touches identify blocks/deflections together, not body versus stick subroutines.',
            'Traffic touches/catches after a goalie touch are rebounds, not direct shot interventions.',
            'Unconfirmed same-frame launches/whiffs are explicit and are not inferred successful shots.',
            'No claim of visual screening, universal distance limits, or full-team match strength.',
        ],
    }
    if args.output:
        _write_report(args.output, report)
    for cell in grid:
        print(f"distance={cell['distance']} lateral={cell['lateral']} traffic={cell['traffic']}: "
              + ' '.join(f"{rating}/{policy}/{height}={summary['contact_free_goals']}/{summary['attempts']}"
                         for rating, policies in cell['results'].items()
                         for policy, heights in policies.items() for height, summary in heights.items()))
    return report


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--distances', type=float, nargs='+', default=[20, 34, 50, 70, 90, 120, 150])
    parser.add_argument('--lateral', type=float, nargs='+', default=[1, 1.25, 1.5, 2])
    parser.add_argument('--width', type=float, default=35)
    parser.add_argument('--goalward', type=float, default=0)
    parser.add_argument('--traffic', nargs='+', choices=TRAFFIC_KINDS, default=['none'])
    parser.add_argument('--heights', nargs='+', choices=('low', 'middle', 'high'), default=['middle'])
    parser.add_argument('--policies', nargs='+', choices=('tap', 'held', 'classic'), default=['tap', 'held', 'classic'])
    parser.add_argument('--goalies', nargs='+', choices=GOALIE_LEVELS, default=list(GOALIE_LEVELS))
    parser.add_argument('--seed', type=int, default=86000)
    parser.add_argument('--seeds', type=int, default=4)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--press-frame', type=int, default=0)
    parser.add_argument('--hold-frames', type=int, default=24)
    parser.add_argument('--aim', type=int, choices=(-1, 1), default=1)
    parser.add_argument('--trace', action='store_true')
    parser.add_argument('--output')
    parser.add_argument('--reanalyze', nargs='+', help='Reclassify existing reports without replaying emulator frames')
    return parser


if __name__ == '__main__':
    options = build_parser().parse_args()
    if options.reanalyze:
        reanalyze(options.reanalyze)
    else:
        run(options)

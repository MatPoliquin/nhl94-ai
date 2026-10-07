"""Matched isolated cross-crease attempts against weak and strong CPU goalies."""
import argparse
from collections import Counter
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from nhl94_ai.agents.classic_v1 import ClassicAIV1Model
from nhl94_ai.agents.cross_crease import SEQUENCE_FRAMES
from nhl94_ai.agents.goalie import DIVE_ANIMATION, SAVE_ANIMATIONS
from nhl94_ai.env.factory import make_retro
from nhl94_ai.evaluation.benchmark import RAM, away_view, update_state
from nhl94_ai.game.constants import GameConsts as Buttons
from nhl94_ai.game.ram import GOALIE_INPUT_ATTRIBUTES, restore_away_control, validate_rom_seed
from nhl94_ai.game.state import NHL94GameState
from nhl94_ai.tasks.cross_crease_setup import (
    GOALIE_LEVELS, OBJECT_BASE, OBJECT_STRIDE, prepare_isolated_crossing, set_goalie_level,
)


STARTS = {
    'deep': (35, 200, 6500),
    'near': (35, 216, 6500),
    'slow': (35, 216, 4500),
    'wide': (50, 216, 7500),
    'rest': (100, 180, 0),
    'rest-near': (70, 216, 0),
}
FRAME_BUDGET = SEQUENCE_FRAMES + 64


@dataclass(frozen=True)
class ShotTiming:
    press_frame: int = 0
    hold_frames: int = 24
    aim: int = 1

    def __post_init__(self):
        if not 0 <= self.press_frame <= 24 or not 1 <= self.hold_frames <= 60 or self.aim not in (-1, 1):
            raise ValueError('Use press delay 0..24, hold 1..60 and aim relative to crossing -1/+1.')

    def action(self, frame, direction):
        action = np.zeros(Buttons.INPUT_MAX, dtype=np.int8)
        side = direction if frame < self.press_frame else direction * self.aim
        action[Buttons.INPUT_RIGHT if side > 0 else Buttons.INPUT_LEFT] = 1
        action[Buttons.INPUT_C] = int(self.press_frame <= frame < self.press_frame + self.hold_frames)
        return action


def _validate_isolation(env, info, carrier, goalie, inactive, positions=None):
    memory = env.data.memory
    if any(memory.extract(OBJECT_BASE + slot * OBJECT_STRIDE + 0x34, '>i2') != -1 for slot in inactive):
        raise AssertionError('An isolated actor became active again.')
    if memory.extract(OBJECT_BASE + goalie * OBJECT_STRIDE + 0x62, '|u1') & 8:
        raise AssertionError('The opposing goalie unexpectedly acquired a joystick.')
    if info['bench_team2'] != 0 or info['bench_team1'] != 1 + (carrier >= 6):
        raise AssertionError('The goalie side must remain CPU-controlled.')
    if positions is not None:
        for slot, position in zip(inactive, positions):
            base = OBJECT_BASE + slot * OBJECT_STRIDE
            if tuple(memory.extract(base + offset, '>i4') for offset in (0, 0x14)) != position:
                raise AssertionError(('Inactive actor moved', slot))


def trial(env, snapshot, *, carrier, goalie, inactive, level, direction, policy,
          timing=None, trace=False, observer=None):
    if policy not in ('tap', 'held', 'stationary', 'classic'):
        raise ValueError('Use tap, held, stationary or classic policy.')
    timing = (replace(timing, press_frame=0, hold_frames=4, aim=1) if timing is not None
              else ShotTiming(hold_frames=4)) if policy == 'tap' else timing or ShotTiming()
    env.em.set_state(snapshot)
    attributes = set_goalie_level(env, goalie, level)
    if policy == 'stationary':
        for slot in (carrier, 14):
            for offset in (0x28, 0x2A):
                env.data.memory.assign(OBJECT_BASE + slot * OBJECT_STRIDE + offset, '>i2', 0)
    env.data.update_ram()
    info = env.data.lookup_all()
    initial_ram = hashlib.sha256(env.get_ram().tobytes()).hexdigest()
    inactive_positions = [tuple(env.data.memory.extract(OBJECT_BASE + slot * OBJECT_STRIDE + offset, '>i4')
                                for offset in (0, 0x14)) for slot in inactive]
    side = 1 + (carrier >= 6)
    state = NHL94GameState(5)
    model = ClassicAIV1Model(SimpleNamespace(action_type='FILTERED', cross_crease=True)) if policy == 'classic' else None
    digest = hashlib.sha256()
    timeline = []
    first_c = windup = release = commit = recorded = goal_frame = caught = None
    c_frames = contacts = 0
    impacts = (0, 0)
    released_seen = False
    opportunity_lost = False
    minimum = math.inf
    initial = charge_geometry = release_geometry = None
    initial_score = info[f'p{side}_score']
    initial_shots = info[f'bench_shots{side}']
    for frame in range(FRAME_BUDGET + 1):
        update_state(state, info, env)
        view = state if side == 1 else away_view(state)
        player, keeper = view.team1.get_player_by_scnum(carrier), view.team2.goalie
        _validate_isolation(env, info, carrier, goalie, inactive, inactive_positions)
        if frame == 0:
            if view.is_shootout_active or view.engine.puck_owner != carrier or view.team1.defense_control != carrier:
                raise AssertionError('Fixture must start in ordinary play with controlled carrier possession.')
            actual = {name: env.data.memory.extract(
                OBJECT_BASE + goalie * OBJECT_STRIDE + GOALIE_INPUT_ATTRIBUTES[name][0], '|u1')
                      for name in attributes}
            if actual != attributes:
                raise AssertionError(('Goalie rating feedback mismatch', actual, attributes))
            initial = {'player': (player.precise_x, player.precise_y), 'puck': (view.puck.x, view.puck.y),
                       'goalie': (keeper.precise_x, keeper.precise_y),
                       'player_velocity': (player.motion_x, player.motion_y),
                       'goalie_velocity': (keeper.motion_x, keeper.motion_y),
                       'goalie_attributes': actual, 'carrier': carrier, 'inactive': inactive,
                       'goalie_cpu': True, 'shootout': False}
        minimum = min(minimum, math.dist((player.x, player.y), (keeper.x, keeper.y)))
        fresh = keeper.contact_impact or 0, player.contact_impact or 0
        if ((keeper.contact_player == carrier and fresh[0] > impacts[0])
                or player.contact_player == goalie and fresh[1] > impacts[1]):
            contacts += 1
        impacts = fresh
        if view.is_shooting and view.engine.puck_owner == carrier:
            windup = frame if windup is None else windup
            if commit is None and keeper.live_anim in (*SAVE_ANIMATIONS, DIVE_ANIMATION):
                commit = frame
        fresh_release = view.has_released_shot and not released_seen
        released_seen = view.has_released_shot
        loose_release = view.engine.puck_owner < 0 and view.engine.last_puck_player == carrier
        if (windup is not None and release is None and view.engine.shot_player == carrier
                and (fresh_release or loose_release)):
            release = frame
            release_geometry = {
                'player': (player.precise_x, player.precise_y), 'puck': (view.puck.x, view.puck.y),
                'puck_velocity': (view.puck.motion_x, view.puck.motion_y),
                'goalie': (keeper.precise_x, keeper.precise_y), 'goalie_anim': keeper.live_anim,
            }
        if view.team1.stats.shots > initial_shots and view.engine.shot_player == carrier:
            recorded = frame if recorded is None else recorded
        if (view.team1.stats.score > initial_score and release is not None
                and view.engine.shot_player == carrier):
            goal_frame = frame
        if caught is None and view.engine.puck_owner == goalie:
            caught = frame
        opportunity_lost = opportunity_lost or view.engine.puck_owner != carrier
        observed_stop = observer(frame, view, release) if observer is not None else False
        if opportunity_lost:
            if model is not None:
                model.cross_crease.observe(view, frame + 1)
            action = np.zeros(Buttons.INPUT_MAX, dtype=np.int8)
        elif model is not None:
            action = model.predict_frame(view, 4)[0]
        else:
            action = timing.action(frame, direction)
        if action[Buttons.INPUT_C]:
            if first_c is None:
                first_c = frame
                charge_geometry = {
                    'player': (player.precise_x, player.precise_y),
                    'player_velocity': (player.motion_x, player.motion_y),
                    'puck': (view.puck.x, view.puck.y),
                    'goalie': (keeper.precise_x, keeper.precise_y),
                    'goalie_velocity': (keeper.motion_x, keeper.motion_y),
                }
            c_frames += 1
        digest.update(action.tobytes())
        if trace:
            timeline.append({
                'frame': frame, 'player': (player.x, player.y), 'goalie': (keeper.x, keeper.y),
                'puck': (view.puck.x, view.puck.y), 'goalie_anim': keeper.live_anim,
                'owner': view.engine.puck_owner, 'windup': view.is_shooting,
                'c': bool(action[Buttons.INPUT_C]), 'decision': model._last_decision if model else policy,
            })
        if goal_frame is not None or view.engine.clock_stopped or frame == FRAME_BUDGET or observed_stop:
            break
        if caught is not None and frame - caught >= 8:
            break
        *_, info = env.step(action)
        if side == 2:
            info = restore_away_control(env.data, info, controller_prefix='bench', player_prefix='cpu', slots=6)
    outcome = ('goal' if goal_frame is not None else 'goalie-possession' if caught is not None
               else 'play-stopped' if view.engine.clock_stopped else 'unreleased' if release is None else 'no-goal')
    return {
        'policy': policy, 'level': level, 'timing': None if model else asdict(timing), 'initial': initial,
        'initial_snapshot_sha256': hashlib.sha256(snapshot).hexdigest(), 'initial_ram_sha256': initial_ram,
        'actions_sha256': digest.hexdigest(), 'first_c': first_c, 'c_frames': c_frames,
        'windup_frame': windup, 'release_frame': release, 'recorded_frame': recorded,
        'commit_frame': commit, 'goal_frame': goal_frame, 'charge_geometry': charge_geometry,
        'release_geometry': release_geometry,
        'goalie_possession_frame': caught,
        'minimum_goalie_separation': minimum, 'goalie_contact_impulses': contacts,
        'goal': goal_frame is not None, 'shot': recorded is not None,
        'outcome': outcome, 'frames': frame + 1,
        'controller_metrics': dict(model.cross_crease.metrics) if model else {},
        'controller_diagnostics': dict(model.cross_crease.diagnostics) if model else {},
        'controller_events': model.cross_crease.events if model else [],
        'timeline': timeline,
    }


def snapshot_start(env, *, scenario, seed, side, direction, handedness, jitter=0):
    env.reset(seed=seed)
    width, depth, speed = STARTS[scenario]
    dx, dy, dv = env.np_random.uniform(-jitter, jitter, 3) if jitter else (0, 0, 0)
    slots = prepare_isolated_crossing(
        env, side=side, direction=direction, width=width + dx, depth=depth + dy,
        speed=speed + round(dv * 256) if speed else 0, handedness=handedness)
    *_, info = env.step(np.zeros(Buttons.INPUT_MAX, dtype=np.int8))
    _validate_isolation(env, info, *slots)
    env.data.set_value('bench_rng', seed)
    return env.em.get_state(), slots


def summarize(rows):
    return {
        'attempts': len(rows), 'goals': sum(row['goal'] for row in rows),
        'shots': sum(row['shot'] for row in rows),
        'accepted_windups': sum(row['windup_frame'] is not None for row in rows),
        'releases': sum(row['release_frame'] is not None for row in rows),
        'pre_release_saves': sum(row['commit_frame'] is not None and row['release_frame'] is not None
                                 and row['commit_frame'] < row['release_frame'] for row in rows),
        'goalie_contact_impulses': sum(row['goalie_contact_impulses'] for row in rows),
        'outcomes': dict(Counter(row['outcome'] for row in rows)),
    }


def run(args):
    validate_rom_seed(args.seed)
    if args.seeds < 1 or args.seed + args.seeds > 2**32 or not 0 <= args.jitter <= 6:
        raise ValueError('Use a positive uint32 seed count and position jitter from zero to six.')
    timing = ShotTiming(args.press_frame, args.hold_frames, args.aim)
    timings = ([ShotTiming(press, hold, aim) for press in (0, 4, 8)
                for hold in (1, 4, 8, 12, 16, 24, 60) for aim in (-1, 1)] if args.sweep else [timing])
    env = make_retro(game='NHL94-Genesis-v0', state='PenguinsVsSenators.DefenseZone', num_players=1)
    rows = []
    try:
        for name, (address, kind) in RAM.items():
            env.data.set_variable(name, {'address': address, 'type': kind})
        for scenario in args.scenarios:
            for side in (1, 2):
                for direction in (-1, 1):
                    for seed in range(args.seed, args.seed + args.seeds):
                        handedness = seed % 2
                        snapshot, slots = snapshot_start(env, scenario=scenario, seed=seed, side=side,
                                                         direction=direction, handedness=handedness, jitter=args.jitter)
                        for rating in args.goalies:
                            for policy in args.policies:
                                for candidate in timings if policy == 'held' else [timing]:
                                    row = trial(env, snapshot, carrier=slots[0], goalie=slots[1], inactive=slots[2],
                                                level=GOALIE_LEVELS[rating], direction=direction, policy=policy,
                                                timing=candidate, trace=args.trace)
                                    row.update(scenario=scenario, side=side, direction=direction,
                                               seed=seed, handedness=handedness, goalie_rating=rating)
                                    rows.append(row)
    finally:
        env.close()
    summary = {rating: {policy: summarize([row for row in rows if row['goalie_rating'] == rating
                                          and row['policy'] == policy]) for policy in args.policies}
               for rating in args.goalies}
    timing_summary = [
        {'timing': asdict(candidate), **{rating: summarize(
            [row for row in rows if row['goalie_rating'] == rating and row['policy'] == 'held'
             and row['timing'] == asdict(candidate)]) for rating in args.goalies}}
        for candidate in timings
    ] if 'held' in args.policies else []
    root = Path(__file__).resolve().parents[2]
    sources = {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
               for path in (Path(__file__), root / 'nhl94_ai/tasks/cross_crease_setup.py',
                            root / 'nhl94_ai/tasks/defense_setup.py', root / 'nhl94_ai/game/ram.py',
                            root / 'nhl94_ai/game/state.py', root / 'nhl94_ai/agents/cross_crease.py',
                            root / 'nhl94_ai/agents/classic_v1.py', root / 'nhl94_ai/agents/goalie.py',
                            root / 'nhl94_ai/agents/offense.py', root / 'nhl94_ai/agents/motion.py',
                            root / 'nhl94_ai/agents/passing.py', root / 'nhl94_ai/agents/defense.py',
                            root / 'nhl94_ai/env/factory.py', root / 'nhl94_ai/env/target_control.py',
                            root / 'nhl94_ai/evaluation/benchmark.py')}
    report = {'protocol': 'isolated-cross-crease-v1', 'settings': vars(args), 'sources': sources,
              'limitations': ['Prepared starts and two isolated from-rest approaches, not match strength.',
                              'Other ten player objects are inactive; goalie remains ordinary ROM CPU.',
                              'Ratings are controlled neutral-hot/cold gameplay bytes, not roster menu overalls.',
                              'Stationary charged control differs only in initial carrier/puck velocity.',
                              'One shot opportunity per trial; no fallback play after release.',
                              'Fixed horizon can censor long rebounds.'],
              'summary': summary, 'timings': timing_summary, 'trials': rows}
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, separators=(',', ':')) + '\n', encoding='utf-8')
    print(json.dumps(timing_summary if args.sweep else summary, indent=2))
    return report


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scenarios', nargs='+', choices=STARTS, default=list(STARTS))
    parser.add_argument('--goalies', nargs='+', choices=GOALIE_LEVELS, default=list(GOALIE_LEVELS))
    parser.add_argument('--policies', nargs='+', choices=('tap', 'held', 'stationary', 'classic'),
                        default=['tap', 'held', 'stationary', 'classic'])
    parser.add_argument('--seed', type=int, default=81000)
    parser.add_argument('--seeds', type=int, default=2)
    parser.add_argument('--jitter', type=float, default=0)
    parser.add_argument('--press-frame', type=int, default=0)
    parser.add_argument('--hold-frames', type=int, default=24)
    parser.add_argument('--aim', type=int, choices=(-1, 1), default=1)
    parser.add_argument('--trace', action='store_true')
    parser.add_argument('--sweep', action='store_true', help='Evaluate 42 press/hold/aim settings on matched starts')
    parser.add_argument('--output')
    return parser


if __name__ == '__main__':
    run(build_parser().parse_args())

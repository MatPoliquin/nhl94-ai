"""Native, paired movement probes for a short-horizon puck-retention predictor.

Only a single ordinary four-frame carrying decision is replaced. Classic then
resumes, including defensive feedback and pass/shot lifecycles. All branches
start from complete emulator/RNG and controller history; probes never drive the
collection prefix. These are local outcomes, not playing-strength estimates.
"""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
import gzip
import hashlib
import json
from pathlib import Path
import pickle
from unittest.mock import patch

import numpy as np

from nhl94_ai.agents import classic_v1
from nhl94_ai.agents.retention_features import (
    INTERVAL, PADS, ACTION_NAMES, attack_sign, movement_action, movement_id,
    available, ordinary_movement, features, admissible,
)
from nhl94_ai.env import factory
from nhl94_ai.evaluation import cpu_benchmark
from nhl94_ai.evaluation.carry_outcomes import CarryOutcomes
from nhl94_ai.evaluation.goalie import source_hashes, MATCHUPS
from nhl94_ai.game.ram import pass_geometry_info, restore_away_control
from nhl94_ai.tasks.cross_crease_setup import crossing_touch_player


HORIZON = 24
SCHEMA = 'native-carry-retention-v1'


def digest_ram(env):
    return hashlib.sha256(bytes(env.get_ram())).hexdigest()


class RetentionOutcome:
    """Accidental separation and stable opponent ownership are different labels.

    Confirmed native passes/shots end exposure to carrying loss. Merely pressing
    a button or entering a windup never establishes a successful release.
    """

    def __init__(self, state, goals, touch):
        self.owner = state.engine.puck_owner
        self.initial_passes = state.team1.pass_attempts
        self.native = CarryOutcomes(state, goals, touch)
        self.separated_run = self.opponent_run = 0
        self.first_separation = self.first_turnover = self.release = None
        self.release_kind = None

    def observe(self, frame, state, goals, touch):
        self.native.observe(frame, state, goals, touch)
        owner = state.engine.puck_owner
        if self.release is None and (state.team1.pass_attempts > self.initial_passes
                                     or self.native.shot_events):
            self.release = frame
            self.release_kind = 'pass' if state.team1.pass_attempts > self.initial_passes else 'shot'
        if self.release is None:
            self.separated_run = self.separated_run + 1 if owner != self.owner else 0
            self.opponent_run = self.opponent_run + 1 if state.team2.owns_scnum(owner) else 0
            if self.separated_run >= 4 and self.first_separation is None:
                self.first_separation = frame - 3
            if self.opponent_run >= 4 and self.first_turnover is None:
                self.first_turnover = frame - 3

    def summary(self):
        return dict(loss=int(self.first_separation is not None),
                    turnover=int(self.first_turnover is not None),
                    separation_frame=self.first_separation, turnover_frame=self.first_turnover,
                    release_frame=self.release, release_kind=self.release_kind,
                    **self.native.summary())


def restore(env, sample):
    env.em.set_state(sample['snapshot'])
    env.data.update_ram()
    if digest_ram(env) != sample['ram_sha256']:
        raise RuntimeError('Retention replay failed to restore identical initial RAM')


def branch(env, sample, command, side, predict, *, horizon=HORIZON):
    restore(env, sample)
    cloned = deepcopy({key: sample[key] for key in ('root_state', 'view', 'model')})
    state, view, model = (cloned[key] for key in ('root_state', 'view', 'model'))
    sign, owner = attack_sign(view), view.engine.puck_owner
    initial_y = view.team1.get_player_by_scnum(owner).y
    first = movement_action(command, sign)
    # The actual Classic decision has already advanced all clocks and edges.
    # Only its direction and cached repeat change, preserving remaining=3.
    model.scheduler.action = first[None].copy()
    info = env.data.lookup_all()
    outcomes = RetentionOutcome(view, info[f'p{side}_score'], crossing_touch_player(env))
    trace, action_hash = [], hashlib.sha256()
    progress4 = None
    for elapsed in range(horizon):
        action = first if elapsed == 0 else predict(model, view, frame_skip=INTERVAL)[0]
        action_hash.update(np.asarray(action, dtype=np.int8).tobytes())
        *_, info = env.step(action)
        if side == 2:
            info = restore_away_control(env.data, info, controller_prefix='bench', player_prefix='cpu', slots=6)
        view = cpu_benchmark.cpu_view(state, pass_geometry_info(env, info), side)
        outcomes.observe(elapsed + 1, view, info[f'p{side}_score'], crossing_touch_player(env))
        player = view.team1.get_player_by_scnum(owner)
        progress = sign * (player.y - initial_y)
        trace.append(dict(frame=elapsed + 1, owner=view.engine.puck_owner,
                          carrier_xy=[player.x, player.y], puck_xy=[view.puck.x, view.puck.y],
                          decision=model._last_decision, buttons=np.asarray(action).tolist()))
        if elapsed + 1 == INTERVAL:
            progress4 = progress
        if view.engine.clock_stopped:
            break
    return dict(command=command, action=ACTION_NAMES[command], features=features(sample['view'], command),
                progress4=progress4, progress=progress, final_owner=view.engine.puck_owner,
                label_valid=elapsed + 1 == horizon or outcomes.release is not None or outcomes.first_separation is not None,
                final_ram_sha256=digest_ram(env), actions_sha256=action_hash.hexdigest(),
                trace=trace, **outcomes.summary())


def collect_game(job):
    matchup, seed, output, maximum = job
    original_factory, original_view = factory.make_retro, cpu_benchmark.cpu_view
    original_predict = classic_v1.ClassicAIV1Model.predict_frame
    holder, rows = {}, []
    frame, last_saved = 0, -1000
    state_name, side = cpu_benchmark.MATCHUPS[matchup]

    def make_env(**kwargs):
        holder['env'] = original_factory(**kwargs)
        return holder['env']

    def view_state(state, info, view_side):
        holder['root_state'] = state
        return original_view(state, info, view_side)

    def predict(model, state, *args, **kwargs):
        nonlocal frame, last_saved
        frame += 1
        ready = len(rows) < maximum and frame - last_saved >= HORIZON and available(model, state)
        tick = model.scheduler.decisions
        action = original_predict(model, state, *args, **kwargs)
        if not (ready and model.scheduler.decisions == tick + 1 and ordinary_movement(model, action[0])):
            return action
        env = holder['env']
        sample = deepcopy(dict(root_state=holder['root_state'], view=state, model=model))
        sample.update(snapshot=env.em.get_state(), ram_sha256=digest_ram(env))
        baseline = movement_id(action[0], attack_sign(state))
        native_rows = []
        # Do not let replay refreshes replace the prefix's root-state pointer.
        with patch.object(cpu_benchmark, 'cpu_view', new=original_view):
            for command in range(len(PADS)):
                native_rows.append(branch(env, sample, command, side, original_predict))
            repeated = branch(env, sample, baseline, side, original_predict)
        if repeated != native_rows[baseline]:
            raise RuntimeError('Repeated baseline replay differs after alternative probes')
        restore(env, sample)
        name = f'{matchup}-{seed}-{frame}'
        path = Path(output) / (name + '.pkl.gz')
        with gzip.open(path, 'wb') as stream:
            pickle.dump(sample, stream, protocol=5)
        passes, _ = sample['model'].offense.passes(state, 'position')
        one_timers, _ = sample['model'].offense.passes(state, 'one-timer')
        row = dict(id=name, path=str(path.resolve()), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                   matchup=matchup, seed=seed, frame=frame, state=state_name, side=side,
                   source=f'{state_name}:{seed}', baseline=baseline,
                   initial_ram_sha256=sample['ram_sha256'], repeated_baseline_identical=True,
                   initial_safe_pass=bool(passes), initial_one_timer=bool(one_timers),
                   admitted=admissible(native_rows, baseline), branches=native_rows)
        path.with_suffix(path.suffix + '.json').write_text(json.dumps(
            {key: value for key, value in row.items() if key != 'branches'}, indent=2) + '\n', encoding='utf-8')
        rows.append(row)
        last_saved = frame
        return action

    with patch.object(factory, 'make_retro', side_effect=make_env), \
            patch.object(cpu_benchmark, 'cpu_view', side_effect=view_state), \
            patch.object(classic_v1.ClassicAIV1Model, 'predict_frame', new=predict):
        period = cpu_benchmark.cpu_match(('classic-v1', matchup, seed, 300, INTERVAL, 'FILTERED', 'off'))
    if not period['completed']:
        raise RuntimeError('Retention collection did not complete its source period')
    report = dict(matchup=matchup, seed=seed, starts=rows, period=period)
    (Path(output) / f'{matchup}-{seed}.json').write_text(json.dumps(report) + '\n', encoding='utf-8')
    return report


def summarize(rows):
    counts = Counter(starts=len(rows))
    sources = set()
    for row in rows:
        base = row['branches'][row['baseline']]
        alternatives = [row['branches'][i] for i in row['admitted']]
        counts['classic_losses'] += base['loss']
        counts['classic_turnovers'] += base['turnover']
        counts['classic_releases'] += base['release_frame'] is not None
        counts['any_avoidable_losses'] += base['loss'] and any(x['label_valid'] and not x['loss'] for x in row['branches'])
        avoidable = base['loss'] and any(x['label_valid'] and not x['loss'] and x['progress'] >= max(0, base['progress'] - 2)
                                         for x in alternatives)
        counts['progress_preserving_avoidable_losses'] += avoidable
        if avoidable:
            sources.add(row['source'])
        counts['progress_preserving_avoidable_turnovers'] += base['turnover'] and any(
            x['label_valid'] and not x['loss'] and x['progress'] >= max(0, base['progress'] - 2) for x in alternatives)
        counts['starts_with_safe_pass'] += row['initial_safe_pass']
        counts['starts_with_one_timer'] += row['initial_one_timer']
    return dict(counts, source_games=len({row['source'] for row in rows}),
                sources_with_avoidable_losses=len(sources),
                training_gate=counts['progress_preserving_avoidable_losses'] >= 20 and len(sources) >= 4)


def run(args):
    if min(args.trials, args.maximum, args.workers) < 1 or not 0 <= args.seed < 2**32 - args.trials:
        raise ValueError('Positive counts and uint32 seeds required')
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError('Use a new directory; retention measurements are immutable')
    hashes = source_hashes()
    jobs = [(matchup, seed, str(output), args.maximum)
            for seed in range(args.seed, args.seed + args.trials) for matchup in args.matchups]
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for row in executor.map(collect_game, jobs):
            results.append(row)
            print(row['matchup'], row['seed'], summarize(row['starts']), flush=True)
    if hashes != source_hashes():
        raise RuntimeError('Retention collection sources changed during measurement')
    starts = [row for result in results for row in result['starts']]
    report = dict(schema=SCHEMA, interval=INTERVAL, horizon=HORIZON, options=vars(args),
                  source_hashes=hashes, summary=summarize(starts), games=results)
    (output / 'report.json').write_text(json.dumps(report) + '\n', encoding='utf-8')
    print('COMPLETE', json.dumps(report['summary']), flush=True)
    return report


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--trials', type=int, default=4)
    parser.add_argument('--maximum', type=int, default=32)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--matchups', nargs='+', choices=MATCHUPS, default=list(MATCHUPS))
    parser.add_argument('--output', required=True)
    return parser


if __name__ == '__main__':
    run(build_parser().parse_args())

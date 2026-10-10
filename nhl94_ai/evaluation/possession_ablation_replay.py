"""Matched-state offensive ablations with a common legacy continuation policy.

Sample first spaced disagreements on a complete legacy trajectory, before
observing fork outcomes. Restore ROM/RNG, decoded state and controller history
for each branch. Only its first decision differs; existing action lifecycles
finish normally and later tactical decisions use the unchanged legacy policy.
No ROM or emulator snapshot bytes are written to disk.
"""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from nhl94_ai.agents.classic_v1 import ClassicAIV1Model
from nhl94_ai.agents.defense import controlled_slot
from nhl94_ai.agents.possession_ablation import LegacyValueRanker, PROFILES
from nhl94_ai.evaluation.cpu_benchmark import CPU_RAM, MATCHUPS
from nhl94_ai.evaluation.refinement_replay import advance, fork, read_view, restore
from nhl94_ai.env.factory import make_retro
from nhl94_ai.game.state import NHL94GameState
from nhl94_ai.evaluation.cpu_benchmark import select_side


def configured(history, profile):
    model = deepcopy(history)
    model.possession_ablation = LegacyValueRanker(profile)
    model.offense.rank_pass_risk = model.possession_ablation.relaxed_risk
    model.args = deepcopy(model.args)
    model.args.possession_ablation = profile
    return model


def common_continuation(policy):
    model = deepcopy(policy)
    model.possession_ablation = None
    model.offense.rank_pass_risk = False
    model.args.possession_ablation = None
    return model


def observation(view):
    fields = ('x', 'y', 'precise_x', 'precise_y', 'motion_x', 'motion_y', 'facing',
              'role', 'speed', 'agility', 'weight', 'energy', 'passing', 'stick',
              'shot_power', 'shot_accuracy', 'stick_x', 'stick_y', 'live_anim')
    return {'owner': view.engine.puck_owner, 'controlled': controlled_slot(view.team1),
            'net_y': view.team2.net.y,
            'puck': {name: getattr(view.puck, name) for name in ('x', 'y', 'motion_x', 'motion_y', 'height')},
            'teams': [[{name: getattr(player, name) for name in fields}
                       for player in (*team.players, team.goalie)] for team in (view.team1, view.team2)]}


def signatures(details):
    return [(row['proposal'], row['kind'], row['target'], row['slot'])
            for row in details.get('possession_candidates', ())]


def run_job(job):
    matchup, seed, horizon, limit, spacing, goalie_policy = job
    save, side = MATCHUPS[matchup]
    env = make_retro(game='NHL94-Genesis-v0', state=save, num_players=1, goalie_policy=goalie_policy)
    cases, counts, last_sample = [], Counter(), -spacing
    try:
        for name, (address, kind) in CPU_RAM.items():
            env.data.set_variable(name, {'address': address, 'type': kind})
        env.reset(seed=seed)
        env.data.update_ram()
        select_side(env.data, env.data.lookup_all(), side)
        env.data.set_value('bench_rng', seed)
        env.data.set_value('bench_clock', 300)
        env.step(np.zeros(12, dtype=np.int8))
        model = ClassicAIV1Model(SimpleNamespace(
            action_type='FILTERED', goalie_policy=goalie_policy, possession_ablation='legacy'))
        state, frame, digest = NHL94GameState(5), 1, hashlib.sha256()
        while frame < 42000:
            view, info = read_view(env, state, side)
            if info['bench_clock'] == 0:
                break
            eligible = (model.scheduler.remaining == 0 and not view.engine.clock_stopped
                        and view.engine.puck_owner == controlled_slot(view.team1)
                        and LegacyValueRanker.available(view))
            before = deepcopy(model) if eligible else None
            before_count = model.possession_ablation.metrics['decisions']
            action = model.predict_frame(view, frame_skip=4)[0]
            if before is not None and model.possession_ablation.metrics['decisions'] > before_count:
                counts['evaluated_states'] += 1
                policies, actions = {'legacy': model}, {'legacy': action}
                for profile in PROFILES[1:]:
                    alternate = configured(before, profile)
                    actions[profile] = alternate.predict_frame(view, frame_skip=4)[0]
                    policies[profile] = alternate
                # Continuation credit must never change the proposal pool on the same state/history.
                for left, right in (('legacy', 'rank'), ('rank', 'continuations'), ('risk', 'continuations-risk')):
                    assert signatures(policies[left].possession_ablation.diagnostics) == signatures(
                        policies[right].possession_ablation.diagnostics), (matchup, seed, frame, left, right)
                changed = [profile for profile in PROFILES[1:]
                           if (not np.array_equal(actions[profile], action)
                               or policies[profile]._last_decision != model._last_decision
                               or policies[profile]._last_target != model._last_target)]
                counts.update('disagreement:' + profile for profile in changed)
                if changed and len(cases) < limit and frame - last_sample >= spacing:
                    snapshot = env.em.get_state()
                    ram_digest = hashlib.sha256(env.get_ram().tobytes()).hexdigest()
                    case = {'frame': frame, 'ram_sha256': ram_digest, 'observation': observation(view),
                            'changed': changed, 'branches': {}}
                    for profile in PROFILES:
                        policy = policies[profile]
                        result = fork(env, snapshot, ram_digest, state, side,
                                      common_continuation(policy), actions[profile], horizon)
                        case['branches'][profile] = {
                            'decision': policy._last_decision, 'target': policy._last_target,
                            'buttons': actions[profile].tolist(),
                            'prediction': deepcopy(policy.possession_ablation.diagnostics), 'outcome': result}
                    if not cases:
                        repeated = fork(env, snapshot, ram_digest, state, side,
                                        common_continuation(model), action, horizon)
                        assert repeated == case['branches']['legacy']['outcome'], 'Nondeterministic restored fork'
                    restore(env, snapshot, ram_digest)
                    cases.append(case)
                    last_sample = frame
            digest.update(np.asarray(action, dtype=np.int8).tobytes())
            advance(env, action, side)
            frame += 1
        _, info = read_view(env, state, side)
        if info['bench_clock'] != 0:
            raise RuntimeError('Incomplete baseline replay period')
        return {'matchup': matchup, 'side': side, 'seed': seed, 'frames': frame,
                'completed': True, 'clock_remaining': info['bench_clock'],
                'goals': [info['p1_score'], info['p2_score']],
                'actions_sha256': digest.hexdigest(), 'counts': dict(counts), 'cases': cases}
    finally:
        env.close()


def source_hashes():
    root = Path(__file__).resolve().parents[2]
    paths = {path for directory in ('agents', 'env', 'game', 'tasks')
             for path in (root / 'nhl94_ai' / directory).glob('*.py')}
    paths.update(root / 'nhl94_ai/evaluation' / name for name in (
        'possession_ablation_replay.py', 'refinement_replay.py', 'cpu_benchmark.py',
        'benchmark.py', 'carry_outcomes.py', 'carry_replay.py'))
    return {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(paths)}


def summarize(rows):
    totals = {profile: Counter() for profile in PROFILES}
    for row in rows:
        for case in row['cases']:
            for profile, branch in case['branches'].items():
                outcome = branch['outcome']
                totals[profile].update(cases=1, goals_for=outcome['goals_for'], goals_against=outcome['goals_against'],
                                       shots=outcome['shots'], one_timers=outcome['one_timers'],
                                       ordinary_turnovers=outcome['possession']['ordinary_turnovers'])
    return {profile: dict(counts) for profile, counts in totals.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, default=24961)
    parser.add_argument('--trials', type=int, default=2)
    parser.add_argument('--horizon', type=int, default=180)
    parser.add_argument('--limit', type=int, default=12)
    parser.add_argument('--spacing', type=int, default=120)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--goalie-policy', choices=('off', 'selective'), default='selective')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    if min(args.trials, args.horizon, args.limit, args.spacing, args.workers) < 1 or not 0 <= args.seed < 2**32 - args.trials:
        raise ValueError('Use positive limits and valid uint32 seeds')
    sources = source_hashes()
    jobs = [(matchup, seed, args.horizon, args.limit, args.spacing, args.goalie_policy)
            for matchup in ('sabres-ducks-manual', 'ducks-sabres-manual')
            for seed in range(args.seed, args.seed + args.trials)]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(run_job, jobs))
    if source_hashes() != sources:
        raise RuntimeError('Sources changed during replay; discard results')
    report = {'protocol': 'possession-ablation-common-continuation-v1', 'settings': vars(args),
              'sources': sources, 'matches': rows, 'summary': summarize(rows),
              'limitations': ['First spaced disagreements on baseline trajectories; not an unbiased state sample.',
                              'All branches use legacy tactical policy after their first decision.',
                              'Local finite-horizon outcomes do not establish full-game strength.']}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report['summary'], indent=2))


if __name__ == '__main__':
    main()

"""Fork Ducks possessions at their first default/experimental input divergence."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

from nhl94_ai.agents.classic_v1 import ClassicAIV1Model
from nhl94_ai.agents.defense import controlled_slot, eligible
from nhl94_ai.agents.passing import PassOption, pad_direction, selected_receiver
from nhl94_ai.env.factory import make_retro
from nhl94_ai.env.intents import HOCKEY_INTENT_NOOP
from nhl94_ai.evaluation.cpu_benchmark import CPU_RAM, cpu_view, select_side
from nhl94_ai.evaluation.carry_outcomes import CarryOutcomes
from nhl94_ai.game.constants import GameConsts as Buttons
from nhl94_ai.game.geometry import aim_pass
from nhl94_ai.game.ram import pass_geometry_info, restore_away_control
from nhl94_ai.game.state import NHL94GameState
from nhl94_ai.tasks.cross_crease_setup import crossing_touch_player


DECISION_INTERVAL = 4


def load_reference(runtime):
    """Load an explicitly supplied local snapshot without replacing live modules."""
    names = ('carry', 'offense', *(('possession',) if (
        runtime / 'nhl94_ai/agents/possession.py').is_file() else ()), 'classic_v1')
    saved = {f'nhl94_ai.agents.{name}': sys.modules.get(f'nhl94_ai.agents.{name}') for name in names}
    try:
        for name in names:
            key = f'nhl94_ai.agents.{name}'
            path = runtime / 'nhl94_ai' / 'agents' / f'{name}.py'
            spec = importlib.util.spec_from_file_location(key, path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[key] = module
            spec.loader.exec_module(module)
        return sys.modules['nhl94_ai.agents.classic_v1'].ClassicAIV1Model
    finally:
        for name, module in saved.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


def one_timer_options(model, state):
    options, _ = model.offense.passes(state, 'one-timer')
    return [{'slot': option.slot, 'value': option.shot_value, 'margin': option.margin} for option in options]


def policy_from_history(history, model_class, *, experimental):
    policy = model_class(SimpleNamespace(action_type='FILTERED', uncertain_carry=experimental))
    offense = policy.offense
    policy.__dict__.update(deepcopy(history.__dict__))
    offense.__dict__.update(deepcopy(history.offense.__dict__))
    if hasattr(offense, 'allow_uncertified'):
        offense.allow_uncertified = experimental
    offense.risk_cache.clear()
    policy.offense = offense
    return policy


def sampling_strata(frame, last_one_timer_frame, options, passes, decisions, *, divergent=True):
    result = {'general'} if divergent else set()
    if passes:
        result.add('safe-pass')
    if options or divergent and (
            frame - last_one_timer_frame <= 32 or any('one-timer' in item for item in decisions)):
        result.add('one-timer')
    return result


def forced_policy(history, state):
    """Advance the same first-frame bookkeeping without inventing an ordinary plan."""
    model = deepcopy(history)
    model._observe_ordinary_pass(state)
    model._tick += 1
    model.defense.idle(1)
    model.defense_diagnostics = model.offense_diagnostics = {}
    model._was_defending = False
    model._defense_elapsed = DECISION_INTERVAL
    return model


def frame_action(model, state):
    return model.predict_frame(state, frame_skip=DECISION_INTERVAL)[0]


def branch(env, sample, policy, action, horizon, *, outcome_ended=False):
    env.em.set_state(sample['snapshot'])
    env.data.update_ram()
    digest = hashlib.sha256(env.get_ram().tobytes()).hexdigest()
    if digest != sample['ram_sha256']:
        raise RuntimeError('Replay did not restore identical RAM.')
    state = deepcopy(sample['state'])
    model = deepcopy(policy)
    model._frame_action = model._encode(np.asarray(action, dtype=np.int8), HOCKEY_INTENT_NOOP)
    model._frame_remaining = DECISION_INTERVAL - 1
    owner_before = sample['owner']
    goals_before = env.data.lookup_all()['p2_score']
    shots_before, attempts_before = state.team2.stats.shots, state.team2.one_timer_attempts
    trace, decisions = [], Counter()
    loss = None
    passes_before = state.team2.pass_attempts
    receptions = []
    launched = None
    view = cpu_view(state, pass_geometry_info(env, env.data.lookup_all()), 2)
    outcomes = CarryOutcomes(view, goals_before, crossing_touch_player(env))
    for elapsed in range(horizon):
        buttons = np.asarray(action if elapsed == 0 else frame_action(model, view), dtype=np.int8)
        decisions[model._last_decision] += 1
        options = one_timer_options(model, view) if view.team1.owns_scnum(view.engine.puck_owner) else []
        outcomes.record_options(options)
        trace.append({'elapsed': elapsed, 'owner': view.engine.puck_owner,
                      'decision': model._last_decision, 'one_timer_options': options})
        *_, info = env.step(buttons)
        info = restore_away_control(env.data, info, controller_prefix='bench', player_prefix='cpu', slots=6)
        if info['p2_pass_attempts'] > passes_before:
            passes_before = info['p2_pass_attempts']
            launched = {'frame': elapsed + 1, 'native_target': info['pass_target'],
                        'passer': view.engine.puck_owner}
            receptions.append(launched)
        if launched is not None and info['puck_owner'] < 0:
            launched['flight_observed'] = True
        if (launched is not None and launched.get('flight_observed')
                and info['puck_owner'] >= 6 and 'actual_receiver' not in launched):
            launched['actual_receiver'], launched['received_frame'] = info['puck_owner'], elapsed + 1
        if info['puck_owner'] >= 0 and info['puck_owner'] < 6 and loss is None:
            loss = elapsed + 1
        env.data.update_ram()
        view = cpu_view(state, pass_geometry_info(env, env.data.lookup_all()), 2)
        ended = outcomes.observe(elapsed + 1, view, info['p2_score'], crossing_touch_player(env))
        if outcome_ended and ended:
            break
    return {
        **outcomes.summary(),
        'initial_ram_sha256': digest, 'initial_owner': owner_before,
        'goals': env.data.lookup_all()['p2_score'] - goals_before,
        'shots': view.team1.stats.shots - shots_before,
        'one_timer_attempts': view.team1.one_timer_attempts - attempts_before,
        'first_opponent_possession_frame': loss, 'decisions': dict(decisions),
        'frames_with_one_timer_option': sum(bool(frame['one_timer_options']) for frame in trace),
        'final_owner': view.engine.puck_owner, 'pass_events': receptions, 'trace': trace,
        'completed_passes': sum(event.get('actual_receiver', -1) >= 6
                                and event['actual_receiver'] != event['passer'] for event in receptions),
    }


def replay_seed(seed, reference_class, *, search_frames, horizon, min_depth,
                outcome_ended=False, strata=('general',)):
    env = make_retro(game='NHL94-Genesis-v0', state='SabresVsMightyDucks.ManualGoalie.Start', num_players=1)
    try:
        for name, (address, kind) in CPU_RAM.items():
            env.data.set_variable(name, {'address': address, 'type': kind})
        env.reset(seed=seed)
        env.data.update_ram()
        select_side(env.data, env.data.lookup_all(), 2)
        env.data.set_value('bench_rng', seed)
        env.data.set_value('bench_clock', 300)
        env.step(np.zeros(12, dtype=np.int8))
        default = ClassicAIV1Model(SimpleNamespace(action_type='FILTERED', uncertain_carry=False))
        state = NHL94GameState(5)
        rows, covered = [], set()
        digest, applied_frames = hashlib.sha256(), 0
        last_one_timer_frame = -100
        for frame in range(search_frames):
            env.data.update_ram()
            info = pass_geometry_info(env, env.data.lookup_all())
            view = cpu_view(state, info, 2)
            owned = view.engine.puck_owner == controlled_slot(view.team1) and view.engine.puck_owner >= 6
            carrier = view.team1.get_player_by_scnum(view.engine.puck_owner)
            sign = 1 if view.team2.net.y > view.team1.net.y else -1
            pending = default.offense.pending
            free_reception = pending is not None and owned and view.engine.puck_owner != pending['passer']
            ready = (bool(strata) and owned and carrier is not view.team1.goalie
                     and carrier.y * sign >= min_depth and default._one_timer is None
                     and (pending is None or free_reception) and default._tick >= default._shot_until
                     and not view.engine.clock_stopped)
            default_before = deepcopy(default) if ready else None
            action = frame_action(default, view)
            candidate = ready and default._tick != default_before._tick
            if candidate:
                old = policy_from_history(default_before, reference_class, experimental=True)
                revised = policy_from_history(default_before, ClassicAIV1Model, experimental=True)
                models = [default_before, deepcopy(old), deepcopy(revised)]
                actions = [action, frame_action(old, view), frame_action(revised, view)]
                initial = one_timer_options(default_before, view)
                if initial:
                    last_one_timer_frame = frame
            divergent = candidate and any(not np.array_equal(actions[0], item) for item in actions[1:])
            selected = []
            if candidate:
                remaining = set(strata) - covered
                decisions = [policy._last_decision for policy in (default, old, revised)]
                eligible_strata = sampling_strata(
                    frame, last_one_timer_frame, initial, [], decisions, divergent=divergent)
                passes, pass_details = (revised.offense.passes(view, 'position')
                                       if 'safe-pass' in remaining or eligible_strata.intersection(remaining)
                                       else ([], []))
                eligible_strata = sampling_strata(
                    frame, last_one_timer_frame, initial, passes, decisions, divergent=divergent)
                selected = sorted(eligible_strata.intersection(remaining))
            if selected:
                sample = {'snapshot': env.em.get_state(), 'state': deepcopy(state),
                          'owner': view.engine.puck_owner,
                          'ram_sha256': hashlib.sha256(env.get_ram().tobytes()).hexdigest()}
                choices = {'default': (default, actions[0]), 'old-experiment': (old, actions[1]),
                           'revised-experiment': (revised, actions[2])}
                unavailable = {}
                carry_model = forced_policy(models[2], view)
                carry_action = np.zeros(12, dtype=np.int8)
                carry_model._steer(carry_action, carrier, *carry_model.offense.carry_target(view, carrier))
                carry_model._last_decision = 'forced-carry'
                choices['carry'] = carry_model, carry_action
                carries = revised.offense.short_carries(view, carrier)
                escape = min((item for item in carries[1:] if item[1]['carry_bounded']
                              and item[1]['carry_clearance'] > 0),
                             key=lambda item: (item[1].get('carry_risk', 1), item[1]['carry_progress']),
                             default=None)
                if escape is not None:
                    escape_model = forced_policy(models[2], view)
                    escape_action = np.zeros(12, dtype=np.int8)
                    escape_model._steer(escape_action, carrier, *escape[0])
                    escape_model._last_decision = 'forced-escape'
                    choices['escape'] = escape_model, escape_action
                else:
                    unavailable['escape'] = 'no geometrically clear bounded alternative'
                forced_pass = False
                if not passes:
                    candidates = [receiver for receiver in view.team1.players
                                  if receiver is not carrier and eligible(receiver)]
                    proposals = []
                    for receiver in candidates:
                        direction = pad_direction(carrier, receiver)
                        slot = selected_receiver(view.team1, view.puck, view.engine.puck_owner, direction)
                        if slot is not None:
                            index = slot - view.team1.skater_scnum_base()
                            actual = view.team1.get_player_by_scnum(slot)
                            proposals.append((sign * (actual.y - carrier.y), index, actual, direction))
                    if proposals:
                        _, index, receiver, direction = max(proposals, key=lambda value: value[0])
                        passes = [PassOption(index, view.team1.skater_scnum_base() + index,
                                             (receiver.x, receiver.y), horizon, direction,
                                             0, 0, 0, 0, 0, 0)]
                        forced_pass = True
                    else:
                        unavailable['pass'] = {'status': 'no-ROM-selected-recipient', 'evaluations': pass_details}
                if passes:
                    pass_model = forced_policy(models[2], view)
                    option = passes[0]
                    pass_model.offense.start_pass(view, option, pass_model.defense.frames, 'position-pass')
                    pass_action = np.zeros(12, dtype=np.int8)
                    aim_pass(pass_action, carrier, view.team1.get_player_by_scnum(option.slot))
                    pass_action[Buttons.INPUT_B] = 1
                    pass_model._last_decision = 'forced-pass'
                    choices['pass'] = pass_model, pass_action
                outcomes = {name: branch(env, sample, model, action, horizon, outcome_ended=outcome_ended)
                            for name, (model, action) in choices.items()}
                rows.append({'seed': seed, 'strata': selected,
                        'capture_frame': frame, 'divergence_frame': frame if divergent else None,
                        'sample_kind': 'input-divergence' if divergent else 'opportunity-control',
                        'initial_owner': sample['owner'],
                        'initial_position': (carrier.x, carrier.y),
                        'initial_one_timer_options': initial,
                        'initial_decisions': {name: model._last_decision for name, (model, _) in choices.items()},
                        'initial_actions': {name: [int(button) for button in action]
                                            for name, (_, action) in choices.items()},
                        'divergent_policies': [name for name, candidate_action in zip(
                            ('old-experiment', 'revised-experiment'), actions[1:])
                                              if not np.array_equal(actions[0], candidate_action)],
                        'initial_ram_sha256': sample['ram_sha256'], 'branches': outcomes,
                        'unavailable_alternatives': unavailable,
                        'pass_model_status': ('forced-despite-rejected-evaluation' if forced_pass else
                                              'modeled-safe' if 'pass' in choices else 'unavailable'),
                        'pass_evaluations': pass_details})
                covered.update(selected)
                if covered.issuperset(strata):
                    break
                env.em.set_state(sample['snapshot'])
                env.data.update_ram()
                if hashlib.sha256(env.get_ram().tobytes()).hexdigest() != sample['ram_sha256']:
                    raise RuntimeError('Replay failed to restore the default-driven search prefix.')
            *_, info = env.step(action)
            digest.update(np.asarray(action, dtype=np.int8).tobytes())
            applied_frames += 1
            restore_away_control(env.data, info, controller_prefix='bench', player_prefix='cpu', slots=6)
            if info['bench_clock'] == 0:
                break
        return {'seed': seed, 'cases': rows, 'missing_strata': sorted(set(strata) - covered),
                'searched_frames': frame + 1,
                'applied_action_frames': applied_frames, 'benchmark_frames': applied_frames + 1,
                'actions_sha256': digest.hexdigest(), 'clock_remaining': info['bench_clock'],
                'status': 'prefix-only' if not strata else 'captured' if rows else
                'no-qualifying-capture-within-search'}
    finally:
        env.close()


def _seed_job(job):
    seed, runtime, settings = job
    return replay_seed(seed, load_reference(Path(runtime).resolve()), **settings)


def source_hashes(runtime=None):
    root = Path(__file__).resolve().parents[1] if runtime is None else Path(runtime) / 'nhl94_ai'
    names = ('carry', 'offense', 'classic_v1', 'passing', 'skating') if runtime is None else (
        'carry', 'offense', 'classic_v1')
    if (root / 'agents/possession.py').is_file():
        names = (*names, 'possession')
    result = {f'nhl94_ai/agents/{name}.py': hashlib.sha256(
        (root / 'agents' / f'{name}.py').read_bytes()).hexdigest() for name in names}
    if runtime is None:
        for name in ('evaluation/carry_replay.py', 'evaluation/carry_outcomes.py', 'evaluation/benchmark.py',
                     'evaluation/cpu_benchmark.py', 'game/ram.py', 'game/state.py', 'tasks/cross_crease_setup.py',
                     'agents/base.py', 'agents/registry.py', 'agents/defense.py', 'agents/motion.py',
                     'env/factory.py', 'env/target_control.py'):
            result[f'nhl94_ai/{name}'] = hashlib.sha256((root / name).read_bytes()).hexdigest()
    return result


def run(args):
    if min(args.seeds, args.search_frames, args.horizon, args.workers) < 1:
        raise ValueError('Replay counts and frame horizons must be positive.')
    if args.seed < 0 or args.seed + args.seeds > 2**32 or not math.isfinite(args.min_depth):
        raise ValueError('Replay requires uint32 ROM seeds and a finite attacking depth.')
    if len(set(args.strata)) != len(args.strata):
        raise ValueError('Choose distinct replay strata.')
    sources, reference_sources = source_hashes(), source_hashes(args.reference_runtime)
    settings = {key: getattr(args, key) for key in ('search_frames', 'horizon', 'min_depth', 'outcome_ended', 'strata')}
    if args.verify_prefix:
        settings['strata'] = ()
    jobs = [(seed, args.reference_runtime, settings) for seed in range(args.seed, args.seed + args.seeds)]
    searches = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for row in executor.map(_seed_job, jobs):
            searches.append(row)
            print(f"Seed {row['seed']}: {len(row['cases'])} captures; missing {row['missing_strata']}", flush=True)
    if sources != source_hashes() or reference_sources != source_hashes(args.reference_runtime):
        raise RuntimeError('Replay sources changed during native measurement.')
    rows = [case for search in searches for case in search['cases']]
    settings = {**vars(args), 'strata': [] if args.verify_prefix else args.strata,
                'frame_skip': DECISION_INTERVAL, 'action_type': 'FILTERED',
                'reference_runtime': 'frozen pre-fix experimental runtime; see reference_sources'}
    report = {'settings': settings, 'sources': sources, 'reference_sources': reference_sources,
              'searches': [{key: value for key, value in search.items() if key != 'cases'}
                           for search in searches],
              'cases': rows, 'limitations': [
        'First qualifying state per predefined stratum and seed; overlapping strata share one capture.',
        'General stratum requires an input divergence; opportunity controls can have identical policy inputs.',
        'One-timer stratum: a viable option now, or a divergence within 32 frames of an option or with a setup decision.',
        'Safe-pass stratum: a modeled-safe position pass is available in the revised evaluator.',
        'Native shot release combines fresh counters or normal-shot animation, ownership and puck impulse.',
        'Ordinary turnovers require four consecutive frames of opponent skater ownership, outside shot follow-through.',
        'Loose saves/deflections remain pending; net-plane misses settle for 24 frames and timeouts remain unresolved.',
        'Branch durations differ; possession fractions and event times accompany counts.',
        'Direct controllers use predict_frame(..., frame_skip=4), matching the production ScriptedAgent wrapper.',
        'Counterfactual branches share full emulator/RNG state; subsequent CPU choices can differ.',
        'Modeled one-timer options are not native shot attempts or guaranteed goals.',
        'Forced pass/escape controls are diagnostic alternatives, not a tuned production policy.',
    ]}
    if args.output:
        Path(args.output).write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps([{'seed': row['seed'], 'frame': row['capture_frame'],
                       'outcomes': {name: {key: value[key] for key in (
                           'goals', 'shots', 'one_timer_attempts', 'first_opponent_possession_frame',
                           'frames_with_one_timer_option')} for name, value in row.get('branches', {}).items()}}
                      for row in rows], indent=2))
    return report


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-runtime', required=True)
    parser.add_argument('--seed', type=int, default=20262004)
    parser.add_argument('--seeds', type=int, default=6)
    parser.add_argument('--search-frames', type=int, default=6000)
    parser.add_argument('--horizon', type=int, default=180)
    parser.add_argument('--outcome-ended', action='store_true',
                        help='Follow the first attack to a resolved shot, confirmed turnover, stoppage or timeout.')
    parser.add_argument('--strata', nargs='+', choices=('general', 'one-timer', 'safe-pass'), default=['general'])
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--verify-prefix', action='store_true',
                        help='Run only the default prefix to verify its full action digest against the CPU benchmark.')
    parser.add_argument('--min-depth', type=float, default=140)
    parser.add_argument('--output')
    return parser


if __name__ == '__main__':
    run(build_parser().parse_args())

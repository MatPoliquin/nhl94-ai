"""Same-state chance-window diagnosis without changing Classic's admission gates."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from nhl94_ai.agents.carry import carry_pad, carry_path
from nhl94_ai.agents.classic_v1 import ClassicAIV1Model
from nhl94_ai.agents.defense import controlled_slot, eligible
from nhl94_ai.agents.passing import PassOption, evaluate_pass, pad_direction, selected_receiver
from nhl94_ai.agents.responses import response_future
from nhl94_ai.env.factory import make_retro
from nhl94_ai.env.intents import HOCKEY_INTENT_NOOP
from nhl94_ai.evaluation.carry_outcomes import CarryOutcomes
from nhl94_ai.evaluation.benchmark import away_view
from nhl94_ai.evaluation.carry_replay import DECISION_INTERVAL, forced_policy, frame_action
from nhl94_ai.evaluation.carry_replay_gate import validate_benchmark, validate_sources
from nhl94_ai.evaluation.cpu_benchmark import CPU_RAM, cpu_view, select_side
from nhl94_ai.game.constants import GameConsts as Buttons
from nhl94_ai.game.geometry import aim_pass
from nhl94_ai.game.ram import pass_geometry_info, restore_away_control
from nhl94_ai.game.state import NHL94GameState
from nhl94_ai.tasks.cross_crease_setup import crossing_touch_player


SAVE = 'SabresVsMightyDucks.ManualGoalie.Start'
FRESH_STRATA = ('uncertified', 'unsupported-response', 'no-window', 'below-margin', 'above-margin')


def clone_policy(model):
    """Copy gameplay history, excluding this tool's read-only decision observer."""
    clone = ClassicAIV1Model()
    clone.__dict__.update(deepcopy({name: value for name, value in model.__dict__.items()
                                   if name != '_predict_decision'}))
    return clone


def restore_sample(env, sample, side):
    env.em.set_state(sample['snapshot'])
    env.data.update_ram()
    if hashlib.sha256(env.get_ram().tobytes()).hexdigest() != sample['ram_sha256']:
        raise RuntimeError('Counterfactual did not restore identical initial RAM.')
    state = deepcopy(sample['state'])
    view = cpu_view(state, pass_geometry_info(env, env.data.lookup_all()), side)
    return state, view


def capture(env, state, model):
    return {'snapshot': env.em.get_state(), 'ram_sha256': hashlib.sha256(env.get_ram().tobytes()).hexdigest(),
            'state': deepcopy(state), 'model': clone_policy(model)}


def step(env, state, action, side):
    *_, info = env.step(np.asarray(action, dtype=np.int8))
    if side == 2:
        info = restore_away_control(env.data, info, controller_prefix='bench', player_prefix='cpu', slots=6)
    return cpu_view(state, pass_geometry_info(env, info), side), info


def planner_windows(model, view, frame):
    result = []
    for purpose in ('position', 'one-timer'):
        available = model.one_timer.retry_at_frame if purpose == 'one-timer' else model.offense.pass_at
        if frame < available:
            continue
        options, _ = model.offense.passes(view, purpose, continuations=False)
        result.extend({'receiver': option.slot, 'purpose': purpose, 'value': option.shot_value,
                       'margin': option.margin, 'point': option.point,
                       'flight_frames': option.flight_frames} for option in options if option.shot_value > 0)
    return result


def gate_reason(row, baseline, first=False):
    if row['status'] != 'modeled-window':
        return row['status']
    if first:
        return 'straight-control'
    if row['receiver'] is None:
        return 'no-window'
    return 'below-margin' if row['value'] <= baseline + 10 else 'above-margin'


def sampling_strata(rows, baseline):
    reasons = {gate_reason(row, baseline, index == 0) for index, row in enumerate(rows)}
    return sorted({name for name in FRESH_STRATA if
                   (name == 'uncertified' and 'uncertified-carry' in reasons)
                   or (name == 'unsupported-response' and 'no-supported-response' in reasons)
                   or name in reasons})


def confusion(predicted, observed):
    return 'true-positive' if predicted and observed else 'false-positive' if predicted else (
        'false-negative' if observed else 'true-negative')


def natural_branch(env, sample, side, horizon, *, enabled, expected_action=None, receiver=None,
                   forecast_target=None, forecast_frames=18):
    state, view = restore_sample(env, sample, side)
    model = deepcopy(sample['model'])
    model.offense.chance_creation = enabled
    info = env.data.lookup_all()
    outcomes = CarryOutcomes(view, info[f'p{side}_score'], crossing_touch_player(env))
    initial_owner, goals = view.engine.puck_owner, info[f'p{side}_score']
    passes = view.team1.pass_attempts
    trace, pass_events = [], []
    pending = None
    matching_prefix = 0
    diverged = False
    for elapsed in range(horizon):
        action = frame_action(model, view)
        if elapsed == 0 and expected_action is not None and not np.array_equal(action, expected_action):
            raise RuntimeError('Replayed selected first action differs from the benchmark prefix.')
        windows = planner_windows(model, view, model.scheduler.frames) if view.engine.puck_owner == initial_owner else []
        trace.append({'elapsed': elapsed, 'owner': view.engine.puck_owner, 'decision': model._last_decision,
                      'buttons': action.tolist(), 'target': model._last_target,
                      'live_planner_windows': windows,
                      'predicted_receiver_window': any(item['receiver'] == receiver for item in windows)})
        before_owner = view.engine.puck_owner
        if forecast_target is not None and elapsed < forecast_frames and not diverged:
            if before_owner != initial_owner:
                diverged = True
            else:
                if elapsed % DECISION_INTERVAL == 0:
                    pad = carry_pad(view.team1.get_player_by_scnum(initial_owner), forecast_target)
                expected = np.zeros(Buttons.INPUT_MAX, dtype=np.int8)
                expected[Buttons.INPUT_LEFT if pad[0] < 0 else Buttons.INPUT_RIGHT] = abs(pad[0])
                expected[Buttons.INPUT_DOWN if pad[1] < 0 else Buttons.INPUT_UP] = abs(pad[1])
                diverged = not np.array_equal(expected, action)
                matching_prefix += not diverged
        view, info = step(env, state, action, side)
        if view.team1.pass_attempts > passes:
            passes = view.team1.pass_attempts
            pending = {'elapsed': elapsed + 1, 'owner_before_counter': before_owner,
                       'native_target': view.engine.pass_target, 'first_possession_owner': None,
                       'received_requested': False}
            pass_events.append(pending)
        if pending is not None:
            pending['flight_observed'] = pending.get('flight_observed', False) or view.engine.puck_owner < 0
            if view.engine.puck_owner >= 0 and (
                    view.engine.puck_owner != pending['owner_before_counter'] or pending['flight_observed']):
                pending['first_possession_owner'] = view.engine.puck_owner
                pending['first_possession_elapsed'] = elapsed + 1
                pending['received_requested'] = view.engine.puck_owner == pending['native_target']
                pending = None
        if outcomes.observe(elapsed + 1, view, info[f'p{side}_score'], crossing_touch_player(env)):
            break
        if info['bench_clock'] == 0:
            break
    return {**outcomes.summary(), 'initial_ram_sha256': sample['ram_sha256'],
            'goals': info[f'p{side}_score'] - goals, 'final_owner': view.engine.puck_owner,
            'pass_events': pass_events, 'trace': trace,
            'forecast_route_matched_prefix_frames': matching_prefix}


def forced_route(env, sample, side, target, frames):
    """A held-input mechanics probe; never pretend the policy selected a rejected route."""
    state, view = restore_sample(env, sample, side)
    model = deepcopy(sample['model'])
    owner = view.engine.puck_owner
    model.offense.chance_owner, model.offense.chance_until = None, 0
    trace = []
    for elapsed in range(frames):
        if view.engine.puck_owner != owner or view.engine.clock_stopped:
            break
        if elapsed % DECISION_INTERVAL == 0:
            model = forced_policy(model, view)
            pad = carry_pad(view.team1.get_player_by_scnum(owner), target)
            model.scheduler.remaining = DECISION_INTERVAL
        else:
            model._observe_ordinary_pass(view)
            model._idle(1)
        action = np.zeros(Buttons.INPUT_MAX, dtype=np.int8)
        action[Buttons.INPUT_LEFT if pad[0] < 0 else Buttons.INPUT_RIGHT] = abs(pad[0])
        action[Buttons.INPUT_DOWN if pad[1] < 0 else Buttons.INPUT_UP] = abs(pad[1])
        model._last_decision = 'forced-window-route'
        model.scheduler.action = model._encode(action, HOCKEY_INTENT_NOOP)
        model.scheduler.remaining -= 1
        model.scheduler.was_defending = False
        trace.append({'elapsed': elapsed, 'owner': owner, 'buttons': action.tolist()})
        view, info = step(env, state, action, side)
        if info['bench_clock'] == 0:
            break
    endpoint = capture(env, state, model)
    return endpoint, view, {'elapsed_frames': len(trace), 'retained_original_carrier':
                           view.engine.puck_owner == owner, 'clock_stopped': view.engine.clock_stopped,
                           'final_owner': view.engine.puck_owner, 'trace': trace}


def request_pass(history, view, receiver, purpose):
    model = forced_policy(history, view)
    player = view.team1.get_player_by_scnum(view.engine.puck_owner)
    target = view.team1.get_player_by_scnum(receiver)
    index = receiver - view.team1.skater_scnum_base()
    action = np.zeros(Buttons.INPUT_MAX, dtype=np.int8)
    aim_pass(action, player, target)
    action[Buttons.INPUT_B] = 1
    if purpose == 'position':
        option = PassOption(index, receiver, (target.x, target.y), 48,
                            pad_direction(player, target), 0, 0, 0, 0, 0, 0)
        model.offense.start_pass(view, option, model.scheduler.frames, 'position-pass')
    elif purpose == 'one-timer':
        model.one_timer.start(view, receiver, model.scheduler.frames)
    else:
        raise ValueError(f'Unknown window purpose: {purpose}')
    model._last_decision = 'forced-window-pass'
    model.scheduler.action = model._encode(action, HOCKEY_INTENT_NOOP)
    model.scheduler.remaining = DECISION_INTERVAL - 1
    return model, action


def dangerous_reception(view, receiver):
    player = view.team1.get_player_by_scnum(receiver)
    sign = 1 if view.team2.net.y > view.team1.net.y else -1
    return player is not None and abs(player.x) <= 71 and 150 < player.y * sign < 245


def native_pass_probe(env, endpoint, side, receiver, purpose, horizon):
    if horizon < 1:
        raise ValueError('Native pass probes require a positive frame budget.')
    state, view = restore_sample(env, endpoint, side)
    info = env.data.lookup_all()
    passes_before, attempts_before = view.team1.pass_attempts, view.team1.one_timer_attempts
    shots_before, goals_before = view.team1.stats.shots, info[f'p{side}_score']
    available = endpoint['model'].one_timer.retry_at_frame if purpose == 'one-timer' else endpoint['model'].offense.pass_at
    policy_available = endpoint['model'].scheduler.frames + 1 >= available
    policy_available &= view.engine.puck_owner == controlled_slot(view.team1)
    passer = view.team1.get_player_by_scnum(view.engine.puck_owner)
    target = view.team1.get_player_by_scnum(receiver)
    index = receiver - view.team1.skater_scnum_base()
    _, evaluation = evaluate_pass(view, passer, index, target, purpose,
                                  decision_interval=DECISION_INTERVAL)
    selector = selected_receiver(view.team1, view.puck, view.engine.puck_owner, pad_direction(passer, target))
    model, first = request_pass(endpoint['model'], view, receiver, purpose)
    outcomes = CarryOutcomes(view, goals_before, crossing_touch_player(env))
    actual_target = target_elapsed = None
    receipt = None
    run = 0
    accepted = recorded = scored = False
    c_requests = []
    c_down = model.buttons.c_down
    for elapsed in range(horizon):
        action = first if elapsed == 0 else frame_action(model, view)
        if action[Buttons.INPUT_C] and not c_down:
            c_requests.append(elapsed)
        c_down = bool(action[Buttons.INPUT_C])
        view, info = step(env, state, action, side)
        if actual_target is None and view.team1.pass_attempts > passes_before:
            actual_target = view.engine.pass_target
            target_elapsed = elapsed + 1
        if actual_target == receiver and view.engine.puck_owner == receiver:
            run += 1
            if receipt is None:
                player = view.team1.get_player_by_scnum(receiver)
                receipt = {'elapsed': elapsed + 1, 'position': (player.x, player.y),
                           'dangerous': dangerous_reception(view, receiver)}
        else:
            run = 0
        attributed = view.engine.shot_player == receiver
        accepted |= view.team1.one_timer_attempts > attempts_before and attributed
        recorded |= view.team1.stats.shots > shots_before and attributed
        scored |= info[f'p{side}_score'] > goals_before and attributed
        ended = outcomes.observe(elapsed + 1, view, info[f'p{side}_score'], crossing_touch_player(env))
        retained = receipt is not None and receipt['dangerous'] and run >= 4
        if purpose == 'position' and retained or ended or info['bench_clock'] == 0:
            break
    usable = actual_target == receiver and (
        retained if purpose == 'position' else accepted and (
            dangerous_reception(view, receiver) or recorded or scored))
    return {'receiver': receiver, 'purpose': purpose, 'initial_ram_sha256': endpoint['ram_sha256'],
            'policy_available': policy_available, 'native_target': actual_target,
            'native_target_elapsed': target_elapsed,
            'live_recipient_prediction': selector, 'live_pass_evaluation': evaluation,
            'first_action': first.tolist(), 'fresh_c_request_frames': c_requests,
            'received': receipt, 'settled_receiver_frames': run, 'native_one_timer': accepted,
            'recorded_receiver_shot': recorded, 'receiver_goal': scored,
            'usable_native_window': bool(usable), 'outcome': outcomes.summary()}


def forecasts(sample, side, row, frames):
    model = deepcopy(sample['model'])
    view = sample['state'] if side == 1 else away_view(sample['state'])
    player = view.team1.get_player_by_scnum(view.engine.puck_owner)
    path = carry_path(player, tuple(row['target']), frames=frames, decision_interval=DECISION_INTERVAL)
    if path is None:
        return [], {'status': 'unbounded-route'}
    future, response = response_future(view, path)
    response['friendly_positions'] = [
        {'slot': future.team1.skater_scnum_base() + index, 'position': (player.x, player.y),
         'assignment': player.assignment}
        for index, player in enumerate(future.team1.players)]
    issued = bool(response['response_slots']) and response['response_clearance'] > 0
    model.offense.one_timer_at = model.one_timer.retry_at_frame
    options = planner_windows(model, future, model.scheduler.frames + 1 + frames) if issued else []
    return options, {**response, 'status': 'issued' if issued else 'abstained',
                     'counterfactual_outside_carry_admission': not row['carry_safe']}


def investigate_case(env, sample, side, model, action, *, selected, horizon, pass_horizon):
    diagnostics = deepcopy(model.offense_diagnostics)
    rows = diagnostics.get('chance_candidates', [])
    chosen_target = tuple(model._last_target)
    chosen_receiver = diagnostics.get('chance_receiver')
    available_frames = (sample['model'].offense.chance_until - sample['model'].scheduler.frames - 1
                        if sample['model'].offense.chance_owner is not None else 18)
    frames = min((row['forecast_frames'] for row in rows if 'forecast_frames' in row),
                 default=max(1, min(18, available_frames)))
    result = {'initial_ram_sha256': sample['ram_sha256'], 'selected': selected,
              'initial_owner': sample['state'].engine.puck_owner,
              'decision': model._last_decision, 'diagnostics': diagnostics, 'routes': []}
    if selected:
        result['natural'] = natural_branch(env, sample, side, horizon, enabled=True,
                                           expected_action=action, receiver=chosen_receiver,
                                           forecast_target=chosen_target, forecast_frames=frames)
        result['without_chance'] = natural_branch(env, sample, side, horizon, enabled=False,
                                                  receiver=chosen_receiver,
                                                  forecast_target=chosen_target, forecast_frames=frames)
    for index, row in enumerate(rows):
        endpoint, view, route = forced_route(env, sample, side, tuple(row['target']), frames)
        predicted, response = forecasts(sample, side, row, frames)
        probes = []
        if route['elapsed_frames'] == frames and route['retained_original_carrier'] and not route['clock_stopped']:
            for player_index, player in enumerate(view.team1.players):
                slot = view.team1.skater_scnum_base() + player_index
                if slot == view.engine.puck_owner or not eligible(player):
                    continue
                for purpose in ('position', 'one-timer'):
                    probes.append(native_pass_probe(env, endpoint, side, slot, purpose, pass_horizon))
        expected = {(option['receiver'], option['purpose']) for option in predicted}
        forecast_issued = response['status'] == 'issued'
        for probe in probes:
            probe['predicted'] = (probe['receiver'], probe['purpose']) in expected
            probe['confusion'] = (confusion(probe['predicted'], probe['usable_native_window'])
                                  if forecast_issued and probe['policy_available'] else 'abstained-or-unavailable')
        result['routes'].append({'target': row['target'], 'selected_target': tuple(row['target']) == chosen_target,
                                 'selector_row': row, 'gate': gate_reason(
                                     row, diagnostics.get('chance_baseline', 0), index == 0),
                                 'prediction': predicted, 'response': response, 'native_route': route,
                                 'native_endpoint_positions': [
                                     {'slot': view.team1.skater_scnum_base() + player_index,
                                      'position': (player.x, player.y), 'assignment': player.assignment}
                                     for player_index, player in enumerate(view.team1.players)],
                                 'native_pass_probes': probes})
    restore_sample(env, sample, side)
    return result


def ready(model, view):
    owner = view.engine.puck_owner
    player = view.team1.get_player_by_scnum(owner)
    sign = 1 if view.team2.net.y > view.team1.net.y else -1
    return (owner == controlled_slot(view.team1) and player is not None
            and player is not view.team1.goalie and 88 < player.y * sign < 218
            and not view.engine.clock_stopped)


def replay_period(job):
    side, seed, expected, fresh_limit, horizon, pass_horizon = job
    env = make_retro(game='NHL94-Genesis-v0', state=SAVE, num_players=1)
    try:
        for name, (address, kind) in CPU_RAM.items():
            env.data.set_variable(name, {'address': address, 'type': kind})
        env.reset(seed=seed)
        env.data.update_ram()
        select_side(env.data, env.data.lookup_all(), side)
        env.data.set_value('bench_rng', seed)
        env.data.set_value('bench_clock', 300)
        *_, info = env.step(np.zeros(12, dtype=np.int8))
        state = NHL94GameState(5)
        model = ClassicAIV1Model(SimpleNamespace(action_type='FILTERED', chance_creation=True))
        observed = {}
        def observe_decision(view, deterministic=True):
            if ready(model, view):
                observed['before'] = clone_policy(model)
            return ClassicAIV1Model._predict_decision(model, view, deterministic)
        model._predict_decision = observe_decision
        digest, cases, covered = hashlib.sha256(), [], set()
        gates, safety_reasons, outer_gates = Counter(), Counter(), Counter()
        selected_count = candidates = frames = 0
        while info['bench_clock'] > 0 and frames < 300 * 120 + 5999:
            view = cpu_view(state, pass_geometry_info(env, info), side)
            observed.clear()
            action = frame_action(model, view)
            before = observed.get('before')
            if before is not None:
                rows = model.offense_diagnostics.get('chance_candidates', [])
                baseline = model.offense_diagnostics.get('chance_baseline', 0)
                if rows:
                    candidates += 1
                    for index, row in enumerate(rows):
                        gates[gate_reason(row, baseline, index == 0)] += 1
                    labels = sampling_strata(rows, baseline)
                else:
                    labels = []
                    outer_gates[model.offense_diagnostics.get('status', model._last_decision)] += 1
                selected = model._last_decision == 'create-chance'
                selected_count += selected
                fresh = (expected is None and len(cases) < fresh_limit
                         and bool(set(labels) - covered))
                if expected is not None and selected or fresh:
                    sample = capture(env, state, before)
                    case = investigate_case(env, sample, side, model, action, selected=selected,
                                            horizon=horizon, pass_horizon=pass_horizon)
                    case.update(side=side, seed=seed, capture_frame=frames + 1,
                                strata=labels, sample_kind='benchmark-selected' if expected else 'fresh-stratified')
                    cases.append(case)
                    covered.update(labels)
                # Reevaluate rejection details on a separate controller, never the live prefix.
                if rows:
                    probe = deepcopy(before.offense)
                    player = view.team1.get_player_by_scnum(view.engine.puck_owner)
                    for row in rows:
                        if not row['carry_safe']:
                            target, detail = probe._carry_option(view, player, tuple(row['target']))
                            safety_reasons[detail['carry_status']] += 1
            view, info = step(env, state, action, side)
            digest.update(np.asarray(action, dtype=np.int8).tobytes())
            frames += 1
        if info['bench_clock'] != 0:
            raise RuntimeError(f'Incomplete replay prefix: side {side}, seed {seed}.')
        parity = {'frames': frames + 1, 'actions_sha256': digest.hexdigest(), 'clock_remaining': 0}
        if expected is not None and (
                parity['frames'] != expected['frames'] or parity['actions_sha256'] != expected['actions_sha256']
                or selected_count != expected['decisions'].get('create-chance', 0)):
            raise RuntimeError(f'Candidate replay differs from measured production actions: side {side}, seed {seed}.')
        return {'side': side, 'seed': seed, 'prefix': parity,
                'benchmark_action_parity': expected is not None, 'selected_decisions': selected_count,
                'candidate_evaluations': candidates, 'candidate_gates': dict(gates),
                'carry_rejection_reasons': dict(safety_reasons), 'outer_gates': dict(outer_gates),
                'covered_strata': sorted(covered), 'missing_strata': sorted(set(FRESH_STRATA) - covered),
                'cases': cases}
    finally:
        env.close()


def summarize(periods):
    gates, safety, matrices = Counter(), Counter(), {}
    selected = []
    for period in periods:
        gates.update(period['candidate_gates'])
        safety.update(period['carry_rejection_reasons'])
        for case in period['cases']:
            if case['sample_kind'] == 'benchmark-selected':
                selected.append({'side': case['side'], 'seed': case['seed'], 'frame': case['capture_frame'],
                                 'receiver': case['diagnostics']['chance_receiver'],
                                 'purpose': case['diagnostics']['chance_purpose'],
                                 'natural_end': case['natural']['end_reason'],
                                 'natural_passes': case['natural']['pass_events'],
                                 'natural_shots': case['natural']['shot_events']})
            for route in case['routes']:
                cohort = ('certified-carry' if route['selector_row']['carry_safe'] else 'rejected-carry')
                key = case['sample_kind'] + '/' + cohort
                counter = matrices.setdefault(key, Counter())
                counter['routes'] += 1
                counter['lost_before_window'] += not route['native_route']['retained_original_carrier']
                for probe in route['native_pass_probes']:
                    counter[probe['confusion']] += 1
                    counter['usable_native_windows'] += probe['usable_native_window'] and probe['policy_available']
    return {'periods': len(periods), 'cases': sum(len(period['cases']) for period in periods),
            'candidate_gates': dict(gates), 'carry_rejection_reasons': dict(safety),
            'target_purpose_confusion': {key: dict(value) for key, value in matrices.items()},
            'selected_decisions': selected}


def run(args):
    root = Path(__file__).resolve().parents[2]
    candidate = json.loads(Path(args.benchmark).read_text(encoding='utf-8'))
    gate = json.loads(Path(args.gate).read_text(encoding='utf-8'))
    if not gate.get('passed'):
        raise ValueError('A passed current-source native cadence gate is required.')
    validate_sources(gate['sources'], root)
    if candidate.get('protocol') != 'nhl94-cpu-first-period-v2' or not candidate['settings'].get('chance_creation'):
        raise ValueError('Use a completed chance-creation CPU benchmark.')
    validate_sources(candidate['sources'], root)
    settings = candidate['settings']
    if (settings['frame_skip'] != DECISION_INTERVAL or settings['action_type'] != 'FILTERED'
            or settings['goalie_policy'] != 'off' or settings['seconds'] != 300
            or any(settings.get(name) for name in ('deke', 'cross_crease', 'uncertain_carry'))):
        raise ValueError('Investigation requires the standard four-frame goalie-off Classic experiment.')
    if min(args.fresh_seeds, args.samples_per_period, args.horizon, args.pass_horizon, args.workers) < 1:
        raise ValueError('Use positive bounded sample counts, replay horizons and workers.')
    recorded = {row['seed'] for row in candidate['matches']}
    fresh_seeds = range(args.fresh_seed, args.fresh_seed + args.fresh_seeds)
    if args.fresh_seed < 0 or args.fresh_seed + args.fresh_seeds > 2**32 or recorded.intersection(fresh_seeds):
        raise ValueError('Fresh seeds must be uint32 values disjoint from the benchmark.')
    jobs = []
    for row in candidate['matches']:
        if not row['completed'] or row['clock_remaining'] != 0:
            raise ValueError('Do not investigate incomplete benchmark periods.')
        if row['decisions'].get('create-chance', 0):
            jobs.append((row['side'], row['seed'], row, 0, args.horizon, args.pass_horizon))
    expected_selected = sum(row['decisions'].get('create-chance', 0) for row in candidate['matches'])
    if expected_selected != 4:
        raise ValueError('This investigation requires the recorded four selected decisions.')
    jobs.extend((side, seed, None, args.samples_per_period, args.horizon, args.pass_horizon)
                for seed in fresh_seeds for side in (1, 2))
    names = ('evaluation/chance_creation_replay.py', 'evaluation/carry_replay.py',
             'evaluation/carry_outcomes.py', 'tasks/cross_crease_setup.py')
    sources = {**candidate['sources'], **gate['sources'],
               **{f'nhl94_ai/{name}': hashlib.sha256((root / 'nhl94_ai' / name).read_bytes()).hexdigest()
                  for name in names}}
    if args.output and Path(args.output).resolve() in {
            Path(args.benchmark).resolve(), Path(args.gate).resolve(),
            *((root / name).resolve() for name in sources)}:
        raise ValueError('Replay output must not overwrite its benchmark, gate or measured sources.')
    if args.workers == 1:
        periods = [replay_period(job) for job in jobs]
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            periods = list(pool.map(replay_period, jobs))
    validate_sources(sources, root)
    summary = summarize(periods)
    if len(summary['selected_decisions']) != expected_selected:
        raise RuntimeError('Not all four selected decisions were replayed.')
    report = {'protocol': 'nhl94-chance-window-replay-v1', 'settings': vars(args), 'sources': sources,
              'summary': summary, 'periods': periods,
              'definitions': {
                  'prediction': 'Native-response forecast accepts a positive-valued receiver/purpose option at the endpoint.',
                  'native-window': 'Fresh native pass selects the requested receiver; ordinary reception settles four frames '
                                   'inside |x|<=71, attack depth (150,245), or a requested one-timer is accepted '
                                   'and has usable geometry, a recorded shot or goal.',
                  'confusion-unit': 'Receiver/purpose probe, conditional on a retained carrier and available retry deadline.',
                  'abstention': 'Unsupported/contacting response scenario or an unavailable policy retry; not a predicted negative.',
                  'forced-route': 'Diagnostic held-input prefix, including rejected candidates; never a normal-policy alternative.'},
              'limitations': [
                  'Fresh cases are first-in-stratum, not a random match-play error-rate estimate.',
                  'Receiver/purpose probes from one state overlap and are not independent samples.',
                  'Lost carriers, missing strata and abstentions remain visible instead of becoming true negatives.',
                  'Native reception/attempt proves bounded executability, not a calibrated goal probability.',
                  'Natural-policy traces can abandon a predicted route; held-route probes test its conditional forecast.',
                  'The prior sparse nonnegative bootstrap interval cannot rule out unobserved downside.']}
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(summary, indent=2))
    return report


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--benchmark', default='docs/benchmarks/classic-v1-chance-creation-candidate.json')
    parser.add_argument('--gate', default='docs/benchmarks/classic-v1-chance-creation-cadence.json')
    parser.add_argument('--fresh-seed', type=int, default=20262801)
    parser.add_argument('--fresh-seeds', type=int, default=4)
    parser.add_argument('--samples-per-period', type=int, default=4)
    parser.add_argument('--horizon', type=int, default=240)
    parser.add_argument('--pass-horizon', type=int, default=120)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--output')
    return parser


if __name__ == '__main__':
    run(build_parser().parse_args())

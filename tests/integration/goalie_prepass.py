"""Button-only component experiments from before four archived CPU passes."""
import argparse
from collections import deque
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
from unittest.mock import patch

import numpy as np

from nhl94_ai.agents import classic_v1, goalie as current
from nhl94_ai.agents.defense import controlled_slot
from nhl94_ai.env import factory
from nhl94_ai.evaluation import cpu_benchmark
from nhl94_ai.game.constants import GameConsts as Buttons
from nhl94_ai.game.ram import pass_geometry_info, restore_away_control
from tests.integration.goalie_timing import baseline_controller, CapturedEnough


VARIANTS = ('baseline', 'corrected', 'anticipation-only', 'alignment-only',
            'no-save-commitment', 'early-takeover-only', 'cpu-goalie')


def blocked_skater(model):
    return (model.one_timer.pending is not None or model.offense.pass_action.pending is not None
            or model.cross_crease is not None and model.cross_crease.plan is not None
            or model.scheduler.decisions < model.shot.until_decision or model.defense.pending_check is not None
            or model.defense.pending_switch is not None)


def trace_row(frame, state, model, info, action):
    goalie, manager = state.team1.goalie, model.goalie
    selected = controlled_slot(state.team1) == state.team1.defense_goalie
    return {
        'frame': frame, 'scores': [info['p1_score'], info['p2_score']],
        'clock_stopped': state.engine.clock_stopped,
        'puck': [state.puck.x, state.puck.y, state.puck.height],
        'puck_velocity': [state.puck.motion_x, state.puck.motion_y, state.puck.motion_z],
        'owner': state.engine.puck_owner, 'last_toucher': state.engine.last_puck_player,
        'pass_target': state.engine.pass_target, 'passes': state.team2.pass_attempts,
        'one_timers': state.team2.one_timer_attempts, 'shot_player': state.engine.shot_player,
        'one_timer_goals': info[f'bench_one_timer_goals{state.team2.controller}'],
        'goalie': [goalie.x, goalie.y], 'goalie_velocity': [goalie.motion_x, goalie.motion_y],
        'selected': selected, 'effective_manual': bool(
            selected and goalie.selection_flags & 8 and not goalie.live_state_flags & 4),
        'locked': bool(goalie.unavailable & 2 or goalie.selection_flags & 0x20),
        'animation': goalie.live_anim, 'cpu_fallback': bool(goalie.live_state_flags & 4),
        'hold_count': state.engine.goalie_hold_counts[0],
        'mode': manager.phase, 'buttons': action.tolist(),
        'target': manager.diagnostics.get('destination'),
        'crossing_frames': manager.diagnostics.get('crossing_frames'),
        'crossing_x': manager.diagnostics.get('crossing_x'),
        'deadline': manager.diagnostics.get('deadline'),
        'reach_frames': manager.diagnostics.get('reach_frames'),
        'reason': manager.diagnostics.get('reason'),
        'pending_save': manager.pending_save, 'save_at': manager.save_at,
        'blocked_skater': blocked_skater(model),
        'receiver': manager.diagnostics.get('receiver'),
        'reception_frames': manager.diagnostics.get('reception_frames'),
        'anticipation': manager.diagnostics.get('experiment_anticipation'),
        'opponent_players': [
            {'slot': state.team2.skater_scnum_base() + index, 'position': [player.x, player.y],
             'velocity': [player.motion_x, player.motion_y], 'stick': [player.stick_x, player.stick_y]}
            for index, player in enumerate(state.team2.players)],
        'metrics': dict(manager.metrics),
    }


def archived_attack(case, baseline, anchor=None):
    original_factory = factory.make_retro
    original_view = cpu_benchmark.cpu_view
    original_predict = classic_v1.ClassicAIV1Model.predict_frame
    holder, trace = {}, deque(maxlen=512)
    frame, score_at_release, last_takeover = 0, None, None

    def make_env(**kwargs):
        holder['env'] = original_factory(**kwargs)
        return holder['env']

    def view_state(state, info, side):
        holder['root_state'] = state
        return original_view(state, info, side)

    def predict(model, state, *args, **kwargs):
        nonlocal frame, score_at_release, last_takeover
        frame += 1
        if frame == anchor:
            holder['sample'] = deepcopy({
                'root_state': holder['root_state'], 'view': state, 'model': model})
            holder['sample']['snapshot'] = holder['env'].em.get_state()
            holder['sample']['ram_sha256'] = hashlib.sha256(bytes(holder['env'].get_ram())).hexdigest()
        info = holder['env'].data.lookup_all()
        before_requests = model.goalie.metrics['takeover_requests']
        action = original_predict(model, state, *args, **kwargs)
        if model.goalie.metrics['takeover_requests'] > before_requests:
            last_takeover = frame
        row = trace_row(frame, state, model, info, action[0])
        trace.append(row)
        if frame == case['frame']:
            assert row['shot_player'] == case['one_timer_release']['shot_player']
            assert row['last_toucher'] == row['shot_player']
            assert row['one_timers'] == case['one_timer_release']['attempts_after']
            score_at_release = row['scores'][2 - state.team1.controller]
        if score_at_release is not None:
            if row['scores'][2 - state.team1.controller] > score_at_release:
                holder['last_takeover'] = last_takeover
                raise CapturedEnough()
            assert frame < case['frame'] + 100, 'Archived failed one-timer did not score'
        return action

    with patch.object(factory, 'make_retro', side_effect=make_env), \
            patch.object(cpu_benchmark, 'cpu_view', side_effect=view_state), \
            patch.object(classic_v1, 'GoalieController', baseline.GoalieController), \
            patch.object(classic_v1.ClassicAIV1Model, 'predict_frame', new=predict):
        try:
            cpu_benchmark.cpu_match(('classic-v1', case['matchup'], case['seed'],
                                     300, 4, 'FILTERED', 'selective'))
        except CapturedEnough:
            pass
    assert score_at_release is not None and 'last_takeover' in holder
    rows = list(trace)
    if anchor is not None:
        assert 'sample' in holder, 'Pre-pass snapshot not reached'
        return holder['sample'], [row for row in rows if row['frame'] >= anchor]
    passes = [row for before, row in zip(rows, rows[1:])
              if row['frame'] < case['frame'] and row['passes'] > before['passes']]
    assert passes, 'No fresh accepted CPU pass before the attributed release'
    accepted = passes[-1]
    assert accepted['pass_target'] == case['one_timer_release']['shot_player']
    start = max(1, accepted['frame'] - 180)
    takeover = holder['last_takeover']
    if takeover is not None and accepted['frame'] - 180 <= takeover < start:
        start = max(1, takeover - 2)
    return {
        'accepted_pass_frame': accepted['frame'], 'pass_target': accepted['pass_target'],
        'pass_owner': accepted['owner'], 'last_takeover_request_frame': takeover,
        'start_frame': start, 'release_frame': case['frame'], 'goal_frame': rows[-1]['frame'],
        'accepted_pass': accepted,
    }


def component_controller(baseline, manager, variant):
    if variant not in VARIANTS:
        raise ValueError(f'Unknown goalie experiment: {variant}')
    if variant == 'baseline':
        return deepcopy(manager)
    if variant == 'corrected':
        updated = current.GoalieController('selective')
    else:
        class ComponentGoalie(baseline.GoalieController):
            def __init__(self):
                super().__init__('selective')
                self.phase = 'skater'
                self.save_at = 0
                self.observer = current.GoalieController('selective')
                self.interventions = 0

            def step(self, state, *, blocked=False):
                self.observer.frames = self.frames + 1
                self.observer._pass_feedback(state)
                raw_plan = baseline.goalie_target(state)
                advanced = current.goalie_target(state, self.observer.pass_windup)
                plan = raw_plan
                if variant == 'anticipation-only' and advanced.receiver is not None:
                    plan = replace(raw_plan, target=advanced.target,
                                   reason='diagnostic receiver-angle positioning only')
                    self.interventions += 1
                with patch.object(baseline, 'goalie_target', return_value=plan):
                    action = super().step(state, blocked=blocked)
                self.diagnostics['experiment_anticipation'] = {
                    'receiver': advanced.receiver, 'reception_frames': advanced.reception_frames,
                    'target': advanced.target, 'windup': self.observer.pass_windup}
                return action

            def _takeover(self, state, plan, blocked, *args):
                if variant == 'cpu-goalie':
                    return False, 'diagnostic retain CPU goalie'
                return super()._takeover(state, plan, blocked, *args)

            def _step(self, state, plan, slot, selected, blocked):
                if variant == 'cpu-goalie':
                    busy = (state.team1.goalie.unavailable & 2
                            or state.team1.goalie.selection_flags & 0x20
                            or state.team1.goalie.live_state_flags & 4
                            or self.outlet is not None or self.pending_save is not None
                            or state.engine.puck_owner == slot)
                    returning = self.phase in ('request-skater', 'neutral-handoff')
                    paused = state.engine.clock_stopped or state.engine.input_block
                    if selected and not busy and not returning and not paused:
                        self.interventions += 1
                        return self._return()
                    return super()._step(state, plan, slot, selected, blocked)
                suppress = variant == 'no-save-commitment'
                if variant == 'alignment-only' and plan.crossing_frames is not None:
                    steering = current.goalie_steer(state.team1.goalie, plan.target)
                    suppress = (abs(plan.crossing_x - state.team1.goalie.x) > 8
                                and plan.height <= 15 and plan.crossing_frames >= 2
                                and current.goalie_arrival(state.team1.goalie, plan.target) + 1
                                <= plan.crossing_frames and bool(steering[6:8].any()))
                eligible = (selected and not state.team1.goalie.live_state_flags & 4
                            and not state.team1.goalie.unavailable & 2
                            and state.engine.puck_owner != slot and self.pending_save is None)
                if suppress and eligible:
                    previous = self.save_at
                    self.save_at = math.inf
                    try:
                        action = super()._step(state, plan, slot, selected, blocked)
                    finally:
                        self.save_at = previous
                    self.interventions += 1
                    return action
                return super()._step(state, plan, slot, selected, blocked)

        updated = ComponentGoalie()
    for name, value in vars(manager).items():
        if hasattr(updated, name):
            setattr(updated, name, deepcopy(value))
    if variant == 'early-takeover-only':
        updated.policy = 'always'
    return updated


def replay(case, capture, sample, baseline, variant):
    state_name, side = cpu_benchmark.MATCHUPS[case['matchup']]
    env = factory.make_retro(game='NHL94-Genesis-v0', state=state_name, num_players=1,
                             goalie_policy='selective')
    try:
        for name, (address, kind) in cpu_benchmark.CPU_RAM.items():
            env.data.set_variable(name, {'address': address, 'type': kind})
        env.reset(seed=case['seed'])
        env.em.set_state(sample['snapshot'])
        env.data.update_ram()
        assert hashlib.sha256(bytes(env.get_ram())).hexdigest() == sample['ram_sha256']
        info = env.data.lookup_all()
        data = deepcopy(sample)
        model, root_state, view = data['model'], data['root_state'], data['view']
        model.goalie = component_controller(baseline, model.goalie, variant)
        trace = []
        live_seen = False
        horizon = capture['goal_frame'] - capture['start_frame'] + 65
        for offset in range(horizon):
            frame = capture['start_frame'] + offset
            if offset:
                info = pass_geometry_info(env, info)
                view = cpu_benchmark.cpu_view(root_state, info, side)
            memory = bytes(env.get_ram())
            with patch.object(env.data, 'set_value', side_effect=AssertionError('Policy RAM write')):
                action = model.predict_frame(view, 4)[0]
            assert bytes(env.get_ram()) == memory, 'Policy modified gameplay RAM'
            assert action.shape == (12,) and np.isin(action, (0, 1)).all()
            row = trace_row(frame, view, model, info, action)
            trace.append(row)
            if not view.engine.clock_stopped:
                live_seen = True
            elif live_seen:
                break
            *_, info = env.step(action)
            if side == 2:
                info = restore_away_control(env.data, info, controller_prefix='bench',
                                           player_prefix='cpu', slots=6)
        first, last = trace[0], trace[-1]
        attempts = [row for previous, row in zip(trace, trace[1:])
                    if row['one_timers'] > previous['one_timers']]
        return {
            'variant': variant, 'trace': trace,
            'goals_against': last['scores'][2 - side] - first['scores'][2 - side],
            'goals_for': last['scores'][side - 1] - first['scores'][side - 1],
            'opponent_one_timers': last['one_timers'] - first['one_timers'],
            'opponent_one_timer_goals': last['one_timer_goals'] - first['one_timer_goals'],
            'requested_receiver_release': any(
                row['shot_player'] == capture['pass_target'] and row['last_toucher'] == capture['pass_target']
                for row in attempts),
            'intervention_frames': getattr(model.goalie, 'interventions', None),
            'ended_on_stoppage': view.engine.clock_stopped,
            'stop_frame': last['frame'],
        }
    finally:
        env.close()


def summarize_comparison(reference, candidate, release_frame):
    before, after = reference['trace'], candidate['trace']
    pairs = list(zip(before, after))
    divergence = next((a['frame'] for a, b in pairs if a['buttons'] != b['buttons']), None)
    flight = ('puck', 'puck_velocity', 'owner', 'last_toucher', 'pass_target', 'passes', 'one_timers')
    difference = next((a['frame'] for a, b in pairs if any(a[name] != b[name] for name in flight)), None)
    through_release = [pair for pair in pairs if pair[0]['frame'] <= release_frame]
    shared = (before[0]['frame'] == after[0]['frame'] and after[-1]['frame'] >= release_frame
              and all(all(a[name] == b[name] for name in flight) for a, b in through_release))
    return {'first_button_difference': divergence, 'same_puck_through_original_release': shared,
            'first_puck_difference': difference,
            'reference_save_requests': [
                row['frame'] for previous, row in zip(before, before[1:])
                if (row['metrics'].get('save_requests', 0) + row['metrics'].get('dive_requests', 0)
                    > previous['metrics'].get('save_requests', 0) + previous['metrics'].get('dive_requests', 0))],
            'candidate_save_requests': [
                row['frame'] for previous, row in zip(after, after[1:])
                if (row['metrics'].get('save_requests', 0) + row['metrics'].get('dive_requests', 0)
                    > previous['metrics'].get('save_requests', 0) + previous['metrics'].get('dive_requests', 0))]}


def check_results(results):
    assert len(results) == 4
    for case in results:
        variants = {row['variant']: row for row in case['variants']}
        assert set(variants) == set(VARIANTS)
        assert variants['baseline']['goals_against'] == 1
        assert variants['baseline']['opponent_one_timer_goals'] == 1
        assert variants['baseline']['requested_receiver_release']
        assert variants['alignment-only']['comparison']['first_button_difference'] is None
    case = next(row for row in results
                if row['matchup'] == 'ducks-campbell-manual' and row['seed'] == 20261211)
    before, after = case['variants'][0], next(
        row for row in case['variants'] if row['variant'] == 'no-save-commitment')
    assert after['goals_against'] == 0 and after['opponent_one_timer_goals'] == 0
    assert after['comparison']['same_puck_through_original_release']
    change = after['comparison']['first_button_difference']
    original = next(row for row in before['trace'] if row['frame'] == change)
    modified = next(row for row in after['trace'] if row['frame'] == change)
    assert original['buttons'][Buttons.INPUT_C]
    assert modified['buttons'][Buttons.INPUT_LEFT] and not modified['buttons'][Buttons.INPUT_C]
    assert modified['frame'] > case['capture']['release_frame']
    assert any(row['owner'] == 5 and row['effective_manual'] for row in after['trace'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-root', required=True)
    parser.add_argument('--cases', default='docs/benchmarks/classic-v1-goalie-timing-native.json')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    source = json.loads(Path(args.cases).read_text(encoding='utf-8'))
    root = Path(args.baseline_root)
    for name, digest in source['baseline_sources'].items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == digest, name
    repository = Path(__file__).resolve().parents[2]
    for name, digest in source['corrected_sources'].items():
        assert hashlib.sha256((repository / name).read_bytes()).hexdigest() == digest, name
    baseline = baseline_controller(root)
    cases = [case for case in source['cases'] if case['one_timer_release']]
    results = []
    for case in cases:
        capture = archived_attack(case, baseline)
        sample, reference_trace = archived_attack(case, baseline, capture['start_frame'])
        assert capture['start_frame'] < capture['accepted_pass_frame']
        reference = replay(case, capture, sample, baseline, 'baseline')
        assert reference['trace'] == reference_trace, 'Restored full-model replay differs from archived attack'
        duplicate = replay(case, capture, sample, baseline, 'baseline')
        assert duplicate == reference, 'Baseline replay is not deterministic'
        assert reference['goals_against'] == 1 and reference['requested_receiver_release']
        variants = [reference]
        for variant in VARIANTS[1:]:
            row = replay(case, capture, sample, baseline, variant)
            row['comparison'] = summarize_comparison(reference, row, capture['release_frame'])
            variants.append(row)
        result = {
            'matchup': case['matchup'], 'seed': case['seed'], 'capture': capture,
            'snapshot_ram_sha256': sample['ram_sha256'], 'variants': variants,
            'baseline_full_trace_parity': True, 'baseline_repeat_parity': True,
        }
        results.append(result)
        print(case['matchup'], case['seed'], 'start/pass/release/goal',
              capture['start_frame'], capture['accepted_pass_frame'],
              capture['release_frame'], capture['goal_frame'], flush=True)
        for row in variants:
            print(row['variant'], 'GA', row['goals_against'], 'OT goals', row['opponent_one_timer_goals'],
                  'receiver release', row['requested_receiver_release'],
                  'first change', row.get('comparison', {}).get('first_button_difference'), flush=True)
    check_results(results)
    output = {
        'protocol': 'nhl94-goalie-prepass-component-experiment-v1',
        'baseline_sources': source['baseline_sources'],
        'corrected_sources': source['corrected_sources'],
        'harness_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'validation': {
            'unique_continuations': len(cases) * len(VARIANTS),
            'duplicate_baseline_continuations': len(cases),
            'policy_RAM_unchanged_bytes_per_call': 65536,
            'archived_full_model_trace_and_duplicate_controls_exact': True,
            'preserved_shot_save_commitment_counterexample_verified': True,
        },
        'limits': [
            'Four selected failures, not a strength benchmark or random sample.',
            'Start 180 frames before the last accepted pass; extend earlier for a recent B request.',
            'Full Classic model and decoded-state history restored, not idle skater continuations.',
            'Anticipation changes receiver-angle target only; original crossing/deadline/save rules remain.',
            'Alignment changes save commitment only; original targets and takeover rules remain.',
            'No-save commitment is a diagnostic, not a proposed deployable policy.',
            'Early takeover uses the existing always policy; CPU goalie returns by B, never RAM selection.',
            'Changing takeover also changes skater defense and can alter the original CPU attack.',
            'Cut off at the first stoppage or 64 frames after the archived goal; no full-period outcomes.',
            'Policy calls checked against all 65536 gameplay RAM bytes; setup/routing repairs unchanged.',
        ],
        'cases': results,
    }
    Path(args.output).write_text(json.dumps(output, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()

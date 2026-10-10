"""Matched ordinary-ROM skating dekes, including live nearby CPU defenders."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from nhl94_ai.agents.classic_v1 import ClassicAIV1Model
from nhl94_ai.agents.deke import DekePlan, SEQUENCE_FRAMES
from nhl94_ai.env.actions import HockeyActionController
from nhl94_ai.env.factory import make_retro
from nhl94_ai.evaluation.benchmark import RAM, away_view, update_state
from nhl94_ai.game.constants import GameConsts as Buttons
from nhl94_ai.game.ram import restore_away_control
from nhl94_ai.game.state import NHL94GameState
from nhl94_ai.tasks.defense_setup import _place_object, _rebuild_object_order, _reset_player
from tests.integration.cross_crease import _crossing_setup


SCENARIOS = ('clear', 'wing', 'wing-pressure', 'trailing', 'closing', 'blocked', 'open-shot', 'fast')


def _setup(env, *, side, direction, scenario, seed, handedness, jitter):
    _crossing_setup(env, side=side, direction=direction, depth=180, prepare=True, width=0)
    memory = env.data.memory
    actual = memory.extract(0xFFC320, '>i2')
    sign = 1 if side == 1 else -1
    rng = np.random.default_rng(seed)
    x, depth = rng.uniform(-jitter, jitter, 2) + (0, 180)
    if scenario == 'open-shot':
        x, depth = -direction * 24, 208
    elif scenario in ('wing', 'wing-pressure'):
        x = -direction * 28 + x
    role = memory.extract(0xFFB04A + actual * 0x80 + 0x34, '>i2')
    assignment = 1 if role in (1, 2) else 6 if role == 4 else 4
    _reset_player(memory, actual, (x, sign * depth), 0 if side == 1 else 4,
                  assignment, True, (0x10, 0x11))
    base = 0xFFB04A + actual * 0x80
    memory.assign(base + 0x2A, '>i2', sign * (12000 if scenario == 'fast' else 1500))
    memory.assign(base + 0x6C, '|u1', 30)
    memory.assign(base + 0x6D, '|u1', 20)
    memory.assign(base + 0x76, '|u1', handedness)
    goalie = 11 if side == 1 else 5
    _reset_player(memory, goalie, (direction * 28 if scenario == 'open-shot' else 0, sign * 250),
                  4 if side == 1 else 0, 0x0E, False)
    defender = 6 if side == 1 else 0
    if scenario in ('trailing', 'closing', 'blocked', 'wing-pressure'):
        point = ((-direction * 18, sign * (depth - 36)) if scenario == 'trailing' else
                 (direction * 45, sign * (depth + 8)) if scenario in ('closing', 'wing-pressure') else
                 (x, sign * (depth + 18)))
        role = memory.extract(0xFFB04A + defender * 0x80 + 0x34, '>i2')
        assignment = 2 if role in (1, 2) else 5 if role == 4 else 3
        _reset_player(memory, defender, point, 0 if side == 1 else 4, assignment, False, (0x11,))
        if scenario in ('closing', 'wing-pressure'):
            memory.assign(0xFFB04A + defender * 0x80 + 0x28, '>i2',
                          (-direction if scenario == 'closing' else direction) * 5000)
    _place_object(memory, 14, (x, sign * (depth + 8)))
    _rebuild_object_order(memory)
    env.data.set_value('bench_rng', seed)


def trial(*, enabled, side=1, direction=1, scenario='clear', seed=71000, handedness=0,
          jitter=0, schema='FILTERED', interval=4, interrupt=None, snapshot=None, capture=None, env=None,
          cut_fixture=False):
    if cut_fixture and (not enabled or scenario != 'open-shot'):
        raise ValueError('The native cut-finish fixture requires enabled dekes and an open-shot start.')
    owned = env is None
    if owned:
        env = make_retro(game='NHL94-Genesis-v0', state='PenguinsVsSenators.DefenseZone', num_players=1)
    try:
        for name, (address, kind) in RAM.items():
            env.data.set_variable(name, {'address': address, 'type': kind})
        if snapshot is None:
            env.reset(seed=seed)
            _setup(env, side=side, direction=direction, scenario=scenario, seed=seed,
                   handedness=handedness, jitter=jitter)
            *_, info = env.step(np.zeros(12, dtype=np.int8))
            snapshot = env.em.get_state()
        env.em.set_state(snapshot)
        env.data.update_ram()
        info = env.data.lookup_all()
        if capture is not None:
            capture['snapshot'] = snapshot
        # Re-serialization can contain changing core-private bytes. Both branches
        # restore the same payload in the same emulator and verify all exposed RAM.
        initial = hashlib.sha256(snapshot).hexdigest()
        initial_ram = hashlib.sha256(env.get_ram().tobytes()).hexdigest()
        state = NHL94GameState(5)
        model = ClassicAIV1Model(SimpleNamespace(action_type=schema, deke=enabled))
        context = SimpleNamespace(action_type=schema, game_state=state)
        processor = HockeyActionController(context)
        macro = processor._new_action_state()
        shots_before, goals_before = info[f'bench_shots{side}'], info[f'p{side}_score']
        digest, timeline = hashlib.sha256(), []
        first_decision = None
        impacts = 0
        last_impact = None
        for frame in range(SEQUENCE_FRAMES + 64):
            if interrupt == frame:
                opponent = 6 if side == 1 else 0
                env.data.memory.assign(0xFFB7AA, '>i2', opponent)
                env.data.memory.assign(0xFFBEDA, '>i2', opponent)
                env.data.update_ram()
                info = env.data.lookup_all()
            update_state(state, info, env)
            view = state if side == 1 else away_view(state)
            if cut_fixture and frame == 0:
                sign = 1 if view.team2.net.y > view.team1.net.y else -1
                plan = DekePlan(view.engine.puck_owner, direction, (direction * 28, sign * 200),
                                (-direction * 32, sign * 206), 0, 0, 60, -direction)
                model.deke.start(plan, view, 0)
                model.deke.event['bait_position'] = plan.bait
                model.deke._phase('cut', 0, 'native-close-finish-fixture')
            action = model.predict_frame(view, interval)[0]
            if frame == 0:
                first_decision = model._last_decision
            context.game_state = view
            buttons = processor._process_action(action, macro)[0] if schema != 'FILTERED' else action
            digest.update(buttons.tobytes())
            impact = view.team2.goalie.contact_impact
            if (last_impact is not None and impact is not None and impact > last_impact
                    and view.team2.goalie.contact_player == view.team1.defense_control):
                impacts += 1
            last_impact = impact
            player = view.team1.get_player_by_scnum(view.team1.defense_control)
            timeline.append({'frame': frame, 'decision': model._last_decision,
                             'player': (player.x, player.y) if player else None,
                             'goalie': (view.team2.goalie.x, view.team2.goalie.y),
                             'puck': (view.puck.x, view.puck.y), 'windup': view.is_shooting,
                             'c': bool(buttons[Buttons.INPUT_C]),
                             'owner': view.engine.puck_owner,
                             'deke': {key: value for key, value in model.offense_diagnostics.get('deke', {}).items()
                                      if key in ('phase', 'status', 'pressure', 'reason', 'outcome',
                                                 'follow', 'cut_distance', 'release_point', 'release_x_bounds')}})
            if interrupt == frame:
                assert not np.any(buttons), (schema, frame, buttons)
                assert model.deke.plan is None
                break
            if view.team1.stats.score > goals_before or view.engine.clock_stopped:
                break
            *_, info = env.step(buttons)
            if side == 2:
                info = restore_away_control(env.data, info, controller_prefix='bench', player_prefix='cpu', slots=6)
        if model.deke is not None and model.deke.plan is not None:
            model.deke._finish(frame, 'fixture-cutoff')
        return {
            'enabled': enabled, 'scenario': scenario, 'seed': seed, 'side': side, 'direction': direction,
            'handedness': handedness, 'schema': schema, 'interval': interval, 'jitter': jitter,
            'cut_fixture': cut_fixture,
            'initial_state_sha256': initial, 'initial_ram_sha256': initial_ram,
            'actions_sha256': digest.hexdigest(),
            'first_decision': first_decision, 'frames': frame + 1,
            'shots': view.team1.stats.shots - shots_before, 'goals': view.team1.stats.score - goals_before,
            'goalie_contacts': impacts, 'metrics': dict(model.deke.metrics) if model.deke else {},
            'events': model.deke.events if model.deke else [], 'timeline': timeline,
        }
    finally:
        if owned:
            env.close()


def paired_trial(**options):
    env = make_retro(game='NHL94-Genesis-v0', state='PenguinsVsSenators.DefenseZone', num_players=1)
    try:
        capture = {}
        baseline = trial(enabled=False, capture=capture, env=env, **options)
        candidate = trial(enabled=True, snapshot=capture['snapshot'], env=env, **options)
        assert baseline['initial_state_sha256'] == candidate['initial_state_sha256']
        assert baseline['initial_ram_sha256'] == candidate['initial_ram_sha256'], 'Paired initial RAM differs'
        return [baseline, candidate]
    finally:
        env.close()


def native_contracts():
    for schema in ('FILTERED', 'HOCKEY_INTENT_DPAD'):
        row = trial(enabled=True, side=2, scenario='open-shot', seed=71000, schema=schema, cut_fixture=True)
        assert row['metrics'].get('accepted_windups', 0) > 0, row['metrics']
        assert row['metrics'].get('observed_releases', 0) > 0, row['metrics']
        assert row['metrics'].get('recorded_shots', 0) > 0, row['metrics']
        assert not any(event['contact'] for event in row['events']), row['events']
        for event in row['events']:
            holds = [transition['frame'] for transition in event['timeline'] if transition['phase'] == 'hold']
            if holds:
                start, duration = holds[-1], event['hold_frames']
                pressed = [sample['c'] for sample in row['timeline']
                           if start <= sample['frame'] + 1 < start + duration]
                assert len(pressed) == duration and all(pressed), (schema, event, pressed)
        print(f'PASS: {schema}: native cut-finish fixture accepts windup and attributes shot without goalie contact')
        row = trial(enabled=True, side=1, scenario='wing-pressure', seed=71000, schema=schema)
        assert row['metrics'].get('candidate-rejected-goalie-contact', 0) > 0, row['metrics']
        assert not row['metrics'].get('plans'), row['metrics']
        print(f'PASS: {schema}: unsafe pressured maneuvers are rejected by rollout before execution')
        trial(enabled=True, schema=schema, scenario='open-shot', cut_fixture=True, interrupt=2)
        print(f'PASS: {schema}: RAM ownership change interrupts native maneuver with neutral input')
    baseline, disabled = paired_trial(scenario='fast')
    assert baseline['actions_sha256'] == disabled['actions_sha256']
    assert not disabled['metrics'].get('plans'), disabled['metrics']
    print('PASS: unsafe high-speed approach rejects dekes and retains identical baseline actions')


def run(args):
    if args.seeds < 1 or args.jitter < 0 or args.seed < 0 or args.seed + args.seeds > 2**32:
        raise ValueError('Use a positive seed count, nonnegative jitter and uint32 seeds.')
    results = []
    for scenario in args.scenarios:
        for side in (1, 2):
            for seed in range(args.seed, args.seed + args.seeds):
                direction, handedness = (-1 if seed % 2 else 1), (seed // 2) % 2
                pair = paired_trial(side=side, direction=direction, scenario=scenario, seed=seed,
                                    handedness=handedness, jitter=args.jitter)
                results.extend(pair)
    summary = {}
    for enabled in (False, True):
        rows = [row for row in results if row['enabled'] == enabled]
        metrics = sum((Counter(row['metrics']) for row in rows), Counter())
        summary['deke' if enabled else 'baseline'] = {
            'opportunities': len(rows), 'goals': sum(row['goals'] for row in rows),
            'shots': sum(row['shots'] for row in rows),
            'goalie_contacts': sum(row['goalie_contacts'] for row in rows), 'metrics': dict(metrics),
        }
    root = Path(__file__).resolve().parents[2]
    files = ('nhl94_ai/agents/deke.py', 'nhl94_ai/agents/classic_v1.py',
             'nhl94_ai/agents/cross_crease.py', 'nhl94_ai/agents/offense.py', 'nhl94_ai/agents/motion.py',
             'nhl94_ai/agents/skating.py',
             'nhl94_ai/agents/passing.py', 'nhl94_ai/game/ram.py', 'nhl94_ai/game/state.py',
             'nhl94_ai/env/target_control.py', 'nhl94_ai/env/actions.py',
             'tests/integration/deke.py', 'tests/integration/cross_crease.py',
             'tests/integration/classic_offense.py', 'nhl94_ai/tasks/defense_setup.py')
    sources = {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in files}
    report = {'protocol': 'classic-skating-deke-scenarios-v2', 'settings': vars(args), 'sources': sources,
              'limitations': ['Short ordinary-game opportunities, not full matches or goal probabilities.',
                              'Policies share exact restored initial snapshots, not identical later CPU decisions.',
                              'ROM seeds vary decisions, not saved hot/cold tables or player ratings.',
                              'A deke abort does not end the opportunity; ordinary fallback play continues.',
                              'Rollout shares live predicates, but future CPU choices, puck sprites and saves are uncertain.',
                              'Future facing changes use a conservative ROM-derived all-direction release envelope.',
                              'Goalie contacts count fresh native impulses, not unique collision episodes.'],
              'summary': summary, 'trials': results}
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(summary, indent=2))
    return report


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scenarios', nargs='+', choices=SCENARIOS, default=list(SCENARIOS))
    parser.add_argument('--seed', type=int, default=71000)
    parser.add_argument('--seeds', type=int, default=2)
    parser.add_argument('--jitter', type=float, default=0)
    parser.add_argument('--output')
    parser.add_argument('--check', action='store_true', help='Also assert fixed native execution/safety contracts')
    return parser


if __name__ == '__main__':
    arguments = build_parser().parse_args()
    if arguments.check:
        native_contracts()
    run(arguments)

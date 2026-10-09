"""Matched native support-response traces for the opt-in chance-creation model."""
import argparse
from functools import partial
import hashlib
from itertools import product
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from nhl94_ai.agents.carry import CARRY_FRAMES, carry_path
from nhl94_ai.agents.classic_v1 import ClassicAIV1Model
from nhl94_ai.agents.motion import velocity
from nhl94_ai.agents.responses import response_future
from nhl94_ai.env.factory import make_retro
from nhl94_ai.env.actions import HockeyActionController
from nhl94_ai.evaluation.benchmark import RAM, away_view, update_state
from nhl94_ai.game.constants import GameConsts as Buttons
from nhl94_ai.game.state import NHL94GameState
from nhl94_ai.tasks.cross_crease_setup import OBJECT_BASE, OBJECT_STRIDE, prepare_isolated_crossing
from nhl94_ai.tasks.defense_setup import _place_object, _rebuild_object_order, _reset_player


POSITION_TOLERANCE = 4
RESPONSE_SIGNAL_THRESHOLD = 0.5


def response_comparison(rows):
    if not rows or len(rows) % 2:
        raise ValueError('Response evidence requires complete paired native traces.')
    pairs = []
    for left, right in zip(rows[::2], rows[1::2]):
        keys = ('style', 'width', 'puck_offset', 'side', 'decision_interval', 'action_type',
                'initial_ram_sha256')
        if any(left[key] != right[key] for key in keys) or (left['direction'], right['direction']) != (-1, 1):
            raise ValueError('Conditional response traces must share their initial state and fixture.')
        native = [a - b for a, b in zip(left['actual'], right['actual'])]
        predicted = [a - b for a, b in zip(left['predicted'], right['predicted'])]
        pairs.append({**{key: left[key] for key in keys},
                      'native_delta': native, 'predicted_delta': predicted,
                      'delta_error': max(abs(a - b) for a, b in zip(native, predicted))})
    native_signal = [max(map(abs, pair['native_delta'])) >= RESPONSE_SIGNAL_THRESHOLD for pair in pairs]
    modeled_signal = [max(map(abs, pair['predicted_delta'])) >= RESPONSE_SIGNAL_THRESHOLD for pair in pairs]
    return {'signal_threshold': RESPONSE_SIGNAL_THRESHOLD, 'pairs': pairs,
            'native_signal_pairs': sum(native_signal), 'modeled_signal_pairs': sum(modeled_signal),
            'missed_signal_pairs': sum(native and not predicted
                                       for native, predicted in zip(native_signal, modeled_signal)),
            'spurious_signal_pairs': sum(predicted and not native
                                         for native, predicted in zip(native_signal, modeled_signal)),
            'maximum_delta_error': max(pair['delta_error'] for pair in pairs)}


def _timing_choice(_state, frame, *, target):
    return ('create-chance', target, None) if frame < CARRY_FRAMES + 1 else None


def run(args):
    env = make_retro(game='NHL94-Genesis-v0', state='PenguinsVsSenators.DefenseZone', num_players=1)
    rows = []
    try:
        for name, (address, kind) in RAM.items():
            env.data.set_variable(name, {'address': address, 'type': kind})
        cases = ((35, 'lateral-offset'), (45, 'forward-offset'), (55, 'forward-offset'))
        for (width, puck_offset), style in product(cases, ('center', 'wing', 'defense', 'nearest')):
            for side, interval, schema in product((1, 2), (1, 4, 10), ('FILTERED', 'HOCKEY_INTENT_DPAD')):
                env.reset(seed=94001)
                carrier, _, inactive = prepare_isolated_crossing(env, side=side, width=width, depth=160, speed=0)
                sign = 1 if side == 1 else -1
                actor_side = 3 - side if style == 'nearest' else side
                receiver = next(slot for slot in range((actor_side - 1) * 6, (actor_side - 1) * 6 + 5)
                                if slot != carrier)
                memory = env.data.memory
                base = OBJECT_BASE + receiver * OBJECT_STRIDE
                role, assignment = {'center': (4, 6), 'wing': (5, 4), 'defense': (1, 1),
                                    'nearest': (3, 0x11)}[style]
                memory.assign(base + 0x34, '>i2', role)
                position = ((45, sign * 200) if style == 'nearest' else
                            (-30, sign * 110) if style == 'defense' else (0, sign * 190))
                _reset_player(memory, receiver, position,
                              6 if style == 'nearest' else 0 if side == 1 else 4, assignment, False)
                memory.assign(base + 0x62, '|u1', memory.extract(base + 0x62, '|u1') & ~2)
                destination = (60, 230) if style == 'wing' else (0, 0) if style == 'nearest' else (0, 170)
                for offset, kind, value in ((0x40, '|i1', 30), (0x42, '|i1', 0),
                                             (0x43, '|u1', 8), (0x44, '>i2', destination[0]),
                                             (0x46, '>i2', destination[1]), (0x48, '>i2', 16)):
                    memory.assign(base + offset, kind, value)
                memory.assign(OBJECT_BASE + carrier * OBJECT_STRIDE + 0x54, '>u4',
                              (0 if side == 1 else 4) << 16)
                if puck_offset == 'forward-offset':
                    _place_object(memory, 14, (-width, sign * 170))
                _rebuild_object_order(memory)
                snapshot = env.em.get_state()
                initial = hashlib.sha256(env.get_ram().tobytes()).hexdigest()
                for direction in (-1, 1):
                    env.em.set_state(snapshot)
                    env.data.update_ram()
                    assert hashlib.sha256(env.get_ram().tobytes()).hexdigest() == initial
                    state = NHL94GameState(5)
                    update_state(state, env.data.lookup_all(), env)
                    view = state if side == 1 else away_view(state)
                    player = view.team1.get_player_by_scnum(carrier)
                    target = player.x + direction * 26, player.y + sign * 8
                    path = carry_path(player, target, decision_interval=interval)
                    predicted, diagnostics = response_future(view, path)
                    assert receiver in diagnostics['response_slots'], diagnostics
                    model = ClassicAIV1Model(SimpleNamespace(
                        action_type=schema, chance_creation=True, one_timers=False))
                    model.offense.chance_owner, model.offense.chance_until = carrier, CARRY_FRAMES + 1
                    model.offense.choose = partial(_timing_choice, target=target)
                    context = SimpleNamespace(action_type=schema, game_state=view)
                    processor = HockeyActionController(context)
                    macro = processor._new_action_state()
                    idle_positions = {
                        slot: tuple(memory.extract(OBJECT_BASE + slot * OBJECT_STRIDE + offset, '>i4')
                                    for offset in (0, 0x14))
                        for slot in inactive if slot != receiver
                    }
                    trajectory = []
                    for elapsed in range(CARRY_FRAMES):
                        view = state if side == 1 else away_view(state)
                        action = model.predict_frame(view, frame_skip=interval)[0]
                        assert model._last_decision == 'create-chance'
                        context.game_state = view
                        buttons = processor._process_action(action, macro)[0] if schema == 'HOCKEY_INTENT_DPAD' else action
                        assert not buttons[Buttons.INPUT_B] and not buttons[Buttons.INPUT_C]
                        *_, info = env.step(buttons)
                        update_state(state, info, env)
                        view = state if side == 1 else away_view(state)
                        actual = (view.team2 if style == 'nearest' else view.team1).get_player_by_scnum(receiver)
                        assert view.engine.puck_owner == carrier, ('lost carrier', side, interval, elapsed)
                        assert actual.assignment == assignment, (
                            'support assignment changed', style, side, interval, elapsed)
                        assert all(tuple(memory.extract(OBJECT_BASE + slot * OBJECT_STRIDE + offset, '>i4')
                                         for offset in (0, 0x14)) == point
                                   for slot, point in idle_positions.items())
                        trajectory.append({'position': (actual.precise_x, actual.precise_y),
                                           'velocity': velocity(actual), 'steering': actual.steering})
                    actual = (view.team2 if style == 'nearest' else view.team1).get_player_by_scnum(receiver)
                    forecast = (predicted.team2 if style == 'nearest' else predicted.team1).get_player_by_scnum(receiver)
                    error = max(abs(actual.precise_x - forecast.precise_x),
                                abs(actual.precise_y - forecast.precise_y))
                    model.predict_frame(view, frame_skip=interval)
                    assert model.scheduler.frames == CARRY_FRAMES + 1
                    assert model._last_decision != 'create-chance', ('cached setup exceeded deadline', schema, interval)
                    rows.append({'style': style, 'width': width, 'puck_offset': puck_offset,
                                 'side': side, 'decision_interval': interval, 'direction': direction, 'action_type': schema,
                                 'initial_ram_sha256': initial, 'target': target,
                                 'predicted': (forecast.precise_x, forecast.precise_y),
                                 'actual': (actual.precise_x, actual.precise_y),
                                 'position_error': error, 'trajectory': trajectory,
                                 'response': diagnostics, 'deadline_interrupt_frame': model.scheduler.frames})
        root = Path(__file__).resolve().parents[2]
        names = ('agents/responses.py', 'agents/carry.py', 'agents/skating.py',
                 'agents/classic_v1.py', 'agents/offense.py', 'env/actions.py',
                 'game/ram.py', 'game/state.py', 'evaluation/chance_creation_probe.py')
        report = {'protocol': 'nhl94-chance-response-probe-v1', 'traces': rows,
                  'maximum_position_error': max(row['position_error'] for row in rows),
                  'conditional_responses': response_comparison(rows),
                  'sources': {f'nhl94_ai/{name}': hashlib.sha256(
                      (root / 'nhl94_ai' / name).read_bytes()).hexdigest() for name in names},
                  'limitations': ['Isolated stable support/containment, not full-team tactic strength.',
                                  'Movement/deadline choices are injected fixture controls, not normal-policy decisions.',
                                  'Future sprite changes, assignment changes and RNG are not simulated.']}
        if args.output:
            Path(args.output).write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
        print(f'{len(rows)} native response traces; maximum position error '
              f'{report["maximum_position_error"]:.4f} rink units; '
              f'{report["conditional_responses"]["missed_signal_pairs"]} conditional signals missed.')
        if report['maximum_position_error'] > POSITION_TOLERANCE:
            raise AssertionError('Native support forecast exceeds the declared four-unit probe tolerance.')
        return report
    finally:
        env.close()


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output')
    return parser


if __name__ == '__main__':
    run(build_parser().parse_args())

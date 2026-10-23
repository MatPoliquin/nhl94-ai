"""Bounded native held-C timing and Classic cross-crease execution fixtures."""
from types import SimpleNamespace
import hashlib
import math

import numpy as np

from nhl94_ai.agents.classic_v1 import ClassicAIV1Model
from nhl94_ai.agents.cross_crease import SEQUENCE_FRAMES
from nhl94_ai.agents.goalie import DIVE_ANIMATION, SAVE_ANIMATIONS
from nhl94_ai.env.actions import HockeyActionController
from nhl94_ai.env.factory import make_retro
from nhl94_ai.evaluation.benchmark import RAM, away_view, update_state
from nhl94_ai.game.constants import GameConsts as Buttons
from nhl94_ai.game.ram import select_cpu_side
from nhl94_ai.game.state import NHL94GameState
from nhl94_ai.tasks.defense_setup import _place_object, _rebuild_object_order, _reset_player
from tests.integration.classic_offense import _feint_setup


def _crossing_setup(env, *, side=1, direction=1, power=30, handedness=0, depth=216,
                    goalie_depth=250, prepare=False, width=None, net_front=False):
    _feint_setup(env, 'NHL94-Genesis-v0')
    env.data.update_ram()
    select_cpu_side(env.data, env.data.lookup_all(), side)
    memory = env.data.memory
    actual = memory.extract(0xFFC320, '>i2')
    width = (55 if prepare else 35) if width is None else width
    sign = 1 if side == 1 else -1
    for team in (1, 2):
        teammates = team == side
        others = iter(((-100, -120), (-55, -120), (55, -120), (100, -120))
                      if teammates else ((-100, -180), (-55, -180), (0, -180), (55, -180), (100, -180)))
        nearest = actual if teammates else (team - 1) * 6
        for slot in range((team - 1) * 6, (team - 1) * 6 + 5):
            point = (-direction * width, sign * depth) if slot == actual else next(others)
            if slot != actual:
                point = point[0], sign * point[1]
            role = memory.extract(0xFFB04A + slot * 0x80 + 0x34, '>i2')
            assignment = (1 if teammates else 2) if role in (1, 2) else (
                (6 if teammates else 5) if role == 4 else (4 if teammates else 3))
            temporary = (0x10, 0x11) if slot == actual else ((0x11,) if slot == nearest else ())
            _reset_player(memory, slot, point, 2 if direction > 0 else 6,
                          assignment, slot == actual, temporary)
            if net_front and slot != actual and slot % 6 == 2:
                _place_object(memory, slot, (direction * (28 if teammates else -28), sign * 244))
        goalie = (team - 1) * 6 + 5
        _reset_player(memory, goalie, (0, -sign * 250) if teammates else (-direction * 12, sign * goalie_depth),
                      0 if team == 1 else 4, 0x0E, False)
    base = 0xFFB04A + actual * 0x80
    memory.assign(base + 0x28, '>i2', 0 if prepare else direction * 6500)
    memory.assign(base + 0x6C, '|u1', power)
    memory.assign(base + 0x6D, '|u1', 20)
    memory.assign(base + 0x76, '|u1', handedness)
    _place_object(memory, 14, (-direction * (width - 10), sign * depth))
    memory.assign(0xFFB7AA, '>i2', actual)
    memory.assign(0xFFBEDA, '>i2', actual)
    _rebuild_object_order(memory)
    env.data.set_value('bench_rng', 7)


def _fixture(*, side=1, direction=1, power=30, handedness=0, schema='FILTERED', interval=4,
             hold=None, depth=216, goalie_depth=250, prepare=False, width=None, net_front=False):
    env = make_retro(game='NHL94-Genesis-v0', state='PenguinsVsSenators.DefenseZone', num_players=1)
    try:
        for name, (address, kind) in RAM.items():
            env.data.set_variable(name, {'address': address, 'type': kind})
        env.reset(seed=7)
        _crossing_setup(env, side=side, direction=direction, power=power, handedness=handedness,
                        depth=depth, goalie_depth=goalie_depth, prepare=prepare, width=width,
                        net_front=net_front)
        *_, info = env.step(np.zeros(12, dtype=np.int8))
        state = NHL94GameState(5)
        model = ClassicAIV1Model(SimpleNamespace(action_type=schema, cross_crease=True))
        context = SimpleNamespace(action_type=schema, game_state=state)
        processor = HockeyActionController(context)
        macro = processor._new_action_state()
        digest = hashlib.sha256()
        first_windup = first_commit = release = None
        held, minimum, shot_x, release_x, release_body_x = 0, math.inf, None, None, None
        actual = info['bench_control1']
        before_shots = info[f'bench_shots{side}']
        had_windup = False
        for frame in range(SEQUENCE_FRAMES + 2):
            update_state(state, info, env)
            view = state if side == 1 else away_view(state)
            player, goalie = view.team1.get_player_by_scnum(actual), view.team2.goalie
            minimum = min(minimum, math.dist((player.x, player.y), (goalie.x, goalie.y)))
            assert not (goalie.contact_player == actual and goalie.contact_impact), (frame, side, minimum)
            if view.is_shooting and view.engine.puck_owner == actual:
                first_windup = frame if first_windup is None else first_windup
                had_windup = True
                if first_commit is None and goalie.live_anim in (*SAVE_ANIMATIONS, DIVE_ANIMATION):
                    first_commit = frame
            if had_windup and view.engine.puck_owner < 0 and release is None:
                release, release_x, release_body_x = frame, view.puck.x, player.x
            if view.team1.stats.shots > before_shots and shot_x is None:
                shot_x = player.x
            if hold is None:
                action = model.predict_frame(view, interval)[0]
                if frame == 0:
                    assert model.cross_crease.plan is not None, model.offense_diagnostics
                context.game_state = view
                buttons = processor._process_action(action, macro)[0] if schema != 'FILTERED' else action
            else:
                buttons = np.zeros(12, dtype=np.int8)
                buttons[Buttons.INPUT_C] = int(frame < hold)
                buttons[Buttons.INPUT_RIGHT if direction > 0 else Buttons.INPUT_LEFT] = 1
            held += bool(buttons[Buttons.INPUT_C] and view.engine.puck_owner == actual)
            digest.update(buttons.tobytes())
            if hold is None and model.cross_crease.events:
                break
            if hold is not None and frame >= 70:
                break
            *_, info = env.step(buttons)
        return {
            'release_frame': release, 'release_x': release_x, 'shot_x': shot_x,
            'release_body_x': release_body_x,
            'windup_frame': first_windup, 'commit_frame': first_commit,
            'held_frames': held, 'minimum_separation': minimum,
            'shots': view.team1.stats.shots - before_shots,
            'goals': view.team1.stats.score,
            'actions_sha256': digest.hexdigest(),
            'metrics': dict(model.cross_crease.metrics), 'events': model.cross_crease.events,
            'diagnostics': dict(model.cross_crease.diagnostics),
        }
    finally:
        env.close()


def native_timing():
    for handedness in (0, 1):
        tap = _fixture(hold=4, handedness=handedness, depth=218)
        charged = _fixture(hold=60, handedness=handedness, depth=218)
        weak = _fixture(hold=60, power=10, handedness=handedness, depth=218)
        assert tap['release_frame'] < weak['release_frame'] < charged['release_frame'] < 60, (
            tap, weak, charged)
        assert charged['release_x'] > tap['release_x'] + 25, (tap, charged)
        assert charged['commit_frame'] is not None and charged['commit_frame'] < charged['release_frame'], charged
        print(f"PASS: hand={handedness}: native releases tap/weak/charged at "
              f"{tap['release_frame']}/{weak['release_frame']}/{charged['release_frame']}; "
              f"goalie commits before charged release")


def classic_execution():
    goals = 0
    for side in (1, 2):
        for direction in (-1, 1):
            for schema in ('FILTERED', 'HOCKEY_INTENT_DPAD'):
                rows = [_fixture(side=side, direction=direction, schema=schema, interval=interval)
                        for interval in (1, 4, 10)]
                assert len({row['actions_sha256'] for row in rows}) == 1, rows
                for row in rows:
                    assert row['held_frames'] >= 8, row
                    assert row['release_x'] * direction > 0, row
                    assert row['release_body_x'] * direction > 0, row
                    assert row['shots'] == 1 and row['metrics']['recorded_shots'] == 1, row
                    assert row['commit_frame'] is not None and row['commit_frame'] < row['release_frame'], row
                    assert row['metrics'].get('goalie_contacts', 0) == 0 and row['minimum_separation'] > 16, row
                goals += rows[0]['goals']
                print(f"PASS: side={side}, cross={direction}, {schema}: held-C crossing, "
                      f"pre-release commitment, confirmed shot, no contact; intervals 1/4/10 identical")
    assert goals > 0, 'Crossing fixtures never produced an actual goal'


def prepared_execution():
    for schema in ('FILTERED', 'HOCKEY_INTENT_DPAD'):
        row = _fixture(prepare=True, depth=214, schema=schema)
        assert row['events'][0]['plan']['approach_frames'] > 16, row
        assert row['held_frames'] >= 8 and row['shots'] == 1, row
        assert row['release_x'] > 0, row
        assert row['commit_frame'] is not None and row['commit_frame'] < row['release_frame'], row
        assert not row['metrics'].get('goalie_contacts', 0), row
        print(f'PASS: {schema}: builds momentum with ordinary skating before the held-C crossing')


def wing_execution():
    goals = 0
    for side in (1, 2):
        for direction in (-1, 1):
            for schema in ('FILTERED', 'HOCKEY_INTENT_DPAD'):
                row = _fixture(prepare=True, width=100, depth=180, side=side,
                               direction=direction, schema=schema)
                event = row['events'][0]
                assert event['start_position'] == (-direction * 100, (1 if side == 1 else -1) * 180), row
                assert event['plan']['approach_frames'] > 48 and event.get('approach_elapsed', 0) > 30, row
                assert row['held_frames'] >= 8 and row['shots'] == 1, row
                assert row['release_x'] * direction > 0, row
                assert event['windup'] and event['shot'] and event['committed'], row
                assert event['commit_frame'] - 1 < row['release_frame'], row
                assert not event['contact'] and row['minimum_separation'] > 16, row
                goals += row['goals']
                print(f'PASS: side={side}, cross={direction}, {schema}: stationary wide-wing '
                      'start, planned approach, observed commitment and attributed shot')
    deep = _fixture(prepare=True, width=100, depth=160)
    assert deep['shots'] == 1 and deep['release_x'] > 0 and deep['goals'] == 1, deep
    assert goals + deep['goals'] > 0, 'Wing-origin crossings never produced an actual goal'
    print('PASS: deeper (-100, 160) wing start plans its approach and scores an attributed goal')
    missed = _fixture(prepare=True, width=100, depth=160, side=2, direction=-1)
    assert missed['events'][0]['outcome'] == 'crossing-window-missed', missed
    assert not missed['held_frames'] and not missed['metrics'].get('goalie_contacts', 0), missed
    print('PASS: a missed deep-wing window exits without forcing an unsafe C press')
    traffic = _fixture(prepare=True, width=100, depth=160, net_front=True)
    event = traffic['events'][0]
    assert event['start_position'] == (-100, 160), traffic
    assert event['outcome'] == 'approach-invalidated' and not event['pressed'], traffic
    print('PASS: net-front players allow a plan; live pursuit into its route cancels the approach')


if __name__ == '__main__':
    native_timing()
    classic_execution()
    prepared_execution()
    wing_execution()

"""Actual goalie contact and recorded shots during coherent approach fixtures."""
from dataclasses import replace
from functools import partial
import math

from nhl94_ai.agents.classic_v1 import ClassicAIV1Model
from nhl94_ai.env.factory import build_single_nhl94_env
from nhl94_ai.evaluation.play import parse_cmdline
from nhl94_ai.tasks.defense_setup import _place_object, _rebuild_object_order
from nhl94_ai.tasks.registry import TASKS, get_task, register_task
from nhl94_ai.training.datasets import get_game_state
from tests.integration.classic_offense import _feint_setup


def _approach_setup(env, game, *, x, y, goalie_x, goalie_y):
    _feint_setup(env, game)
    memory = env.data.memory
    actual = memory.extract(0xFFC320, '>i2')
    _place_object(memory, actual, (x, y))
    memory.assign(0xFFB04A + actual * 0x80 + 0x2A, '>i2', 6500)
    memory.assign(0xFFB04A + actual * 0x80 + 0x54, '>u2', 0)
    for slot in range(6, 11):
        _place_object(memory, slot, (110, -180))
    _place_object(memory, 11, (goalie_x, goalie_y))
    _place_object(memory, 14, (x, y + 10))
    _rebuild_object_order(memory)


def goalie_approach():
    name = 'ClassicGoalieApproachProbe'
    for x, y, goalie_x, goalie_y in ((0, 180, 0, 228), (-35, 185, -30, 232)):
        register_task(name, replace(
            get_task('DefenseZone'),
            initialize=partial(_approach_setup, x=x, y=y, goalie_x=goalie_x, goalie_y=goalie_y),
            reward=lambda _: 0.0, done=lambda _: False))
        try:
            for schema in ('FILTERED', 'HOCKEY_INTENT_DPAD'):
                args = parse_cmdline([
                    '--mode=model_vs_game', '--nn=ClassicAIV1', '--env=NHL94-Genesis-v0',
                    '--state=PenguinsVsSenators.DefenseZone', f'--rf={name}', f'--action_type={schema}',
                ])
                env = build_single_nhl94_env(args, {'clip_reward': False}, use_frame_skip=False)
                try:
                    env.reset(seed=7)
                    state = get_game_state(env)
                    actual = state.team1.defense_control
                    before = state.team1.stats.shots
                    memory = env.unwrapped.data.memory
                    model = ClassicAIV1Model(args)
                    minimum = math.inf
                    for frame in range(70):
                        state = get_game_state(env)
                        player, goalie = state.team1.get_player_by_scnum(actual), state.team2.goalie
                        minimum = min(minimum, math.dist((player.x, player.y), (goalie.x, goalie.y)))
                        contact = memory.extract(0xFFB04A + 11 * 0x80 + 0x2E, '>i2') == actual
                        impact = memory.extract(0xFFB04A + 11 * 0x80 + 0x32, '>u2')
                        assert not contact or impact == 0, (schema, frame, minimum, model._last_decision)
                        action = model.predict_frame(state)[0]
                        if frame == 0:
                            assert model._last_decision == 'goalie-avoid', model.offense_diagnostics
                        env.step(action)
                    assert minimum > 16, (schema, minimum)
                    shots = get_game_state(env).team1.stats.shots - before
                    if x != 0:
                        assert shots > 0, (schema, 'Goalie avoidance suppressed finishing')
                    print(f'PASS: {schema} approach x={x} avoids actual goalie contact; '
                          f'minimum separation={minimum:.1f}, recorded shots={shots}')
                finally:
                    env.close()
        finally:
            TASKS.pop(name)


if __name__ == '__main__':
    goalie_approach()

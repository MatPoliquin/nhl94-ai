"""Native carrier-trap fixtures. Setup moves actors once; play uses buttons only."""
from types import SimpleNamespace
import unittest

from nhl94_ai.agents.defense import eligible
from nhl94_ai.agents.teammates import TeammateCoordinator
from nhl94_ai.evaluation.teammates import DEFAULT_STATE, make_match_env, prepare_match
from nhl94_ai.tasks.defense_setup import _place_object, _rebuild_object_order


def args(size, side='home'):
    return SimpleNamespace(team_size=size, side=side, mode='team_vs_game',
                           action_type='FILTERED', goalie_policy='off',
                           hyperparams_dict={'frame_skip': 4})


def fixture(env, size, side='home'):
    physical = 1 if side == 'home' else 2
    own = range((physical - 1) * 6, (physical - 1) * 6 + 5)
    other = range((2 - physical) * 6, (2 - physical) * 6 + 5)
    info = prepare_match(env, seed=95, seconds=120, team_size=size, side=physical)
    agent = TeammateCoordinator(args(size, side))
    for _ in range(5000):
        action = agent.act(info).action
        state = agent.states[0]
        slots = agent.last_slots
        if (not state.engine.clock_stopped and state.engine.puck_owner in other
                and len(set(slots)) == size and all(slot is not None and slot in own for slot in slots)
                and all(eligible(state.team1.get_player_by_scnum(slot)) for slot in slots)):
            break
        _, _, _, truncated, info = env.step(action)
        if truncated:
            raise AssertionError('Native game ended before a carrier fixture was available')
    else:
        raise AssertionError('No live opponent possession for fixture')
    memory = env.data.memory
    carrier = state.engine.puck_owner
    # Preserve real roster attributes, native carrier ownership and controller selections.
    friends = dict(zip(slots, ((-24, -24), (24, 24), (0, -72))))
    available = [slot for slot in own if slot not in slots]
    for slot, point in zip(available, ((0, -72), (-100, -180), (100, -180)) if size == 2
                           else ((-100, -180), (100, -180))):
        friends[slot] = point
    enemies = {carrier: (0, 0)}
    enemies.update(zip((slot for slot in other if slot != carrier),
                       ((-100, 100), (100, 100), (-50, 180), (50, 180))))
    sign = 1 if state.team1.net.y < 0 else -1
    goalies = {state.team1.goalie_scnum(): (0, -252), state.team2.goalie_scnum(): (0, 252)}
    for slot, (x, y) in {**friends, **enemies, **goalies, 14: (0, 0)}.items():
        _place_object(memory, slot, (x, y * sign))
    for slot, heading in ((slots[0], 1), (slots[1], 5), (carrier, 4)):
        memory.assign(0xFFB04A + slot * 0x80 + 0x54, '>u4',
                      (heading if sign == 1 else (4 - heading) % 8) << 16)
    _rebuild_object_order(memory)
    env.data.update_ram()
    return env.data.lookup_all()


class NativePressureTests(unittest.TestCase):
    def test_two_and_three_bots_execute_trap_from_native_feedback(self):
        for size, side in ((2, 'home'), (3, 'home'), (2, 'away'), (3, 'away')):
            with self.subTest(size=size, side=side):
                env = make_match_env(DEFAULT_STATE, team_size=size)
                try:
                    info = fixture(env, size, side)
                    agent = TeammateCoordinator(args(size, side))
                    initial = agent.act(info)
                    self.assertEqual(agent.last_roles, ['contain', 'close'] + (['cover'] if size == 3 else []),
                                     initial.diagnostics)
                    if size == 3:
                        self.assertFalse(agent.previous[2, 0] or agent.previous[2, 8])
                    frames = safety_exits = 0
                    for _ in range(48):
                        # No setup writes after this point: all movement and contact is native.
                        _, _, _, truncated, info = env.step(initial.action)
                        initial = agent.act(info)
                        if 'contain' in agent.last_roles:
                            frames += 1
                            self.assertLessEqual(sum(bool(a[0] or a[8]) for a in agent.previous), 1)
                        elif initial.diagnostics['teammates']['pressure'].get('reason') == 'unavailable-or-missing-feedback':
                            safety_exits += 1
                            self.assertFalse(any(role in ('contain', 'close', 'cover') for role in agent.last_roles))
                        if truncated:
                            break
                    # A native shooting animation can lock an attacker on the
                    # next frame; the trap must then release its assignments.
                    if side == 'away' and size == 2 and frames == 0:
                        self.assertGreater(safety_exits, 0)
                    else:
                        self.assertGreater(frames, 0)
                    self.assertEqual(env.data.lookup_all()['defense_team1'], 1 if side == 'home' else 2)
                finally:
                    env.close()


if __name__ == '__main__':
    unittest.main()

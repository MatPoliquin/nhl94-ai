"""ROM-backed defensive reset invariants. Run explicitly; does not train or save models."""
import math

import numpy as np

from nhl94_ai.config import load_hyperparams, default_config_path, resolve_hyperparams_for_model
from nhl94_ai.env.factory import build_single_nhl94_env
from nhl94_ai.evaluation.play import parse_cmdline
from nhl94_ai.tasks.defensezone import isdone_defensezone
from nhl94_ai.training.datasets import get_game_state


def main():
    args = parse_cmdline([
        '--mode=model_vs_game', '--env=NHL94-Genesis-v0', '--nn=MlpPolicy',
        '--rf=DefenseZone', '--state=PenguinsVsSenators.DefenseZone', '--action_type=FILTERED',
    ])
    params = resolve_hyperparams_for_model(load_hyperparams(default_config_path('nhl94')), args.nn)
    env = build_single_nhl94_env(args, params, use_frame_skip=False, use_sticky_action=False)
    try:
        signatures, owners, gaps = set(), set(), []
        reference = None
        for seed in [*range(200), 0]:
            observation, info = env.reset(seed=seed)
            observation = np.asarray(observation)
            state = get_game_state(env)
            assert observation.shape == (310,)
            assert not isdone_defensezone(state), f'Already-terminal reset: seed={seed}'
            owner = info['puck_owner']
            assert owner in range(6, 11), f'Not opponent possession: seed={seed}, owner={owner}'
            owners.add(owner)
            carrier = state.team2.get_player_by_scnum(owner)
            assert math.dist((carrier.x, carrier.y), (state.puck.x, state.puck.y)) < 24
            assert -182 <= carrier.y <= -123 and abs(carrier.x) <= 97
            positions = [(p.x, p.y) for team in (state.team1, state.team2) for p in [*team.players, team.goalie]]
            assert all(-102 <= x <= 102 and -252 <= y <= 252 for x, y in positions)
            assert min(math.dist(a, b) for i, a in enumerate(positions) for b in positions[i + 1:]) >= 18
            defenders = [p for p in state.team1.players if p.y < carrier.y]
            gap = min(math.dist((p.x, p.y), (carrier.x, carrier.y)) for p in defenders)
            assert gap <= 56, f'No reachable goal-side defender: seed={seed}, gap={gap}'
            gaps.append(gap)
            assert abs(state.team1.goalie.x) <= 16 and -252 <= state.team1.goalie.y <= -248
            assert state.team1.control == env.unwrapped.data.memory.extract(0xFFC320, '>i2') + 1
            signature = tuple(positions)
            if seed == 0:
                if reference is None:
                    reference = observation.copy()
                else:
                    np.testing.assert_array_equal(observation, reference)
            signatures.add(signature)
            for _ in range(8):
                _, _, terminated, truncated, _ = env.step(np.zeros(12, dtype=np.int8))
                if terminated or truncated:
                    break
        assert len(signatures) == 200
        assert owners == set(range(6, 11))
        print(f'PASS: 200 distinct seeded defensive starts; all five carriers; '
              f'goal-side defender gap {min(gaps):.1f}-{max(gaps):.1f}; repeatable observations.')
    finally:
        env.close()


if __name__ == '__main__':
    main()

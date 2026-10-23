"""ROM regressions for reset-induced launches and missing CPU possession handoffs."""
import math

import numpy as np

from nhl94_ai.env.factory import build_single_nhl94_env
from tests.integration.target_control import _args


def main():
    for mode in ('FILTERED', 'TARGET_POSITION'):
        env = build_single_nhl94_env(_args(action_type=mode), {'clip_reward': False}, use_frame_skip=False)
        frames, handoffs, max_stale, max_speed = 0, 0, 0, 0.0
        try:
            memory = env.unwrapped.data.memory
            for seed in range(64):
                _, info = env.reset(seed=seed)
                rng = np.random.default_rng(seed)
                seen_owners = {info['puck_owner']}
                previous_owner, stale = info['puck_owner'], 0
                for frame in range(1200):
                    if frame % 4 == 0:
                        action = rng.normal(size=2).clip(-1, 1) if mode == 'TARGET_POSITION' else np.zeros(12, dtype=np.int8)
                    _, _, done, truncated, info = env.step(action)
                    for slot in range(12):
                        base = 0xFFB04A + slot * 0x80
                        speed = math.hypot(*(memory.extract(base + offset, '>i2') * 17 / 65536 for offset in (0x28, 0x2A)))
                        max_speed = max(max_speed, speed)
                        assert speed < 6, ('Reset-induced launch', mode, seed, frame, slot, speed)
                        expected_radii = (8, 4)
                        assert tuple(memory.extract(base + offset, '>u2') for offset in (0x4A, 0x4C)) == expected_radii
                    owner = info['puck_owner']
                    if owner in range(6, 11):
                        if owner not in seen_owners:
                            handoffs += 1
                            seen_owners.add(owner)
                        base = 0xFFB04A + owner * 0x80
                        index = memory.extract(base + 0x36, '>u2')
                        assignment = memory.extract(base + 0x38 + index, '|u1')
                        stale = stale + 1 if owner == previous_owner and assignment in range(1, 7) else 0
                        max_stale = max(stale, max_stale)
                        assert stale < 60, ('Carrier is stuck in support-player AI', mode, seed, frame, owner, assignment)
                    else:
                        stale = 0
                    previous_owner = owner
                    frames += 1
                    if done or truncated:
                        break
            assert handoffs >= 5, (mode, handoffs)
            print(f'PASS: {mode}: {frames} frames, {handoffs} CPU ownership handoffs; '
                  f'peak speed {max_speed:.2f}, longest carrier handoff delay {max_stale} frames.')
        finally:
            env.close()


if __name__ == '__main__':
    main()

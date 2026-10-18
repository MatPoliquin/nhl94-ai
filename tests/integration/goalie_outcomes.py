"""Native conditional catch/rebound audit, independent of planner selection."""
import argparse
from collections import defaultdict
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path

import numpy as np

from nhl94_ai.agents.goalie_outcomes import body_rebound, catch_probability
from nhl94_ai.agents.goalie_physics import KeeperMotion, PuckFlight
from nhl94_ai.env.factory import make_retro
from nhl94_ai.game.ram import ROM_RNG_ADDRESS
from tests.integration.goalie_physics import buttons
from tests.integration.manual_goalie import STATE, advance, restore, shot_fixture, takeover


def run(output, repeats):
    env = make_retro(game='NHL94-Genesis-v0', state=STATE, num_players=1, goalie_policy='always')
    cases = []
    seeds = np.random.default_rng(193000).integers(1, 2**32, size=repeats).tolist()
    try:
        snapshot = takeover(env)
        for control in (4, 16, 28):
            for speed in (6000, 14000, 22000):
                for save in (False, True):
                    for seed in seeds:
                        restore(env, snapshot)
                        shot_fixture(env, x=8, y=-210, raw_vy=-speed)
                        memory = env.data.memory
                        base = 0xFFB04A+5*0x80
                        memory.assign(base+0x6C, '|u1', control)
                        memory.assign(0xFFB74A+0x18, '>i4', 12*65536)
                        memory.assign(0xFFBF7A, '>i2', -1)
                        memory.assign(ROM_RNG_ADDRESS, '>u4', seed)
                        action = buttons((0, 0), save)
                        # Finish the CPU update containing fixture writes.
                        state, _ = advance(env, action)
                        assert memory.extract(0xFFBF7A, '>i2') != 5
                        result = {'control': control, 'speed': speed, 'requested_save': save,
                                  'seed': seed, 'contact': False}
                        for frame in range(1, 25):
                            before = state
                            flight = PuckFlight.observe(before.puck)
                            state, _ = advance(env, action if frame == 1 else buttons((0, 0)))
                            if memory.extract(0xFFBF7A, '>i2') != 5:
                                continue
                            keeper, puck = KeeperMotion.observe(state.team1.goalie), PuckFlight.observe(state.puck)
                            contact_puck = puck
                            animated = bool(state.team1.goalie.unavailable & 2)
                            probability = catch_probability(control, flight.vx, flight.vy,
                                animated=animated, dive=state.team1.goalie.live_anim == 0x2F4,
                                unavailable=bool(state.team1.goalie.unavailable & 4))
                            incoming_at_contact = replace(contact_puck, vx=flight.vx, vy=flight.vy, vz=flight.vz)
                            predicted = body_rebound(keeper, incoming_at_contact)
                            # a2touchpuck runs before randomd0 and the catch/deflect
                            # branch. A video-frame boundary can split that code.
                            pending_frames = 0
                            while (state.engine.puck_owner != 5 and (puck.vx, puck.vy) == (flight.vx, flight.vy)
                                   and pending_frames < 2):
                                state, _ = advance(env, buttons((0, 0)))
                                puck = PuckFlight.observe(state.puck)
                                pending_frames += 1
                            caught = state.engine.puck_owner == 5
                            rebound_candidates = [predicted, predicted.step()] if predicted else []
                            exact = any((puck.vx, puck.vy, puck.vz) == (p.vx, p.vy, p.vz)
                                        for p in rebound_candidates[:1+bool(pending_frames)])
                            result.update(contact=True, frame=frame, caught=caught,
                                pending_frames=pending_frames,
                                animated=animated, previously_animated=bool(before.team1.goalie.unavailable & 2),
                                height_before=before.puck.height, height_after=state.puck.height,
                                probability=probability, goalie_animation=state.team1.goalie.live_anim,
                                actual=asdict(puck), before=asdict(flight), keeper=asdict(keeper),
                                predicted_rebound=asdict(predicted) if predicted else None,
                                rebound_velocity_exact=None if caught or predicted is None else exact)
                            break
                        cases.append(result)
                    print(json.dumps({'control': control, 'speed': speed, 'save': save,
                                      'completed': len(cases)}), flush=True)
    finally:
        env.close()
    groups = defaultdict(list)
    for row in cases:
        if row['contact']:
            groups[f"pc={row['control']},speed={row['speed']},animated={row['animated']}"] .append(row)
    summary = {key: {'n': len(rows), 'predicted_catches': sum(r['probability'] for r in rows),
                    'actual_catches': sum(r['caught'] for r in rows),
                    'rebound_checks': sum(r['rebound_velocity_exact'] is not None for r in rows),
                    'exact_rebounds': sum(r['rebound_velocity_exact'] is True for r in rows),
                    'stick_eligible': sum(r['height_before'] <= 5 for r in rows),
                    'animation_transition': sum(r['animated'] != r['previously_animated'] for r in rows)}
               for key, rows in groups.items()}
    report = {'protocol': 'native-goalie-contact-outcomes-v2', 'repeats': repeats,
              'scope': 'Conditional body-contact formula audit; injected fixtures, not match strength.',
              'source_hashes': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (
                  Path(__file__), Path('nhl94_ai/agents/goalie_outcomes.py'))},
              'cases': cases, 'summary': summary}
    Path(output).write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--repeats', type=int, default=128)
    args = parser.parse_args()
    run(args.output, args.repeats)

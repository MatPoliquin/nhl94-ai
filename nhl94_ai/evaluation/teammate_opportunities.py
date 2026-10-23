"""Read-only, bounded opportunity estimates for native teammate input traces.

Rejections describe the model's candidate population, not missed native shots.
Live contact and both shot poses remain uncertain until the ROM resolves them.
"""
from copy import copy

from nhl94_ai.game.actors import controllable_skater
from nhl94_ai.game.motion import velocity
from nhl94_ai.agents.passing import evaluate_pass, shot_value
from nhl94_ai.agents.receiving import passive_reception, reachable_reception
from nhl94_ai.game.constants import GameConsts as Buttons


def opportunity_diagnostics(state, diagnostics, previous):
    """Sample all four other skaters every eight frames and every pass frame."""
    plan = diagnostics.get('pending_pass')
    events = diagnostics.get('events', [])
    if diagnostics.get('frame', 0) % 8 and plan is None and not events:
        return None
    attack = 1 if state.team2.net.y > state.team1.net.y else -1
    slots = [row['slot'] for row in diagnostics['controllers']]
    rows = []
    for row in diagnostics['controllers']:
        player = state.team1.get_player_by_scnum(row['slot'])
        if player is None:
            continue
        index = row['controller'] - 1
        rows.append({'slot': row['slot'], 'position': [player.x, player.y],
                     'velocity': velocity(player), 'target': row.get('target'),
                     'role': row['role'], 'buttons': row.get('buttons'),
                     'c_edge': bool(row.get('buttons', [0] * 12)[Buttons.INPUT_C]
                                    and not previous[index][Buttons.INPUT_C])})
    result = {'estimate': 'approximate-not-native-opportunity', 'controllers': rows,
              'native_receiver': state.engine.pass_target, 'owner': state.engine.puck_owner,
              'shot_history_player': state.engine.last_puck_player, 'shooter': state.engine.shot_player,
              'shots': state.team1.stats.shots, 'attempts': state.team1.one_timer_attempts,
              'candidates': [], 'receiving': None}
    carrier = state.team1.get_player_by_scnum(state.engine.puck_owner)
    if carrier is not None and not state.engine.clock_stopped:
        for index, receiver in enumerate(state.team1.players):
            if receiver is carrier:
                continue
            slot = state.team1.skater_scnum_base() + index
            for purpose in ('one-timer', 'support'):
                option, details = evaluate_pass(state, carrier, index, receiver, purpose,
                                                release_prediction=True)
                result['candidates'].append({**details, 'purpose': purpose,
                    'controlled_partner': slot in slots, 'feasible': option is not None,
                    'shot_value': shot_value(state, receiver)})
    receiver = state.team1.get_player_by_scnum(plan['receiver']) if plan is not None else None
    if receiver is not None:
        contact = passive_reception(state, receiver)
        poses = []
        for offset in receiver.one_timer_offsets or ():
            shooter = copy(receiver)
            shooter.stick_x, shooter.stick_y = offset
            poses.append({'offset': offset, 'contact': passive_reception(state, shooter, stick_only=True)})
        forecasts = [pose['contact'] for pose in poses]
        shot = max(forecasts, key=lambda item: item['gap']) if forecasts and all(forecasts) else None
        reasons = []
        if plan['purpose'] != 'one-timer':
            reasons.append('purpose-classification')
        if plan['phase'] != 'flight':
            reasons.append('not-in-flight')
        if not controllable_skater(receiver):
            reasons.append('animation-unavailable')
        if shot is None:
            reasons.append('animation-uncertainty')
        elif shot['gap'] > -2:
            reasons.append('shot-contact-gap')
        elif not 1 < shot['frames'] <= 16:
            reasons.append('cue-timing')
        if not (175 < receiver.y * attack < 245 and abs(receiver.x) < 70):
            reasons.append('shot-position')
        value = shot_value(state, receiver)
        if value <= 0:
            reasons.append('poor-shot-value')
        result['receiving'] = {'plan': plan, 'controlled_partner': plan['receiver_controller'] is not None,
            'passive_contact': contact, 'reachable': reachable_reception(state, receiver),
            'shot_poses': poses, 'shot_value': value, 'rejections': reasons}
    return result

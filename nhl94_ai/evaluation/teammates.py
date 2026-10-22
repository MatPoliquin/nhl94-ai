"""Seeded native-frame teammate matches and opt-in input/decision recordings."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from contextlib import ExitStack
import hashlib
import gzip
import json
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

import numpy as np

from nhl94_ai.agents.classic_v1 import ClassicAIV1Model
from nhl94_ai.agents.teammates import TeammateCoordinator
from nhl94_ai.env.factory import make_retro
from nhl94_ai.env.actions import filter_menu_actions
from nhl94_ai.game.ram import (
    ROM_RNG_ADDRESS, configure_teammates, decode_shot_count, register_teammate_state,
    restore_away_control, select_cpu_side, teammate_controller_info, validate_rom_seed,
)
from nhl94_ai.game.specs import get_game
from nhl94_ai.game.state import NHL94GameState
from nhl94_ai.evaluation.cpu_benchmark import CPU_RAM, lineup
from nhl94_ai.evaluation.offense_metrics import OffenseMetrics
from nhl94_ai.tasks.cross_crease_setup import crossing_touch_player
from nhl94_ai.evaluation.teammate_metrics import ReceptionOutcomes
from nhl94_ai.evaluation.teammate_opportunities import opportunity_diagnostics
from nhl94_ai.evaluation.pressure_metrics import PressureMetrics
from nhl94_ai.evaluation.teammate_diagnostics import TeammateDiagnostics
from nhl94_ai.evaluation.teammate_evidence import PassEvidence


DEFAULT_STATE = 'SabresVsMightyDucks.ManualGoalie.Start'


def make_match_env(state, *, single=False, team_size=2, human=False):
    """Named saves retain their behavior; external native saves are opt-in."""
    path = Path(state).expanduser()
    external = path.is_absolute()
    if external and not path.is_file():
        raise ValueError(f'External starting save does not exist: {path}')
    env = make_retro(game='NHL94-Genesis-v0', state=DEFAULT_STATE if external else state,
                     num_players=1 if single else team_size + human, allow_menu_buttons=True)
    if external:
        try:
            env.initial_state = gzip.decompress(path.read_bytes())
        except Exception:
            env.close()
            raise
    return env


def prepare_match(env, *, seed, seconds, single=False, side=1, team_size=2, human=False):
    """Rebuild the same reset boundary for evaluation and input replay."""
    register_teammate_state(env)
    env.reset(seed=seed)
    if not single:
        if team_size == 3 or human:
            from nhl94_ai.game.multitap import prepare_multitap_save
            prepare_multitap_save(env)
        configure_teammates(env.data, side, team_size=team_size, human=human)
    else:
        native = env.data.lookup_all()
        if (native['defense_team1'], native['defense_team2']) != (1, 0):
            raise ValueError('The single-controller baseline requires a home-versus-CPU save')
        select_cpu_side(env.data, native, side)
        for address in (0xFFD052, 0xFFD05A, 0xFFD05C):
            env.data.memory.assign(address, '>u2', 1)
    memory = env.data.memory
    if (memory.extract(0xFFC466, '>u2'), memory.extract(0xFFC468, '>u2')) != (0, 300):
        raise ValueError('Evaluation requires a five-minute first-period starting save')
    memory.assign(ROM_RNG_ADDRESS, '>u4', seed)
    memory.assign(0xFFC468, '>u2', seconds)
    env.data.update_ram()
    return env.data.lookup_all()


def play_match(*, state=DEFAULT_STATE, seed=94, strategy='coordinated', periods=1,
               seconds=300, frame_skip=4, max_frames=None, trace=None, side='home', support_options=False,
               team_size=2, human_input=None, pressure=True, offpuck=False, intent_arbitration=False,
               defensive_support=False, primary_policy='baseline', role_balance=False, event_trace=None,
               transient_receiving=None, cue_anticipation=False):
    """Return measured outcomes; a bounded/incomplete run never counts as a win."""
    validate_rom_seed(seed)
    human = human_input is not None
    if isinstance(team_size, bool) or team_size not in (2, 3):
        raise ValueError('team_size must be 2 or 3')
    if (team_size == 3 or human) and strategy == 'single':
        raise ValueError('Multitap matches require a teammate strategy')
    if strategy not in ('single', 'independent', 'coordinated'):
        raise ValueError('Unknown teammate strategy')
    if transient_receiving is None:
        transient_receiving = strategy == 'coordinated'
    if support_options and strategy != 'coordinated':
        raise ValueError('Support positioning requires coordinated teammates')
    if any((offpuck, intent_arbitration, defensive_support, role_balance, transient_receiving, cue_anticipation,
            primary_policy != 'baseline')) and strategy != 'coordinated':
        raise ValueError('Teammate variants require coordinated strategy')
    if offpuck and not support_options:
        raise ValueError('Off-puck movement requires scored support positioning')
    if periods not in (1, 3) or not 1 <= seconds <= 300 or periods == 3 and seconds != 300:
        raise ValueError('Use one period of 1..300 seconds or three 300-second regulation periods')
    if isinstance(frame_skip, bool) or not isinstance(frame_skip, int) or frame_skip < 1:
        raise ValueError('frame_skip must be a positive integer')
    if max_frames is not None and max_frames < 1:
        raise ValueError('max_frames must be positive')
    if side not in ('home', 'away'):
        raise ValueError('AI side must be home or away')
    physical_side = 1 if side == 'home' else 2
    args = SimpleNamespace(env='NHL94-Genesis-v0', action_type='FILTERED', goalie_policy='off',
                           hyperparams_dict={'frame_skip': frame_skip}, side=side,
                           teammate_support_options=support_options, teammate_pressure=pressure, team_size=team_size,
                           teammate_offpuck=offpuck,
                           teammate_intent_arbitration=intent_arbitration,
                           teammate_defensive_support=defensive_support,
                           teammate_primary_policy=primary_policy,
                           teammate_role_balance=role_balance, teammate_transient_receiving=transient_receiving,
                           teammate_cue_anticipation=cue_anticipation,
                           mode='team_vs_player' if human else 'team_vs_game')
    single = strategy == 'single'
    agent = ClassicAIV1Model(args) if single else TeammateCoordinator(args, strategy=strategy)
    env = make_match_env(state, single=single, team_size=team_size, human=human)
    started = perf_counter()
    recording = None
    resources = ExitStack()
    try:
        for name, (address, kind) in CPU_RAM.items():
            env.data.set_variable(name, {'address': address, 'type': kind})
        info = prepare_match(env, seed=seed, seconds=seconds, single=single, side=physical_side, team_size=team_size, human=human)
        initial_ram_hash = hashlib.sha256(env.get_ram().tobytes()).hexdigest()
        rom_path = Path(env.pass_geometry_rom)
        starting_lineup = lineup(info, rom_path.read_bytes())
        ratings = [[env.data.memory.extract(0xFFB04A + slot * 0x80 + offset, '|u1')
                    for offset in range(0x67, 0x77)] for slot in range(12)]
        teams = [info['home_team'], info['away_team']]
        state_object = NHL94GameState(5)
        if physical_side == 2:
            state_object.Flip()
        digest, roles = hashlib.sha256(), Counter()
        possession_frames, selected_frames = 0, [0] * (1 if single else team_size)
        period_rows = []
        started_periods = {0}
        ordered_sides = (physical_side, 3 - physical_side)
        first_scores = [info[f'p{i}_score'] for i in ordered_sides]
        first_shots = [decode_shot_count(info[f'p{i}_shots']) for i in ordered_sides]
        passes = info.get(f'p{physical_side}_pass_attempts', 0)
        one_timers = info.get(f'p{physical_side}_one_timer_attempts', 0)
        one_timer_goals = info[f'bench_one_timer_goals{physical_side}']
        offense = OffenseMetrics()
        receptions = ReceptionOutcomes()
        pressure_metrics = PressureMetrics()
        measurements = TeammateDiagnostics()
        evidence = PassEvidence(event_trace)
        period_native = {'score': first_scores, 'shots': first_shots, 'pass_attempts': passes,
                         'one_timer_attempts': one_timers, 'one_timer_goals': one_timer_goals}
        period_observed = {}
        last_menu = -120
        completed = False
        frame = 0
        previous_owner = -1
        previous_selected = []
        previous_action = np.zeros(12, dtype=np.int8)
        previous_passes = passes
        pending = None
        controlled_passes = completed_passes = turnovers = 0
        last_possession = None
        loose_frames = 0
        transient_frames = transient_streak = max_transient_streak = transient_trap_exits = 0
        was_trapping = False
        if trace:
            path = Path(trace).expanduser()
            path.parent.mkdir(parents=True, exist_ok=True)
            recording = resources.enter_context(path.open('w', encoding='utf-8'))
            recording.write(json.dumps({'schema': 'nhl94-teammate-inputs-v2' if team_size == 3 or human else 'nhl94-teammate-inputs-v1',
                                         **(trace_layout(team_size, human, physical_side) if team_size == 3 or human else {}), 'state': state,
                                         'seed': seed, 'strategy': strategy, 'periods': periods,
                                         'seconds': seconds, 'frame_skip': frame_skip,
                                         'side': side,
                                         'support_options': support_options,
                                         'pressure': pressure,
                                         'rom_sha256': hashlib.sha256(rom_path.read_bytes()).hexdigest(),
                                         'initial_ram_sha256': initial_ram_hash}) + '\n')
        budget = max_frames or (seconds * 120 + 6000) * periods
        backend = get_game(args.env).backend
        for frame in range(1, budget + 1):
            info = backend.frame_info(env, info)
            highlight = bool(info.get('goalie_clock_flags', 0) & 0x10)
            if single:
                if not highlight and (info['defense_team1'], info['defense_team2']) != (physical_side, 0):
                    raise RuntimeError('The baseline opponent is no longer under CPU control')
                if not highlight and physical_side == 2:
                    info = restore_away_control(env.data, info)
                native = {**info, f'p{physical_side}_control_slot': info['defense_control1'],
                          f'p{3-physical_side}_control_slot': -1,
                          'time': info['teammate_clock']}
                state_object.BeginFrame(native, TeammateCoordinator._game_action(previous_action))
                action = (np.zeros(12, dtype=np.int8) if highlight
                          else agent.predict_frame(state_object, frame_skip=frame_skip)[0])
                if highlight or native['time'] == 0 or native['sflags'] & 0x0100:
                    action = np.zeros(12, dtype=np.int8)
                    if ((highlight or native['period'] in (1, 2) and native['time'] > 0)
                            and frame - last_menu >= 120):
                        action[3] = 1
                        last_menu = frame
                diagnostics = {'controllers': [{'controller': 1, 'slot': native[f'p{physical_side}_control_slot'],
                                                  'role': 'highlight' if highlight else agent._last_decision,
                                                  'decision_tick': agent.scheduler.decisions,
                                                  'decision': agent._last_decision,
                                                  'buttons': action.tolist()}]}
                state_object.EndFrame()
            else:
                previous_teammate_inputs = agent.previous.copy() if recording else None
                result = agent.act(info)
                action = result.action
                diagnostics = result.diagnostics['teammates']
                native = teammate_controller_info(info, side=physical_side, team_size=team_size, human=human)
            view_state = state_object if single else agent.states[0]
            transient = bool(diagnostics.get('transient'))
            transient_frames += transient
            transient_streak = transient_streak + 1 if transient else 0
            max_transient_streak = max(max_transient_streak, transient_streak)
            transient_trap_exits += transient and was_trapping
            was_trapping = diagnostics.get('pressure', {}).get('phase') in ('contain', 'close')
            if not highlight:
                pressure_metrics.observe(frame, view_state, diagnostics)
            release = receptions.shots.observe(frame, view_state, highlight=highlight)
            if not highlight:
                if release is not None:
                    offense.after_shot = True
                offense.observe(frame, view_state)
            for row in diagnostics['controllers']:
                roles[f"p{row['controller']}:{row['role']}"] += 1
                selected_frames[row['controller'] - 1] += row['slot'] >= 0
            base = (physical_side - 1) * 6
            possession_frames += not highlight and base <= native['puck_owner'] < base + 5
            owner = native['puck_owner']
            pass_count = info.get(f'p{physical_side}_pass_attempts', 0)
            selected = [row['slot'] for row in diagnostics['controllers'] if row['slot'] >= 0]
            receptions.observe(frame, view_state, selected, highlight=highlight, release=release)
            controlled_pass = (previous_owner in previous_selected
                               and info.get('pass_target') in previous_selected
                               and info.get('pass_target') != previous_owner)
            if not highlight and pass_count > previous_passes and controlled_pass:
                pending = (info['pass_target'], frame + 90)
                controlled_passes += 1
            if highlight:
                pending = None
            if pending is not None:
                if owner == pending[0]:
                    completed_passes += 1
                    pending = None
                elif owner >= 0 or frame > pending[1]:
                    pending = None
            # A turnover is an observed home-skater -> away-skater possession
            # change, allowing a short loose interval and excluding stoppages.
            if highlight or info.get('goalie_clock_flags', 0) & 1:
                last_possession = None
                loose_frames = 0
            elif 0 <= owner < 5 or 6 <= owner < 11:
                owner_side = 1 if owner < 5 else 2
                turnovers += last_possession == physical_side and owner_side != physical_side and loose_frames <= 90
                last_possession, loose_frames = owner_side, 0
            elif owner in (5, 11):
                last_possession = None
            else:
                loose_frames += 1
            previous_owner, previous_passes = (-1 if highlight else owner), pass_count
            previous_selected = [] if highlight else selected
            if human:
                human_buttons = np.asarray(human_input(frame, dict(info)))
                if human_buttons.shape != (12,) or not np.isin(human_buttons, (0, 1)).all():
                    raise ValueError('Human input source must return 12 binary buttons')
                action = np.concatenate((action, human_buttons))
            action = filter_menu_actions(env.data, action)
            measurements.observe(frame, view_state, diagnostics, action, highlight=highlight)
            evidence.observe(frame, view_state, diagnostics, action, receptions)
            if single:
                previous_action = np.asarray(action, dtype=np.int8).copy()
            digest.update(np.asarray(action, dtype=np.int8).tobytes())
            if recording:
                if not single and not highlight and team_size == 2:
                    diagnostics['opportunities'] = opportunity_diagnostics(
                        view_state, diagnostics, previous_teammate_inputs)
                    if diagnostics['opportunities'] is not None:
                        diagnostics['opportunities']['native_touch'] = crossing_touch_player(env)
                recording.write(json.dumps({'frame': frame, 'period': info['period'] + 1,
                                             'clock': info['teammate_clock'], 'owner': info['puck_owner'],
                                             'buttons': action.tolist(), 'diagnostics': diagnostics}) + '\n')
            _, _, _, truncated, info = env.step(action)
            if not info.get('goalie_clock_flags', 0) & 0x10 and info['teammate_clock'] > 0:
                started_periods.add(info['period'])
            if (not info.get('goalie_clock_flags', 0) & 0x10
                    and info['period'] in started_periods and info['teammate_clock'] == 0
                    and info['period'] == len(period_rows)):
                current = {'score': [info[f'p{i}_score'] for i in ordered_sides],
                           'shots': [decode_shot_count(info[f'p{i}_shots']) for i in ordered_sides],
                           'pass_attempts': info.get(f'p{physical_side}_pass_attempts', 0),
                           'one_timer_attempts': info.get(f'p{physical_side}_one_timer_attempts', 0),
                           'one_timer_goals': info[f'bench_one_timer_goals{physical_side}']}
                totals = {key: [a-b for a, b in zip(value, period_native[key])] if isinstance(value, list)
                          else value-period_native[key] for key, value in current.items()}
                observed = dict(measurements.counts, skater_turnovers=turnovers)
                observed_delta = {key: value-period_observed.get(key, 0) for key, value in observed.items()
                                  if key != 'max_goalie_hold_frames'}
                period_rows.append({'period': info['period'] + 1, 'frame': frame,
                                    'score': [info['p1_score'], info['p2_score']],
                                    'native_totals': totals, 'observed_totals': observed_delta})
                period_native, period_observed = current, observed
                if len(period_rows) == periods:
                    completed = True
                    break
            if truncated:
                break
        score = [info[f'p{i}_score'] - before for i, before in zip(ordered_sides, first_scores)]
        # Consume the final native frame as well, including a buzzer-beating shot.
        if not info.get('goalie_clock_flags', 0) & 0x10:
            final_view = ({**info, f'p{physical_side}_control_slot': info['defense_control1'],
                           f'p{3-physical_side}_control_slot': -1, 'time': info['teammate_clock']}
                          if single else teammate_controller_info(info, side=physical_side, team_size=team_size, human=human))
            view_state.BeginFrame(final_view, TeammateCoordinator._game_action(previous_action)
                                  if single else TeammateCoordinator._game_action(agent.previous[0]))
            release = receptions.shots.observe(frame + 1, view_state)
            if release is not None:
                offense.after_shot = True
            offense.observe(frame + 1, view_state)
            receptions.observe(frame + 1, view_state, selected, release=release)
            view_state.EndFrame()
        receptions.finish(frame + 1, 'period-ended' if completed else 'incomplete-run')
        shot_counts = [decode_shot_count(info[f'p{i}_shots']) for i in ordered_sides]
        return {'schema': 'nhl94-teammate-match-v1', 'state': state, 'seed': seed,
                'side': 'home' if physical_side == 1 else 'away', 'teams': teams,
                'score_order': ['ai', 'human' if human else 'cpu'],
                'lineup': starting_lineup, 'initial_ratings': ratings,
                'strategy': strategy, 'controllers': 1 if single else team_size + human,
                'team_size': 1 if single else team_size, 'opponent': 'human' if human else 'cpu', 'goalies': 'automatic',
                'support_options': support_options,
                'offpuck': offpuck,
                'intent_arbitration': intent_arbitration,
                'defensive_support': defensive_support,
                'primary_policy': primary_policy,
                'role_balance': role_balance, 'transient_receiving': transient_receiving,
                'cue_anticipation': cue_anticipation,
                'pressure': pressure, 'pressure_outcomes': pressure_metrics.summary(),
                'controller_health': {'transient_frames': transient_frames,
                                      'max_transient_streak': max_transient_streak,
                                      'transient_trap_exits': transient_trap_exits},
                'frame_skip': frame_skip, 'requested_periods': periods, 'seconds': seconds,
                'completed': completed, 'frames': frame, 'elapsed_seconds': perf_counter() - started,
                'score': score, 'shots': [a-b for a, b in zip(shot_counts, first_shots)],
                'pass_attempts': info.get(f'p{physical_side}_pass_attempts', 0) - passes,
                'one_timer_attempts': info.get(f'p{physical_side}_one_timer_attempts', 0) - one_timers,
                'one_timer_goals': info[f'bench_one_timer_goals{physical_side}'] - one_timer_goals,
                'offense': offense.summary(),
                'teammate_diagnostics': measurements.summary(),
                'reception_outcomes': receptions.summary(),
                'decision_evidence': evidence.close(receptions),
                'pass_attacks': receptions.events,
                'native_shot_releases': receptions.shots.events,
                'passes_to_controlled_teammate': controlled_passes,
                'completed_passes_to_controlled_teammate': completed_passes,
                'skater_turnovers': turnovers,
                'possession_frames': possession_frames, 'selected_frames': selected_frames,
                'roles': dict(roles), 'periods': period_rows, 'actions_sha256': digest.hexdigest(),
                'initial_ram_sha256': initial_ram_hash,
                'initial_state_sha256': hashlib.sha256(env.initial_state).hexdigest(),
                'final_clock': info['teammate_clock'], 'final_period': info['period'] + 1,
                'final_ram_sha256': hashlib.sha256(env.get_ram().tobytes()).hexdigest(),
                'rom_sha256': hashlib.sha256(rom_path.read_bytes()).hexdigest(),
                'source_sha256': source_hash()}
    finally:
        resources.close()
        env.close()


def trace_layout(team_size, human, side=1):
    """Versioned physical layout and executable provenance for input-only replay."""
    import stable_retro
    import stable_retro._retro as native
    core = Path(stable_retro.core_path()) / 'genesis_plus_gx_libretro.so'
    return {'team_size': team_size, 'opponent': 'human' if human else 'cpu',
            'ports': list(range(team_size + human)), 'physical_capacity': 4,
            'teams': [side] * team_size + ([3 - side] if human else []) + [0] * (4 - team_size - human),
            'reset_setup': 'ea-wayplay-genplus-1.7.5-v1', 'device': 1025,
            'native_sha256': hashlib.sha256(Path(native.__file__).read_bytes()).hexdigest(),
            'core_sha256': hashlib.sha256(core.read_bytes()).hexdigest()}


def replay_trace(path):
    """Replay recorded native inputs from their verified reset boundary."""
    with Path(path).expanduser().open(encoding='utf-8') as recording:
        header = json.loads(next(recording))
        if header.get('schema') not in ('nhl94-teammate-inputs-v1', 'nhl94-teammate-inputs-v2'):
            raise ValueError('Unsupported teammate input recording')
        single = header['strategy'] == 'single'
        if header.get('side', 'home') not in ('home', 'away'):
            raise ValueError('Invalid replay bot side')
        side = 1 if header.get('side', 'home') == 'home' else 2
        team_size, human = 2, False
        if header['schema'] == 'nhl94-teammate-inputs-v2':
            team_size, human = header.get('team_size'), header.get('opponent') == 'human'
            # Keep the replay-layout rejection conditions together and short-circuited.
            if (single or isinstance(team_size, bool) or team_size not in (2, 3)  # pylint: disable=too-many-boolean-expressions
                    or team_size == 2 and not human or header.get('opponent') not in ('human', 'cpu')):
                raise ValueError('Invalid multitap replay layout')
            expected = trace_layout(team_size, human, side)
            if any(header.get(key) != value for key, value in expected.items()):
                raise ValueError('Incompatible replay controller layout or dependency/core provenance')
        elif 'team_size' in header or 'ports' in header:
            raise ValueError('v1 traces must use their historical controller layout')
        env = make_match_env(header['state'], single=single, team_size=team_size, human=human)
        try:
            prepare_match(env, seed=header['seed'], seconds=header['seconds'], single=single, side=side, team_size=team_size, human=human)
            rom_hash = hashlib.sha256(Path(env.pass_geometry_rom).read_bytes()).hexdigest()
            ram_hash = hashlib.sha256(env.get_ram().tobytes()).hexdigest()
            if (rom_hash, ram_hash) != (header['rom_sha256'], header['initial_ram_sha256']):
                raise ValueError('Replay ROM or initial RAM does not match the recording')
            frames, digest = 0, hashlib.sha256()
            for line in recording:
                row = json.loads(line)
                buttons = np.asarray(row['buttons'])
                if (row['frame'] != frames + 1 or buttons.shape != (12 if single else 12 * (team_size + human),)
                        or not np.isin(buttons, (0, 1)).all()):
                    raise ValueError('Invalid teammate replay frame or controller buttons')
                buttons = buttons.astype(np.int8)
                env.step(buttons)
                frames += 1
                digest.update(buttons.tobytes())
            return {'frames': frames, 'actions_sha256': digest.hexdigest(),
                    'final_ram_sha256': hashlib.sha256(env.get_ram().tobytes()).hexdigest()}
        finally:
            env.close()


def source_hash():
    digest = hashlib.sha256()
    import nhl94_ai
    root = Path(nhl94_ai.__file__).resolve().parent
    for path in sorted(root.rglob('*.py')):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def build_parser():
    parser = argparse.ArgumentParser(description='Measure two or three AI teammates against the built-in CPU')
    parser.add_argument('--state', default=DEFAULT_STATE,
                        help='Named starting save or absolute path to an external gzip native save')
    parser.add_argument('--seed', type=int, default=94)
    parser.add_argument('--trials', type=int, default=1)
    parser.add_argument('--team-size', type=int, choices=(2, 3), default=2)
    parser.add_argument('--strategy', choices=('single', 'independent', 'coordinated'), default='coordinated')
    parser.add_argument('--side', choices=('home', 'away'), default='home', help='Bots team; the opposite team is CPU-controlled')
    parser.add_argument('--teammate-support-options', action=argparse.BooleanOptionalAction, default=False,
                        help='Scored reachable support positions; --no-teammate-support-options retains baseline steering')
    parser.add_argument('--no-pressure', action='store_true',
                        help='Disable coordinated carrier traps for a baseline comparison')
    parser.add_argument('--teammate-offpuck', action='store_true', help='Evaluate scored cuts/flanks with forecast-matched steering')
    parser.add_argument('--teammate-intent-arbitration', action='store_true', help='Separate native poke, switch and pass requests')
    parser.add_argument('--teammate-defensive-support', action='store_true', help='Threat-aware fallback screen coverage')
    parser.add_argument('--teammate-primary-policy', choices=('baseline', 'short-lease', 'lost-race'), default='baseline')
    parser.add_argument('--teammate-transient-receiving', action=argparse.BooleanOptionalAction, default=None,
                        help='Preserve released receptions during unknown controller tables (default: on in coordinated mode); '
                             '--no-teammate-transient-receiving disables only that preservation for comparison')
    parser.add_argument('--teammate-cue-anticipation', action='store_true',
                        help='Time bot-receiver one-timer cue from shared pass intent (default: off)')
    parser.add_argument('--teammate-role-balance', action='store_true', help='Position-based three-bot safety assignments')
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--periods', type=int, choices=(1, 3), default=1)
    parser.add_argument('--seconds', type=int, default=300)
    parser.add_argument('--frame-skip', type=int, default=4)
    parser.add_argument('--max-frames', type=int)
    parser.add_argument('--trace', help='Optional per-frame buttons and decisions; requires one trial')
    parser.add_argument('--replay', help='Replay a previously recorded input trace instead of evaluating')
    parser.add_argument('--output', required=True)
    return parser


def run(args):
    if args.replay:
        report = replay_trace(args.replay)
        output = Path(args.output).expanduser()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
        return report
    if args.trials < 1 or args.trace and args.trials != 1 or args.workers < 1:
        raise ValueError('Use positive trials; --trace requires exactly one trial')
    validate_rom_seed(args.seed)
    frozen_source = source_hash()
    jobs = [dict(state=args.state, seed=(args.seed + index) % 2**32, strategy=args.strategy,
                 periods=args.periods, seconds=args.seconds, frame_skip=args.frame_skip,
                 max_frames=args.max_frames, trace=args.trace, side=args.side,
                 support_options=args.teammate_support_options, team_size=args.team_size,
                 pressure=not args.no_pressure, offpuck=args.teammate_offpuck,
                 intent_arbitration=args.teammate_intent_arbitration,
                 defensive_support=args.teammate_defensive_support,
                 primary_policy=args.teammate_primary_policy,
                 role_balance=args.teammate_role_balance, transient_receiving=getattr(args, 'teammate_transient_receiving', None),
                 cue_anticipation=getattr(args, 'teammate_cue_anticipation', False)) for index in range(args.trials)]
    matches = []
    with ExitStack() as resources:
        if args.workers > 1:
            pool = resources.enter_context(ProcessPoolExecutor(max_workers=args.workers))
            rows = pool.map(_play_job, jobs)
        else:
            rows = map(_play_job, jobs)
        for row in rows:
            if row['source_sha256'] != frozen_source or source_hash() != frozen_source:
                raise RuntimeError('Source changed during evaluation; discard this run')
            matches.append(row)
            print(f"Seed {row['seed']}: {row['score'][0]}-{row['score'][1]}, "
                  f"{'complete' if row['completed'] else 'incomplete'}, {row['frames']} native frames", flush=True)
    report = {'schema': 'nhl94-teammate-evaluation-v1', 'strategy': args.strategy,
              'completed': all(row['completed'] for row in matches), 'matches': matches}
    output = Path(args.output).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    return report


def _play_job(job):
    return play_match(**job)


if __name__ == '__main__':
    run(build_parser().parse_args())

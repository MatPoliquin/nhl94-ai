"""Run native matches against explicitly loaded external adapters.

Adapters declare read-only RAM observations and return twelve native buttons.
Only the requested adapter is imported; normal gameplay needs no adapter package.
"""
import argparse
from collections import Counter
from collections.abc import Mapping
from dataclasses import asdict, dataclass
import hashlib
import importlib
import json
from pathlib import Path
from time import perf_counter, perf_counter_ns
from types import SimpleNamespace

import numpy as np

from nhl94_ai.agents.classic_tuning import DEFAULT, load_params, scope
from nhl94_ai.agents.registry import resolve_goalie_positioning

PROTOCOL_VERSION = 1
BUTTONS = ('B', 'A', 'MODE', 'START', 'UP', 'DOWN', 'LEFT', 'RIGHT', 'C', 'Y', 'X', 'Z')


@dataclass(frozen=True)
class ExternalFrame:
    number: int
    side: int
    observations: Mapping
    phase: str


def validate_action(action):
    value = np.asarray(action)
    if value.shape != (len(BUTTONS),) or not np.isin(value, (0, 1)).all():
        raise ValueError('External action must contain twelve binary native buttons')
    return value.astype(np.int8, copy=True)


class AdapterSession:
    """Validate observations and own one adapter's lifecycle and callback count."""

    def __init__(self, adapter, side):
        self.adapter, self.side, self.frames = adapter, side, 0
        self.closed = False
        try:
            for method in ('reset', 'act', 'close'):
                if not callable(getattr(adapter, method, None)):
                    raise ValueError(f'External adapter must implement {method}()')
            self.ram = dict(adapter.ram)
            for key, declaration in self.ram.items():
                if (not isinstance(key, str) or not isinstance(declaration, (tuple, list))
                        or len(declaration) != 2):
                    raise ValueError('Invalid external RAM field declaration')
                address, kind = declaration
                if (isinstance(address, bool) or not isinstance(address, int)
                        or not isinstance(kind, str)
                        or kind not in ('|u1', '|i1', '>u2', '>i2', '>u4', '>i4')
                        or not 0xFF0000 <= address <= 0x1000000 - int(kind[-1])):
                    raise ValueError(f'Invalid external RAM address/type for {key}')
            self.metadata = dict(adapter.metadata)
            json.dumps(self.metadata, allow_nan=False)
            adapter.reset()
        except BaseException:
            self.close()
            raise

    def action(self, env, phase):
        memory = env.unwrapped.data.memory
        observations = {key: int(memory.extract(address, kind))
                        for key, (address, kind) in self.ram.items()}
        result = validate_action(self.adapter.act(
            ExternalFrame(self.frames, self.side, observations, phase)))
        self.frames += 1
        return result

    def close(self):
        if self.closed:
            return
        self.closed = True
        close = getattr(self.adapter, 'close', None)
        if close is not None:
            close()


def load_adapter(spec, *, side, seed, config=None):
    module_name, separator, factory_name = spec.partition(':')
    if not separator or not module_name or not factory_name.isidentifier():
        raise ValueError('Use an external adapter in module:factory format')
    try:
        factory = getattr(importlib.import_module(module_name), factory_name)
    except (ImportError, AttributeError) as error:
        raise ValueError(f'Cannot load external adapter {spec}: {error}') from error
    if not callable(factory):
        raise ValueError(f'External adapter factory is not callable: {spec}')
    return AdapterSession(factory(protocol_version=PROTOCOL_VERSION, side=side,
                                  seed=seed, config=config), side)


def latency_summary(values):
    if not values:
        return {'calls': 0, 'p50': None, 'p99': None, 'max': None}
    return {'calls': len(values), 'p50': float(np.quantile(values, .5)/1e6),
            'p99': float(np.quantile(values, .99)/1e6), 'max': max(values)/1e6}


FIXTURES = {
    'sabres-ducks': 'SabresVsMightyDucks.ManualGoalie.Start',
    'ducks-campbell': 'MightyDucksVsAllStarCampbell.ManualGoalie.2P.Start',
    'penguins-senators': 'PenguinsVsSenators.start.2P',
    'canadiens-nordiques': 'CanadiensvsNordiques.start.2P',
}
TRAIN_FIXTURES = ('sabres-ducks', 'ducks-campbell')
OPPONENTS = ('external', 'classic')


def warm_imports():
    """Import before fork so workers share the read-only Python/runtime pages."""
    from nhl94_ai.agents.classic_v1 import ClassicAIV1Model
    from nhl94_ai.env.factory import make_retro
    from nhl94_ai.evaluation.benchmark import away_view, update_state
    from nhl94_ai.evaluation.cpu_benchmark import CPU_RAM, lineup
    from nhl94_ai.game.state import NHL94GameState
    from stable_retro.data import get_romfile_path


def play_match(job):
    """Run one independently seeded game and mark incomplete games explicitly."""
    from nhl94_ai.agents.classic_v1 import ClassicAIV1Model
    from nhl94_ai.env.factory import make_retro
    from nhl94_ai.evaluation.benchmark import away_view, update_state
    from nhl94_ai.evaluation.cpu_benchmark import CPU_RAM, lineup
    from nhl94_ai.game.state import NHL94GameState
    from stable_retro.data import get_romfile_path

    params = load_params(job.get('params', asdict(DEFAULT)))
    side, seed = job['side'], job['seed']
    periods = job.get('periods', 3)
    opponent = job['opponent']
    human = job.get('human', False)
    if side not in (1, 2) or opponent not in OPPONENTS or periods not in (1, 3):
        raise ValueError('Unsupported match configuration')
    if human and opponent != 'external':
        raise ValueError('Human external matches require an external adapter')
    args = SimpleNamespace(action_type='FILTERED', goalie_policy='selective',
                           goalie_planner=job.get('goalie_planner'),
                           goalie_positioning=job.get('goalie_positioning'))
    candidate = None if human else ClassicAIV1Model(args)
    other = (load_adapter(job['adapter'], side=3-side, seed=seed, config=job.get('adapter_config'))
             if opponent == 'external' else ClassicAIV1Model(
                 SimpleNamespace(action_type='FILTERED', goalie_policy='selective')))
    try:
        env = make_retro(game='NHL94-Genesis-v0', state=FIXTURES[job['fixture']],
                         num_players=2, goalie_policy='selective')
    except BaseException:
        if opponent == 'external':
            other.close()
        raise
    started = perf_counter()
    viewer = None
    try:
        if tuple(env.buttons) != BUTTONS:
            raise ValueError('Unexpected native button ordering')
        for name, (address, kind) in CPU_RAM.items():
            env.data.set_variable(name, {'address': address, 'type': kind})
        env.reset(seed=seed)
        env.data.update_ram()
        initial = env.data.lookup_all()
        writes = []
        if initial['bench_team2'] == 0:
            if initial['bench_team1'] != 1 or not 0 <= initial['bench_control1'] < 5:
                raise ValueError('Second-controller setup requires a home skater start')
            target = initial['bench_control1'] + 6
            writes.extend([('bench_team2', 2), ('bench_control2', target),
                           (f'cpu_{target}_flags', initial[f'cpu_{target}_flags'] | 8)])
        for controller in (1, 2):
            key = f'goalie_mode_{controller}'
            if initial[key] != 0:
                writes.append((key, 0))
        for key, value in writes:
            env.data.set_value(key, value)
        env.data.update_ram()
        initial = env.data.lookup_all()
        if (initial['bench_team1'], initial['bench_team2'], initial['period'], initial['bench_clock'],
                initial['p1_score'], initial['p2_score']) != (1, 2, 0, 300, 0, 0):
            raise ValueError('Need scoreless five-minute first-period opposing-controller start')
        roles = [env.data.memory.extract(0xFFB04A + slot * 0x80 + 0x34, '>i2') for slot in range(12)]
        if any(role <= 0 for slot, role in enumerate(roles) if slot % 6 != 5) or (roles[5], roles[11]) != (0, 0):
            raise ValueError('Fixture contains inactive skaters or unexpected goalie roles')
        rom = Path(get_romfile_path('NHL94-Genesis-v0')).read_bytes()
        rosters = lineup(initial, rom)
        attributes = [[env.data.memory.extract(0xFFB04A + slot * 0x80 + offset, '|u1')
                       for offset in range(0x67, 0x74)] for slot in range(12)]
        env.data.set_value('bench_rng', seed)
        env.data.set_value('bench_clock', job.get('seconds', 300))
        env.data.update_ram()
        info = env.data.lookup_all()
        # The Genesis serializer includes host pointers in sound-chip structs.
        # Its raw hash is useful within a process, but differs under ASLR.
        # At this pre-step boundary the same save and RAM edits identify the start.
        ram_hash = hashlib.sha256(env.get_ram().tobytes()).hexdigest()
        save_hash = hashlib.sha256(env.initial_state).hexdigest()
        prepared_hash = hashlib.sha256((save_hash + ram_hash).encode()).hexdigest()
        serialized_hash = hashlib.sha256(env.em.get_state()).hexdigest()
        state, frames = NHL94GameState(5), 0
        digest, raw_digest = hashlib.sha256(), hashlib.sha256()
        period_rows, menu_frames, period_nets = [], [], {}
        started_periods, last_menu = {0}, -120
        previous_goals, previous_shots = [0, 0], [0, 0]
        goals = [info['p1_score'], info['p2_score']]
        shots = [info['bench_shots1'], info['bench_shots2']]
        intermission_frames = 0
        decisions = Counter()
        inference_ns = []
        user_quit = terminated = truncated = False
        if job.get('watch') or human:
            viewer = MatchViewer(env, side, seed, info, human=human,
                                 no_gamepad=job.get('no_gamepad', False),
                                 opponent_name=other.metadata.get('name', 'External adapter')
                                 if opponent == 'external' else 'Classic AI')
        while len(period_rows) < periods and (human or frames < (job.get('seconds', 300)*120+6000)*periods):
            if viewer is not None and not viewer.before_frame():
                user_quit = True
                break
            intermission = periods > 1 and (
                info['bench_clock'] == 0 or info['sflags'] & 0x8000
                or info['period'] > 0 and info['bench_clock'] == 300
                and (info['bench_control1'] < 0 or info['bench_control2'] < 0))
            other_buttons = (other.action(env, 'intermission' if intermission else 'play') if opponent == 'external'
                             else np.zeros(12, dtype=np.int8))
            raw_digest.update(other_buttons.tobytes())
            candidate_buttons = (validate_action(viewer.human_action()) if human
                                 else np.zeros(12, dtype=np.int8))
            if intermission:
                intermission_frames += 1
            else:
                update_state(state, info, env)
                views = (state, away_view(state))
                period_nets[info['period']] = [state.team1.net.y, state.team2.net.y]
                if not human:
                    before = candidate.scheduler.decisions
                    with scope(params):
                        inference_started = perf_counter_ns()
                        candidate_buttons = candidate.predict_frame(views[side-1], frame_skip=job.get('frame_skip', 4))[0]
                        inference_ns.append(perf_counter_ns()-inference_started)
                    if candidate.scheduler.decisions != before:
                        decisions[candidate._last_decision] += 1
                if opponent == 'classic':
                    other_buttons = other.predict_frame(views[2-side], frame_skip=job.get('frame_skip', 4))[0]
            buttons = np.concatenate((candidate_buttons, other_buttons) if side == 1
                                     else (other_buttons, candidate_buttons))
            if (intermission and info['period'] in (1, 2) and info['bench_clock'] == 300
                    and info['sflags'] & 0x8000 and frames - last_menu >= 120):
                buttons[3] = 1
                last_menu = frames
                menu_frames.append(frames)
                if len(menu_frames) > 12:
                    raise RuntimeError('Intermission navigation stalled')
            digest.update(buttons.tobytes())
            _, _, terminated, truncated, info = env.step(buttons)
            frames += 1
            if (info['bench_team1'], info['bench_team2'], info['goalie_mode_1'], info['goalie_mode_2']) != (1, 2, 0, 0):
                raise RuntimeError('Controller assignment or goalie setting changed')
            if info['bench_clock'] > 0:
                started_periods.add(info['period'])
            goals = [info['p1_score'], info['p2_score']]
            shots = [info['bench_shots1'], info['bench_shots2']]
            if (info['bench_clock'] == 0 and info['period'] in started_periods
                    and info['period'] == len(period_rows)):
                period_rows.append({'period': info['period']+1, 'frame': frames,
                                    'goals': [a-b for a, b in zip(goals, previous_goals)],
                                    'shots': [a-b for a, b in zip(shots, previous_shots)],
                                    'actions_sha256': digest.hexdigest()})
                previous_goals, previous_shots = goals, shots
            if viewer is not None:
                viewer.show(env, info)
            if truncated or terminated and periods == 1:
                break
        completed = len(period_rows) == periods
        if opponent == 'external' and other.frames != frames:
            raise RuntimeError('External adapter missed native callbacks')
        if candidate is not None:
            candidate.end_one_timer('period-ended' if completed else 'trial-incomplete')
        return {
            'fixture': job['fixture'], 'opponent': opponent, 'side': side, 'seed': seed,
            'local_controller': 'human' if human else 'classic',
            'periods': periods, 'completed': completed, 'user_quit': user_quit,
            'terminated': bool(terminated), 'truncated': bool(truncated), 'frames': frames, 'goals': goals, 'shots': shots,
            'goal_difference': goals[side-1] - goals[2-side],
            'one_timers': [info['bench_one_timers1'], info['bench_one_timers2']],
            'one_timer_goals': [info['bench_one_timer_goals1'], info['bench_one_timer_goals2']],
            'period_results': period_rows, 'period_nets': period_nets,
            'intermission_frames': intermission_frames, 'menu_frames': menu_frames,
            'external_frames': other.frames if opponent == 'external' else None,
            'external_actions_sha256': raw_digest.hexdigest() if opponent == 'external' else None,
            'adapter_metadata': other.metadata if opponent == 'external' else None,
            'external_source_sha256': other.metadata.get('source_sha256') if opponent == 'external' else None,
            'actions_sha256': digest.hexdigest(), 'prepared_start_sha256': prepared_hash,
            'initial_state_sha256': save_hash, 'prepared_ram_sha256': ram_hash,
            'serialized_state_sha256': serialized_hash,
            'rom_sha256': hashlib.sha256(rom).hexdigest(), 'setup_writes': writes,
            'teams': [initial['home_team'], initial['away_team']], 'lineup': rosters,
            'effective_attributes': attributes, 'decisions': dict(decisions),
            'goalie_metrics': {
                'candidate': dict(candidate.goalie.metrics) if candidate is not None else None,
                'opponent': dict(other.goalie.metrics) if opponent == 'classic' else None,
            },
            'goalie_planner': job.get('goalie_planner'),
            'goalie_positioning': None if human else resolve_goalie_positioning(args),
            'controller_latency_ms': latency_summary(inference_ns),
            'planner_latency_ms': ({'calls': len(candidate.goalie.planner_times_ns),
                'p50': float(np.quantile(candidate.goalie.planner_times_ns, .5)/1e6),
                'p99': float(np.quantile(candidate.goalie.planner_times_ns, .99)/1e6),
                'max': max(candidate.goalie.planner_times_ns)/1e6}
                if candidate is not None and getattr(candidate.goalie, 'planner_times_ns', None) else None),
            'elapsed_seconds': perf_counter()-started,
        }
    finally:
        try:
            if viewer is not None:
                viewer.close()
        finally:
            try:
                env.close()
            finally:
                if opponent == 'external':
                    other.close()



class MatchViewer:
    """Pace native frames, with optional human control of the local side."""

    def __init__(self, env, side, seed, info, *, human=False, no_gamepad=False,
                 opponent_name='External adapter'):
        import os
        os.environ.setdefault('PYGAME_HIDE_SUPPORT_PROMPT', '1')
        import pygame
        self.pg = pygame
        pygame.display.init()
        pygame.font.init()
        self.gamepad = None
        self.human = human
        self.input_focused = True
        self.frame = env.render()
        height, width = self.frame.shape[:2]
        self.screen = pygame.display.set_mode((width * 3, height * 3 + 60), pygame.RESIZABLE)
        local_name = 'You' if human else 'Classic AI'
        pygame.display.set_caption(f'{opponent_name} vs {local_name} | seed {seed}')
        self.font = pygame.font.Font(None, 24)
        self.clock = pygame.time.Clock()
        self.fps = round(env.metadata.get('video.frames_per_second', 60))
        self.names = (local_name, opponent_name) if side == 1 else (opponent_name, local_name)
        self.paused = False
        self.info = info
        if human:
            from nhl94_ai.ui.gamepad import GamepadInput
            pygame.joystick.init()
            self.gamepad = GamepadInput(enabled=not no_gamepad)
            print(f'You control {"home / controller 1" if side == 1 else "away / controller 2"}. '
                  'Keyboard: arrows move, Z clear/hold, X pass/switch, C shoot/check, Enter Start. '
                  'M30: D-pad and matching Genesis buttons. Space pauses playback; Esc quits.')
        self._draw()

    def human_action(self):
        from nhl94_ai.ui.human_input import read_human_input
        if not self.human:
            raise RuntimeError('Human input is unavailable in spectator mode')
        return read_human_input(self.gamepad, self.input_focused)

    def _draw(self):
        pygame = self.pg
        self.screen.fill((18, 22, 29))
        width, height = self.screen.get_size()
        surface = pygame.surfarray.make_surface(self.frame.swapaxes(0, 1))
        bounds = pygame.Rect(0, 60, max(1, width), max(1, height - 60))
        target = surface.get_rect().fit(bounds)
        self.screen.blit(pygame.transform.scale(surface, target.size), target)
        title = (f'{self.names[0]} (Home)  {self.info["p1_score"]} - '
                 f'{self.info["p2_score"]}  {self.names[1]} (Away)')
        controls = ('PAUSED | ' if self.paused else '') + 'Space: pause/resume | Esc: quit'
        if self.human:
            controls += ' | M30 / arrows + Z X C'
        for y, text in ((8, title), (33, controls)):
            self.screen.blit(self.font.render(text, True, (235, 240, 245)), (12, y))
        pygame.display.flip()

    def before_frame(self):
        """Keep processing window events while paused, without advancing either AI."""
        pygame = self.pg
        while True:
            for event in pygame.event.get():
                if self.gamepad is not None:
                    self.gamepad.process_event(event)
                if event.type in (pygame.WINDOWFOCUSLOST, pygame.WINDOWFOCUSGAINED):
                    self.input_focused = event.type == pygame.WINDOWFOCUSGAINED
                if event.type == pygame.QUIT or event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    return False
                if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE and not getattr(event, 'repeat', False):
                    self.paused = not self.paused
            if not self.paused:
                return True
            self._draw()
            self.clock.tick(30)

    def show(self, env, info):
        self.frame, self.info = env.render(), info
        self._draw()
        self.clock.tick(self.fps)

    def close(self):
        if self.gamepad is not None:
            self.gamepad.close()
            self.pg.joystick.quit()
        self.pg.display.quit()
        self.pg.font.quit()



def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--agent', choices=('classic-v1',), default='classic-v1')
    parser.add_argument('--adapter', required=True, help='Installed Python module:factory')
    parser.add_argument('--adapter-config', help='Optional adapter configuration file')
    parser.add_argument('--fixture', choices=tuple(FIXTURES), default='ducks-campbell')
    parser.add_argument('--side', choices=('both', 'home', 'away'), default='both',
                        help='Local agent or human assignment; both plays paired AI matches')
    parser.add_argument('--pairs', type=int, default=1)
    parser.add_argument('--seed', type=int, default=26534)
    parser.add_argument('--periods', type=int, choices=(1, 3), default=3)
    parser.add_argument('--seconds', type=int, default=300, help='First-period clock; regulation requires 300')
    parser.add_argument('--frame-skip', type=int, default=4, help='Local Classic interval; human input and external adapters always run at 60 Hz')
    parser.add_argument('--goalie-positioning', choices=('classic', 'deep', 'deep-csave'))
    parser.add_argument('--watch', action='store_true')
    parser.add_argument('--human', action='store_true',
                        help='Play against the adapter with keyboard/M30 at 60 Hz; choose --side home or away')
    parser.add_argument('--no-gamepad', action='store_true', help='Use keyboard only in human mode')
    parser.add_argument('--output', help='Explicit report path; no files are written by default')
    return parser


def validate_options(args):
    if args.human and args.side == 'both':
        raise ValueError('Human play requires --side home or --side away')
    if args.pairs < 1 or args.frame_skip < 1 or not 1 <= args.seconds <= 300:
        raise ValueError('Use positive pairs/frame-skip and 1..300 seconds')
    if not 0 <= args.seed < 2**32 or args.seed + args.pairs > 2**32:
        raise ValueError('All trial seeds must fit uint32')
    if args.periods not in (1, 3) or args.periods == 3 and args.seconds != 300:
        raise ValueError('Full regulation requires three five-minute periods')


def run(args):
    validate_options(args)
    sides = {'both': (1, 2), 'home': (1,), 'away': (2,)}[args.side]
    rows = []
    for seed in range(args.seed, args.seed+args.pairs):
        for side in sides:
            job = dict(vars(args), side=side, seed=seed, opponent='external')
            row = play_match(job)
            rows.append(row)
            print(f"seed={seed} local_side={side} goals={row['goals']} "
                  f"frames={row['frames']} completed={row['completed']}", flush=True)
            if row['user_quit']:
                break
        if rows[-1]['user_quit']:
            break
    complete = [row for row in rows if row['completed']]
    scores = [row['goal_difference'] for row in complete]
    summary = dict(completed=len(complete), incomplete=len(rows)-len(complete),
                   local_wins=sum(score > 0 for score in scores),
                   draws=sum(score == 0 for score in scores),
                   external_wins=sum(score < 0 for score in scores))
    report = dict(protocol='external-native-matches-v1', settings=vars(args),
                  summary=summary, matches=rows)
    if args.output:
        path = Path(args.output).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(summary), flush=True)
    if any(not row['completed'] and not row['user_quit'] for row in rows):
        raise RuntimeError('Incomplete matches; partial scores are excluded from the summary')
    return report


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        run(args)
    except (ValueError, RuntimeError, OSError) as error:
        parser.exit(1, f'{error}\n')


if __name__ == '__main__':
    main()

"""Bounded native-frame playback timing with exact applied-input references."""
import argparse
import cProfile
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from nhl94_ai.config import load_hyperparams, resolve_hyperparams_for_model
from nhl94_ai.evaluation.play import NHL94Player, parse_cmdline


def timing_summary(rows):
    result = {}
    for cohort in ('all', 'ai-skater-possession', 'other'):
        samples = rows if cohort == 'all' else [row for row in rows if row['cohort'] == cohort]
        result[cohort] = {'frames': len(samples)}
        for stage in ('predict', 'diagnostics', 'step', 'draw', 'total', 'paced'):
            values = [row['timing_ms'][stage] for row in samples]
            if values:
                result[cohort][stage] = {
                    'mean_ms': float(np.mean(values)), 'p95_ms': float(np.percentile(values, 95)),
                    'max_ms': max(values), 'over_60hz_budget': sum(value > 1000 / 60 for value in values)}
    return result


def run(args):
    import pygame
    if args.frames < 1:
        raise ValueError('Profile frame budget must be positive.')
    settings = {'frames': args.frames, 'side': 'away', 'seed': 12000, 'goalie_policy': 'selective',
                'state': 'SabresVsMightyDucks.ManualGoalie.Start', 'frame_skip': 4,
                'action_type': 'FILTERED', 'paced': args.paced}
    reference = json.loads(Path(args.reference).read_text(encoding='utf-8')) if args.reference else None
    if args.output and args.reference and Path(args.output).resolve() == Path(args.reference).resolve():
        raise ValueError('Do not overwrite the original playback timing reference.')
    if reference is not None and reference['settings'] != settings:
        raise ValueError('Timing reference must use the identical native prefix and pacing settings.')
    root = Path(__file__).resolve().parents[2]
    names = ('evaluation/play.py', 'agents/base.py', 'agents/decisions.py', 'agents/classic_v1.py',
             'agents/offense.py', 'agents/carry.py', 'agents/passing.py', 'agents/motion.py',
             'env/target_control.py', 'env/observation.py', 'agents/multi_model.py',
             'game/state.py', 'game/ram.py', 'agents/skating.py', 'agents/goalie.py', 'ui/debug.py', 'ui/decision_panel.py',
             'evaluation/playback_profile.py')
    sources = {f'nhl94_ai/{name}': hashlib.sha256((root / 'nhl94_ai' / name).read_bytes()).hexdigest()
               for name in names}
    play_args = parse_cmdline([
        '--nn=ClassicAIV1', '--env=NHL94-Genesis-v0', '--mode=model_vs_game',
        '--state=' + settings['state'], '--side=away', '--goalie-policy=selective',
        '--max_playback_speed=1.0', '--seed=12000', '--rf=PostPlay'])
    play_args.hyperparams_dict = resolve_hyperparams_for_model(
        load_hyperparams(play_args.hyperparams, required=True), play_args.nn)
    player = NHL94Player(play_args, None, need_display=True)
    try:
        display = player.display_env
        observation = display.reset()
        info = player._scripted_reset_info()
        agent = player.ai_sys.models[player.ai_sys.model_in_use]
        rows, draw_duration = [], [0.0]
        original_draw = display.draw_frame
        def timed_draw(*draw_args, **kwargs):
            start = time.perf_counter()
            result = original_draw(*draw_args, **kwargs)
            draw_duration[0] += time.perf_counter() - start
            return result
        display.draw_frame = timed_draw
        initial_ram = hashlib.sha256(display.env.env_method('get_ram')[0].tobytes()).hexdigest()
        digest = hashlib.sha256()
        profiler = cProfile.Profile() if args.profile else None
        if profiler:
            profiler.enable()
        player._reset_playback_timer()
        started = time.perf_counter()
        for frame in range(args.frames):
            start = time.perf_counter()
            draw_duration[0] = 0.0
            player._wait_for_display()
            actions = player.ai_sys.predict(observation, info=info, deterministic=True)
            predicted = time.perf_counter()
            display.set_ai_sys_info(player.ai_sys)
            diagnosed = time.perf_counter()
            observation, _, done, info = display.step([actions[0]])
            stepped = time.perf_counter()
            if args.paced:
                player._throttle_display_frame()
            ended = time.perf_counter()
            controller = agent.controller
            owner = player.ai_sys.game_state.engine.puck_owner
            row = {
                'frame': frame + 1, 'action': actions[0].tolist(), 'owner': owner,
                'cohort': 'ai-skater-possession' if 6 <= owner < 11 else 'other',
                'decision': controller._last_decision,
                'clocks': [controller._tick, controller.defense.frames, controller._frame_remaining],
                'timing_ms': {
                    'predict': (predicted - start) * 1000, 'diagnostics': (diagnosed - predicted) * 1000,
                    'step': (stepped - diagnosed - draw_duration[0]) * 1000,
                    'draw': draw_duration[0] * 1000, 'total': (stepped - start) * 1000,
                    'paced': (ended - start) * 1000}}
            rows.append(row)
            digest.update(np.asarray(actions[0], dtype=np.int8).tobytes())
            if reference is not None:
                expected = reference['rows'][frame]
                if any(row[key] != expected[key] for key in ('action', 'owner', 'decision', 'clocks')):
                    raise RuntimeError(f'Native playback behavior differs at frame {frame + 1}.')
            if np.any(done):
                raise RuntimeError(f'Playback episode ended before the declared {args.frames}-frame budget.')
        elapsed = time.perf_counter() - started
        if profiler:
            profiler.disable()
            profiler.dump_stats(args.profile)
        final_ram = hashlib.sha256(display.env.env_method('get_ram')[0].tobytes()).hexdigest()
        if reference is not None and (initial_ram != reference['initial_ram_sha256']
                                      or final_ram != reference['final_ram_sha256']
                                      or digest.hexdigest() != reference['actions_sha256']):
            raise RuntimeError('Timing replay does not match the reference RAM and applied-input prefix.')
        for name, digest_before in sources.items():
            if hashlib.sha256((root / name).read_bytes()).hexdigest() != digest_before:
                raise RuntimeError(f'Measured playback source changed during profiling: {name}')
        report = {
            'protocol': 'nhl94-debug-native-timing-v1', 'settings': settings,
            'profiled': bool(profiler), 'elapsed_seconds': elapsed, 'native_fps': args.frames / elapsed,
            'window_driver': pygame.display.get_driver(), 'window_size': display.window.get_size(),
            'sources': sources,
            'initial_ram_sha256': initial_ram, 'final_ram_sha256': final_ram,
            'actions_sha256': digest.hexdigest(), 'reference_parity': reference is not None,
            'timing': timing_summary(rows), 'rows': rows,
            'limitations': [
                'A bounded seeded prefix is not a match-play strength comparison or a global worst-case CPU bound.',
                'SDL dummy timing covers the real rendering work but not a desktop compositor, vsync or GPU contention.',
                'Mean throughput does not imply every native frame fits the 16.67 ms budget.',
                'cProfile instrumentation, if enabled, is included and must not be compared with unprofiled wall timing.']}
        if reference is not None:
            report['baseline'] = {key: value for key, value in reference.items() if key not in ('rows', 'baseline')}
        if args.output:
            path = Path(args.output)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
        print(json.dumps({key: report[key] for key in ('native_fps', 'reference_parity', 'timing')}, indent=2))
        return report
    finally:
        player.close()


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--frames', type=int, default=1800)
    parser.add_argument('--paced', action='store_true', help='Include the production 1x playback throttle.')
    parser.add_argument('--profile', help='Optional cProfile output; its overhead is included in timing.')
    parser.add_argument('--reference', help='Require per-frame actions, lifecycle and final RAM parity.')
    parser.add_argument('--output')
    return parser


if __name__ == '__main__':
    run(build_parser().parse_args())

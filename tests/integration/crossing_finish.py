"""Observe actual crossing inputs, windups and releases in full CPU periods."""
import argparse
from collections import deque
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np

from nhl94_ai.agents.registry import create_scripted
from nhl94_ai.env.factory import make_retro
from nhl94_ai.evaluation.cpu_benchmark import cpu_match
from nhl94_ai.evaluation.cross_crease_full_team import full_lineup


def inspect_period(matchup, seed, video=None):
    environment, timeline = [], []
    images, writer, video_until = deque(maxlen=90), None, None

    def factory(**kwargs):
        env = make_retro(**kwargs)
        environment.append(env)
        return env

    def agent_factory(*args, **kwargs):
        agent = create_scripted(*args, **kwargs)
        predict = agent.predict_game_state

        def observe(state):
            nonlocal writer, video_until
            before = len(agent.crossing_finish.events)
            action = predict(state)
            frame = agent.scheduler.frames
            timeline.append({
                'frame': frame, 'buttons': np.flatnonzero(action[0]).tolist(),
                'decision': agent._last_decision, 'owner': state.engine.puck_owner,
                'shot_player': state.engine.shot_player, 'last_player': state.engine.last_puck_player,
                'shooting': bool(state.is_shooting), 'release_flag': bool(state.engine.sflags2 & 0x1000),
                'shots': state.team1.stats.shots, 'goals': state.team1.stats.score,
                'full_lineup': full_lineup(state), 'goalie_modes': state.engine.goalie_modes,
            })
            if video and (video_until is None or frame <= video_until):
                import cv2
                image = environment[0].render().copy()
                cv2.putText(image, f'{frame} {agent._last_decision}', (4, 15),
                            cv2.FONT_HERSHEY_SIMPLEX, .4, (255, 255, 0), 1)
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                if writer is None:
                    images.append(image)
                if len(agent.crossing_finish.events) > before and video_until is None:
                    Path(video).parent.mkdir(parents=True, exist_ok=True)
                    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*'mp4v'), 60,
                                            (image.shape[1], image.shape[0]))
                    assert writer.isOpened(), video
                    for previous in images:
                        writer.write(previous)
                    video_until = frame + 420
                elif writer is not None:
                    writer.write(image)
            return action

        agent.predict_game_state = observe
        return agent

    options = ('off', False, False, False, False, False, (), False, None, None, None,
               False, None, False, 'classic', None, None, None, True)
    try:
        with patch('nhl94_ai.env.factory.make_retro', side_effect=factory), patch(
                'nhl94_ai.evaluation.cpu_benchmark.create_scripted', side_effect=agent_factory):
            result = cpu_match(('classic-v1', matchup, seed, 300, 4, 'FILTERED', *options))
    finally:
        if writer is not None:
            writer.release()
    assert result['completed']
    assert result['inactive_skater_frames'] == 0
    assert all(row['full_lineup'] and tuple(row['goalie_modes']) == (1, 1) for row in timeline)
    assert result['crossing_finish_events'], 'This fixture must exercise the finisher.'
    events = []
    for event in result['crossing_finish_events']:
        rows = [row for row in timeline if event['frame'] <= row['frame'] <= event['frame'] + 60]
        assert [8 in row['buttons'] for row in rows[:13]] == [True] * 12 + [False]
        assert all(not ({0, 1} & set(row['buttons'])) for row in rows[:13])
        assert any(row['shooting'] for row in rows), 'C request never entered native windup.'
        release = next((row for previous, row in zip(rows, rows[1:])
                        if row['shot_player'] == event['slot'] and (
                            row['release_flag'] and not previous['release_flag']
                            or previous['shooting'] and row['owner'] < 0
                            and row['last_player'] == event['slot'])), None)
        assert release is not None, 'No fresh native release after windup.'
        events.append({**event, 'release_frame': release['frame'],
                       'shots_within_60_frames': rows[-1]['shots'] - rows[0]['shots'],
                       'goals_within_60_frames': rows[-1]['goals'] - rows[0]['goals']})
    return {'match': result, 'observed_crossings': events}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='tmp/crossing-finish/native-contracts.json')
    parser.add_argument('--video-dir')
    args = parser.parse_args()
    results = []
    for matchup in ('penguins-senators', 'nordiques-canadiens'):
        video = Path(args.video_dir) / f'{matchup}.mp4' if args.video_dir else None
        row = inspect_period(matchup, 104001, video)
        results.append(row)
        print(matchup, row['observed_crossings'], flush=True)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, indent=2) + '\n', encoding='utf-8')
    print('PASS: full rosters, native goalies, twelve C frames, windups and fresh releases on both sides')


if __name__ == '__main__':
    main()

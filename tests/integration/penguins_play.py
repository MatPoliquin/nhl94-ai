"""The supplied Penguins command must complete deliberate one-timers in live play."""
import os
from unittest.mock import patch

from nhl94_ai.cli import main
from nhl94_ai.config import load_hyperparams, resolve_hyperparams_for_model
from nhl94_ai.evaluation.play import NHL94Player


class Finished(Exception):
    """Stop on the terminal transition before vector auto-reset feedback is reused."""


def check(args):
    assert (args.nn, args.mode, args.side, args.state, args.num_players) == (
        'ClassicAIV1', 'model_vs_game', 'home', None, 1)
    assert args.max_playback_speed == 1.0
    args.hyperparams_dict = resolve_hyperparams_for_model(load_hyperparams(args.hyperparams), args.nn)
    player = NHL94Player(args, None, need_display=True)
    try:
        assert args.state == 'PenguinsVsSenators.start'
        player.display_frame_interval = 0
        shown_step = player.display_env.step
        frames, before, attempts = 0, None, 0
        shots, terminal = [], {}

        def capture(actions):
            nonlocal frames, before, attempts
            if before is None:
                initial = player.display_env.env.reset_infos[0]
                assert (initial['defense_team1'], initial['defense_team2']) == (1, 0)
                before = initial['p1_onetimer'] & 65535
            controller = player.ai_sys.models[1].controller
            pending = controller._one_timer
            request = controller._last_pass_request
            result = shown_step(actions)
            info = result[3][0]
            assert (info['defense_team1'], info['defense_team2']) == (1, 0)
            current = info['p1_onetimer'] & 65535
            if current > before:
                attempts += current - before
                if (pending is not None and request is not None and request['purpose'] == 'one-timer'
                        and info['shot_player'] == pending[1] == request['receiver']):
                    shots.append((frames, info['time'] & 65535, info['shot_player']))
                before = current
            frames += 1
            assert frames < 45000, 'Supplied Penguins playback did not finish its first period'
            if result[2][0]:
                terminal.update(info)
                raise Finished
            return result

        with patch.object(player.display_env, 'step', side_effect=capture):
            try:
                player.play(continuous=False, need_reset=False)
            except Finished:
                pass
        assert terminal
        assert len(shots) >= 2, ('Too few completed deliberate one-timers', shots, attempts)
        assert shots[0][1] > 150, ('First deliberate one-timer was too late in the period', shots)
        assert attempts == len(shots), ('Counter included unattributed autonomous shots', shots, attempts)
        print(f'PASS: exact supplied Penguins command completes {attempts} deliberate ROM-counted '
              f'one-timers; events (frame, clock, receiver)={shots}; '
              f'Penguins {terminal["p1_score"]}, Senators {terminal["p2_score"]}')
    finally:
        player.close()


if __name__ == '__main__':
    with patch.dict(os.environ, {'SDL_VIDEODRIVER': 'dummy', 'SDL_AUDIODRIVER': 'dummy'}), \
            patch('nhl94_ai.evaluation.play.run', side_effect=check):
        main(['play', '--agent', 'classic-v1', '--env', 'NHL94-Genesis-v0', '--mode', 'model_vs_game',
              '--side', 'home', '--max_playback_speed', '1.0'])

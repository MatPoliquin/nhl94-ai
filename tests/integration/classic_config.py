"""Exercise the player config through real native playback; requires the ROM."""
import json
from pathlib import Path
import tempfile
from unittest.mock import patch

from tests.integration import control_rate


def main():
    parse = control_rate.parse_cmdline
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / 'classic_bot.json'

        def configured(argv):
            # Reuse the cadence assertions, but source settings only from JSON.
            argv = list(argv)
            for flag in ('--classic-control-interval', '--goalie-policy'):
                index = argv.index(flag)
                del argv[index:index + 2]
            args = parse([*argv, '--classic-config', str(path)])
            assert args.player_ratings_consideration == ratings
            assert (args.goalie_policy != 'off') == goalie
            return args

        for hz, ratings, goalie in ((60, True, True), (60, False, True), (15, False, True),
                                   (10, False, True), (6, False, True), (10, False, False)):
            path.write_text('// Player config\n' + json.dumps({
                'decision_hz': hz, 'ai_goalie': goalie,
                'player_ratings_consideration': ratings}), encoding='utf-8')
            with patch.object(control_rate, 'parse_cmdline', side_effect=configured):
                control_rate.replay(0 if hz == 60 else 60 // hz, 'home', frames=241)
            print(f'PASS: config {hz} Hz, ratings={ratings}, goalie={goalie}')


if __name__ == '__main__':
    main()

"""Installed commands. Configuration precedence: defaults < file < CLI."""
from nhl94_ai.agents.registry import ALIASES
import argparse
import importlib
import sys
from pathlib import Path

COMMANDS = {
    'train': 'training.live', 'play': 'evaluation.play',
    'collect': 'training.collect', 'bc': 'training.bc', 'dagger': 'training.dagger',
    'export': 'export', 'evaluate': 'evaluation.runner',
    'benchmark': 'evaluation.benchmark',
    'benchmark-cpu': 'evaluation.cpu_benchmark',
    'benchmark-teammates': 'evaluation.teammates',
    'external': 'evaluation.external',
    'record-human': 'evaluation.human_session',
    'replay-human': 'evaluation.human_replay',
    'replay-match': 'evaluation.human_replay',
    'record-snes': 'evaluation.snes_session',
    'replay-snes': 'evaluation.snes_replay',
    'offense': 'evaluation.offense_policy',
    'goalie': 'evaluation.goalie_policy',
}


def parse_configured(parser, argv):
    from nhl94_ai.config import load_json_dict, namespace_from_mapping, resolve_declared_paths
    probe = argparse.ArgumentParser(add_help=False)
    probe.add_argument('--config')
    config_args, _ = probe.parse_known_args(argv)
    parser.add_argument('--config', help='JSON options or training hyperparameters')
    if config_args.config:
        path = Path(config_args.config).expanduser().resolve()
        payload = load_json_dict(str(path))
        options = payload.get('options', payload)
        if any(key in options for key in ('common', 'n_steps', 'learning_rate', 'model_overrides')):
            options = {'hyperparams': str(path)}
        options = resolve_declared_paths(options, path.parent)
        # Required values may be supplied by explicit CLI overrides.
        required = [action for action in parser._actions if action.required]
        for action in required:
            action.required = False
        try:
            defaults = vars(namespace_from_mapping(parser, options))
        finally:
            for action in required:
                action.required = action.dest not in options
        parser.set_defaults(**defaults)
    return parser.parse_args(argv)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(prog='nhl94', description='NHL94 training, agents, and evaluation')
    parser.add_argument('command', choices=[*COMMANDS, 'curriculum', 'compare'])
    if not argv or argv[0] in ('-h', '--help'):
        parser.print_help()
        return 0
    command = parser.parse_args(argv[:1]).command
    rest = argv[1:]
    if command in ('train', 'collect', 'bc', 'dagger', 'curriculum', 'compare') or (
            command in ('offense', 'goalie') and rest and rest[0] in ('train', 'collect', 'bc')):
        from nhl94_ai.training.unavailable import main as unavailable_main
        return unavailable_main(rest)
    module = importlib.import_module('nhl94_ai.' + COMMANDS[command])
    command_parser = module.build_parser()
    if command == 'play':
        command_parser.add_argument('--agent', choices=list(ALIASES))
        command_parser.set_defaults(mode='model_vs_game', num_players=None, rf='PostPlay')
    args = parse_configured(command_parser, rest)
    if command == 'play':
        if args.agent:
            args.nn = ALIASES[args.agent]
        module.resolve_play_controllers(args)
        module.resolve_play_goalie_defaults(args)
    module.run(args)
    return 0

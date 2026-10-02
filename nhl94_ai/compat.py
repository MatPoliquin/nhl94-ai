"""Import aliases used only when reading historical custom-policy checkpoints."""
import importlib
import sys


def install_checkpoint_aliases():
    aliases = {
        'models': 'nhl94_ai.models.networks',
        'es': 'nhl94_ai.training.es',
    }
    for old, new in aliases.items():
        if old not in sys.modules or getattr(sys.modules[old], '__file__', None) is None:
            sys.modules[old] = importlib.import_module(new)

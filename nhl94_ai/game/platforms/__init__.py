"""Per-environment console backends; tactical code consumes common game state."""
from functools import lru_cache
from importlib import import_module


@lru_cache(maxsize=2)
def get_backend(platform='genesis'):
    implementations = {'genesis': 'GenesisBackend', 'snes': 'SnesBackend'}
    if platform not in implementations:
        raise ValueError(f'Unsupported NHL94 platform: {platform}')
    # Load console-specific ROM helpers only after selecting a game. This keeps
    # the registry independent of RAM readers and optional agent capabilities.
    module = import_module(f'nhl94_ai.game.platforms.{platform}')
    return getattr(module, implementations[platform])()

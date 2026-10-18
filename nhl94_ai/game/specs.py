"""Supported game variants; controller count is distinct from skater count."""
from dataclasses import dataclass


@dataclass(frozen=True)
class GameSpec:
    env_id: str
    skaters_per_team: int
    default_state: str
    platform: str = 'genesis'

    @property
    def backend(self):
        from nhl94_ai.game.platforms import get_backend
        return get_backend(self.platform)


GAMES = {
    spec.env_id: spec for spec in (
        GameSpec('NHL941on1-Genesis-v0', 1, 'PenguinsVsSenators.start'),
        GameSpec('NHL942on2-Genesis-v0', 2, 'PenguinsVsWhalers.start'),
        GameSpec('NHL94-Genesis-v0', 5, 'PenguinsVsSenators.start'),
        GameSpec('NHL94-Snes-v0', 5, 'DucksVsSabres.ManualGoalie.Start', 'snes'),
    )
}


def get_game(env_id):
    try:
        return GAMES[env_id]
    except KeyError as error:
        raise ValueError(f"Unsupported environment '{env_id}'. Supported NHL94 environments: {', '.join(GAMES)}") from error

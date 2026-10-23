"""Stateless component selection. Display dependencies are loaded on demand."""
from dataclasses import dataclass
from nhl94_ai.game.specs import GAMES, get_game

NHL94_ENVS = tuple(GAMES)


@dataclass(frozen=True)
class EnvironmentBindings:
    env_id: str

    @property
    def obs_env(self):
        from nhl94_ai.env.observation import NHL94Observation2PEnv
        return NHL94Observation2PEnv

    @property
    def ai_sys(self):
        from nhl94_ai.agents.multi_model import NHL94AISystem
        return NHL94AISystem

    @property
    def sp_display_env(self):
        from nhl94_ai.ui.debug import NHL94DebugDisplay
        return NHL94DebugDisplay

    @property
    def pvp_display_env(self):
        from nhl94_ai.ui.pvp import NHL94PvPGameDisplayEnv
        return NHL94PvPGameDisplayEnv


def get_bindings(args):
    return EnvironmentBindings(get_game(args.env).env_id)

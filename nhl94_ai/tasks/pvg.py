"""Goal reward and one small bonus for creating an opening before shot release."""
from nhl94_ai.game.ram import ROM_RNG_ADDRESS, register_shootout_scores
from nhl94_ai.tasks.pvg_setup import randomize_pvg_start
from nhl94_ai.tasks.setup import _randint_inclusive


def init_pvg(env, env_name):
    register_shootout_scores(env)
    env.data.memory.assign(ROM_RNG_ADDRESS, '>u4', _randint_inclusive(env.np_random, 1, 2**32 - 1))
    randomize_pvg_start(env)


def rf_pvg(state):
    if state.team1.stats.score > state.team1.last_stats.score:
        return 1.0
    return 0.05 if state.created_pre_shot_opening else 0.0


def isdone_pvg(state):
    return (
        state.team1.stats.score > state.team1.last_stats.score
        or state.team2.stats.score > state.team2.last_stats.score
        or not state.is_shootout_active
    )

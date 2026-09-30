"""Historical tuple registration API; new callers use tasks.registry."""
from typing import Tuple, Callable
from nhl94_ai.env.encoding import init_model, set_model_input
from nhl94_ai.tasks.setup import RandomPos, RandomPosAttackZone, RandomPosDefenseZone, _env_rng, _randint_inclusive, _random_pos_from_rng, _is_inside_net_area, _sample_non_net_pos_from_rng, _sample_non_net_pos_uniform, _choose_index, SampleRandomPos, SampleRandomPosAttackZone, SampleRandomPosDefenseZone, SampleRandomPosNeutralZone, ClampCameraToPlayer, input_overide, input_overide_no_shoot, input_overide_empty, SetRandomSkaterPositions, _safe_set_value, _sample_position_away_from, _set_getpuck_puck_state, _set_team1_puck_state, init_attackzone, init_general

from nhl94_ai.tasks.breakaway import init_breakaway, isdone_breakaway, rf_breakaway
from nhl94_ai.tasks.az_lostpuck import init_az_lostpuck, isdone_az_lostpuck, rf_az_lostpuck
from nhl94_ai.tasks.createopp import input_overide_createopp, isdone_createopp, rf_createopp
from nhl94_ai.tasks.defensezone import init_defensezone, isdone_defensezone, rf_defensezone
from nhl94_ai.tasks.general import isdone_general, init_general_v2, isdone_general_v2, rf_general, rf_general_v2
from nhl94_ai.tasks.getpuck import init_getpuck, init_getpuck_az, init_getpuck_dz, init_getpuck_nz, isdone_getpuck, isdone_getpuck_az, isdone_getpuck_dz, isdone_getpuck_nz, rf_getpuck
from nhl94_ai.tasks.keeppuck import init_keeppuck, isdone_keeppuck, rf_keeppuck
from nhl94_ai.tasks.passing import isdone_passing, rf_passing
from nhl94_ai.tasks.postplay import init_postplay, isdone_postplay, rf_postplay
from nhl94_ai.tasks.scoregoal import isdone_crosscrease_v2, isdone_scoregoal, isdone_scoregoal_cc, isdone_scoregoal_ot, isdone_scoregoal_v4, isdone_scoregoal_v2, rf_crosscrease_v2, rf_scoregoal, rf_scoregoal_cc, rf_scoregoal_ot, rf_scoregoal_v4, rf_scoregoal_v2
from nhl94_ai.tasks.selfplay import init_selfplay, init_selfplay_defense, init_selfplay_offense, isdone_selfplay, isdone_selfplay_defense, isdone_selfplay_offense, rf_selfplay, rf_selfplay_defense, rf_selfplay_offense

# =====================================================================
# Register Functions
# =====================================================================
MODEL_INPUT_FUNCTIONS = (init_model, set_model_input)

_reward_function_map = {
    "GetPuck": (init_getpuck, rf_getpuck, isdone_getpuck, *MODEL_INPUT_FUNCTIONS, input_overide_empty),
    "GetPuckAZ": (init_getpuck_az, rf_getpuck, isdone_getpuck_az, *MODEL_INPUT_FUNCTIONS, input_overide_empty),
    "GetPuckNZ": (init_getpuck_nz, rf_getpuck, isdone_getpuck_nz, *MODEL_INPUT_FUNCTIONS, input_overide_empty),
    "GetPuckDZ": (init_getpuck_dz, rf_getpuck, isdone_getpuck_dz, *MODEL_INPUT_FUNCTIONS, input_overide_empty),
    "AZLostPuck": (init_az_lostpuck, rf_az_lostpuck, isdone_az_lostpuck, *MODEL_INPUT_FUNCTIONS, input_overide_empty),
    "ScoreGoalCC": (init_attackzone, rf_scoregoal_cc, isdone_scoregoal_cc, *MODEL_INPUT_FUNCTIONS, input_overide_empty),
    "CrossCreaseV2": (init_attackzone, rf_crosscrease_v2, isdone_crosscrease_v2, *MODEL_INPUT_FUNCTIONS, input_overide_empty),
    "ScoreGoalOT": (init_attackzone, rf_scoregoal_ot, isdone_scoregoal_ot, *MODEL_INPUT_FUNCTIONS, input_overide_empty),
    "ScoreGoal": (init_attackzone, rf_scoregoal, isdone_scoregoal, *MODEL_INPUT_FUNCTIONS, input_overide_empty),
    "ScoreGoalV2": (init_attackzone, rf_scoregoal_v2, isdone_scoregoal_v2, *MODEL_INPUT_FUNCTIONS, input_overide_empty),
    "ScoreGoalV4": (init_attackzone, rf_scoregoal_v4, isdone_scoregoal_v4, *MODEL_INPUT_FUNCTIONS, input_overide_empty),
    "Breakaway": (init_breakaway, rf_breakaway, isdone_breakaway, *MODEL_INPUT_FUNCTIONS, input_overide_empty),
    "CreateOpp": (init_attackzone, rf_createopp, isdone_createopp, *MODEL_INPUT_FUNCTIONS, input_overide_createopp),
    "KeepPuck": (init_keeppuck, rf_keeppuck, isdone_keeppuck, *MODEL_INPUT_FUNCTIONS, input_overide_empty),
    "DefenseZone": (init_defensezone, rf_defensezone, isdone_defensezone, *MODEL_INPUT_FUNCTIONS, input_overide_empty),
    "Passing": (init_attackzone, rf_passing, isdone_passing, *MODEL_INPUT_FUNCTIONS, input_overide_no_shoot),
    "General": (init_general, rf_general, isdone_general, *MODEL_INPUT_FUNCTIONS, input_overide_empty),
    "GeneralV2": (init_general_v2, rf_general_v2, isdone_general_v2, *MODEL_INPUT_FUNCTIONS, input_overide_empty),
    "PostPlay": (init_postplay, rf_postplay, isdone_postplay, *MODEL_INPUT_FUNCTIONS, input_overide_empty),
    "SelfPlay": (init_selfplay, rf_selfplay, isdone_selfplay, *MODEL_INPUT_FUNCTIONS, input_overide_empty),
    "SelfPlayOffenseFinetune": (init_selfplay_offense, rf_selfplay_offense, isdone_selfplay_offense, *MODEL_INPUT_FUNCTIONS, input_overide_empty),
    "SelfPlayDefenseFinetune": (init_selfplay_defense, rf_selfplay_defense, isdone_selfplay_defense, *MODEL_INPUT_FUNCTIONS, input_overide_empty),
}

def register_functions(name: str) -> Tuple[Callable, Callable, Callable, Callable, Callable, Callable]:
    if name not in _reward_function_map:
        raise ValueError(f"Unsupported Reward Function: {name}")
    return _reward_function_map[name]

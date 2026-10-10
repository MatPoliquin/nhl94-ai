"""End on skater possession (+1), opponent shots or goals (-1)."""

_TIMEOUT = 200


def init_defensezone(env, env_name):
    if env_name == 'NHL94-Genesis-v0':
        from nhl94_ai.tasks.defense_setup import init_full_team_defense
        init_full_team_defense(env)
        return

    from nhl94_ai.tasks.setup import SampleRandomPosDefenseZone

    if env_name == 'NHL941on1-Genesis-v0':
        x, y = SampleRandomPosDefenseZone(env)
        env.set_value("p1_x", x)
        env.set_value("p1_y", y)
        x, y = SampleRandomPosDefenseZone(env)
        env.set_value("p2_x", x)
        env.set_value("p2_y", y)

    elif env_name == 'NHL942on2-Genesis-v0':
        x, y = SampleRandomPosDefenseZone(env)
        env.set_value("p1_x", x)
        env.set_value("p1_y", y)
        x, y = SampleRandomPosDefenseZone(env)
        env.set_value("p2_x", x)
        env.set_value("p2_y", y)
        x, y = SampleRandomPosDefenseZone(env)
        env.set_value("p2_2_x", x)
        env.set_value("p2_2_y", y)
        x, y = SampleRandomPosDefenseZone(env)
        env.set_value("p1_2_x", x)
        env.set_value("p1_2_y", y)
    else:
        raise ValueError(f"Invalid environment name, got '{env_name}'")


def _puck_carrier(state, team):
    if state.engine.puck_owner != -1:
        return team.get_player_by_scnum(state.engine.puck_owner)

    # Older integrations omit puck_owner; -256 is a known loose puck, not missing data.
    if team.player_haspuck or team.goalie_haspuck:
        return team.get_possession_player()
    return None


def isdone_defensezone(state):
    carrier = _puck_carrier(state, state.team1)
    return (
        state.team2.stats.score > state.team2.last_stats.score
        or state.team2.stats.shots > state.team2.last_stats.shots
        or (carrier is not None and carrier is not state.team1.goalie)
        or state.time < _TIMEOUT
    )


def rf_defensezone(state):
    if (state.team2.stats.score > state.team2.last_stats.score
            or state.team2.stats.shots > state.team2.last_stats.shots):
        return -1.0

    carrier = _puck_carrier(state, state.team1)
    if carrier is not None and carrier is not state.team1.goalie:
        return 1.0
    return 0.0

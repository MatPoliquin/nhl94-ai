"""Environment wrapper access and reset/step result compatibility."""


def find_wrapper_with_attr(env, attr_name):
    current = env
    while current is not None:
        if hasattr(current, attr_name):
            return current
        current = getattr(current, "env", None)
    raise AttributeError(f"Could not find wrapper attribute: {attr_name}")


def get_game_state(env):
    return find_wrapper_with_attr(env, "game_state").game_state


def normalize_reset_result(reset_result):
    if isinstance(reset_result, tuple) and len(reset_result) == 2:
        return reset_result
    return reset_result, {}


def normalize_step_result(step_result):
    if len(step_result) == 5:
        obs, reward, terminated, truncated, info = step_result
        return obs, reward, bool(terminated or truncated), info
    obs, reward, done, info = step_result
    return obs, reward, bool(done), info

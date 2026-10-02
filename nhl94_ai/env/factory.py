from nhl94_ai.agents.registry import CONTROLLERS
import os
import numpy as np
from stable_baselines3.common.atari_wrappers import WarpFrame
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecFrameStack
from stable_baselines3.common.monitor import Monitor
import gymnasium as gym
import stable_retro as retro
import stable_retro.data
import nhl94_ai.env.components as games
import cv2
from nhl94_ai.env.wrappers import EpisodeROMSeed, StochasticFrameSkip, WarpFrameDict, RewardClipper
from nhl94_ai.config import EnvironmentConfig, resolve_clip_reward, resolve_sticky_action_settings


from nhl94_ai.game.specs import GAMES, get_game
from nhl94_ai.game.ram import (
    register_defense_state, register_goalie_control, register_goalie_motion, register_pass_state,
    register_skater_ratings, register_target_state, validate_rom_seed,
)
NHL94_DEFAULT_STATES = {name: spec.default_state for name, spec in GAMES.items()}


def isMLP(name):
    return name == 'MlpPolicy' or name == 'MlpDropoutPolicy' or name == 'CombinedPolicy' \
          or name == 'AttentionMLPPolicy' or name == 'HockeyMultiHeadPolicy' \
          or name == 'ResidualMlpPolicy' \
          or name == 'HybridMambaPolicy' or name == 'GRUMlpPolicy' \
          or name in CONTROLLERS


def resolve_backend_action_type(args, num_players):
    requested_action_type = getattr(args, 'action_type', 'FILTERED').upper()
    if requested_action_type in ('HOCKEY_INTENT_DPAD', 'TARGET_POSITION'):
        return 'FILTERED'
    return requested_action_type


def resolve_default_state(game, state):
    if state is not None:
        return state
    return NHL94_DEFAULT_STATES.get(game, state)


def resolve_retro_state_name(game, state, num_players, inttype):
    if num_players <= 1:
        return state

    if state in (None, retro.State.DEFAULT, retro.State.NONE):
        return state

    if not isinstance(state, str):
        return state

    base_state = state[:-6] if state.endswith('.state') else state
    if base_state.endswith('.2P'):
        return base_state

    candidate = f"{base_state}.2P"
    if stable_retro.data.get_file_path(game, f"{candidate}.state", inttype):
        return candidate

    return base_state


def make_retro(
    *,
    game,
    state=None,
    num_players,
    max_episode_steps=4500,
    action_type='FILTERED',
    goalie_policy='off',
    inttype=retro.data.Integrations.ALL,
    **kwargs,
):
    import stable_retro as retro  # pylint: disable=import-outside-toplevel,reimported
    state = resolve_default_state(game, state)
    if state is None:
        state = retro.State.DEFAULT

    # Convert action_type string to retro.Actions enum
    action_map = {
        'FILTERED': retro.Actions.FILTERED,
        'DISCRETE': retro.Actions.DISCRETE,
        'MULTI_DISCRETE': retro.Actions.MULTI_DISCRETE
    }
    action_enum = action_map.get(action_type.upper(), retro.Actions.FILTERED)
    if goalie_policy != 'off':
        if goalie_policy not in ('selective', 'always') or game != 'NHL94-Genesis-v0' or action_type.upper() != 'FILTERED':
            raise ValueError('Manual goalie buttons require full-team FILTERED controls')
        # The integration's FILTERED button group omits A (dives and clears).
        # ALL has the same 12 binary fields; the opt-in controller owns button edges.
        action_enum = retro.Actions.ALL
    state = resolve_retro_state_name(game, state, num_players, inttype)

    env = retro.make(
        game,
        state,
        **kwargs,
        players=num_players,
        render_mode="rgb_array",
        use_restricted_actions=action_enum,
        inttype=inttype,
    )
    if game in GAMES:
        env.pass_geometry_rom = stable_retro.data.get_romfile_path(game, inttype)
        register_skater_ratings(env, GAMES[game].skaters_per_team)
        register_pass_state(env)
        register_goalie_motion(env, GAMES[game].skaters_per_team)
        register_goalie_control(env, GAMES[game].skaters_per_team)
        register_defense_state(env, GAMES[game].skaters_per_team)
    #env = NHL94Discretizer(env)
    #if max_episode_steps is not None:
    #    env = TimeLimit(env, max_episode_steps=max_episode_steps)
    return env

def build_single_nhl94_env(
    args,
    hyperparams,
    *,
    num_players=1,
    output_path=None,
    use_sticky_action=False,
    use_frame_skip=True,
    monitor=False,
    action_seed=None,
    episode_rom_seed=None,
):
    EnvironmentConfig.from_args(args)
    if episode_rom_seed is not None:
        validate_rom_seed(episode_rom_seed)
        get_game(args.env)
    target_mode = getattr(args, 'action_type', '').upper() == 'TARGET_POSITION'
    if target_mode and num_players != 1:
        raise ValueError('TARGET_POSITION requires one emulator controller')
    interval = hyperparams.get('frame_skip', 4)
    if target_mode and (isinstance(interval, bool) or not isinstance(interval, int) or interval < 1):
        raise ValueError('TARGET_POSITION frame_skip must be a positive integer')
    args.hyperparams_dict = hyperparams
    games.get_bindings(args)
    args.state = resolve_default_state(args.env, getattr(args, "state", None))
    backend_action_type = resolve_backend_action_type(args, num_players)
    env = make_retro(
        game=args.env,
        action_type=backend_action_type,
        state=args.state,
        num_players=num_players,
        goalie_policy=getattr(args, 'goalie_policy', 'off'),
    )
    if episode_rom_seed is not None:
        env = EpisodeROMSeed(env, episode_rom_seed)
    if target_mode:
        register_target_state(env)

    if action_seed is not None:
        env.action_space.seed(action_seed)

    if isMLP(args.nn):
        env = games.get_bindings(args).obs_env(env, args, num_players, args.rf)
        if target_mode and action_seed is not None:
            env.action_space.seed(action_seed)

    if monitor or output_path:
        env = Monitor(env, output_path, allow_early_resets=True)

    if use_frame_skip:
        frame_skip = max(1, int(hyperparams.get("frame_skip", 4)))
        sticky_enabled, sticky_prob = resolve_sticky_action_settings(
            use_sticky_action,
            hyperparams,
        )
        env = StochasticFrameSkip(env, n=frame_skip, stickprob=sticky_prob if sticky_enabled else -1,
                                  stop_on_truncation=target_mode)

    if not isMLP(args.nn):
        env = WarpFrame(env)
    elif args.nn == "CombinedPolicy":
        env = WarpFrameDict(env)

    if resolve_clip_reward(args, hyperparams):
        env = RewardClipper(env, low=-1.0, high=1.0)

    return env

def init_env(
    output_path,
    num_env,
    state,
    num_players,
    args,
    hyperparams,
    use_sticky_action=True,
    use_display=False,
    use_frame_skip=True,
    *,
    episode_rom_seed=None,
):
    EnvironmentConfig.from_args(args)
    args.hyperparams_dict = hyperparams
    wrapper_kwargs = {}

    if getattr(args, 'selfplay', False) and not isMLP(args.nn):
        raise ValueError('Self-play currently requires an MLP-style observation wrapper.')

    clip_reward = resolve_clip_reward(args, hyperparams)
    args.clip_reward = clip_reward
    sticky_actions_enabled, sticky_action_prob = resolve_sticky_action_settings(
        use_sticky_action,
        hyperparams,
    )
    frame_skip = max(1, int(hyperparams.get('frame_skip', 4)))

    seed = getattr(args, "seed", 0)
    if seed is None:
        seed = 0
    start_index = 0
    start_method = os.environ.get('RETRO_VECENV_START_METHOD')
    allow_early_resets=True
    env_num_players = 2 if getattr(args, 'selfplay', False) else num_players
    state = resolve_default_state(args.env, state)
    if getattr(args, 'state', None) is None and state is not None:
        args.state = state

    # Prefer spawn for self-play unless explicitly overridden.
    if start_method is None and getattr(args, 'selfplay', False):
        start_method = 'spawn'

    def make_env(rank):
        def _thunk():
            env = build_single_nhl94_env(
                args, hyperparams, num_players=env_num_players,
                output_path=output_path and os.path.join(output_path, str(rank)),
                use_sticky_action=use_sticky_action, use_frame_skip=use_frame_skip,
                monitor=True, action_seed=seed + rank,
                episode_rom_seed=episode_rom_seed,
            )
            return env
        return _thunk

    env = SubprocVecEnv([make_env(i + start_index) for i in range(num_env)], start_method=start_method)

    env.seed(seed)

    if not isMLP(args.nn) or args.nn == 'CombinedPolicy':
        env = VecFrameStack(env, n_stack=4)
        #env = VecTransposeImage(env)

    return env


def get_button_names(args):
    action_map = {
        'FILTERED': retro.Actions.FILTERED,
        'DISCRETE': retro.Actions.DISCRETE,
        'MULTI_DISCRETE': retro.Actions.MULTI_DISCRETE,
        'HOCKEY_INTENT_DPAD': retro.Actions.FILTERED,
    }
    backend_action_type = resolve_backend_action_type(args, 2 if getattr(args, 'selfplay', False) else args.num_players)
    action_enum = action_map.get(backend_action_type.upper(), retro.Actions.FILTERED)
    if getattr(args, 'goalie_policy', 'off') != 'off':
        action_enum = retro.Actions.ALL
    state = resolve_default_state(args.env, args.state)
    if getattr(args, 'state', None) is None and state is not None:
        args.state = state

    env = retro.make(
        game=args.env,
        state=state,
        use_restricted_actions=action_enum,
        players=2 if getattr(args, 'selfplay', False) else args.num_players,
        inttype=retro.data.Integrations.ALL,
    )
    try:
        games.get_bindings(args)
        if isMLP(args.nn) and games.get_bindings(args).obs_env is not None:
            wrapped_env = games.get_bindings(args).obs_env(
                env,
                args,
                2 if getattr(args, 'selfplay', False) else args.num_players,
                getattr(args, 'rf', ''),
            )
            if hasattr(wrapped_env, 'get_action_names'):
                return wrapped_env.get_action_names()
        return env.buttons
    finally:
        env.close()

def init_play_env(args, num_players, hyperparams, is_pvp_display=False, need_display=True, use_frame_skip=True):
    button_names = get_button_names(args)

    env = init_env(
        None,
        1,
        args.state,
        num_players,
        args,
        hyperparams,
        use_sticky_action=False,
        use_display=False,
        use_frame_skip=use_frame_skip,
        episode_rom_seed=getattr(args, 'seed', None),
    )

    if not need_display:
        return env

    games.get_bindings(args)

    if is_pvp_display:
        display_env = env = games.get_bindings(args).pvp_display_env(env, args, args.model1_desc, args.model2_desc, None, None, button_names)
    else:
        display_env = env = games.get_bindings(args).sp_display_env(env, args, 0, args.model1_desc, button_names)

    return display_env

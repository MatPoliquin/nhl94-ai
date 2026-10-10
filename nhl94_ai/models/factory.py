
from nhl94_ai.artifacts import load_policy
import math
import os
import torch as th

# Avoid noisy NNPACK init warnings on some CPUs (common on some AMD systems)
# and force PyTorch to use other backends.
try:
    import torch.backends.nnpack as nnpack

    nnpack.set_flags(False)
except Exception:
    pass

from stable_baselines3 import PPO
from nhl94_ai.models.networks import CustomMlpPolicy, CustomPolicy, ViTPolicy, DartPolicy, AttentionMLPPolicy, CustomCNN, CustomImpalaFeatureExtractor, CNNTransformer, HockeyMultiHeadPolicy, HybridMambaPolicy, GRUMlpPolicy, ResidualMlpPolicy
from nhl94_ai.models.mlp import SquashedMlpPolicy
from nhl94_ai.training.es import EvolutionStrategies


VECTOR_POLICIES = (
    'MlpPolicy',
    'AttentionMLPPolicy',
    'CustomMlpPolicy',
    'ResidualMlpPolicy',
    'HybridMambaPolicy',
    'GRUMlpPolicy',
)


def load_model_for_inference(player_model, player_alg='ppo2'):
    if not player_model:
        raise ValueError('player_model must be provided for inference loading')

    if player_alg == 'ppo2':
        return load_policy(os.path.expanduser(player_model), device='cpu')

    raise NotImplementedError(f"Inference loading is not implemented for algorithm '{player_alg}'")

def get_num_parameters(model):
    if not hasattr(model, "policy"):
        return 0
    total_params = sum(p.numel() for p in model.policy.parameters() if p.requires_grad)
    return total_params

def get_model_probabilities(model, state):
    if hasattr(model, "get_action_preferences"):
        return model.get_action_preferences(state)

    #obs = obs_as_tensor(state, model.policy.device)
    obs = model.policy.obs_to_tensor(state)[0]
    dis = model.policy.get_distribution(obs)
    distribution = dis.distribution
    if isinstance(distribution, list):
        probs = th.cat([dist.probs for dist in distribution], dim=1)
    else:
        probs = distribution.probs
    probs_np = probs.detach().cpu().numpy()
    return probs_np

def print_model_summary(args, env, player_model, model):
    if not getattr(args, 'alg_verbose', True):
        return

    if not hasattr(model, "policy"):
        return

    obs_space = getattr(model, "observation_space", None) or env.observation_space
    print(obs_space)

    if args.alg == 'es':
        return

    # Handle policies
    if args.nn in VECTOR_POLICIES:
        obs_shape = obs_space.shape
        pytorch_obs_shape = tuple(obs_shape)
    elif args.nn in ('CnnPolicy', 'ImpalaCnnPolicy'):
        # SB3 model.observation_space is already channels-first and includes frame stack
        obs_shape = obs_space.shape  # (C, H, W)
        pytorch_obs_shape = (obs_shape[0], obs_shape[1], obs_shape[2])
    else:
        # Other types are not supported for summary for now
        return

    from torchsummary import summary
    summary(model.policy, pytorch_obs_shape)

def build_MlpPolicy(size, hyperparams):
    nn_type = 'MlpPolicy'
    policy_kwargs = dict(activation_fn=th.nn.ReLU, net_arch=hyperparams.get('net_arch', dict(pi=[size, size], vf=[size, size])))
    squash_output = hyperparams.get('squash_output', False)
    if not isinstance(squash_output, bool):
        raise TypeError('squash_output must be a JSON boolean')
    if squash_output:
        nn_type = SquashedMlpPolicy
    if 'log_std_init' in hyperparams:
        log_std_init = float(hyperparams['log_std_init'])
        if not math.isfinite(log_std_init):
            raise ValueError('log_std_init must be finite')
        policy_kwargs['log_std_init'] = log_std_init
    return nn_type, policy_kwargs


def build_CustomMlpPolicy(size, hyperparams):
    nn_type = CustomMlpPolicy
    policy_kwargs = dict(activation_fn=th.nn.ReLU, net_arch=hyperparams.get('net_arch', [size, size]), dropout_prob=hyperparams.get('dropout_prob', 0.3))
    return nn_type, policy_kwargs


def build_ResidualMlpPolicy(size, hyperparams):
    nn_type = ResidualMlpPolicy
    residual_kwargs = hyperparams.get('residual_mlp', {}) if hyperparams else {}
    if not isinstance(residual_kwargs, dict):
        raise TypeError("hyperparams['residual_mlp'] must be a dict when provided")
    policy_kwargs = dict(residual_mlp=residual_kwargs)
    return nn_type, policy_kwargs


def build_CustomCnnPolicy(size, hyperparams):
    nn_type = 'CnnPolicy'
    policy_kwargs = dict(features_extractor_class=CustomCNN, features_extractor_kwargs=dict(features_dim=hyperparams.get('features_dim', 128)))
    return nn_type, policy_kwargs


def build_ImpalaCnnPolicy(size, hyperparams):
    nn_type = 'CnnPolicy'
    policy_kwargs = dict(features_extractor_class=CustomImpalaFeatureExtractor)
    return nn_type, policy_kwargs


def build_CombinedPolicy(size, hyperparams):
    nn_type = CustomPolicy
    policy_kwargs = dict(net_arch=hyperparams.get('net_arch', dict(pi=[size, size], vf=[size, size])), features_extractor_kwargs=dict(features_dim=hyperparams.get('features_dim', 128)))
    return nn_type, policy_kwargs


def build_ViTPolicy(size, hyperparams):
    nn_type = ViTPolicy
    policy_kwargs = {}
    return nn_type, policy_kwargs


def build_DartPolicy(size, hyperparams):
    nn_type = DartPolicy
    policy_kwargs = dict(net_arch=hyperparams.get('net_arch', dict(pi=[size, size], vf=[size, size])), features_extractor_kwargs=dict(features_dim=hyperparams.get('features_dim', 128)))
    return nn_type, policy_kwargs


def build_AttentionMLPPolicy(size, hyperparams):
    nn_type = AttentionMLPPolicy
    attention_kwargs = hyperparams.get('attention_mlp', {}) if hyperparams else {}
    if not isinstance(attention_kwargs, dict):
        raise TypeError("hyperparams['attention_mlp'] must be a dict when provided")
    features_extractor_kwargs = dict(attention_kwargs)
    features_extractor_kwargs.setdefault('features_dim', hyperparams.get('features_dim', 128))
    policy_kwargs = dict(activation_fn=th.nn.ReLU, net_arch=hyperparams.get('net_arch', dict(pi=[size, size], vf=[size, size])), features_extractor_kwargs=features_extractor_kwargs)
    return nn_type, policy_kwargs


def build_CnnTransformerPolicy(size, hyperparams):
    nn_type = 'CnnPolicy'
    policy_kwargs = dict(features_extractor_class=CNNTransformer, features_extractor_kwargs=dict(features_dim=512, nhead=8, num_layers=3))
    return nn_type, policy_kwargs


def build_HockeyMultiHeadPolicy(size, hyperparams):
    nn_type = HockeyMultiHeadPolicy
    policy_kwargs = dict(features_extractor_kwargs=dict())
    return nn_type, policy_kwargs


def build_HybridMambaPolicy(size, hyperparams):
    nn_type = HybridMambaPolicy
    hybrid_kwargs = hyperparams.get('hybrid_mamba', {}) if hyperparams else {}
    if not isinstance(hybrid_kwargs, dict):
        raise TypeError("hyperparams['hybrid_mamba'] must be a dict when provided")
    features_extractor_kwargs = dict(hybrid_kwargs)
    features_extractor_kwargs.setdefault('features_dim', hyperparams.get('features_dim', 256))
    policy_kwargs = dict(activation_fn=th.nn.ReLU, net_arch=hyperparams.get('net_arch', dict(pi=[size, size], vf=[size, size])), features_extractor_kwargs=features_extractor_kwargs)
    return nn_type, policy_kwargs


def build_GRUMlpPolicy(size, hyperparams):
    nn_type = GRUMlpPolicy
    gru_kwargs = hyperparams.get('gru_mlp', {}) if hyperparams else {}
    if not isinstance(gru_kwargs, dict):
        raise TypeError("hyperparams['gru_mlp'] must be a dict when provided")
    features_extractor_kwargs = dict(gru_kwargs)
    features_extractor_kwargs.setdefault('features_dim', hyperparams.get('features_dim', 256))
    policy_kwargs = dict(activation_fn=th.nn.ReLU, net_arch=hyperparams.get('net_arch', dict(pi=[size, size], vf=[size, size])), features_extractor_kwargs=features_extractor_kwargs)
    return nn_type, policy_kwargs


MODEL_BUILDERS = {
    'MlpPolicy': build_MlpPolicy,
    'CustomMlpPolicy': build_CustomMlpPolicy,
    'ResidualMlpPolicy': build_ResidualMlpPolicy,
    'CustomCnnPolicy': build_CustomCnnPolicy,
    'ImpalaCnnPolicy': build_ImpalaCnnPolicy,
    'CombinedPolicy': build_CombinedPolicy,
    'ViTPolicy': build_ViTPolicy,
    'DartPolicy': build_DartPolicy,
    'AttentionMLPPolicy': build_AttentionMLPPolicy,
    'CnnTransformerPolicy': build_CnnTransformerPolicy,
    'HockeyMultiHeadPolicy': build_HockeyMultiHeadPolicy,
    'HybridMambaPolicy': build_HybridMambaPolicy,
    'GRUMlpPolicy': build_GRUMlpPolicy,
    'CnnPolicy': lambda size, hyperparams: ('CnnPolicy', None),
}

def register_model(name, builder):
    if name in MODEL_BUILDERS:
        raise ValueError(f'Model already registered: {name}')
    MODEL_BUILDERS[name] = builder


def init_model(output_path, player_model, player_alg, args, env, logger, hyperparams):
    from nhl94_ai.artifacts import run_metadata
    policy_kwargs = None
    nn_type = args.nn

    from nhl94_ai.agents.registry import CONTROLLERS, create_scripted
    if args.nn in CONTROLLERS:
        return create_scripted(args.nn, args, env)

    size = args.nnsize

    if hyperparams is None:
        raise ValueError("hyperparams must be provided; load them via utils.load_hyperparams")

    try:
        nn_type, policy_kwargs = MODEL_BUILDERS[args.nn](size, hyperparams)
    except KeyError as error:
        raise ValueError(f'Unknown neural policy: {args.nn}') from error

    if player_alg == 'ppo2':
        if player_model == '':
            batch_size = hyperparams.get('batch_size', 256)
            if getattr(args, 'alg_verbose', True):
                print("batch_size:%d" % batch_size)
            model = PPO(
                policy=nn_type,
                env=env,
                policy_kwargs=policy_kwargs,
                verbose=1 if getattr(args, 'alg_verbose', True) else 0,
                n_steps=hyperparams.get('n_steps', 2048),
                n_epochs=hyperparams.get('n_epochs', 4),
                batch_size=batch_size,
                learning_rate=hyperparams.get('learning_rate', 2.5e-4),
                clip_range=hyperparams.get('clip_range', 0.2),
                vf_coef=hyperparams.get('vf_coef', 0.5),
                ent_coef=hyperparams.get('ent_coef', 0.01),
                max_grad_norm=hyperparams.get('max_grad_norm', 0.5),
                clip_range_vf=hyperparams.get('clip_range_vf', None),
                gamma=hyperparams.get('gamma', 0.99),
                gae_lambda=hyperparams.get('gae_lambda', 0.95),
                normalize_advantage=hyperparams.get('normalize_advantage', True),
                target_kl=hyperparams.get('target_kl', None),
                seed=getattr(args, 'policy_seed', None)
            )
        else:
            model = load_policy(os.path.expanduser(player_model), env=env, expected=run_metadata(args, hyperparams))

        model.set_logger(logger)

    elif player_alg == 'es':
        es = EvolutionStrategies(env, args, 1, None)
        return es

    from nhl94_ai.artifacts import run_metadata
    metadata = run_metadata(args, hyperparams)
    if player_model:
        metadata['parent_artifact'] = {'path': str(player_model), 'compatibility': getattr(model, '_nhl94_metadata', {}).get('compatibility', 'known')}
    model._nhl94_metadata = metadata
    print_model_summary(args, env.unwrapped, player_model, model)


    return model

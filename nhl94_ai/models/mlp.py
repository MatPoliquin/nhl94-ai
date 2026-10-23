from gymnasium import spaces
import torch.nn as nn
from torch.distributions import TanhTransform
from stable_baselines3.common.distributions import SquashedDiagGaussianDistribution
from stable_baselines3.common.policies import ActorCriticPolicy


class SquashedGaussianDistribution(SquashedDiagGaussianDistribution):
    def entropy(self):
        # Fresh reparameterized samples estimate bounded entropy, not the entropy
        # of the unbounded Gaussian or cross-entropy against old rollout actions.
        raw = self.distribution.rsample()
        log_jacobian = TanhTransform().log_abs_det_jacobian(raw, raw.tanh())
        return (self.distribution.entropy() + log_jacobian).sum(dim=1)


class SquashedMlpPolicy(ActorCriticPolicy):
    """Ordinary MLP heads with a tanh-Gaussian distribution for target actions."""

    def _build(self, lr_schedule):
        if (not isinstance(self.action_space, spaces.Box)
                or self.action_space.shape != (2,)
                or not (self.action_space.low == -1).all()
                or not (self.action_space.high == 1).all()):
            raise ValueError('SquashedMlpPolicy requires a two-coordinate Box(-1, 1) target action space')
        if self.use_sde:
            raise ValueError('SquashedMlpPolicy uses independent Gaussian noise, not gSDE')
        self.action_dist = SquashedGaussianDistribution(2)
        self._squash_output = True
        super()._build(lr_schedule)


def _resolve_activation(activation_name):
    activations = {
        "relu": nn.ReLU,
        "gelu": nn.GELU,
        "silu": nn.SiLU,
        "tanh": nn.Tanh,
    }
    normalized_name = str(activation_name).strip().lower()
    if normalized_name not in activations:
        raise ValueError(
            f"Unsupported activation '{activation_name}'. "
            f"Expected one of: {', '.join(sorted(activations))}"
        )
    return activations[normalized_name]


class ResidualMLPBlock(nn.Module):
    def __init__(self, hidden_dim, activation_cls):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.activation = activation_cls()

    def forward(self, x):
        residual = x
        x = self.norm(x)
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return residual + x


class ResidualMLPHead(nn.Module):
    def __init__(self, input_dim, head_dim, activation_cls):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, head_dim),
            nn.LayerNorm(head_dim),
            activation_cls(),
        )

    def forward(self, x):
        return self.net(x)


class ResidualMLPExtractor(nn.Module):
    def __init__(
        self,
        feature_dim,
        hidden_dim=512,
        num_blocks=4,
        head_dim=256,
        activation="relu",
    ):
        super().__init__()

        if num_blocks < 1:
            raise ValueError("ResidualMLPExtractor requires at least one residual block.")

        activation_cls = _resolve_activation(activation)
        self.input_proj = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            activation_cls(),
        )
        self.shared_blocks = nn.ModuleList(
            [ResidualMLPBlock(hidden_dim, activation_cls) for _ in range(num_blocks)]
        )
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.policy_head = ResidualMLPHead(hidden_dim, head_dim, activation_cls)
        self.value_head = ResidualMLPHead(hidden_dim, head_dim, activation_cls)

        self.latent_dim_pi = head_dim
        self.latent_dim_vf = head_dim

    def _shared_forward(self, features):
        shared = self.input_proj(features)
        for block in self.shared_blocks:
            shared = block(shared)
        return self.output_norm(shared)

    def forward(self, features):
        shared = self._shared_forward(features)
        return self.policy_head(shared), self.value_head(shared)

    def forward_actor(self, features):
        shared = self._shared_forward(features)
        return self.policy_head(shared)

    def forward_critic(self, features):
        shared = self._shared_forward(features)
        return self.value_head(shared)


class ResidualMlpPolicy(ActorCriticPolicy):
    def __init__(self, *args, residual_mlp=None, **kwargs):
        self.residual_mlp_kwargs = dict(residual_mlp or {})
        super().__init__(*args, **kwargs)

    def _build_mlp_extractor(self):
        extractor_kwargs = dict(self.residual_mlp_kwargs)
        extractor_kwargs.setdefault("hidden_dim", 512)
        extractor_kwargs.setdefault("num_blocks", 4)
        extractor_kwargs.setdefault("head_dim", 256)
        extractor_kwargs.setdefault("activation", "relu")
        self.mlp_extractor = ResidualMLPExtractor(self.features_dim, **extractor_kwargs)

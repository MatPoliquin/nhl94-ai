import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy
import numpy as np


class HockeyFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=512):
        super().__init__(observation_space, features_dim)
        with torch.no_grad():
            dummy = torch.as_tensor(observation_space.sample()[None]).float()
            input_dim = dummy.shape[1]

        self.net = nn.Sequential(
            nn.Linear(input_dim, features_dim*2),
            nn.ReLU(),
            nn.Linear(features_dim*2, features_dim),
            nn.ReLU()
        )
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.net(observations)


class HockeyMultiHeadExtractor(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        # Offense network
        self.offense_net = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256)
        )
        # Defense network
        self.defense_net = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256)
        )
        # Shared network
        self.shared_net = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU()
        )
        # Value network
        self.value_net = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256)
        )

        # Required by SB3
        self.latent_dim_pi = 256 + 256 + 128  # offense + defense + shared
        self.latent_dim_vf = 256  # value network output

    def forward(self, features):
        """Main forward pass (required by SB3)"""
        return self.forward_actor(features), self.forward_critic(features)

    def forward_actor(self, features):
        """Forward pass for policy network"""
        offense = self.offense_net(features)
        defense = self.defense_net(features)
        shared = self.shared_net(features)
        return torch.cat([offense, defense, shared], dim=-1)

    def forward_critic(self, features):
        """Forward pass for value network"""
        return self.value_net(features)


class HockeyMultiHeadPolicy(ActorCriticPolicy):
    def __init__(self, observation_space, action_space, lr_schedule, **kwargs):
        features_extractor_kwargs = kwargs.pop('features_extractor_kwargs', {})
        features_extractor_kwargs['features_dim'] = 512  # Match this with your extractor output

        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            features_extractor_class=HockeyFeatureExtractor,
            features_extractor_kwargs=features_extractor_kwargs,
            net_arch=[],  # We handle architecture manually
            **kwargs
        )

        # Replace default extractor with our custom one
        self.mlp_extractor = HockeyMultiHeadExtractor(self.features_dim)

        # Manually define policy and value networks with correct dimensions
        self.action_net = nn.Linear(self.mlp_extractor.latent_dim_pi, action_space.n)
        self.value_net = nn.Linear(self.mlp_extractor.latent_dim_vf, 1)

        # Initialize weights properly
        self.apply(self._initialize_weights)

    def _initialize_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            nn.init.constant_(module.bias, 0.0)

    def forward(self, obs, deterministic=False):
        features = self.extract_features(obs)
        latent_pi, latent_vf = self.mlp_extractor(features)

        # Get action distribution
        distribution = self._get_action_dist_from_latent(latent_pi)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)

        # Get value estimate
        values = self.value_net(latent_vf)

        return actions, values, log_prob

    def predict_values(self, obs):
        features = self.extract_features(obs)
        _, latent_vf = self.mlp_extractor(features)
        return self.value_net(latent_vf)


class AttentionMLP(BaseFeaturesExtractor):
    """
    Custom MLP with self-attention for hockey.
    Input: Structured data (e.g., player positions, velocities).
    Output: Feature vector for RL policy/value heads.
    """
    def __init__(self, observation_space, features_dim=64, hidden_dim=128):
        super().__init__(observation_space, features_dim)
        self.num_features = observation_space.shape[0]
        self.hidden_dim = hidden_dim

        # Self-attention layer
        self.query = nn.Linear(self.num_features, self.num_features)
        self.key = nn.Linear(self.num_features, self.num_features)

        # MLP layers
        self.mlp = nn.Sequential(
            nn.Linear(self.num_features, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, features_dim),
            nn.ReLU()
        )

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        # Self-attention
        q = self.query(x)  # [batch_size, num_features]
        k = self.key(x)    # [batch_size, num_features]

        # Compute attention scores
        # q.unsqueeze(1): [batch_size, 1, num_features]
        # k.unsqueeze(2): [batch_size, num_features, 1]
        # bmm result: [batch_size, 1, 1]
        attention_scores = torch.bmm(q.unsqueeze(1), k.unsqueeze(2)) / (self.num_features ** 0.5)

        # Apply softmax - need to specify dimension
        weights = torch.softmax(attention_scores, dim=-1)  # Apply along last dimension

        # Apply attention weights
        attended_x = weights.squeeze(-1) * x  # [batch_size, num_features]

        # MLP
        return self.mlp(attended_x)


class AttentionMLPPolicy(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        features_extractor_kwargs = kwargs.pop("features_extractor_kwargs", {})
        if not isinstance(features_extractor_kwargs, dict):
            raise TypeError("features_extractor_kwargs must be a dict when provided.")
        features_extractor_kwargs = dict(features_extractor_kwargs)
        features_extractor_kwargs.setdefault("features_dim", 128)

        super().__init__(
            *args,
            **kwargs,
            features_extractor_class=AttentionMLP,
            features_extractor_kwargs=features_extractor_kwargs,
        )

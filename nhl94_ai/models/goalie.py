"""Masked goalie policies and versioned, independent checkpoint contracts."""
import hashlib
import json
from pathlib import Path

import torch
from torch import nn
from stable_baselines3.common.distributions import CategoricalDistribution
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from nhl94_ai.agents.learned_goalie import N_ACTIONS, SCHEMA, TOKEN_COUNT, TOKEN_WIDTH
from nhl94_ai.models.decision_ppo import DecisionPPO


class GoalieExtractor(BaseFeaturesExtractor):
    """Current-state MLP or entity attention over the identical physical inputs."""

    def __init__(self, observation_space, features_dim=128, architecture='transformer', hidden_dim=64):
        super().__init__(observation_space, features_dim)
        self.architecture = architecture
        width = TOKEN_COUNT * TOKEN_WIDTH
        if observation_space.shape != (width + N_ACTIONS,):
            raise ValueError('Goalie observation schema mismatch')
        if architecture == 'mlp':
            self.network = nn.Sequential(nn.Linear(width, hidden_dim * 2), nn.LayerNorm(hidden_dim * 2),
                                         nn.Tanh(), nn.Linear(hidden_dim * 2, features_dim), nn.Tanh())
        elif architecture == 'transformer':
            if hidden_dim % 4:
                raise ValueError('Goalie transformer width must be divisible by four')
            self.input_projection = nn.Linear(TOKEN_WIDTH, hidden_dim)
            self.entities = nn.Parameter(torch.zeros(1, TOKEN_COUNT, hidden_dim))
            nn.init.normal_(self.entities, std=0.02)
            layer = nn.TransformerEncoderLayer(hidden_dim, 4, hidden_dim * 2, dropout=0,
                                               activation='gelu', batch_first=True, norm_first=True)
            self.attention = nn.TransformerEncoder(layer, 2, norm=nn.LayerNorm(hidden_dim),
                                                   enable_nested_tensor=False)
            self.output = nn.Sequential(nn.Linear(hidden_dim * 2, features_dim), nn.Tanh())
        else:
            raise ValueError('Goalie architecture must be mlp or transformer')

    def forward(self, observations):
        physical = observations[:, :-N_ACTIONS]
        if self.architecture == 'mlp':
            return self.network(physical)
        tokens = physical.reshape(-1, TOKEN_COUNT, TOKEN_WIDTH)
        tokens = self.attention(self.input_projection(tokens) + self.entities)
        return self.output(torch.cat((tokens[:, 0], tokens[:, -1]), dim=-1))


class GoaliePolicy(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault('features_extractor_class', GoalieExtractor)
        kwargs.setdefault('net_arch', dict(pi=[64], vf=[64]))
        super().__init__(*args, **kwargs)

    def _distribution(self, obs, latent):
        mask = obs[:, -N_ACTIONS:] > 0.5
        logits = self.action_net(latent).masked_fill(~mask, -1e9)
        return CategoricalDistribution(N_ACTIONS).proba_distribution(logits)

    def forward(self, obs, deterministic=False):
        pi, vf = self.mlp_extractor(self.extract_features(obs))
        distribution = self._distribution(obs, pi)
        actions = distribution.get_actions(deterministic=deterministic)
        return actions, self.value_net(vf), distribution.log_prob(actions)

    def evaluate_actions(self, obs, actions):
        pi, vf = self.mlp_extractor(self.extract_features(obs))
        distribution = self._distribution(obs, pi)
        return self.value_net(vf), distribution.log_prob(actions), distribution.entropy()

    def get_distribution(self, obs):
        pi = self.mlp_extractor.forward_actor(self.extract_features(obs))
        return self._distribution(obs, pi)


def save_model(model, path, **details):
    path = Path(path).resolve().with_suffix('.zip')
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + '.tmp.zip')
    model.save(temporary)
    temporary.replace(path)
    metadata = {'schema': SCHEMA, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                'algorithm': 'DecisionPPO', 'timesteps': model.num_timesteps,
                'parameters': sum(parameter.numel() for parameter in model.policy.parameters()), **details}
    Path(str(path) + '.goalie.json').write_text(json.dumps(metadata, indent=2) + '\n', encoding='utf-8')
    return path


def load_model(path, *, device='cpu', interval=None):
    path = Path(path).resolve().with_suffix('.zip')
    metadata = json.loads(Path(str(path) + '.goalie.json').read_text(encoding='utf-8'))
    if metadata['schema'] != SCHEMA or metadata['sha256'] != hashlib.sha256(path.read_bytes()).hexdigest():
        raise ValueError('Goalie checkpoint schema or checksum mismatch')
    if interval is not None and metadata['interval'] != interval:
        raise ValueError('Goalie checkpoint control interval mismatch')
    model = DecisionPPO.load(path, device=device)
    if not isinstance(model.policy, GoaliePolicy):
        raise ValueError('Checkpoint does not contain a goalie policy')
    model.goalie_metadata = metadata
    return model

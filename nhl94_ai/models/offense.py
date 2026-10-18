"""Finite-history GRU policy with mechanical action/ownership masks.

On Classic-owned ticks only coast is allowed. Its log probability and entropy
are identically zero: PPO gets no policy gradient for Classic's actions, while
the critic still trains on all transitions and rewards.
"""
import hashlib
import json
from pathlib import Path

import torch
from torch import nn
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.distributions import CategoricalDistribution
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from nhl94_ai.agents.learned_offense import N_ACTIONS, SCHEMA, ENTITY_FIELDS
from nhl94_ai.models.temporal import GRUMlpExtractor


class CurrentOffenseExtractor(BaseFeaturesExtractor):
    """Current-observation MLP baseline with the same action/observation schema."""
    def __init__(self, observation_space, features_dim=256, hidden_dim=128):
        super().__init__(observation_space, features_dim)
        self.network = nn.Sequential(nn.Linear(observation_space.shape[-1], hidden_dim), nn.ReLU(),
                                     nn.Linear(hidden_dim, features_dim), nn.ReLU())

    def forward(self, observations):
        return self.network(observations[:, -1, :])


class ReceiverOrderedExtractor(GRUMlpExtractor):
    """Align carrier and receiver inputs with the receiver-specific commands.

    Raw observations and checkpoints keep their stable roster identities. Only
    this extractor's internal view places the current carrier before receivers
    1..4, preserving roster order among those receivers.
    """
    @staticmethod
    def ordered(observations):
        width = len(ENTITY_FIELDS)
        players = observations[..., :5 * width].reshape(*observations.shape[:-1], 5, width)
        owner = players[..., ENTITY_FIELDS.index('has_puck')]
        controlled = players[..., ENTITY_FIELDS.index('controlled')]
        carrier = (2 * owner + controlled).argmax(dim=-1)
        slots = torch.arange(5, device=observations.device)
        priority = slots.expand_as(owner) - (slots == carrier.unsqueeze(-1)) * 5
        order = priority.argsort(dim=-1)
        ordered = players.gather(-2, order.unsqueeze(-1).expand(*order.shape, width))
        return torch.cat((ordered.flatten(-2), observations[..., 5 * width:]), dim=-1)

    def forward(self, observations):
        return super().forward(self.ordered(observations))


class OffensePolicy(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault('features_extractor_class', GRUMlpExtractor)
        kwargs.setdefault('features_extractor_kwargs', dict(
            features_dim=256, hidden_dim=128, spatial_hidden_dim=128, gru_layers=1))
        kwargs.setdefault('net_arch', dict(pi=[128], vf=[128]))
        super().__init__(*args, **kwargs)

    def _distribution(self, obs, latent):
        mask = obs[:, -1, -N_ACTIONS:] > 0.5
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


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    temporary.replace(path)


def model_path(path):
    path = Path(path).expanduser().resolve()
    return path if path.suffix == '.zip' else path.with_suffix('.zip')


def save_offense_model(model, path, **details):
    path = model_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + '.tmp.zip')
    model.save(temporary)
    temporary.replace(path)
    write_json(str(path) + '.offense.json', dict(
        schema=SCHEMA, sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        timesteps=model.num_timesteps, gamma=model.gamma,
        parameters=sum(parameter.numel() for parameter in model.policy.parameters()),
        algorithm=type(model).__name__, **details))
    return str(path)


def load_offense_model(path, env=None, **kwargs):
    path = model_path(path)
    metadata = json.loads(Path(str(path) + '.offense.json').read_text(encoding="utf-8"))
    if metadata['schema'] != SCHEMA:
        raise ValueError('Offense checkpoint observation/action/timing schema mismatch')
    if metadata['sha256'] != hashlib.sha256(path.read_bytes()).hexdigest():
        raise ValueError('Offense checkpoint hash mismatch')
    expected_task = kwargs.pop('expected_task', None)
    allow_transfer = kwargs.pop('allow_task_transfer', False)
    saved_task = metadata.get('training_contract', {}).get('task', 'period')
    if expected_task is not None and saved_task != expected_task and not allow_transfer:
        raise ValueError(f'Checkpoint task is {saved_task}, requested {expected_task}; explicitly allow task transfer for this comparison')
    expected_scope = kwargs.pop('expected_control_scope', None)
    allow_scope_transfer = kwargs.pop('allow_control_transfer', False)
    saved_scope = metadata.get('training_contract', {}).get('control_scope', 'offense')
    if expected_scope is not None and saved_scope != expected_scope and not allow_scope_transfer:
        raise ValueError(f'Checkpoint control scope is {saved_scope}, requested {expected_scope}; explicitly allow control transfer')
    from nhl94_ai.training.decision_ppo import DecisionPPO
    decision_only = kwargs.pop('decision_only', metadata.get('algorithm') == 'DecisionPPO')
    algorithm = DecisionPPO if decision_only else PPO
    model = algorithm.load(path, env=env, device=kwargs.pop('device', 'cpu'), **kwargs)
    if not isinstance(model.policy, OffensePolicy):
        raise ValueError('Checkpoint does not contain an offense policy')
    model.offense_control_scope = saved_scope
    return model

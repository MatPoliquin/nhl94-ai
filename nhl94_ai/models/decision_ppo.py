"""Inference compatibility for checkpoints produced by the retired DecisionPPO.

SB3 restores the saved rollout-buffer class even when loading only for prediction.
Keep both class names, but reject learning so ordinary PPO cannot silently replace
historical duration-aware training semantics.
"""
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.buffers import RolloutBuffer


class DecisionRolloutBuffer(RolloutBuffer):
    """Legacy serialized class name; no duration-aware training implementation."""


class DecisionPPO(PPO):
    """Load and predict with existing decision-policy checkpoints."""

    def __init__(self, *args, **kwargs):
        kwargs['rollout_buffer_class'] = DecisionRolloutBuffer
        super().__init__(*args, **kwargs)
        if hasattr(self, 'action_space') and (not isinstance(self.action_space, spaces.Discrete) or self.use_sde):
            raise ValueError('DecisionPPO requires discrete actions without SDE')

    def _setup_model(self):
        self.rollout_buffer_class = DecisionRolloutBuffer
        super()._setup_model()

    def learn(self, *args, **kwargs):
        from nhl94_ai.training.unavailable import unavailable
        unavailable()

    def train(self):
        from nhl94_ai.training.unavailable import unavailable
        unavailable()

    def collect_rollouts(self, *args, **kwargs):
        from nhl94_ai.training.unavailable import unavailable
        unavailable()

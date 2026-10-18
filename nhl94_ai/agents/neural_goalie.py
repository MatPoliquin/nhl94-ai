"""Optional neural inference; ordinary Classic imports remain torch-free."""
import math

import torch

from nhl94_ai.agents.learned_goalie import CommandGoalie, decision_available, observation
from nhl94_ai.models.goalie import load_model


class NeuralGoalie(CommandGoalie):
    """Deploy the trained positioning policy inside Classic's goalie lifecycle."""

    def __init__(self, policy, model_path, *, interval=None, **kwargs):
        super().__init__(policy, **kwargs)
        # Small CPU inference batches are slower with a large BLAS thread pool.
        torch.set_num_threads(1)
        self.model = load_model(model_path, interval=interval)
        self.interval = self.model.goalie_metadata['interval']
        self.next_choice = 0

    def step(self, state, *, blocked=False):
        # Match training's observation boundary, before native lifecycle updates.
        available = decision_available(state, self)
        if available and not self.next_choice:
            self.next_choice = self.frames
        if available and self.frames > self.next_choice:
            self.next_choice += math.ceil((self.frames - self.next_choice) / self.interval) * self.interval
            self.command = 0
        if available and self.frames == self.next_choice:
            self.command = int(self.model.predict(observation(state, self), deterministic=True)[0])
            self.next_choice = self.frames + self.interval
        if not available:
            self.command = 0
        return super().step(state, blocked=blocked)

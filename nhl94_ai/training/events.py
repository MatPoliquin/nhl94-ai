from __future__ import annotations
from dataclasses import dataclass
from typing import Deque, Optional, Tuple
import threading

@dataclass
class LiveTrainingState:
    """Shared state between the training process and the display thread."""

    best_model_path: str
    reward_history: Deque[Tuple[int, float]]
    best_mean_reward: float = float("-inf")
    latest_eval_reward: Optional[float] = None
    latest_eval_timesteps: int = 0
    model_version: int = 0
    stop_requested: bool = False
    training_active: bool = True
    latest_model_path: Optional[str] = None
    latest_model_timesteps: int = 0

    def __post_init__(self) -> None:
        self.lock = threading.Lock()


@dataclass
class LiveTrainingResult:
    final_model_path: str
    best_model_path: str
    output_dir: str

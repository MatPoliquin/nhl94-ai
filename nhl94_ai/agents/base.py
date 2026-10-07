"""One gameplay interface for scripted and learned agents."""
from dataclasses import dataclass, field
from typing import Any, Protocol
import numpy as np
from nhl94_ai.agents.decisions import DecisionSnapshot, classic_decision_snapshot


@dataclass(frozen=True)
class AgentInput:
    game_state: Any
    observation: Any = None


@dataclass
class AgentOutput:
    action: np.ndarray
    diagnostics: dict = field(default_factory=dict)
    decision: DecisionSnapshot | None = None


class Agent(Protocol):
    input_schema: str
    action_schema: str

    def reset(self) -> None: ...
    def act(self, inputs: AgentInput, deterministic: bool = True) -> AgentOutput: ...


class ScriptedAgent:
    input_schema = 'game-state'

    def __init__(self, controller_class, args, env=None):
        self.controller_class = controller_class
        self.args, self.env = args, env
        self.action_schema = getattr(args, 'action_type', 'FILTERED')
        self.frame_skip = None
        self.reset()

    def reset(self):
        self.controller = self.controller_class(args=self.args, env=self.env)

    def act(self, inputs, deterministic=True):
        if self.frame_skip is None:
            action = self.controller.predict_game_state(inputs.game_state, deterministic)[0]
        else:
            action = self.controller.predict_frame(inputs.game_state, self.frame_skip, deterministic)[0]
        plan = getattr(self.controller, '_last_plan', None)
        self.last_output = AgentOutput(np.asarray(action), {'plan': getattr(plan, 'name', None),
            'decision': getattr(self.controller, '_last_decision', ''),
            'target': getattr(self.controller, '_last_target', (0, 0)),
            'classic_defense': getattr(self.controller, 'defense_diagnostics', {}),
            'classic_offense': getattr(self.controller, 'offense_diagnostics', {})})
        if getattr(self.controller, 'goalie', None) is not None:
            self.last_output.diagnostics['classic_goalie'] = self.controller.goalie_diagnostics
        if hasattr(self.controller, 'offense_diagnostics'):
            self.last_output.decision = classic_decision_snapshot(self.controller, inputs.game_state, action)
        return self.last_output

    # Existing integrations use these inference methods while callers migrate.
    def predict_game_state(self, state, deterministic=True):
        return np.asarray([self.act(AgentInput(state), deterministic).action])

    def predict(self, observation, deterministic=True):
        return self.controller.predict(observation, deterministic)

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return getattr(self.controller, name)

    def get_action_preferences(self, observation=None):
        return self.controller.get_action_preferences(observation)

    def act_frame(self, game_state, observation):
        return self.act(AgentInput(game_state, observation))


def configure_scripted_frames(agent, env, *, record=False, teacher_only=False):
    """Opt in to reactive substeps without changing learned-agent frame skipping."""
    if not isinstance(agent, ScriptedAgent):
        return None
    from nhl94_ai.env.wrappers import StochasticFrameSkip
    current = env
    while current is not None:
        if isinstance(current, StochasticFrameSkip):
            if current.stickprob > 0:
                raise ValueError('Reactive scripted agents require non-sticky actions')
            agent.frame_skip = current.n
            current.frame_agent = agent
            current.record_scripted_frames = record
            current.scripted_teacher_only = teacher_only
            return current.n
        current = getattr(current, 'env', None)
    agent.frame_skip = 1
    return 1


class LearnedAgent:
    input_schema = 'encoded-observation'

    def __init__(self, model, action_schema='FILTERED'):
        self.model, self.action_schema = model, action_schema
        self.reset()

    def reset(self):
        self.recurrent_state = None
        self.episode_start = True

    def act(self, inputs, deterministic=True):
        action, self.recurrent_state = self.model.predict(
            inputs.observation, state=self.recurrent_state,
            episode_start=np.asarray([self.episode_start]), deterministic=deterministic)
        self.episode_start = False
        return AgentOutput(np.asarray(action))


class FrameRepeatAgent:
    """Render/execute every frame while keeping the policy's decision interval."""

    def __init__(self, agent, frames):
        if int(frames) != frames or frames < 1:
            raise ValueError('Policy decision interval must be a positive integer')
        self.agent, self.frames = agent, int(frames)
        self.input_schema, self.action_schema = agent.input_schema, agent.action_schema
        self.reset()

    def reset(self):
        self.agent.reset()
        self.remaining = 0
        self.output = None

    def act(self, inputs, deterministic=True):
        if self.remaining == 0:
            self.output = self.agent.act(inputs, deterministic)
            self.remaining = self.frames
        self.remaining -= 1
        return AgentOutput(self.output.action.copy(), dict(self.output.diagnostics), self.output.decision)


class MultiModelAgent:
    input_schema = 'game-state-and-observation'
    action_schema = 'FILTERED'

    def __init__(self, system):
        self.system = system

    def reset(self):
        self.system.shooting = False
        self.system.pass_button_pressed = False
        self.system.scoregoal_state = None
        for model in self.system.models:
            if hasattr(model, 'reset'):
                model.reset()

    def act(self, inputs, deterministic=True):
        if getattr(getattr(self.system, 'args', None), 'action_type', 'FILTERED').upper() == 'TARGET_POSITION':
            raise ValueError('TARGET_POSITION uses a single learned agent, not the button-based multi-model adapter')
        from nhl94_ai.game.constants import GameConsts
        self.system.game_state = inputs.game_state
        if self.system.num_models == 1:
            action = self.system.Predict(self.system.model_in_use, inputs.observation, deterministic)
            if not hasattr(self.system.models[self.system.model_in_use], 'predict_game_state'):
                action[GameConsts.INPUT_MODE] = 0
        elif self.system.models[1] and self.system.models[2]:
            action = self.system.Think_TwoModels(inputs.observation, inputs.game_state, deterministic)
        else:
            action = np.zeros(GameConsts.INPUT_MAX, dtype=np.int8)
        return AgentOutput(np.asarray(action), {'model_index': self.system.model_in_use},
                           self.system.last_diagnostics.get('decision_inspector'))

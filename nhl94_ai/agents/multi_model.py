"""
NHL AI
"""
from nhl94_ai.agents.registry import CONTROLLERS

from copy import copy
import numpy as np
from nhl94_ai.game.constants import GameConsts
from nhl94_ai.game.state import NHL94GameState
from nhl94_ai.game.ram import controller_team_info
from nhl94_ai.models.factory import init_model, get_num_parameters, get_model_probabilities


MODEL_NONE = 0 #code
MODEL_DEFENSEZONE = 1
MODEL_SCOREOPP = 2

class NHL94AISystem():
    def __init__(self, args, env, logger):
        self.args = args
        self.logger = logger
        self.env = env

        self.scoregoal_state = None
        self.pass_button_pressed = False
        self.shooting = False

        from nhl94_ai.game.specs import get_game
        num_players_per_team = get_game(args.env).skaters_per_team

        self.game_state = NHL94GameState(num_players_per_team, platform=get_game(args.env).backend)

        self.models = [None, None, None]
        # Each slot is an independent controller stream, even for the same policy.
        self._learned_agents = {}
        self.model_params = [None, None, None]
        self.display_probs = (1,0,0,0,0,0,0,0,0,0,0,0,0)
        self.model_in_use = 0
        self.num_models = 0
        self.last_diagnostics = {}

    def SetModels(self, model_paths):
        if self.args.action_type.upper() == 'TARGET_POSITION':
            raise ValueError('TARGET_POSITION is not supported by the button-based multi-model system')
        self.models = [None, None, None]
        self.model_params = [None, None, None]
        self._learned_agents.clear()
        self.model_in_use = 0
        self.shooting = self.pass_button_pressed = False
        self.scoregoal_state = None
        self.last_diagnostics = {}
        i=1
        self.num_models = 0
        for p in model_paths:
            should_init = (p != '')
            if self.args.nn in CONTROLLERS and i == 1:
                should_init = True

            if should_init:
                self.models[i] = init_model(
                    None,
                    p,
                    self.args.alg,
                    self.args,
                    self.env,
                    self.logger,
                    getattr(self.args, "hyperparams_dict", None),
                )
                self.model_params[i] = get_num_parameters(self.models[i])
                self._prediction_agent(i)
                self.num_models += 1
                self.model_in_use = i
            i += 1

    def _prediction_agent(self, model_index):
        """Switching resumes a slot; replacing its model starts a fresh stream."""
        from nhl94_ai.agents.base import LearnedAgent
        model = self.models[model_index]
        if hasattr(model, 'act'):
            return model
        # Historical integrations may construct the system without __init__.
        if not hasattr(self, '_learned_agents'):
            self._learned_agents = {}
        cached = self._learned_agents.get(model_index)
        if cached is None or cached[0] is not model:
            cached = model, LearnedAgent(model, self.args.action_type)
            self._learned_agents[model_index] = cached
        return cached[1]

    def reset(self):
        """End the episode for every stream, including currently inactive slots."""
        self.shooting = self.pass_button_pressed = False
        self.scoregoal_state = None
        self.last_diagnostics = {}
        for index, model in enumerate(self.models):
            if model is None:
                continue
            agent = self._prediction_agent(index)
            agent.reset()
            if agent is not model and hasattr(model, 'reset'):
                model.reset()

    def GotoTarget(self, p1_actions, target_vec):
        if target_vec[0] > 0:
            p1_actions[GameConsts.INPUT_LEFT] = 1
        else:
            p1_actions[GameConsts.INPUT_RIGHT] = 1

        if target_vec[1] > 0:
            p1_actions[GameConsts.INPUT_DOWN] = 1
        else:
            p1_actions[GameConsts.INPUT_UP] = 1

    def Predict(self, model_index, model_input, deterministic):
        from nhl94_ai.agents.base import AgentInput
        agent = self._prediction_agent(model_index)
        game_state = self.game_state
        if getattr(self.args, 'side', 'home') == 'away':
            game_state = copy(game_state)
            game_state.Flip()
        result = agent.act(AgentInput(game_state, model_input), deterministic)
        p1_actions = result.action
        self.last_diagnostics = dict(result.diagnostics)
        if result.decision is not None:
            self.last_diagnostics['decision_inspector'] = result.decision

        p1_actions = np.asarray(p1_actions)
        if p1_actions.ndim > 1 and p1_actions.shape[0] == 1:
            p1_actions = p1_actions[0]

        self.display_probs = get_model_probabilities(self.models[model_index], model_input)[0]
        self.model_in_use = model_index

        return p1_actions

    def Think_TwoModels(self, model_input, state, deterministic):
        p1_actions = [0] * GameConsts.INPUT_MAX
        self.last_diagnostics = {}

        self.model_in_use = MODEL_NONE

        t1 = state.team1
        t2 = state.team2

        if t1.player_haspuck == True:
            # If in attack zone use ScoreGoal model
            # Otherwise go to attack zone
            if t1.players[0].y >= GameConsts.ATACKZONE_POS_Y:
                p1_actions = self.Predict(MODEL_SCOREOPP, model_input, deterministic)
                p1_actions[GameConsts.INPUT_C] = 0
                p1_actions[GameConsts.INPUT_B] = 0
                # check if we are in good scoring opportunity
                if t1.players[0].y < 230 and t1.players[0].y > 210:
                    if t1.players[0].vx >= 30 and t1.players[0].vx > -23 and state.puck.x < 0:
                        p1_actions[GameConsts.INPUT_C] = 1
                        self.shooting = True
                    elif t1.players[0].vx <= -30 and t1.players[0].vx < 23 and state.puck.x > 0:
                        p1_actions[GameConsts.INPUT_C] = 1
                        self.shooting = True
            else:
                self.GotoTarget(p1_actions, [t1.players[0].x - 0, t1.players[0].y - 99])

        elif t1.goalie_haspuck:
            p1_actions[GameConsts.INPUT_B] = 1

        else:
            self.shooting = False

            if t1.players[0].y < GameConsts.DEFENSEZONE_POS_Y and t2.player_haspuck:
                p1_actions = self.Predict(MODEL_DEFENSEZONE, model_input, deterministic)
            else:
                pp_vec = [t1.players[0].x - state.puck.x, t1.players[0].y - state.puck.y]
                self.GotoTarget(p1_actions, pp_vec)

        if self.shooting == True:
            p1_actions[GameConsts.INPUT_MODE] = 1
            p1_actions[GameConsts.INPUT_C] = 1

        return p1_actions

    def predict(self, state, info, deterministic):
        if info is None:
            p1_actions = [[0] * GameConsts.INPUT_MAX]
            return p1_actions

        frame_info = info[0]
        if getattr(self.args, 'side', 'home') == 'away':
            frame_info = controller_team_info(frame_info)
        self.game_state.BeginFrame(frame_info, [0] * 6)

        if self.num_models == 1:
            p1_actions = self.Predict(self.model_in_use, state, deterministic)
            if self.args.nn not in CONTROLLERS:
                p1_actions[GameConsts.INPUT_MODE] = 0
            p1_actions = [p1_actions]
        elif self.models[1] and self.models[2]:
            p1_actions = [self.Think_TwoModels(state, self.game_state, deterministic)]
        else:
            p1_actions = [[0] * GameConsts.INPUT_MAX]

        self.game_state.EndFrame()

        #self.display_probs = tuple(p1_actions[0])

        return p1_actions

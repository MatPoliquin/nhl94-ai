"""Recurrent histories belong to controller streams and end at playback resets."""
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from nhl94_ai.agents.base import AgentInput, AgentOutput, MultiModelAgent
from nhl94_ai.agents.multi_model import NHL94AISystem
from nhl94_ai.evaluation.play import NHL94Player


class StatefulPolicy:
    def __init__(self):
        self.calls = []

    def predict(self, observation, *, state, episode_start, deterministic):
        self.calls.append((np.asarray(observation).copy(), state, bool(episode_start[0]), deterministic))
        return np.zeros((1, 12), dtype=np.int8), 1 if state is None else state + 1

    def get_action_preferences(self, observation):
        return np.full((1, 12), 0.5)


class Finished(Exception):
    """Stop after the caller processes two terminal transitions."""


class AutoResetVector:
    def __init__(self, length=12):
        self.length = length
        self.frame = self.completed = self.resets = 0
        self.applied = []

    def observation(self):
        return np.asarray([[self.completed, self.frame]], dtype=np.float32)

    def reset(self):
        self.resets += 1
        self.frame = 0
        return self.observation()

    def step(self, action):
        self.applied.append(np.asarray(action).copy())
        self.frame += 1
        info = [{'episode': self.completed}]
        done = self.frame == self.length
        if done:
            self.completed += 1
            self.frame = 0
        return self.observation(), np.zeros(1), np.asarray([done]), info


def system():
    args = SimpleNamespace(env='NHL94-Genesis-v0', action_type='FILTERED', nn='MlpPolicy',
                           alg='ppo2', side='home', mode='model_vs_game', deterministic=True)
    result = NHL94AISystem(args, None, None)
    result.models[1] = StatefulPolicy()
    result.num_models, result.model_in_use = 1, 1
    # RAM decoding has its own native tests; these fixtures isolate stream history.
    result.game_state = SimpleNamespace(BeginFrame=Mock(), EndFrame=Mock())
    return result


class LearnedStreamTests(unittest.TestCase):
    def test_prediction_retains_state_and_episode_reset_clears_it(self):
        model = system()
        policy = model.models[1]
        for _ in range(2):
            action = model.Predict(1, np.zeros((1, 2)), True)
            self.assertEqual(action.shape, (12,))
        self.assertEqual([call[1:3] for call in policy.calls], [(None, True), (1, False)])
        np.testing.assert_array_equal(model.display_probs, np.full(12, 0.5))
        adapter = model._prediction_agent(1)
        model.reset()
        self.assertIs(model._prediction_agent(1), adapter)
        model.Predict(1, np.zeros((1, 2)), False)
        self.assertEqual(policy.calls[-1][1:], (None, True, False))

    def test_switching_resumes_each_slot_and_independent_systems_do_not_share_history(self):
        model, other = system(), system()
        policy = model.models[1]
        model.models[2] = other.models[1] = policy
        for slot in (1, 2, 1, 2):
            model.Predict(slot, np.zeros((1, 2)), True)
        other.Predict(1, np.zeros((1, 2)), True)
        self.assertEqual([call[1:3] for call in policy.calls],
                         [(None, True), (None, True), (1, False), (1, False), (None, True)])
        model.reset()
        model.Predict(2, np.zeros((1, 2)), True)
        other.Predict(1, np.zeros((1, 2)), True)
        self.assertEqual([call[1:3] for call in policy.calls[-2:]], [(None, True), (1, False)])

    def test_model_replacement_invalidates_only_that_slots_adapter(self):
        model = system()
        model.Predict(1, np.zeros((1, 2)), True)
        old = model._prediction_agent(1)
        replacement = StatefulPolicy()
        model.models[1] = replacement
        model.Predict(1, np.zeros((1, 2)), True)
        self.assertIsNot(model._prediction_agent(1), old)
        self.assertEqual(replacement.calls[0][1:3], (None, True))

    def test_set_models_replaces_slots_and_removes_omitted_models(self):
        model = system()
        first, second, replacement = (StatefulPolicy() for _ in range(3))
        with patch('nhl94_ai.agents.multi_model.init_model', side_effect=[first, second, replacement]):
            model.SetModels(['first', 'second'])
            model.Predict(1, np.zeros((1, 2)), True)
            model.SetModels(['replacement', ''])
        self.assertEqual((model.num_models, model.model_in_use), (1, 1))
        self.assertIsNone(model.models[2])
        self.assertEqual(set(model._learned_agents), {1})
        model.Predict(1, np.zeros((1, 2)), True)
        self.assertEqual(replacement.calls[0][1:3], (None, True))

    def test_compatibility_predict_and_agent_act_share_the_retained_stream(self):
        model = system()
        policy = model.models[1]
        model.predict(np.zeros((1, 2)), [{}], True)
        agent = MultiModelAgent(model)
        output = agent.act(AgentInput(model.game_state, np.zeros((1, 2))))
        self.assertEqual(output.action.shape, (12,))
        self.assertEqual([call[1:3] for call in policy.calls], [(None, True), (1, False)])
        agent.reset()
        agent.act(AgentInput(model.game_state, np.zeros((1, 2))))
        self.assertEqual(policy.calls[-1][1:3], (None, True))

    def test_agent_models_keep_their_own_act_and_reset_contract(self):
        model = system()
        agent = Mock()
        agent.act.return_value = AgentOutput(np.zeros(12), {'evidence': 'kept'})
        agent.get_action_preferences.return_value = np.zeros((1, 12))
        model.models[1] = agent
        model.Predict(1, np.zeros((1, 2)), True)
        self.assertEqual(model.last_diagnostics, {'evidence': 'kept'})
        self.assertFalse(model._learned_agents)
        model.reset()
        agent.reset.assert_called_once()

    @staticmethod
    def player(model, vector):
        player = NHL94Player.__new__(NHL94Player)
        player.args, player.need_display = model.args, False
        player.ai_sys, player.display_env = model, vector

        def stop_after_two(_env=None):
            if vector.completed == 2:
                raise Finished

        player._wait_for_display = stop_after_two
        return player

    def test_playback_resets_learned_streams_for_explicit_and_vector_auto_resets(self):
        for explicit in (False, True):
            with self.subTest(explicit=explicit):
                model, vector = system(), AutoResetVector()
                player = self.player(model, vector)
                with self.assertRaises(Finished):
                    player.play_player_or_game_mode(continuous=True, need_reset=explicit)
                calls = model.models[1].calls
                self.assertEqual([call[1:3] for call in calls],
                                 [(None, True), (1, False), (None, True), (1, False)])
                self.assertEqual([int(call[0][0, 0]) for call in calls], [0, 0, 1, 1])
                self.assertEqual(vector.resets, 3 if explicit else 1)
                self.assertEqual(len(vector.applied), 24)
                self.assertIsNone(model._prediction_agent(1).recurrent_state)

    def test_learned_model_vs_model_playback_has_two_independent_episode_streams(self):
        for explicit in (False, True):
            with self.subTest(explicit=explicit):
                model, vector = system(), AutoResetVector(length=2)
                player = self.player(model, vector)
                player.play_env = vector
                # Sharing weights must not share the two controllers' recurrent state.
                policy = StatefulPolicy()
                player.p1_model = player.p2_model = policy
                with self.assertRaises(Finished):
                    player.play_model_vs_model(continuous=True, need_reset=explicit)
                self.assertEqual([call[1:3] for call in policy.calls],
                                 [(None, True), (None, True), (1, False), (1, False)] * 2)
                self.assertTrue(all(call[3] is False for call in policy.calls))
                self.assertEqual(vector.resets, 3 if explicit else 1)

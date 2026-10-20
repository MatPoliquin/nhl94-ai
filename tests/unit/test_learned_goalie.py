"""Goalie action ownership, native contact timing and learned-policy contracts."""
from copy import deepcopy
import pickle
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import gymnasium as gym
import numpy as np
import torch

from nhl94_ai.agents.classic_v1 import ClassicAIV1Model
from nhl94_ai.agents.goalie import GoalieController, GoaliePlan, moving_body_contact
from nhl94_ai.agents.learned_goalie import (
    N_ACTIONS, CommandGoalie, SampledGoalie, command_buttons, observation,
    decision_available,
)
from nhl94_ai.agents.registry import add_classic_arguments
from nhl94_ai.game.constants import GameConsts as Buttons
from nhl94_ai.env.goalie import GoalieEnv
from nhl94_ai.models.goalie import GoaliePolicy, load_model, save_model
from nhl94_ai.models.decision_ppo import DecisionPPO
from tests.unit.test_manual_goalie import goalie_state


class ContactTimingTests(unittest.TestCase):
    def test_motion_can_reach_body_contact_before_center_alignment(self):
        state = goalie_state(True)
        goalie = state.team1.goalie
        goalie.x, goalie.y = -6, -238
        goalie.motion_x, goalie.motion_y = -1.5255279541015625, 0.157196044921875
        state.puck.x, state.puck.y, state.puck.height = -26, -204, 7.1614990234375
        state.puck.motion_x, state.puck.motion_y, state.puck.motion_z = 0.41815185546875, -4.6834564208984375, 0.3880615234375
        state.engine.puck_owner = -249
        state.engine.word_ffc2f8 = 0x0200
        plan = GoaliePlan((-22.96438659651066, -238), 'counterexample',
                          7.259595679867073, -22.96438659651066, 8, 12.8)
        before = pickle.dumps(state)
        with patch('nhl94_ai.agents.goalie.goalie_arrival', return_value=7.024311090071476):
            classic = GoalieController('selective')._position(state, plan)
            candidate = GoalieController('selective', save_timing='contact')._position(state, plan)
        self.assertTrue(classic[Buttons.INPUT_C])
        self.assertTrue(candidate[Buttons.INPUT_LEFT])
        self.assertFalse(candidate[Buttons.INPUT_C])
        self.assertLess(moving_body_contact(state, candidate, plan.crossing_frames), plan.crossing_frames)
        self.assertEqual(pickle.dumps(state), before)
        # A high contact needs the save animation even with favorable movement.
        high = GoaliePlan(plan.target, plan.reason, plan.crossing_frames, plan.crossing_x, 16, plan.deadline)
        with patch('nhl94_ai.agents.goalie.goalie_arrival', return_value=7.024311090071476):
            self.assertTrue(GoalieController('selective', save_timing='contact')._position(state, high)[Buttons.INPUT_C])

    def test_missing_motion_attributes_cannot_justify_delaying_save(self):
        state = goalie_state(True)
        state.team1.goalie.energy = None
        self.assertIsNone(moving_body_contact(state, command_buttons(3, state), 8))

    def test_keep_save_when_its_slower_motion_already_covers_the_shot(self):
        state = goalie_state(True)
        goalie = state.team1.goalie
        goalie.x, goalie.y = -14, -246
        goalie.motion_x, goalie.motion_y = 0.2202301025390625, 0
        state.puck.x, state.puck.y, state.puck.height = -30, -216, 3.540802001953125
        state.puck.motion_x, state.puck.motion_y = 2.8118896484375, -2.97686767578125
        state.puck.motion_z = 0.32684326171875
        state.engine.puck_owner, state.engine.word_ffc2f8 = -249, 0x0200
        plan = GoaliePlan((-1.66, -246), 'native regression', 10.0777, -1.66, 4, 16)
        with patch('nhl94_ai.agents.goalie.goalie_arrival', return_value=20):
            candidate = GoalieController('selective', save_timing='contact')._position(state, plan)
        self.assertTrue(candidate[Buttons.INPUT_C])


class LearnedGoalieTests(unittest.TestCase):
    def test_native_one_timer_input_uses_the_rom_byte_without_changing_legacy_field(self):
        state, manager = goalie_state(True), CommandGoalie()
        state.engine.one_timer_collision_mode = 0.25
        state.engine.word_ffc2f8 = 0x0002
        self.assertFalse(state.engine.native_one_timer_shot)
        self.assertEqual(observation(state, manager)[276], 0)
        state.engine.word_ffc2f8 = 0x0200
        self.assertTrue(state.engine.native_one_timer_shot)
        self.assertEqual(observation(state, manager)[276], 1)
        self.assertEqual(state.engine.one_timer_collision_mode, 0.25)

    def test_returning_to_skater_does_not_offer_a_goalie_choice(self):
        state, manager = goalie_state(True), CommandGoalie()
        state.engine.puck_owner = -256
        manager.phase = 'positioning'
        state.puck.y = 0
        state.puck.motion_x = state.puck.motion_y = 0
        self.assertFalse(decision_available(state, manager))
        self.assertEqual(observation(state, manager)[-N_ACTIONS:].sum(), 1)
        manager.step(state)
        self.assertEqual(manager.commands_used, 0)

    def test_goal_reward_precedes_possession_and_requires_confirmed_recovery(self):
        env = GoalieEnv.__new__(GoalieEnv)
        env.view = goalie_state(True)
        env.initial_goals = env.view.team2.stats.score
        env.frames, env.horizon = 10, 180
        env.owner_run, env.previous_owner = 0, None
        env.view.engine.puck_owner = env.view.team1.defense_goalie
        for _ in range(3):
            self.assertEqual(env._ending(), (None, 0.0))
        self.assertEqual(env._ending(), ('manual-catch', 1.0))
        env.view.team2.stats.score += 1
        self.assertEqual(env._ending(), ('goal', -1.0))
        env.view.team2.stats.score = env.initial_goals
        env.view.engine.puck_owner = -256
        env.frames = env.horizon
        self.assertEqual(env._ending(), ('timeout', 0.0))
        env.view.engine.clock_stopped = True
        self.assertEqual(env._ending(), ('stoppage', 0.0))

    def test_checkpoint_load_preserves_actions_and_rejects_wrong_cadence_or_bytes(self):
        torch.set_num_threads(1)
        state = goalie_state(True)
        state.engine.puck_owner = -256
        manager = CommandGoalie()
        manager.phase = 'positioning'
        obs = observation(state, manager)
        env = gym.Env()
        env.observation_space = gym.spaces.Box(-1, 1, obs.shape, np.float32)
        env.action_space = gym.spaces.Discrete(N_ACTIONS)
        model = DecisionPPO(GoaliePolicy, env, n_steps=2, batch_size=2, seed=7, device='cpu')
        with tempfile.TemporaryDirectory() as directory:
            path = save_model(model, Path(directory) / 'goalie.zip', interval=4)
            loaded = load_model(path, interval=4)
            np.testing.assert_equal(model.predict(obs, deterministic=True)[0],
                                    loaded.predict(obs, deterministic=True)[0])
            with self.assertRaisesRegex(ValueError, 'interval'):
                load_model(path, interval=10)
            with path.open('ab') as stream:
                stream.write(b'changed')
            with self.assertRaisesRegex(ValueError, 'checksum'):
                load_model(path)

    def test_commands_rotate_and_never_request_a_handoff(self):
        state = goalie_state(True)
        other = deepcopy(state)
        other.team1.net.y = -state.team1.net.y
        for command in range(N_ACTIONS):
            first, second = command_buttons(command, state), command_buttons(command, other)
            self.assertFalse(first[Buttons.INPUT_B])
            self.assertEqual(first[Buttons.INPUT_RIGHT], second[Buttons.INPUT_LEFT])
            self.assertEqual(first[Buttons.INPUT_UP], second[Buttons.INPUT_DOWN])
            self.assertEqual(first[Buttons.INPUT_C], second[Buttons.INPUT_C])
            self.assertFalse(first[Buttons.INPUT_A] and first[Buttons.INPUT_C])

    def test_observations_and_masks_follow_actual_control_and_animation(self):
        state = goalie_state(True)
        state.engine.puck_owner = -256
        manager = CommandGoalie()
        manager.phase = 'positioning'
        obs = observation(state, manager)
        self.assertEqual(obs.shape, (292,))
        self.assertTrue(np.isfinite(obs).all())
        self.assertTrue((np.abs(obs) <= 1).all())
        self.assertEqual(obs[-N_ACTIONS:].sum(), N_ACTIONS)
        manager.save_at = 40
        self.assertEqual(observation(state, manager)[-N_ACTIONS:].sum(), 9)
        for name, value in (('unavailable', 2), ('live_state_flags', 4)):
            locked = deepcopy(state)
            setattr(locked.team1.goalie, name, value)
            self.assertEqual(observation(locked, manager)[-N_ACTIONS:].sum(), 1)
            self.assertFalse(manager.step(locked).any())

    def test_sampled_policy_does_not_reconsider_between_decision_ticks(self):
        state = goalie_state(True)
        manager = SampledGoalie(interval=4)
        plan = GoaliePlan((20, -246), 'position')
        manager.frames = 1
        first = manager._position(state, plan)
        self.assertTrue(first[Buttons.INPUT_RIGHT])
        manager.frames = 2
        opposite = GoaliePlan((-20, -246), 'position')
        np.testing.assert_array_equal(manager._position(state, opposite), first)
        # A lock spanning a scheduled choice cannot create a new off-grid choice.
        manager.frames = 7
        self.assertFalse(manager._position(state, opposite).any())
        manager.frames = 9
        self.assertTrue(manager._position(state, opposite)[Buttons.INPUT_LEFT])

    def test_model_masks_and_attention_gradients(self):
        torch.set_num_threads(1)
        state = goalie_state(True)
        state.engine.puck_owner = -256
        manager = CommandGoalie()
        manager.phase = 'positioning'
        sample = observation(state, manager)
        space = gym.spaces.Box(-1, 1, sample.shape, np.float32)
        for architecture in ('mlp', 'transformer'):
            torch.manual_seed(123)
            policy = GoaliePolicy(space, gym.spaces.Discrete(N_ACTIONS), lambda _: 0.001,
                                  features_extractor_kwargs={'architecture': architecture})
            obs = torch.tensor(np.stack((sample, sample)))
            obs[0, -N_ACTIONS:] = 0
            obs[0, -N_ACTIONS] = 1
            actions, values, log_probs = policy(obs)
            self.assertEqual(actions[0].item(), 0)
            self.assertEqual(log_probs[0].item(), 0)
            loss = values.square().mean() - log_probs[1]
            loss.backward()
            if architecture == 'transformer':
                for layer in policy.features_extractor.attention.layers:
                    gradient = layer.self_attn.in_proj_weight.grad
                    self.assertTrue(torch.isfinite(gradient).all())
                    self.assertGreater(gradient.abs().sum().item(), 0)
            policy.eval()
            expected = policy.get_distribution(obs).distribution.probs
            actual = policy.get_distribution(obs.flip(0)).distribution.probs.flip(0)
            torch.testing.assert_close(actual, expected)

    def test_goalie_options_require_enabled_manual_control(self):
        import argparse
        parser = add_classic_arguments(argparse.ArgumentParser())
        for options in (['--goalie-save-timing', 'contact'], ['--goalie-decision-interval', '4']):
            with self.assertRaisesRegex(ValueError, 'goalie-policy'):
                ClassicAIV1Model(parser.parse_args(options))

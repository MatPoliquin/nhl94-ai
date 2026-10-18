"""Fresh model sizing and checkpoint timing across actual optimizer updates."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
from stable_baselines3.common.vec_env import DummyVecEnv

from nhl94_ai.env.offense_decisions import DecisionEnv
from nhl94_ai.models.offense import load_offense_model
from nhl94_ai.training.offense import build_parser, new_model, run, OffenseTrainingCallback
from tests.unit.test_offense_decisions import TickEnv


class ScratchTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def args(self, *extra):
        return build_parser().parse_args(['train', '--output', '/tmp/not-written.zip', '--scratch',
            '--decision-only', '--architecture', 'mlp', '--hidden-dim', '16', '--features-dim', '16',
            '--head-dim', '16', '--rollout-steps', '2', '--batch-size', '2', *extra])

    def test_scratch_rejects_parent_and_imitation_before_creating_environment(self):
        for flags in [('--model', '/missing.zip'), ('--data', '/missing.npz')]:
            with self.assertRaisesRegex(ValueError, 'without --model or --data'):
                run(self.args(*flags))

    def test_random_initialization_is_seeded_and_width_changes_capacity(self):
        first = new_model(DecisionEnv(TickEnv()), self.args('--seed', '7'))
        same = new_model(DecisionEnv(TickEnv()), self.args('--seed', '7'))
        other = new_model(DecisionEnv(TickEnv()), self.args('--seed', '8'))
        larger = new_model(DecisionEnv(TickEnv()), self.args('--hidden-dim', '32'))
        self.assertEqual(first.num_timesteps, 0)
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(first.policy.parameters(), same.policy.parameters())))
        self.assertFalse(all(torch.equal(a, b) for a, b in zip(first.policy.parameters(), other.policy.parameters())))
        self.assertGreater(sum(p.numel() for p in larger.policy.parameters()),
                           sum(p.numel() for p in first.policy.parameters()))

    def test_initial_intermediate_and_final_checkpoints_have_updated_weights(self):
        model = new_model(DummyVecEnv([lambda: DecisionEnv(TickEnv())]), self.args())
        with tempfile.TemporaryDirectory() as directory:
            callback = OffenseTrainingCallback(directory, ['cpu'], every=0, max_snapshots=1,
                total_steps=4, checkpoint_directory=directory, checkpoint_every=2,
                checkpoint_details=dict(initialization='random', training_contract=dict(task='period')))
            model.learn(total_timesteps=4, callback=callback)
            self.assertEqual(len(callback.checkpoints), 3)
            loaded = [load_offense_model(path) for path in callback.checkpoints]
            metadata = [json.loads(Path(path + '.offense.json').read_text(encoding='utf-8'))
                        for path in callback.checkpoints]
            self.assertEqual([row['timesteps'] for row in metadata], [0, 2, 4])
            self.assertEqual([row['optimizer_updates'] for row in metadata], [0, 4, 8])
            self.assertFalse(all(torch.equal(a, b) for a, b in zip(loaded[0].policy.parameters(), loaded[1].policy.parameters())))
            self.assertTrue(all(torch.equal(a, b) for a, b in zip(model.policy.parameters(), loaded[2].policy.parameters())))
            observation, _ = TickEnv().reset()
            np.testing.assert_equal(model.predict(observation, deterministic=True)[0],
                                    loaded[2].predict(observation, deterministic=True)[0])

"""Configuration, artifact, and agent behavior across application boundaries."""
import argparse
import importlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from gymnasium import spaces
import numpy as np
from stable_baselines3.common.policies import ActorCriticPolicy

from nhl94_ai.agents.base import AgentInput, LearnedAgent
from nhl94_ai.agents.registry import create_scripted
from nhl94_ai.artifacts import read_metadata, run_metadata, validate_schema
from nhl94_ai.cli import parse_configured
from nhl94_ai.config import (
    EnvironmentConfig, default_config_path, load_hyperparams,
    namespace_from_mapping, resolve_hyperparams_for_model,
)
from nhl94_ai.env.encoding import init_model
from nhl94_ai.env.target_control import CONTROLLER_FIELDS
from nhl94_ai.training.datasets import load_demo_arrays, save_demo_shard
from nhl94_ai.models.factory import MODEL_BUILDERS
from nhl94_ai.models.mlp import SquashedMlpPolicy


class PackageContracts(unittest.TestCase):
    def test_nhl94_training_example_matches_packaged_defaults(self):
        path = Path(__file__).resolve().parents[2] / 'configs/training/nhl94.json'
        self.assertEqual(load_hyperparams(str(path)), load_hyperparams(default_config_path('nhl94')))

    def test_nhl94_mlp_retry_configuration_builds_smaller_policy(self):
        config = load_hyperparams(default_config_path('nhl94'))
        params = resolve_hyperparams_for_model(config, 'MlpPolicy')
        self.assertEqual(params['frame_skip'], 4)
        self.assertEqual(params['net_arch'], {'pi': [128, 128], 'vf': [128, 128]})
        self.assertEqual(params['model_input'], config['model_input'])
        other = resolve_hyperparams_for_model(config, 'CustomMlpPolicy')
        self.assertEqual(other['net_arch'], {'pi': [512, 512, 256], 'vf': [512, 512, 256]})
        policy_name, kwargs = MODEL_BUILDERS['MlpPolicy'](512, params)
        self.assertEqual(policy_name, 'MlpPolicy')
        observation_size = init_model(5, params['model_input'])
        self.assertEqual(observation_size, 310)
        policy = ActorCriticPolicy(
            spaces.Box(-1, 1, shape=(observation_size,), dtype=np.float32),
            spaces.MultiBinary(12), lambda _: params['learning_rate'], **kwargs,
        )
        self.assertEqual(sum(parameter.numel() for parameter in policy.parameters()), 114317)

    def test_defense_target_example_uses_bounded_mlp_without_changing_shared_defaults(self):
        from nhl94_ai.training.live import build_parser, prepare_args
        path = Path(__file__).resolve().parents[2] / 'configs/training/defense-target.json'
        args = prepare_args(parse_configured(build_parser(), ['--config', str(path)]))
        EnvironmentConfig.from_args(args)
        self.assertEqual((args.nn, args.rf, args.action_type),
                         ('MlpPolicy', 'DefenseZone', 'TARGET_POSITION'))
        params = args.hyperparams_dict
        baseline = resolve_hyperparams_for_model(
            load_hyperparams(default_config_path('nhl94')), 'MlpPolicy',
        )
        self.assertEqual(params['net_arch'], {'pi': [256, 256], 'vf': [256, 256]})
        self.assertEqual(baseline['net_arch'], {'pi': [128, 128], 'vf': [128, 128]})
        target_options = ('net_arch', 'squash_output', 'log_std_init')
        self.assertEqual({key: value for key, value in params.items() if key not in target_options},
                         {key: value for key, value in baseline.items() if key not in target_options})
        self.assertIs(params['squash_output'], True)
        self.assertAlmostEqual(np.exp(params['log_std_init']), 0.5)
        self.assertNotIn('squash_output', baseline)
        self.assertNotIn('log_std_init', baseline)
        validate_schema(run_metadata(args, baseline), run_metadata(args, params))
        policy_name, kwargs = MODEL_BUILDERS[args.nn](args.nnsize, params)
        self.assertIs(policy_name, SquashedMlpPolicy)
        observation_size = init_model(5, params['model_input']) + len(CONTROLLER_FIELDS)
        self.assertEqual(observation_size, 322)
        action_space = spaces.Box(-1, 1, shape=(2,), dtype=np.float32)
        policy = policy_name(
            spaces.Box(-1, 1, shape=(observation_size,), dtype=np.float32),
            action_space, lambda _: params['learning_rate'], **kwargs,
        )
        for network in (policy.mlp_extractor.policy_net, policy.mlp_extractor.value_net):
            self.assertEqual([network[index].out_features for index in (0, 2)], [256, 256])
        self.assertEqual(sum(parameter.numel() for parameter in policy.parameters()), 297733)
        action, _ = policy.predict(np.zeros(observation_size, dtype=np.float32), deterministic=True)
        self.assertTrue(action_space.contains(action))

    def test_config_precedence_and_declaring_directory(self):
        with tempfile.TemporaryDirectory() as root:
            config = Path(root) / 'run.json'
            config.write_text(json.dumps({'num_env': 3, 'output': 'new-output'}), encoding='utf-8')
            parser = argparse.ArgumentParser()
            parser.add_argument('--num_env', type=int, default=1)
            parser.add_argument('--output', default='default')
            args = parse_configured(parser, ['--config', str(config), '--num_env', '5'])
            self.assertEqual(args.num_env, 5)
            self.assertEqual(args.output, str(Path(root) / 'new-output'))

    def test_configuration_rejects_boolean_strings(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--selfplay', action='store_true')
        with self.assertRaisesRegex(TypeError, 'JSON boolean'):
            namespace_from_mapping(parser, {'selfplay': 'false'})

    def test_invalid_combinations_fail_before_emulator_creation(self):
        for env in ('NHL941on1-Genesis-v0', 'NHL942on2-Genesis-v0'):
            with self.assertRaisesRegex(ValueError, 'full-team'):
                EnvironmentConfig(env=env, action_type='HOCKEY_INTENT_DPAD').validate()
        with self.assertRaisesRegex(ValueError, 'structured'):
            EnvironmentConfig(nn='CnnPolicy', selfplay=True).validate()
        with self.assertRaisesRegex(ValueError, 'controller count'):
            EnvironmentConfig(num_players=5).validate()

    def test_equal_shapes_with_different_field_order_are_incompatible(self):
        args = SimpleNamespace(env='NHL94-Genesis-v0', nn='MlpPolicy')
        first = run_metadata(args, {'model_input': {'schema_version': 2, 'groups': {'puck': ['x', 'y']}}})
        second = run_metadata(args, {'model_input': {'schema_version': 2, 'groups': {'puck': ['y', 'x']}}})
        with self.assertRaisesRegex(ValueError, 'observation_fields'):
            validate_schema(first, second)
        with tempfile.TemporaryDirectory() as root:
            path = str(Path(root) / 'shard.npz')
            save_demo_shard(path, {'observations': np.zeros((2, 2)), 'actions': np.zeros((2, 12))}, first)
            with self.assertRaisesRegex(ValueError, 'observation_fields'):
                load_demo_arrays([path], expected=second)

    def test_legacy_artifacts_are_explicitly_unknown(self):
        with tempfile.TemporaryDirectory() as root:
            artifact = Path(root) / 'legacy.zip'
            self.assertEqual(read_metadata(artifact)['compatibility'], 'unknown')
            Path(str(artifact) + '.json').write_text('{"env": "old"}', encoding='utf-8')
            self.assertEqual(read_metadata(artifact)['compatibility'], 'unknown')

    def test_reactive_demo_sample_timing_is_explicit_without_relaxing_checkpoints(self):
        args = SimpleNamespace(env='NHL94-Genesis-v0', nn='MlpPolicy')
        expected = run_metadata(args, {'frame_skip': 4})
        actual = run_metadata(args, {'frame_skip': 1})
        actual.update(sample_interval_frames=1, teacher_decision_interval=4)
        arrays = {'observations': np.zeros((2, 3)), 'actions': np.zeros((2, 12))}
        with tempfile.TemporaryDirectory() as root:
            path = str(Path(root) / 'frames.npz')
            save_demo_shard(path, arrays, actual)
            self.assertEqual(load_demo_arrays([path], expected=expected)['actions'].shape, (2, 12))
            with self.assertRaisesRegex(ValueError, 'frame_skip'):
                validate_schema(actual, expected)
            actual['teacher_decision_interval'] = 10
            save_demo_shard(path, arrays, actual)
            with self.assertRaisesRegex(ValueError, 'frame_skip'):
                load_demo_arrays([path], expected=expected)

    def test_learned_agent_resets_episode_state(self):
        calls = []
        class Model:
            def predict(self, observation, **kwargs):
                calls.append(kwargs)
                return np.zeros(12), 'hidden'
        agent = LearnedAgent(Model())
        agent.act(AgentInput(None, np.zeros(3)))
        agent.act(AgentInput(None, np.zeros(3)))
        agent.reset()
        agent.act(AgentInput(None, np.zeros(3)))
        self.assertEqual([bool(call['episode_start'][0]) for call in calls], [True, False, True])
        self.assertEqual([call['state'] for call in calls], [None, 'hidden', None])

    def test_scripted_agent_reset_clears_cooldowns(self):
        args = SimpleNamespace(action_type='FILTERED')
        agent = create_scripted('classic-v1', args)
        agent.controller._pass_at = 100
        agent.reset()
        self.assertEqual(agent.controller._pass_at, 0)

    def test_neural_registry_constructs_matching_policy_names(self):
        self.assertIn('GRUMlpPolicy', MODEL_BUILDERS)
        policy, kwargs = MODEL_BUILDERS['ResidualMlpPolicy'](16, {'residual_mlp': {'hidden_dim': 16}})
        self.assertEqual(policy.__name__, 'ResidualMlpPolicy')
        self.assertEqual(kwargs['residual_mlp']['hidden_dim'], 16)

    def test_checkpoint_aliases_do_not_import_retired_agents(self):
        command = '''import sys
from nhl94_ai.compat import install_checkpoint_aliases
install_checkpoint_aliases()
assert sys.modules['models'].__name__ == 'nhl94_ai.models.networks'
assert sys.modules['es'].__name__ == 'nhl94_ai.training.es'
for name in ('classic_ai', 'classic_ai_v2', 'classic_ai_v3'):
    assert name not in sys.modules, name
'''
        subprocess.run([sys.executable, '-B', '-c', command], check=True)

    def test_headless_imports_do_not_import_display_or_export(self):
        command = '''import sys
import nhl94_ai.training.live
import nhl94_ai.training.curriculum
import nhl94_ai.evaluation.runner
import nhl94_ai.export
assert 'pygame' not in sys.modules
assert 'onnx' not in sys.modules
assert 'onnxruntime' not in sys.modules
assert 'timm' not in sys.modules
'''
        subprocess.run([sys.executable, '-B', '-c', command], cwd=tempfile.gettempdir(), check=True)


if __name__ == '__main__':
    unittest.main()

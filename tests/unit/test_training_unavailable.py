"""Retired workflows fail explicitly without starting runs or writing artifacts."""
import contextlib
import importlib
import io
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

from nhl94_ai.cli import main
from nhl94_ai.training.unavailable import TrainingUnavailableError


class TrainingUnavailableTests(unittest.TestCase):
    def test_cli_workflows_fail_before_reading_configs_or_creating_outputs(self):
        commands = [('train',), ('collect',), ('bc',), ('dagger',), ('curriculum',), ('compare',),
                    ('offense', 'train'), ('offense', 'bc'), ('offense', 'collect'), ('goalie', 'train')]
        with tempfile.TemporaryDirectory() as directory:
            for command in commands:
                with self.subTest(command=command), contextlib.redirect_stderr(io.StringIO()) as error:
                    result = main([*command, '--config', 'missing.json', '--output', directory + '/run'])
                    self.assertEqual(result, 2)
                    self.assertIn('Training is unavailable', error.getvalue())
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_python_entrypoints_raise_not_implemented_before_using_arguments(self):
        entries = [('service', 'train'), ('live', 'run_training_session'), ('rl', 'ModelTrainer'),
                   ('collect', 'collect_demonstrations'), ('bc', 'train_bc'), ('dagger', 'run_dagger'),
                   ('curriculum', 'run_curriculum'), ('offense', 'train'), ('goalie', 'train'),
                   ('shot_placement', 'fit_logistic'), ('shot_direction', 'fit'),
                   ('retention', 'train'), ('retention_ranking', 'train'),
                   ('chance_selection', 'fit'), ('one_timer_admission', 'fit')]
        for module, name in entries:
            with self.subTest(module=module, name=name):
                function = getattr(importlib.import_module('nhl94_ai.training.' + module), name)
                with self.assertRaisesRegex(NotImplementedError, 'Training is unavailable'):
                    function(None)

    def test_python_module_exit_status_is_nonzero_without_traceback(self):
        for module, arguments in [('nhl94_ai', ['train']), ('nhl94_ai.training.bc', []),
                                  ('nhl94_ai.training.offense', ['train'])]:
            with self.subTest(module=module):
                result = subprocess.run([sys.executable, '-m', module, *arguments],
                                        capture_output=True, text=True, check=False)
                self.assertEqual(result.returncode, 2)
                self.assertIn('Training is unavailable', result.stderr)
                self.assertNotIn('Traceback', result.stderr)

    def test_disabled_command_help_explains_status(self):
        with contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(main(['train', '--help']), 0)
        self.assertIn('Training is unavailable', output.getvalue())

    def test_existing_policy_commands_dispatch_to_evaluation(self):
        for command, arguments, module in [
            ('offense', ['evaluate', '--output', 'report.json'], 'offense_policy'),
            ('offense', ['play', '--output', 'report.json'], 'offense_policy'),
            ('offense', ['audit', '--output', 'report.json'], 'offense_policy'),
            ('goalie', ['evaluate', '--banks', 'bank.json', '--output', 'report.json'], 'goalie_policy'),
        ]:
            with self.subTest(command=command, arguments=arguments), \
                    patch('nhl94_ai.evaluation.' + module + '.run') as run:
                self.assertEqual(main([command, *arguments]), 0)
                run.assert_called_once()

    def test_legacy_retention_evaluation_command_still_routes_to_evaluation(self):
        for module in ('retention', 'retention_ranking', 'shot_placement'):
            stub = importlib.import_module('nhl94_ai.training.' + module)
            with self.subTest(module=module), patch('nhl94_ai.evaluation.' + module + '_model.main') as evaluate:
                stub.main(['evaluate', '--model', 'existing.json'])
                evaluate.assert_called_once_with(['--model', 'existing.json'])

    def test_offense_evaluation_runs_and_writes_report_without_training(self):
        import json
        from nhl94_ai.evaluation import offense_policy
        env = Mock()
        env.reset.return_value = (None, {})
        env.step.return_value = (None, 1.0, True, False, {
            'opponent': 'cpu', 'completed': True, 'goals_for': 1, 'goals_against': 0,
        })
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'report.json'
            args = offense_policy.build_parser().parse_args([
                'evaluate', '--output', str(output), '--episodes', '1', '--agents', 'classic',
            ])
            with patch.object(offense_policy, 'OffenseEnv', return_value=env):
                offense_policy.run(args)
            report = json.loads(output.read_text())
            self.assertEqual(report['summary']['classic:cpu']['wins'], 1)
            env.close.assert_called_once()

    def test_goalie_evaluation_runs_and_writes_report_without_training(self):
        import json
        from nhl94_ai.evaluation import goalie_policy
        env = Mock()
        env.reset.return_value = (None, {})
        env.step.return_value = (None, 1.0, True, False, {})
        env.result.return_value = {'start': 'one', 'goal': False, 'outcome': 'save', 'frames': 4}
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'report.json'
            args = goalie_policy.build_parser().parse_args([
                'evaluate', '--banks', 'bank.json', '--output', str(output), '--workers', '1',
            ])
            with patch.object(goalie_policy, 'GoalieEnv', return_value=env), \
                    patch.object(goalie_policy, 'bank_rows', return_value=[{}]), \
                    patch.object(goalie_policy, 'source_hashes', return_value={}):
                goalie_policy.run(args)
            report = json.loads(output.read_text())
            self.assertEqual(report['summary']['attempts'], 1)
            self.assertEqual(report['summary']['goals_against'], 0)
            env.close.assert_called_once()

    def test_es_is_not_loadable_or_aliased(self):
        from nhl94_ai.compat import install_checkpoint_aliases
        from nhl94_ai.models.factory import init_model
        install_checkpoint_aliases()
        self.assertNotIn('es', sys.modules)
        self.assertIsNone(importlib.util.find_spec('nhl94_ai.training.es'))
        with self.assertRaisesRegex(ValueError, 'Unsupported algorithm'):
            init_model(None, None, 'es', None, None, None, None)

    def test_legacy_decision_checkpoint_predicts_but_cannot_resume_learning(self):
        import numpy as np
        from nhl94_ai.models.decision_ppo import DecisionPPO, DecisionRolloutBuffer
        from nhl94_ai.models.offense import OffensePolicy, save_offense_model, load_offense_model
        from tests.support.learned_offense import TickEnv
        env = TickEnv()
        model = DecisionPPO(OffensePolicy, env, n_steps=2, batch_size=2, device='cpu')
        observation, _ = env.reset()
        expected = model.predict(observation, deterministic=True)[0]
        with tempfile.TemporaryDirectory() as directory:
            # Serialize the buffer's original import path, as historical archives did.
            with patch.object(DecisionRolloutBuffer, '__module__', 'nhl94_ai.training.decision_ppo'):
                path = save_offense_model(model, Path(directory) / 'legacy')
            loaded = load_offense_model(path)
            np.testing.assert_array_equal(loaded.predict(observation, deterministic=True)[0], expected)
            with self.assertRaises(TrainingUnavailableError):
                loaded.learn(2)

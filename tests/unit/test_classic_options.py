"""Caller defaults, compatibility, and immutable construction settings."""
from dataclasses import FrozenInstanceError
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from nhl94_ai.agents.classic_options import ClassicOptions, validate_classic_options
from nhl94_ai.agents.classic_v1 import ClassicAIV1Model
from nhl94_ai.config import EnvironmentConfig
from nhl94_ai.evaluation.cpu_benchmark import build_parser, run
from nhl94_ai.evaluation.play import parse_cmdline


class ClassicOptionsTests(unittest.TestCase):
    def test_normalization_does_not_alias_mutable_arguments(self):
        args = SimpleNamespace(classic_refinements=['pass-timing'], action_type='filtered')
        options = validate_classic_options(args)
        args.classic_refinements.append('finishing')
        self.assertEqual(options.classic_refinements, frozenset({'pass-timing'}))
        self.assertEqual(options.action_type, 'FILTERED')
        self.assertEqual(args.action_type, 'filtered')
        with self.assertRaises(FrozenInstanceError):
            options.goalie_policy = 'selective'

    def test_unspecified_explicit_and_disabled_goalies_remain_distinct(self):
        automatic = validate_classic_options(SimpleNamespace(goalie_policy='selective'))
        explicit = validate_classic_options(SimpleNamespace(goalie_policy='selective', goalie_positioning='classic'))
        disabled = validate_classic_options(None)
        self.assertIsNone(automatic.goalie_positioning)
        self.assertEqual(automatic.effective_goalie_positioning, 'deep-csave')
        self.assertEqual(explicit.effective_goalie_positioning, 'classic')
        self.assertEqual(disabled.goalie_policy, 'off')
        self.assertIsNone(ClassicAIV1Model().goalie)
        self.assertIsNone(disabled.classic_decision_interval)
        self.assertEqual(disabled.classic_control_interval, 0)

    def test_legacy_direct_experiments_keep_their_explicit_caller_contract(self):
        for flag in ('offense-lookahead', 'uncertain-carry'):
            args = parse_cmdline(['--nn=ClassicAIV1', '--' + flag])
            with self.subTest(flag=flag):
                model = ClassicAIV1Model(args)
                self.assertTrue(getattr(model.options, flag.replace('-', '_')))
                with self.assertRaises(ValueError):
                    EnvironmentConfig.from_args(args)

    def test_conflicts_are_rejected_before_loading_or_running_at_every_boundary(self):
        flags = dict(goalie_policy='selective', goalie_positioning='deep', goalie_model='unloaded.zip')
        args = SimpleNamespace(env='NHL94-Genesis-v0', nn='ClassicAIV1', **flags)
        with patch('nhl94_ai.agents.classic_v1.create_goalie') as construction:
            for caller in (lambda: ClassicAIV1Model(args), lambda: EnvironmentConfig.from_args(args),
                           lambda: run(build_parser().parse_args([
                               '--goalie-policy', 'selective', '--goalie-positioning', 'deep',
                               '--goalie-model', 'unloaded.zip']))):
                with self.subTest(caller=caller), self.assertRaisesRegex(ValueError, 'Deep goalie positioning'):
                    caller()
            construction.assert_not_called()

    def test_direct_and_benchmark_valid_options_have_the_same_normalized_values(self):
        benchmark = build_parser().parse_args(['--goalie-policy', 'selective', '--goalie-positioning', 'classic',
                                               '--classic-refinements', 'pass-timing', '--goalie-decision-interval', '4'])
        direct = SimpleNamespace(**{name: getattr(benchmark, name) for name in ClassicOptions.__dataclass_fields__
                                    if hasattr(benchmark, name)})
        self.assertEqual(validate_classic_options(benchmark, benchmark=True), validate_classic_options(direct))

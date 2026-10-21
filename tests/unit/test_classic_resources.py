"""Episode state is fresh while immutable artifacts remain pinned to a stream."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import pickle
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from nhl94_ai.agents.base import AgentInput
from nhl94_ai.agents.classic_v1 import ClassicAIV1Model
from nhl94_ai.agents.learned_offense import PossessionController
from nhl94_ai.agents.registry import create_scripted
from nhl94_ai.agents.retention_ranker import load_retention_filter
from tests.support.offense import shooting_state
from tests.unit.test_retention_filter import artifact as tree_artifact
from tests.unit.test_retention_ranking import artifact as ranker_artifact


class ClassicResourceTests(unittest.TestCase):
    def test_retention_dispatch_reads_hashes_and_parses_once(self):
        with TemporaryDirectory() as directory:
            for data in (tree_artifact(), ranker_artifact()):
                path = Path(directory)/'retention.json'
                raw = json.dumps(data).encode()
                path.write_bytes(raw)
                with patch.object(Path, 'read_bytes', autospec=True, return_value=raw) as reads, \
                        patch('nhl94_ai.agents.classic_resources.json.loads', wraps=json.loads) as parses:
                    model = load_retention_filter(path)
                self.assertEqual(reads.call_count, 1)
                self.assertEqual(parses.call_count, 1)
                self.assertEqual(model.sha256, hashlib.sha256(raw).hexdigest())
                with self.assertRaises(TypeError):
                    model.artifact['interval'] = 8
                if hasattr(model, 'trees'):
                    with self.assertRaises(ValueError):
                        model.trees[0]['value'][0] = 1

    def test_resets_reuse_json_without_sharing_episode_metrics_or_mutable_history(self):
        with TemporaryDirectory() as directory:
            for data in (tree_artifact(), ranker_artifact()):
                path = Path(directory)/'retention.json'
                path.write_text(json.dumps(data))
                agent = create_scripted('classic-v1', SimpleNamespace(retention_model=str(path)))
                old = agent.controller
                old.retention_filter.metrics['overrides'] = 9
                old.scheduler.frames = old.offense.feint_at = 100
                old._predict_decision = lambda *_args: None
                with patch.object(Path, 'read_bytes', side_effect=AssertionError('reset reread artifact')):
                    agent.reset()
                self.assertIsNot(agent.controller, old)
                self.assertIs(agent.retention_filter.artifact, old.retention_filter.artifact)
                self.assertEqual((agent.scheduler.frames, agent.offense.feint_at), (0, 0))
                self.assertFalse(agent.retention_filter.metrics)
                self.assertEqual(old.retention_filter.metrics['overrides'], 9)
                self.assertIs(agent.controller._predict_decision.__func__, ClassicAIV1Model._predict_decision)
                restored = pickle.loads(pickle.dumps(agent.controller))
                self.assertEqual(restored.retention_filter.artifact, agent.retention_filter.artifact)

    def test_neural_weights_are_reused_but_opponents_and_commands_are_independent(self):
        models = [SimpleNamespace(goalie_metadata={'interval': 4}) for _ in range(2)]
        args = SimpleNamespace(goalie_policy='selective', goalie_model='weights.zip')
        with patch('nhl94_ai.models.goalie.load_model', side_effect=models) as loader:
            first = create_scripted('classic-v1', args)
            first.goalie.command, first.goalie.next_choice = 3, 400
            first.goalie.metrics['saves'] = 5
            first.reset()
            self.assertIs(first.goalie.model, models[0])
            self.assertEqual((first.goalie.command, first.goalie.next_choice), (0, 0))
            self.assertFalse(first.goalie.metrics)
            second = create_scripted('classic-v1', args)
            self.assertIsNot(first.goalie.model, second.goalie.model)
            self.assertEqual(loader.call_count, 2)
            branch = first.controller.fork()
            self.assertIs(branch.goalie.model, first.goalie.model)
            branch.goalie.command = 7
            branch.goalie.metrics['saves'] = 1
            self.assertEqual(first.goalie.command, 0)
            self.assertFalse(first.goalie.metrics)

    def test_shot_resource_is_pinned_until_explicit_path_replacement(self):
        with TemporaryDirectory() as directory:
            first, second = (Path(directory)/name for name in ('first.json', 'second.json'))
            first.write_text(json.dumps({'version':'native-shot-fixed-hold-v1', 'hold':8}))
            second.write_text(json.dumps({'version':'native-shot-fixed-hold-v1', 'hold':12}))
            args = SimpleNamespace(shot_placement=str(first))
            agent = create_scripted('classic-v1', args)
            artifact = agent.shot_placement.model
            first.write_text('changed file is not silently reloaded during an episode reset')
            agent.reset()
            self.assertIs(agent.shot_placement.model, artifact)
            args.shot_placement = str(second)
            agent.reset()
            self.assertEqual(agent.shot_placement.fixed_hold, 12)
            self.assertEqual(artifact['hold'], 8)
            self.assertEqual(deepcopy(artifact), artifact)

    def test_ordinary_reset_and_learned_reset_clear_all_owned_execution(self):
        state = shooting_state()
        agent = create_scripted('classic-v1', SimpleNamespace(classic_control_interval=4))
        agent.frame_skip = 4
        agent.act(AgentInput(state))
        self.assertIsNotNone(agent.last_output)
        agent.reset()
        self.assertEqual((agent.scheduler.frames, agent.control_rate.frames), (0, 0))
        self.assertFalse(agent.buttons.c_down)
        self.assertIsNone(agent.last_output)
        self.assertEqual(agent.frame_skip, 4)
        learned = PossessionController(mode='teacher')
        learned.teacher_command, learned.command_used, learned.command = 17, True, 4
        learned.scheduler.frames = 100
        learned.reset_episode()
        self.assertEqual(learned.scheduler.frames, 0)
        self.assertIsNone(learned.teacher_command)
        self.assertIsNone(learned.command)
        self.assertFalse(learned.command_used)

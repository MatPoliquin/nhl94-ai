"""Directional execution preserves the chosen shot and bounds outcome attribution."""
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import Mock, patch

import numpy as np

from nhl94_ai.agents.classic_v1 import ClassicAIV1Model
from nhl94_ai.evaluation.shot_direction import (
    DirectionalShotModel, DirectionSelector, MODEL_VERSION, PADS, aim_buttons, direction_features, placed,
)
from nhl94_ai.evaluation.shot_direction_replay import fork_direction
from nhl94_ai.evaluation.shot_direction_benchmark import run_job
from nhl94_ai.evaluation import cpu_benchmark
from nhl94_ai.training.shot_direction import evaluate, fit, read_cases, selected_index
from tests.unit.test_classic_offense import live_one_timer_state, offense_state
from tests.unit.test_possession_value import finish_state


class ShotDirectionTests(TestCase):
    def test_nine_distinct_pads_preserve_c_and_other_buttons(self):
        actions = []
        for pad in PADS:
            action = np.ones(12, dtype=np.int8)
            aim_buttons(action, pad)
            self.assertTrue(action[8])
            np.testing.assert_array_equal(action[:4], np.ones(4))
            np.testing.assert_array_equal(action[8:], np.ones(4))
            self.assertLessEqual(sum(action[4:6]), 1)
            self.assertLessEqual(sum(action[6:8]), 1)
            actions.append(tuple(action))
        self.assertEqual(len(set(actions)), 9)
        with self.assertRaises(ValueError):
            aim_buttons(np.zeros(12), (2, 0))

    def test_baseline_placement_preserves_lifecycle_and_input_history(self):
        state, model = finish_state(), ClassicAIV1Model()
        model.predict_frame(state, 4)
        saved = deepcopy(model)
        branch = placed(model, (model.shot.side, 0))
        for _ in range(12):
            np.testing.assert_array_equal(branch.predict_frame(state, 4), model.predict_frame(state, 4))
        self.assertEqual(asdict(branch.shot), asdict(model.shot))
        high = placed(saved, (1, 1))
        self.assertEqual(high.shot.side, saved.shot.side)
        self.assertEqual(high.buttons, saved.buttons)
        self.assertTrue(high.scheduler.action[0, 4] and high.scheduler.action[0, 7] and high.scheduler.action[0, 8])
        self.assertFalse(saved.scheduler.action[0, 4])

    def test_vertical_aim_preserves_hold_and_stops_on_possession_change(self):
        state, model = finish_state(), DirectionalShotModel()
        model.direction_selector = DirectionSelector({'version': MODEL_VERSION, 'kind': 'fixed-height', 'height': 1})
        actions = [model.predict_frame(state, 4)[0] for _ in range(7)]
        self.assertEqual([bool(a[8]) for a in actions], [True]*4 + [False]*3)
        self.assertTrue(all(a[4] for a in actions))
        state.engine.puck_owner = -1
        model.predict_frame(state, 4)
        self.assertIsNone(model.directional_aim)

    def test_pass_and_one_timer_choices_never_consult_normal_shot_selector(self):
        for factory in (offense_state, live_one_timer_state):
            state, old, new = factory(), ClassicAIV1Model(), DirectionalShotModel()
            new.direction_selector = Mock()
            np.testing.assert_array_equal(new.predict_frame(state), old.predict_frame(state))
            new.direction_selector.choose.assert_not_called()

    def test_same_screen_height_at_both_attacking_ends(self):
        selector = DirectionSelector({'version': MODEL_VERSION, 'kind': 'fixed-height', 'height': 1})
        state = finish_state()
        self.assertEqual(selector.choose(state), (-1, 1))
        state.team1.net.y, state.team2.net.y = state.team2.net.y, state.team1.net.y
        self.assertEqual(selector.choose(state), (-1, 1))

    def test_invalid_model_parameters_are_rejected(self):
        for value in (True, 2, .5):
            with self.assertRaises(ValueError):
                DirectionSelector({'version': MODEL_VERSION, 'kind': 'fixed-height', 'height': value})
        with self.assertRaises(ValueError):
            DirectionSelector({'version': 'unknown', 'kind': 'fixed-height', 'height': 1})

    def test_height_only_scope_cannot_change_horizontal_aim(self):
        model = {'version': MODEL_VERSION, 'kind': 'logistic', 'features': ['aim', 'height'],
                 'mean': [0, 0], 'scale': [1, 1], 'weights': [10, 1], 'bias': 0, 'min_gain': 0,
                 'scope': 'height-only'}
        with patch('nhl94_ai.evaluation.shot_direction.direction_features',
                   side_effect=lambda state, pad: dict(aim=pad[0], height=pad[1])):
            self.assertEqual(DirectionSelector(model).choose(finish_state()), (-1, 1))
            model['scope'] = 'nine'
            self.assertEqual(DirectionSelector(model).choose(finish_state()), (1, 1))

    def test_benchmark_activates_direction_policy_inside_scripted_wrapper(self):
        policy = {'version': MODEL_VERSION, 'kind': 'fixed-height', 'height': 1}

        def match(_fixture):
            agent = cpu_benchmark.create_scripted('classic-v1', SimpleNamespace(action_type='FILTERED'))
            agent.frame_skip = 4
            action = agent.predict_game_state(finish_state())[0]
            self.assertTrue(action[4] and action[8])
            self.assertEqual(agent.controller.direction_selector.metrics['height-overrides'], 1)
            return {'goals': [0, 0]}

        with patch('nhl94_ai.evaluation.shot_direction_benchmark.cpu_match', side_effect=match):
            result = run_job(('sabres-ducks-manual', 1, 'candidate', policy))
        self.assertEqual(result['direction_metrics']['height-overrides'], 1)

    def test_later_goal_cannot_relabel_an_unreleased_original_shot(self):
        state, model = finish_state(), ClassicAIV1Model()
        state.team1.one_timer_attempts = 0
        state.engine.clock_stopped = False
        model.predict_frame(state, 4)
        later = deepcopy(state)
        later.engine.shot_player = model.shot.slot
        later.team1.stats.shots += 1
        info = {'p1_score': 0, 'p2_score': 0, 'bench_clock': 100}

        def new_shot(*_args, **_kwargs):
            model.shot.hold_until_frame += 100
            return np.zeros((1, 12), dtype=np.int8)

        model.predict_frame = Mock(side_effect=new_shot)
        with (patch('nhl94_ai.evaluation.shot_direction_replay.deepcopy', side_effect=lambda item: item),
              patch('nhl94_ai.evaluation.shot_direction_replay.restore'),
              patch('nhl94_ai.evaluation.shot_direction_replay.advance'),
              patch('nhl94_ai.evaluation.shot_direction_replay.crossing_touch_player', return_value=-1),
              patch('nhl94_ai.evaluation.shot_direction_replay.read_view', side_effect=[
                  (state, info), (state, info), (later, {**info, 'p1_score': 1})])):
            result = fork_direction(Mock(), b'', '', state, 1, model, None, 3)
        self.assertEqual(result['goal'], 1)
        self.assertEqual(result['primary_goal'], 0)
        self.assertIsNone(result['release'])
        self.assertEqual(result['primary_end'], 'next-action-before-primary-release')


class ShotDirectionTrainingTests(TestCase):
    @staticmethod
    def case(seed, *, high_wins):
        pads = [(-1, 0), *(pad for pad in PADS if pad != (-1, 0))]
        return {'seed': seed, 'matchup': 'example', 'frame': 100, 'baseline': [-1, 0], 'branches': [
            {'pad': list(pad), 'features': {'height': pad[1]},
             'outcome': {'goal': int(pad[1] == (1 if high_wins else 0)),
                         'primary_goal': 0, 'goals_against': 0}} for pad in pads]}

    def test_fit_uses_attack_goals_and_retains_baseline_on_adverse_validation(self):
        training = [self.case(seed, high_wins=True) for seed in range(1, 5)]
        validation = [self.case(5, high_wins=False)]
        train_info, val_info = {'seeds': [1, 2, 3, 4]}, {'seeds': [5]}
        model, report = fit(training, validation, train_info, val_info)
        self.assertEqual(model['selected_name'], 'legacy')
        self.assertEqual(report['validation_performance']['totals']['selected_attack_goals'], 1)
        learned = next(row['model'] for row in report['alternatives'] if row['name'] == 'nine:0')
        self.assertGreater(learned['weights'][0], 0)
        totals = evaluate(learned, training)['totals']
        self.assertEqual(totals['selected_attack_goals'], 4)
        self.assertEqual(totals['selected_primary_goals'], 0)
        with self.assertRaisesRegex(ValueError, 'complete seeds'):
            fit(training, validation, train_info, {'seeds': [4]})

    def test_offline_selection_matches_runtime_and_missing_features_keep_baseline(self):
        model = {'version': MODEL_VERSION, 'kind': 'logistic', 'features': ['height'],
                 'mean': [0], 'scale': [1], 'weights': [10], 'bias': 0, 'min_gain': 0,
                 'scope': 'height-only'}
        case = self.case(1, high_wins=True)
        by_pad = {tuple(branch['pad']): branch['features'] for branch in case['branches']}
        with patch('nhl94_ai.evaluation.shot_direction.direction_features',
                   side_effect=lambda state, pad: by_pad[pad]):
            self.assertEqual(DirectionSelector(model).choose(finish_state()),
                             tuple(case['branches'][selected_index(model, case)]['pad']))
            case['branches'][0]['features'] = by_pad[(-1, 0)] = None
            self.assertEqual(DirectionSelector(model).choose(finish_state()), (-1, 0))
            self.assertEqual(selected_index(model, case), 0)
            self.assertEqual(evaluate(model, [case])['totals']['cases'], 1)

    def test_reader_requires_complete_periods_and_all_nine_alternatives(self):
        report = {'protocol': MODEL_VERSION, 'matches': [
            {'seed': 1, 'matchup': 'example', 'completed': True, 'cases': [self.case(1, high_wins=True)]}]}
        with TemporaryDirectory() as directory:
            path = Path(directory)/'cases.json'
            path.write_text(json.dumps(report), encoding='utf-8')
            self.assertEqual(read_cases(path)[1]['cases'], 1)
            report['matches'][0]['completed'] = False
            path.write_text(json.dumps(report), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'Incomplete'):
                read_cases(path)
            report['matches'][0]['completed'] = True
            report['matches'][0]['cases'][0]['branches'].pop()
            path.write_text(json.dumps(report), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'all nine'):
                read_cases(path)

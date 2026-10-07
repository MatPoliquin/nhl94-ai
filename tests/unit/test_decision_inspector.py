"""Typed evidence does not turn controller preferences into tactical probabilities."""
from copy import deepcopy
from dataclasses import replace
import importlib.util
import os
import pickle
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from nhl94_ai.agents.base import AgentInput, AgentOutput, FrameRepeatAgent, ScriptedAgent
from nhl94_ai.agents.classic_v1 import ClassicAIV1Model
from nhl94_ai.agents.decisions import (
    ActionCandidate, DecisionScore, DecisionSnapshot, catalogue_snapshot, classic_decision_snapshot,
)
from nhl94_ai.agents.multi_model import NHL94AISystem
from nhl94_ai.game.constants import GameConsts
from nhl94_ai.evaluation.benchmark import away_view
from tests.unit.test_classic_offense import offense_state
from tests.unit import test_debug_playback


def neural_snapshot():
    return DecisionSnapshot(
        'Tactical neural adapter', 4, (
            ActionCandidate('carry', 'Carry puck', 'Tactics', scores=(DecisionScore('Q', 2.5),),
                            probability=0.4, raw_probability=0.3, probability_scope='offense/action',
                            source='Offense network', status='eligible', evaluated_frame=4),
            ActionCandidate('pass:1', 'Pass to LW [1]', 'Tactics', target_slot=1,
                            probability=0.6, probability_scope='offense/action', status='eligible', evaluated_frame=4),
        ), plan_id='carry', action=tuple(np.zeros(12)), evaluation_frame=4)


class DecisionContractTests(unittest.TestCase):
    def test_probability_validation_preserves_zero_and_requires_scope(self):
        candidate = ActionCandidate('shoot', 'Shoot', 'Offense', probability=0, probability_scope='finish/action')
        self.assertEqual(candidate.probability, 0)
        for probability in (-0.1, 1.1, float('nan'), float('inf')):
            with self.subTest(probability=probability), self.assertRaises(ValueError):
                replace(candidate, probability=probability)
        with self.assertRaisesRegex(ValueError, 'scope'):
            replace(candidate, probability_scope=None)
        with self.assertRaises(ValueError):
            DecisionScore('heuristic', float('nan'))

    def test_snapshot_rejects_ambiguous_or_missing_selected_ids(self):
        snapshot = neural_snapshot()
        with self.assertRaisesRegex(ValueError, 'unique'):
            replace(snapshot, candidates=snapshot.candidates * 2)
        with self.assertRaisesRegex(ValueError, 'Selected'):
            replace(snapshot, execution_id='missing')
        with self.assertRaises(ValueError):
            replace(snapshot, frame=-1)

    def test_classic_collection_is_read_only_and_has_no_fabricated_probabilities(self):
        controller, state = ClassicAIV1Model(), offense_state()
        action = controller.predict_frame(state)[0]
        before = pickle.dumps((controller, state))
        snapshot = classic_decision_snapshot(controller, state, action)
        self.assertEqual(pickle.dumps((controller, state)), before)
        self.assertTrue(snapshot.plan_id)
        self.assertTrue(all(candidate.probability is None for candidate in snapshot.candidates))
        self.assertEqual(snapshot.action, tuple(action))
        self.assertEqual(sum(candidate.group == 'Passing' for candidate in snapshot.candidates), 5)
        self.assertEqual(sum(candidate.group == 'One-timers' for candidate in snapshot.candidates), 5)
        self.assertEqual(next(c for c in snapshot.candidates if c.action_id == 'pass:0').status, 'unavailable')
        self.assertEqual(next(c for c in snapshot.candidates if c.action_id == 'slapshot').status, 'disabled')

    def test_defensive_goal_and_execution_are_separate_and_skating_is_not_a_boost(self):
        controller, state = ClassicAIV1Model(), offense_state()
        controller._last_decision = 'protect-lane'
        controller.defense_diagnostics = {'decision': 'protect-lane', 'mode': 'skating'}
        snapshot = classic_decision_snapshot(controller, state, np.zeros(12))
        self.assertEqual((snapshot.plan_id, snapshot.execution_id), ('protect-lane', 'skating'))
        controller.defense_diagnostics['mode'] = 'check-request'
        snapshot = classic_decision_snapshot(controller, state, np.zeros(12))
        self.assertEqual(snapshot.execution_id, 'check-request')

    def test_pending_pass_uses_its_real_receiver_without_renaming_public_actions(self):
        controller, state = ClassicAIV1Model(), offense_state()
        controller._last_decision = 'pass-flight'
        controller.offense.pending = {'receiver': 2}
        snapshot = classic_decision_snapshot(controller, state, np.zeros(12))
        self.assertEqual(snapshot.plan_id, 'pass:2')
        self.assertEqual(snapshot.phase, 'pass-flight')
        self.assertEqual(snapshot.action_schema, 'FILTERED')

    def test_away_roles_use_live_player_roles_and_physical_slots(self):
        state = away_view(offense_state())
        for player, role in zip(state.team1.players, (4, 3, 5, 1, 2)):
            player.role = role
        snapshot = catalogue_snapshot(state.team1, 'No tactical producer', 0)
        passes = [candidate for candidate in snapshot.candidates if candidate.group == 'Passing']
        self.assertEqual([candidate.target_slot for candidate in passes], list(range(6, 11)))
        self.assertEqual([candidate.label for candidate in passes],
                         ['Pass to C [6]', 'Pass to LW [7]', 'Pass to RW [8]', 'Pass to LD [9]', 'Pass to RD [10]'])
        self.assertIsNone(snapshot.plan_id)

    def test_scripted_adapter_preserves_exact_controller_outputs_and_cadence(self):
        agent = ScriptedAgent(ClassicAIV1Model, SimpleNamespace(action_type='FILTERED'))
        agent.frame_skip = 4
        reference = deepcopy(agent.controller)
        state = offense_state()
        for _ in range(32):
            actual = agent.act(AgentInput(state))
            np.testing.assert_array_equal(actual.action, reference.predict_frame(state, 4)[0])
            self.assertIsInstance(actual.decision, DecisionSnapshot)
            self.assertEqual((agent.controller._tick, agent.controller.defense.frames, agent.controller._frame_remaining),
                             (reference._tick, reference.defense.frames, reference._frame_remaining))

    def test_probability_snapshot_survives_multi_model_and_frame_repeat_adapters(self):
        snapshot = neural_snapshot()
        agent = Mock()
        agent.act.return_value = AgentOutput(np.zeros(12), {'native': 'metadata'}, snapshot)
        system = NHL94AISystem.__new__(NHL94AISystem)
        system.args = SimpleNamespace(side='home')
        system.models = [None, agent]
        system.game_state = offense_state()
        with patch('nhl94_ai.agents.multi_model.get_model_probabilities', return_value=np.zeros((1, 12))):
            system.Predict(1, np.zeros(2), True)
        self.assertIs(system.last_diagnostics['decision_inspector'], snapshot)
        self.assertNotIn('decision_inspector', agent.act.return_value.diagnostics)
        repeated = FrameRepeatAgent(agent, 4)
        results = [repeated.act(AgentInput(None)) for _ in range(4)]
        self.assertTrue(all(output.decision is snapshot for output in results))


@unittest.skipUnless(importlib.util.find_spec('pygame'), 'optional display extra is not installed')
class InspectorDisplayTests(unittest.TestCase):
    def setUp(self):
        variables = patch.dict(os.environ, {'SDL_VIDEODRIVER': 'dummy', 'SDL_AUDIODRIVER': 'dummy'})
        variables.start()
        self.addCleanup(variables.stop)
        import pygame
        pygame.init()
        pygame.event.clear()
        self.addCleanup(pygame.quit)

    def test_old_scores_are_gray_but_never_restore_an_old_selected_action(self):
        from nhl94_ai.ui.decision_panel import DecisionInspector
        inspector = DecisionInspector()
        snapshot = neural_snapshot()
        inspector.update(snapshot, 4)
        current = replace(snapshot, frame=9, plan_id='pass:1', evaluation_frame=None,
                          candidates=tuple(replace(candidate, scores=(), probability=None, raw_probability=None,
                                                   evaluated_frame=None) for candidate in snapshot.candidates))
        inspector.update(current, 9)
        rows = inspector.rows()
        self.assertEqual(inspector.selected_ids(np.zeros(12)), ('pass:1',))
        self.assertTrue(rows[0].historical)
        self.assertEqual(rows[0].candidate.scores[0].value, 2.5)
        self.assertIsNone(rows[0].candidate.probability)
        self.assertEqual(inspector.evaluation_at, 4)

    def test_repeated_evaluation_does_not_refresh_age_and_current_rejections_replace_scores(self):
        from nhl94_ai.ui.decision_panel import DecisionInspector
        inspector = DecisionInspector()
        snapshot = neural_snapshot()
        inspector.update(snapshot, 4)
        inspector.update(replace(snapshot, frame=8), 8)
        self.assertEqual(inspector.evaluation_at, 4)
        self.assertTrue(inspector.rows(8)[0].historical)
        rejected = replace(snapshot, frame=9, evaluation_frame=9,
                           candidates=tuple(replace(candidate, scores=(), probability=None, raw_probability=None,
                                                    status='rejected', reason='new rejection', evaluated_frame=9)
                                            for candidate in snapshot.candidates))
        inspector.update(rejected, 9)
        self.assertEqual(inspector.rows()[0].candidate.scores, ())
        self.assertFalse(inspector.rows()[0].historical)

    def test_probability_display_and_actual_action_overrides(self):
        import pygame
        from nhl94_ai.ui.decision_panel import DecisionInspector
        inspector = DecisionInspector()
        inspector.update(neural_snapshot(), 4)
        surface = pygame.Surface((740, 1080))
        font, big = pygame.font.SysFont('Arial', 16), pygame.font.SysFont('Arial', 24)
        inspector.draw(surface, surface.get_rect(), font, big, 4, actual_action=np.zeros(12))
        self.assertEqual(inspector.highlighted_ids, ('carry',))
        self.assertIn('pass:1', inspector.row_rects)
        bounds = inspector.row_rects['carry']
        self.assertEqual(surface.get_at((bounds.right - 1, bounds.bottom - 1))[:3], (24, 65, 42))
        changed = np.zeros(12)
        changed[0] = 1
        inspector.draw(surface, surface.get_rect(), font, big, 4, actual_action=changed)
        self.assertEqual(inspector.highlighted_ids, ())
        self.assertEqual(inspector.selected_ids(np.zeros(12), human_override=True), ())
        changed[:] = 0
        changed[GameConsts.INPUT_C] = changed[GameConsts.INPUT_RIGHT] = 1
        self.assertEqual(inspector.input_label(changed), 'RIGHT+C')
        inspector.reset()
        self.assertIsNone(inspector.evaluation)

    def test_logical_canvas_is_1080p_and_window_resizes_without_changing_game_geometry(self):
        import pygame
        helper = test_debug_playback.DebugPlaybackTests()
        self.addCleanup(helper.doCleanups)
        display, _ = helper.display()
        self.assertEqual(display.screen.get_size(), (1920, 1080))
        self.assertEqual(display.debug_surf.get_size(), (164, 300))
        self.assertEqual(display.game_surf.get_size(), (1040, 780))
        self.assertGreaterEqual(display.game_rect.left, display.ACTION_WIDTH)
        self.assertEqual(display.game_rect.bottom, display.mini_rink_rect.top)
        self.assertEqual(display.mini_rink_rect.right, display.stats_rect.left)
        self.assertEqual(display.stats_rect.bottom, 1080)
        self.assertEqual(display.stats_rect.right, 1920)
        display.set_ai_sys_info(SimpleNamespace(last_diagnostics={'decision_inspector': neural_snapshot()}))
        frame = np.zeros((224, 256, 3), dtype=np.uint8)
        display.draw_frame(frame, [{}], np.zeros(12))
        geometry = display.rink_rect.copy()
        pygame.event.post(pygame.event.Event(pygame.VIDEORESIZE, w=960, h=540))
        display.process_events()
        display.draw_frame(frame, [{}], np.zeros(12))
        self.assertEqual(display.window.get_size(), (960, 540))
        self.assertEqual(display.presentation_rect.size, (960, 540))
        self.assertEqual(display.rink_rect, geometry)
        self.assertEqual(display.screen.get_size(), (1920, 1080))

    def test_catalogue_is_scrollable_while_paused_and_goalie_setting_is_read_only(self):
        import pygame
        from nhl94_ai.ui.decision_panel import DecisionInspector
        inspector = DecisionInspector()
        snapshot = catalogue_snapshot(offense_state().team1, 'Classic rules', 0)
        inspector.update(snapshot, 0)
        surface = pygame.Surface((740, 1080))
        inspector.draw(surface, surface.get_rect(), pygame.font.SysFont('Arial', 16),
                       pygame.font.SysFont('Arial', 24), 0)
        self.assertGreater(inspector.max_scroll, 0)
        inspector.scroll_by(10000)
        self.assertEqual(inspector.scroll, inspector.max_scroll)
        inspector.scroll_by(-10000)
        self.assertEqual(inspector.scroll, 0)
        self.assertEqual(inspector.highlighted_ids, ())


if __name__ == '__main__':
    unittest.main()

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


def mode_snapshot(mode, frame):
    snapshot = catalogue_snapshot(offense_state().team1, 'Classic rules', frame)
    plan = {'offense': 'carry', 'defense': 'protect-lane', 'goalie': 'goalie-save'}[mode]
    candidates = tuple(replace(candidate, scores=(DecisionScore(mode, frame),), evaluated_frame=frame)
                       if candidate.action_id == plan else candidate for candidate in snapshot.candidates)
    return replace(snapshot, active_mode=mode, plan_id=plan, candidates=candidates,
                   action=tuple(np.zeros(12)), goalie_policy='selective', evaluation_frame=frame)


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
        with self.assertRaisesRegex(ValueError, 'mode'):
            replace(snapshot, active_mode='passing')

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

    def test_active_mode_and_shared_evidence_follow_executed_branch(self):
        controller, state = ClassicAIV1Model(), offense_state()
        scenarios = (
            ('carry', {}, 'offense', 'carry'),
            ('pass-flight', {}, 'offense', 'pass:2'),
            ('receive-pass-switch', {}, 'offense', 'pass:2'),
            ('receive-pass', {}, 'offense', 'pass:2'),
            ('one-timer-flight', {}, 'offense', 'one-timer:2'),
            ('goalie-avoid', {}, 'offense', 'goalie-avoid'),
            ('recover-safe', {'decision': 'recover-safe', 'mode': 'skating'}, 'defense', 'recover-safe'),
            ('goalie-avoid', {'decision': 'goalie-avoid', 'mode': 'goalie-avoid'}, 'defense', 'goalie-avoid'),
            ('goalie-hold', {}, 'goalie', 'goalie-hold'),
            ('goalie-outlet', {}, 'goalie', 'goalie-outlet'),
            ('goalie-request-goalie', {}, 'goalie', 'goalie-takeover'),
            ('goalie-neutral-handoff', {}, 'goalie', 'goalie-return'),
        )
        for decision, defense, mode, plan in scenarios:
            with self.subTest(decision=decision, mode=mode):
                controller._last_decision = decision
                controller.offense.pending = {'receiver': 2}
                controller.offense_diagnostics = {'desired_slot': 2, 'target': (12, 34), 'reason': 'offense reason'}
                controller.defense_diagnostics = dict(defense, reason='defense reason') if defense else {}
                controller.goalie_diagnostics = {'mode': decision.removeprefix('goalie-'), 'reason': 'goalie reason'}
                before = pickle.dumps((controller, state))
                snapshot = classic_decision_snapshot(controller, state, np.zeros(12))
                self.assertEqual(pickle.dumps((controller, state)), before)
                self.assertEqual((snapshot.active_mode, snapshot.plan_id), (mode, plan))
                expected = 'offense' if mode == 'offense' or decision in ('goalie-hold', 'goalie-outlet') else mode
                self.assertEqual(snapshot.reason, expected + ' reason')
                if expected == 'offense':
                    self.assertEqual((snapshot.desired_slot, snapshot.target), (2, (12, 34)))
                if decision == 'goalie-hold':
                    self.assertNotEqual(next(c for c in snapshot.candidates if c.action_id == plan).status, 'disabled')

    def test_goalie_outlets_keep_evidence_in_goalie_group_with_and_without_manual_control(self):
        for manual in (False, True):
            controller, state = ClassicAIV1Model(), offense_state()
            controller._last_decision = 'goalie-possession-outlet' if manual else 'goalie-outlet'
            details = {'mode': 'possession-outlet', 'outlet_candidates': [
                {'slot': 1, 'status': 'safe', 'value': 20},
                {'slot': 2, 'status': 'different-rom-recipient'},
            ]}
            if manual:
                controller.goalie = SimpleNamespace(phase='possession-outlet', outlet={'receiver': 1})
                controller.goalie_diagnostics = details
            else:
                controller.offense_diagnostics = details
            snapshot = classic_decision_snapshot(controller, state, np.zeros(12))
            rows = {c.action_id: c for c in snapshot.candidates if c.group == 'Outlets'}
            self.assertEqual(len(rows), 5)
            self.assertEqual(snapshot.active_mode, 'goalie')
            self.assertEqual(rows['outlet:1'].scores[0].value, 20)
            self.assertEqual(rows['outlet:2'].status, 'rejected')
            self.assertEqual(rows['outlet:0'].status, 'not-evaluated')

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

    def test_every_classic_tab_fits_without_scrolling_and_preserves_all_its_rows(self):
        import pygame
        from nhl94_ai.ui.decision_panel import DecisionInspector
        inspector = DecisionInspector()
        surface = pygame.Surface((740, 1080))
        controller, state = ClassicAIV1Model(), offense_state()
        for decision, mode in (('carry', 'offense'), ('recover-safe', 'defense'), ('goalie-outlet', 'goalie')):
            controller._last_decision = decision
            controller.defense_diagnostics = {'decision': decision} if mode == 'defense' else {}
            snapshot = classic_decision_snapshot(controller, state, np.zeros(12))
            inspector.update(snapshot, 0)
            with patch.object(inspector, '_text', wraps=inspector._text) as text:
                inspector.draw(surface, surface.get_rect(), pygame.font.SysFont('Arial', 16),
                               pygame.font.SysFont('Arial', 24), 0)
            self.assertEqual(inspector.selected_mode, mode)
            self.assertEqual(inspector.max_scroll, 0)
            self.assertEqual(set(inspector.row_rects), {row.candidate.action_id for row in inspector.rows()})
            self.assertTrue(all(bounds.height == inspector.ROW_HEIGHT for bounds in inspector.row_rects.values()))
            self.assertTrue(all(bounds.bottom <= 996 for bounds in inspector.row_rects.values()))
            self.assertNotIn('P eff/raw', [call.args[2] for call in text.call_args_list])
            self.assertNotIn('wait', inspector.row_rects)

    def test_mode_history_is_independent_and_pause_inspection_never_highlights_old_actions(self):
        import pygame
        from nhl94_ai.ui.decision_panel import DecisionInspector
        inspector = DecisionInspector()
        surface = pygame.Surface((740, 1080))
        font, big = pygame.font.SysFont('Arial', 16), pygame.font.SysFont('Arial', 24)
        for mode, frame in (('offense', 4), ('defense', 8), ('goalie', 12)):
            inspector.update(mode_snapshot(mode, frame), frame)
            self.assertEqual(inspector.selected_mode, mode)
            self.assertEqual(inspector.evaluation_at, frame)
        inspector.draw(surface, surface.get_rect(), font, big, 12)
        inspector.click(inspector.tab_rects['offense'].center)
        self.assertEqual(inspector.selected_mode, 'goalie')
        inspector.set_paused(True)
        inspector.click(inspector.tab_rects['offense'].center)
        self.assertEqual(inspector.evaluation_at, 4)
        self.assertTrue(inspector.historical)
        self.assertEqual(inspector.snapshot.frame, 4)
        self.assertTrue(all(row.historical for row in inspector.rows(12)))
        with patch.object(inspector, '_text', wraps=inspector._text) as text:
            inspector.draw(surface, surface.get_rect(), font, big, 12, actual_action=np.ones(12))
        self.assertEqual(inspector.highlighted_ids, ())
        texts = [call.args[2] for call in text.call_args_list]
        self.assertTrue(any('HISTORICAL: 8' in value for value in texts))
        self.assertIn('Recorded input: neutral', texts)
        self.assertFalse(any('override' in value for value in texts))
        inspector.set_paused(False)
        self.assertEqual(inspector.selected_mode, 'goalie')
        self.assertFalse(inspector.historical)
        inspector.update(replace(mode_snapshot('offense', 16), evaluation_frame=None,
                                 candidates=tuple(replace(c, scores=(), evaluated_frame=None)
                                                  for c in mode_snapshot('offense', 16).candidates)), 16)
        self.assertEqual(inspector.evaluation_at, 4)
        self.assertEqual(inspector.rows()[0].candidate.scores[0].value, 4)
        self.assertEqual(inspector.selected_ids(), ('carry',))
        inspector.reset()
        inspector.update(mode_snapshot('goalie', 20), 20)
        inspector.set_paused(True)
        inspector.draw(surface, surface.get_rect(), font, big, 20)
        inspector.click(inspector.tab_rects['offense'].center)
        self.assertIsNone(inspector.snapshot)
        self.assertEqual(inspector.rows(), ())
        self.assertIsNone(inspector.evaluation)

    def test_resized_tab_clicks_and_resume_do_not_send_game_input(self):
        import pygame
        helper = test_debug_playback.DebugPlaybackTests()
        self.addCleanup(helper.doCleanups)
        display, _ = helper.display()
        for mode, frame in (('offense', 4), ('defense', 8)):
            display.set_ai_sys_info(SimpleNamespace(last_diagnostics={'decision_inspector': mode_snapshot(mode, frame)}))
        pygame.event.post(pygame.event.Event(pygame.VIDEORESIZE, w=960, h=600))
        display.process_events()
        display.draw_frame(np.zeros((224, 256, 3), dtype=np.uint8), [{}], np.zeros(12))
        logical = display.inspector.tab_rects['offense'].center
        presentation = display.presentation_rect
        position = (presentation.x + round(logical[0] * presentation.width / 1920),
                    presentation.y + round(logical[1] * presentation.height / 1080))
        buttons = list(display.player_actions)
        with patch.object(display.env, 'step') as step:
            pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_SPACE))
            pygame.event.post(pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=position))
            display.process_events()
            self.assertTrue(display.paused)
            self.assertEqual(display.inspector.selected_mode, 'offense')
            pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_SPACE))
            display.process_events()
            self.assertFalse(display.paused)
            self.assertEqual(display.inspector.selected_mode, 'defense')
            step.assert_not_called()
        self.assertEqual(display.player_actions, buttons)


if __name__ == '__main__':
    unittest.main()

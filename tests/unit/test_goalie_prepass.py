"""Contracts for diagnostic interventions, not production goalie heuristics."""
from dataclasses import replace
import unittest
from unittest.mock import patch

from nhl94_ai.agents import goalie
from nhl94_ai.game.constants import GameConsts as Buttons
from tests.integration.goalie_prepass import component_controller, summarize_comparison
from tests.unit.test_manual_goalie import goalie_state


class ComponentExperimentsTests(unittest.TestCase):
    def test_unknown_experiment_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Unknown goalie experiment'):
            component_controller(goalie, goalie.GoalieController('selective'), 'misspelled')

    def test_anticipation_replaces_only_target_and_reason(self):
        state = goalie_state(True)
        raw = goalie.GoaliePlan((0, -246), 'raw', 20, 2, 4, 24)
        advanced = replace(raw, target=(15, -244), deadline=100,
                           receiver=(60, -180), reception_frames=12)
        manager = component_controller(goalie, goalie.GoalieController('selective'), 'anticipation-only')
        with patch.object(goalie, 'goalie_target', side_effect=(raw, advanced)):
            manager.step(state)
        self.assertEqual(manager.diagnostics['destination'], advanced.target)
        self.assertEqual(manager.diagnostics['crossing_frames'], raw.crossing_frames)
        self.assertEqual(manager.diagnostics['crossing_x'], raw.crossing_x)
        self.assertEqual(manager.diagnostics['deadline'], raw.deadline)
        self.assertIsNone(manager.diagnostics['receiver'])

    def test_no_save_commitment_keeps_movement_without_faking_a_save_request(self):
        state = goalie_state(True)
        state.engine.puck_owner = -256
        state.puck.x, state.puck.y, state.puck.motion_y = 14, -215, -4
        state.puck.friction = 1
        state.team1.goalie.motion_x = -1.5
        original = goalie.GoalieController('selective')
        manager = component_controller(goalie, original, 'no-save-commitment')
        self.assertTrue(original.step(state)[Buttons.INPUT_C])
        action = manager.step(state)
        self.assertTrue(action[Buttons.INPUT_RIGHT])
        self.assertFalse(action[Buttons.INPUT_C])
        self.assertFalse(action[Buttons.INPUT_A])
        self.assertIsNone(manager.pending_save)
        self.assertEqual(manager.save_at, 0)
        self.assertEqual(manager.metrics['save_requests'], 0)

    def test_CPU_variant_does_not_try_to_switch_while_holding_the_puck(self):
        state = goalie_state(True)
        state.engine.puck_owner = state.team1.defense_goalie
        manager = component_controller(goalie, goalie.GoalieController('selective'), 'cpu-goalie')
        self.assertFalse(manager.step(state).any())
        self.assertEqual(manager.phase, 'possession-outlet')
        self.assertEqual(manager.metrics['return_requests'], 0)

    def test_CPU_variant_respects_animation_lock_and_never_requests_takeover(self):
        state = goalie_state(True)
        state.team1.goalie.unavailable = 2
        manager = component_controller(goalie, goalie.GoalieController('selective'), 'cpu-goalie')
        self.assertFalse(manager.step(state).any())
        self.assertEqual(manager.metrics['return_requests'], 0)
        state = goalie_state()
        for _ in range(3):
            action = manager.step(state)
        self.assertIsNone(action)
        self.assertEqual(manager.metrics['takeover_requests'], 0)

    def test_initial_history_and_production_controller_are_not_mutated(self):
        original = goalie.GoalieController('selective')
        original.frames, original.save_at = 123, 140
        manager = component_controller(goalie, original, 'early-takeover-only')
        self.assertEqual(manager.frames, 123)
        self.assertEqual(manager.save_at, 140)
        self.assertEqual(manager.policy, 'always')
        self.assertEqual(original.policy, 'selective')

    def test_incomplete_replay_cannot_claim_matching_puck_through_release(self):
        fields = {'puck': [0, 0, 0], 'puck_velocity': [0, 0, 0], 'owner': 6, 'last_toucher': 6,
                  'pass_target': 7, 'passes': 1, 'one_timers': 0, 'buttons': [0] * 12, 'metrics': {}}
        before = {'trace': [dict(fields, frame=1), dict(fields, frame=2)]}
        after = {'trace': [dict(fields, frame=1)]}
        self.assertFalse(summarize_comparison(before, after, 2)['same_puck_through_original_release'])


if __name__ == '__main__':
    unittest.main()

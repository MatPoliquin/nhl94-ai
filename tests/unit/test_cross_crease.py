"""Opportunity arbitration, native edges, per-frame cadence and attribution."""
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from nhl94_ai.agents.classic_v1 import ClassicAIV1Model
from nhl94_ai.agents.cross_crease import (
    APPROACH_FRAMES, COOLDOWN_FRAMES, HOLD_FRAMES, REPLAN_FRAMES, SEQUENCE_FRAMES,
    CrossCreaseController, coast_projection, crossing_clear, evaluate_cross_crease,
)
from nhl94_ai.agents.registry import create_scripted
from nhl94_ai.env.actions import HockeyActionController
from nhl94_ai.game.constants import GameConsts as Buttons
from tests.unit.test_classic_offense import offense_state


def crossing_state(*, direction=1, attack=1):
    state = offense_state()
    player = state.team1.players[0]
    player.x, player.y = -direction * 35, attack * 212
    player.motion_x, player.motion_y = direction * 1.65, 0
    player.shot_power, player.handedness = 30, 0
    player.live_anim, player.live_anim_frame, player.animation_timer = 0, 0, 0
    player.facing = 2 if direction > 0 else 6
    for other in (*state.team1.players[1:], *state.team2.players):
        other.x, other.y = 110, -attack * 180
    state.team1.net.y, state.team2.net.y = -attack * 264, attack * 264
    state.team1.goalie.y = -attack * 250
    state.team2.goalie.x, state.team2.goalie.y = -direction * 12, attack * 250
    state.team2.goalie.live_anim = 2
    state.puck.x, state.puck.y = player.x + direction * 10, player.y
    state.engine.shot_player = -1
    return state


class CrossCreaseEvaluationTests(unittest.TestCase):
    def test_clear_crossing_beats_a_weak_shot_in_each_direction_and_end(self):
        for direction in (-1, 1):
            for attack in (-1, 1):
                with self.subTest(direction=direction, attack=attack):
                    state = crossing_state(direction=direction, attack=attack)
                    plan, diagnostics = evaluate_cross_crease(state, {'shoot': 20})
                    self.assertIsNotNone(plan, diagnostics)
                    self.assertEqual(plan.direction, direction)
                    self.assertEqual(plan.aim_side, direction)
                    self.assertGreater(plan.release_point[0] * direction, 4)

    def test_a_better_one_timer_or_pass_wins(self):
        for action in ('one-timer', 'position-pass', 'shoot', 'carry'):
            plan, diagnostics = evaluate_cross_crease(crossing_state(), {action: 100})
            self.assertIsNone(plan)
            self.assertEqual(diagnostics['status'], 'better-alternative')
            self.assertEqual(diagnostics['alternatives'][action], 100)

    def test_already_worthwhile_crossing_does_not_chase_a_higher_setup_score(self):
        def value(_state, _player, *, point, delay=0):
            return 100 if point[0] > 30 else 50
        with patch('nhl94_ai.agents.cross_crease.shot_value', side_effect=value):
            plan, diagnostics = evaluate_cross_crease(crossing_state(), {'shoot': 20})
        self.assertIsNotNone(plan, diagnostics)
        self.assertEqual(plan.approach_frames, 0)

    def test_motion_power_and_animation_feedback_are_required(self):
        for field in ('motion_x', 'motion_y', 'shot_power', 'handedness', 'live_anim', 'live_anim_frame'):
            state = crossing_state()
            setattr(state.team1.players[0], field, None)
            with self.subTest(field=field):
                plan, diagnostics = evaluate_cross_crease(state, {})
                self.assertIsNone(plan)
                self.assertEqual(diagnostics['status'], 'missing-feedback')

    def test_short_windup_uses_its_actual_smaller_crossing_window(self):
        state = crossing_state()
        state.team1.players[0].shot_power = 19
        plan, diagnostics = evaluate_cross_crease(state, {})
        self.assertIsNotNone(plan, diagnostics)
        self.assertEqual(plan.shot_frames, 32)
        self.assertEqual(evaluate_cross_crease(crossing_state(), {})[0].shot_frames, 38)

    def test_shootouts_keep_ordinary_finishing(self):
        state = crossing_state()
        state.is_shootout_active = True
        self.assertEqual(evaluate_cross_crease(state, {})[1]['status'], 'ineligible-state')

    def test_defender_goalie_and_teammate_obstructions_reject(self):
        for obstacle in ('defender', 'goalie', 'teammate'):
            state = crossing_state()
            player = (state.team2.players[0] if obstacle == 'defender' else
                      state.team2.goalie if obstacle == 'goalie' else state.team1.players[1])
            player.x, player.y = -5, 214
            with self.subTest(obstacle=obstacle):
                self.assertIsNone(evaluate_cross_crease(state, {})[0])

    def test_wrong_momentum_can_be_prepared_but_invalid_net_route_is_rejected(self):
        state = crossing_state()
        state.team1.players[0].motion_x = -2
        plan, diagnostics = evaluate_cross_crease(state, {})
        self.assertIsNotNone(plan, diagnostics)
        self.assertGreater(plan.approach_frames, 0)
        self.assertIsNone(evaluate_cross_crease(state, {}, prepare=False)[0])
        state.team1.players[0].y = 255
        self.assertIsNone(evaluate_cross_crease(state, {})[0])

    def test_evaluation_does_not_modify_game_state(self):
        state = crossing_state()
        before = deepcopy(state)
        evaluate_cross_crease(state, {})
        self.assertEqual(vars(state.team1.players[0]), vars(before.team1.players[0]))
        self.assertEqual(vars(state.team2.goalie), vars(before.team2.goalie))
        self.assertEqual(vars(state.puck), vars(before.puck))
        for player, original in zip(state.team2.players, before.team2.players):
            self.assertEqual(vars(player), vars(original))

    def test_stationary_wide_wing_is_a_planned_start_in_both_directions_and_ends(self):
        for direction in (-1, 1):
            for attack in (-1, 1):
                state = crossing_state(direction=direction, attack=attack)
                player = state.team1.players[0]
                player.x, player.y = -direction * 100, attack * 160
                player.motion_x = player.motion_y = 0
                state.puck.x, state.puck.y = player.x + direction * 10, player.y
                with self.subTest(direction=direction, attack=attack):
                    plan, diagnostics = evaluate_cross_crease(state, {'shoot': 0})
                    self.assertIsNotNone(plan, diagnostics)
                    self.assertGreater(plan.approach_frames, 48)
                    self.assertLessEqual(plan.approach_frames, APPROACH_FRAMES)
                    self.assertGreater(plan.charge_point[0] * direction, player.x * direction)
                    self.assertIsNone(evaluate_cross_crease(state, {}, prepare=False)[0])

    def test_diagonal_crossing_is_judged_by_its_reachable_opening(self):
        state = crossing_state()
        player = state.team1.players[0]
        player.y, player.motion_y = 190, 0.6
        state.puck.y = player.y
        plan, diagnostics = evaluate_cross_crease(state, {}, prepare=False)
        self.assertIsNotNone(plan, diagnostics)
        self.assertEqual(plan.approach_frames, 0)

    def test_distant_skater_projection_through_the_boards_does_not_veto_the_carrier(self):
        state = crossing_state()
        for other in (state.team1.players[1], state.team2.players[0]):
            other.x, other.motion_x = 119, 3
        plan, diagnostics = evaluate_cross_crease(state, {})
        self.assertIsNotNone(plan, diagnostics)

    def test_net_front_occupants_do_not_veto_an_unobstructed_wing_route(self):
        state = crossing_state()
        player = state.team1.players[0]
        player.x, player.y, player.motion_x, player.motion_y = -100, 160, 0, 0
        state.puck.x, state.puck.y = -90, 160
        state.team1.players[1].x, state.team1.players[1].y = 28, 244
        state.team2.players[0].x, state.team2.players[0].y = -28, 244
        plan, diagnostics = evaluate_cross_crease(state, {})
        self.assertIsNotNone(plan, diagnostics)
        self.assertGreater(plan.approach_frames, 48)

    def test_an_imminent_wing_collision_still_vetoes_the_maneuver(self):
        state = crossing_state()
        player = state.team1.players[0]
        player.x, player.y, player.motion_x, player.motion_y = -100, 160, 0, 0
        state.puck.x, state.puck.y = -90, 160
        state.team2.players[0].x, state.team2.players[0].y = -90, 160
        plan, diagnostics = evaluate_cross_crease(state, {})
        self.assertIsNone(plan)
        self.assertEqual(diagnostics['status'], 'no-safe-crossing')

    def test_replanning_does_not_deepen_the_chosen_crossing_lane(self):
        state = crossing_state()
        player = state.team1.players[0]
        player.x, player.y, player.motion_x, player.motion_y = -100, 160, 0, 0
        state.puck.x, state.puck.y = -90, 160
        plan, diagnostics = evaluate_cross_crease(state, {}, max_depth=192)
        self.assertIsNotNone(plan, diagnostics)
        self.assertLessEqual(plan.target[1], 192)
        self.assertTrue(all(point[1] <= 192 for point in plan.route))

    def test_windup_projection_coasts_instead_of_accelerating(self):
        player = crossing_state().team1.players[0]
        x, y = coast_projection(player, 24)
        self.assertGreater(x, player.x)
        self.assertLess(x, player.x + player.motion_x * 24)
        self.assertEqual(y, player.y)

    def test_setup_can_build_lateral_momentum_instead_of_requiring_it_on_entry(self):
        state = crossing_state()
        player = state.team1.players[0]
        player.x, player.y = -55, 214
        player.motion_x, player.motion_y, player.facing = 0, 0, 2
        state.puck.x, state.puck.y = player.x + 10, player.y
        plan, diagnostics = evaluate_cross_crease(state, {})
        self.assertIsNotNone(plan, diagnostics)
        self.assertGreater(plan.approach_frames, 16)
        self.assertLessEqual(plan.approach_frames, APPROACH_FRAMES)

    def test_setup_checks_the_whole_approach_not_only_the_old_18_frame_cut_window(self):
        state = crossing_state()
        player = state.team1.players[0]
        player.x, player.y = -55, 214
        player.motion_x, player.motion_y, player.facing = 0, 0, 2
        state.puck.x, state.puck.y = player.x + 10, player.y
        state.team1.players[1].x, state.team1.players[1].y = -30, 215
        self.assertIsNone(evaluate_cross_crease(state, {})[0])

    def test_possible_pursuit_is_a_risk_cost_not_a_blanket_veto(self):
        state = crossing_state()
        def margin(_opponents, point, _frames):
            return -1 if point[0] > -25 else 60
        with patch('nhl94_ai.agents.cross_crease._arrival_margin', side_effect=margin):
            clear, reason, pressure, _ = crossing_clear(state, state.team1.players[0], 4)
        self.assertTrue(clear)
        self.assertEqual(reason, 'clear')
        self.assertEqual(pressure, -1)


class CrossCreaseExecutionTests(unittest.TestCase):
    def setUp(self):
        self.state = crossing_state()
        plan, diagnostics = evaluate_cross_crease(self.state, {})
        self.assertIsNotNone(plan, diagnostics)
        self.controller = CrossCreaseController()
        self.controller.start(plan, self.state, 0)

    def test_old_c_is_released_before_a_fresh_press(self):
        action = self.controller.step(self.state, 0, c_down=True)
        self.assertFalse(action[Buttons.INPUT_C])
        self.assertEqual(self.controller.phase, 'approach')
        action = self.controller.step(self.state, 1, c_down=False)
        self.assertTrue(action[Buttons.INPUT_C])
        self.assertEqual(self.controller.metrics['attempts'], 1)

    def test_hold_is_finite_without_a_goalie_commit(self):
        self.controller.step(self.state, 0)
        self.state.is_shooting = True
        for frame in range(1, HOLD_FRAMES):
            action = self.controller.step(self.state, frame, c_down=True)
            self.assertTrue(action[Buttons.INPUT_C])
        action = self.controller.step(self.state, HOLD_FRAMES, c_down=True)
        self.assertFalse(action[Buttons.INPUT_C])
        self.assertEqual(self.controller.phase, 'release')
        for frame in range(HOLD_FRAMES + 1, HOLD_FRAMES + 4):
            self.assertFalse(self.controller.step(self.state, frame)[Buttons.INPUT_C])
        self.assertEqual(self.controller.metrics['c_releases'], 1)

    def test_actual_goalie_animation_can_release_an_open_crossing(self):
        self.controller.step(self.state, 0)
        self.state.is_shooting = True
        self.state.team1.players[0].x = 2
        self.state.puck.x = 14
        self.state.team2.goalie.live_anim = 0x250
        action = self.controller.step(self.state, 10, c_down=True)
        self.assertFalse(action[Buttons.INPUT_C])
        self.assertTrue(self.controller.committed)
        self.assertEqual(self.controller.diagnostics['release_reason'], 'committed-opening')
        self.assertEqual(self.controller.metrics['pre_release_commits'], 1)

    def test_early_save_does_not_release_before_the_native_fast_swing_can_cross(self):
        self.controller.step(self.state, 0)
        self.state.is_shooting = True
        self.state.team1.players[0].x = -25
        self.state.puck.x = -7
        self.state.team2.goalie.live_anim = 0x250
        action = self.controller.step(self.state, 10, c_down=True)
        self.assertTrue(action[Buttons.INPUT_C])
        self.assertTrue(self.controller.committed)
        self.assertFalse(self.controller.diagnostics['opening'])

    def test_actual_windup_advance_is_distinct_from_a_save_animation(self):
        self.controller.step(self.state, 0)
        self.state.is_shooting = True
        self.state.team2.goalie.y -= 5
        self.controller.step(self.state, 8, c_down=True)
        self.assertTrue(self.controller.committed)
        self.assertEqual(self.controller.metrics['angle_commits'], 1)
        self.assertEqual(self.controller.metrics['pre_release_commits'], 0)
        self.assertEqual(self.controller.event['commit_kind'], 'windup-advance')

    def test_obstruction_during_windup_releases_without_passing_or_redirecting_aim(self):
        self.controller.step(self.state, 0)
        self.state.is_shooting = True
        self.state.team2.players[0].x, self.state.team2.players[0].y = -20, 213
        action = self.controller.step(self.state, 1, c_down=True)
        self.assertFalse(action[Buttons.INPUT_C] or action[Buttons.INPUT_B])
        self.assertTrue(action[Buttons.INPUT_RIGHT])
        self.assertEqual(self.controller.phase, 'release')

    def test_accepted_windup_does_not_mean_recorded_shot_or_goal(self):
        self.controller.step(self.state, 0)
        self.state.is_shooting = True
        self.controller.step(self.state, 1)
        self.assertEqual(self.controller.metrics['accepted_windups'], 1)
        self.assertEqual(self.controller.metrics['recorded_shots'], 0)
        self.state.engine.puck_owner = -255
        self.state.engine.shot_player = 1
        self.state.team1.stats.shots = 1
        self.state.team1.stats.score = 1
        self.controller.observe(self.state, 2)
        self.assertEqual(self.controller.metrics['recorded_shots'], 0)
        self.assertEqual(self.controller.metrics['goals'], 0)
        self.controller.observe(self.state, SEQUENCE_FRAMES)
        self.assertEqual(self.controller.events[-1]['outcome'], 'shot-unconfirmed')

    def test_confirmed_goal_is_counted_once_for_the_correct_shooter(self):
        self.controller.step(self.state, 0)
        self.state.engine.puck_owner = -255
        self.state.engine.shot_player = 0
        self.state.team1.stats.shots = 1
        self.state.team1.stats.score = 1
        self.controller.observe(self.state, 40)
        self.controller.observe(self.state, 41)
        self.assertEqual(self.controller.metrics['recorded_shots'], 1)
        self.assertEqual(self.controller.metrics['goals'], 1)
        self.assertEqual(len(self.controller.events), 1)

    def test_control_loss_clears_c_without_starting_another_tactic(self):
        self.controller.step(self.state, 0)
        self.state.team1.defense_control = -1
        action = self.controller.step(self.state, 1, c_down=True)
        self.assertFalse(np.any(action))
        self.assertIsNone(self.controller.plan)
        self.assertEqual(self.controller.events[-1]['outcome'], 'control-lost')

    def test_native_goalie_contact_not_proximity_is_counted(self):
        self.state.team2.goalie.contact_player = 0
        self.state.team2.goalie.contact_impact = 0
        self.controller.observe(self.state, 0)
        self.assertEqual(self.controller.metrics['goalie_contacts'], 0)
        self.state.team2.goalie.contact_impact = 4
        self.controller.observe(self.state, 1)
        self.controller.observe(self.state, 2)
        self.assertEqual(self.controller.metrics['goalie_contacts'], 1)

    def test_stale_positive_impact_does_not_count_as_a_new_contact(self):
        state = crossing_state()
        state.team2.goalie.contact_player, state.team2.goalie.contact_impact = 0, 10
        plan, _ = evaluate_cross_crease(state, {})
        controller = CrossCreaseController()
        controller.start(plan, state, 0)
        controller.observe(state, 0)
        self.assertEqual(controller.metrics['goalie_contacts'], 0)
        state.team2.goalie.contact_impact = 15
        controller.observe(state, 1)
        self.assertEqual(controller.metrics['goalie_contacts'], 1)

    def test_benchmark_counts_fresh_goalie_impulses_not_stale_frames(self):
        from nhl94_ai.evaluation.offense_metrics import OffenseMetrics
        metrics = OffenseMetrics()
        self.state.team2.goalie.contact_player, self.state.team2.goalie.contact_impact = 0, 10
        metrics.observe(0, self.state)
        metrics.observe(1, self.state)
        self.state.team2.goalie.contact_impact = 15
        metrics.observe(2, self.state)
        metrics.observe(3, self.state)
        self.assertEqual(metrics.summary()['goalie_contact_impulses'], 1)

    def test_rejected_native_press_ends_neutrally_instead_of_pretending_to_charge(self):
        self.controller.step(self.state, 0)
        action = self.controller.step(self.state, 3, c_down=True)
        self.assertFalse(np.any(action))
        self.assertEqual(self.controller.events[-1]['outcome'], 'shot-not-started')

    def test_missing_motion_during_windup_releases_c_and_reports_missing_feedback(self):
        self.controller.step(self.state, 0)
        self.state.is_shooting = True
        self.state.team1.players[0].motion_x = None
        action = self.controller.step(self.state, 1, c_down=True)
        self.assertFalse(action[Buttons.INPUT_C])
        self.assertTrue(action[Buttons.INPUT_RIGHT])
        self.assertEqual(self.controller.diagnostics['release_reason'], 'feedback-lost')
        self.assertIn('motion_x', self.controller.diagnostics['missing_feedback'])

    def test_loose_puck_transition_releases_without_dereferencing_a_missing_owner(self):
        self.controller.step(self.state, 0)
        self.state.is_shooting = True
        self.controller.step(self.state, 1, c_down=True)
        self.state.is_shooting = False
        self.state.engine.puck_owner = -255
        action = self.controller.step(self.state, 2, c_down=True)
        self.assertFalse(action[Buttons.INPUT_C])
        self.assertEqual(self.controller.diagnostics['release_reason'], 'possession-lost')

    def test_changed_route_is_explicitly_replanned_and_recorded(self):
        previous = self.controller.plan
        replacement = replace(previous, route=((-55, 192), (50, 192)), target=(50, 192))
        with patch('nhl94_ai.agents.cross_crease._windup_option',
                   return_value=(None, {'status': 'crossing-not-reachable-during-windup'})), \
                patch('nhl94_ai.agents.cross_crease.evaluate_cross_crease',
                      return_value=(replacement, {'status': 'selected'})), \
                patch('nhl94_ai.agents.cross_crease._path_clear', return_value=(True, 'clear', 60, None)):
            action = self.controller.step(self.state, REPLAN_FRAMES)
        self.assertIs(self.controller.plan, replacement)
        self.assertEqual(self.controller.event['replans'][0]['previous_route'], previous.route)
        self.assertEqual(self.controller.metrics['route_replans'], 1)
        self.assertEqual(self.controller.diagnostics['waypoint'], replacement.route[0])
        self.assertFalse(action[Buttons.INPUT_C])

    def test_retry_after_preparation_cancel_is_short_but_a_pressed_attempt_cools_down(self):
        self.controller._finish(10, 'approach-invalidated')
        self.assertEqual(self.controller.retry_at, 22)
        plan, _ = evaluate_cross_crease(self.state, {})
        self.controller.start(plan, self.state, 22)
        self.controller.step(self.state, 22)
        self.controller._finish(23, 'possession-lost')
        self.assertEqual(self.controller.retry_at, 23 + COOLDOWN_FRAMES)


class ClassicCrossCreaseTests(unittest.TestCase):
    def test_classic_command_options_are_explicitly_off_by_default(self):
        from nhl94_ai.evaluation import play, runner, cpu_benchmark
        from nhl94_ai.training import collect, dagger
        for module in (play, runner, cpu_benchmark, collect, dagger):
            parser = module.build_parser()
            with self.subTest(command=module.__name__):
                self.assertFalse(parser.get_default('cross_crease'))
                self.assertTrue(parser.parse_known_args(['--cross-crease'] +
                                (['--model', 'policy', '--base_datasets', 'data'] if module is dagger else []))[0].cross_crease)

    def test_tactic_is_opt_in_and_keeps_default_actions(self):
        state = crossing_state()
        default = ClassicAIV1Model()
        disabled = ClassicAIV1Model(SimpleNamespace(cross_crease=False))
        for _ in range(12):
            np.testing.assert_array_equal(default.predict_frame(state), disabled.predict_frame(state))
        self.assertIsNone(default.cross_crease)

    def test_crossing_is_not_overridden_by_close_finishing(self):
        state = crossing_state()
        model = ClassicAIV1Model(SimpleNamespace(cross_crease=True))
        with patch('nhl94_ai.agents.classic_v1.shot_value', return_value=10), \
                patch.object(model.offense, 'choose', return_value=None):
            action = model.predict_frame(state)[0]
        self.assertIsNotNone(model.cross_crease.plan, model.offense_diagnostics)
        self.assertEqual(model._last_decision, 'cross-crease-hold')
        self.assertTrue(action[Buttons.INPUT_C])

    def test_approach_overlay_uses_the_current_waypoint_not_the_final_crossing_target(self):
        state = crossing_state()
        model = ClassicAIV1Model(SimpleNamespace(cross_crease=True))
        plan, _ = evaluate_cross_crease(state, {})
        plan = replace(plan, route=((-55, 192), (50, 192)), target=(50, 192))
        model.cross_crease.start(plan, state, 0)
        with patch('nhl94_ai.agents.cross_crease._windup_option',
                   return_value=(None, {'status': 'crossing-not-reachable-during-windup'})), \
                patch('nhl94_ai.agents.cross_crease._path_clear', return_value=(True, 'clear', 60, None)):
            model.predict_frame(state)
        self.assertEqual(model.offense_diagnostics['waypoint'], plan.route[0])
        self.assertEqual(model.offense_diagnostics['destination'], plan.target)

    def test_active_hold_uses_emulator_frames_in_both_action_formats(self):
        for schema in ('FILTERED', 'HOCKEY_INTENT_DPAD'):
            for interval in (1, 4, 10):
                with self.subTest(schema=schema, interval=interval):
                    state = crossing_state()
                    model = ClassicAIV1Model(SimpleNamespace(cross_crease=True, action_type=schema))
                    plan, _ = evaluate_cross_crease(state, {})
                    model.cross_crease.start(plan, state, 0)
                    context = SimpleNamespace(action_type=schema, game_state=state)
                    processor = HockeyActionController(context)
                    macro = processor._new_action_state()
                    for frame in range(HOLD_FRAMES + 2):
                        action = model.predict_frame(state, interval)[0]
                        buttons = processor._process_action(action, macro)[0] if schema != 'FILTERED' else action
                        self.assertEqual(bool(buttons[Buttons.INPUT_C]), frame < HOLD_FRAMES)
                        self.assertFalse(buttons[Buttons.INPUT_B])
                        state.is_shooting = True

    def test_turnover_sends_neutral_before_defensive_c_or_b(self):
        state = crossing_state()
        model = ClassicAIV1Model(SimpleNamespace(cross_crease=True))
        plan, _ = evaluate_cross_crease(state, {})
        model.cross_crease.start(plan, state, 0)
        self.assertTrue(model.predict_frame(state)[0, Buttons.INPUT_C])
        state.engine.puck_owner = 6
        self.assertFalse(np.any(model.predict_frame(state)[0]))
        self.assertIsNone(model.cross_crease.plan)
        required = {'mode', 'reason', 'actual_slot', 'desired_slot', 'target', 'destination', 'waypoint'}
        self.assertLessEqual(required, model.offense_diagnostics.keys())
        self.assertEqual(model.offense_diagnostics['reason'], 'possession-lost')

    def test_reset_clears_pending_crossing_and_metrics(self):
        agent = create_scripted('classic', SimpleNamespace(cross_crease=True, action_type='FILTERED'))
        state = crossing_state()
        plan, _ = evaluate_cross_crease(state, {})
        agent.cross_crease.start(plan, state, 0)
        agent.reset()
        self.assertIsNone(agent.cross_crease.plan)
        self.assertFalse(agent.cross_crease.metrics)

    def test_unsupported_controls_and_variants_fail_explicitly(self):
        for options in ({'action_type': 'TARGET_POSITION'}, {'env': 'NHL941on1-Genesis-v0'}):
            with self.subTest(options=options), self.assertRaisesRegex(ValueError, 'Cross-crease'):
                ClassicAIV1Model(SimpleNamespace(cross_crease=True, **options))

    def test_learned_evaluation_does_not_silently_ignore_the_classic_flag(self):
        from nhl94_ai.evaluation import runner
        args = runner.build_parser().parse_args(['--cross-crease', '--model', 'policy'])
        with self.assertRaisesRegex(ValueError, 'Classic tactic'):
            runner.run(args)

    def test_configuration_rejects_non_classic_cross_crease_and_supports_teachers(self):
        from nhl94_ai.config import EnvironmentConfig
        with self.assertRaisesRegex(ValueError, 'Cross-crease'):
            EnvironmentConfig.from_args(SimpleNamespace(env='NHL94-Genesis-v0', cross_crease=True))
        config = EnvironmentConfig.from_args(SimpleNamespace(
            env='NHL94-Genesis-v0', agent='classic', nn='MlpPolicy', cross_crease=True))
        self.assertEqual(config.env, 'NHL94-Genesis-v0')


if __name__ == '__main__':
    unittest.main()

"""Skating-deke selection, observed reactions, native shot edges and CLI contracts."""
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from nhl94_ai.agents.classic_v1 import ClassicAIV1Model
from nhl94_ai.agents.deke import (
    BAIT_FRAMES, SHOT_HOLD_FRAMES, SKATING_FRAMES, DekeController, DekePlan, evaluate_deke,
    _advance_preview, preview_deke,
)
from nhl94_ai.agents.registry import create_scripted
from nhl94_ai.config import EnvironmentConfig
from nhl94_ai.game.constants import GameConsts as Buttons
from tests.unit.test_cross_crease import crossing_state


def deke_state():
    state = crossing_state()
    player = state.team1.players[0]
    player.x, player.y, player.motion_x, player.motion_y, player.facing = 0, 180, 0, 0.2, 0
    state.puck.x, state.puck.y = 0, 188
    state.team2.goalie.x = 0
    for team in (state.team1, state.team2):
        for other in (*team.players, team.goalie):
            other.precise_x, other.precise_y = float(other.x), float(other.y)
            other.facing_phase, other.sprite_flipped_x = float(other.facing), False
    player.shot_projection_offsets_x, player.shot_projection_offsets_y = (-24, 24), (-24, 24)
    player.shot_offsets_x, player.shot_offsets_y = (-6, 6), (12, 20)
    player.shot_durations = player.shot_projection_durations = (4,) * 8
    goalie = state.team2.goalie
    goalie.decision_timer, goalie.decision_interval, goalie.steering = 0, 2, 8
    goalie.cpu_tracking_active = True
    return state


def controller(state):
    plan = DekePlan(0, -1, (-28, 188), (32, 206), 60, 50, 60, 1)
    deke = DekeController()
    deke.start(plan, state, 0)
    return deke


class DekeSelectionTests(unittest.TestCase):
    def test_unreachable_finish_is_rejected_before_live_execution(self):
        plan, diagnostics = evaluate_deke(deke_state(), {})
        self.assertIsNone(plan)
        self.assertEqual(diagnostics['status'], 'no-safe-deke')
        self.assertTrue(all(row['status'] != 'feasible' for row in diagnostics['candidates']))

    def test_preview_runs_the_same_phases_and_full_release_gate_as_execution(self):
        state = deke_state()
        state.team2.goalie.x = state.team2.goalie.precise_x = 18
        state.team2.goalie.decision_interval = 32
        plan, diagnostics = evaluate_deke(state, {})
        self.assertIsNotNone(plan, diagnostics)
        finish, forecast = preview_deke(state, plan, {})
        live = DekeController()
        scene = deepcopy(state)
        live.start(plan, scene, 0)
        for frame in range(finish.frames + 1):
            action = live.step(scene, frame)
            if frame < finish.frames:
                self.assertFalse(action[Buttons.INPUT_C])
                _advance_preview(scene, plan.slot, action)
        self.assertTrue(action[Buttons.INPUT_C])
        self.assertGreaterEqual(scene.team1.players[0].y, 202)
        self.assertEqual(live.event['timeline'], forecast['timeline'])
        self.assertEqual(live.diagnostics['release_x_bounds'], forecast['release_x_bounds'])
        self.assertEqual(live.event['hold_frames'], forecast['hold_frames'])

    def test_stationary_unresponsive_goalie_is_not_teleported_into_an_opening(self):
        with patch('nhl94_ai.agents.deke._goalie_step', side_effect=lambda state: deepcopy(state.team2.goalie)):
            plan, diagnostics = evaluate_deke(deke_state(), {})
        self.assertIsNone(plan)
        self.assertTrue(all(row['status'] == 'bait-timeout' for row in diagnostics['candidates']))

    def test_both_ends_and_directions_consider_bounded_mirrored_routes(self):
        for attack in (-1, 1):
            for start_x in (-20, 20):
                state = deke_state()
                state.team1.net.y, state.team2.net.y = -attack * 264, attack * 264
                player = state.team1.players[0]
                player.x, player.y, player.motion_y, player.facing = start_x, attack * 180, attack * 0.6, 0 if attack > 0 else 4
                state.puck.x, state.puck.y = start_x, attack * 188
                state.team2.goalie.y = attack * 250
                for other in (*state.team1.players[1:], *state.team2.players):
                    other.y = -attack * 180
                for team in (state.team1, state.team2):
                    for other in (*team.players, team.goalie):
                        other.precise_x, other.precise_y = float(other.x), float(other.y)
                        other.facing_phase = float(other.facing)
                with self.subTest(attack=attack, x=start_x):
                    plan, diagnostics = evaluate_deke(state, {})
                    self.assertEqual({row['direction'] for row in diagnostics['candidates']}, {-1, 1})
                    self.assertTrue(all(row['frames'] <= SKATING_FRAMES for row in diagnostics['candidates']))
                    self.assertTrue(all(abs(row['target'][1]) >= 202 for row in diagnostics['candidates']))
                    if plan is not None:
                        self.assertEqual(plan.aim_side, -plan.direction)

    def test_nearby_trailing_defender_is_not_a_breakaway_requirement(self):
        state = deke_state()
        defender = state.team2.players[0]
        defender.x, defender.y, defender.motion_y = 35, 145, 0.4
        _, diagnostics = evaluate_deke(state, {})
        self.assertEqual(len(diagnostics['candidates']), 4)
        self.assertNotEqual(diagnostics['status'], 'ineligible-state')

    def test_imminent_obstacle_and_goalie_contact_reject(self):
        for other in ('defender', 'goalie'):
            state = deke_state()
            obstacle = state.team2.players[0] if other == 'defender' else state.team2.goalie
            obstacle.x, obstacle.y = 0, 190
            with self.subTest(other=other):
                self.assertIsNone(evaluate_deke(state, {})[0])

    def test_better_available_finish_or_pass_wins(self):
        for option in ('shoot', 'carry', 'one-timer', 'position-pass'):
            with self.subTest(option=option):
                plan, diagnostics = evaluate_deke(deke_state(), {option: 100})
                self.assertIsNone(plan)
                self.assertIn(diagnostics['status'], ('better-alternative', 'no-safe-deke'))

    def test_open_shot_does_not_get_overplayed(self):
        state = deke_state()
        state.team1.players[0].x, state.team1.players[0].y = -24, 208
        state.puck.x, state.puck.y = -24, 216
        state.team2.goalie.x = 28
        self.assertEqual(evaluate_deke(state, {})[1]['status'], 'shot-already-open')

    def test_missing_feedback_and_shootouts_keep_normal_finishing(self):
        for field in ('motion_x', 'motion_y', 'shot_power', 'handedness', 'live_anim', 'live_anim_frame'):
            state = deke_state()
            setattr(state.team1.players[0], field, None)
            with self.subTest(field=field):
                self.assertEqual(evaluate_deke(state, {})[1]['status'], 'missing-feedback')
        state = deke_state()
        state.is_shootout_active = True
        self.assertEqual(evaluate_deke(state, {})[1]['status'], 'ineligible-state')

    def test_evaluation_does_not_mutate_live_actors(self):
        state = deke_state()
        before = deepcopy(state)
        evaluate_deke(state, {})
        for team, original in ((state.team1, before.team1), (state.team2, before.team2)):
            for player, previous in zip((*team.players, team.goalie), (*original.players, original.goalie)):
                self.assertEqual(vars(player), vars(previous))
        self.assertEqual(vars(state.puck), vars(before.puck))


class DekeExecutionTests(unittest.TestCase):
    def test_bait_requires_observed_movement_before_cutting(self):
        state = deke_state()
        deke = controller(state)
        direction = deke.plan.direction
        state.team2.goalie.x = direction * 8
        deke.step(state, 1)
        self.assertEqual(deke.phase, 'bait')
        state.team1.players[0].x = direction * 16
        state.puck.x = direction * 16
        action = deke.step(state, 2)
        self.assertEqual(deke.phase, 'cut')
        self.assertEqual(deke.metrics['observed_tracking'], 1)
        self.assertFalse(action[Buttons.INPUT_C])
        self.assertEqual(deke.diagnostics['waypoint'], deke.plan.target)

    def test_unresponsive_goalie_has_a_bounded_fallback(self):
        state = deke_state()
        deke = controller(state)
        self.assertFalse(np.any(deke.step(state, BAIT_FRAMES)))
        self.assertIsNone(deke.plan)
        self.assertEqual(deke.events[-1]['outcome'], 'bait-timeout')
        self.assertGreater(deke.retry_at, BAIT_FRAMES)

    def test_changed_route_is_aborted_not_silently_redirected(self):
        state = deke_state()
        deke = controller(state)
        with patch('nhl94_ai.agents.deke._path_clear', return_value=(False, 'occupied-crossing', 0, 0)):
            self.assertFalse(np.any(deke.step(state, 1)))
        self.assertEqual(deke.events[-1]['outcome'], 'route-invalidated')
        self.assertEqual(deke.diagnostics['reason'], 'occupied-crossing')

    def test_fresh_c_press_is_separate_from_windup_and_attributed_release(self):
        state = deke_state()
        deke = controller(state)
        deke.phase = 'cut'
        deke.event['bait_position'] = (deke.plan.direction * 12, 180)
        details = {'release_point': (0, 208), 'status': 'open'}
        with patch('nhl94_ai.agents.deke.quick_finish', return_value=(True, details)), \
                patch('nhl94_ai.agents.deke.shot_value', return_value=100):
            self.assertTrue(deke.step(state, 1)[Buttons.INPUT_C])
        self.assertFalse(deke.event['windup'])
        state.is_shooting = True
        deke.observe(state, 2)
        self.assertEqual(deke.metrics['accepted_windups'], 1)
        self.assertTrue(deke.step(state, 2, c_down=True)[Buttons.INPUT_C])
        self.assertFalse(deke.step(state, 1 + SHOT_HOLD_FRAMES, c_down=True)[Buttons.INPUT_C])
        state.engine.puck_owner, state.engine.shot_player, state.engine.last_puck_player = -1, 0, 0
        state.is_shooting = False
        deke.observe(state, 6)
        self.assertTrue(deke.event['released'])
        self.assertEqual(deke.metrics['observed_releases'], 1)
        self.assertFalse(deke.event['shot'])
        state.team1.stats.shots += 1
        deke.observe(state, 7)
        self.assertEqual(deke.metrics['recorded_shots'], 1)
        state.team1.stats.score += 1
        deke.observe(state, 8)
        self.assertEqual(deke.events[-1]['outcome'], 'goals')

    def test_post_shot_unavailability_does_not_discard_native_evidence(self):
        state = deke_state()
        deke = controller(state)
        deke.phase = 'release'
        deke.event['pressed'] = deke.event['windup'] = True
        state.team1.players[0].unavailable = 4
        state.team1.defense_control = -1
        state.engine.puck_owner, state.engine.shot_player, state.engine.last_puck_player = -1, 0, 0
        deke.observe(state, 3)
        self.assertIsNotNone(deke.plan)
        self.assertTrue(deke.event['released'])
        self.assertFalse(np.any(deke.step(state, 3)))

    def test_other_shooters_do_not_get_credit_and_stale_impacts_do_not_count(self):
        state = deke_state()
        state.team2.goalie.contact_player, state.team2.goalie.contact_impact = 0, 4
        deke = controller(state)
        deke.event['pressed'] = True
        state.team1.stats.shots += 1
        state.engine.shot_player = 1
        deke.observe(state, 1)
        self.assertFalse(deke.event['shot'])
        self.assertFalse(deke.event['contact'])
        state.team2.goalie.contact_impact = 5
        deke.observe(state, 2)
        deke.observe(state, 3)
        self.assertEqual(deke.metrics['goalie_contacts'], 1)

    def test_unaccepted_presses_have_bounded_explicit_retries(self):
        state = deke_state()
        deke = controller(state)
        deke.event['pressed'] = True
        for retry in range(3):
            deke.phase = 'release'
            deke.pressed = retry * 10
            deke.step(state, deke.pressed + SHOT_HOLD_FRAMES + 3)
            if retry < 2:
                self.assertEqual(deke.phase, 'cut')
        self.assertIsNone(deke.plan)
        self.assertEqual(deke.metrics['unaccepted_presses'], 3)
        self.assertEqual(deke.events[-1]['outcome'], 'shot-not-started')

    def test_another_carriers_windup_does_not_confirm_our_c_request(self):
        state = deke_state()
        deke = controller(state)
        deke.event['pressed'] = True
        deke.phase = 'hold'
        state.is_shooting = True
        state.engine.puck_owner, state.engine.shot_player = 6, 6
        deke.observe(state, 1)
        self.assertEqual(deke.metrics.get('accepted_windups', 0), 0)
        self.assertEqual(deke.events[-1]['outcome'], 'possession-lost')


class ClassicDekeContracts(unittest.TestCase):
    def test_disabled_actions_and_episode_reset_are_unchanged(self):
        state = deke_state()
        default, disabled = ClassicAIV1Model(), ClassicAIV1Model(SimpleNamespace(deke=False))
        for _ in range(16):
            np.testing.assert_array_equal(default.predict_frame(state), disabled.predict_frame(state))
        self.assertIsNone(default.deke)
        agent = create_scripted('classic', SimpleNamespace(deke=True, action_type='FILTERED'))
        agent.deke.start(controller(state).plan, state, 0)
        agent.reset()
        self.assertIsNone(agent.deke.plan)
        self.assertFalse(agent.deke.metrics)

    def test_turnover_and_control_changes_release_buttons_next_frame(self):
        for change in ('turnover', 'control'):
            state = deke_state()
            model = ClassicAIV1Model(SimpleNamespace(deke=True))
            model.deke.start(controller(state).plan, state, 0)
            model.predict_frame(state, 10)
            if change == 'turnover':
                state.engine.puck_owner = 6
            else:
                state.team1.defense_control = 1
            self.assertFalse(np.any(model.predict_frame(state, 10)))
            self.assertIsNone(model.deke.plan)

    def test_active_execution_uses_emulator_frames_at_each_interval(self):
        for schema in ('FILTERED', 'HOCKEY_INTENT_DPAD'):
            outcomes = []
            for interval in (1, 4, 10):
                state = deke_state()
                model = ClassicAIV1Model(SimpleNamespace(deke=True, action_type=schema))
                model.deke.start(controller(state).plan, state, 0)
                for _ in range(BAIT_FRAMES + 1):
                    model.predict_frame(state, interval)
                outcomes.append(model.deke.events[-1]['end_frame'])
            self.assertEqual(outcomes, [BAIT_FRAMES] * 3)

    def test_all_classic_entry_points_expose_the_opt_in(self):
        from nhl94_ai.evaluation import cpu_benchmark, play, runner
        from nhl94_ai.training import collect, dagger
        for module in (cpu_benchmark, play, runner, collect, dagger):
            parser = module.build_parser()
            self.assertFalse(parser.get_default('deke'))
            arguments = ['--deke'] + (['--model', 'policy', '--base_datasets', 'data'] if module is dagger else [])
            self.assertTrue(parser.parse_known_args(arguments)[0].deke)

    def test_incompatible_configuration_is_rejected(self):
        for options in ({'action_type': 'TARGET_POSITION'}, {'env': 'NHL941on1-Genesis-v0'}):
            with self.assertRaisesRegex(ValueError, 'Deke'):
                ClassicAIV1Model(SimpleNamespace(deke=True, **options))
        with self.assertRaisesRegex(ValueError, 'Deke'):
            EnvironmentConfig.from_args(SimpleNamespace(env='NHL94-Genesis-v0', deke=True))
        config = EnvironmentConfig.from_args(SimpleNamespace(
            env='NHL94-Genesis-v0', agent='classic', nn='MlpPolicy', deke=True))
        self.assertEqual(config.env, 'NHL94-Genesis-v0')
        from nhl94_ai.evaluation import runner
        args = runner.build_parser().parse_args(['--deke', '--model', 'policy'])
        with self.assertRaisesRegex(ValueError, 'Classic tactic'):
            runner.run(args)

    def test_cpu_trials_preserve_deke_options_and_aggregate_events(self):
        from nhl94_ai.evaluation import cpu_benchmark
        fixtures = []
        def match(fixture):
            fixtures.append(fixture)
            return dict(matchup=fixture[1], side=1, completed=True, goals=[0, 0],
                        one_timers=[0, 0], one_timer_goals=[0, 0], decisions={},
                        deke_metrics={'plans': 3, 'attempts': 1})
        args = cpu_benchmark.build_parser().parse_args(['--deke', '--trials', '1'])
        with patch.object(cpu_benchmark, 'cpu_match', side_effect=match), patch('builtins.print'):
            report = cpu_benchmark.run(args)
        self.assertTrue(all(fixture[-2:] == (False, True) for fixture in fixtures))
        self.assertTrue(report['settings']['deke'])
        for row in report['summary'].values():
            self.assertEqual(row['deke_metrics'], {'plans': 3, 'attempts': 1})

    def test_deke_and_cross_crease_compare_values_without_overlapping_sequences(self):
        from nhl94_ai.agents.cross_crease import evaluate_cross_crease
        crossing, _ = evaluate_cross_crease(crossing_state(), {})
        plan = controller(deke_state()).plan
        diagnostics = {'status': 'selected', 'candidates': []}
        for value, expected in ((crossing.value - 1, 'cross_crease'), (crossing.value + 1, 'deke')):
            model = ClassicAIV1Model(SimpleNamespace(deke=True, cross_crease=True))
            with patch('nhl94_ai.agents.classic_v1.evaluate_cross_crease',
                       return_value=(crossing, dict(diagnostics))), \
                    patch('nhl94_ai.agents.classic_v1.evaluate_deke',
                          return_value=(replace(plan, value=value), dict(diagnostics))):
                self.assertEqual(model._choose_finisher(deke_state(), None, None), expected)
            self.assertEqual(model.deke.plan is not None, expected == 'deke')
            self.assertEqual(model.cross_crease.plan is not None, expected == 'cross_crease')

    def test_paired_benchmark_label_retains_the_same_one_timer_controller(self):
        from nhl94_ai.evaluation.benchmark import make_agent
        candidate, baseline = make_agent('classic-v1-deke'), make_agent('classic-v1')
        self.assertIsNotNone(candidate.deke)
        self.assertIsNone(baseline.deke)
        self.assertTrue(candidate._one_timers)
        self.assertIs(type(candidate.controller), type(baseline.controller))


if __name__ == '__main__':
    unittest.main()

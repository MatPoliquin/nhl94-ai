"""Coordinated traps, safety exits and role-specific defensive execution."""
from copy import deepcopy
from dataclasses import replace
import unittest
from unittest.mock import patch

from nhl94_ai.agents.classic_v1 import ClassicAIV1Model
from nhl94_ai.agents.defense import DefenseController
from nhl94_ai.agents.teammate_pressure import SharedPressure
from nhl94_ai.agents.teammates import TeammateCoordinator
from nhl94_ai.evaluation.pressure_metrics import PressureMetrics
from nhl94_ai.game.state import NHL94GameState
from nhl94_ai.game.constants import GameConsts as Buttons
from tests.unit.test_classic_defense import defense_state
from tests.unit.test_teammates import team_args


def pressure_state():
    state = defense_state()
    state.time = 300
    state.engine.clock_stopped = False
    state.engine.puck_owner_known = True
    state.team1.goalie.x, state.team1.goalie.y = 0, -264
    state.team2.goalie.x, state.team2.goalie.y = 0, 264
    for team in (state.team1, state.team2):
        for player in team.players:
            player.role = -1
            player.selection_flags = 8
    for i, (x, y, facing) in enumerate(((-24, -24, 1), (24, 24, 5), (0, -72, 0))):
        player = state.team1.players[i]
        player.x, player.y, player.facing, player.role = x, y, facing, 1
    carrier = state.team2.players[0]
    carrier.x = carrier.y = state.puck.x = state.puck.y = 0
    carrier.role, carrier.facing, carrier.selection_flags = 1, 4, 0x40
    return state


class SharedPressureTests(unittest.TestCase):
    def test_three_bots_reserve_cover_and_approach_opposite_sides(self):
        state, planner = pressure_state(), SharedPressure()
        before = deepcopy(state)
        result = planner.choose(state, [0, 1, 2], 1)
        self.assertEqual([a.plan.mode for a in result], ['contain', 'close', 'cover'])
        self.assertLess(result[0].plan.target[1], state.puck.y)
        self.assertGreater(result[1].plan.target[1], state.puck.y)
        self.assertLess(result[0].plan.target[0] * result[1].plan.target[0], 0)
        self.assertEqual(sum(a.contact for a in result), 1)
        self.assertFalse(result[2].contact)
        self.assertEqual(vars(state.team1.players[0]), vars(before.team1.players[0]))
        self.assertEqual(vars(state.team1.players[2]), vars(before.team1.players[2]))

    def test_assignments_follow_geometry_not_controller_number(self):
        result = SharedPressure().choose(pressure_state(), [2, 1, 0], 1)
        self.assertEqual([a.plan.mode for a in result], ['cover', 'close', 'contain'])

    def test_deep_cover_does_not_take_nearest_attacker_out_of_the_trap(self):
        state = pressure_state()
        state.team1.players[2].y = -150
        result = SharedPressure().choose(state, [0, 1, 2], 1)
        self.assertEqual([a.plan.mode for a in result], ['contain', 'close', 'cover'])
        self.assertEqual(result[2].plan.target[1], -150)

    def test_staged_approach_withholds_contact_until_both_attackers_are_close(self):
        state, planner = pressure_state(), SharedPressure()
        for player, (x, y) in zip(state.team1.players, ((-40, -55), (38, 55), (0, -110))):
            player.x, player.y = x, y
        result = planner.choose(state, [0, 1, 2], 1)
        self.assertIsNotNone(result)
        self.assertEqual(planner.diagnostics['phase'], 'contain')
        self.assertFalse(any(a.contact for a in result))

    def test_two_bots_require_remaining_cover(self):
        state = pressure_state()
        self.assertIsNotNone(SharedPressure().choose(state, [0, 1], 1))
        state.team1.players[2].role = -1
        planner = SharedPressure()
        self.assertIsNone(planner.choose(state, [0, 1], 1))
        self.assertEqual(planner.diagnostics['reason'], 'exposed-lane')

    def test_cpu_cover_that_drifts_away_cannot_license_a_trap(self):
        state = pressure_state()
        state.team1.players[2].motion_x = 2
        self.assertIsNone(SharedPressure().choose(state, [0, 1], 1))

    def test_open_receiver_and_breakaway_reject_trap_even_with_three(self):
        state = pressure_state()
        receiver = state.team2.players[1]
        receiver.role, receiver.x, receiver.y = 1, 60, -170
        self.assertIsNone(SharedPressure().choose(state, [0, 1, 2], 1))
        receiver.role = -1
        state.team2.players[0].motion_y = -4
        self.assertIsNone(SharedPressure().choose(state, [0, 1, 2], 1))

    def test_receiver_entering_open_lane_during_approach_rejects_trap(self):
        state = pressure_state()
        receiver = state.team2.players[1]
        receiver.role, receiver.x, receiver.y, receiver.motion_y = 1, 70, 5, -3
        self.assertIsNone(SharedPressure().choose(state, [0, 1, 2], 1))

    def test_board_trap_keeps_both_approaches_inside_rink(self):
        state = pressure_state()
        state.team2.players[0].x = state.puck.x = 100
        for p, (x, y) in zip(state.team1.players, ((76, -24), (94, 24), (65, -72))):
            p.x, p.y = x, y
        result = SharedPressure().choose(state, [0, 1, 2], 1)
        self.assertIsNotNone(result)
        self.assertLess(result[0].plan.target[0], 100)
        self.assertLess(result[1].plan.target[0], 100)
        self.assertLess(result[0].plan.target[1], 0)
        self.assertGreater(result[1].plan.target[1], 0)

    def test_obstructed_or_late_closer_does_not_commit(self):
        for point in ((-24, -24), (110, 150)):
            with self.subTest(point=point):
                state = pressure_state()
                state.team1.players[1].x, state.team1.players[1].y = point
                self.assertIsNone(SharedPressure().choose(state, [0, 1, 2], 1))

    def test_roles_stay_stable_but_safety_overrides_lease(self):
        state, planner = pressure_state(), SharedPressure()
        first = planner.choose(state, [0, 1, 2], 1)
        state.team1.players[0].x += 1
        second = planner.choose(state, [0, 1, 2], 2)
        self.assertEqual([a.plan.mode for a in first], [a.plan.mode for a in second])
        state.team1.players[2].is_falling = 1
        self.assertIsNone(planner.choose(state, [0, 1, 2], 3))

    def test_pass_loose_puck_selection_and_unknown_feedback_abort(self):
        for change in ('pass', 'loose', 'selection', 'unknown', 'stoppage'):
            with self.subTest(change=change):
                state, planner = pressure_state(), SharedPressure()
                self.assertIsNotNone(planner.choose(state, [0, 1, 2], 1))
                slots = [0, 1, 2]
                if change == 'pass':
                    state.engine.puck_owner = 7
                    state.team2.players[1].role = 1
                elif change == 'loose':
                    state.engine.puck_owner = -1
                elif change == 'selection':
                    slots = [1, 0, 2]
                elif change == 'unknown':
                    state.engine.puck_owner_known = False
                else:
                    state.engine.clock_stopped = True
                self.assertIsNone(planner.choose(state, slots, 2))

    def test_timeout_requires_regroup(self):
        state, planner = pressure_state(), SharedPressure()
        self.assertIsNotNone(planner.choose(state, [0, 1, 2], 1))
        self.assertIsNone(planner.choose(state, [0, 1, 2], 121))
        self.assertIsNone(planner.choose(state, [0, 1, 2], 140))
        self.assertIsNotNone(planner.choose(state, [0, 1, 2], 157))

    def test_mirrored_ends_have_mirrored_targets(self):
        state = pressure_state()
        first = SharedPressure().choose(state, [0, 1, 2], 1)
        for team in (state.team1, state.team2):
            team.net.y *= -1
            for player in (*team.players, team.goalie):
                player.y *= -1
                player.facing = (4 - player.facing) % 8 if player.facing is not None else None
        second = SharedPressure().choose(state, [0, 1, 2], 1)
        self.assertEqual([a.plan.mode for a in first], [a.plan.mode for a in second])
        self.assertEqual([(a.plan.target[0], -a.plan.target[1]) for a in first],
                         [a.plan.target for a in second])


class PressureExecutionTests(unittest.TestCase):
    def test_coordinator_runs_all_assignments_then_reassigns_loose_puck(self):
        state = pressure_state()
        team = TeammateCoordinator(team_args(team_size=3))
        team.states = [deepcopy(state) for _ in range(3)]
        for index, view in enumerate(team.states):
            view.team1.defense_control = index
        native = [dict(teammate_highlight=False, teammate_transient=False, p1_control_slot=i, time=300, sflags=0)
                  for i in range(3)]
        with patch('nhl94_ai.agents.teammates.teammate_controller_info',
                   side_effect=lambda info, port, *args, **kwargs: native[port - 1]), \
                patch.object(NHL94GameState, 'BeginFrame'), patch.object(NHL94GameState, 'EndFrame'):
            team.act({})
            self.assertEqual(team.last_roles, ['contain', 'close', 'cover'])
            self.assertEqual([c.scheduler.frames for c in team.controllers], [1, 1, 1])
            for view in team.states:
                view.engine.puck_owner = -1
                view.puck.x, view.puck.y = 24, 24
            team.act({})
            self.assertEqual(team.primary, 1)
            self.assertEqual(team.last_roles[1], 'retrieve')
            self.assertFalse(any(role in ('contain', 'close', 'cover') for role in team.last_roles))
            self.assertTrue(all(c.defense.pending_check is None for c in team.controllers))
            self.assertEqual([c.scheduler.frames for c in team.controllers], [2, 2, 2])

    def test_assigned_defender_cannot_switch_or_boost_out_of_assignment(self):
        state = pressure_state()
        assignment = SharedPressure().choose(state, [0, 1, 2], 1)[0]
        controller = DefenseController()
        with patch.object(controller, 'select_player', side_effect=AssertionError('independent switch')):
            action = controller.step(state, assignment=replace(assignment, contact=False))
        self.assertFalse(action[Buttons.INPUT_B] or action[Buttons.INPUT_C])
        self.assertIsNone(controller.pending_switch)

    def test_covered_rear_check_relaxes_only_goal_side_requirement(self):
        state = pressure_state()
        player, carrier = state.team1.players[1], state.team2.players[0]
        player.x, player.y, player.facing, player.weight = 0, 24, 4, 32
        state.team1.players[0].x = -60
        controller = DefenseController()
        self.assertEqual(controller._body_check(state, 1, player)['status'], 'not-goal-side')
        self.assertEqual(controller._body_check(state, 1, player, covered_closer=True)['status'], 'ready')
        player.energy = 0
        self.assertEqual(controller._body_check(state, 1, player, covered_closer=True)['status'], 'low-energy')
        player.energy = 4096
        state.team1.players[0].x, state.team1.players[0].y = 0, 20
        self.assertEqual(controller._body_check(state, 1, player, covered_closer=True)['status'], 'obstructed-contact')

    def test_executor_advances_once_and_preserves_contact_follow_through(self):
        state = pressure_state()
        assignment = SharedPressure().choose(state, [0, 1, 2], 1)[0]
        controller = ClassicAIV1Model()
        controller.predict_team_defense(state, assignment)
        self.assertEqual(controller.scheduler.frames, 1)
        self.assertEqual(controller.input_owner, 'team-defense')
        controller.defense.pending_check = {'from_slot': 0, 'carrier_slot': 6, 'until': 6}
        controller.predict_team_defense(state, assignment)
        self.assertEqual(controller.scheduler.frames, 2)
        self.assertEqual(controller.defense.diagnostics['mode'], 'check-follow-through')
        controller.cancel_actions('trap-ended')
        self.assertIsNone(controller.defense.pending_check)


class PressureMetricsTests(unittest.TestCase):
    def test_counts_observed_recovery_after_trap(self):
        state, metrics = pressure_state(), PressureMetrics()
        active = {'pressure': {'phase': 'close'}, 'controllers': [
            {'slot': i, 'role': role} for i, role in enumerate(('contain', 'close', 'cover'))]}
        metrics.observe(1, state, active)
        state.engine.puck_owner = -1
        metrics.observe(2, state, {'pressure': {'phase': 'idle', 'reason': 'loose-puck'}})
        state.engine.puck_owner = 1
        metrics.observe(3, state, {})
        self.assertEqual(metrics.summary()['counts']['attempts'], 1)
        self.assertEqual(metrics.summary()['counts']['team_recoveries_within_60_frames'], 1)
        self.assertEqual(len(metrics.events), 1)

    def test_shots_after_an_escape_still_count_in_aftermath_window(self):
        state, metrics = pressure_state(), PressureMetrics()
        metrics.observe(1, state, {'pressure': {'phase': 'close'}, 'controllers': []})
        state.engine.puck_owner = 7
        metrics.observe(2, state, {'pressure': {'phase': 'idle'}})
        state.team2.stats.shots += 1
        metrics.observe(20, state, {})
        self.assertEqual(metrics.summary()['counts']['opponent_receptions_within_60_frames'], 1)
        self.assertEqual(metrics.summary()['counts']['opponent_shots_during_or_after_trap'], 1)


if __name__ == '__main__':
    unittest.main()

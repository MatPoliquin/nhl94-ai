"""Ownership, action priority and tactical variant regression cases."""
import unittest

from nhl94_ai.agents.teammate_faceoff import FaceoffInput
from nhl94_ai.game.constants import GameConsts as Buttons
from nhl94_ai.game.state import NHL94GameState
from tests.support.teammates import team_info


def draw_state(side=1, reverse=False):
    state = NHL94GameState(5)
    if side == 2:
        state.Flip()
    state.BeginFrame(team_info(), [0] * 6)
    state.time = 120
    state.engine.clock_stopped = True
    state.engine.puck_owner = -1
    state.engine.sflags = 0
    state.puck.x, state.puck.y = 0, 0
    for team, y in ((state.team1, -15), (state.team2, 15)):
        for player in team.players:
            player.assignment, player.role = 0x16, 3
        player = team.players[2]
        player.assignment, player.role, player.x, player.y = 0x17, 4, 0, y
        player.unavailable, player.is_falling = 0, False
    if reverse:
        state.team1.net.y *= -1
        state.team2.net.y *= -1
    return state


class FaceoffOwnershipTests(unittest.TestCase):
    def test_every_controller_and_both_sides_release_press_and_direction(self):
        for side in (1, 2):
            for reverse in (False, True):
                for count in (2, 3):
                    for index in range(count):
                        state = draw_state(side, reverse)
                        base = state.team1.skater_scnum_base()
                        slots = [base, base + 1, base + 3][:count]
                        slots[index] = base + 2
                        policy = FaceoffInput(draw_back=True)
                        owner, action, center = policy.choose(state, slots, 1)
                        self.assertEqual((owner, center), (index, base + 2))
                        self.assertTrue(action[Buttons.INPUT_B])
                        actions = [policy.choose(state, slots, f)[1] for f in range(2, 10)]
                        self.assertEqual([int(a[Buttons.INPUT_B]) for a in actions], [1]*2 + [0]*4 + [1]*2)
                        direction = Buttons.INPUT_UP if state.team1.net.y > 0 else Buttons.INPUT_DOWN
                        self.assertTrue(all(a[direction] for a in actions))

    def test_cpu_center_transient_and_unrelated_stoppages_stay_neutral(self):
        state = draw_state()
        policy = FaceoffInput()
        for slots, transient in (([0, 1], False), ([-1, -256], False), ([2, 0], True)):
            self.assertFalse(policy.choose(state, slots, 2, transient=transient)[1].any())
        state.team1.players[2].assignment = 0x16
        self.assertIsNone(policy.choose(state, [2, 0], 3)[2])
        state.team1.players[2].assignment = 0x17
        state.engine.puck_owner = 5
        self.assertIsNone(policy.choose(state, [2, 0], 4)[2])


class CpuCarrierTests(unittest.TestCase):
    def test_context_requires_known_selections_and_friendly_skater(self):
        from nhl94_ai.agents.teammate_context import TeamControlContext
        state = draw_state()
        state.engine.puck_owner_known = True
        context = TeamControlContext((0, 1), True, (45, 76))
        for owner, expected in ((2, True), (0, False), (5, False), (6, False), (-1, False)):
            state.engine.puck_owner = owner
            self.assertEqual(context.cpu_carrier(state), expected)
        state.engine.puck_owner = 2
        self.assertFalse(TeamControlContext((-1, 1), False, (0, 0)).cpu_carrier(state))

    def test_explicit_context_supports_cpu_carrier_without_switching(self):
        from nhl94_ai.agents.teammates import TeammateCoordinator
        from tests.support.teammates import team_args
        for side in ('home', 'away'):
            base = 0 if side == 'home' else 6
            physical = 1 if side == 'home' else 2
            ai = TeammateCoordinator(team_args(side=side))
            info = team_info(defense_team1=physical, defense_team2=physical,
                             defense_control1=base, defense_control2=base+1, puck_owner=base+2)
            result = ai.act(info)
            controller = ai.controllers[ai.primary]
            self.assertEqual(controller._last_decision, 'cpu-carrier-support')
            self.assertFalse(result.action[Buttons.INPUT_B::12].any())
            self.assertTrue(all(c.team_context.known for c in ai.controllers))
            ai.act({**info, 'puck_owner': base})
            self.assertFalse(ai.controllers[0].team_context.cpu_carrier(ai.states[0]))

    def test_single_mode_label_does_not_enable_context_and_history_preserves_it(self):
        from nhl94_ai.agents.classic_v1 import ClassicAIV1Model
        from nhl94_ai.agents.teammate_context import TeamControlContext
        from tests.support.teammates import team_args
        model = ClassicAIV1Model(team_args())
        self.assertIsNone(model.team_context)
        model.team_context = TeamControlContext((0, 1), True, (45, 76))
        self.assertEqual(model.fork().team_context, model.team_context)


class GoalieOwnershipTests(unittest.TestCase):
    def test_only_selected_goalie_can_request_outlet_on_either_controller(self):
        from unittest.mock import patch
        import numpy as np
        from nhl94_ai.agents.teammates import TeammateCoordinator
        from tests.support.teammates import team_args
        for side in ('home', 'away'):
            base, physical = (0, 1) if side == 'home' else (6, 2)
            for index in (0, 1):
                ai = TeammateCoordinator(team_args(side=side))
                slots = [base, base+1]
                slots[index] = base+5
                info = team_info(defense_team1=physical, defense_team2=physical,
                                 defense_control1=slots[0], defense_control2=slots[1], puck_owner=base+5)
                calls = []
                def outlet(_state, action, calls=calls):  # pylint: disable=dangerous-default-value
                    calls.append(_state.team1.controlled_scnum())
                    action[Buttons.INPUT_B] = 1
                    return action, 0
                with patch.object(ai.controllers[index], '_goalie_outlet', side_effect=outlet):
                    result = ai.act(info)
                self.assertEqual(calls, [base+5])
                self.assertEqual(result.action[index*12+Buttons.INPUT_B], 1)
                self.assertIsNone(ai.controllers[1-index].pass_intention())
                self.assertFalse(result.action[(1-index)*12+Buttons.INPUT_B])

    def test_unselected_goalie_does_not_create_skater_commitments(self):
        from nhl94_ai.agents.teammates import TeammateCoordinator
        from tests.support.teammates import team_args
        for strategy in ('coordinated', 'independent'):
            ai = TeammateCoordinator(team_args(), strategy=strategy)
            result = ai.act(team_info(puck_owner=5))
            self.assertFalse(result.action[Buttons.INPUT_B::12].any())
            self.assertTrue(all(c.pass_intention() is None for c in ai.controllers))

    def test_provisional_goalie_target_never_reserves_receiver(self):
        from nhl94_ai.agents.teammate_receiving import SharedReceiving
        shared, state = SharedReceiving(), draw_state()
        shared.propose({'passer': 5, 'receiver': 0}, 1, [5, 0], state, 1)
        self.assertIsNone(shared.pending)


class OffPuckContractTests(unittest.TestCase):
    def test_execution_matches_forecast_pad_and_never_bursts(self):
        import numpy as np
        from nhl94_ai.agents.teammate_offpuck import movement_action
        from nhl94_ai.agents.teammate_support import support_forecast
        from nhl94_ai.game.skating import grounded_step
        from tests.support.offense import offense_state
        state = offense_state()
        slot = state.team1.skater_scnum_base() + 1
        player = state.team1.players[1]
        for coast in (False, True):
            for target in ((-50, 76), (30, 200), (player.x, player.y)):
                action, _ = movement_action(player, target, coast=coast)
                self.assertFalse(action[[Buttons.INPUT_B, Buttons.INPUT_C]].any())
                pad = (int(action[Buttons.INPUT_RIGHT])-int(action[Buttons.INPUT_LEFT]),
                       int(action[Buttons.INPUT_UP])-int(action[Buttons.INPUT_DOWN]))
                expected = grounded_step(player, pad)
                actual = support_forecast(state, slot, target, coast=coast, frames=1).team1.players[1]
                np.testing.assert_allclose((actual.x, actual.y), (expected.x, expected.y))

    def test_scored_cut_target_is_owned_for_forecast_window_and_invalidates_on_turnover(self):
        from nhl94_ai.agents.teammate_support import SupportPlanner
        from tests.support.offense import offense_state
        state = offense_state()
        state.engine.puck_owner = 0
        planner = SupportPlanner()
        first = planner.choose(state, 1, 1, 0, movement=True)
        self.assertEqual(planner.until, 13)
        self.assertEqual(first, planner.choose(state, 1, 12, 0, movement=True))
        state.engine.puck_owner = 6
        self.assertEqual(planner.choose(state, 1, 12, 0, movement=True)[1], 'coverage')
        self.assertFalse(planner.coast)


class ArbitrationTests(unittest.TestCase):
    def test_poke_does_not_consume_shared_switch_lease(self):
        from nhl94_ai.agents.teammate_arbitration import BArbitrator
        gate = BArbitrator(3)
        args = dict(frame=10, previous=False, owns_puck=False, known=True, protected=False)
        self.assertTrue(gate.allow(intent='poke', controller=0, likely_keeps_selection=True, **args)[0])
        self.assertTrue(gate.allow(intent='switch', controller=1, **args)[0])
        self.assertFalse(gate.allow(intent='switch', controller=2, **args)[0])
        self.assertTrue(gate.allow(intent='poke', controller=2, likely_keeps_selection=True, **args)[0])

    def test_protection_transients_native_poke_meaning_and_release_handoff(self):
        from nhl94_ai.agents.teammate_arbitration import BArbitrator
        base = dict(frame=1, controller=0, previous=False, owns_puck=False, known=True, protected=False)
        for change in ({'known': False}, {'protected': True}, {'owns_puck': True}, {'previous': True}):
            self.assertFalse(BArbitrator(2).allow(intent='switch', **{**base, **change})[0])
        self.assertFalse(BArbitrator(2).allow(intent='poke', **base)[0])
        self.assertTrue(BArbitrator(2).allow(intent='pass', **{**base, 'owns_puck': True, 'protected': True})[0])
        self.assertFalse(BArbitrator(2).allow(intent='goalie-outlet', **base)[0])


class SharedRoleTests(unittest.TestCase):
    def test_safety_rotates_by_position_and_excludes_reserved_input(self):
        from nhl94_ai.agents.teammate_offpuck import SharedRoles
        from tests.support.offense import offense_state
        state = offense_state()
        state.engine.puck_owner = -1
        state.team1.players[1].x, state.team1.players[1].y = 0, -130
        state.team1.players[2].x, state.team1.players[2].y = 100, 210
        roles = SharedRoles()
        chosen = roles.choose(state, [0, 1, 2], 0, 1)
        self.assertEqual(set(chosen), {1})
        self.assertEqual(chosen[1][1], 'safety-cover')
        self.assertEqual(set(roles.choose(state, [0, 1, 2], 0, 2, reserved=(1,))), {2})
        self.assertEqual(roles.choose(state, [0, 1, 2], 0, 3, reserved=(1, 2)), {})

    def test_attack_outlet_stays_onside_and_turnover_invalidates_role(self):
        from nhl94_ai.agents.teammate_offpuck import SharedRoles
        from tests.support.offense import offense_state
        state = offense_state()
        state.engine.puck_owner = 0
        state.engine.offsides_enabled = True
        state.team1.players[0].y, state.puck.y = 50, 50
        for player in state.team2.players:
            player.y = 150
        roles = SharedRoles()
        chosen = roles.choose(state, [0, 1, 2], 0, 1)
        self.assertEqual(next(iter(chosen.values()))[1], 'high-slot-outlet')
        self.assertLess(next(iter(chosen.values()))[0][1], 88)
        state.engine.puck_owner = 6
        chosen = roles.choose(state, [0, 1, 2], 0, 2)
        self.assertEqual(next(iter(chosen.values()))[1], 'safety-cover')


class InputHandoffTests(unittest.TestCase):
    def test_animation_cut_window_is_not_priced_and_releases_after_native_completion(self):
        from nhl94_ai.agents.teammate_support import SupportPlanner
        from tests.support.offense import offense_state
        state, planner = offense_state(), SupportPlanner()
        state.team1.players[1].unavailable = 2
        self.assertEqual(planner.choose(state, 1, 1, 0, movement=True)[1], 'animation-hold')
        self.assertEqual(planner.diagnostics['candidates'], [])
        self.assertTrue(planner.coast)
        state.team1.players[1].unavailable = 0
        self.assertNotEqual(planner.choose(state, 1, 2, 0, movement=True)[1], 'animation-hold')

    def test_offpuck_acquisition_uses_applied_edges_and_preserves_valid_pass(self):
        from nhl94_ai.agents.teammates import TeammateCoordinator
        from nhl94_ai.agents.lifecycle import PassState
        from tests.support.teammates import team_args
        ai = TeammateCoordinator(team_args(teammate_support_options=True, teammate_offpuck=True))
        ai.act(team_info(puck_owner=0))
        controller = ai.controllers[1]
        self.assertFalse(controller.buttons.c_down)
        self.assertFalse(controller.buttons.b_down)
        ai.act(team_info(puck_owner=1))
        self.assertEqual(ai.primary, 1)
        self.assertEqual(controller.scheduler.frames, 2)
        controller.offense.pass_action.pending = PassState(1, 2, 2, 60)
        controller.scheduler.interrupt()
        ai.act(team_info(puck_owner=1))
        self.assertIsNotNone(controller.offense.pass_action.pending)
        self.assertNotEqual(controller.input_owner, 'external')

    def test_primary_variants_release_stale_selection_and_receiver_leases(self):
        from unittest.mock import patch
        from nhl94_ai.agents.teammates import TeammateCoordinator
        from tests.support.teammates import team_args
        from tests.support.offense import offense_state
        state = offense_state()
        state.engine.puck_owner = -1
        for policy in ('short-lease', 'lost-race'):
            ai = TeammateCoordinator(team_args(teammate_primary_policy=policy))
            with patch('nhl94_ai.agents.teammates.arrival_time', side_effect=[10, 30]):
                self.assertEqual(ai._choose_primary(state, [0, 1]), 0)
            ai.frames = 1
            with patch('nhl94_ai.agents.teammates.arrival_time', side_effect=[30, 2]):
                self.assertEqual(ai._choose_primary(state, [2, 1]), 1)
            state.engine.puck_owner = 2
            self.assertEqual(ai._choose_primary(state, [2, 1]), 0)
            state.engine.puck_owner = -1


class FinalOwnershipEdgeTests(unittest.TestCase):
    def test_loose_sentinel_is_not_a_selected_carrier(self):
        from nhl94_ai.agents.teammates import TeammateCoordinator
        from tests.support.teammates import team_args
        from tests.support.offense import offense_state
        ai, state = TeammateCoordinator(team_args()), offense_state()
        state.engine.puck_owner = -1
        self.assertIsNone(ai._choose_primary(state, [-1, -1]))

    def test_cpu_support_waits_for_puck_to_cross_blue_line_in_both_directions(self):
        from nhl94_ai.agents.classic_v1 import ClassicAIV1Model
        from nhl94_ai.agents.teammate_context import TeamControlContext
        from tests.support.teammates import team_args
        from tests.support.offense import offense_state
        import numpy as np
        for sign in (1, -1):
            state, model = offense_state(), ClassicAIV1Model(team_args())
            state.team1.net.y, state.team2.net.y = -264*sign, 264*sign
            state.engine.offsides_enabled = True
            state.puck.y = 85*sign
            model.team_context = TeamControlContext((0, 1), True, (45, 140*sign))
            model._team_support(state, np.zeros(12), state.team1.players[0])
            self.assertEqual(model._last_target, (45, 76*sign))


class RoleSpacingTests(unittest.TestCase):
    def test_guard_target_is_excluded_before_pass_scoring(self):
        import math
        from nhl94_ai.agents.teammate_support import SupportPlanner
        from tests.support.offense import offense_state
        state, planner = offense_state(), SupportPlanner()
        reserved = (-45, -90)
        planner.choose(state, 1, 1, 0, movement=True, reserved_targets=(reserved,))
        self.assertTrue(all(math.dist(candidate['point'], reserved) >= 24
                            for candidate in planner.diagnostics['candidates']))


class ThreeBotGoalieTests(unittest.TestCase):
    def test_each_goalie_controller_and_transient_stoppage_handoffs_on_both_sides(self):
        from unittest.mock import patch
        from nhl94_ai.agents.teammates import TeammateCoordinator
        from tests.support.teammates import team_args, three_info
        for side, base in (('home', 0), ('away', 6)):
            for controller_index in range(3):
                slots = [base, base+1, base+2]
                slots[controller_index] = base+5
                values = {f'defense_team{i+1}': 1 if base == 0 else 2 for i in range(3)}
                values.update({f'defense_control{i+1}': slot for i, slot in enumerate(slots)})
                values.update({f'defense_{slot}_flags': (8 if slot in slots else 0) | (64 if slot >= 6 else 0)
                               for slot in range(12)})
                info = three_info(**values, puck_owner=base+5)
                ai = TeammateCoordinator(team_args(side=side, team_size=3))
                def outlet(_state, action):
                    action[Buttons.INPUT_B] = 1
                    return action, 0
                with patch.object(ai.controllers[controller_index], '_goalie_outlet', side_effect=outlet):
                    result = ai.act(info)
                self.assertEqual(result.action[controller_index*12], 1)
                self.assertEqual(sum(result.action[0::12]), 1)
                self.assertTrue(all(c.pass_intention() is None for c in ai.controllers))
                stopped = ai.act({**info, 'goalie_clock_flags': 1})
                self.assertFalse(stopped.action.any())
                transient = ai.act({**info, 'defense_control2': slots[0]})
                self.assertTrue(transient.diagnostics['teammates']['transient'])
                self.assertFalse(transient.action.any())


class DrawCadenceTests(unittest.TestCase):
    def test_default_preserves_original_pulse_but_releases_a_held_handoff(self):
        import numpy as np
        state, policy = draw_state(), FaceoffInput()
        for frame in range(1, 17):
            _, action, _ = policy.choose(state, [2, 0], frame)
            self.assertEqual(action[Buttons.INPUT_B], frame % 8 < 4)
            self.assertFalse(action[4:8].any())
        previous = np.zeros((2, 12), dtype=np.int8)
        previous[1, Buttons.INPUT_B] = 1
        _, action, _ = policy.choose(state, [0, 2], 17, previous=previous)
        self.assertFalse(action.any())
        self.assertTrue(policy.choose(state, [0, 2], 18, previous=np.zeros((2, 12)))[1][Buttons.INPUT_B])

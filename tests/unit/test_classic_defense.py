"""Tactical destinations, controllable puck races and frame-level defense."""
from copy import deepcopy
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from nhl94_ai.agents.base import AgentInput, configure_scripted_frames
from nhl94_ai.agents.classic_v1 import ClassicAIV1Model
from nhl94_ai.agents.defense import DefenseController, DefensePlan, owns_puck
from nhl94_ai.agents.motion import arrival_time, boost_safe, puck_path, skating
from nhl94_ai.agents.registry import create_scripted
from nhl94_ai.env.target_control import project_target
from nhl94_ai.env.actions import HockeyActionController
from nhl94_ai.env.intents import HOCKEY_INTENT_CHANGE_PLAYER
from nhl94_ai.env.wrappers import StochasticFrameSkip
from nhl94_ai.game.constants import GameConsts as Buttons
from nhl94_ai.game.ram import register_defense_state
from nhl94_ai.game.state import NHL94GameState, Player
from tests.unit.test_environment_contracts import wrapped


def defense_state():
    state = NHL94GameState(5)
    state.team1.net.y, state.team2.net.y = -264, 264
    state.team1.control = state.team2.control = 1
    state.team1.defense_control, state.team2.defense_control = 0, 6
    state.engine.puck_owner = 6
    for team, y in ((state.team1, -215), (state.team2, 70)):
        for index, player in enumerate(team.players):
            player.x, player.y = -90 + index * 40, y
            player.motion_x = player.motion_y = 0
            player.speed = player.agility = player.stick = 20
            player.weight, player.energy, player.checking = 64, 4096, 20
            player.role, player.facing = 4, 0
    state.team1.players[0].x = 0
    state.team2.players[0].x, state.team2.players[0].y = 0, -150
    state.puck.x, state.puck.y = 0, -150
    state.puck.motion_x = state.puck.motion_y = 0
    state.puck.height, state.puck.motion_z = 0, 0
    return state


class DefenseMotionTests(unittest.TestCase):
    def test_stats_momentum_and_energy_change_arrival(self):
        player = defense_state().team1.players[0]
        target = (0, -80)
        base = arrival_time(player, target)
        player.motion_y = 1.5
        self.assertLess(arrival_time(player, target), base)
        player.motion_y = -1.5
        self.assertGreater(arrival_time(player, target), base)
        player.motion_y = 0
        player.speed = player.agility = 0
        player.weight, player.energy = 112, 1024
        self.assertGreater(arrival_time(player, target), base)

    def test_boost_needs_facing_and_stopping_room(self):
        player = defense_state().team1.players[0]
        self.assertTrue(boost_safe(player, (0, 0)))
        self.assertFalse(boost_safe(player, (0, -205)))
        self.assertLess(arrival_time(player, (0, 0), boost=True), arrival_time(player, (0, 0)))
        player.facing = 4
        self.assertFalse(boost_safe(player, (0, 0)))

    def test_unknown_attributes_are_conservative_and_never_zero_filled(self):
        player = Player(x=0, y=0)
        self.assertGreater(arrival_time(player, (0, 100)), arrival_time(player, (0, 100), optimistic=True))
        self.assertGreater(skating(player)[0], 0)

    def test_puck_path_stops_before_uncertain_net_or_wall_collision(self):
        puck = Player(x=15, y=245, motion_x=0, motion_y=2, height=0)
        path = puck_path(puck)
        self.assertTrue(path)
        for _, point, _ in path:
            self.assertEqual(point, project_target(point))
        self.assertLess(path[-1][0], 64)

    def test_telemetry_reads_roster_energy_without_replacing_legacy_fields(self):
        env = Mock()
        register_defense_state(env, 2)
        fields = dict(call.args for call in env.data.set_variable.call_args_list)
        self.assertEqual(fields['defense_7_vx'], {'address': 0xFFB04A + 7 * 0x80 + 0x28, 'type': '>i2'})
        self.assertNotIn('defense_2_vx', fields)
        self.assertNotIn('p1_vel_x', fields)
        self.assertEqual(fields['defense_energy_2_19'], {'address': 0xFFC700 + 0x364 + 38, 'type': '>u2'})


class DefensiveTargetTests(unittest.TestCase):
    def test_carrier_target_is_goal_side_and_independent_of_controlled_skater(self):
        state = defense_state()
        planner = DefenseController()
        first = planner.choose_target(state)
        self.assertEqual(first.mode, 'protect-lane')
        self.assertLess(first.target[1], state.puck.y)
        self.assertGreater(first.target[1], state.team1.net.y)
        state.team1.defense_control = 4
        self.assertEqual(first, planner.choose_target(state))

    def test_board_containment_closes_the_inside_escape(self):
        state = defense_state()
        state.team2.players[0].x = state.puck.x = 100
        state.team1.players[0].x, state.team1.players[0].y = 65, -190
        plan = DefenseController().choose_target(state)
        self.assertEqual(plan.mode, 'contain-boards')
        self.assertLess(plan.target[0], 100)
        self.assertLess(plan.target[1], -150)

    def test_loose_puck_with_clear_margin_is_recovered(self):
        state = defense_state()
        state.engine.puck_owner = -256
        state.puck.x, state.puck.y = 0, -200
        for player in state.team2.players:
            player.y = 180
        plan = DefenseController().choose_target(state)
        self.assertEqual(plan.mode, 'recover-safe')
        self.assertGreater(plan.opponent_arrival, plan.puck_arrival)

    def test_losing_race_defends_future_reception_before_it_happens(self):
        state = defense_state()
        state.engine.puck_owner = -256
        state.engine.last_puck_player = 6
        state.puck.x, state.puck.y = -60, -160
        state.puck.motion_x = 3
        for player in state.team1.players:
            player.y = -245
        state.team2.players[0].x, state.team2.players[0].y = 40, -160
        plan = DefenseController().choose_target(state)
        self.assertEqual(plan.mode, 'deny-reception')
        self.assertIsNotNone(plan.receiver)
        self.assertLess(plan.target[1], plan.receiver[1])
        self.assertGreater(plan.target[0], state.puck.x)

    def test_pass_interception_precedes_receiver_and_has_a_safety_margin(self):
        state = defense_state()
        state.engine.puck_owner, state.engine.last_puck_player = -256, 6
        state.puck.x, state.puck.y, state.puck.motion_x = -60, -160, 3
        state.team1.players[0].x, state.team1.players[0].y = 0, -160
        state.team2.players[0].x, state.team2.players[0].y = 80, -160
        plan = DefenseController().choose_target(state)
        self.assertEqual(plan.mode, 'intercept-pass')
        self.assertLess(plan.puck_arrival + 5, plan.opponent_arrival)
        self.assertLess(plan.target[0], state.team2.players[0].x)

    def test_switch_delay_cannot_justify_an_unreachable_puck_race(self):
        state = defense_state()
        state.engine.puck_owner = -256
        state.puck.x, state.puck.y, state.puck.motion_x = -60, -160, 3
        state.team1.players[0].x, state.team1.players[0].y = -100, -240
        state.team1.players[1].x, state.team1.players[1].y = -60, -160
        state.team1.players[1].is_falling = 1
        planner = DefenseController()
        planner.switch_at = 1000
        state.team1.players[2].x, state.team1.players[2].y = 0, -160
        state.team2.players[0].x, state.team2.players[0].y = 65, -160
        self.assertNotIn(planner.choose_target(state).mode, ('intercept-pass', 'recover-safe'))

    def test_missing_puck_height_never_claims_a_safe_interception(self):
        state = defense_state()
        state.engine.puck_owner = -256
        state.puck.height = None
        planner = DefenseController()
        planner.step(state)
        self.assertNotIn(planner.plan.mode, ('recover-safe', 'intercept-pass'))
        self.assertIn('puck.height', planner.diagnostics['missing_feedback'])

    def test_airborne_puck_is_not_treated_as_immediate_pickup(self):
        state = defense_state()
        state.engine.puck_owner = -256
        state.puck.height, state.puck.motion_z = 100, 3
        plan = DefenseController().choose_target(state)
        self.assertNotIn(plan.mode, ('recover-safe', 'intercept-pass'))

    def test_last_defender_does_not_abandon_slot_for_bait(self):
        state = defense_state()
        state.team1.players = state.team1.players[:1]
        state.team1.num_players = 1
        state.engine.puck_owner = -256
        state.puck.y = -140
        state.puck.motion_y = 1.5
        for player in state.team2.players:
            player.y = 180
        plan = DefenseController().choose_target(state)
        self.assertNotEqual(plan.mode, 'recover-safe')

    def test_best_defender_accounts_for_motion_and_excludes_fallen(self):
        state = defense_state()
        state.team1.players[0].x, state.team1.players[0].y = 60, -190
        state.team1.players[0].motion_y = -2
        state.team1.players[1].x, state.team1.players[1].y = 0, -205
        state.team1.players[1].motion_y = 2
        planner = DefenseController()
        best, _ = planner.select_player(state, (0, -130))
        self.assertEqual(best, 1)
        state.team1.players[1].is_falling = 1
        best, _ = planner.select_player(state, (0, -130))
        self.assertNotEqual(best, 1)

    def test_negative_selection_waits_and_reset_has_no_target(self):
        state = defense_state()
        state.team1.defense_control = -1
        agent = ClassicAIV1Model()
        self.assertFalse(agent.predict_frame(state).any())
        self.assertEqual(agent.defense_diagnostics['mode'], 'waiting-for-selection')

    def test_reduced_variant_goalie_ownership_is_not_a_loose_puck(self):
        state = defense_state()
        state.team1.defense_goalie = 2
        state.team1.players = state.team1.players[:2]
        self.assertTrue(owns_puck(state.team1, 2))
        self.assertFalse(owns_puck(state.team1, 5))


class DefensiveSwitchTests(unittest.TestCase):
    def setUp(self):
        self.state = defense_state()
        for player, position in zip(self.state.team1.players, (
                (100, 0), (0, -151), (0, -172), (90, -70), (-90, -70))):
            player.x, player.y = position
        self.controller = DefenseController()

    def test_switches_to_useful_reachable_skater_even_if_ideal_is_unreachable(self):
        action = self.controller.step(self.state)
        details = self.controller.diagnostics
        self.assertEqual(details['ideal_slot'], 2)
        self.assertEqual(details['desired_slot'], 1)
        self.assertEqual(details['likely_switch_slot'], 1)
        self.assertEqual(details['mode'], 'switch-request')
        self.assertEqual(action[Buttons.INPUT_B], 1)
        self.assertIsNone(details['last_switch_result'])
        self.assertEqual(details['pending_switch']['from_slot'], 0)

    def test_rom_keeps_current_instead_of_cycling_to_next_closest(self):
        self.state.team1.defense_control = 1
        self.controller.step(self.state)
        self.assertEqual(self.controller.diagnostics['ideal_slot'], 2)
        self.assertEqual(self.controller.diagnostics['desired_slot'], 1)
        self.assertEqual(self.controller.diagnostics['switch_status'], 'rom-keeps-current')
        self.assertNotEqual(self.controller.diagnostics['mode'], 'switch-request')

    def test_selection_flags_exclude_locked_and_other_controlled_players(self):
        self.state.team1.players[1].selection_flags = 0x20
        self.state.team1.players[2].selection_flags = 8
        self.assertNotIn(self.controller._likely_switch(self.state), (1, 2))
        self.state.team1.defense_control = 2
        self.assertEqual(self.controller._likely_switch(self.state), 2)

    def test_live_selection_flags_take_precedence_over_legacy_animation_flags(self):
        self.state.team1.players[1].is_falling = 1
        self.state.team1.players[1].selection_flags = 0
        self.assertEqual(self.controller._likely_switch(self.state), 1)
        self.controller.step(self.state)
        self.assertEqual(self.controller.diagnostics['desired_slot'], 1)

    def test_signed_high_byte_projection_and_ties_match_rom_order(self):
        self.state.puck.x = self.state.puck.y = 0
        self.state.puck.motion_x = -257 * 17 / 65536
        first, second = self.state.team1.players[:2]
        first.x, first.y, second.x, second.y = -2, 0, -1, 0
        self.assertEqual(self.controller._likely_switch(self.state), 0)
        second.x = -2
        self.assertEqual(self.controller._likely_switch(self.state), 1)

    def test_small_advantage_does_not_churn_but_meaningful_improvement_switches(self):
        costs = {0: 12, 1: 8, 2: 0, 3: 100, 4: 100}
        with patch.object(self.controller, '_cost', side_effect=lambda _, slot, *args: costs[slot]):
            self.assertEqual(self.controller.select_player(self.state, (0, -172))[0], 0)
            costs[0] = 13
            self.assertEqual(self.controller.select_player(self.state, (0, -172))[0], 1)

    def test_hysteresis_cannot_lose_a_safe_interception_deadline(self):
        self.controller.plan = DefensePlan((0, -172), 'intercept-pass', 'test', puck_arrival=15)
        costs = {0: 11, 1: 8, 2: 0, 3: 100, 4: 100}
        with patch.object(self.controller, '_cost', side_effect=lambda _, slot, *args: costs[slot]):
            self.assertEqual(self.controller.select_player(self.state, (0, -172))[0], 1)

    def test_ignored_switches_retry_with_backoff_instead_of_permanent_lockout(self):
        presses = []
        for frame in range(150):
            action = self.controller.step(self.state)
            if self.controller.diagnostics['mode'] == 'switch-request':
                presses.append(frame)
            if self.controller.pending_switch:
                self.assertFalse(action[Buttons.INPUT_C])
        self.assertGreaterEqual(len(presses), 4)
        self.assertTrue(all(b - a >= 12 for a, b in zip(presses, presses[1:])))
        self.assertEqual(self.controller.last_switch_result['outcome'], 'unconfirmed')

    def test_release_then_feedback_confirms_selection_not_the_button_press(self):
        self.controller.step(self.state)
        release = self.controller.step(self.state)
        self.assertFalse(release[Buttons.INPUT_B])
        self.assertIsNone(self.controller.last_switch_result)
        self.assertEqual(self.controller.diagnostics['switch_status'], 'awaiting-selection')
        self.state.team1.defense_control = 1
        self.controller.step(self.state)
        self.assertIsNone(self.controller.pending_switch)
        self.assertEqual(self.controller.last_switch_result['outcome'], 'confirmed')
        self.assertEqual(self.controller.last_switch_result['elapsed_frames'], 2)
        self.assertEqual(self.controller.switch_attempts, 0)

    def test_unexpected_selection_and_possession_clear_pending_request(self):
        self.controller.step(self.state)
        self.state.team1.defense_control = 3
        self.controller.step(self.state)
        self.assertEqual(self.controller.last_switch_result['outcome'], 'unexpected-selection')
        self.assertEqual(self.controller.last_switch_result['actual_slot'], 3)
        self.state.engine.puck_owner = 3
        self.assertFalse(self.controller.step(self.state).any())
        self.assertIsNone(self.controller.pending_switch)
        self.assertEqual(self.controller.diagnostics, {})

    def test_cooldown_and_prior_b_press_prevent_a_new_request(self):
        self.controller.switch_at = 5
        self.controller.step(self.state)
        self.assertEqual(self.controller.diagnostics['switch_status'], 'cooldown')
        self.controller.switch_at = 0
        self.controller.b_down = True
        self.assertFalse(self.controller.step(self.state)[Buttons.INPUT_B])
        self.assertEqual(self.controller.diagnostics['switch_status'], 'release-b')
        self.assertEqual(self.controller.step(self.state)[Buttons.INPUT_B], 1)

    def test_goalie_and_unavailable_skater_never_become_switch_candidates(self):
        self.state.team1.players[1].role = 0
        self.state.team1.players[2].unavailable = 4
        self.assertNotIn(self.controller._likely_switch(self.state), (1, 2))

    def test_stale_stars_cannot_suppress_defensive_switch_intents(self):
        context = SimpleNamespace(game_state=self.state, action_type='HOCKEY_INTENT_DPAD')
        controller = HockeyActionController(context)
        self.state.team1.player_haspuck = True
        self.state.team1.goalie_haspuck = True
        for owner in (-256, -1, 6, 0, 5):
            with self.subTest(owner=owner):
                self.state._update_engine_state({'puck_owner': owner})
                buttons = np.zeros(12, dtype=np.int8)
                controller._apply_hockey_intent(
                    [HOCKEY_INTENT_CHANGE_PLAYER, 0, 0, 0, 0, 0],
                    buttons, [0] * 6, controller._new_action_state())
                self.assertEqual(bool(buttons[Buttons.INPUT_B]), owner not in (0, 5))
                self.assertEqual(controller._player_has_puck(self.state.team1), owner == 0)
        self.state._update_engine_state({})
        self.assertTrue(controller._team_has_puck(self.state.team1))


class DefensiveCadenceTests(unittest.TestCase):
    def test_defensive_cooldowns_use_elapsed_frames_during_possession(self):
        state = defense_state()
        model = ClassicAIV1Model()
        model.predict_frame(state, 4)
        state.engine.puck_owner = 0
        for _ in range(39):
            model.predict_frame(state, 4)
        self.assertEqual(model.defense.frames, 40)

    def test_offensive_actions_keep_the_original_four_frame_interval(self):
        from tests.unit.test_v1_one_timers import cross_slot_state
        for schema in ('FILTERED', 'HOCKEY_INTENT_DPAD'):
            args = SimpleNamespace(action_type=schema)
            decision_model, frame_model = ClassicAIV1Model(args), ClassicAIV1Model(args)
            state = cross_slot_state()
            for tick in range(12):
                if tick == 1:
                    state.engine.puck_owner = -256
                elif tick == 3:
                    state.engine.puck_owner, state.team1.control = 1, 2
                expected = decision_model.predict_game_state(state)
                for _ in range(4):
                    np.testing.assert_array_equal(frame_model.predict_frame(state, 4), expected)

    def test_teacher_only_observes_each_frame_without_overriding_clone(self):
        inner = wrapped()
        env = StochasticFrameSkip(inner, 4, -1)
        self.addCleanup(env.close)
        agent = create_scripted('classic-v1', SimpleNamespace(action_type='FILTERED'))
        configure_scripted_frames(agent, env, record=True, teacher_only=True)
        observation, _ = env.reset(seed=7)
        inner.game_state = defense_state()
        agent.act(AgentInput(inner.game_state, observation))
        clone = np.zeros(12, dtype=np.int8)
        _, _, _, _, info = env.step(clone)
        self.assertEqual(len(info['scripted_frames']), 4)
        for _, emitted, _, _, diagnostics in info['scripted_frames']:
            np.testing.assert_array_equal(emitted, clone)
            self.assertEqual(diagnostics['teacher_action'].shape, (12,))

    def test_pass_retargets_on_next_frame_without_advancing_offense_tick(self):
        state = defense_state()
        model = ClassicAIV1Model()
        model.predict_frame(state, 4)
        first, tick = model._last_target, model._tick
        state.engine.puck_owner = -256
        state.engine.last_puck_player = 6
        state.puck.x, state.puck.y, state.puck.motion_x = -60, -160, 3
        state.team2.players[0].x, state.team2.players[0].y = 40, -160
        model.predict_frame(state, 4)
        self.assertEqual(model._tick, tick)
        self.assertNotEqual(model._last_target, first)

    def test_recovery_releases_defensive_buttons_within_the_interval(self):
        state = defense_state()
        model = ClassicAIV1Model()
        model.predict_frame(state, 4)
        state.engine.puck_owner = 0
        self.assertFalse(model.predict_frame(state, 4).any())
        self.assertEqual(model.defense_diagnostics, {})

    def test_frame_skip_collects_actual_substep_observations_and_actions(self):
        inner = wrapped()
        env = StochasticFrameSkip(inner, 4, -1)
        self.addCleanup(env.close)
        agent = create_scripted('classic-v1', SimpleNamespace(action_type='FILTERED'))
        configure_scripted_frames(agent, env, record=True)
        observation, _ = env.reset(seed=7)
        inner.game_state = defense_state()
        action = agent.act(AgentInput(inner.game_state, observation)).action
        _, _, _, _, info = env.step(action)
        self.assertEqual(len(info['scripted_frames']), 4)
        self.assertEqual(agent._tick, 1)
        self.assertEqual(agent.defense.frames, 4)
        self.assertEqual(info['scripted_frames'][0][0].shape, np.asarray(observation).shape)
        self.assertEqual(info['scripted_frames'][0][1].shape, (12,))


if __name__ == '__main__':
    unittest.main()

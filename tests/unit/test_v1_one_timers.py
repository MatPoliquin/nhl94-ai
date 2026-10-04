"""Pass selection, button edges, and cancellation for V1's one-timer sequence."""
from copy import deepcopy
from types import SimpleNamespace
import unittest

from nhl94_ai.agents.classic_v1 import ClassicAIV1Model
from nhl94_ai.agents.registry import create_scripted
from nhl94_ai.env.actions import HockeyActionController
from nhl94_ai.evaluation.benchmark import away_view
from nhl94_ai.game.constants import GameConsts as Buttons
from nhl94_ai.game.state import NHL94GameState


def cross_slot_state():
    state = NHL94GameState(5)
    state.team1.net.y, state.team2.net.y = -264, 264
    state.team1.control = 1
    state.engine.puck_owner = 0
    state.team1.player_haspuck = True
    for player in state.team1.players:
        player.x, player.y = 80, -180
    for player in state.team2.players:
        player.x, player.y = 90, 0
    state.team1.players[0].x, state.team1.players[0].y = -50, 190
    state.team1.players[0].orientation = 1  # UP+RIGHT, toward this receiver.
    state.team1.players[1].x, state.team1.players[1].y = 35, 210
    state.puck.x, state.puck.y = -48, 190
    return state


def converted_buttons(model, processor, action_state):
    action = model.predict_game_state(processor.game_state)[0]
    return processor._process_action(action, action_state)[0]


class OneTimerContracts(unittest.TestCase):
    def test_pass_then_fresh_c_press_and_release_in_both_action_schemas(self):
        for schema in ('FILTERED', 'HOCKEY_INTENT_DPAD'):
            with self.subTest(schema=schema):
                state = cross_slot_state()
                model = ClassicAIV1Model(SimpleNamespace(action_type=schema))
                context = SimpleNamespace(action_type=schema, game_state=state)
                processor = HockeyActionController(context)
                action_state = processor._new_action_state()
                initial = converted_buttons(model, processor, action_state)
                self.assertEqual(initial[Buttons.INPUT_B], 1)
                self.assertEqual(initial[Buttons.INPUT_C], 0)
                self.assertEqual(initial[Buttons.INPUT_RIGHT], 1)
                self.assertEqual(initial[Buttons.INPUT_UP], 0)
                self.assertEqual(model._last_decision, 'one-timer-pass')
                # The environment repeats each decision across four emulator
                # frames; advance macro timing without asking the agent again.
                if schema == 'HOCKEY_INTENT_DPAD':
                    repeated = model.get_action_preferences()[0].astype('int8')
                    for _ in range(3):
                        held = processor._process_action(repeated, action_state)[0]
                        self.assertEqual(held[Buttons.INPUT_B], 1)
                        self.assertEqual(held[Buttons.INPUT_C], 0)
                state.engine.puck_owner = -256
                state.team1.player_haspuck = False
                shot = converted_buttons(model, processor, action_state)
                self.assertEqual(shot[Buttons.INPUT_B], 0)
                self.assertEqual(shot[Buttons.INPUT_C], 1)
                self.assertEqual(model._last_decision, 'one-timer-shoot')
                self.assertEqual(converted_buttons(model, processor, action_state)[Buttons.INPUT_C], 0)

    def test_pending_pass_tracks_slots_in_fresh_state_objects(self):
        state, model = cross_slot_state(), ClassicAIV1Model()
        model.predict_game_state(state)
        current = deepcopy(state)
        current.team1.players[1].x, current.team1.players[1].y = 39, 205
        release = model.predict_game_state(current)[0]
        self.assertEqual(model._last_target, (39, 205))
        self.assertEqual(release[Buttons.INPUT_B], 0)
        self.assertEqual(release[Buttons.INPUT_C], 0)

    def test_intent_wrapper_holds_pass_aim_then_releases_b(self):
        state = cross_slot_state()
        model = ClassicAIV1Model(SimpleNamespace(action_type='HOCKEY_INTENT_DPAD'))
        processor = HockeyActionController(SimpleNamespace(action_type='HOCKEY_INTENT_DPAD', game_state=state))
        macro = processor._new_action_state()
        action = model.predict_game_state(state)[0]
        first = processor._process_action(action, macro)[0]
        for _ in range(3):
            # A new movement direction cannot redirect a pass already started.
            held = processor._process_action([0, 0, 1, 1, 0, 0], macro)[0]
            self.assertEqual(held[Buttons.INPUT_B], 1)
            self.assertEqual(held[4:8].tolist(), first[4:8].tolist())
        release = processor._process_action([0, 0, 0, 0, 0, 0], macro)[0]
        self.assertEqual(release[Buttons.INPUT_B], 0)
        processor._reset_action_state(macro)
        self.assertEqual(macro['pass_press_frames'], 0)

    def test_interception_reception_and_knockdown_cancel(self):
        for situation in ('intercepted', 'received', 'knocked-down'):
            with self.subTest(situation=situation):
                state, model = cross_slot_state(), ClassicAIV1Model()
                model.predict_game_state(state)
                if situation == 'intercepted':
                    state.engine.puck_owner = 6
                elif situation == 'received':
                    state.engine.puck_owner, state.team1.control = 1, 2
                else:
                    state.team1.players[1].is_falling = 1
                model.predict_game_state(state)
                self.assertIsNone(model._one_timer)
                self.assertFalse(model._last_decision.startswith('one-timer'))

    def test_timeout_returns_to_skating_and_does_not_immediately_repass(self):
        state, model = cross_slot_state(), ClassicAIV1Model()
        model.predict_game_state(state)
        for _ in range(18):
            model.predict_game_state(state)
        self.assertIsNone(model._one_timer)
        self.assertEqual(model._last_decision, 'carry')

    def test_fresh_receiver_release_ends_the_sequence_even_without_a_shot_on_goal(self):
        for feedback in ('attempt', 'recorded-shot'):
            with self.subTest(feedback=feedback):
                state, model = cross_slot_state(), ClassicAIV1Model()
                state.team1.one_timer_attempts = 3
                model.predict_game_state(state)
                state.engine.puck_owner, state.engine.shot_player = -256, 1
                if feedback == 'attempt':
                    state.team1.one_timer_attempts = 4
                else:
                    state.team1.stats.shots += 1
                model.predict_game_state(state)
                self.assertIsNone(model._one_timer)
                self.assertEqual(model.one_timer_metrics['shot-released'], 1)
                self.assertFalse(model._last_decision.startswith('one-timer'))

    def test_stale_or_other_shooter_feedback_does_not_finish_the_sequence(self):
        for feedback in ('stale', 'other-shooter'):
            with self.subTest(feedback=feedback):
                state, model = cross_slot_state(), ClassicAIV1Model()
                state.team1.one_timer_attempts = 3
                state.engine.shot_player = 1
                model.predict_game_state(state)
                state.engine.puck_owner = -256
                if feedback == 'other-shooter':
                    state.team1.one_timer_attempts = 4
                    state.team1.stats.shots += 1
                    state.engine.shot_player = 2
                model.predict_game_state(state)
                self.assertIsNotNone(model._one_timer)

    def test_stoppage_clears_the_sequence_between_offensive_decisions(self):
        for schema in ('FILTERED', 'HOCKEY_INTENT_DPAD'):
            with self.subTest(schema=schema):
                state = cross_slot_state()
                model = ClassicAIV1Model(SimpleNamespace(action_type=schema))
                model.predict_frame(state, frame_skip=4)
                state.engine.clock_stopped = True
                action = model.predict_frame(state, frame_skip=4)[0]
                self.assertIsNone(model._one_timer)
                self.assertFalse(action.any())
                self.assertEqual(model.one_timer_metrics['play-stopped'], 1)
                self.assertEqual(model._frame_remaining, 0)

    def test_receiver_release_interrupts_a_repeated_c_action_on_the_next_frame(self):
        state, model = cross_slot_state(), ClassicAIV1Model()
        state.team1.one_timer_attempts = 0
        for _ in range(4):
            model.predict_frame(state, frame_skip=4)
        state.engine.puck_owner = -256
        self.assertTrue(model.predict_frame(state, frame_skip=4)[0, Buttons.INPUT_C])
        state.team1.one_timer_attempts, state.engine.shot_player = 1, 1
        model.predict_frame(state, frame_skip=4)
        self.assertIsNone(model._one_timer)
        self.assertEqual(model.one_timer_metrics['shot-released'], 1)

    def test_legacy_timeout_is_in_emulator_frames_at_all_decision_intervals(self):
        for interval in (1, 4, 10):
            with self.subTest(interval=interval):
                state, model = cross_slot_state(), ClassicAIV1Model()
                model.predict_frame(state, frame_skip=interval)
                self.assertEqual(model._one_timer[2] - model._one_timer_started, 72)
                deadline = model._one_timer[2]
                while model.defense.frames < deadline - 1:
                    model.predict_frame(state, frame_skip=interval)
                    self.assertIsNotNone(model._one_timer)
                model.predict_frame(state, frame_skip=interval)
                self.assertIsNone(model._one_timer)
                self.assertEqual(model.defense.frames, deadline)
                self.assertEqual(model.one_timer_metrics['timeout'], 1)

    def test_interval_one_keeps_raw_b_held_for_the_rom_pass_sampling_window(self):
        state, model = cross_slot_state(), ClassicAIV1Model()
        for _ in range(4):
            self.assertTrue(model.predict_frame(state, frame_skip=1)[0, Buttons.INPUT_B])
        self.assertFalse(model.predict_frame(state, frame_skip=1)[0, Buttons.INPUT_B])

    def test_blocked_lane_covered_receiver_and_wrong_zone_are_rejected(self):
        for situation in ('lane', 'receiver', 'zone', 'side', 'falling'):
            with self.subTest(situation=situation):
                state = cross_slot_state()
                if situation == 'lane':
                    state.team2.players[0].x, state.team2.players[0].y = -7, 200
                elif situation == 'receiver':
                    state.team2.players[0].x, state.team2.players[0].y = 33, 209
                elif situation == 'zone':
                    state.team1.players[1].y = -100
                elif situation == 'side':
                    state.team1.players[1].x = -35
                else:
                    state.team1.players[1].is_falling = 1
                model = ClassicAIV1Model()
                model.predict_game_state(state)
                self.assertIsNone(model._one_timer)

    def test_lower_net_uses_away_slots_and_the_correct_pass_direction(self):
        state = cross_slot_state()
        state.team2.control = 1
        state.team2.players[0].x, state.team2.players[0].y = 50, -190
        state.team2.players[1].x, state.team2.players[1].y = -35, -210
        state.engine.puck_owner = 6
        model = ClassicAIV1Model()
        action = model.predict_game_state(away_view(state))[0]
        self.assertEqual(action[Buttons.INPUT_B], 1)
        self.assertEqual(action[Buttons.INPUT_LEFT], 1)
        self.assertEqual(model._one_timer[:2], (6, 7))

    def test_reset_discards_pending_pass_and_cooldown(self):
        agent = create_scripted('classic-v1', SimpleNamespace(action_type='FILTERED'))
        state = cross_slot_state()
        agent.predict_game_state(state)
        self.assertIsNotNone(agent._one_timer)
        agent.reset()
        self.assertIsNone(agent._one_timer)
        self.assertEqual(agent._pass_at, 0)
        self.assertEqual(agent.predict_game_state(state)[0, Buttons.INPUT_B], 1)

    def test_setup_skates_across_the_receiver_then_expires(self):
        state, model = cross_slot_state(), ClassicAIV1Model()
        state.team1.players[0].x = 30
        action = model.predict_game_state(state)[0]
        self.assertEqual(model._last_decision, 'one-timer-setup')
        self.assertEqual(action[Buttons.INPUT_LEFT], 1)
        self.assertEqual(action[Buttons.INPUT_B], 0)
        for _ in range(8):
            model.predict_game_state(deepcopy(state))
        self.assertEqual(model._last_decision, 'carry')

    def test_weak_receiver_needs_a_closer_one_timer_position(self):
        state = cross_slot_state()
        state.team1.players[1].y = 185
        receiver = state.team1.players[1]
        receiver.shot_accuracy = 30
        strong = ClassicAIV1Model()._one_timer_target(state.team1, state.team2, 1)
        receiver.shot_accuracy = 0
        weak = ClassicAIV1Model()._one_timer_target(state.team1, state.team2, 1)
        self.assertIsNotNone(strong)
        self.assertIsNone(weak)

    def test_pass_aim_uses_the_nearest_of_eight_directions(self):
        from nhl94_ai.game.geometry import aim_pass
        import numpy as np
        passer = SimpleNamespace(x=0, y=0)
        for point, expected in (((100, 10), [0, 0, 0, 1]),
                                ((-100, 10), [0, 0, 1, 0]),
                                ((10, 100), [1, 0, 0, 0]),
                                ((-10, -100), [0, 1, 0, 0]),
                                ((80, 80), [1, 0, 0, 1])):
            with self.subTest(point=point):
                action = np.ones(12, dtype=np.int8)
                aim_pass(action, passer, SimpleNamespace(x=point[0], y=point[1]))
                self.assertEqual(action[4:8].tolist(), expected)

    def test_setup_cancels_on_turnover_and_reset(self):
        state = cross_slot_state()
        state.team1.players[0].x = 30
        agent = create_scripted('classic-v1', SimpleNamespace(action_type='FILTERED'))
        agent.predict_game_state(state)
        self.assertEqual(agent._last_decision, 'one-timer-setup')
        state.engine.puck_owner = 6
        agent.predict_game_state(state)
        self.assertEqual(agent._setup_until, 0)
        agent.reset()
        self.assertEqual(agent._setup_at, 0)
        self.assertIsNone(agent._setup_slot)

    def test_macro_reset_cancels_an_unfinished_pass(self):
        context = SimpleNamespace(action_type='HOCKEY_INTENT_DPAD', game_state=cross_slot_state())
        processor = HockeyActionController(context)
        macro = processor._new_action_state()
        converted_buttons(ClassicAIV1Model(SimpleNamespace(action_type=context.action_type)), processor, macro)
        self.assertGreater(macro['pass_press_frames'], 0)
        processor._reset_action_state(macro)
        action = processor._process_action([0, 0, 0, 0, 0, 0], macro)[0]
        self.assertEqual(action[Buttons.INPUT_B], 0)
        self.assertEqual(action[4:8].tolist(), [0, 0, 0, 0])

    def test_direct_shot_and_single_skater_need_no_pass_sequence(self):
        state = cross_slot_state()
        state.team1.players[0].x, state.team1.players[0].y = -30, 225
        state.team2.goalie.x = 12  # Already displaced: no need for a goalie deke.
        model = ClassicAIV1Model()
        self.assertEqual(model.predict_game_state(state)[0, Buttons.INPUT_C], 1)
        self.assertIsNone(model._one_timer)
        state = cross_slot_state()
        state.team1.players = state.team1.players[:1]
        state.team1.num_players = 1
        model = ClassicAIV1Model()
        model.predict_game_state(state)
        self.assertIsNone(model._one_timer)


class OneTimerCompletionContracts(unittest.TestCase):
    def test_original_passer_recovery_interrupts_wait_between_decisions(self):
        for interval in (1, 4, 8, 10):
            with self.subTest(interval=interval):
                state, model = cross_slot_state(), ClassicAIV1Model()
                model.predict_frame(state, frame_skip=interval)
                state.engine.puck_owner = -256
                model.predict_frame(state, frame_skip=interval)
                state.engine.puck_owner = 0
                model.predict_frame(state, frame_skip=interval)
                self.assertIsNone(model._one_timer)
                self.assertEqual(model.one_timer_metrics, {'recovered-by-passer': 1})
                self.assertFalse(model._last_decision.startswith('one-timer'))

    def test_initial_passer_possession_is_not_mistaken_for_recovery(self):
        state, model = cross_slot_state(), ClassicAIV1Model()
        for _ in range(8):
            model.predict_frame(state)
        self.assertIsNotNone(model._one_timer)
        self.assertEqual(model.one_timer_metrics, {})

    def test_interception_and_period_end_record_exactly_one_outcome(self):
        for outcome in ('possession-changed', 'period-ended'):
            with self.subTest(outcome=outcome):
                state, model = cross_slot_state(), ClassicAIV1Model()
                model.predict_frame(state)
                if outcome == 'possession-changed':
                    state.engine.puck_owner = 6
                    model.predict_frame(state)
                else:
                    model._end_one_timer(outcome)
                model._end_one_timer('period-ended')
                self.assertEqual(model.one_timer_starts, 1)
                self.assertEqual(model.one_timer_metrics, {outcome: 1})

    def test_outcome_logging_does_not_reschedule_cached_defense(self):
        state, model = cross_slot_state(), ClassicAIV1Model()
        model.predict_frame(state)
        tick = model._tick
        state.engine.puck_owner = 6
        model.predict_frame(state)
        self.assertEqual(model._tick, tick)
        self.assertEqual(model._frame_remaining, 2)
        self.assertEqual(model.one_timer_metrics, {'possession-changed': 1})

    def test_release_finishes_even_when_the_original_passer_has_already_recovered(self):
        state, model = cross_slot_state(), ClassicAIV1Model()
        state.team1.one_timer_attempts = 0
        model.predict_game_state(state)
        state.team1.one_timer_attempts, state.engine.shot_player = 1, 1
        self.assertEqual(state.engine.puck_owner, 0)
        model.predict_game_state(state)
        self.assertIsNone(model._one_timer)
        self.assertEqual(model.one_timer_metrics['shot-released'], 1)
        self.assertFalse(model._last_decision.startswith('one-timer'))


if __name__ == '__main__':
    unittest.main()

"""Distance/traffic setup, human shot heights and conservative native attribution."""
from dataclasses import replace
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from nhl94_ai.evaluation.cross_crease_distance import (
    AimedShotTiming, CrossingObservations, build_parser, classify_traffic_events,
    summarize_distance, validate_options,
)
from nhl94_ai.game.constants import GameConsts as Buttons
from nhl94_ai.game.state import NHL94GameState
from nhl94_ai.evaluation.cross_crease_probe import trial
from nhl94_ai.tasks.cross_crease_setup import (
    CrossingTraffic, NATIVE_TOUCH_PLAYER, OBJECT_BASE, OBJECT_STRIDE, keep_traffic_idle,
    crossing_traffic_position, place_crossing_traffic, set_crossing_velocity,
)
from tests.ram_fixture import FixtureMemory


def environment():
    memory = FixtureMemory({})
    return SimpleNamespace(data=SimpleNamespace(memory=memory))


class DistanceSetupTests(unittest.TestCase):
    def test_farther_geometry_has_coherent_full_positions_and_momentum(self):
        env = environment()
        set_crossing_velocity(env, 7, direction=-1, lateral=1.5, goalward=0, depth=100)
        for slot, x in ((7, 35), (14, 25)):
            base = OBJECT_BASE + slot * OBJECT_STRIDE
            self.assertEqual(env.data.memory.extract(base, '>i4'), x * 65536)
            self.assertEqual(env.data.memory.extract(base + 0x14, '>i4'), -100 * 65536)
            self.assertEqual(env.data.memory.extract(base + 0x20, '>i4'), -100 * 65536)

    def test_initial_lane_traffic_is_active_and_idle_or_native_pursuing(self):
        for carrier, goalie in ((1, 11), (7, 5)):
            for kind in ('friendly-initial', 'opponent-initial', 'pursuit-initial'):
                env = environment()
                sign = 1 if carrier < 6 else -1
                keeper = OBJECT_BASE + goalie * OBJECT_STRIDE
                env.data.memory.assign(keeper, '>i4', -12 * 65536)
                env.data.memory.assign(keeper + 0x14, '>i4', sign * 250 * 65536)
                set_crossing_velocity(env, carrier, direction=1, lateral=1.5, goalward=0, depth=160)
                traffic = place_crossing_traffic(env, carrier, goalie, kind=kind, direction=1)
                self.assertEqual(traffic.position, (-18.5, sign * 205))
                base = OBJECT_BASE + traffic.slot * OBJECT_STRIDE
                self.assertEqual(env.data.memory.extract(base + 0x34, '>i2'), 3)
                self.assertFalse(env.data.memory.extract(base + 0x62, '|u1') & 8)
                index = env.data.memory.extract(base + 0x36, '>u2')
                self.assertEqual(env.data.memory.extract(base + 0x38 + index, '|u1'),
                                 0x11 if traffic.pursuing else 0)
                self.assertEqual(traffic.opponent, traffic.slot // 6 != carrier // 6)

    def test_idle_override_preserves_contact_impulse_and_fractional_position(self):
        env = environment()
        traffic = CrossingTraffic(6, 'opponent-initial', (0, 200), True, False)
        base = OBJECT_BASE + traffic.slot * OBJECT_STRIDE
        env.data.memory.assign(base, '>i4', 12345)
        env.data.memory.assign(base + 0x28, '>i2', 3210)
        env.data.memory.assign(base + 0x38, '|u1', 0x11)
        keep_traffic_idle(env, traffic)
        self.assertEqual(env.data.memory.extract(base + 0x38, '|u1'), 0)
        self.assertEqual(env.data.memory.extract(base, '>i4'), 12345)
        self.assertEqual(env.data.memory.extract(base + 0x28, '>i2'), 3210)

    def test_pursuer_keeps_its_native_assignment(self):
        env = environment()
        traffic = CrossingTraffic(6, 'pursuit-initial', (0, 200), True, True)
        base = OBJECT_BASE + traffic.slot * OBJECT_STRIDE
        env.data.memory.assign(base + 0x38, '|u1', 0x11)
        keep_traffic_idle(env, traffic)
        self.assertEqual(env.data.memory.extract(base + 0x38, '|u1'), 0x11)

    def test_close_between_body_placement_has_valid_clearance_without_moving_the_shooter(self):
        point = crossing_traffic_position((-35, 216), (-12, 250), (-25, 216), 'opponent-between', 1)
        self.assertEqual(point, (-23.5, 233))
        validate_options(build_parser().parse_args(['--distances', '34', '--traffic', 'opponent-between']))
        with self.assertRaises(ValueError):
            crossing_traffic_position((-35, 230), (-12, 250), (-25, 230), 'opponent-between', 1)

    def test_invalid_distance_duplicate_and_overlapping_traffic_cells_are_rejected(self):
        for tokens in (['--distances', '19'], ['--distances', '171'],
                       ['--lateral', 'nan'], ['--distances', '50', '50'],
                       ['--traffic', 'opponent-initial', '--distances', '34']):
            with self.assertRaises(ValueError):
                validate_options(build_parser().parse_args(tokens))


class HeightTests(unittest.TestCase):
    def test_vertical_input_is_world_shot_height_for_both_crossing_directions(self):
        for direction in (-1, 1):
            for height in ('low', 'middle', 'high'):
                timing = AimedShotTiming(height=height)
                action = timing.action(0, direction)
                self.assertTrue(action[Buttons.INPUT_C])
                self.assertTrue(action[Buttons.INPUT_UP] == (height == 'high'))
                self.assertTrue(action[Buttons.INPUT_DOWN] == (height == 'low'))
                self.assertTrue(action[Buttons.INPUT_RIGHT if direction > 0 else Buttons.INPUT_LEFT])
                tap = replace(timing, press_frame=0, hold_frames=4, aim=1)
                self.assertEqual(tap.height, height)
                self.assertFalse(tap.action(4, direction)[Buttons.INPUT_C])
        with self.assertRaises(ValueError):
            AimedShotTiming(height='goalward')


class NativeObservationTests(unittest.TestCase):
    def test_rebounds_are_not_direct_shot_blocks_and_classification_is_idempotent(self):
        events = [
            {'frame': 38, 'kind': 'shot-block-or-deflection'},
            {'frame': 40, 'kind': 'goalie-shot-touch'},
            {'frame': 55, 'kind': 'shot-block-or-deflection'},
            {'frame': 55, 'kind': 'shot-catch'},
        ]
        classified = classify_traffic_events(events)
        self.assertEqual([event['kind'] for event in classified],
                         ['shot-block-or-deflection', 'goalie-shot-touch', 'rebound-traffic-touch', 'rebound-catch'])
        self.assertEqual(classify_traffic_events(classified), classified)
        self.assertEqual(events[-1]['kind'], 'shot-catch')

    def observation(self):
        env = environment()
        state = NHL94GameState(5)
        for team in (state.team1, state.team2):
            for player in (*team.players, team.goalie):
                player.precise_x, player.precise_y = 0, 200
                player.motion_x, player.motion_y = 0, 0
                player.role = 3
        state.engine.puck_owner = state.engine.last_puck_player = 1
        env.data.memory.assign(NATIVE_TOUCH_PLAYER, '>i2', 1)
        traffic = CrossingTraffic(6, 'opponent-initial', (0, 200), True, False)
        observer = CrossingObservations(env, 1, 11, traffic)
        observer(0, state, None)
        return observer, state

    def test_touch_and_possession_before_release_are_a_steal_not_a_shot_block(self):
        observer, state = self.observation()
        state.engine.puck_owner = 6
        observer.env.data.memory.assign(NATIVE_TOUCH_PLAYER, '>i2', 6)
        observer(1, state, None)
        kinds = [event['kind'] for event in observer.events]
        self.assertIn('steal', kinds)
        self.assertIn('loss-before-release', kinds)
        self.assertNotIn('shot-block-or-deflection', kinds)
        self.assertTrue(observer(9, state, None))

    def test_touch_after_release_is_a_block_and_catch_not_a_steal(self):
        observer, state = self.observation()
        state.engine.puck_owner = 6
        observer.env.data.memory.assign(NATIVE_TOUCH_PLAYER, '>i2', 6)
        state.engine.shot_player = 1
        observer(37, state, 37)
        kinds = [event['kind'] for event in observer.events]
        self.assertIn('shot-block-or-deflection', kinds)
        self.assertIn('shot-catch', kinds)
        self.assertNotIn('steal', kinds)

    def test_unconfirmed_launch_or_whiff_is_not_misclassified_as_a_steal(self):
        observer, state = self.observation()
        state.engine.puck_owner = 6
        observer.env.data.memory.assign(NATIVE_TOUCH_PLAYER, '>i2', 6)
        state.engine.shot_player = 1
        observer(37, state, None)
        kinds = [event['kind'] for event in observer.events]
        self.assertIn('unconfirmed-launch-or-whiff', kinds)
        self.assertNotIn('steal', kinds)
        self.assertNotIn('shot-block-or-deflection', kinds)

    def test_body_contact_goal_is_not_contact_free(self):
        row = dict(goal=True, shot=True, windup_frame=1, release_frame=37, commit_frame=None,
                   goalie_contact_impulses=0, controller_metrics={}, outcome='goal',
                   terminal_cause='goal',
                   traffic_events=[{'kind': 'traffic-body-contact'}],
                   separations={'initial': {'body_goalie': 41, 'longitudinal': 34},
                                'charge': None, 'release': None})
        summary = summarize_distance([row])
        self.assertEqual(summary['goals'], 1)
        self.assertEqual(summary['contact_free_goals'], 0)
        self.assertEqual(summary['traffic_contact_trials'], 1)
        self.assertIsNone(summary['charge_separation']['mean_body_goalie'])

    def test_possession_recovery_cannot_start_a_second_attempt(self):
        env = environment()
        actions = []
        info = {'index': 0, 'p1_score': 0, 'bench_shots1': 0}
        env.em = SimpleNamespace(set_state=lambda _: None)
        env.get_ram = lambda: np.zeros(16, dtype=np.uint8)
        env.data.update_ram = lambda: None
        env.data.lookup_all = lambda: info

        def step(action):
            actions.append(action.copy())
            info['index'] += 1
            return None, 0, False, False, info

        def update(state, current, _env):
            state.engine.puck_owner = -1 if current['index'] == 1 else 1
            state.engine.clock_stopped = current['index'] == 3
            state.team1.defense_control = 1
            for actor, point in ((state.team1.get_player_by_scnum(1), (-35, 216)),
                                 (state.team2.goalie, (-12, 250))):
                actor.precise_x, actor.precise_y = point
                actor.motion_x, actor.motion_y = 0, 0

        env.step = step
        with patch('nhl94_ai.evaluation.cross_crease_probe.update_state', side_effect=update), \
                patch('nhl94_ai.evaluation.cross_crease_probe._validate_isolation'):
            row = trial(env, b'snapshot', carrier=1, goalie=11, inactive=(), level=6, direction=1, policy='held')
        self.assertTrue(actions[0][Buttons.INPUT_C])
        self.assertFalse(np.any(actions[1]))
        self.assertFalse(np.any(actions[2]))
        self.assertEqual(row['c_frames'], 1)


if __name__ == '__main__':
    unittest.main()

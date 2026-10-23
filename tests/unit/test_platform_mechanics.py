"""Stage-2 mechanics dispatch: SNES calls SNES hooks, Genesis keeps Genesis hooks."""
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from nhl94_ai.agents import motion, passing
from nhl94_ai.agents.motion import (
    VELOCITY_SCALE, arrival_time, boost_impulse, coast_projection, direction, puck_path,
    rom_direction, skate_step, stop_projection,
)
from nhl94_ai.agents.passing import pass_contact, pass_release_frames, selected_receiver
from nhl94_ai.game.geometry import aim_pass
from nhl94_ai.game.platforms import get_backend
from nhl94_ai.game.platforms.genesis import GenesisBackend
from nhl94_ai.game.platforms.snes import SnesBackend


def player(platform, **overrides):
    values = dict(x=0.0, y=-100.0, motion_x=0.5, motion_y=0.25, vx=0, vy=0,
                  speed=20, agility=20, weight=32, energy=4096, movement_bonus=0,
                  facing=2, orientation=2, projection_uncertainty=0, platform=platform,
                  role=1, unavailable=0, selection_flags=0, stick_x=None, stick_y=None,
                  passing=20)
    values.update(overrides)
    return SimpleNamespace(**values)


def puck(platform, **overrides):
    values = dict(x=0.0, y=0.0, motion_x=0.5, motion_y=0.25, height=0, motion_z=None,
                  friction=255 / 256, platform=platform)
    values.update(overrides)
    return SimpleNamespace(**values)


def team(platform):
    players = [player(platform, x=-40, y=-60), player(platform, x=30, y=-40),
               player(platform, x=10, y=20)]
    return SimpleNamespace(players=players, skater_scnum_base=lambda: 0)


class GenesisBehaviorPreservedTests(unittest.TestCase):
    def test_coast_and_stop_keep_exact_genesis_friction_and_braking(self):
        skater = player('genesis')
        x, y = 0.0, -100.0
        vx, vy = 0.5, 0.25
        for _ in range(12):
            vx, vy = vx * 63 / 64, vy * 63 / 64
            x, y = x + vx, y + vy
        self.assertEqual(coast_projection(skater, 12), (x, y))
        import math
        sx, sy = 0.0, -100.0
        svx, svy = 0.5, 0.25
        for _ in range(4):
            svx = math.copysign(max(0, abs(svx) - 150 * VELOCITY_SCALE), svx) * 63 / 64
            svy = math.copysign(max(0, abs(svy) - 150 * VELOCITY_SCALE), svy) * 63 / 64
            sx, sy = sx + svx, sy + svy
        self.assertEqual(stop_projection(skater, 4), ((sx, sy), (svx, svy)))

    def test_direction_and_release_timing_keep_rom_rules(self):
        self.assertEqual(direction(player('genesis'), 100, 50), rom_direction(100, 50))
        self.assertEqual(pass_release_frames(player('genesis')), 4)
        self.assertEqual(pass_release_frames(player('genesis', stick_x=0, stick_y=4)), 2)

    def test_airborne_puck_integrates_genesis_gravity(self):
        path = puck_path(puck('genesis', height=10), 24, sample_interval=4)
        heights = [z for _, _, z in path]
        self.assertEqual(len(heights), 6)
        self.assertTrue(all(earlier > later for earlier, later in zip(heights, heights[1:])))


class SnesEstimatesTests(unittest.TestCase):
    def test_airborne_puck_integrates_rom_gravity_and_bounce(self):
        # ROM: zvel -= 96 per substep, z += zvel/16, landing bounces with zvel = -zvel/2.
        path = puck_path(puck('snes', motion_x=0.0, motion_y=0.0, height=10, motion_z=0.0),
                         3, sample_interval=1)
        self.assertEqual([z for _, _, z in path], [4, 0, 0])

    def test_coast_uses_the_declared_snes_friction_estimate(self):
        x, _ = coast_projection(player('snes'), 2)
        self.assertAlmostEqual(x, 0.5 * (125 / 128) + 0.5 * (125 / 128) ** 2)

    def test_estimates_are_labeled_in_diagnostics_and_docstrings(self):
        backend = get_backend('snes')
        for name in (*backend.pass_recipient_models, *backend.pass_geometry_models):
            self.assertTrue(name.startswith('rom-'), name)
        self.assertEqual(get_backend().pass_recipient_models,
                         ('native-slot-order-held-steering', 'current-position'))
        for method in ('boost_impulse', 'stop_projection', 'skate_step', 'reception_drift_shift'):
            self.assertIn('Estimated', getattr(SnesBackend, method).__doc__, method)
        for method in ('direction', 'aim_pass', 'puck_path', 'selected_receiver',
                       'pass_release_frames', 'pass_contact'):
            self.assertIn('ROM', getattr(SnesBackend, method).__doc__, method)

    def test_release_timing_and_drift_shift_do_not_read_genesis_telemetry(self):
        receiver = player('snes', stick_x=3, stick_y=4, selection_flags=1)
        self.assertEqual(pass_release_frames(receiver), 2)
        self.assertEqual(get_backend('snes').reception_drift_shift(receiver), 6)
        self.assertEqual(get_backend().reception_drift_shift(player('genesis', selection_flags=1)), 9)
        self.assertEqual(get_backend().reception_drift_shift(player('genesis', selection_flags=0)), 6)

    def test_pass_contact_uses_rom_launch_without_genesis_integer_vectors(self):
        # Cardinal capture geometry: passer at (0,0), puck carried at (-2,21),
        # receiver at (100,0); connects with the ROM-aimed hotspot offset.
        passer = player('snes', x=0.0, y=0.0, motion_x=0.0, motion_y=0.0)
        receiver = player('snes', x=100.0, y=0.0, motion_x=0.0, motion_y=0.0, facing=0)
        with patch('nhl94_ai.game.passing.rom_pass_vector', side_effect=AssertionError('ROM rule')):
            contact = pass_contact(puck('snes', x=-2, y=21, motion_x=0.0, motion_y=0.0),
                                   passer, receiver)
        self.assertIsNotNone(contact)
        self.assertEqual(contact[0], (-2, 21))
        self.assertGreater(contact[2], 0)

    def test_launch_view_is_unavailable_without_native_launch_telemetry(self):
        self.assertIsNone(get_backend('snes').pass_launch_view(None, player('snes')))
        self.assertIsNone(get_backend('snes').live_pass_contact(None, None, None, 0))


class SnesNativePassingTests(unittest.TestCase):
    """ROM $9BB445/$9E:E9BF/$9E:EB2E rules, verified against passing-cardinal-v2
    and the pass-aim-v1 probe set (/tmp/nhl94-snes-mechanics/passing_findings.md)."""

    def test_direction_octants_use_rom_2_to_1_boundaries(self):
        backend = get_backend('snes')
        self.assertEqual(backend.direction(0, 0), 8)
        self.assertEqual(backend.direction(0, 10), 0)
        self.assertEqual(backend.direction(10, 0), 2)
        self.assertEqual(backend.direction(0, -10), 4)
        self.assertEqual(backend.direction(-10, 0), 6)
        self.assertEqual(backend.direction(10, 10), 1)
        self.assertEqual(backend.direction(10, -10), 3)
        self.assertEqual(backend.direction(-10, -10), 5)
        self.assertEqual(backend.direction(-10, 10), 7)
        # Exactly 2:1 is diagonal; strictly beyond is axial.
        self.assertEqual(backend.direction(10, 5), 1)
        self.assertEqual(backend.direction(21, 10), 2)
        self.assertEqual(backend.direction(10, 20), 1)
        self.assertEqual(backend.direction(10, 21), 0)

    def test_selected_receiver_matches_cardinal_capture(self):
        # passing-cardinal-v2: passer slot 2 at (0,0), puck carried at (-2,21).
        backend = get_backend('snes')
        teammates = [player('snes', x=0, y=-100, motion_x=0, motion_y=0, carrier_state=5),
                     player('snes', x=100, y=0, motion_x=0, motion_y=0, carrier_state=3),
                     player('snes', x=0, y=0, motion_x=0, motion_y=0, carrier_state=4),
                     player('snes', x=0, y=100, motion_x=0, motion_y=0, carrier_state=2),
                     player('snes', x=-100, y=0, motion_x=0, motion_y=0, carrier_state=1)]
        side = SimpleNamespace(players=teammates, skater_scnum_base=lambda: 0)
        carried = puck('snes', x=-2, y=21, motion_x=0.0, motion_y=0.0)
        self.assertEqual(backend.selected_receiver(side, carried, 2, 0), 3)
        self.assertEqual(backend.selected_receiver(side, carried, 2, 2), 1)
        self.assertEqual(backend.selected_receiver(side, carried, 2, 1), 3)
        self.assertEqual(backend.selected_receiver(side, carried, 2, 5), 4)

    def test_selected_receiver_penalizes_adjacent_octant_and_filters(self):
        backend = get_backend('snes')
        exact = player('snes', x=5, y=100, motion_x=0, motion_y=0, carrier_state=1)
        adjacent = player('snes', x=40, y=40, motion_x=0, motion_y=0, carrier_state=1)
        side = SimpleNamespace(players=[exact, adjacent], skater_scnum_base=lambda: 0)
        still = puck('snes', x=0, y=0, motion_x=0.0, motion_y=0.0)
        # Exact octant wins despite the greater distance.
        self.assertEqual(backend.selected_receiver(side, still, 9, 0), 0)
        # The 0x1000 octant penalty flips the choice once the exact candidate is far.
        exact.y = 300
        self.assertEqual(backend.selected_receiver(side, still, 9, 0), 1)
        # carrier_state 0 (goalie/special) and motion_flags 0x1000 are ineligible.
        exact.y = 100
        exact.carrier_state = 0
        self.assertEqual(backend.selected_receiver(side, still, 9, 0), 1)
        adjacent.motion_flags = 0x1000
        self.assertIsNone(backend.selected_receiver(side, still, 9, 0))

    def test_pass_contact_matches_cardinal_capture_geometry(self):
        # passing-cardinal-v2 pass-right: receiver (100,0) facing up, puck (-2,21);
        # ROM aim (113,10), cheapdist 117, speed clamp((2*117)/3) = 78,
        # launch (76, -7) << 8 (jitter-free; the ROM drew +4), reception (104,11) at frame 23.
        backend = get_backend('snes')
        carried = puck('snes', x=-2, y=21, motion_x=0.0, motion_y=0.0)
        passer = player('snes', x=0, y=0, motion_x=0.0, motion_y=0.0)
        receiver = player('snes', x=100, y=0, motion_x=0.0, motion_y=0.0, ori_x=0.0, ori_y=1.0)
        start, point, flight, speed = pass_contact(carried, passer, receiver)
        self.assertEqual(start, (-2, 21))
        self.assertAlmostEqual(point[0], 97.9, places=0)
        self.assertAlmostEqual(point[1], 11.8, places=0)
        self.assertEqual(flight, 22)
        self.assertAlmostEqual(speed, 4.37, places=1)

    def test_pass_contact_none_when_unreachable(self):
        backend_puck = puck('snes', x=0, y=0, motion_x=0.0, motion_y=0.0)
        passer = player('snes', motion_x=0.0, motion_y=0.0)
        # Receiver beyond the boards: the route leaves the rink, so no contact.
        receiver = player('snes', x=0.0, y=400.0, motion_x=0.0, motion_y=0.0, facing=0)
        self.assertIsNone(pass_contact(backend_puck, passer, receiver))


class SnesNativeShotTests(unittest.TestCase):
    """ROM $9E:E146/$9E:E220 windup and launch; capture-verified in shooting-v2."""

    def test_release_frames_match_tap_and_hold_captures(self):
        from nhl94_ai.agents.offense import normal_shot_release_frames
        shooter = player('snes')
        # A held 5 frames released at frame 7; 12 frames at 19; a full hold at 27.
        self.assertEqual(normal_shot_release_frames(shooter, 5), 7)
        self.assertEqual(normal_shot_release_frames(shooter, 12), 19)
        self.assertEqual(normal_shot_release_frames(shooter, 30), 27)

    def test_shot_speed_matches_capture_charge_levels(self):
        from nhl94_ai.agents.finishing import shot_speed
        backend = get_backend('snes')
        shooter = player('snes')
        # (9584 * charge * 28) >> 16 with charge min(35, 14 + hold); raw speed 136x final.
        self.assertAlmostEqual(shot_speed(shooter, 5), 136 * 77 * backend.velocity_scale)
        self.assertAlmostEqual(shot_speed(shooter, 30), 136 * 143 * backend.velocity_scale)
        self.assertAlmostEqual(shot_speed(shooter, 30, one_timer=True),
                               136 * 61 * backend.velocity_scale)

    def test_goalie_motion_uses_capture_observed_positioning(self):
        self.assertEqual(get_backend('snes').goalie_motion(player('snes')), (0.05, 0.68))


class SnesNativeMotionTests(unittest.TestCase):
    def native_player(self, debt):
        return player('snes', x=-20, y=100, precise_x=-19.5, precise_y=100.75,
                      position_frac_x=32768, position_frac_y=49152,
                      raw_vel_x=-1000, raw_vel_y=250,
                      motion_x=-1000 / 4096, motion_y=250 / 4096,
                      physics_frame_debt=debt)

    def test_zero_debt_holds_motion_and_arms_next_physics_frame(self):
        backend = get_backend('snes')
        current = self.native_player(debt=0)
        future = backend.grounded_step(current, (1, 0))
        self.assertEqual((future.x, future.y), (-20, 100))
        self.assertEqual((future.precise_x, future.precise_y), (-19.5, 100.75))
        self.assertEqual((future.raw_vel_x, future.raw_vel_y), (-1000, 250))
        self.assertEqual((future.motion_x, future.motion_y), (current.motion_x, current.motion_y))
        self.assertEqual(future.facing, current.facing)
        self.assertEqual(future.physics_frame_debt, 1)
        self.assertEqual(future.physics_step_frames, 0)

    def test_pending_frames_apply_rom_friction_before_fixed_point_integration(self):
        backend = get_backend('snes')
        current = self.native_player(debt=1)
        future = backend.grounded_step(current, (0, 0))
        self.assertEqual((future.raw_vel_x, future.raw_vel_y), (-976, 244))
        expected_x = -20 * 65536 + 32768 - 976 * 16 * 2
        expected_y = 100 * 65536 + 49152 + 244 * 16 * 2
        self.assertEqual((future.x, future.position_frac_x),
                         (expected_x >> 16, expected_x & 0xFFFF))
        self.assertEqual((future.y, future.position_frac_y),
                         (expected_y >> 16, expected_y & 0xFFFF))
        self.assertAlmostEqual(future.precise_x, (expected_x >> 16) + (expected_x & 0xFFFF) / 65536)
        self.assertAlmostEqual(future.precise_y, (expected_y >> 16) + (expected_y & 0xFFFF) / 65536)
        self.assertEqual(future.physics_frame_debt, 0)
        self.assertEqual(future.physics_step_frames, 2)

    def test_native_turn_accumulates_fractional_phase_before_changing_facing(self):
        backend = get_backend('snes')
        current = self.native_player(debt=1)
        current.facing = current.orientation = 0
        current.native_facing_phase = 0
        current.turning_state = 0
        current.raw_vel_x = current.raw_vel_y = 0
        current.motion_x = current.motion_y = 0.0
        future = backend.grounded_step(current, (1, 0))
        self.assertEqual(future.facing, 0)
        self.assertEqual(future.native_facing_phase, 26112)
        self.assertAlmostEqual(future.facing_phase, 26112 / 65536)

    def test_no_acceleration_until_native_facing_aligns(self):
        backend = get_backend('snes')
        current = self.native_player(debt=1)
        current.facing = current.orientation = 0
        current.native_facing_phase = 0
        current.turning_state = 0
        current.raw_vel_x = current.raw_vel_y = 0
        current.motion_x = current.motion_y = 0.0
        current.motion_weight = 48
        current.motion_agility = 15
        future = backend.grounded_step(current, (1, 0))
        self.assertEqual(future.facing, 0)
        self.assertEqual((future.raw_vel_x, future.raw_vel_y), (0, 0))

    def test_acceleration_changes_velocity_after_native_integration(self):
        backend = get_backend('snes')
        neutral = backend.grounded_step(self.native_player(debt=1), (0, 0))
        pushed = backend.grounded_step(self.native_player(debt=1), (1, 0))
        self.assertEqual((pushed.x, pushed.y), (neutral.x, neutral.y))
        self.assertEqual(pushed.raw_vel_y, neutral.raw_vel_y)
        self.assertGreater(pushed.raw_vel_x, neutral.raw_vel_x)
        self.assertEqual(pushed.physics_step_frames, 2)

    def test_rom_rating_factor_and_squared_high_byte_speed_cap(self):
        backend = get_backend('snes')
        current = self.native_player(debt=1)
        current.motion_weight = 48
        current.motion_agility = 15
        current.speed_rating = 9
        acceleration, limit, _ = backend.skating(current)
        self.assertAlmostEqual(acceleration, 288 / 4096)
        self.assertAlmostEqual(limit, 874 ** 0.5 / 16)
        current.speed_rating = 0
        current.raw_vel_x, current.raw_vel_y = 6000, 0
        current.motion_x, current.motion_y = 6000 / 4096, 0.0
        current.facing = current.orientation = 2
        current.native_facing_phase = 0
        current.turning_state = 0
        future = backend.grounded_step(current, (1, 0))
        self.assertEqual(future.raw_vel_x, 5859)


class BackendDispatchTests(unittest.TestCase):
    def check_dispatch(self, genesis_attr, snes_attr, genesis_call, snes_call):
        with patch.object(GenesisBackend, genesis_attr, autospec=True,
                          side_effect=getattr(GenesisBackend, genesis_attr)) as genesis_spy, \
                patch.object(SnesBackend, snes_attr, autospec=True,
                             side_effect=getattr(SnesBackend, snes_attr)) as snes_spy:
            genesis_call()
            self.assertEqual(genesis_spy.call_count, 1, genesis_attr)
            self.assertFalse(snes_spy.called, snes_attr)
            snes_call()
            self.assertEqual(snes_spy.call_count, 1, snes_attr)
            self.assertEqual(genesis_spy.call_count, 1, genesis_attr)

    def test_motion_projection_dispatches_on_player_platform(self):
        self.check_dispatch('coast_projection', 'coast_projection',
                            lambda: coast_projection(player('genesis'), 4),
                            lambda: coast_projection(player('snes'), 4))
        self.check_dispatch('stop_projection', 'stop_projection',
                            lambda: stop_projection(player('genesis'), 2),
                            lambda: stop_projection(player('snes'), 2))
        self.check_dispatch('arrival_time', 'arrival_time',
                            lambda: arrival_time(player('genesis'), (10, -50)),
                            lambda: arrival_time(player('snes'), (10, -50)))
        self.check_dispatch('boost_impulse', 'boost_impulse',
                            lambda: boost_impulse(player('genesis')),
                            lambda: boost_impulse(player('snes')))
        self.check_dispatch('skate_step', 'skate_step',
                            lambda: skate_step(player('genesis'), ((0.0, 0.0), (0.5, 0.0)), 2, 0, (1, 0), 0.1, 3.0),
                            lambda: skate_step(player('snes'), ((0.0, 0.0), (0.5, 0.0)), 2, 0, (1, 0), 0.1, 3.0))
        self.check_dispatch('direction', 'direction',
                            lambda: direction(player('genesis'), 3, 1),
                            lambda: direction(player('snes'), 3, 1))

    def test_puck_path_dispatches_on_puck_platform(self):
        self.check_dispatch('puck_path', 'puck_path',
                            lambda: puck_path(puck('genesis'), 8),
                            lambda: puck_path(puck('snes'), 8))

    def test_pass_mechanics_dispatch_on_puck_and_receiver_platform(self):
        self.check_dispatch('selected_receiver', 'selected_receiver',
                            lambda: selected_receiver(team('genesis'), puck('genesis'), 0, 2),
                            lambda: selected_receiver(team('snes'), puck('snes'), 0, 2))
        self.check_dispatch('pass_release_frames', 'pass_release_frames',
                            lambda: pass_release_frames(player('genesis')),
                            lambda: pass_release_frames(player('snes')))
        passer = dict(motion_x=0.0, motion_y=0.0)
        receiver = dict(x=0.0, y=-70.0, motion_x=0.0, motion_y=0.0, facing=0)
        self.check_dispatch('pass_contact', 'pass_contact',
                            lambda: pass_contact(puck('genesis', motion_x=0.0, motion_y=0.0),
                                                 player('genesis', **passer), player('genesis', **receiver)),
                            lambda: pass_contact(puck('snes', motion_x=0.0, motion_y=0.0),
                                                 player('snes', **passer), player('snes', **receiver)))

    def test_aim_pass_dispatches_on_passer_platform(self):
        def genesis_call():
            action = np.zeros(12, dtype=np.int8)
            aim_pass(action, player('genesis'), SimpleNamespace(x=10, y=-99))
            return action

        def snes_call():
            action = np.zeros(12, dtype=np.int8)
            aim_pass(action, player('snes'), SimpleNamespace(x=10, y=-99))
            return action

        self.check_dispatch('aim_pass', 'aim_pass', genesis_call, snes_call)
        np.testing.assert_array_equal(genesis_call(), snes_call())

    def test_passive_reception_uses_console_scale_and_drift_shift(self):
        state = SimpleNamespace(puck=puck('genesis', motion_x=0.0, motion_y=0.0))
        receiver = player('genesis', x=0.0, y=-20.0, precise_x=None, precise_y=None,
                          selection_flags=1, stick_x=0, stick_y=6, motion_x=0.0, motion_y=0.0)
        from nhl94_ai.agents.receiving import passive_reception
        with patch.object(GenesisBackend, 'reception_drift_shift', autospec=True,
                          side_effect=GenesisBackend.reception_drift_shift) as genesis_spy, \
                patch.object(SnesBackend, 'reception_drift_shift') as snes_spy:
            forecast = passive_reception(state, receiver)
        self.assertIsNotNone(forecast)
        self.assertEqual(genesis_spy.call_count, 1)
        self.assertFalse(snes_spy.called)
        state.puck.platform = 'snes'
        receiver.platform = 'snes'
        with patch.object(SnesBackend, 'reception_drift_shift', autospec=True,
                          side_effect=SnesBackend.reception_drift_shift) as snes_spy:
            forecast = passive_reception(state, receiver)
        self.assertIsNotNone(forecast)
        self.assertEqual(snes_spy.call_count, 1)


if __name__ == '__main__':
    unittest.main()

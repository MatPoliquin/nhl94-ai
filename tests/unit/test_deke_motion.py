"""Fixed-point mechanics used only by experimental deke forecasts."""
import unittest

from nhl94_ai.agents.motion import VELOCITY_SCALE, skating
from nhl94_ai.agents.skating import grounded_step
from tests.unit.test_deke import deke_state


class NativeCarrierMotionTests(unittest.TestCase):
    def player(self, *, vx=0, vy=1500, heading=0):
        player = deke_state().team1.players[0]
        player.motion_x, player.motion_y = vx * VELOCITY_SCALE, vy * VELOCITY_SCALE
        player.facing = heading
        player.facing_phase = float(heading)
        player.movement_bonus = 2
        return player

    def test_integrates_friction_before_acceleration_and_keeps_subpixel_position(self):
        player = self.player()
        future = grounded_step(player, (0, 1))
        self.assertEqual(future.precise_y, player.precise_y + (1500 - (1500 >> 6)) * VELOCITY_SCALE)
        self.assertGreater(future.motion_y, player.motion_y)
        self.assertEqual(player.precise_y, 180)

    def test_turning_accelerates_along_current_facing_and_retains_fractional_turn(self):
        player = self.player()
        future = grounded_step(player, (1, 0))
        self.assertEqual(future.facing, 0)
        self.assertEqual(future.facing_phase, (0x300 - ((1500 - (1500 >> 6)) ** 2 >> 16)) * 16 / 65536)
        self.assertGreater(future.motion_y, player.motion_y)
        for _ in range(16):
            future = grounded_step(future, (1, 0))
        self.assertEqual(future.facing, 2)
        self.assertGreater(future.facing_phase, 2)
        self.assertEqual(grounded_step(future, (1, 0)).facing_phase, future.facing_phase)

    def test_turn_at_near_standstill_does_not_accelerate(self):
        player = self.player(vy=200)
        self.assertEqual(grounded_step(player, (1, 0)).motion_y, (200 - (200 >> 6)) * VELOCITY_SCALE)

    def test_signed_diagonal_acceleration_and_small_velocity_friction(self):
        player = self.player(vx=44, vy=-44, heading=5)
        future = grounded_step(player, (-1, -1))
        increment = (-141 * 35) >> 5
        self.assertEqual(future.motion_x, (43 + increment) * VELOCITY_SCALE)
        self.assertEqual(future.motion_y, (-43 + increment) * VELOCITY_SCALE)

    def test_above_speed_limit_impulse_does_not_allow_extra_acceleration(self):
        player = self.player(vy=12000)
        self.assertGreater(player.motion_y, skating(player)[1])
        future = grounded_step(player, (0, 1))
        self.assertEqual(future.motion_y, (12000 - (12000 >> 6)) * VELOCITY_SCALE)

    def test_opposite_request_brakes_forward_momentum_before_rotating(self):
        player = self.player()
        future = grounded_step(player, (0, -1))
        self.assertEqual(future.facing_phase, 0)
        self.assertEqual(future.motion_y, (1500 - (1500 >> 6) - 150) * VELOCITY_SCALE)
        for flipped, expected in ((False, 1), (True, 7)):
            player = self.player(vy=0)
            player.sprite_flipped_x = flipped
            future = grounded_step(player, (0, -1))
            self.assertEqual(future.facing, expected)
            self.assertEqual(future.motion_y, 0)


if __name__ == '__main__':
    unittest.main()

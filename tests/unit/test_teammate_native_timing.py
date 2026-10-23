"""Native elapsed ticks scale integration/acceleration, not friction calls."""
import unittest

from nhl94_ai.game.skating import grounded_step
from nhl94_ai.game.motion import VELOCITY_SCALE
from nhl94_ai.agents.teammate_support import support_forecast
from tests.support.offense import offense_state


class NativeTimingTests(unittest.TestCase):
    def test_two_elapsed_ticks_are_one_friction_update(self):
        player = offense_state().team1.players[1]
        player.precise_x, player.precise_y = 10., 20.
        player.motion_x, player.motion_y = 640*VELOCITY_SCALE, -640*VELOCITY_SCALE
        actual = grounded_step(player, (0, 0), elapsed_ticks=2)
        self.assertEqual(actual.precise_x, 10+630*VELOCITY_SCALE*2)
        self.assertEqual(actual.precise_y, 20-630*VELOCITY_SCALE*2)
        twice = grounded_step(grounded_step(player, (0,0)), (0,0))
        self.assertNotEqual(actual.precise_x, twice.precise_x)

    def test_receiving_friction_is_explicit_and_default_is_unchanged(self):
        player = offense_state().team1.players[1]
        player.motion_x = 1024*VELOCITY_SCALE
        old = grounded_step(player, (0,0))
        explicit = grounded_step(player, (0,0), elapsed_ticks=1, friction_shift=6)
        self.assertEqual(vars(old), vars(explicit))
        coast = grounded_step(player, (0,0), elapsed_ticks=1, friction_shift=9)
        self.assertGreater(coast.motion_x, old.motion_x)

    def test_no_native_updates_does_not_advance_receiver(self):
        state = offense_state(); player = state.team1.players[1]
        future = support_forecast(state, 1, (40,180), coast=True, native_updates=[])
        self.assertEqual(vars(future.team1.players[1]), vars(player))
        with self.assertRaises(ValueError):
            support_forecast(state, 1, (40,180), native_updates=[0])

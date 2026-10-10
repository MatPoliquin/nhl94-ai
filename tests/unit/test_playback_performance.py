"""Batched conservative contact queries and rendering caches retain exact semantics."""
from copy import deepcopy
import importlib.util
import math
import os
import random
import unittest
from unittest.mock import patch

import numpy as np

from nhl94_ai.agents.carry import (
    ENVELOPE_FIELDS, PROJECTION_FIELDS, body_clearance, cached_reach_envelopes, carry_path, forecast_carry, interception_time,
    interception_times, reach_envelopes,
)
from nhl94_ai.env.target_control import SETTINGS, project_target
from nhl94_ai.evaluation.playback_profile import timing_summary
from tests.unit.test_classic_offense import offense_state
from tests.unit import test_debug_playback


class CarryPerformanceParityTests(unittest.TestCase):
    def test_batched_contact_matches_scalar_for_bursts_fractional_delays_and_stick_reach(self):
        rng = random.Random(12000)
        state = offense_state()
        player = state.team2.players[0]
        for _ in range(100):
            player.x, player.y = rng.uniform(-120, 120), rng.uniform(-250, 250)
            player.precise_x, player.precise_y = player.x, player.y
            player.motion_x, player.motion_y = rng.uniform(-3, 3), rng.uniform(-3, 3)
            player.facing_phase = rng.uniform(0, 8)
            player.facing = int(player.facing_phase)
            player.energy = rng.randrange(0, 4097)
            player.burst_full_energy = rng.choice((True, False, None))
            player.stick_x, player.stick_y = rng.randrange(-20, 21), rng.randrange(-20, 21)
            player.shot_projection_offsets_x = (-20, 20)
            player.shot_projection_offsets_y = (-18, 18)
            player.projection_uncertainty = rng.uniform(0, 5)
            delay = rng.uniform(0, 30)
            extent = rng.uniform(0, 30)
            deadlines = [delay + elapsed for elapsed in range(1, 37)]
            envelopes = reach_envelopes(player, math.ceil(max(deadlines)))
            points = [(rng.uniform(-120, 120), rng.uniform(-250, 250)) for _ in deadlines]
            bodies = [rng.choice((True, False)) for _ in deadlines]
            expected = [interception_time(
                state, player, point, deadline, body=body, envelopes=envelopes,
                extra_radius=0 if body else extent) for point, deadline, body in zip(points, deadlines, bodies)]
            actual = interception_times(player, points, deadlines, bodies, envelopes, puck_extent=extent)
            np.testing.assert_array_equal(actual, expected)

    def test_contact_boundaries_and_inactive_players_match_scalar(self):
        state = offense_state()
        player = state.team2.players[0]
        player.x = player.y = player.motion_x = player.motion_y = 0
        envelopes = reach_envelopes(player, 21)
        points = [(16 + delta, 0) for delta in (-1e-12, 0, 1e-12)]
        np.testing.assert_array_equal(interception_times(player, points, [0] * 3, [True] * 3, envelopes), [0, 0, 1])
        player.role = -1
        np.testing.assert_array_equal(interception_times(player, points, [2.5] * 3, [True] * 3, envelopes), [4] * 3)

    def test_complete_forecast_details_match_scalar_queries_including_windup_extent(self):
        rng = random.Random(94)
        def scalar(player, points, deadlines, bodies, envelopes, *, puck_extent=0):
            return np.asarray([interception_time(
                state, player, point, deadline, body=body, envelopes=envelopes,
                extra_radius=0 if body else puck_extent) for point, deadline, body in zip(points, deadlines, bodies)])
        for _ in range(24):
            state = offense_state()
            for team in (state.team1, state.team2):
                for player in (*team.players, team.goalie):
                    player.x, player.y = rng.randrange(-100, 101), rng.randrange(-230, 231)
                    player.motion_x, player.motion_y = rng.uniform(-2, 2), rng.uniform(-2, 2)
                    player.projection_uncertainty = rng.uniform(0, 4)
            player = state.team1.players[0]
            target = player.x + rng.uniform(-48, 48), player.y + rng.uniform(-48, 48)
            delay = rng.uniform(0, 40)
            actual = forecast_carry(state, player, target, pressure_delay=delay, risk_cache={})[1]
            with patch('nhl94_ai.agents.carry.interception_times', side_effect=scalar):
                expected = forecast_carry(state, player, target, pressure_delay=delay)[1]
            self.assertEqual(actual, expected)

    def test_envelope_cache_keys_every_grounded_physics_field_and_horizon(self):
        player = offense_state().team2.players[0]
        cache = {}
        original = cached_reach_envelopes(player, 21, cache)
        self.assertIs(cached_reach_envelopes(deepcopy(player), 21, cache), original)
        self.assertEqual(original, reach_envelopes(player, 21))
        self.assertIsNot(cached_reach_envelopes(player, 22, cache), original)
        for field in ENVELOPE_FIELDS:
            changed = deepcopy(player)
            value = getattr(changed, field)
            setattr(changed, field, 0.5 if value is None else not value if isinstance(value, bool) else value + 0.5)
            with self.subTest(field=field):
                self.assertIsNot(cached_reach_envelopes(changed, 21, cache), original)

    def test_shared_body_projections_match_uncached_clearance_and_live_feedback_changes(self):
        state = offense_state()
        player, other = state.team1.players[0], state.team2.players[0]
        path = carry_path(player, (0, -106))
        cache = {}
        self.assertEqual(body_clearance(state, player, path, cache=cache), body_clearance(state, player, path))
        size = len(cache)
        body_clearance(state, player, path, cache=cache)
        self.assertEqual(len(cache), size)
        for field in PROJECTION_FIELDS:
            before = len(cache)
            value = getattr(other, field)
            setattr(other, field, 0.5 if value is None else value + 0.5)
            self.assertEqual(body_clearance(state, player, path, cache=cache), body_clearance(state, player, path))
            self.assertGreater(len(cache), before)

    def test_scalar_projection_matches_prior_numpy_clip_at_edges_and_nonfinite_values(self):
        def original(point):
            x = float(np.clip(point[0], -SETTINGS.scale_x, SETTINGS.scale_x))
            y = float(np.clip(point[1], -SETTINGS.scale_y, SETTINGS.scale_y))
            cx, cy = max(0.0, abs(x) - 72), max(0.0, abs(y) - 222)
            radius = math.hypot(cx, cy)
            if radius > 48:
                x, y = math.copysign(72 + cx * 48 / radius, x), math.copysign(222 + cy * 48 / radius, y)
            if abs(x) < 28 and abs(y) > 254:
                if 28 - abs(x) < abs(y) - 254:
                    x = math.copysign(28, x)
                else:
                    y = math.copysign(254, y)
            return x, y
        rng = random.Random(94)
        points = [(rng.uniform(-400, 400), rng.uniform(-400, 400)) for _ in range(1000)]
        points.extend((x, y) for x in (float('nan'), float('inf'), -float('inf'), -0.0, 28, -120)
                      for y in (float('nan'), float('inf'), -float('inf'), 254, -270))
        for point in points:
            np.testing.assert_equal(project_target(point), original(point))

    def test_timing_summary_keeps_spikes_and_cohorts_visible(self):
        rows = [{'cohort': cohort, 'timing_ms': {stage: duration for stage in (
            'predict', 'diagnostics', 'step', 'draw', 'total', 'paced')}}
                for cohort, duration in (('ai-skater-possession', 100), ('other', 5))]
        summary = timing_summary(rows)
        self.assertEqual(summary['all']['total']['over_60hz_budget'], 1)
        self.assertEqual(summary['ai-skater-possession']['total']['max_ms'], 100)
        self.assertEqual(summary['other']['total']['mean_ms'], 5)


@unittest.skipUnless(importlib.util.find_spec('pygame'), 'optional display extra is not installed')
class DisplayPerformanceParityTests(unittest.TestCase):
    def setUp(self):
        variables = patch.dict(os.environ, {'SDL_VIDEODRIVER': 'dummy', 'SDL_AUDIODRIVER': 'dummy'})
        variables.start()
        self.addCleanup(variables.stop)
        import pygame
        pygame.init()
        self.addCleanup(pygame.quit)

    def test_text_cache_keys_font_color_and_clipping_width_and_is_bounded(self):
        import pygame
        from nhl94_ai.ui.decision_panel import _text_image
        font = pygame.font.SysFont('Arial', 16)
        _text_image.cache_clear()
        expected = _text_image(font, 'carry', (255, 255, 255), 200)
        self.assertIs(_text_image(font, 'carry', (255, 255, 255), 200), expected)
        self.assertIsNot(_text_image(font, 'carry', (100, 255, 100), 200), expected)
        self.assertIsNot(_text_image(font, 'carry', (255, 255, 255), 10), expected)
        for index in range(1100):
            _text_image(font, str(index), (255, 255, 255), 200)
        self.assertIsNot(_text_image(font, 'carry', (255, 255, 255), 200), expected)

    def test_reused_scaling_buffers_match_original_pixels_and_reallocate_on_resize(self):
        import pygame
        helper = test_debug_playback.DebugPlaybackTests()
        self.addCleanup(helper.doCleanups)
        display, _ = helper.display()
        frame = np.random.default_rng(94).integers(0, 256, (224, 256, 3), dtype=np.uint8)
        display.draw_frame(frame, [{}])
        raw = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
        expected = pygame.transform.scale(raw, display.game_surf.get_size())
        np.testing.assert_array_equal(pygame.surfarray.array3d(display.game_surf), pygame.surfarray.array3d(expected))
        buffers = display._frame_surface, display._presentation_surface
        display.draw_frame(frame, [{}])
        self.assertIs(display._frame_surface, buffers[0])
        self.assertIs(display._presentation_surface, buffers[1])
        pygame.event.post(pygame.event.Event(pygame.VIDEORESIZE, w=960, h=540))
        display.process_events()
        display.draw_frame(frame, [{}])
        self.assertIsNot(display._presentation_surface, buffers[1])
        expected = pygame.transform.smoothscale(display.screen, display.presentation_rect.size)
        np.testing.assert_array_equal(pygame.surfarray.array3d(display._presentation_surface),
                                      pygame.surfarray.array3d(expected))


if __name__ == '__main__':
    unittest.main()

"""Forecasts compose with native movement without returning to observed positions."""
from copy import deepcopy
from dataclasses import replace
import math
import unittest

import numpy as np

from nhl94_ai.agents.carry import carry_path
from nhl94_ai.agents.cross_crease import _route_motion, _shot_scene
from nhl94_ai.agents.deke import _advance_preview
from nhl94_ai.agents.offense import carry_future, projected_state, reception_state
from nhl94_ai.agents.passing import evaluate_pass, option_receiver
from nhl94_ai.agents.responses import _puck_at, response_future
from nhl94_ai.agents.skating import grounded_step
from tests.support.offense import crossing_state, deke_state, offense_state, response_state


def actors(state):
    return [state.puck, *(actor for team in (state.team1, state.team2)
                          for actor in (*team.players, team.goalie))]


def precise_state(state):
    for actor in actors(state):
        actor.precise_x, actor.precise_y = float(actor.x), float(actor.y)
    return state


class ForecastCoordinatesTests(unittest.TestCase):
    def assert_continuous_step(self, projected):
        stepped = grounded_step(projected, (0, 1), goalie=projected.role == 0)
        distance = math.dist((projected.x, projected.y), (stepped.x, stepped.y))
        self.assertLess(distance, math.hypot(projected.motion_x, projected.motion_y) + 2)

    def assert_puck_position(self, puck):
        self.assertEqual((puck.precise_x, puck.precise_y), (float(puck.x), float(puck.y)))

    def test_legacy_projection_composes_without_mutating_observation(self):
        state = precise_state(offense_state())
        state.team1.players[0].motion_y = 1.6
        state.team2.players[0].motion_x = -0.75
        state.team2.goalie.motion_x = -0.5
        before = deepcopy([vars(actor) for actor in actors(state)])
        future = projected_state(state, (0, -80), frames=12)
        self.assertGreater(future.team1.players[0].y - state.team1.players[0].y, 20)
        for actor in actors(future)[1:]:
            with self.subTest(role=actor.role, position=(actor.x, actor.y)):
                self.assertEqual((actor.precise_x, actor.precise_y), (float(actor.x), float(actor.y)))
                self.assert_continuous_step(actor)
        self.assert_puck_position(future.puck)
        # The legacy carry model translates the puck but retains observed velocity.
        self.assertEqual((future.puck.motion_x, future.puck.motion_y), (0, 0))
        self.assertEqual([vars(actor) for actor in actors(state)], before)

    def test_native_carry_translates_puck_without_changing_velocity_model(self):
        state = precise_state(offense_state())
        player = state.team1.players[0]
        player.motion_y = 1.6
        before = deepcopy([vars(actor) for actor in actors(state)])
        path = carry_path(player, (0, -80), 12, decision_interval=4)
        future = carry_future(state, player, path[-1], frames=12)
        self.assert_continuous_step(future.team1.players[0])
        self.assert_puck_position(future.puck)
        self.assertEqual((future.puck.motion_x, future.puck.motion_y), (0, 0))
        self.assertEqual([vars(actor) for actor in actors(state)], before)

    def test_reception_puck_keeps_contact_point_and_receiver_velocity(self):
        state = precise_state(offense_state())
        passer, receiver = state.team1.players[:2]
        receiver.motion_x = -0.5
        option, details = evaluate_pass(state, passer, 1, receiver, 'advance')
        self.assertIsNotNone(option, details)
        before = deepcopy([vars(actor) for actor in actors(state)])
        future = reception_state(state, option)
        self.assert_puck_position(future.puck)
        self.assertEqual((future.puck.motion_x, future.puck.motion_y), (-0.5, 0))
        self.assertEqual([vars(actor) for actor in actors(state)], before)

    def test_effective_receiver_origin_composes_with_native_step(self):
        state = precise_state(offense_state())
        passer, receiver = state.team1.players[:2]
        option, details = evaluate_pass(state, passer, 1, receiver, 'advance')
        self.assertIsNotNone(option, details)
        before = deepcopy(vars(receiver))
        option = replace(option, receiver_origin=(-55.5, -10.25, 0.25, 1.0))
        future = option_receiver(receiver, option)
        self.assert_continuous_step(future)
        self.assertEqual(vars(receiver), before)
        self.assertIs(option_receiver(receiver, replace(option, receiver_origin=None)), receiver)

    def test_cpu_response_puck_follows_same_carrier_displacement(self):
        state = precise_state(response_state())
        player = state.team1.players[0]
        player.motion_y = 1.6
        path = carry_path(player, (-50, 210), 12, decision_interval=4)
        before = deepcopy([vars(actor) for actor in actors(state)])
        future, _ = response_future(state, path)
        for puck in (_puck_at(state, player, path[-1]), future.puck):
            self.assert_puck_position(puck)
            self.assertEqual((puck.motion_x, puck.motion_y), (path[-1].motion_x, path[-1].motion_y))
        self.assertEqual([vars(actor) for actor in actors(state)], before)

    def test_crossing_route_and_shot_scene_compose_with_native_step(self):
        state = precise_state(crossing_state())
        before = deepcopy([vars(actor) for actor in actors(state)])
        _, carrier, _ = list(_route_motion(state.team1.players[0], ((70, 212),), horizon=12))[-1]
        self.assert_continuous_step(carrier)
        scene = _shot_scene(state, carrier, 1, 12)
        for actor in (*scene.team2.players, scene.team2.goalie):
            self.assert_continuous_step(actor)
        self.assertEqual([vars(actor) for actor in actors(state)], before)

    def test_deke_preview_updates_puck_precise_coordinates_each_step(self):
        state = precise_state(deke_state())
        state.team1.players[0].motion_y = 1.6
        before = deepcopy([vars(actor) for actor in actors(state)])
        scene = deepcopy(state)
        for _ in range(12):
            _advance_preview(scene, 0, np.zeros(12, dtype=np.int8))
            self.assert_puck_position(scene.puck)
        self.assertEqual([vars(actor) for actor in actors(state)], before)

"""Native three-bot table gaps must not erase a confirmed pass in flight."""
import unittest

from nhl94_ai.agents.teammate_receiving import PassCommitment, SharedReceiving
from tests.support.offense import offense_state


def flight():
    state = offense_state()
    state.engine.puck_owner = -1
    state.team1.pass_attempts = 0
    receiver = SharedReceiving()
    receiver.last_count = 0
    receiver.pending = PassCommitment(0, 1, 1, 2, (40, 180), 10, 100, 'support', 0, 0, 0, 0,
                                     phase='flight', released=12)
    return state, receiver


class TransientReceivingTests(unittest.TestCase):
    def test_confirmed_flight_survives_unknown_table_then_resumes_known_receiver(self):
        state, receiving = flight()
        plan = receiving.pending
        receiving.observe(state, [-1, -1, -1], 13)
        self.assertIs(receiving.pending, plan)
        receiving.observe(state, [0, 1, 2], 14)
        self.assertIs(receiving.pending, plan)
        state.engine.puck_owner = 1
        receiving.observe(state, [0, 1, 2], 15)
        self.assertEqual(receiving.pending.phase, 'received')
        self.assertEqual(receiving.pending.received, 15)

    def test_confirmed_selection_loss_and_native_endings_still_cancel(self):
        for reason in ('known-loss', 'partial-unknown', 'interception', 'stoppage', 'expiry', 'unreleased'):
            state, receiving = flight()
            slots, frame = [-1, -1, -1], 13
            if reason == 'known-loss': slots = [0, 2, 3]
            if reason == 'partial-unknown': slots = [0, 2, -1]
            if reason == 'interception': state.engine.puck_owner = 6
            if reason == 'stoppage': state.engine.clock_stopped = True
            if reason == 'expiry': frame = 100
            if reason == 'unreleased': receiving.pending.phase = 'release'; receiving.pending.released = None
            receiving.observe(state, slots, frame)
            with self.subTest(reason=reason):
                self.assertIsNone(receiving.pending)


    def test_comparison_option_disables_only_transient_preservation(self):
        state, receiving = flight()
        self.assertTrue(receiving.preserve_transient)
        receiving.preserve_transient = False
        receiving.observe(state, [-1, -1, -1], 13)
        self.assertIsNone(receiving.pending)

    def test_coordinator_and_parsers_enable_preservation_only_for_coordinated_play(self):
        from nhl94_ai.agents.teammates import TeammateCoordinator
        from nhl94_ai.agents.registry import add_classic_arguments
        from nhl94_ai.evaluation.teammates import build_parser
        from argparse import ArgumentParser
        from tests.support.teammates import team_args
        self.assertTrue(TeammateCoordinator(team_args()).receiving.preserve_transient)
        self.assertTrue(TeammateCoordinator(team_args(teammate_transient_receiving=True)).receiving.preserve_transient)
        self.assertFalse(TeammateCoordinator(team_args(teammate_transient_receiving=False)).receiving.preserve_transient)
        self.assertFalse(TeammateCoordinator(team_args(), strategy='independent').receiving.preserve_transient)
        with self.assertRaises(ValueError):
            TeammateCoordinator(team_args(teammate_transient_receiving=True), strategy='independent')
        gameplay = ArgumentParser()
        add_classic_arguments(gameplay)
        for parser, required in ((build_parser(), ['--output', '/tmp/unused']), (gameplay, [])):
            with self.subTest(parser=parser):
                default = parser.parse_args(required)
                self.assertIsNone(default.teammate_transient_receiving)
                self.assertTrue(TeammateCoordinator(default).receiving.preserve_transient)
                self.assertFalse(TeammateCoordinator(default, strategy='independent').receiving.preserve_transient)
                self.assertTrue(parser.parse_args(required + ['--teammate-transient-receiving']).teammate_transient_receiving)
                self.assertFalse(parser.parse_args(required + ['--no-teammate-transient-receiving']).teammate_transient_receiving)

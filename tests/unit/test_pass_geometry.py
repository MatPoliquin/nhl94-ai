"""Live GetHot feedback and passto's signed, quantized launch arithmetic."""
import json
import math
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from tempfile import TemporaryDirectory

from nhl94_ai.agents.motion import VELOCITY_SCALE
from nhl94_ai.agents.passing import (
    _divide, _rom_sqrt, evaluate_pass, pass_contact, pass_release_frames, rom_pass_vector,
)
from nhl94_ai.game.ram import _animation_frames, _animation_rom, _stick_hotspots, pass_geometry_info
from nhl94_ai.game.state import NHL94GameState
from tests.ram_fixture import FixtureMemory
from tests.unit.test_classic_offense import offense_state


def geometry_env(game='NHL94-Genesis-v0'):
    return SimpleNamespace(unwrapped=SimpleNamespace(gamename=game),
                           data=SimpleNamespace(memory=FixtureMemory({})))


class SpriteHotspotTests(unittest.TestCase):
    def setUp(self):
        self.env = geometry_env()
        table = bytearray(12)
        table[6:8] = bytes((24, 252))
        self.table = bytes(table)
        self.env.data.memory.assign(0xFFB04A + 6, '>i2', 3)
        self.shots = tuple(((4,) * 8, ((0, 0),)) for _ in range(8))
        self.mock_shots = patch('nhl94_ai.game.ram._shot_animations', return_value=self.shots)
        self.mock_shots.start()
        self.addCleanup(self.mock_shots.stop)

    def _read(self, flags, *, sflags=0):
        self.env.data.memory.assign(0xFFB04A + 4, '|u1', flags)
        with patch('nhl94_ai.game.ram._stick_hotspots', return_value=self.table):
            return pass_geometry_info(self.env, {'sflags': sflags})

    def test_live_sprite_flip_and_horizontal_view_follow_gethot(self):
        for flags, expected in ((0, (24, 4)), (8, (-24, 4)), (16, (24, -4)), (24, (-24, -4))):
            with self.subTest(flags=flags):
                info = self._read(flags)
                self.assertEqual((info['offense_0_stick_x'], info['offense_0_stick_y']), expected)
                rotated = self._read(flags, sflags=0x8000)
                self.assertEqual((rotated['offense_0_stick_x'], rotated['offense_0_stick_y']),
                                 (expected[1], -expected[0]))

    def test_nonpositive_sprite_returns_body_and_invalid_frame_raises(self):
        for frame in (0, -1):
            self.env.data.memory.assign(0xFFB04A + 6, '>i2', frame)
            info = self._read(0)
            self.assertEqual((info['offense_0_stick_x'], info['offense_0_stick_y']), (0, 0))
        self.env.data.memory.assign(0xFFB04A + 6, '>i2', 10)
        with self.assertRaisesRegex(ValueError, 'Invalid stick sprite'):
            self._read(0)

    def test_variants_only_decode_existing_skater_slots_and_do_not_mutate_info(self):
        for game, count in (('NHL941on1-Genesis-v0', 1), ('NHL942on2-Genesis-v0', 2), ('NHL94-Genesis-v0', 5)):
            with self.subTest(game=game):
                env = geometry_env(game)
                original = {'p1_vel_x': 15, 'sflags': 0}
                with patch('nhl94_ai.game.ram._stick_hotspots', return_value=self.table):
                    info = pass_geometry_info(env, original)
                slots = [*range(count), *range(6, 6 + count)]
                expected = {f'offense_{slot}_{field}' for slot in slots
                            for field in ('stick_x', 'stick_y', 'shot_durations', 'shot_offsets_y', 'shot_offsets_x',
                                          'sprite_flipped_x', 'shot_projection_offsets_x', 'shot_projection_offsets_y',
                                          'shot_projection_durations')}
                self.assertEqual(set(info) - set(original), expected)
                self.assertEqual(original, {'p1_vel_x': 15, 'sflags': 0})

    def test_optional_feedback_loads_without_entering_normalized_neural_fields(self):
        info = json.loads((Path(__file__).resolve().parents[1] /
                           'fixtures/NHL94-Genesis-v0.json').read_text(encoding='utf-8'))
        info.update(defense_control1=0, defense_team1=1, defense_team2=0,
                    offense_0_stick_x=-24, offense_0_stick_y=4,
                    offense_0_shot_durations=(4,) * 8, offense_0_shot_offsets_y=(12, 20),
                    offense_0_shot_offsets_x=(-18, 24),
                    offense_0_shot_projection_offsets_x=(-24, 24),
                    offense_0_shot_projection_offsets_y=(-24, 24),
                    offense_0_shot_projection_durations=(4,) * 8,
                    offense_0_sprite_flipped_x=True,
                    defense_0_precise_x=-98304, defense_0_precise_y=786560,
                    defense_0_facing_phase=98304,
                    g1_control_precise_x=16384, g1_control_precise_y=-16379904,
                    g1_control_decision_timer=7, g1_control_decision_delay=41,
                    g1_control_steering=6, g1_control_flags=0x20, g1_control_unavailable=0)
        state = NHL94GameState(5)
        state.BeginFrame(info, [0] * 6)
        self.assertEqual((state.team1.players[0].stick_x, state.team1.players[0].stick_y), (-24, 4))
        self.assertIsNone(state.team2.players[0].stick_x)
        self.assertIsNone(state.team1.nz_players[0].stick_x)
        self.assertEqual(state.team1.players[0].shot_offsets_y, (12, 20))
        self.assertEqual(state.team1.players[0].shot_offsets_x, (-18, 24))
        self.assertIsNone(state.team1.nz_players[0].shot_offsets_x)
        self.assertIsNone(state.team1.nz_players[0].shot_offsets_y)
        self.assertIsNone(state.team1.nz_players[0].shot_durations)
        player, normalized = state.team1.players[0], state.team1.nz_players[0]
        self.assertEqual((player.precise_x, player.precise_y, player.facing_phase), (-1.5, 12 + 128 / 65536, 1.5))
        self.assertTrue(player.sprite_flipped_x)
        for field in ('precise_x', 'precise_y', 'facing_phase', 'sprite_flipped_x',
                      'shot_projection_offsets_x', 'shot_projection_offsets_y', 'shot_projection_durations'):
            self.assertIsNone(getattr(normalized, field), field)
        goalie = state.team1.goalie
        self.assertEqual((goalie.precise_x, goalie.precise_y), (0.25, -250 + 4096 / 65536))
        self.assertEqual((goalie.decision_timer, goalie.decision_interval, goalie.steering), (7, 10, 6))
        self.assertFalse(goalie.cpu_tracking_active)
        self.assertIsNone(state.team1.nz_goalie.decision_timer)
        del info['offense_0_stick_x']
        state.BeginFrame(info, [0] * 6)
        self.assertIsNone(state.team1.players[0].stick_x)
        del info['defense_control1']
        state.BeginFrame(info, [0] * 6)
        self.assertIsNone(state.team1.goalie.decision_timer)
        self.assertIsNone(state.team1.goalie.cpu_tracking_active)

    def test_release_x_envelope_includes_old_sprite_and_respects_flips_and_rotation(self):
        shots = tuple(((4,) * 8, ((-2, -3), (5, 6))) for _ in range(8))
        with patch('nhl94_ai.game.ram._shot_animations', return_value=shots), \
                patch('nhl94_ai.game.ram._animation_frames', return_value=tuple(((3, 4),) for _ in range(8))):
            for flags in (0, 8, 16, 24):
                for rotated in (False, True):
                    info = self._read(flags, sflags=0x8000 if rotated else 0)
                    points = [(-sx if flags & 8 else sx, sy if flags & 16 else -sy)
                              for sx, sy in ((-2, -3), (5, 6), (24, -4))]
                    if rotated:
                        points = [(sy, -sx) for sx, sy in points]
                    self.assertEqual(info['offense_0_shot_offsets_x'],
                                     (min(sx for sx, _ in points), max(sx for sx, _ in points)))
                    self.assertEqual(info['offense_0_shot_offsets_y'],
                                     (min(sy for _, sy in points), max(sy for _, sy in points)))

    def test_romless_fixture_retains_explicitly_optional_geometry(self):
        env = SimpleNamespace(unwrapped=SimpleNamespace())
        info = {'p1_vel_x': 15}
        self.assertIs(pass_geometry_info(env, info), info)

    def test_rom_table_is_read_only_cached_and_requires_verified_gethot_signature(self):
        signature = bytes.fromhex('48e700c0206f000c424042414a6800066f00')
        rom = bytearray(80)
        rom[:len(signature)] = signature
        rom[20:26] = bytes.fromhex('227c00000028')
        rom[40:44] = bytes((0, 0, 24, 252))
        _stick_hotspots.cache_clear()
        self.addCleanup(_stick_hotspots.cache_clear)
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'fixture.rom'
            path.write_bytes(rom)
            with patch('stable_retro.data.get_romfile_path', return_value=str(path)) as locate:
                self.assertEqual(_stick_hotspots('fixture')[:4], bytes((0, 0, 24, 252)))
                self.assertEqual(_stick_hotspots('fixture')[:4], bytes((0, 0, 24, 252)))
                locate.assert_called_once()
                self.assertEqual(path.read_bytes(), rom)
                path.write_bytes(bytes(rom[1:]))
                with self.assertRaisesRegex(ValueError, 'Unsupported GetHot'):
                    _stick_hotspots('unsupported')

    def test_animation_offsets_are_signed_and_long_native_pose_durations_are_valid(self):
        rom = bytearray(0x7000)
        base = 0x5B1C + 0x50C
        for direction in range(8):
            rom[base + direction * 2:base + direction * 2 + 2] = (-20).to_bytes(2, 'big', signed=True)
        rom[base - 20:base - 12] = bytes.fromhex('000300780004ffff')
        _animation_frames.cache_clear()
        _animation_rom.cache_clear()
        self.addCleanup(_animation_frames.cache_clear)
        self.addCleanup(_animation_rom.cache_clear)
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'animation.rom'
            path.write_bytes(rom)
            frames = _animation_frames('fixture', 0x50C, str(path))
        self.assertEqual(frames, (((3, 120), (4, -1)),) * 8)


class IntegerPassGeometryTests(unittest.TestCase):
    def test_passto_vectors_retain_signed_word_operations_and_quantized_time(self):
        cases = (
            ((55, -30, 3393, 7518, 22), (18295, -819)),
            ((55, -28, 3558, 7401, 22), (18568, -273)),
            ((51, -23, 5167, 5410, 22), (18841, -1092)),
            ((17, -31, 425, 4787, 22), (9284, -12561)),
            ((-53, -56, -425, 8410, 22), (-14745, -7099)),
            ((-13, 49, -1508, -1848, 30), (-8192, 25122)),
        )
        for inputs, expected in cases:
            with self.subTest(inputs=inputs):
                self.assertEqual(rom_pass_vector(*inputs), expected)
        self.assertEqual(_divide(-5, 2), -2)
        self.assertEqual(_divide(5, -2), -2)
        self.assertEqual(_divide(-5, -2), 2)

    def test_rom_square_root_matches_observed_integer_algorithm(self):
        for value in (0, 1, 2, 1599, 1600, 1601, 2500, 0xF00000, 0xF00001, 0xFFFFFFFF):
            with self.subTest(value=value):
                root = _rom_sqrt(value)
                self.assertLessEqual(abs(root - math.isqrt(value)), 1)
        self.assertEqual(_rom_sqrt(0xFFFFFFFF), 65534)

    def test_live_contact_uses_rom_hotspot_and_launch_not_ideal_interception(self):
        state = offense_state()
        passer, receiver = state.team1.players[:2]
        receiver.stick_x, receiver.stick_y = -24, 4
        with patch('nhl94_ai.agents.passing.rom_pass_vector', return_value=(-10000, 10000)) as launch:
            pass_contact(state.puck, passer, receiver)
        self.assertEqual(launch.call_args.args, (-104, 94, 0, 0, 20))
        receiver.stick_x = receiver.stick_y = None
        with patch('nhl94_ai.agents.passing.rom_pass_vector') as launch:
            self.assertIsNotNone(pass_contact(state.puck, passer, receiver))
        launch.assert_not_called()

    def test_launch_that_cannot_reach_live_stick_is_not_a_safe_reception(self):
        state = offense_state()
        passer, receiver = state.team1.players[:2]
        receiver.stick_x = receiver.stick_y = 0
        with patch('nhl94_ai.agents.passing.rom_pass_vector', return_value=(10000, 0)):
            self.assertIsNone(pass_contact(state.puck, passer, receiver))
        with patch('nhl94_ai.agents.passing.rom_pass_vector', return_value=None):
            self.assertIsNone(pass_contact(state.puck, passer, receiver))

    def test_live_reception_speed_accounts_for_flight_friction(self):
        state = offense_state()
        passer, receiver = state.team1.players[:2]
        receiver.x, receiver.y = -80, -120
        receiver.stick_x = receiver.stick_y = 0
        with patch('nhl94_ai.agents.passing.rom_pass_vector', return_value=(-10000, 0)):
            contact = pass_contact(state.puck, passer, receiver)
        self.assertIsNotNone(contact)
        self.assertEqual(contact[3], 10000 * VELOCITY_SCALE * state.puck.friction**32)
        self.assertEqual(contact[2], 32)

    def test_live_launch_and_collision_checks_share_release_timing(self):
        state = offense_state()
        passer, receiver = state.team1.players[:2]
        self.assertEqual(pass_release_frames(receiver), 4)
        receiver.stick_x, receiver.stick_y = 0, 10
        self.assertEqual(pass_release_frames(receiver), 2)
        self.assertEqual(pass_release_frames(receiver, 1), 1)
        contact = ((0, -120), (-80, -20), 10, 2)
        with patch('nhl94_ai.agents.passing.pass_contact', return_value=contact), \
                patch('nhl94_ai.agents.passing._swept_contact', return_value=False) as sweep:
            option, details = evaluate_pass(state, passer, 1, receiver)
        self.assertIsNotNone(option, details)
        self.assertEqual(details['release_frames'], 2)
        times = [call.args[3] for call in sweep.call_args_list]
        self.assertEqual(times[0], (2, 6))
        self.assertEqual(times[-1][-1], 12)

    def test_stick_can_intercept_a_lane_that_the_body_does_not_cover(self):
        state = offense_state()
        passer, receiver = state.team1.players[:2]
        opponent = state.team2.players[0]
        opponent.x, opponent.y = -15, -75
        with patch('nhl94_ai.agents.passing.arrival_time', return_value=90):
            option, details = evaluate_pass(state, passer, 1, receiver)
            self.assertIsNotNone(option, details)
            opponent.stick_x, opponent.stick_y = -18, -8
            option, details = evaluate_pass(state, passer, 1, receiver)
        self.assertIsNone(option)
        self.assertEqual(details['status'], 'moving-stick-interception')

    def test_friendly_stick_does_not_silently_steal_the_intended_reception(self):
        state = offense_state()
        passer, receiver = state.team1.players[:2]
        friend = state.team1.players[2]
        friend.x, friend.y = -15, -75
        with patch('nhl94_ai.agents.passing.arrival_time', return_value=90):
            option, details = evaluate_pass(state, passer, 1, receiver)
            self.assertIsNotNone(option, details)
            friend.stick_x, friend.stick_y = -18, -8
            option, details = evaluate_pass(state, passer, 1, receiver)
        self.assertIsNone(option)
        self.assertEqual(details['status'], 'friendly-stick-obstruction')


if __name__ == '__main__':
    unittest.main()

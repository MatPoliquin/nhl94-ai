"""Named input selection, compatibility and compact player-centered geometry."""
import copy
from dataclasses import replace
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from nhl94_ai.artifacts import run_metadata, validate_schema
from nhl94_ai.cli import parse_configured
from nhl94_ai.env.encoding import (
    DEFAULT_MODEL_INPUT_GROUPS, ObservationEncoder, _normalize_model_input_config,
    init_model, set_model_input,
)
from nhl94_ai.game.ram import GOALIE_INPUT_ATTRIBUTES, SKATER_INPUT_ATTRIBUTES
from nhl94_ai.game.state import NHL94GameState
from nhl94_ai.model_inputs import default_model_input_path, load_model_input
from nhl94_ai.tasks.registry import get_task, resolve_task_model_input as resolve_model_input
from nhl94_ai.training.bc import _compute_element_weights
from nhl94_ai.training.live import build_parser, prepare_args
from tests.unit.test_environment_contracts import FixtureEnv
from nhl94_ai.env.observation import NHL94Observation2PEnv


def input_state(skaters=5):
    state = NHL94GameState(skaters)
    state.team1.control = 1
    state.team1.defense_control = 0
    state.update_nets()
    for index, player in enumerate(state.team1.players):
        player.x, player.y = 20 + index * 10, 180
        player.anim, player.anim_frame = 500, 3
        player.input_state_known = True
        player.input_state = {name: scale // 2 for name, (_, scale) in SKATER_INPUT_ATTRIBUTES.items()}
        player.input_state.update(vx=8192, vy=-4096, facing=2, has_puck=int(index == 0))
    goalie = state.team2.goalie
    goalie.x, goalie.y = 5, 245
    goalie.input_state_known = True
    goalie.input_state = {name: scale // 2 for name, (_, scale) in GOALIE_INPUT_ATTRIBUTES.items()}
    goalie.input_state.update(vx=4096, vy=2048, facing=4, has_puck=0)
    return state


def field_index(group, field, config):
    index = 0
    for name, fields in config['groups'].items():
        if name == group:
            return index + fields.index(field)
        index += len(fields)
    raise AssertionError(group)


class InputRegistryTests(unittest.TestCase):
    def test_editable_and_packaged_registries_match(self):
        root = Path(__file__).resolve().parents[2]
        editable = json.loads((root / 'configs/model_input.json').read_text(encoding='utf-8'))
        packaged = json.loads(Path(default_model_input_path()).read_text(encoding='utf-8'))
        self.assertEqual(editable, packaged)
        self.assertEqual(set(editable), {'default', 'pvg', 'pvg-v3', 'legacy'})

    def test_legacy_and_default_preserve_existing_ordered_fields(self):
        self.assertEqual(_normalize_model_input_config('legacy'), DEFAULT_MODEL_INPUT_GROUPS)
        expected = copy.deepcopy(DEFAULT_MODEL_INPUT_GROUPS)
        expected['friendly_player'].remove('is_controlled')
        expected['friendly_player'].insert(12, 'is_controlled')
        expected['friendly_goalie'].remove('is_controlled')
        expected['friendly_goalie'].insert(10, 'is_controlled')
        self.assertEqual(_normalize_model_input_config('default'), expected)
        self.assertEqual(init_model(5, 'default'), 310)
        self.assertEqual(init_model(5, 'legacy'), 310)
        self.assertEqual(_normalize_model_input_config(None), DEFAULT_MODEL_INPUT_GROUPS)

    def test_old_pvg_fields_remain_available_without_reinterpreting_weights(self):
        old = load_model_input('pvg-v3')
        current = load_model_input('pvg')
        self.assertEqual(old['schema_version'], 3)
        self.assertEqual(current['schema_version'], 4)
        self.assertEqual(init_model(5, old), 50)
        self.assertEqual(current['groups']['player'][:-2], old['groups']['player'])
        state = input_state()
        np.testing.assert_array_equal(set_model_input(state, old),
                                      (*set_model_input(state, current)[:23],
                                       *set_model_input(state, current)[25:]))
        args = SimpleNamespace(env='NHL94-Genesis-v0', nn='MlpPolicy', rf='PvG')
        previous = run_metadata(SimpleNamespace(**vars(args), model_input='pvg-v3'))
        with self.assertRaisesRegex(ValueError, 'Incompatible artifact'):
            validate_schema(previous, run_metadata(args))

    def test_pvg_task_default_and_explicit_reuse_or_override(self):
        params = {'model_input': load_model_input('default')}
        selected = resolve_model_input(SimpleNamespace(rf='PvG'), params)
        self.assertEqual(selected['model_input']['variant'], 'pvg')
        overridden = resolve_model_input(SimpleNamespace(rf='PvG', model_input='default'), params)
        self.assertEqual(overridden['model_input']['variant'], 'default')
        reused = resolve_model_input(SimpleNamespace(rf='PostPlay', model_input='pvg'), params)
        self.assertEqual(reused['model_input']['variant'], 'pvg')
        task = replace(get_task('PostPlay'), model_input_variant='pvg')
        self.assertEqual(resolve_model_input(SimpleNamespace(), params, task=task)['model_input']['variant'], 'pvg')
        self.assertEqual(params['model_input']['variant'], 'default')

    def test_inline_legacy_definitions_remain_supported(self):
        inline = {'schema_version': 2, 'groups': {'puck': ['x', 'y']}}
        params = {'model_input': inline}
        resolved = resolve_model_input(SimpleNamespace(rf='PvG'), params)
        self.assertEqual(_normalize_model_input_config(resolved['model_input']),
                         _normalize_model_input_config(inline))
        args = SimpleNamespace(env='NHL94-Genesis-v0', nn='MlpPolicy')
        old = run_metadata(args, params)
        validate_schema(old, run_metadata(args, copy.deepcopy(params)))

    def test_custom_registry_variants_and_declaring_directory(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            registry = json.loads(Path(default_model_input_path()).read_text(encoding='utf-8'))
            registry['small'] = copy.deepcopy(registry['pvg'])
            registry['small']['groups'] = {'player': ['speed'], 'goalie': [], 'net': ['left_x']}
            (root / 'inputs.json').write_text(json.dumps(registry), encoding='utf-8')
            config = {'options': {
                'env': 'NHL94-Genesis-v0', 'nn': 'MlpPolicy', 'rf': 'PostPlay',
                'model_input': 'small', 'model_input_config': 'inputs.json',
            }}
            (root / 'run.json').write_text(json.dumps(config), encoding='utf-8')
            args = prepare_args(parse_configured(build_parser(), ['--config', str(root / 'run.json')]))
            self.assertEqual(args.model_input_config, str(root / 'inputs.json'))
            self.assertEqual(init_model(5, args.hyperparams_dict['model_input']), 2)
            self.assertEqual(len(set_model_input(input_state(), args.hyperparams_dict['model_input'])), 2)
            changed = load_model_input('small', root / 'inputs.json')
            changed['groups']['player'].append('agility')
            self.assertEqual(load_model_input('small', root / 'inputs.json')['groups']['player'], ['speed'])

    def test_invalid_names_layouts_fields_and_registries_fail_explicitly(self):
        with self.assertRaisesRegex(ValueError, 'Unknown model input variant'):
            load_model_input('missing')
        with self.assertRaisesRegex(TypeError, 'variant name or an inline JSON object'):
            resolve_model_input(SimpleNamespace(), {'model_input': False})
        for update in ({'layout': 'unknown'}, {'normalization': 'unknown'},
                       {'groups': {'player': ['unknown']}},
                       {'groups': {'player': ['speed', 'speed']}},
                       {'groups': {'buttons': []}}):
            config = load_model_input('pvg')
            config.update(update)
            with self.subTest(update=update), self.assertRaises(ValueError):
                init_model(5, config)
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'invalid.json'
            path.write_text('{}', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'default variant'):
                load_model_input('default', path)

    def test_artifacts_reject_wrong_fields_layout_or_reference_frame(self):
        args = SimpleNamespace(env='NHL94-Genesis-v0', nn='MlpPolicy', rf='PvG')
        pvg = run_metadata(args)
        default = run_metadata(SimpleNamespace(**vars(args), model_input='default'))
        with self.assertRaisesRegex(ValueError, 'Incompatible artifact'):
            validate_schema(default, pvg)
        for key, value in (
            ('normalization', 'other-frame'), ('observation_layout', 'other-layout'),
            ('observation_fields', {'player': list(reversed(pvg['schema']['observation_fields']['player']))}),
        ):
            changed = copy.deepcopy(pvg)
            changed['schema'][key] = value
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, key):
                validate_schema(changed, pvg)

    def test_saved_configuration_is_a_snapshot_not_a_mutable_registry_reference(self):
        args = SimpleNamespace(rf='PvG')
        params = resolve_model_input(args, {})
        params['model_input']['groups']['player'] = ['speed']
        self.assertEqual(resolve_model_input(args, params)['model_input']['groups']['player'], ['speed'])

    def test_partial_definitions_materialize_inherited_fields_in_the_snapshot(self):
        config = load_model_input('pvg')
        config.pop('variant')
        config['groups'] = {'player': ['speed']}
        params = resolve_model_input(SimpleNamespace(), {'model_input': config})
        snapshot = params['model_input']
        self.assertEqual(list(snapshot['groups']), ['player', 'goalie', 'net'])
        self.assertEqual(init_model(5, snapshot), 28)
        changed_defaults = load_model_input('pvg')
        changed_defaults['groups'] = {'player': [], 'goalie': [], 'net': []}
        with patch('nhl94_ai.env.compact_input.load_model_input', return_value=changed_defaults):
            self.assertEqual(init_model(5, snapshot), 28)


class CompactInputTests(unittest.TestCase):
    def setUp(self):
        self.config = load_model_input('pvg')

    def value(self, state, group, field):
        return set_model_input(state, self.config)[field_index(group, field, self.config)]

    def test_exact_size_is_independent_of_roster_size_and_has_only_three_blocks(self):
        self.assertEqual(list(self.config['groups']), ['player', 'goalie', 'net'])
        self.assertEqual([len(fields) for fields in self.config['groups'].values()], [25, 22, 5])
        for skaters in (1, 2, 5):
            self.assertEqual(init_model(skaters, self.config), 52)
            values = np.asarray(set_model_input(input_state(skaters), self.config))
            self.assertEqual(values.shape, (52,))
            self.assertTrue(np.isfinite(values).all())
            self.assertTrue((np.abs(values) <= 1).all())

    def test_position_and_velocity_geometry_are_player_relative(self):
        state = input_state()
        self.assertAlmostEqual(self.value(state, 'goalie', 'x'), (5 - 20) / 240)
        self.assertAlmostEqual(self.value(state, 'goalie', 'y'), (245 - 180) / 540)
        self.assertAlmostEqual(self.value(state, 'player', 'vx'), 8192 / 32768)
        self.assertAlmostEqual(self.value(state, 'goalie', 'vx'), (4096 - 8192) / 65536)
        self.assertAlmostEqual(self.value(state, 'net', 'left_x'), (-19 - 20) / 240)
        self.assertAlmostEqual(self.value(state, 'net', 'right_y'), (264 - 180) / 540)

    def test_translation_and_unused_entities_do_not_change_inputs(self):
        original = input_state()
        moved = copy.deepcopy(original)
        for entity in (moved.team1.players[0], moved.team2.goalie):
            entity.x += 35
            entity.y -= 100
        moved.team2.net.left += 35
        moved.team2.net.right += 35
        moved.team2.net.y -= 100
        for entity in [*moved.team1.players[1:], *moved.team2.players, moved.team1.goalie, moved.puck]:
            entity.x, entity.y, entity.vx, entity.vy = 900, 900, 90, 90
        moved.engine.pass_speed = 255
        np.testing.assert_array_equal(set_model_input(original, self.config),
                                      set_model_input(moved, self.config))

    def test_facing_changes_do_not_rotate_goalie_or_net_coordinates(self):
        state = input_state()
        before = set_model_input(state, self.config)
        state.team1.players[0].input_state['facing'] = 0
        after = set_model_input(state, self.config)
        self.assertNotEqual(before[field_index('player', 'facing_x', self.config)],
                            after[field_index('player', 'facing_x', self.config)])
        for group, fields in (('goalie', ('x', 'y')), ('net', ('left_x', 'left_y', 'right_x', 'right_y'))):
            for field in fields:
                index = field_index(group, field, self.config)
                self.assertEqual(before[index], after[index])

    def test_live_attributes_use_role_specific_encodings(self):
        state = input_state()
        player, goalie = state.team1.players[0], state.team2.goalie
        player.input_state.update(shot_power=30, weight=80, defensive_delay=7, handedness=1)
        goalie.input_state.update(puck_control=20, glove_left=9, stick_right=25)
        self.assertEqual(self.value(state, 'player', 'shot_power'), 1.0)
        self.assertAlmostEqual(self.value(state, 'player', 'weight'), 80 / 120)
        self.assertAlmostEqual(self.value(state, 'player', 'defensive_delay'), 7 / 15)
        self.assertEqual(self.value(state, 'player', 'handedness'), 1.0)
        self.assertAlmostEqual(self.value(state, 'goalie', 'puck_control'), 20 / 30)
        self.assertAlmostEqual(self.value(state, 'goalie', 'glove_left'), 9 / 15)
        self.assertAlmostEqual(self.value(state, 'goalie', 'stick_right'), 25 / 30)

    def test_current_control_changes_the_single_player_not_the_shape(self):
        state = input_state()
        state.team1.defense_control = 1
        state.team1.players[1].input_state['shot_power'] = 30
        self.assertEqual(self.value(state, 'player', 'shot_power'), 1.0)
        self.assertAlmostEqual(self.value(state, 'goalie', 'x'), (5 - 30) / 240)
        state.team1.players[0].x = 900
        self.assertAlmostEqual(self.value(state, 'goalie', 'x'), (5 - 30) / 240)

    def test_unselected_or_absent_entities_have_explicit_presence_masks(self):
        state = input_state()
        state.team1.defense_control = -1
        values = set_model_input(state, self.config)
        self.assertEqual(values[:25], (0.0,) * 25)
        self.assertEqual(self.value(state, 'goalie', 'is_present'), 1.0)
        self.assertEqual(values[-5:], (0.0,) * 5)
        state.team1.defense_control = 0
        state.team1.players[0].input_state = None
        self.assertEqual(set_model_input(state, self.config)[:25], (0.0,) * 25)
        state.team2.goalie.input_state = None
        self.assertEqual(set_model_input(state, self.config)[25:47], (0.0,) * 22)

    def test_c_feedback_is_observable_and_hold_duration_saturates(self):
        state = input_state()
        state.c_pressed = True
        state.c_frames_held = 12
        self.assertEqual(self.value(state, 'player', 'c_pressed'), 1.0)
        self.assertEqual(self.value(state, 'player', 'c_frames_held'), 0.2)
        state.c_frames_held = 80
        self.assertEqual(self.value(state, 'player', 'c_frames_held'), 1.0)
        state.team1.defense_control = -1
        self.assertEqual(self.value(state, 'player', 'is_controlled'), 0.0)
        self.assertEqual(self.value(state, 'player', 'c_pressed'), 1.0)

    def test_missing_feedback_and_nonfinite_values_are_errors(self):
        state = input_state()
        state.team1.players[0].input_state_known = False
        with self.assertRaisesRegex(ValueError, 'live RAM feedback'):
            set_model_input(state, self.config)
        state = input_state()
        state.team2.goalie.x = float('nan')
        with self.assertRaisesRegex(ValueError, 'nonfinite'):
            set_model_input(state, self.config)

    def test_other_tasks_can_use_pvg_without_changing_their_rewards(self):
        args = SimpleNamespace(env='NHL94-Genesis-v0', nn='MlpPolicy', action_type='FILTERED',
                               model_input='pvg', hyperparams_dict={})
        env = NHL94Observation2PEnv(FixtureEnv(), args, 1, 'PostPlay')
        self.addCleanup(env.close)
        observation, _ = env.reset(seed=0)
        self.assertEqual(np.asarray(observation).shape, (52,))
        _, reward, terminated, _, _ = env.step(np.zeros(12, dtype=np.int8))
        self.assertEqual(reward, 0.0)
        self.assertFalse(terminated)
        self.assertIs(env.task.reward, get_task('PostPlay').reward)

    def test_encoder_is_task_independent_and_bc_does_not_treat_attributes_as_buttons(self):
        encoder = ObservationEncoder(get_task('PostPlay'), 5, self.config)
        self.assertEqual(encoder.size, 52)
        observations = np.zeros((1, 52), dtype=np.float32)
        actions = np.zeros((1, 12), dtype=np.float32)
        np.testing.assert_array_equal(_compute_element_weights(observations, actions, 1, 5, self.config),
                                      np.ones_like(actions))
        with self.assertRaisesRegex(ValueError, 'button fields'):
            _compute_element_weights(observations, actions, 2, 5, self.config)

    def test_bc_release_weights_follow_the_selected_default_field_order(self):
        config = load_model_input('default')
        observations = np.zeros((1, 310), dtype=np.float32)
        actions = np.zeros((1, 12), dtype=np.float32)
        observations[0, 294 + config['groups']['buttons'].index('c')] = 1
        weights = _compute_element_weights(observations, actions, 3, 5, config)
        expected = np.ones_like(actions)
        expected[0, 8] = 3
        np.testing.assert_array_equal(weights, expected)

    def test_opponent_schema_and_mirrored_feedback_are_preserved(self):
        args = SimpleNamespace(env='NHL94-Genesis-v0', nn='MlpPolicy', action_type='FILTERED',
                               model_input='default', hyperparams_dict={})
        env = NHL94Observation2PEnv(FixtureEnv(), args, 1, 'PostPlay')
        self.addCleanup(env.close)
        env.reset(seed=0)
        model = SimpleNamespace(_nhl94_metadata={
            'schema': {'action_type': 'FILTERED'}, 'hyperparams': {'model_input': self.config},
        })
        with patch('nhl94_ai.env.observation.load_model_for_inference', return_value=model):
            env.set_opponent_model('opponent.zip')
        self.assertEqual(env.opponent_encoder.size, 52)
        memory = env.unwrapped.data.memory
        memory.assign(0xFFC32A, '>u2', 2)
        memory.assign(0xFFC322, '>i2', 6)
        memory.assign(0xFFB7AA, '>i2', 6)
        memory.assign(0xFFB04A + 6 * 0x80 + 0x28, '>i2', 1200)
        memory.assign(0xFFB04A + 6 * 0x80 + 0x54, '>u2', 2)
        env.unwrapped.info['period'] = 2
        info = env._model_input_info(env.unwrapped.info)
        self.assertIn('model_input_players', info)
        env.game_state.BeginFrame(info, [0] * 6)
        env.opponent_action_state['last_gamestate_action'][5] = 1
        env.opponent_action_state['slapshot_frames'] = 12
        view = env._build_opponent_view_state()
        values = env.opponent_encoder.encode(view)
        self.assertEqual(len(values), 52)
        self.assertAlmostEqual(values[field_index('player', 'vx', self.config)], -1200 / 32768)
        self.assertAlmostEqual(values[field_index('player', 'facing_x', self.config)], -1.0)
        self.assertEqual(values[field_index('player', 'has_puck', self.config)], 1.0)
        self.assertEqual(values[field_index('player', 'c_pressed', self.config)], 1.0)
        self.assertEqual(values[field_index('player', 'c_frames_held', self.config)], 0.2)
        self.assertEqual(env.game_state.team2.players[0].input_state['vx'], 1200)


if __name__ == '__main__':
    unittest.main()

"""Replay branches preserve implementations, aliases and proposal side effects."""
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from nhl94_ai.agents.classic_v1 import ClassicAIV1Model
from nhl94_ai.agents.defense import DefenseController
from nhl94_ai.agents.history import COMPONENT_FIELDS, CONTROLLER_FIELDS
from nhl94_ai.agents.learned_offense import PossessionController
from tests.support.offense import shooting_state


class ClassicHistoryTests(unittest.TestCase):
    def test_declared_history_covers_every_constructed_owner(self):
        for flags in ({}, {'goalie_policy': 'selective'}, {'cross_crease': True}, {'deke': True},
                      {'possession_value': True}, {'possession_ablation': 'risk'},
                      {'receiver_selection': 'all-safe'}, {'opportunistic_cross_crease': True}):
            model = ClassicAIV1Model(SimpleNamespace(**flags))
            history = model.export_history()
            self.assertFalse(set(vars(model)) - set(CONTROLLER_FIELDS) - set(COMPONENT_FIELDS) - {'env'})
            for path, captured in history.components.items():
                if captured is None:
                    continue
                owner = model
                for name in path.split('.'):
                    owner = getattr(owner, name)
                children = {name[len(path)+1:] for name in COMPONENT_FIELDS if name.startswith(path+'.')}
                with self.subTest(flags=flags, owner=path):
                    self.assertFalse(set(vars(owner)) - set(captured.values) - children - {'offense'})

    def test_import_retains_target_implementation_and_environment(self):
        class TargetDefense(DefenseController):
            def step(self, *_args, **_kwargs):
                return 'target implementation'

        source, target = ClassicAIV1Model(), ClassicAIV1Model(env=object())
        target.defense = TargetDefense()
        source.defense.frames, source.defense.switch_at = 75, 92
        source.defense.pending_switch = {'to_slot': 2}
        owner, environment = target.defense, target.env
        target.import_history(source.export_history())
        self.assertIs(target.defense, owner)
        self.assertIs(target.env, environment)
        self.assertEqual(target.defense.step(None), 'target implementation')
        self.assertEqual((target.defense.frames, target.defense.switch_at), (75, 92))
        target.defense.pending_switch['to_slot'] = 3
        self.assertEqual(source.defense.pending_switch['to_slot'], 2)

    def test_multiple_imports_keep_diagnostic_aliases_and_bound_planners_independent(self):
        source = ClassicAIV1Model(SimpleNamespace(receiver_selection='legacy'))
        source.offense_diagnostics = source.offense.diagnostics = {'rows': []}
        source.receiver_selector.offense = source.offense
        history = source.export_history()
        branches = [ClassicAIV1Model().import_history(history) for _ in range(2)]
        for branch in branches:
            self.assertIs(branch.offense_diagnostics, branch.offense.diagnostics)
            self.assertIs(branch.receiver_selector.offense, branch.offense)
        branches[0].offense.diagnostics['rows'].append(1)
        self.assertEqual(branches[1].offense.diagnostics['rows'], [])
        self.assertEqual(source.offense.diagnostics['rows'], [])
        self.assertEqual(history.controller['offense_diagnostics']['rows'], [])

    def test_fork_continues_held_input_and_learned_command_history(self):
        state, model = shooting_state(), ClassicAIV1Model()
        model.predict_frame(state, 4)
        branch = model.fork()
        self.assertIsNot(branch.scheduler.action, model.scheduler.action)
        for _ in range(12):
            np.testing.assert_array_equal(model.predict_frame(state, 4), branch.predict_frame(state, 4))
        learned = PossessionController(mode='teacher')
        learned.teacher_command, learned.command_used = 17, True
        fork = learned.fork()
        self.assertIsInstance(fork, PossessionController)
        self.assertEqual((fork.teacher_command, fork.command_used, fork.control_scope), (17, True, 'offense'))
        fork.last_buttons[0] = 1
        self.assertEqual(learned.last_buttons[0], 0)

    def test_overridden_feint_keeps_retry_history_until_explicit_cancel(self):
        state, model = shooting_state(), ClassicAIV1Model()

        def proposal(_state, frame):
            model.offense.feint_target = (50, 120)
            model.offense.feint_until, model.offense.feint_at = frame + 24, frame + 72
            return 'feint', (50, 120), None

        with patch.object(model.offense, 'choose', side_effect=proposal):
            model.predict_frame(state, 4)
        self.assertEqual(model._last_decision, 'shoot')
        self.assertEqual(model.offense.feint_target, (50, 120))
        branch = model.fork()
        retry = branch.offense.feint_at
        branch.cancel_actions('stoppage')
        self.assertIsNone(branch.offense.feint_target)
        self.assertEqual(branch.offense.feint_at, retry)
        self.assertEqual(model.offense.feint_target, (50, 120))

    def test_overridden_chance_cleanup_preserves_previous_decision_retry_rule(self):
        for previous, expected in (('create-chance', 73), ('carry', 0)):
            state, model = shooting_state(), ClassicAIV1Model()
            model._last_decision = previous

            def proposal(_state, frame, controller=model):
                controller.offense.chance_owner, controller.offense.chance_until = 0, frame + 24
                return 'create-chance', (50, 120), None

            with self.subTest(previous=previous), patch.object(model.offense, 'choose', side_effect=proposal):
                model.predict_frame(state, 4)
                self.assertEqual(model._last_decision, 'shoot')
                self.assertIsNone(model.offense.chance_owner)
                self.assertEqual((model.offense.chance_until, model.offense.chance_at), (0, expected))

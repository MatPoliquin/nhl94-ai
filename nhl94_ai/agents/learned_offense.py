"""Optional possession decisions using Classic's native action lifecycles.

Commands use attacking coordinates. Receiver numbers refer to roster order with
the carrier removed, never distance order. No tactical score masks an action.
"""
import math
from types import SimpleNamespace

import numpy as np

from nhl94_ai.agents.classic_v1 import ClassicAIV1Model
from nhl94_ai.game.actors import controlled_slot, observable_body
from nhl94_ai.agents.lifecycle import ONE_TIMER_TIMEOUT_FRAMES, ONE_TIMER_CONTACT_GRACE
from nhl94_ai.game.motion import velocity
from nhl94_ai.game.passing import pad_direction, pass_speed, pass_contact, pass_release_frames
from nhl94_ai.env.intents import HOCKEY_INTENT_NOOP, HOCKEY_INTENT_NORMAL_SHOOT, HOCKEY_INTENT_PASS_START
from nhl94_ai.game.constants import GameConsts as Buttons


PADS = ((0, 0), (0, 1), (1, 1), (1, 0), (1, -1),
        (0, -1), (-1, -1), (-1, 0), (-1, 1))
ACTION_NAMES = ('coast', 'up', 'up-right', 'right', 'down-right', 'down',
                'down-left', 'left', 'up-left',
                *(f'pass-{i}' for i in range(1, 5)),
                *(f'one-timer-{i}' for i in range(1, 5)), 'shoot', 'slapshot')
N_ACTIONS = len(ACTION_NAMES)


def attack_sign(state):
    return 1 if state.team2.net.y > state.team1.net.y else -1


def receivers(state):
    base = state.team1.skater_scnum_base()
    return [(base + i, p) for i, p in enumerate(state.team1.players)
            if base + i != state.engine.puck_owner]


class PossessionController(ClassicAIV1Model):  # pylint: disable=abstract-method
    """Classic remains unchanged unless an external command is supplied.

Teacher mode captures the choice at _attack, before it becomes a button macro.
External commands are consumed once. Subsequent frames use the inherited pass,
shot, one-timer, goalie and defense lifecycles.
    """

    def __init__(self, *, mode='learned', goalie_policy='selective', interval=4, control_scope='offense'):
        super().__init__(SimpleNamespace(action_type='FILTERED', goalie_policy=goalie_policy))
        if control_scope not in ('offense', 'carry'):
            raise ValueError('Control scope must be offense or carry')
        self.control_scope = control_scope
        self.mode = mode
        self.interval = interval
        self._reset_commands()

    def reset_episode(self):
        super().reset_episode()
        self._reset_commands()

    def _reset_commands(self):
        self.command = None
        self.teacher_command = None
        self.teacher_allowed = False
        self.command_used = False
        self.last_buttons = np.zeros(12, dtype=np.int8)

    def available(self, state):
        team, owner = state.team1, state.engine.puck_owner
        player = team.get_player_by_scnum(owner)
        return bool(
            team.owns_scnum(owner) and owner != team.goalie_scnum()
            and owner == controlled_slot(team) and player is not None
            and observable_body(player) and not player.is_falling
            and player.y * attack_sign(state) >= 100
            and not state.engine.clock_stopped
            and not self.scheduler.was_defending
            and self.one_timer.pending is None and self.offense.pass_action.pending is None
            and not self.shot.active(self.scheduler.decisions)
            and (self.goalie is None or self.goalie.phase == 'skater'))

    def action_mask(self, state):
        mask = np.zeros(N_ACTIONS, dtype=np.float32)
        mask[0] = 1
        if not self.available(state):
            return mask
        mask[:9] = 1
        if self.control_scope == 'carry':
            return mask
        if not self.buttons.b_down:
            for i, (_, player) in enumerate(receivers(state)[:4]):
                if observable_body(player) and not player.is_falling:
                    mask[9 + i] = mask[13 + i] = 1
        if not self.buttons.c_down:
            mask[17:] = 1
        return mask

    def begin_step(self, state, command=0):
        self.teacher_command = None
        self.teacher_allowed = False
        self.command_used = False
        self.command = None
        active = self.available(state)
        if self.mode != 'classic' and active:
            # All new possession choices are on the policy's decision clock.
            self.scheduler.interrupt()
            self.teacher_allowed = self.mode == 'teacher'
            if self.mode == 'learned':
                if not 0 <= int(command) < N_ACTIONS or not self.action_mask(state)[int(command)]:
                    raise ValueError(f'Unavailable offense command: {command}')
                self.command = int(command)
        return active

    def frame(self, state):
        action = self.predict_frame(state, self.interval)[0]
        self.last_buttons = action.copy()
        return action

    def _attack(self, state, action, player, actual, attack):
        if self.mode == 'classic' or player.y * attack < 100:
            return super()._attack(state, action, player, actual, attack)
        if self.control_scope == 'carry':
            return self._carry_attack(state, action, player, actual, attack)
        if self.mode == 'teacher':
            if not self.teacher_allowed:
                return action, HOCKEY_INTENT_NOOP
            self.teacher_allowed = False
            result = super()._attack(state, action, player, actual, attack)
            if self.teacher_command is None:
                self.teacher_command = self._label(state, result[0])
            return result
        command, self.command = self.command, None
        if command is None:
            # An ownership change between decision ticks waits for the next tick.
            return action, HOCKEY_INTENT_NOOP
        self.command_used = True
        self.input_owner = 'offense'
        self._last_decision = 'learned-' + ACTION_NAMES[command]
        self._record_selection(self._last_decision)
        self.offense_diagnostics = {'command': ACTION_NAMES[command]}
        self.setup.cancel()
        if command < 9:
            dx, dy = PADS[command]
            dx, dy = dx * attack, dy * attack
            if dx:
                action[Buttons.INPUT_RIGHT if dx > 0 else Buttons.INPUT_LEFT] = 1
            if dy:
                action[Buttons.INPUT_UP if dy > 0 else Buttons.INPUT_DOWN] = 1
            self._last_target = player.x + dx * 30, player.y + dy * 30
            return action, HOCKEY_INTENT_NOOP
        if command < 17:
            index = (command - 9) % 4
            slot, receiver = receivers(state)[index]
            point = receiver.x, receiver.y
            flight = math.dist((player.x, player.y), point) / max(pass_speed(player) or 4, 1)
            contact = pass_contact(state.puck, player, receiver)
            if contact is not None:
                _, point, flight, _ = contact
            if command >= 13:
                duration = (math.ceil(flight + pass_release_frames(receiver)) + ONE_TIMER_CONTACT_GRACE
                            if contact is not None else ONE_TIMER_TIMEOUT_FRAMES)
                result = self._start_one_timer_pass(state, action, player, index, slot, point, duration)
                purpose = 'one-timer'
            else:
                option = SimpleNamespace(slot=slot, point=point, flight_frames=flight,
                                         direction=pad_direction(player, receiver))
                result = self._start_pass(state, action, player, option, index, 'learned-pass')
                purpose = 'learned-pass'
            self._last_pass_request = dict(frame=self.scheduler.frames, passer=actual,
                                           receiver=slot, purpose=purpose)
            return result
        self.input_owner = 'shot'
        self._last_decision = 'shoot'
        self._record_selection(self._last_decision)
        self.shot.side = -1 if state.team2.goalie.x > 0 else 1
        hold = self.interval if command == 17 else 18
        result = self._start_shot(state, action, hold)
        self.shot.extended_hold = command == 18
        self._last_target = self.shot.side * 13, state.team2.net.y
        return result

    def _carry_attack(self, state, action, player, actual, attack):
        """Keep Classic's decisions/executors; replace only an uncommitted carry.

        The teacher shares the learner's decision clock. A finishing decision
        can consume a tick without using the external movement command.
        """
        if self.command is None and not self.teacher_allowed:
            return action, HOCKEY_INTENT_NOOP
        command, self.command = self.command, None
        self.teacher_allowed = False
        result, intent = ClassicAIV1Model._attack(self, state, action, player, actual, attack)
        committed = (self.one_timer.pending is not None or self.offense.pass_action.pending is not None
                     or self.shot.active(self.scheduler.decisions) or self.input_owner != 'offense'
                     or result[Buttons.INPUT_B] or result[Buttons.INPUT_C])
        if committed:
            return result, intent
        if self.mode == 'teacher':
            self.teacher_command = self._label(state, result)
            self.command_used = True
            return result, intent
        self.command_used = True
        result[:] = 0
        dx, dy = PADS[command]
        dx, dy = dx * attack, dy * attack
        if dx:
            result[Buttons.INPUT_RIGHT if dx > 0 else Buttons.INPUT_LEFT] = 1
        if dy:
            result[Buttons.INPUT_UP if dy > 0 else Buttons.INPUT_DOWN] = 1
        self._last_decision = 'learned-carry-' + ACTION_NAMES[command]
        self._record_selection(self._last_decision)
        self._last_target = player.x + dx * 30, player.y + dy * 30
        self.offense_diagnostics['carry_command'] = ACTION_NAMES[command]
        return result, HOCKEY_INTENT_NOOP

    def _label(self, state, buttons):
        slots = [slot for slot, _ in receivers(state)]
        if buttons[Buttons.INPUT_B] and self._last_pass_request:
            slot = self._last_pass_request['receiver']
            if slot in slots:
                return (13 if self.one_timer.pending is not None else 9) + slots.index(slot)
        if buttons[Buttons.INPUT_C]:
            return 17
        sign = attack_sign(state)
        pad = (sign * int(buttons[Buttons.INPUT_RIGHT] - buttons[Buttons.INPUT_LEFT]),
               sign * int(buttons[Buttons.INPUT_UP] - buttons[Buttons.INPUT_DOWN]))
        return PADS.index(pad)


RATINGS = (('speed', 30), ('agility', 30), ('weight', 120), ('passing', 30),
           ('shot_accuracy', 30), ('shot_power', 30), ('handedness', 1), ('energy', 4096))
ENTITY_FIELDS = ('x', 'y', 'vx', 'vy', 'rel_puck_x', 'rel_puck_y', 'facing_x', 'facing_y',
                 'controlled', 'has_puck', 'on_ice', 'falling', 'pad_stack', 'dive',
                 'animation_frame', *(name for name, _ in RATINGS), 'ratings_known')
CONTEXT_FIELDS = ('puck_x', 'puck_y', 'puck_vx', 'puck_vy', 'own_possession',
                  'opponent_possession', 'loose', 'clock', 'score_difference', 'net_y',
                  'clock_stopped', 'can_choose', 'pass_active', 'one_timer_active',
                  'shot_active', 'shot_hold_remaining', 'decision_remaining',
                  'up', 'down', 'left', 'right', 'a', 'b', 'c')
OBSERVATION_FIELDS = tuple(
    f'{team}_{slot}_{field}' for team in ('own', 'opponent')
    for slot in ('1', '2', '3', '4', '5', 'goalie') for field in ENTITY_FIELDS
) + CONTEXT_FIELDS + tuple('legal_' + name for name in ACTION_NAMES)


def encode_offense(state, controller, *, clock, score_difference):
    """Physical state, scaled once and rotated 180 degrees when attacking down."""
    sign, puck = attack_sign(state), state.puck
    owner = state.engine.puck_owner
    values = []
    for team in (state.team1, state.team2):
        for i, player in enumerate([*team.players, team.goalie]):
            slot = team.skater_scnum_base() + i
            vx, vy = velocity(player)
            facing = player.facing if player.facing is not None else player.orientation
            angle = facing * math.pi / 4
            values.extend((sign * player.x / 128, sign * player.y / 288,
                           sign * vx / 16, sign * vy / 16,
                           sign * (player.x - puck.x) / 256, sign * (player.y - puck.y) / 576,
                           sign * math.sin(angle), sign * math.cos(angle),
                           float(controlled_slot(team) == slot), float(owner == slot),
                           float(observable_body(player)), player.is_falling, player.is_pad_stack,
                           player.is_dive, player.anim_frame / 32))
            values.extend((getattr(player, name, None) or 0) / scale for name, scale in RATINGS)
            values.append(float(all(getattr(player, name, None) is not None for name, _ in RATINGS)))
    vx, vy = velocity(puck)
    buttons = controller.last_buttons
    up, down = (Buttons.INPUT_UP, Buttons.INPUT_DOWN) if sign == 1 else (Buttons.INPUT_DOWN, Buttons.INPUT_UP)
    left, right = (Buttons.INPUT_LEFT, Buttons.INPUT_RIGHT) if sign == 1 else (Buttons.INPUT_RIGHT, Buttons.INPUT_LEFT)
    values.extend((sign * puck.x / 128, sign * puck.y / 288, sign * vx / 16, sign * vy / 16,
                   float(state.team1.owns_scnum(owner)), float(state.team2.owns_scnum(owner)),
                   float(owner < 0), clock / 1200, score_difference / 10,
                   sign * state.team2.net.y / 288, float(bool(state.engine.clock_stopped)),
                   float(controller.available(state)), float(controller.offense.pass_action.pending is not None),
                   float(controller.one_timer.pending is not None), float(controller.shot.active(controller.scheduler.decisions)),
                   max(0, controller.shot.hold_until_frame - controller.scheduler.frames) / 60,
                   controller.scheduler.remaining / max(controller.interval, 1),
                   *(float(buttons[i]) for i in (up, down, left, right, Buttons.INPUT_A, Buttons.INPUT_B, Buttons.INPUT_C))))
    values.extend(controller.action_mask(state))
    result = np.asarray(values, dtype=np.float32)
    if result.shape != (len(OBSERVATION_FIELDS),) or not np.isfinite(result).all():
        raise ValueError('Invalid offense observation')
    return np.clip(result, -1, 1)


SCHEMA = dict(version=1, history=16, interval=4, actions=list(ACTION_NAMES),
              fields=list(OBSERVATION_FIELDS), reward="goal-difference-v1",
              execution="classic-native-v1", coordinates="attack-up-180")

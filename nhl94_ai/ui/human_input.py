"""Shared keyboard and gamepad controls for playable windows."""

import pygame

from nhl94_ai.game.constants import GameConsts as Buttons


KEY_ACTION_MAP = {
    pygame.K_UP: Buttons.INPUT_UP, pygame.K_DOWN: Buttons.INPUT_DOWN,
    pygame.K_LEFT: Buttons.INPUT_LEFT, pygame.K_RIGHT: Buttons.INPUT_RIGHT,
    pygame.K_z: Buttons.INPUT_A, pygame.K_x: Buttons.INPUT_B, pygame.K_c: Buttons.INPUT_C,
    pygame.K_a: Buttons.INPUT_X, pygame.K_s: Buttons.INPUT_Y, pygame.K_d: Buttons.INPUT_Z,
    pygame.K_RETURN: Buttons.INPUT_START, pygame.K_TAB: Buttons.INPUT_MODE,
}


def read_human_input(gamepad, focused, key_action_map=None):
    """Read current buttons; releases and focus loss never retain old input."""
    pygame.event.pump()
    if not focused:
        return [0] * Buttons.INPUT_MAX
    keys = pygame.key.get_pressed()
    actions = gamepad.read()
    if key_action_map is None:
        key_action_map = KEY_ACTION_MAP
    for key, action in key_action_map.items():
        if keys[key]:
            actions[action] = 1
    for first, second in ((Buttons.INPUT_UP, Buttons.INPUT_DOWN),
                          (Buttons.INPUT_LEFT, Buttons.INPUT_RIGHT)):
        if actions[first] and actions[second]:
            actions[first] = actions[second] = 0
    return actions

"""B arbitration uses executor intent and confirmed native ownership."""


class BArbitrator:
    def __init__(self, controllers, *, switch_spacing=8):
        self.next_switch = 0
        self.next_controller = [0] * controllers
        self.switch_spacing = switch_spacing

    def allow(self, *, intent, frame, controller, previous, owns_puck,
              known, protected, likely_keeps_selection=False):
        if intent == 'faceoff':
            return True, 'faceoff-owner'
        if not known:
            return False, 'unknown-selection'
        if intent in ('pass', 'goalie-outlet'):
            return owns_puck, 'owned-release' if owns_puck else 'not-input-owner'
        if protected:
            return False, 'pass-protected'
        if owns_puck:
            return False, 'defensive-input-would-pass'
        if previous:
            return False, 'release-b'
        if frame < self.next_controller[controller]:
            return False, 'controller-cooldown'
        if intent == 'poke' and likely_keeps_selection:
            self.next_controller[controller] = frame + 16
            return True, 'poke-keeps-selection'
        if intent == 'switch':
            if frame < self.next_switch:
                return False, 'shared-switch-cooldown'
            self.next_switch = frame + self.switch_spacing
            self.next_controller[controller] = frame + 8
            return True, 'serialized-switch'
        return False, 'unconfirmed-b-meaning'

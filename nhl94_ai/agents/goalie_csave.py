"""Deeper positioning with native body-contact checks before committing C."""
import math

from nhl94_ai.agents.goalie import goalie_steer
from nhl94_ai.agents.goalie_physics import KeeperMotion, PuckFlight, contact_margin
from nhl94_ai.agents.goalie_positioning import DeepGoalie
from nhl94_ai.game.constants import GameConsts as Buttons


def contact_frame(state, action, horizon, *, save=False):
    """Project body contact under continued movement or C's reduced momentum.

    This short free-flight model excludes animation-specific reach, intervening
    skaters and rebound outcomes. It must not be used to compare A dives, which
    have different motion and collision cooldowns.
    """
    if not 0 < horizon <= 12:
        return None
    keeper = KeeperMotion.observe(state.team1.goalie)
    puck = PuckFlight.observe(state.puck)
    if keeper is None or puck is None:
        return None
    sign = 1 if state.team1.net.y > 0 else -1
    pad = (int(action[Buttons.INPUT_RIGHT]) - int(action[Buttons.INPUT_LEFT]),
           int(action[Buttons.INPUT_UP]) - int(action[Buttons.INPUT_DOWN]))
    if not any(pad):
        return None
    for frame in range(1, math.ceil(horizon) + 1):
        keeper = keeper.step(pad, save=save)
        puck = puck.step()
        if abs(puck.x >> 16) >= 128 or (puck.y >> 16) * sign >= 261:
            return None
        if contact_margin(keeper, puck, one_timer=state.engine.native_one_timer_shot) >= 0:
            return frame
    return None


class CSaveGoalie(DeepGoalie):
    """Keep moving when projected body contact would be lost by pressing C."""

    def _position(self, state, plan):
        if plan.crossing_frames is not None and self.frames >= self.save_at:
            gap = abs(plan.crossing_x - state.team1.goalie.x)
            # Preserve the A-dive branch (gap > 18); this model describes C.
            if 8 < gap <= 18 and plan.height <= 15:
                action = goalie_steer(state.team1.goalie, plan.target)
                moving = contact_frame(state, action, plan.crossing_frames)
                saving = contact_frame(state, action, plan.crossing_frames, save=True)
                self.diagnostics.update(native_moving_contact=moving, native_saving_contact=saving)
                if moving is not None and saving is None:
                    self.diagnostics['save_decision'] = 'native-body-contact-before-C'
                    return action
        return super()._position(state, plan)

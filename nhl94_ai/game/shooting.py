"""Normal-shot release timing and speed, independent of tactical shot scoring."""
from nhl94_ai.game.motion import VELOCITY_SCALE


def normal_shot_release_frames(player, hold_frames):
    """Bound updateanim/ShotMode through contact, including the C-release jump."""
    if getattr(player, 'platform', 'genesis') != 'genesis':
        from nhl94_ai.game.platforms import get_backend
        return get_backend(player.platform).shot_release_frames(player, hold_frames)
    durations = player.shot_durations or (4,) * 8
    index, timer = 0, -1
    for elapsed in range(1, hold_frames + 2 * sum(duration + 1 for duration in durations) + 3):
        if index >= 7:
            return elapsed + 1
        if timer < 0:
            timer = durations[index]
        else:
            timer -= 1
            if timer < 0:
                index += 1
                if index < 7:
                    timer = durations[index]
        if index >= 7:
            return elapsed + 1
        if index < 4 and (elapsed == hold_frames or player.shot_power is not None
                          and player.shot_power < 20 and index > 2):
            index = 7 - index
    raise RuntimeError('Shot animation did not reach native contact.')

def shot_speed(player, hold_frames, *, one_timer=False):
    """doshot's rating/energy scaling with an estimated charged passspeed."""
    if getattr(player, 'platform', 'genesis') != 'genesis':
        from nhl94_ai.game.platforms import get_backend
        return get_backend(player.platform).shot_speed(player, hold_frames, one_timer=one_timer)
    power = 15 if player.shot_power is None else player.shot_power
    energy = 4096 if player.energy is None else max(0, min(4096, player.energy))
    charge = 31 if one_timer else 15 + hold_frames
    rating = (power // 2) * energy // 4096
    speed = ((20 + rating) * charge * 0x5249) >> 16
    return max(1.0, speed * 68 * VELOCITY_SCALE)

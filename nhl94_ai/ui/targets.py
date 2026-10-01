"""Shared target-policy annotations for playback and the live training viewer."""
import math

import pygame


TARGET_GREEN = (0, 255, 0)
CYAN = (0, 195, 225)
GREEN = (20, 175, 75)
PURPLE = (175, 80, 220)
OFFENSE_AMBER = (255, 165, 0)


def draw_target_box(surface, position, size=16, color=TARGET_GREEN):
    rect = pygame.Rect(0, 0, size, size)
    rect.center = round(position[0]), round(position[1])
    pygame.draw.rect(surface, color, rect, max(2, round(size / 7)))


def draw_game_target(surface, rect, frame_size, info):
    """Draw the policy target on the supported vertical-view NHL94 framebuffer."""
    diagnostics = info.get('target_control') or info.get('classic_defense') or info.get('classic_offense')
    if not diagnostics or diagnostics['target'] is None:
        return
    x, y = diagnostics['target']
    # ROM find3d at ice height, minus the Genesis sprite coordinate bias (128).
    x = 128 + x - info['target_camera_x']
    y = 112 - y + info['target_camera_y']
    scale_x, scale_y = rect.width / frame_size[0], rect.height / frame_size[1]
    position = rect.x + x * scale_x, rect.y + y * scale_y
    previous_clip = surface.get_clip()
    surface.set_clip(rect.clip(previous_clip))
    if diagnostics.get('phase') == 'offense':
        draw_target_box(surface, position, max(10, round(14 * min(scale_x, scale_y))), OFFENSE_AMBER)
    else:
        draw_target_box(surface, position, max(10, round(14 * min(scale_x, scale_y))))
    surface.set_clip(previous_clip)


def _target_team(state, diagnostics):
    for slot in (diagnostics['actual_slot'], diagnostics['desired_slot']):
        if slot is not None and slot >= 0:
            for team in (state.team1, state.team2):
                if team.owns_scnum(slot):
                    return team
    return state.team1


def draw_target_overlay(surface, transform, state, diagnostics, font, origin=(12, 12)):
    if not diagnostics or diagnostics['target'] is None:
        return
    if diagnostics.get('phase') == 'offense':
        draw_offense_overlay(surface, transform, state, diagnostics, font, origin)
        return
    target, destination = diagnostics['target'], diagnostics['destination']
    waypoint = diagnostics['waypoint']
    classic = 'reason' in diagnostics
    lines = (
        f"Target ({target[0]:.0f}, {target[1]:.0f})",
        diagnostics['mode'] if diagnostics['actual_slot'] != -1 else 'No player selected',
        'Green square: defensive target' if classic else 'Green square: policy target',
        'Green: controlled / cyan: desired',
    )
    if classic:
        costs = diagnostics.get('arrival_frames', {})
        eta = costs.get(diagnostics['desired_slot'])
        lines += (diagnostics['decision'], diagnostics['reason'],
                  f"Arrival: {eta:.1f} frames" if eta is not None else 'No eligible defender',
                  f"Skater: {diagnostics['actual_slot']} -> {diagnostics['desired_slot']}; "
                  f"ideal: {diagnostics.get('ideal_slot')}; B: {diagnostics.get('likely_switch_slot')}",
                  f"Switch: {diagnostics.get('switch_status', 'unknown')}",
                  f"Receiver: {diagnostics.get('threat_slot')}; puck ETA: {diagnostics.get('puck_arrival')}",
                  f"Missing feedback: {', '.join(diagnostics.get('missing_feedback', [])) or 'none'}")
        result = diagnostics.get('last_switch_result')
        if result:
            lines += (f"Last switch: {result['outcome']} ({result['from_slot']} -> "
                      f"{result['actual_slot']}, {result['elapsed_frames']} frames)",)
        if diagnostics.get('lanes'):
            covered = sum(bool(lane['blockers']) for lane in diagnostics['lanes'])
            lines += (f"Covered lanes: {covered}/{len(diagnostics['lanes'])}; "
                      f"keep skaters: {diagnostics.get('reserved_slots', ())}",)
        check = diagnostics.get('check')
        if check:
            lines += (f"Body check: {check['status']}; impact/threshold: "
                      f"{check.get('impact_estimate', '-')}/{check.get('weight_threshold', '-')}",)
    for index, text in enumerate(lines):
        label = font.render(text, True, (15, 20, 30), (230, 242, 249))
        surface.blit(label, (origin[0], origin[1] + index * (font.get_linesize() + 1)))
    team = _target_team(state, diagnostics)
    actual = team.get_player_by_scnum(diagnostics['actual_slot'])
    desired = (team.get_player_by_scnum(diagnostics['desired_slot'])
               if diagnostics['desired_slot'] is not None else None)
    if actual is not None:
        position = transform(actual.x, actual.y)
        pygame.draw.circle(surface, GREEN, position, 12, 3)
        pygame.draw.lines(surface, PURPLE, False, [position, transform(*waypoint), transform(*destination)], 2)
    if desired is not None:
        pygame.draw.circle(surface, CYAN, transform(desired.x, desired.y), 16, 2)
    if math.dist(target, destination) > 0.5:
        pygame.draw.circle(surface, CYAN, transform(*destination), 7, 2)
    if math.dist(waypoint, destination) > 0.5:
        pygame.draw.circle(surface, PURPLE, transform(*waypoint), 5, 2)
    if classic:
        for lane in diagnostics.get('lanes', ()):
            pygame.draw.line(surface, GREEN if lane['blockers'] else PURPLE,
                             transform(*lane['origin']), transform(*lane['goal']), 1)
        if diagnostics.get('lanes') and diagnostics.get('shot_goal'):
            pygame.draw.line(surface, CYAN, transform(*diagnostics['lanes'][-1]['origin']),
                             transform(*diagnostics['shot_goal']), 2)
        path = diagnostics.get('puck_path', [])
        if len(path) > 1:
            pygame.draw.lines(surface, CYAN, False, [transform(*point) for _, point in path], 1)
        receiver = diagnostics.get('receiver')
        if receiver is not None:
            pygame.draw.line(surface, PURPLE, transform(*receiver), transform(0, team.net.y), 1)
    draw_target_box(surface, transform(*target))


def draw_offense_overlay(surface, transform, state, diagnostics, font, origin):
    lines = [
        diagnostics['mode'], diagnostics['reason'], 'Amber square: offensive destination',
        f"Receiver: {diagnostics['desired_slot']}; scenarios: {diagnostics.get('scenario_samples', '-')}",
    ]
    candidate = diagnostics.get('receiver')
    if candidate:
        lines.append(f"Flight: {candidate['flight_frames']:.1f}; margin: {candidate['margin']:.1f}; "
                     f"robustness: {candidate['robustness']:.0%}")
    last = diagnostics.get('last_pass')
    if last:
        lines.append(f"Last pass: {last['outcome']}; intended/actual: "
                     f"{last['receiver']}/{last['actual_receiver']}")
    for index, text in enumerate(lines):
        surface.blit(font.render(text, True, (15, 20, 30), (230, 242, 249)),
                     (origin[0], origin[1] + index * (font.get_linesize() + 1)))
    actual = _target_team(state, diagnostics).get_player_by_scnum(diagnostics['actual_slot'])
    if actual is not None:
        pygame.draw.circle(surface, GREEN, transform(actual.x, actual.y), 12, 3)
        for option in diagnostics.get('candidates', []):
            if option.get('point'):
                pygame.draw.line(surface, GREEN if option['status'] == 'safe' else PURPLE,
                                 transform(state.puck.x, state.puck.y), transform(*option['point']), 1)
        pygame.draw.line(surface, OFFENSE_AMBER, transform(actual.x, actual.y),
                         transform(*diagnostics['target']), 2)
    draw_target_box(surface, transform(*diagnostics['target']), color=OFFENSE_AMBER)


def draw_target_rink(surface, rect, state, diagnostics, font):
    previous_clip = surface.get_clip()
    surface.set_clip(rect)
    surface.fill((210, 233, 246), rect)
    scale = min((rect.width - 12) / 260, (rect.height - 12) / 580)

    def transform(x, y):
        return round(rect.centerx + x * scale), round(rect.centery - y * scale)

    for y, color in ((-88, (50, 100, 200)), (0, (200, 65, 65)), (88, (50, 100, 200))):
        pygame.draw.line(surface, color, transform(-130, y), transform(130, y), 1)
    for y in (-264, 264):
        pygame.draw.line(surface, (180, 50, 50), transform(-19, y), transform(19, y), 3)
    if state is not None:
        for team, color in ((state.team1, (200, 55, 55)), (state.team2, (55, 85, 205))):
            for player in (*team.players, team.goalie):
                pygame.draw.circle(surface, color, transform(player.x, player.y), 4)
        pygame.draw.circle(surface, (20, 20, 20), transform(state.puck.x, state.puck.y), 3)
        draw_target_overlay(surface, transform, state, diagnostics, font, (rect.x + 6, rect.y + 6))
    surface.set_clip(previous_clip)

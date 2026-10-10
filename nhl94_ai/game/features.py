from __future__ import annotations
import math
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from nhl94_ai.game.state import Player, Team, Net
from nhl94_ai.game.constants import GameConsts


class TacticalFeatures:
    """Hockey geometry and derived tactical features for decoded state."""

    @staticmethod
    def _team_attacking_top(team: Team) -> bool:
        return team.controller == 1

    def _normalized_attack_y(self, player: Player, team: Team) -> int:
        return player.y if self._team_attacking_top(team) else -player.y

    def _is_receiver_in_attack_band(self, player: Player, team: Team) -> bool:
        normalized_y = self._normalized_attack_y(player, team)
        return GameConsts.ATACKZONE_POS_Y <= normalized_y <= GameConsts.P2_NET_Y

    @staticmethod
    def _is_receiver_in_central_lane(player: Player) -> bool:
        return GameConsts.SLOT_BAND_MIN_X <= player.x <= GameConsts.SLOT_BAND_MAX_X

    def _get_passer_for_team(self, team: Team) -> Player | None:
        puck_owner = self.engine.puck_owner
        if team.owns_scnum(puck_owner):
            return team.get_player_by_scnum(puck_owner)

        return team.get_possession_player()

    def _get_puck_controlled_skater_for_team(self, team: Team) -> Player | None:
        puck_controller = self._get_passer_for_team(team)
        if any(puck_controller is player for player in team.players):
            return puck_controller

        return None

    def _is_goalie_player(self, player: Player) -> bool:
        return player is self.team1.goalie or player is self.team2.goalie

    def _passing_intercept_radius(self, player: Player) -> int:
        if not self._is_goalie_player(player):
            return 14

        if player.is_pad_stack:
            return 18

        goalie_radius = self.engine.goalie_chk_body if self.engine.goalie_chk_body > 0 else 12
        return max(12, min(goalie_radius, 18))

    def _goalie_shot_collision_radius(self, goalie: Player) -> int:
        return self._passing_intercept_radius(goalie)

    @staticmethod
    def _receiver_catch_point(player: Player) -> tuple[int, int]:
        catch_offset = 10
        return (
            player.x + round(player.ori_x * catch_offset),
            player.y + round(player.ori_y * catch_offset),
        )

    def _attacking_net_for_team(self, team: Team) -> Net:
        return self.team2.net if self._team_attacking_top(team) else self.team1.net

    @staticmethod
    def _shot_target_points_for_net(net: Net) -> list[tuple[int, int]]:
        center_x = (net.left + net.right) // 2
        post_inset = 6
        return [
            (center_x, net.y),
            (net.left + post_inset, net.y),
            (net.right - post_inset, net.y),
        ]

    @staticmethod
    def _shotdir_metric(
        puck_x: float,
        puck_y: float,
        goalie_x: float,
        goalie_y: float,
        post_x: float,
        post_y: float,
        goalie_distance: float,
    ) -> float:
        return ((goalie_x - puck_x) * (post_y - puck_y) - (goalie_y - puck_y) * (post_x - puck_x)) / goalie_distance

    def _shot_target_points_for_player(self, shooter: Player, team: Team, opponents: Team) -> list[tuple[int, int]]:
        attacking_net = self._attacking_net_for_team(team)
        fallback_targets = self._shot_target_points_for_net(attacking_net)
        goalie = opponents.goalie
        goalie_x = goalie.x + (goalie.vx / 2)
        goalie_y = goalie.y + (goalie.vy / 2)
        goalie_distance = math.hypot(goalie_x - shooter.x, goalie_y - shooter.y)
        if goalie_distance == 0:
            return fallback_targets

        left_metric = self._shotdir_metric(
            shooter.x,
            shooter.y,
            goalie_x,
            goalie_y,
            attacking_net.left,
            attacking_net.y,
            goalie_distance,
        )
        right_metric = self._shotdir_metric(
            shooter.x,
            shooter.y,
            goalie_x,
            goalie_y,
            attacking_net.right,
            attacking_net.y,
            goalie_distance,
        )
        crease_metric = left_metric + right_metric

        if abs(crease_metric) > 0x2C:
            return fallback_targets

        center_x = (attacking_net.left + attacking_net.right) // 2
        post_inset = 6
        side_target = (
            attacking_net.right - post_inset
            if crease_metric >= 0
            else attacking_net.left + post_inset
        )
        opposite_target = (
            attacking_net.left + post_inset
            if crease_metric >= 0
            else attacking_net.right - post_inset
        )
        return [
            (side_target, attacking_net.y),
            (center_x, attacking_net.y),
            (opposite_target, attacking_net.y),
        ]

    def _get_preferred_shot_lane_target(self, shooter: Player, team: Team, opponents: Team) -> tuple[int, int] | None:
        target_points = self._shot_target_points_for_player(shooter, team, opponents)
        return target_points[0] if target_points else None

    def _get_clear_shot_lane_target(self, shooter: Player, team: Team, opponents: Team) -> tuple[int, int] | None:
        if not self._is_in_front_of_attacking_goal_line(shooter, team):
            return None

        shot_start = (shooter.x, shooter.y)

        for target_point in self._shot_target_points_for_player(shooter, team, opponents):
            if self._is_passing_lane_clear(shot_start, target_point, opponents.players):
                return target_point

        return None

    def _has_clear_shot_lane(self, shooter: Player, team: Team, opponents: Team) -> bool:
        return self._get_clear_shot_lane_target(shooter, team, opponents) is not None

    def _is_goalie_blocking_shot_path(self, shot_start: tuple[int, int], target_point: tuple[int, int], goalie: Player) -> bool:
        radius = self._goalie_shot_collision_radius(goalie)
        return self._line_intersects_circle(shot_start, target_point, (goalie.x, goalie.y), radius)

    def _is_in_front_of_attacking_goal_line(self, shooter: Player, team: Team) -> bool:
        return self._normalized_attack_y(shooter, team) < GameConsts.P2_NET_Y

    def _get_open_net_shot_target(self, shooter: Player, team: Team, opponents: Team) -> tuple[int, int] | None:
        if not self._is_in_front_of_attacking_goal_line(shooter, team):
            return None

        shot_start = (shooter.x, shooter.y)

        for target_point in self._shot_target_points_for_player(shooter, team, opponents):
            if not self._is_passing_lane_clear(shot_start, target_point, opponents.players):
                continue
            if not self._is_goalie_blocking_shot_path(shot_start, target_point, opponents.goalie):
                return target_point

        return None

    def _has_open_net_shot(self, shooter: Player, team: Team, opponents: Team) -> bool:
        return self._get_open_net_shot_target(shooter, team, opponents) is not None

    def _has_viable_shooting_opening(self, max_distance: float = 100) -> bool:
        team = self.team1
        controlled = team.defense_control if team.defense_control is not None else team.controlled_scnum()
        shooter = team.get_player_by_scnum(controlled)
        if shooter is None or shooter is team.goalie or self.engine.puck_owner != controlled:
            return False
        net = self._attacking_net_for_team(team)
        return (
            shooter.open_net_shot
            and self._is_in_front_of_attacking_goal_line(shooter, team)
            and math.hypot(shooter.x - (net.left + net.right) / 2, shooter.y - net.y) <= max_distance
        )

    def _get_clear_one_timer_shot_target(self, shooter: Player, team: Team, opponents: Team) -> tuple[int, int] | None:
        shot_start = (shooter.x, shooter.y)
        attacking_net = self._attacking_net_for_team(team)

        for target_point in self._shot_target_points_for_net(attacking_net):
            if self._is_passing_lane_clear(shot_start, target_point, opponents.players):
                return target_point

        return None

    def _has_clear_one_timer_shot_lane(self, shooter: Player, team: Team, opponents: Team) -> bool:
        return self._get_clear_one_timer_shot_target(shooter, team, opponents) is not None

    def _pass_direction_matches_target(self, passer: Player, target_point: tuple[int, int]) -> bool:
        if not 0 <= self.engine.pass_dir <= 7:
            return True

        delta_x = target_point[0] - passer.x
        delta_y = target_point[1] - passer.y
        distance = math.hypot(delta_x, delta_y)
        if distance == 0:
            return False

        target_x = delta_x / distance
        target_y = delta_y / distance
        pass_angle = self.engine.pass_dir * (2 * math.pi / 8)
        pass_dir_x = math.cos(pass_angle)
        pass_dir_y = math.sin(pass_angle)
        dot_product = (target_x * pass_dir_x) + (target_y * pass_dir_y)
        return dot_product >= math.cos(math.pi / 4)

    def _is_passing_lane_clear(self, start_pos, end_pos, opponents):
        """
        Check if a straight-line path between two points is obstructed.

        Based on NHL94 ROM data:
        - Stick interception: 14 units from stick hotspot (dist^2 <= 196)
        - Body collision: 8 units from player center (dist^2 <= 64)
        - Broad-phase Y-filter: 22 units (ROM optimization)

        Using radius=18 (14 + 4 buffer for moving stick hotspot)
        """
        for opponent in opponents:
            radius = self._passing_intercept_radius(opponent)

            # Broad-phase: skip if Y difference is too large (ROM optimization)
            min_y_dist = min(abs(opponent.y - start_pos[1]), abs(opponent.y - end_pos[1]))
            if min_y_dist > radius + 8:
                continue

            if self._line_intersects_circle(start_pos, end_pos, (opponent.x, opponent.y), radius=radius):
                return False
        return True

    def _line_intersects_circle(self, start, end, circle_center, radius):
        """Check if a line segment from 'start' to 'end' intersects a circle.

        Args:
            start: Tuple (x, y) of line segment start point
            end: Tuple (x, y) of line segment end point
            circle_center: Tuple (x, y) of circle center
            radius: Radius of the circle

        Returns:
            bool: True if the line segment intersects the circle
        """
        # Vector from start to end
        line_vec = (end[0] - start[0], end[1] - start[1])
        # Vector from start to circle center
        circle_vec = (circle_center[0] - start[0], circle_center[1] - start[1])

        # Length of line segment squared
        line_len_sq = line_vec[0]**2 + line_vec[1]**2

        # Projection of circle_vec onto line_vec (dot product)
        projection = circle_vec[0] * line_vec[0] + circle_vec[1] * line_vec[1]

        # Normalized projection (0 to 1 means closest point is on the segment)
        t = max(0, min(1, projection / line_len_sq)) if line_len_sq != 0 else 0

        # Closest point on the line segment to the circle center
        closest_point = (
            start[0] + t * line_vec[0],
            start[1] + t * line_vec[1]
        )

        # Distance from closest point to circle center
        distance_sq = (circle_center[0] - closest_point[0])**2 + \
                    (circle_center[1] - closest_point[1])**2

        return distance_sq <= radius**2

    def _line_intersects_line(self, line1_start, line1_end, line2_start, line2_end):
        """Check if two line segments intersect.

        Args:
            line1_start: Tuple (x, y) of first line's start point
            line1_end: Tuple (x, y) of first line's end point
            line2_start: Tuple (x, y) of second line's start point
            line2_end: Tuple (x, y) of second line's end point

        Returns:
            bool: True if the line segments intersect
        """
        # Implementation of the line segment intersection algorithm
        def ccw(A, B, C):
            return (C[1]-A[1])*(B[0]-A[0]) > (B[1]-A[1])*(C[0]-A[0])

        A = line1_start
        B = line1_end
        C = line2_start
        D = line2_end

        # Check if lines intersect
        return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)

    def _update_passing_lanes(self):
        """Compute pass lanes and one-timer lanes for both teams."""
        for team in [self.team1, self.team2]:
            for player in team.players:
                player.passing_lane_clear = False
                player.one_timer_lane_good = False

            opponents = self.team2 if team.controller == 1 else self.team1
            passer = self._get_passer_for_team(team)
            if passer is None:
                continue

            passer_in_attack_band = self._is_receiver_in_attack_band(passer, team)

            # Get all potential obstacles
            obstacles = []

            # 1. Add opponent players
            obstacles.extend(opponents.players)

            # 2. Add opponent goalie
            if passer is not opponents.goalie:
                obstacles.append(opponents.goalie)

            for i, player in enumerate(team.players):
                if player is not passer:
                    catch_point = self._receiver_catch_point(player)
                    path_clear = self._pass_direction_matches_target(passer, catch_point)
                    if path_clear:
                        path_clear = self._is_passing_lane_clear(
                            (passer.x, passer.y),
                            catch_point,
                            obstacles,
                        )

                    in_attack_band = self._is_receiver_in_attack_band(player, team)
                    in_central_lane = self._is_receiver_in_central_lane(player)
                    shot_lane_clear = False
                    if path_clear and in_attack_band and in_central_lane:
                        shot_lane_clear = self._has_clear_one_timer_shot_lane(player, team, opponents)

                    player.passing_lane_clear = path_clear
                    player.one_timer_lane_good = (
                        path_clear
                        and passer_in_attack_band
                        and in_attack_band
                        and in_central_lane
                        and shot_lane_clear
                    )

    def _update_shot_lanes(self):
        """Compute shot lane flags only for the skater controlling the puck."""
        for team in [self.team1, self.team2]:
            for player in team.players:
                player.clear_shot_lane = False
                player.open_net_shot = False

            puck_controller = self._get_puck_controlled_skater_for_team(team)
            if puck_controller is None:
                continue

            opponents = self.team2 if team.controller == 1 else self.team1
            puck_controller.clear_shot_lane = self._has_clear_shot_lane(puck_controller, team, opponents)
            if puck_controller.clear_shot_lane:
                puck_controller.open_net_shot = self._has_open_net_shot(puck_controller, team, opponents)

    def _update_opponent_controlled_distances(self):
        """Update distances to the opponent's controlled player for all players."""
        for team in [self.team1, self.team2]:
            opponent_team = self.team2 if team.controller == 1 else self.team1
            opp_controlled_player = opponent_team.get_controlled_player()

            # Update distances for all players in the current team
            for player in team.players + [team.goalie]:
                player.dist_to_controlled_opp = 0 if opp_controlled_player is None else GameConsts.Distance(
                    (player.x, player.y),
                    (opp_controlled_player.x, opp_controlled_player.y)
                )

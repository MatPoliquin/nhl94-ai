from __future__ import annotations

from nhl94_ai.artifacts import load_policy, run_metadata
from nhl94_ai.agents.base import AgentInput, LearnedAgent, FrameRepeatAgent
from nhl94_ai.config import default_config_path
import argparse
import os
import random
import sys
import threading
from collections import deque
from dataclasses import dataclass
from typing import Callable, Deque, List, Optional, Sequence, Tuple
import numpy as np
import pygame
import pygame.freetype
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from nhl94_ai.training.logging import com_print, create_output_dir, get_model_file_name, init_logger
from nhl94_ai.env.factory import get_button_names, init_env
from nhl94_ai.models.factory import get_model_probabilities, init_model
from nhl94_ai.config import load_hyperparams, resolve_hyperparams_for_model
import nhl94_ai.env.components as games
from nhl94_ai.game.state import NHL94GameState
from nhl94_ai.training.events import LiveTrainingState
from nhl94_ai.evaluation.metrics import LiveTeamTotals, NHL94_ENVS, resolve_nhl94_players_per_team, reset_team_totals
from nhl94_ai.evaluation.metrics import _first_item
from nhl94_ai.artifacts import ensure_zip_path
SIM_STEPS_PER_SECOND = 60


def format_playtime(steps: int) -> str:
    if steps <= 0:
        return "0h:00m"

    total_seconds = int(steps / SIM_STEPS_PER_SECOND)
    hours, remainder = divmod(total_seconds, 3600)
    minutes = remainder // 60
    return f"{hours}h:{minutes:02d}m"


@dataclass
class ButtonLayout:
    buttons_per_row: int
    cell_width: int
    cell_height: int
    rows: int
    total_height: int


class LiveTrainingDisplay(threading.Thread):
    """Pygame window that replays the latest evaluated model and plots rewards."""

    BACKGROUND_COLOR = (38, 62, 116)
    BACKGROUND_PATTERN_COLOR = (90, 125, 185)
    PANEL_BG_COLOR = (54, 82, 142)
    PANEL_BORDER_COLOR = (120, 168, 242)
    GRAPH_BG = (44, 70, 126)
    GRAPH_AXIS = (136, 176, 236)
    GRAPH_LINE = (122, 194, 255)
    GRAPH_POINT = (245, 190, 90)
    TEXT_COLOR = (230, 238, 255)
    TITLE_COLOR = (255, 255, 255)
    BUTTON_BORDER_ACTIVE = (255, 210, 130)
    BUTTON_BORDER_INACTIVE = (130, 150, 200)
    BUTTON_BG = (62, 90, 148)
    BUTTON_BG_INACTIVE = (52, 78, 128)
    BUTTON_TEXT_ACTIVE = (255, 228, 150)
    BUTTON_TEXT_INACTIVE = (210, 220, 245)

    BUTTON_LABEL_OVERRIDES = {
        "UP": "↑",
        "DOWN": "↓",
        "LEFT": "←",
        "RIGHT": "→",
        "START": "S",
        "MODE": "M",
    }

    def __init__(
        self,
        args: argparse.Namespace,
        shared_state: LiveTrainingState,
        fps: int,
        graph_width: int,
    ) -> None:
        super().__init__(daemon=True)
        self.args = args
        self.is_target_policy = getattr(args, 'action_type', 'FILTERED').upper() == 'TARGET_POSITION'
        self.target_agent = None
        self.target_diagnostics = None
        self.target_frame_info = {}
        self.target_font = None
        self.shared_state = shared_state
        self.fps = max(1, fps)
        self.graph_width = graph_width

        self.info_height = 120
        self.margin = 20
        self.loaded_model_name: Optional[str] = None
        self.loaded_model_timesteps: Optional[int] = None
        self.display_model_param_count: Optional[int] = None

        self.running = True
        self.model_version_loaded = -1
        self.display_model: Optional[PPO] = None
        self.obs = None

        self.env = None
        self.screen: Optional[pygame.Surface] = None
        self.clock = pygame.time.Clock()
        self.header_font = None
        self.subheader_font = None
        self.body_font = None
        self.button_font = None
        self.button_prob_font = None
        self.background_pattern = None
        self.button_names: List[str] = []
        self.button_probs: Optional[np.ndarray] = None
        self.button_actions: Optional[np.ndarray] = None
        self.uses_nhl94_gamestate = args.env in NHL94_ENVS
        nhl94_players_per_team = resolve_nhl94_players_per_team(args.env, args.num_players)
        self.game_state = NHL94GameState(nhl94_players_per_team) if self.uses_nhl94_gamestate else None
        self.total_sessions_played = 0
        self.total_team1_stats = LiveTeamTotals()
        self.total_team2_stats = LiveTeamTotals()
        self.last_team1_session_stats: Optional[LiveTeamTotals] = None
        self.last_team2_session_stats: Optional[LiveTeamTotals] = None

    def stop(self) -> None:
        self.running = False

    def _ensure_replay_env(self) -> bool:
        if self.env is not None:
            return True

        try:
            self.env = init_env(
                None,
                1,
                self.args.state,
                self.args.num_players,
                self.args,
                self.args.hyperparams_dict,
                use_sticky_action=False,
                use_frame_skip=False,
            )
            self.obs = self.env.reset()
            return True
        except Exception as exc:  # pylint: disable=broad-except
            com_print(f"[Live UI] Failed to create replay environment: {exc}")
            self.env = None
            self.obs = None
            return False

    def _ensure_model_loaded(self) -> None:
        # The writer uses this same lock for the checkpoint and its sidecar.
        with self.shared_state.lock:
            version = self.shared_state.model_version
            base_path = self.shared_state.latest_model_path
            if base_path is None or version == 0 or version == self.model_version_loaded:
                return

            if not self._ensure_replay_env():
                return

            zip_path = ensure_zip_path(base_path)
            try:
                expected = run_metadata(self.args, self.args.hyperparams_dict) if self.is_target_policy else None
                model = load_policy(zip_path, env=self.env, expected=expected)
                target_agent = (
                    FrameRepeatAgent(LearnedAgent(model, 'TARGET_POSITION'),
                                     self.args.hyperparams_dict.get('frame_skip', 4))
                    if self.is_target_policy else None
                )
                observation = self.env.reset()
                self.display_model = model
                self.target_agent = target_agent
                self.obs = observation
                self._begin_new_replay_session()
                self.model_version_loaded = version
                self.loaded_model_name = os.path.basename(zip_path)
                self.loaded_model_timesteps = self.shared_state.latest_model_timesteps
                try:
                    self.display_model_param_count = sum(
                        p.numel() for p in model.policy.parameters() if p.requires_grad
                    )
                except Exception:  # pylint: disable=broad-except
                    self.display_model_param_count = None
                com_print(f"[Live UI] Loaded latest model at {self.loaded_model_timesteps:,} steps "
                          f"(version {version}) from {zip_path}")
            except Exception as exc:  # pylint: disable=broad-except
                com_print(f"[Live UI] Failed to load latest model: {exc}")

    @staticmethod
    def _extract_frame(env) -> Optional[np.ndarray]:
        frame = None
        try:
            frame = env.render(mode="rgb_array")
            if isinstance(frame, Sequence):
                frame = next((img for img in frame if img is not None), None)
        except TypeError:
            frame = None

        if frame is None and hasattr(env, "get_images"):
            try:
                images = env.get_images()
                if images:
                    frame = next((img for img in images if img is not None), None)
            except (NotImplementedError, TypeError):
                frame = None

        if frame is None:
            try:
                frames = env.env_method("render")
                if isinstance(frames, Sequence):
                    frame = next((img for img in frames if img is not None), None)
                else:
                    frame = frames
            except (TypeError, AttributeError):
                frame = None

        return frame

    def _fill_with_pattern(self, surface: pygame.Surface, rect: pygame.Rect) -> None:
        surface.fill(self.BACKGROUND_COLOR, rect)
        if self.background_pattern is None:
            return
        pattern_w, pattern_h = self.background_pattern.get_size()
        start_x = rect.left - (rect.left % pattern_w)
        start_y = rect.top - (rect.top % pattern_h)
        for x in range(start_x, rect.right, pattern_w):
            for y in range(start_y, rect.bottom, pattern_h):
                surface.blit(self.background_pattern, (x, y))

    def _prepare_action_info(
        self,
        probabilities: Optional[np.ndarray],
        actions: Optional[np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray]:
        button_count = len(self.button_names)

        probs = np.array(probabilities if probabilities is not None else [], dtype=float)
        if probs.ndim > 1:
            probs = probs.reshape(probs.shape[0], -1)[0]
        probs = probs.reshape(-1)

        acts = np.array(actions if actions is not None else [], dtype=float)
        if acts.ndim > 1:
            acts = acts.reshape(acts.shape[0], -1)[0]
        acts = acts.reshape(-1)

        if button_count == 0:
            return probs, acts

        if probs.size < button_count:
            if probs.size == 0:
                probs = np.zeros(button_count, dtype=float)
            else:
                probs = np.pad(probs, (0, button_count - probs.size), constant_values=0.0)
        elif probs.size > button_count:
            probs = probs[:button_count]

        if acts.size < button_count:
            if acts.size == 1 and button_count > 1:
                discrete_index = int(acts[0])
                one_hot = np.zeros(button_count, dtype=float)
                if 0 <= discrete_index < button_count:
                    one_hot[discrete_index] = 1.0
                acts = one_hot
            elif acts.size == 0:
                acts = np.zeros(button_count, dtype=float)
            else:
                acts = np.pad(acts, (0, button_count - acts.size), constant_values=0.0)
        elif acts.size > button_count:
            acts = acts[:button_count]

        return probs, acts

    def _draw_reward_graph(self, surface: pygame.Surface, rect: pygame.Rect) -> None:
        surface.fill(self.GRAPH_BG, rect)
        pygame.draw.rect(surface, self.PANEL_BORDER_COLOR, rect, width=2, border_radius=12)

        with self.shared_state.lock:
            history = list(self.shared_state.reward_history)
            latest_step = self.shared_state.latest_eval_timesteps
            latest_reward = self.shared_state.latest_eval_reward

        if len(history) < 2:
            font = pygame.freetype.SysFont("Arial", 22)
            text = "Waiting for evaluations..." if not history else "Collecting more data..."
            font.render_to(
                surface,
                (rect.x + 20, rect.y + rect.height // 2),
                text,
                self.TEXT_COLOR,
            )
            return

        steps = np.array([pt[0] for pt in history], dtype=float)
        rewards = np.array([pt[1] for pt in history], dtype=float)

        min_reward = float(np.min(rewards))
        max_reward = float(np.max(rewards))
        if np.isclose(min_reward, max_reward):
            max_reward += 1.0
            min_reward -= 1.0

        min_step = float(np.min(steps))
        max_step = float(np.max(steps))
        if np.isclose(min_step, max_step):
            max_step += 1.0

        margin = 20
        graph_rect = pygame.Rect(
            rect.x + margin,
            rect.y + margin,
            rect.width - 2 * margin,
            rect.height - 2 * margin,
        )
        pygame.draw.rect(surface, self.GRAPH_AXIS, graph_rect, width=2, border_radius=8)

        norm_points: List[Tuple[int, int]] = []
        for step_val, reward_val in history:
            x_pos = graph_rect.left + int((step_val - min_step) / (max_step - min_step) * graph_rect.width)
            y_pos = graph_rect.bottom - int((reward_val - min_reward) / (max_reward - min_reward) * graph_rect.height)
            norm_points.append((x_pos, y_pos))

        if len(norm_points) >= 2:
            pygame.draw.lines(surface, self.GRAPH_LINE, False, norm_points, width=3)

        if latest_reward is not None:
            latest_x = graph_rect.left + int((latest_step - min_step) / (max_step - min_step) * graph_rect.width)
            latest_y = graph_rect.bottom - int((latest_reward - min_reward) / (max_reward - min_reward) * graph_rect.height)
            pygame.draw.circle(surface, self.GRAPH_POINT, (latest_x, latest_y), 6)
            label_font = pygame.freetype.SysFont("Arial", 18)
            label_font.render_to(
                surface,
                (latest_x + 8, latest_y - 10),
                f"{latest_reward:.2f}",
                self.GRAPH_POINT,
            )

        caption_font = pygame.freetype.SysFont("Arial", 20)
        caption_font.render_to(
            surface,
            (graph_rect.left - 10, graph_rect.top - 2),
            f"{max_reward:.1f}",
            self.TEXT_COLOR,
        )
        caption_font.render_to(
            surface,
            (graph_rect.left - 10, graph_rect.bottom - 20),
            f"{min_reward:.1f}",
            self.TEXT_COLOR,
        )

    def _create_background_pattern(self) -> pygame.Surface:
        pattern = pygame.Surface((16, 16), pygame.SRCALPHA)
        color = (*self.BACKGROUND_PATTERN_COLOR, 90)
        pygame.draw.line(pattern, color, (0, 0), (15, 15), 2)
        pygame.draw.line(pattern, color, (0, 8), (7, 15), 2)
        pygame.draw.line(pattern, color, (8, 0), (15, 7), 2)
        return pattern

    def _draw_background(self, surface: pygame.Surface, width: int, height: int) -> None:
        surface.fill(self.BACKGROUND_COLOR)
        if self.background_pattern is None:
            return
        pattern_w, pattern_h = self.background_pattern.get_size()
        for x in range(0, width, pattern_w):
            for y in range(0, height, pattern_h):
                surface.blit(self.background_pattern, (x, y))

    def _compute_button_layout(self, available_width: int) -> ButtonLayout:
        if not self.button_names:
            return ButtonLayout(buttons_per_row=1, cell_width=80, cell_height=40, rows=1, total_height=60)

        padding = 24
        cell_gap = 6
        width = max(1, available_width - padding)
        button_count = len(self.button_names)
        width_for_buttons = max(1, width - (button_count - 1) * cell_gap)
        base_width = width_for_buttons / max(1, button_count)

        cell_width = min(96.0, base_width)
        if cell_width < 18.0:
            cell_width = base_width
        if cell_width * button_count + cell_gap * (button_count - 1) > width:
            cell_width = (width - cell_gap * (button_count - 1)) / max(1, button_count)
        cell_width = max(12.0, cell_width)

        cell_width_int = max(16, int(cell_width))
        cell_height = max(28, min(44, int(cell_width_int * 0.55)))
        total_height = cell_height + 16

        return ButtonLayout(
            buttons_per_row=button_count,
            cell_width=cell_width_int,
            cell_height=cell_height,
            rows=1,
            total_height=total_height,
        )

    def _draw_button_row(self, surface: pygame.Surface, rect: pygame.Rect, layout: ButtonLayout) -> None:
        self._fill_with_pattern(surface, rect)
        pygame.draw.rect(surface, self.PANEL_BORDER_COLOR, rect, width=2, border_radius=10)

        if self.button_font is None or self.button_prob_font is None:
            return
        if self.is_target_policy:
            from nhl94_ai.game.constants import GameConsts
            buttons = self.target_diagnostics['buttons'] if self.target_diagnostics else [0] * 12
            names = [(GameConsts.INPUT_UP, 'UP'), (GameConsts.INPUT_DOWN, 'DOWN'),
                     (GameConsts.INPUT_LEFT, 'LEFT'), (GameConsts.INPUT_RIGHT, 'RIGHT'),
                     (GameConsts.INPUT_B, 'B'), (GameConsts.INPUT_C, 'C')]
            active = ' '.join(name for index, name in names if buttons[index]) or 'none'
            self.body_font.render_to(surface, (rect.x + 12, rect.y + 12),
                                     f"Controller buttons: {active}", self.TEXT_COLOR)
            return

        if not self.button_names:
            self.body_font.render_to(
                surface,
                (rect.x + 16, rect.centery - 10),
                "Button metadata unavailable",
                self.TEXT_COLOR,
            )
            return

        probs = self.button_probs if self.button_probs is not None else np.zeros(len(self.button_names))
        actions = self.button_actions if self.button_actions is not None else np.zeros(len(self.button_names))

        buttons = list(zip(range(len(self.button_names)), self.button_names, probs, actions))
        cell_gap = 8

        for idx, name, prob, action_val in buttons:
            column = idx % max(1, layout.buttons_per_row)
            row = idx // max(1, layout.buttons_per_row)

            pad_x = rect.x + 12 + column * (layout.cell_width + cell_gap)
            pad_y = rect.y + 8 + row * (layout.cell_height + cell_gap)

            button_rect = pygame.Rect(pad_x, pad_y, layout.cell_width, layout.cell_height)
            is_active = bool(action_val >= 0.5)
            bg_color = self.BUTTON_BG if is_active else self.BUTTON_BG_INACTIVE
            border_color = self.BUTTON_BORDER_ACTIVE if is_active else self.BUTTON_BORDER_INACTIVE

            pygame.draw.rect(surface, bg_color, button_rect, border_radius=8)
            pygame.draw.rect(surface, border_color, button_rect, width=2 if is_active else 1, border_radius=8)

            label_raw = name if isinstance(name, str) and name else f"BTN{idx}"
            label = self.BUTTON_LABEL_OVERRIDES.get(str(label_raw).upper(), str(label_raw))
            label_color = self.BUTTON_TEXT_ACTIVE if is_active else self.BUTTON_TEXT_INACTIVE
            label_surface, _ = self.button_font.render(label, fgcolor=label_color)
            label_surface.set_alpha(220 if is_active else 150)
            label_rect = label_surface.get_rect()
            label_rect.midtop = (button_rect.centerx, button_rect.top + 2)
            surface.blit(label_surface, label_rect)

            prob_surface, _ = self.button_prob_font.render(f"{prob:0.2f}", fgcolor=self.TEXT_COLOR)
            prob_surface.set_alpha(220 if is_active else 170)
            prob_rect = prob_surface.get_rect()
            prob_rect.midbottom = (button_rect.centerx, button_rect.bottom - 4)
            surface.blit(prob_surface, prob_rect)

    def _load_button_names(self) -> List[str]:
        try:
            names = get_button_names(self.args)
            return [str(name) for name in names]
        except Exception:  # pylint: disable=broad-except
            return []

    def _begin_new_replay_session(self) -> None:
        self.total_sessions_played += 1
        self.last_team1_session_stats = None
        self.last_team2_session_stats = None
        if self.uses_nhl94_gamestate:
            self.last_team1_session_stats, self.last_team2_session_stats = reset_team_totals(self.env)
        if self.target_agent is not None:
            self.target_agent.reset()
            self.target_diagnostics = None
            self.target_frame_info = {}

    def _update_total_game_stats(self) -> None:
        if not self.uses_nhl94_gamestate or self.game_state is None:
            return

        team1_stats = LiveTeamTotals.from_game_stats(self.game_state.team1.stats)
        team2_stats = LiveTeamTotals.from_game_stats(self.game_state.team2.stats)

        self.total_team1_stats.add_delta(self.last_team1_session_stats, team1_stats)
        self.total_team2_stats.add_delta(self.last_team2_session_stats, team2_stats)

        self.last_team1_session_stats = team1_stats
        self.last_team2_session_stats = team2_stats

    def run(self) -> None:  # noqa: D401
        pygame.init()
        pygame.freetype.init()

        games.get_bindings(self.args)

        self.header_font = pygame.freetype.SysFont("Arial", 30)
        self.subheader_font = pygame.freetype.SysFont("Arial", 22)
        self.body_font = pygame.freetype.SysFont("Arial", 20)
        self.button_font = pygame.freetype.SysFont("Arial", 20)
        self.button_prob_font = pygame.freetype.SysFont("Courier New", 18)
        self.target_font = pygame.font.SysFont('Arial', 14)
        self.background_pattern = self._create_background_pattern()
        self.button_names = self._load_button_names()
        if self.button_names:
            zeros = np.zeros(len(self.button_names), dtype=float)
            self.button_probs = zeros.copy()
            self.button_actions = zeros.copy()

        window_width = self.args.live_window_width
        window_height = self.args.live_window_height
        self.screen = pygame.display.set_mode((window_width, window_height))
        pygame.display.set_caption("Live Training Viewer")

        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    with self.shared_state.lock:
                        self.shared_state.stop_requested = True
                    self.running = False

            if not self.running:
                break

            self._ensure_model_loaded()

            frame = None
            if self.display_model is not None and self.obs is not None and self.env is not None:
                if self.is_target_policy:
                    action = self.target_agent.act(AgentInput(None, self.obs)).action
                else:
                    action, _state = self.display_model.predict(self.obs, deterministic=True)
                    prob_vals = None
                    try:
                        prob_vals = get_model_probabilities(self.display_model, self.obs)
                    except Exception:  # pylint: disable=broad-except
                        prob_vals = None
                    action_arr = np.array(action, dtype=float)
                    probs, acts = self._prepare_action_info(prob_vals, action_arr)
                    self.button_probs = probs
                    self.button_actions = acts
                try:
                    self.obs, _reward, done, info = self.env.step(action)
                    if self.is_target_policy:
                        self.target_diagnostics = info[0]['target_control']
                        self.target_frame_info = info[0]
                    if self.uses_nhl94_gamestate and self.game_state is not None:
                        self.game_state.BeginFrame(info[0], [0] * 6)
                        self.game_state.EndFrame()
                        self._update_total_game_stats()
                    if np.any(done):
                        if self.is_target_policy:
                            self.game_state = self.env.get_attr('game_state')[0]
                        else:
                            self.obs = self.env.reset()
                        self._begin_new_replay_session()
                except Exception as exc:  # pylint: disable=broad-except
                    com_print(f"[Live UI] env.step failed: {exc}")
                    if self.is_target_policy:
                        with self.shared_state.lock:
                            self.shared_state.stop_requested = True
                        self.running = False
                        break
                    self.obs = self.env.reset()
                    self._begin_new_replay_session()

                try:
                    frame = self._extract_frame(self.env)
                except Exception:  # pylint: disable=broad-except
                    frame = None

            if self.screen is None:
                break

            current_width, current_height = self.screen.get_size()
            self._draw_background(self.screen, current_width, current_height)

            info_rect = pygame.Rect(0, 0, current_width, self.info_height)
            game_width = max(1, current_width - self.graph_width - 3 * self.margin)
            button_layout = self._compute_button_layout(game_width)
            buttons_height = button_layout.total_height
            game_height = max(1, current_height - self.info_height - 3 * self.margin - buttons_height)
            graph_height = max(1, game_height * 0.5)

            game_rect = pygame.Rect(self.margin, self.info_height + self.margin, game_width, game_height)
            graph_rect = pygame.Rect(
                current_width - self.graph_width - self.margin,
                self.info_height + self.margin,
                self.graph_width,
                graph_height,
            )
            buttons_rect = pygame.Rect(
                game_rect.x,
                game_rect.bottom + self.margin,
                game_rect.width,
                buttons_height,
            )

            self._draw_info_bar(self.screen, info_rect)

            replay_label = "Latest model"
            if self.loaded_model_timesteps is not None:
                replay_label += f" - {self.loaded_model_timesteps:,} steps"
            self.body_font.render_to(self.screen, (game_rect.x, game_rect.y - self.margin + 2),
                                     replay_label, self.TEXT_COLOR, size=14)
            self.screen.fill(self.PANEL_BG_COLOR, game_rect)
            image_rect = game_rect.copy()
            target_rect = None
            if self.is_target_policy:
                target_width = max(140, game_rect.width // 3)
                image_rect.width -= target_width
                target_rect = pygame.Rect(image_rect.right, game_rect.y, target_width, game_rect.height)

            if frame is not None:
                try:
                    frame_surface = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
                    frame_surface = pygame.transform.smoothscale(frame_surface, image_rect.size)
                    self.screen.blit(frame_surface, image_rect)
                    if self.is_target_policy:
                        from nhl94_ai.ui.targets import draw_game_target
                        draw_game_target(self.screen, image_rect, (frame.shape[1], frame.shape[0]),
                                         self.target_frame_info)
                except Exception as exc:  # pylint: disable=broad-except
                    com_print(f"[Live UI] Failed to blit frame: {exc}")
            else:
                font = pygame.freetype.SysFont("Arial", 28)
                font.render_to(
                    self.screen,
                    (game_rect.x + 40, game_rect.centery),
                    "Waiting for first training snapshot...",
                    self.TEXT_COLOR,
                )

            pygame.draw.rect(self.screen, self.PANEL_BORDER_COLOR, game_rect, width=3, border_radius=12)
            if target_rect is not None:
                from nhl94_ai.ui.targets import draw_target_rink
                draw_target_rink(self.screen, target_rect, self.game_state, self.target_diagnostics, self.target_font)

            self._draw_button_row(self.screen, buttons_rect, button_layout)

            self._draw_reward_graph(self.screen, graph_rect)
            self._draw_game_stats(self.screen, graph_rect)

            pygame.display.flip()
            self.clock.tick(self.fps)
            if self.args.live_sleep_per_step > 0:
                pygame.time.delay(int(self.args.live_sleep_per_step * 1000))

            with self.shared_state.lock:
                if not self.shared_state.training_active and self.display_model is None:
                    self.running = False

        pygame.quit()
        if self.env is not None:
            self.env.close()

    def _draw_info_bar(self, surface: pygame.Surface, rect: pygame.Rect) -> None:
        self._fill_with_pattern(surface, rect)

        if self.header_font is None or self.subheader_font is None or self.body_font is None:
            return

        algo_text = self.args.alg.upper()
        policy_text = self.args.nn
        env_text = self.args.env
        param_text = (
            f"{self.display_model_param_count:,}"
            if self.display_model_param_count is not None
            else "—"
        )

        with self.shared_state.lock:
            best_reward_value = self.shared_state.best_mean_reward
            latest_steps = self.shared_state.latest_eval_timesteps

        best_reward_text = (
            f"{best_reward_value:.2f}" if best_reward_value != float("-inf") else "—"
        )
        playtime_text = format_playtime(latest_steps)
        sessions_text = f"{self.total_sessions_played:,}"

        info_items = [
            ("Environment", env_text),
            ("Algorithm", algo_text),
            ("Policy", policy_text),
            ("Parameters", param_text),
            ("Best Reward", best_reward_text),
            ("Playtime", playtime_text),
            ("Total Sessions", sessions_text),
        ]

        count = len(info_items)
        available_width = rect.width - 2 * self.margin
        spacing = self.margin
        box_height = rect.height - 2 * self.margin
        box_width = max(140, (available_width - (count - 1) * spacing) // count)
        x = rect.x + self.margin
        y = rect.y + self.margin

        for idx, (title, value) in enumerate(info_items):
            remaining_width = rect.right - self.margin - x
            current_width = box_width if idx < count - 1 else max(160, remaining_width)
            box_rect = pygame.Rect(int(x), int(y), int(current_width), int(box_height))
            surface.fill(self.PANEL_BG_COLOR, box_rect)
            pygame.draw.rect(surface, self.PANEL_BORDER_COLOR, box_rect, width=2, border_radius=10)

            self.body_font.render_to(
                surface,
                (box_rect.x + 12, box_rect.y + 10),
                title,
                self.TITLE_COLOR,
            )
            self.header_font.render_to(
                surface,
                (box_rect.x + 12, box_rect.y + box_rect.height // 2 + 4),
                value,
                self.TEXT_COLOR,
            )

            x += current_width + spacing

    def _draw_game_stats(self, surface: pygame.Surface, graph_rect: pygame.Rect) -> None:
        if self.body_font is None or self.subheader_font is None:
            return

        if not self.uses_nhl94_gamestate or self.game_state is None:
            return

        stats_rect = pygame.Rect(
            graph_rect.x,
            graph_rect.bottom + self.margin,
            graph_rect.width,
            120,
        )
        surface.fill(self.PANEL_BG_COLOR, stats_rect)
        pygame.draw.rect(surface, self.PANEL_BORDER_COLOR, stats_rect, width=2, border_radius=10)

        t1 = self.total_team1_stats
        t2 = self.total_team2_stats

        self.subheader_font.render_to(surface, (stats_rect.x + 14, stats_rect.y + 10), "Team 1 Totals", self.TITLE_COLOR)
        self.subheader_font.render_to(surface, (stats_rect.x + stats_rect.width // 2 + 14, stats_rect.y + 10), "Team 2 Totals", self.TITLE_COLOR)

        rows = [
            ("Goals", t1.goals, t2.goals),
            ("Shots", t1.shots, t2.shots),
            ("Passes", t1.passes, t2.passes),
            ("One Timers", t1.one_timers, t2.one_timers),
        ]

        for idx, (label, v1, v2) in enumerate(rows):
            line_y = stats_rect.y + 40 + idx * 20
            self.body_font.render_to(surface, (stats_rect.x + 14, line_y), f"{label}: {v1}", self.TEXT_COLOR)
            self.body_font.render_to(surface, (stats_rect.x + stats_rect.width // 2 + 14, line_y), f"{label}: {v2}", self.TEXT_COLOR)

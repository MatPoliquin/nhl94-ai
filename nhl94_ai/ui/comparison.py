import argparse
import copy
import io
import json
import math
import os
import time
from contextlib import redirect_stdout
from typing import List, Optional, Tuple
import cv2
import numpy as np
import pygame
import pygame.freetype
from stable_baselines3 import PPO
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from nhl94_ai.training.logging import init_logger
from nhl94_ai.env.factory import init_env, get_button_names
from nhl94_ai.models.factory import get_num_parameters, get_model_probabilities
from torchsummary import summary
import nhl94_ai.env.components as games
import nhl94_ai.training.rl as train
from nhl94_ai.config import load_hyperparams, resolve_hyperparams_for_model
DEFAULT_OUTPUT_DIR = os.path.expanduser("~/OUTPUT/compare_models")
DEFAULT_FPS = 60
DEFAULT_DURATION_SECONDS = 120
DEFAULT_HYPERPARAMS = os.path.join(os.path.dirname(__file__), "..", "hyperparams", "default.json")
DEFAULT_WINDOW_WIDTH = 1920
DEFAULT_WINDOW_HEIGHT = 1080
DEFAULT_MARGIN = 20
DEFAULT_FOOTER_HEIGHT = 120
DEFAULT_FONT = "Arial"
BUTTON_LABEL_OVERRIDES = {
    "UP": "↑",
    "DOWN": "↓",
    "LEFT": "←",
    "RIGHT": "→",
    "START": "S",
    "MODE": "M",
}
from nhl94_ai.evaluation.frames import get_env_frame, prepare_action_info


def total_surface_height(surfaces: List["pygame.Surface"], line_gap: int) -> int:
    if not surfaces:
        return 0
    total = 0
    for idx, surface in enumerate(surfaces):
        total += surface.get_height()
        if idx < len(surfaces) - 1:
            total += line_gap
    return total


def draw_centered_stack(target: "pygame.Surface", rect: "pygame.Rect",
                        surfaces: List["pygame.Surface"], line_gap: int, padding: int) -> None:
    if not surfaces:
        return
    content_height = total_surface_height(surfaces, line_gap)
    available = rect.height - content_height
    start_y = rect.top + max(padding, available / 2)
    current_y = start_y
    for idx, surface in enumerate(surfaces):
        surface_rect = surface.get_rect()
        surface_rect.midtop = (rect.centerx, int(round(current_y)))
        target.blit(surface, surface_rect)
        current_y += surface.get_height()
        if idx < len(surfaces) - 1:
            current_y += line_gap


def create_reward_curve_surface(
    window_size: Tuple[int, int],
    rewards_a: List[Tuple[float, float]],
    rewards_b: List[Tuple[float, float]],
    label_a: str,
    label_b: str,
) -> Optional["pygame.Surface"]:
    if not rewards_a and not rewards_b:
        return None

    pygame.font.init()
    plot_width, plot_height = window_size
    surface = pygame.Surface(window_size)
    surface.fill((0, 0, 0))

    title_font = pygame.font.SysFont(DEFAULT_FONT, 48)
    subtitle_font = pygame.font.SysFont(DEFAULT_FONT, 32)

    title = title_font.render("Training Reward Curve", True, (255, 255, 255))
    surface.blit(title, title.get_rect(center=(plot_width // 2, 60)))

    subtitle = subtitle_font.render("Mean episode reward per training step", True, (180, 180, 180))
    surface.blit(subtitle, subtitle.get_rect(center=(plot_width // 2, 120)))

    margin = 120
    graph_rect = pygame.Rect(
        margin,
        margin + 80,
        plot_width - 2 * margin,
        plot_height - (margin + 80) - margin,
    )
    pygame.draw.rect(surface, (60, 60, 60), graph_rect, width=2)

    plotted_any = False

    all_rewards = [pt[1] for pt in rewards_a] + [pt[1] for pt in rewards_b]
    all_steps = [pt[0] for pt in rewards_a] + [pt[0] for pt in rewards_b]

    reward_min = min(all_rewards) if all_rewards else 0.0
    reward_max = max(all_rewards) if all_rewards else 1.0
    step_min = min(all_steps) if all_steps else 0.0
    step_max = max(all_steps) if all_steps else 1.0

    def draw_series(points: List[Tuple[float, float]], color: Tuple[int, int, int]) -> None:
        nonlocal plotted_any
        if len(points) < 2:
            return
        steps = [pt[0] for pt in points]
        rewards = [pt[1] for pt in points]
        min_reward = min(rewards)
        max_reward = max(rewards)
        if math.isclose(max_reward, min_reward):
            max_reward += 1.0
            min_reward -= 1.0

        step_min = min(steps)
        step_max = max(steps)
        if math.isclose(step_max, step_min):
            step_max += 1.0

        normalized_points = []
        for step, reward in points:
            x = graph_rect.left + (step - step_min) / (step_max - step_min) * graph_rect.width
            y = graph_rect.bottom - (reward - min_reward) / (max_reward - min_reward) * graph_rect.height
            normalized_points.append((x, y))

        pygame.draw.lines(surface, color, False, normalized_points, width=3)
        plotted_any = True

    draw_series(rewards_a, (255, 99, 71))
    draw_series(rewards_b, (30, 144, 255))

    legend_font = pygame.font.SysFont(DEFAULT_FONT, 28)
    legend_y = graph_rect.top - 40
    legend_padding = 20
    line_width = 40

    def draw_legend(color: Tuple[int, int, int], text: str, x_offset: int) -> int:
        pygame.draw.line(
            surface,
            color,
            (graph_rect.left + x_offset, legend_y),
            (graph_rect.left + x_offset + line_width, legend_y),
            width=6,
        )
        label = legend_font.render(text, True, (220, 220, 220))
        label_rect = label.get_rect()
        label_rect.topleft = (graph_rect.left + x_offset + line_width + 10, legend_y - label_rect.height // 2)
        surface.blit(label, label_rect)
        return x_offset + line_width + label_rect.width + legend_padding + 10

    x_offset = 0
    if rewards_a:
        x_offset = draw_legend((255, 99, 71), label_a, x_offset)
    if rewards_b:
        draw_legend((30, 144, 255), label_b, x_offset)

    axis_font = pygame.font.SysFont(DEFAULT_FONT, 28)
    axis_label_font = pygame.font.SysFont(DEFAULT_FONT, 32)

    has_step_range = bool(all_steps) and step_max > step_min
    has_reward_range = bool(all_rewards) and reward_max > reward_min

    # X-axis ticks
    if has_step_range:
        num_ticks = 5
        step_interval = (step_max - step_min) / (num_ticks - 1)
        for idx in range(num_ticks):
            step_value = step_min + idx * step_interval
            x = graph_rect.left + (step_value - step_min) / (step_max - step_min) * graph_rect.width
            pygame.draw.line(surface, (90, 90, 90), (x, graph_rect.bottom), (x, graph_rect.bottom + 10), 2)
            tick_text = axis_font.render(f"{int(step_value):,}", True, (200, 200, 200))
            text_rect = tick_text.get_rect()
            text_rect.center = (x, graph_rect.bottom + 30)
            surface.blit(tick_text, text_rect)
    elif all_steps:
        x = graph_rect.centerx
        pygame.draw.line(surface, (90, 90, 90), (x, graph_rect.bottom), (x, graph_rect.bottom + 10), 2)
        tick_text = axis_font.render(f"{int(step_min):,}", True, (200, 200, 200))
        text_rect = tick_text.get_rect()
        text_rect.center = (x, graph_rect.bottom + 30)
        surface.blit(tick_text, text_rect)

    # Y-axis ticks
    if has_reward_range:
        num_ticks = 5
        reward_interval = (reward_max - reward_min) / (num_ticks - 1)
        for idx in range(num_ticks):
            reward_value = reward_min + idx * reward_interval
            y = graph_rect.bottom - (reward_value - reward_min) / (reward_max - reward_min) * graph_rect.height
            pygame.draw.line(surface, (90, 90, 90), (graph_rect.left - 10, y), (graph_rect.left, y), 2)
            tick_text = axis_font.render(f"{reward_value:.1f}", True, (200, 200, 200))
            text_rect = tick_text.get_rect()
            text_rect.right = graph_rect.left - 15
            text_rect.centery = y
            surface.blit(tick_text, text_rect)
    elif all_rewards:
        y = graph_rect.centery
        pygame.draw.line(surface, (90, 90, 90), (graph_rect.left - 10, y), (graph_rect.left, y), 2)
        tick_text = axis_font.render(f"{reward_min:.1f}", True, (200, 200, 200))
        text_rect = tick_text.get_rect()
        text_rect.right = graph_rect.left - 15
        text_rect.centery = y
        surface.blit(tick_text, text_rect)

    # Axis labels
    x_axis_label = axis_label_font.render("Timesteps", True, (220, 220, 220))
    x_rect = x_axis_label.get_rect()
    x_rect.center = (graph_rect.centerx, graph_rect.bottom + 60)
    surface.blit(x_axis_label, x_rect)

    y_axis_label = axis_label_font.render("Reward", True, (220, 220, 220))
    y_axis_label = pygame.transform.rotate(y_axis_label, 90)
    y_rect = y_axis_label.get_rect()
    y_rect.center = (graph_rect.left - 70, graph_rect.centery)
    surface.blit(y_axis_label, y_rect)

    if not plotted_any:
        info_font = pygame.font.SysFont(DEFAULT_FONT, 32)
        message = "Not enough reward data to plot a curve"
        info_surface = info_font.render(message, True, (200, 200, 200))
        surface.blit(info_surface, info_surface.get_rect(center=graph_rect.center))

    return surface


def play_and_record(model1: PPO, env1, obs1: np.ndarray, label1: str, params1: int,
                    algorithm1: str, training_reward1: float,
                    model2: PPO, env2, obs2: np.ndarray, label2: str, params2: int,
                    algorithm2: str, training_reward2: float,
                    button_names: List[str],
                    reward_series1: List[Tuple[float, float]],
                    reward_series2: List[Tuple[float, float]],
                    reward_curve_duration: float,
                    hyperparams_text1: str, hyperparams_text2: str, hyperparams_duration: float,
                    summary_text1: str, summary_text2: str, summary_duration: float,
                    initial_frame1: np.ndarray, initial_frame2: np.ndarray,
                    video_path: str, duration_seconds: float, fps: int,
                    window_width: int, window_height: int) -> None:
    pygame.init()
    pygame.freetype.init()

    frame_height, frame_width = initial_frame1.shape[:2]
    aspect_ratio = frame_width / frame_height if frame_height else 4 / 3

    margin = DEFAULT_MARGIN
    footer_height = DEFAULT_FOOTER_HEIGHT

    available_width = window_width - (margin * 3)
    available_height = window_height - (margin * 2) - footer_height

    target_width = available_width // 2
    target_height = int(target_width / aspect_ratio)

    if target_height > available_height:
        target_height = available_height
        target_width = int(target_height * aspect_ratio)

    left_x = margin
    right_x = window_width - margin - target_width

    screen = pygame.display.set_mode((window_width, window_height))
    pygame.display.set_caption("Model Comparison")

    font_label = pygame.freetype.SysFont("Arial", 36)
    font_detail = pygame.freetype.SysFont("Arial", 28)
    font_timer = pygame.freetype.SysFont("Arial", 24)
    font_mono = pygame.freetype.SysFont("Courier New", 14)
    font_prob = pygame.freetype.SysFont("Courier New", 20)

    params_text1 = f"Parameters: {params1:,}"
    params_text2 = f"Parameters: {params2:,}"
    algo_text1 = f"Algorithm: {str(algorithm1).upper()}"
    algo_text2 = f"Algorithm: {str(algorithm2).upper()}"
    reward_text1 = f"Final Reward: {training_reward1:.2f}"
    reward_text2 = f"Final Reward: {training_reward2:.2f}"

    def draw_button_info(x_pos: int, start_y: int, names, probs, actions) -> int:
        buttons = list(zip(range(len(names)), names, probs, actions))
        if not buttons:
            return start_y

        cell_width = 76
        cell_height = 40
        available_width = window_width - 2 * margin
        buttons_per_row = max(1, min(len(buttons), max(1, available_width // cell_width)))

        for idx, name, prob, action in buttons:
            label_raw = name if isinstance(name, str) and name else f"BTN{idx}"
            display_label = BUTTON_LABEL_OVERRIDES.get(str(label_raw).upper(), str(label_raw))
            is_pressed = action >= 0.5
            col = idx % buttons_per_row
            row = idx // buttons_per_row
            base_x = x_pos + col * cell_width
            base_y = start_y + row * cell_height

            rect = pygame.Rect(base_x + 3, base_y + 3, cell_width - 9, cell_height - 9)
            bg_color = (60, 60, 60)
            border_color = (255, 200, 120) if is_pressed else (110, 110, 110)
            pygame.draw.rect(screen, bg_color, rect)
            pygame.draw.rect(screen, border_color, rect, width=3 if is_pressed else 1)

            label_color = (255, 220, 140) if is_pressed else (210, 210, 210)
            label_surface, _ = font_detail.render(display_label, fgcolor=label_color)
            label_surface.set_alpha(170 if is_pressed else 110)
            label_rect = label_surface.get_rect()
            label_rect.midtop = (rect.centerx, rect.top + 2)
            screen.blit(label_surface, label_rect)

            prob_text = f"{prob:>5.2f}"
            prob_surface, _ = font_prob.render(prob_text, fgcolor=(230, 230, 230))
            prob_rect = prob_surface.get_rect()
            prob_rect.center = rect.center
            screen.blit(prob_surface, prob_rect)

        rows_used = (len(buttons) + buttons_per_row - 1) // buttons_per_row
        return start_y + rows_used * cell_height

    os.makedirs(os.path.dirname(video_path) or ".", exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(video_path, fourcc, fps, (window_width, window_height))
    if not writer.isOpened():
        raise RuntimeError(f"Unable to open video writer for {video_path}")

    if fps <= 0:
        raise ValueError("FPS must be a positive integer")

    clock = pygame.time.Clock()

    requested_summary_frames = max(0, int(summary_duration * fps))
    summary_line_height = font_mono.get_sized_height() + 2
    max_summary_lines = max(1, (window_height - 2 * margin - footer_height - 80) // summary_line_height)
    requested_hyperparams_frames = max(0, int(hyperparams_duration * fps))
    hyperparams_line_height = summary_line_height

    def prepare_summary_lines(text: str) -> List[str]:
        raw_lines = [line.rstrip() for line in text.strip().splitlines()]
        if not raw_lines:
            raw_lines = ["Summary unavailable"]
        if len(raw_lines) > max_summary_lines:
            truncated = raw_lines[:max_summary_lines - 1]
            truncated.append("... (truncated) ...")
            return truncated
        return raw_lines

    summary_lines1 = prepare_summary_lines(summary_text1)
    summary_lines2 = prepare_summary_lines(summary_text2)
    hyperparams_lines1 = prepare_summary_lines(hyperparams_text1)
    hyperparams_lines2 = prepare_summary_lines(hyperparams_text2)

    reward_curve_surface = create_reward_curve_surface(
        (window_width, window_height), reward_series1, reward_series2, label1, label2
    )
    reward_curve_frames = max(0, int(reward_curve_duration * fps))

    def render_text_block(lines_left: List[str], lines_right: List[str], heading_left: str, heading_right: str) -> None:
        screen.fill((15, 15, 15))

        heading_offset = margin
        font_label.render_to(screen, (left_x, heading_offset), heading_left, (200, 200, 200))
        font_label.render_to(screen, (right_x, heading_offset), heading_right, (200, 200, 200))

        text_y_left = heading_offset + 40
        for line in lines_left:
            font_mono.render_to(screen, (left_x, text_y_left), line, (210, 210, 210))
            text_y_left += hyperparams_line_height

        text_y_right = heading_offset + 40
        for line in lines_right:
            font_mono.render_to(screen, (right_x, text_y_right), line, (210, 210, 210))
            text_y_right += hyperparams_line_height

    if reward_curve_surface is not None and reward_curve_frames > 0:
        for frame_idx in range(reward_curve_frames):
            quit_requested = False
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    quit_requested = True
                    break

            if quit_requested:
                writer.release()
                pygame.quit()
                return

            screen.blit(reward_curve_surface, (0, 0))

            remaining_time = max(reward_curve_duration - frame_idx / fps, 0.0)
            timer_text = f"Hyperparameters in {remaining_time:0.1f}s"
            font_timer.render_to(
                screen,
                (window_width - margin - 320, window_height - footer_height // 2),
                timer_text,
                (180, 180, 180),
            )

            pygame.display.flip()

            frame_pixels = pygame.surfarray.array3d(screen)
            frame_pixels = np.transpose(frame_pixels, (1, 0, 2))
            writer.write(cv2.cvtColor(frame_pixels, cv2.COLOR_RGB2BGR))
            clock.tick(fps)

    hyperparams_frames = requested_hyperparams_frames

    if hyperparams_frames > 0:
        for frame_idx in range(hyperparams_frames):
            quit_requested = False
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    quit_requested = True
                    break

            if quit_requested:
                writer.release()
                pygame.quit()
                return

            render_text_block(hyperparams_lines1, hyperparams_lines2,
                              f"{label1} Hyperparameters", f"{label2} Hyperparameters")

            remaining_time = max(hyperparams_duration - frame_idx / fps, 0.0)
            timer_text = f"Model summary in {remaining_time:0.1f}s"
            font_timer.render_to(screen, (window_width - margin - 310, window_height - footer_height // 2),
                                 timer_text, (180, 180, 180))

            pygame.display.flip()

            frame_pixels = pygame.surfarray.array3d(screen)
            frame_pixels = np.transpose(frame_pixels, (1, 0, 2))
            writer.write(cv2.cvtColor(frame_pixels, cv2.COLOR_RGB2BGR))
            clock.tick(fps)

    summary_frames = requested_summary_frames

    if summary_frames > 0:
        for frame_idx in range(summary_frames):
            quit_requested = False
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    quit_requested = True
                    break

            if quit_requested:
                writer.release()
                pygame.quit()
                return

            render_text_block(summary_lines1, summary_lines2,
                              f"{label1} Summary", f"{label2} Summary")

            remaining_time = max(summary_duration - frame_idx / fps, 0.0)
            timer_text = f"Gameplay starts in {remaining_time:0.1f}s"
            font_timer.render_to(screen, (window_width - margin - 280, window_height - footer_height // 2),
                                 timer_text, (180, 180, 180))

            pygame.display.flip()

            frame_pixels = pygame.surfarray.array3d(screen)
            frame_pixels = np.transpose(frame_pixels, (1, 0, 2))
            writer.write(cv2.cvtColor(frame_pixels, cv2.COLOR_RGB2BGR))
            clock.tick(fps)

    frame1 = initial_frame1
    frame2 = initial_frame2

    total_reward1 = 0.0
    total_reward2 = 0.0

    gameplay_frames = max(0, int(duration_seconds * fps))
    gameplay_start_time = time.time()
    action_repeat = 4
    repeat_counter = 0
    current_action1 = None
    current_action2 = None
    prob1_vals = np.array([])
    prob2_vals = np.array([])
    actions1_flat = np.array([])
    actions2_flat = np.array([])
    display_probs1 = np.zeros(len(button_names))
    display_probs2 = np.zeros(len(button_names))
    display_actions1 = np.zeros(len(button_names))
    display_actions2 = np.zeros(len(button_names))

    for frame_idx in range(gameplay_frames):
        quit_requested = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                quit_requested = True
                break

        if quit_requested:
            break

        if repeat_counter <= 0:
            try:
                prob1_vals = np.asarray(get_model_probabilities(model1, obs1), dtype=float).flatten()
            except (RuntimeError, AttributeError, ValueError, TypeError):
                prob1_vals = np.array([])
            current_action1, _ = model1.predict(obs1, deterministic=True)
            actions1_flat = np.asarray(current_action1, dtype=float).flatten()

            try:
                prob2_vals = np.asarray(get_model_probabilities(model2, obs2), dtype=float).flatten()
            except (RuntimeError, AttributeError, ValueError, TypeError):
                prob2_vals = np.array([])
            current_action2, _ = model2.predict(obs2, deterministic=True)
            actions2_flat = np.asarray(current_action2, dtype=float).flatten()

            display_probs1, display_actions1 = prepare_action_info(prob1_vals, actions1_flat, len(button_names))
            display_probs2, display_actions2 = prepare_action_info(prob2_vals, actions2_flat, len(button_names))
            repeat_counter = action_repeat

        obs1, reward1_arr, done1, _ = env1.step(current_action1)
        reward1_value = float(np.array(reward1_arr).reshape(-1)[0])
        total_reward1 += reward1_value
        done1_flag = bool(np.any(done1))
        if done1_flag:
            obs1 = env1.reset()
            repeat_counter = 0
        frame1 = get_env_frame(env1)

        obs2, reward2_arr, done2, _ = env2.step(current_action2)
        reward2_value = float(np.array(reward2_arr).reshape(-1)[0])
        total_reward2 += reward2_value
        done2_flag = bool(np.any(done2))
        if done2_flag:
            obs2 = env2.reset()
            repeat_counter = 0
        frame2 = get_env_frame(env2)

        repeat_counter = max(0, repeat_counter - 1)

        screen.fill((15, 15, 15))

        resized1 = cv2.resize(frame1, (target_width, target_height), interpolation=cv2.INTER_AREA)
        resized2 = cv2.resize(frame2, (target_width, target_height), interpolation=cv2.INTER_AREA)

        surf1 = pygame.surfarray.make_surface(np.transpose(resized1, (1, 0, 2)))
        surf2 = pygame.surfarray.make_surface(np.transpose(resized2, (1, 0, 2)))

        screen.blit(surf1, (left_x, margin))
        screen.blit(surf2, (right_x, margin))

        frame_color = (90, 90, 90)
        frame_rect_left = pygame.Rect(left_x - 4, margin - 4, target_width + 8, target_height + 8)
        frame_rect_right = pygame.Rect(right_x - 4, margin - 4, target_width + 8, target_height + 8)
        pygame.draw.rect(screen, frame_color, frame_rect_left, width=2, border_radius=6)
        pygame.draw.rect(screen, frame_color, frame_rect_right, width=2, border_radius=6)

        buttons_start_y = margin + target_height + 16
        next_left_y = draw_button_info(left_x, buttons_start_y, button_names, display_probs1, display_actions1)
        next_right_y = draw_button_info(right_x, buttons_start_y, button_names, display_probs2, display_actions2)

        info_area_top = max(next_left_y, next_right_y) + 12
        info_area_bottom = window_height - footer_height - margin
        box_padding = 12
        box_gap = 14
        line_gap = 6

        label_surface_left, _ = font_label.render(label1, fgcolor=(200, 200, 200))
        label_surface_right, _ = font_label.render(label2, fgcolor=(200, 200, 200))
        params_surface_left, _ = font_detail.render(params_text1, fgcolor=(180, 220, 255))
        params_surface_right, _ = font_detail.render(params_text2, fgcolor=(180, 220, 255))
        algo_surface_left, _ = font_detail.render(algo_text1, fgcolor=(200, 200, 200))
        algo_surface_right, _ = font_detail.render(algo_text2, fgcolor=(200, 200, 200))
        reward_surface_left, _ = font_detail.render(reward_text1, fgcolor=(220, 220, 120))
        reward_surface_right, _ = font_detail.render(reward_text2, fgcolor=(220, 220, 120))

        heights_to_match = [
            total_surface_height([label_surface_left, params_surface_left], line_gap),
            total_surface_height([label_surface_right, params_surface_right], line_gap),
            total_surface_height([algo_surface_left, reward_surface_left], line_gap),
            total_surface_height([algo_surface_right, reward_surface_right], line_gap),
        ]
        max_content_height = max(heights_to_match) if heights_to_match else 0
        box_height = int(box_padding * 2 + max_content_height)

        available_height = info_area_bottom - info_area_top - box_height
        info_top = info_area_top + max(0, available_height / 2)

        box_width = max(60, (target_width - box_gap) // 2)
        secondary_width = target_width - box_gap - box_width

        model_rect_left = pygame.Rect(left_x, int(info_top), box_width, box_height)
        algo_rect_left = pygame.Rect(left_x + box_width + box_gap, int(info_top), secondary_width, box_height)
        model_rect_right = pygame.Rect(right_x, int(info_top), box_width, box_height)
        algo_rect_right = pygame.Rect(right_x + box_width + box_gap, int(info_top), secondary_width, box_height)

        info_bg_color = (25, 25, 25)
        info_border_color = (100, 100, 100)
        for rect in (model_rect_left, algo_rect_left, model_rect_right, algo_rect_right):
            pygame.draw.rect(screen, info_bg_color, rect, border_radius=8)
            pygame.draw.rect(screen, info_border_color, rect, width=2, border_radius=8)

        draw_centered_stack(screen, model_rect_left, [label_surface_left, params_surface_left], line_gap, box_padding)
        draw_centered_stack(screen, algo_rect_left, [algo_surface_left, reward_surface_left], line_gap, box_padding)
        draw_centered_stack(screen, model_rect_right, [label_surface_right, params_surface_right], line_gap, box_padding)
        draw_centered_stack(screen, algo_rect_right, [algo_surface_right, reward_surface_right], line_gap, box_padding)

        wall_elapsed = time.time() - gameplay_start_time
        frame_elapsed = frame_idx / fps
        elapsed_seconds = max(frame_elapsed, wall_elapsed)
        if duration_seconds > 0 and wall_elapsed >= duration_seconds:
            break

        elapsed_seconds = min(elapsed_seconds, duration_seconds)
        timer_text = f"{elapsed_seconds:.1f}s / {duration_seconds:.1f}s"
        font_timer.render_to(screen, (window_width - margin - 200, window_height - footer_height // 2),
                             timer_text, (180, 180, 180))

        divider_x = window_width // 2
        pygame.draw.line(screen, (80, 80, 80),
                         (divider_x, margin),
                         (divider_x, margin + target_height), width=4)

        pygame.display.flip()

        frame_pixels = pygame.surfarray.array3d(screen)
        frame_pixels = np.transpose(frame_pixels, (1, 0, 2))
        writer.write(cv2.cvtColor(frame_pixels, cv2.COLOR_RGB2BGR))
        clock.tick(fps)

        if done1_flag:
            total_reward1 = 0.0
        if done2_flag:
            total_reward2 = 0.0

    writer.release()
    pygame.quit()

"""Read-only action catalogue with explicitly scoped policy probabilities."""
from dataclasses import dataclass, replace
from functools import lru_cache

import pygame

from nhl94_ai.agents.decisions import ActionCandidate, DecisionSnapshot
from nhl94_ai.game.constants import GameConsts
from nhl94_ai.env.intents import HOCKEY_INTENT_DPAD_ACTIONS


BACKGROUND = (18, 24, 34)
GREEN = (80, 240, 130)
GRAY = (135, 145, 160)
WHITE = (225, 230, 240)
CYAN = (100, 205, 235)
RED = (240, 135, 120)
MODE_GROUPS = {
    'offense': ('Offense', 'Passing', 'One-timers', 'Finishing'),
    'defense': ('Defense',),
    'goalie': ('Goalie', 'Outlets'),
}


@lru_cache(maxsize=1024)
def _text_image(font, text, color, width):
    if width is not None and font.size(text)[0] > width:
        while text and font.size(text + '...')[0] > width:
            text = text[:-1]
        text += '...'
    return font.render(text, True, color)


@dataclass(frozen=True)
class InspectorRow:
    candidate: ActionCandidate
    historical: bool = False


class DecisionInspector:
    ROW_HEIGHT = 23
    GROUP_HEIGHT = 25

    def __init__(self):
        self.reset()

    def reset(self):
        self.current = None
        self.selected_mode = 'offense'
        self.paused = False
        self._snapshots = {}
        self._snapshot_at = {}
        self._evaluations = {}
        self.scroll = 0
        self.max_scroll = 0
        self._follow_id = None
        self.highlighted_ids = ()
        self.row_rects = {}
        self.tab_rects = {}

    @property
    def tabbed(self):
        return self.current is not None and (self.current.active_mode is not None or
            {'Offense', 'Defense', 'Goalie'} <= {candidate.group for candidate in self.current.candidates})

    @property
    def snapshot(self):
        if not self.tabbed or self.current.active_mode is None:
            return self.current
        return self._snapshots.get(self.selected_mode)

    @property
    def historical(self):
        return self.tabbed and self.current.active_mode is not None and self.selected_mode != self.current.active_mode

    @property
    def evaluation(self):
        return self._evaluations.get(self.selected_mode if self.tabbed else None, (None, 0))[0]

    @property
    def evaluation_at(self):
        return self._evaluations.get(self.selected_mode if self.tabbed else None, (None, 0))[1]

    def set_paused(self, paused):
        self.paused = paused
        if not paused and self.current and self.current.active_mode:
            self._select_mode(self.current.active_mode)

    def _select_mode(self, mode):
        if mode != self.selected_mode:
            self.selected_mode = mode
            self.scroll = self.max_scroll = 0
            self._follow_id = None

    def click(self, position):
        if not self.paused or not self.tabbed:
            return
        for mode, bounds in self.tab_rects.items():
            if bounds.collidepoint(position):
                self._select_mode(mode)
                break

    def _candidates(self, snapshot):
        if snapshot is None:
            return ()
        if not self.tabbed:
            return snapshot.candidates
        return tuple(candidate for candidate in snapshot.candidates
                     if candidate.group in MODE_GROUPS[self.selected_mode])

    def update(self, snapshot, playback_frame):
        if snapshot is not None and not isinstance(snapshot, DecisionSnapshot):
            raise TypeError('decision_inspector requires a DecisionSnapshot.')
        if snapshot is None or self.current and snapshot.source != self.current.source:
            self.reset()
        if snapshot and (self.current is None or snapshot.plan_id != self.current.plan_id):
            self._follow_id = snapshot.plan_id
        self.current = snapshot
        if snapshot is None:
            return
        mode = snapshot.active_mode
        old = self._snapshots.get(mode)
        self._snapshots[mode] = snapshot
        if old is None or snapshot.frame != old.frame:
            self._snapshot_at[mode] = playback_frame
        if mode and not self.paused:
            self._select_mode(mode)
        evaluation, _ = self._evaluations.get(mode, (None, 0))
        candidates = (candidate for candidate in snapshot.candidates
                      if mode is None or candidate.group in MODE_GROUPS[mode])
        measured = any(candidate.evaluated_frame is not None or candidate.probability is not None
                       or candidate.raw_probability is not None for candidate in candidates)
        key = snapshot.evaluation_frame if snapshot.evaluation_frame is not None else snapshot.frame
        previous = None if evaluation is None else (
            evaluation.evaluation_frame if evaluation.evaluation_frame is not None else evaluation.frame)
        if measured and (evaluation is None or key != previous):
            self._evaluations[mode] = snapshot, playback_frame

    def rows(self, playback_frame=None):
        snapshot = self.snapshot
        if snapshot is None:
            return ()
        previous = {} if self.evaluation is None else {
            candidate.action_id: candidate for candidate in self.evaluation.candidates}
        rows = []
        for candidate in self._candidates(snapshot):
            old = previous.get(candidate.action_id)
            inherited = (old is not None and candidate.evaluated_frame is None
                          and not candidate.scores and candidate.probability is None
                          and candidate.raw_probability is None
                          and (bool(old.scores) or old.status == 'rejected'))
            historical = self.historical or inherited or (
                candidate.evaluated_frame is not None and candidate.evaluated_frame != snapshot.frame)
            if playback_frame is not None and self.evaluation and playback_frame > self.evaluation_at:
                historical |= (candidate.evaluated_frame is not None
                               or candidate.probability is not None or candidate.raw_probability is not None)
            if inherited:
                reason = candidate.reason
                if candidate.status == 'not-evaluated' and old.reason:
                    reason = f'last: {old.reason}'
                candidate = replace(candidate, scores=old.scores, reason=reason)
            rows.append(InspectorRow(candidate, historical))
        return tuple(rows)

    def scroll_by(self, amount):
        self._follow_id = None
        self.scroll = max(0, min(self.max_scroll, self.scroll + amount))

    def selected_ids(self, actual_action=None, *, human_override=False):
        if self.snapshot is None or self.historical or human_override:
            return ()
        if actual_action is not None and self.current.action and tuple(actual_action) != self.current.action:
            return ()
        return tuple(dict.fromkeys(value for value in (self.current.plan_id, self.current.execution_id)
                                   if value is not None))

    def input_label(self, actual_action=None):
        snapshot = self.snapshot
        if snapshot is None:
            return '--'
        values = snapshot.action if actual_action is None else actual_action
        if snapshot.action_schema == 'FILTERED' and len(values) == GameConsts.INPUT_MAX:
            names = ('UP', 'DOWN', 'LEFT', 'RIGHT', 'B', 'C', 'A', 'START', 'MODE', 'X', 'Y', 'Z')
            return '+'.join(name for name in names if values[getattr(GameConsts, f'INPUT_{name}')]) or 'neutral'
        if snapshot.action_schema == 'HOCKEY_INTENT_DPAD' and len(values) == 6:
            intent = int(values[0])
            name = HOCKEY_INTENT_DPAD_ACTIONS[intent] if 0 <= intent < len(HOCKEY_INTENT_DPAD_ACTIONS) else f'intent {intent}'
            directions = '+'.join(name for name, value in zip(('UP', 'DOWN', 'LEFT', 'RIGHT', 'C'), values[1:]) if value)
            return name + (' | ' + directions if directions else '')
        return str(tuple(round(float(value), 3) for value in values))

    @staticmethod
    def _text(surface, font, text, point, color=WHITE, width=None):
        surface.blit(_text_image(font, text, color, width), point)

    def _draw_summary(self, surface, rect, font, playback_frame, actual_action, human_override):
        snapshot = self.snapshot
        x, y, width = rect.x + 16, rect.y + 46, rect.width - 32
        if snapshot is None:
            self._text(surface, font, f'No recorded {self.selected_mode} decision yet.', (x, y), GRAY)
            return
        by_id = {candidate.action_id: candidate for candidate in snapshot.candidates}
        plan = by_id[snapshot.plan_id].label if snapshot.plan_id else '--'
        execution = by_id[snapshot.execution_id].label if snapshot.execution_id else snapshot.phase or '--'
        override = not self.historical and (human_override or snapshot.plan_id is not None and not self.highlighted_ids)
        source = f'{snapshot.source} | frame {snapshot.frame}'
        if self.historical:
            age = max(0, playback_frame - self._snapshot_at[self.selected_mode])
            source += f' | HISTORICAL: {age} playback frames ago'
        self._text(surface, font, source, (x, y), GRAY if self.historical else CYAN, width)
        self._text(surface, font, f'{"Proposed" if override else "Plan"}: {plan}', (x, y + 23),
                   GRAY if override or self.historical else GREEN, width)
        self._text(surface, font, f'Execution: {"human/outer-controller override" if override else execution}',
                   (x, y + 46), WHITE, width)
        self._text(surface, font, f'Reason: {snapshot.reason or "--"}', (x, y + 69), WHITE, width)
        actual = '--' if snapshot.actual_slot is None or snapshot.actual_slot < 0 else str(snapshot.actual_slot)
        desired = '--' if snapshot.desired_slot is None or snapshot.desired_slot < 0 else str(snapshot.desired_slot)
        target = '--' if snapshot.target is None else f'({snapshot.target[0]:.0f}, {snapshot.target[1]:.0f})'
        self._text(surface, font, f'Player slot: {actual} -> {desired} | Target: {target}',
                   (x, y + 92), WHITE, width)
        label = 'Recorded input' if self.historical else 'Applied input'
        self._text(surface, font, f'{label}: {self.input_label(None if self.historical else actual_action)}',
                   (x, y + 115), GRAY, width)
        age = max(0, playback_frame - self.evaluation_at)
        freshness = f'Last evaluated alternatives: {age} playback frames ago' if self.evaluation else (
            'No evaluated alternatives recorded for this mode.')
        self._text(surface, font, freshness, (x, y + 138), GRAY, width)

    def _draw_tabs(self, surface, rect, font):
        self.tab_rects = {}
        if not self.tabbed:
            return
        width = (rect.width - 40) // 3
        for index, mode in enumerate(MODE_GROUPS):
            bounds = pygame.Rect(rect.x + 16 + index * (width + 4), rect.y + 218, width, 32)
            self.tab_rects[mode] = bounds
            selected = self.selected_mode == mode
            pygame.draw.rect(surface, (38, 68, 85) if selected else (28, 35, 47), bounds, border_radius=4)
            active = self.current.active_mode == mode
            label = mode.title() + ('  [active]' if active else '')
            self._text(surface, font, label, (bounds.x + 10, bounds.y + 6), GREEN if active else WHITE)
            if selected:
                pygame.draw.line(surface, CYAN, (bounds.left + 2, bounds.bottom - 1),
                                 (bounds.right - 2, bounds.bottom - 1), 2)
        if self.selected_mode == 'goalie' and self.current.goalie_policy is not None:
            context = f'Goalie policy: {self.current.goalie_policy} (read-only)'
        else:
            context = 'Click a tab to inspect its latest decision.' if self.paused else 'Following active mode. Pause to inspect other tabs.'
        self._text(surface, font, context, (rect.x + 16, rect.y + 263), GRAY, rect.width - 32)

    def draw(self, surface, rect, font, big_font, playback_frame, *, actual_action=None,
             human_override=False, mouse_position=None):
        pygame.draw.rect(surface, BACKGROUND, rect)
        x, y = rect.x + 16, rect.y + 12
        self._text(surface, big_font, 'Decision inspector', (x, y))
        self.row_rects = {}
        self.tab_rects = {}
        self.highlighted_ids = self.selected_ids(actual_action, human_override=human_override)
        if self.current is None:
            self._text(surface, font, 'No tactical decision diagnostics from this agent.', (x, y + 40), CYAN)
            self._text(surface, font, 'Button/target outputs are not tactical probabilities.', (x, y + 66), GRAY)
            return
        self._draw_summary(surface, rect, font, playback_frame, actual_action, human_override)
        self._draw_tabs(surface, rect, font)
        snapshot = self.snapshot
        if snapshot is None:
            self.scroll = self.max_scroll = 0
            return
        by_id = {candidate.action_id: candidate for candidate in snapshot.candidates}
        rows = self.rows(playback_frame)
        probabilities = any(row.candidate.probability is not None or row.candidate.raw_probability is not None
                            for row in rows)
        columns = (x, rect.x + 278, rect.x + 341, rect.x + 431, rect.x + (536 if probabilities else 431))
        headers = [('Action / target', columns[0]), ('Score', columns[1]), ('Kind', columns[2])]
        if probabilities:
            headers.append(('P eff/raw', columns[3]))
        headers.append(('Status / reason', columns[4]))
        header_y = rect.y + (296 if self.tabbed else 218)
        for name, left in headers:
            self._text(surface, font, name, (left, header_y), CYAN)
        clip = pygame.Rect(rect.x + 8, header_y + 26, rect.width - 16, rect.bottom - 84 - header_y - 26)
        offsets, offset, group = [], 0, None
        for row in rows:
            if row.candidate.group != group:
                offset += self.GROUP_HEIGHT
                group = row.candidate.group
            offsets.append(offset)
            offset += self.ROW_HEIGHT
        self.max_scroll = max(0, offset - clip.height)
        if self._follow_id is not None:
            index = next((i for i, row in enumerate(rows) if row.candidate.action_id == self._follow_id), None)
            if index is not None and not self.scroll <= offsets[index] < self.scroll + clip.height:
                self.scroll = max(0, min(self.max_scroll, offsets[index] - clip.height // 2))
            self._follow_id = None
        self.scroll = min(self.scroll, self.max_scroll)
        previous_clip = surface.get_clip()
        surface.set_clip(clip.clip(previous_clip))
        group, hovered = None, None
        for row, offset in zip(rows, offsets):
            candidate = row.candidate
            top = clip.y + offset - self.scroll
            if candidate.group != group:
                self._text(surface, font, candidate.group, (x, top - self.GROUP_HEIGHT + 4), CYAN)
                group = candidate.group
            bounds = pygame.Rect(x - 4, top, rect.width - 24, self.ROW_HEIGHT)
            if not bounds.colliderect(clip):
                continue
            self.row_rects[candidate.action_id] = bounds.clip(clip)
            chosen = candidate.action_id in self.highlighted_ids
            if chosen:
                pygame.draw.rect(surface, (24, 65, 42), bounds)
            color = GRAY if self.historical else GREEN if chosen else GRAY if candidate.status in ('disabled', 'unavailable') else (
                RED if candidate.status == 'rejected' else WHITE)
            self._text(surface, font, candidate.label, (columns[0], top + 2), color, 250)
            score = candidate.scores[0] if candidate.scores else None
            metric_color = GRAY if row.historical else color
            self._text(surface, font, '--' if score is None else f'{score.value:.1f}',
                       (columns[1], top + 2), metric_color, 60)
            self._text(surface, font, '--' if score is None else score.kind,
                       (columns[2], top + 2), metric_color, 85)
            if probabilities:
                probability = '--' if candidate.probability is None else f'{candidate.probability:.0%}'
                if candidate.raw_probability is not None:
                    probability += f'/{candidate.raw_probability:.0%}'
                self._text(surface, font, probability, (columns[3], top + 2), metric_color, 100)
            self._text(surface, font, candidate.reason or candidate.status, (columns[4], top + 2),
                       GRAY if row.historical else color, rect.right - columns[4] - 16)
            if mouse_position is not None and bounds.clip(clip).collidepoint(mouse_position):
                hovered = candidate
        surface.set_clip(previous_clip)
        if self.max_scroll:
            track = pygame.Rect(rect.right - 6, clip.top, 3, clip.height)
            pygame.draw.rect(surface, (45, 55, 70), track)
            thumb_height = max(24, round(clip.height * clip.height / (clip.height + self.max_scroll)))
            thumb = pygame.Rect(track.x, track.y + round(
                self.scroll / self.max_scroll * (track.height - thumb_height)), track.width, thumb_height)
            pygame.draw.rect(surface, GRAY, thumb)
        footer = rect.bottom - 76
        candidate = hovered or by_id.get(snapshot.plan_id)
        if candidate:
            detail = ' | '.join(f'{score.kind}: {score.value:.1f}' for score in candidate.scores)
            self._text(surface, font, candidate.reason or detail or candidate.label, (x, footer), WHITE, rect.width - 32)
            self._text(surface, font, f'Source/head: {candidate.source or snapshot.source}'
                       + (f' | P scope: {candidate.probability_scope}' if candidate.probability_scope else ''),
                       (x, footer + 23), CYAN, rect.width - 32)
        self._text(surface, font, 'Scores are heuristics; P is policy selection, not success.' if probabilities else
                   'Scores are heuristics. Inputs are requests, not confirmed ROM actions.',
                   (x, rect.bottom - 28), GRAY, rect.width - 32)

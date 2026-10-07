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
        self.evaluation = None
        self.evaluation_at = 0
        self.scroll = 0
        self.max_scroll = 0
        self._follow_id = None
        self.highlighted_ids = ()
        self.row_rects = {}

    def update(self, snapshot, playback_frame):
        if snapshot is not None and not isinstance(snapshot, DecisionSnapshot):
            raise TypeError('decision_inspector requires a DecisionSnapshot.')
        if snapshot and (self.current is None or snapshot.plan_id != self.current.plan_id):
            self._follow_id = snapshot.plan_id
        self.current = snapshot
        if snapshot is None:
            return
        if self.evaluation and self.evaluation.source != snapshot.source:
            self.evaluation = None
        measured = any(candidate.evaluated_frame is not None or candidate.probability is not None
                       or candidate.raw_probability is not None for candidate in snapshot.candidates)
        key = snapshot.evaluation_frame if snapshot.evaluation_frame is not None else snapshot.frame
        previous = None if self.evaluation is None else (
            self.evaluation.evaluation_frame if self.evaluation.evaluation_frame is not None else self.evaluation.frame)
        if measured and (self.evaluation is None or key != previous):
            self.evaluation, self.evaluation_at = snapshot, playback_frame

    def rows(self, playback_frame=None):
        if self.current is None:
            return ()
        previous = {} if self.evaluation is None else {
            candidate.action_id: candidate for candidate in self.evaluation.candidates}
        rows = []
        for candidate in self.current.candidates:
            old = previous.get(candidate.action_id)
            inherited = (old is not None and candidate.evaluated_frame is None
                          and not candidate.scores and candidate.probability is None
                          and candidate.raw_probability is None
                          and (bool(old.scores) or old.status == 'rejected'))
            historical = inherited or (
                candidate.evaluated_frame is not None and candidate.evaluated_frame != self.current.frame)
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
        if self.current is None or human_override:
            return ()
        if actual_action is not None and self.current.action and tuple(actual_action) != self.current.action:
            return ()
        return tuple(dict.fromkeys(value for value in (self.current.plan_id, self.current.execution_id)
                                   if value is not None))

    def input_label(self, actual_action=None):
        if self.current is None:
            return '--'
        values = self.current.action if actual_action is None else actual_action
        if self.current.action_schema == 'FILTERED' and len(values) == GameConsts.INPUT_MAX:
            names = ('UP', 'DOWN', 'LEFT', 'RIGHT', 'B', 'C', 'A', 'START', 'MODE', 'X', 'Y', 'Z')
            return '+'.join(name for name in names if values[getattr(GameConsts, f'INPUT_{name}')]) or 'neutral'
        if self.current.action_schema == 'HOCKEY_INTENT_DPAD' and len(values) == 6:
            intent = int(values[0])
            name = HOCKEY_INTENT_DPAD_ACTIONS[intent] if 0 <= intent < len(HOCKEY_INTENT_DPAD_ACTIONS) else f'intent {intent}'
            directions = '+'.join(name for name, value in zip(('UP', 'DOWN', 'LEFT', 'RIGHT', 'C'), values[1:]) if value)
            return name + (' | ' + directions if directions else '')
        return str(tuple(round(float(value), 3) for value in values))

    @staticmethod
    def _text(surface, font, text, point, color=WHITE, width=None):
        surface.blit(_text_image(font, text, color, width), point)

    def draw(self, surface, rect, font, big_font, playback_frame, *, actual_action=None,
             human_override=False, mouse_position=None):
        pygame.draw.rect(surface, BACKGROUND, rect)
        x, y = rect.x + 16, rect.y + 12
        self._text(surface, big_font, 'Decision inspector', (x, y))
        self.row_rects = {}
        self.highlighted_ids = self.selected_ids(actual_action, human_override=human_override)
        if self.current is None:
            self._text(surface, font, 'No tactical decision diagnostics from this agent.', (x, y + 40), CYAN)
            self._text(surface, font, 'Button/target outputs are not tactical probabilities.', (x, y + 66), GRAY)
            return
        snapshot = self.current
        by_id = {candidate.action_id: candidate for candidate in snapshot.candidates}
        plan = by_id[snapshot.plan_id].label if snapshot.plan_id else '--'
        execution = by_id[snapshot.execution_id].label if snapshot.execution_id else snapshot.phase or '--'
        override = human_override or snapshot.plan_id is not None and not self.highlighted_ids
        self._text(surface, font, f'Source: {snapshot.source} | producer frame: {snapshot.frame}', (x, y + 34), CYAN)
        self._text(surface, font, f'{"Proposed" if override else "Plan"}: {plan}', (x, y + 57),
                   GRAY if override else GREEN, rect.width - 32)
        self._text(surface, font, f'Execution: {"human/outer-controller override" if override else execution}',
                   (x, y + 80), WHITE, rect.width - 32)
        mode = f'Goalie policy: {snapshot.goalie_policy} (read-only)' if snapshot.goalie_policy is not None else (
            'Neural P describes selection, not success.')
        age = max(0, playback_frame - self.evaluation_at)
        self._text(surface, font, mode, (x, y + 103), GRAY)
        self._text(surface, font, f'Last scored evaluation: {age} playback frames ago' if self.evaluation else (
            'No scored alternatives in this snapshot.'), (x, y + 126), GRAY)
        self._text(surface, font, f'Applied input: {self.input_label(actual_action)} (requests, not ROM completion)',
                   (x, y + 148), GRAY, rect.width - 32)
        columns = (x, rect.x + 278, rect.x + 341, rect.x + 431, rect.x + 536)
        for name, left in zip(('Action / target', 'Score', 'Kind', 'P eff/raw', 'Status / reason'), columns):
            self._text(surface, font, name, (left, rect.y + 184), CYAN)
        clip = pygame.Rect(rect.x + 8, rect.y + 210, rect.width - 16, rect.height - 294)
        rows = self.rows(playback_frame)
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
            color = GREEN if chosen else GRAY if candidate.status in ('disabled', 'unavailable') else (
                RED if candidate.status == 'rejected' else WHITE)
            self._text(surface, font, candidate.label, (columns[0], top + 2), color, 250)
            score = candidate.scores[0] if candidate.scores else None
            metric_color = GRAY if row.historical else color
            self._text(surface, font, '--' if score is None else f'{score.value:.1f}',
                       (columns[1], top + 2), metric_color, 60)
            self._text(surface, font, '--' if score is None else score.kind,
                       (columns[2], top + 2), metric_color, 85)
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
        self._text(surface, font, 'Scores are heuristics; P is policy selection, not success.',
                   (x, rect.bottom - 28), GRAY, rect.width - 32)

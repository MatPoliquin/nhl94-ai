from __future__ import annotations
import sys
import threading
import time
from typing import Dict, List, Optional
SPINNER_FRAMES = ('|', '/', '-', '\\')
ACTIVE_PHASE_COLOR = '\033[1;36m'
RESET_COLOR = '\033[0m'


class CurriculumTerminalDisplay:
    def __init__(self, phase_names: List[str]) -> None:
        self._tty = sys.stdout.isatty()
        self._phase_rows = [
            {
                "name": name,
                "steps": None,
                "reward": None,
                "best_reward": None,
            }
            for name in phase_names
        ]
        self._active_phase_index: Optional[int] = None
        self._spinner_index = 0
        self._games: Optional[int] = None
        self._team1 = {"goals": None, "shots": None, "one_timers": None, "cross_checks": None}
        self._team2 = {"goals": None, "shots": None, "one_timers": None, "cross_checks": None}
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if not self._tty or self._thread is not None:
            return

        self.render()
        self._thread = threading.Thread(target=self._spin, name="curriculum-terminal-display", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        self.render()

    def activate_phase(self, phase_index: int) -> None:
        with self._lock:
            self._active_phase_index = phase_index
            self._spinner_index = 0
            row = self._phase_rows[phase_index]
            if row["steps"] is None:
                row["steps"] = 0
        self.render()

    def complete_phase(self, phase_index: int) -> None:
        with self._lock:
            if self._active_phase_index == phase_index:
                self._active_phase_index = None
        self.render()

    def update_phase(
        self,
        phase_index: int,
        *,
        steps: Optional[int] = None,
        reward: Optional[float] = None,
        best_reward: Optional[float] = None,
    ) -> None:
        with self._lock:
            row = self._phase_rows[phase_index]
            if steps is not None:
                row["steps"] = int(steps)
            if reward is not None:
                row["reward"] = float(reward)
            if best_reward is not None:
                row["best_reward"] = float(best_reward)
        self.render()

    def update_test_totals(self, games: int, team1_stats, team2_stats) -> None:
        with self._lock:
            self._games = int(games)
            self._team1 = {
                "goals": int(getattr(team1_stats, "goals", 0) or 0),
                "shots": int(getattr(team1_stats, "shots", 0) or 0),
                "one_timers": int(getattr(team1_stats, "one_timers", 0) or 0),
                "cross_checks": int(getattr(team1_stats, "cross_checks", 0) or 0),
            }
            self._team2 = {
                "goals": int(getattr(team2_stats, "goals", 0) or 0),
                "shots": int(getattr(team2_stats, "shots", 0) or 0),
                "one_timers": int(getattr(team2_stats, "one_timers", 0) or 0),
                "cross_checks": int(getattr(team2_stats, "cross_checks", 0) or 0),
            }
        self.render()

    def clear_test_totals(self) -> None:
        with self._lock:
            self._games = None
            self._team1 = {"goals": None, "shots": None, "one_timers": None, "cross_checks": None}
            self._team2 = {"goals": None, "shots": None, "one_timers": None, "cross_checks": None}
        self.render()

    def render(self) -> None:
        with self._lock:
            rows = [dict(row) for row in self._phase_rows]
            active_phase_index = self._active_phase_index
            spinner_index = self._spinner_index
            games = self._games
            team1 = dict(self._team1)
            team2 = dict(self._team2)

        lines: List[str] = []
        for index, row in enumerate(rows, start=1):
            prefix = f"{index}."
            phase_name = row["name"]
            if active_phase_index == index - 1:
                spinner = SPINNER_FRAMES[spinner_index]
                prefix = f"{ACTIVE_PHASE_COLOR}{spinner}{RESET_COLOR}" if self._tty else spinner
                phase_name = f"{ACTIVE_PHASE_COLOR}{phase_name}{RESET_COLOR}" if self._tty else phase_name

            lines.append(
                f"{prefix} {phase_name}: steps={self._format_steps(row['steps'])} "
                f"reward={self._format_reward(row['reward'])} best_reward={self._format_reward(row['best_reward'])}"
            )

        lines.extend(
            [
                "",
                "===============",
                "TEST",
                "===============",
                f"number of games: {games if games is not None else 'N/A'}",
                self._format_team_line("team1", team1),
                self._format_team_line("team2", team2),
            ]
        )

        output = "\n".join(lines)
        if self._tty:
            sys.stdout.write("\033[2J\033[H")
            sys.stdout.write(output + "\n")
            sys.stdout.flush()
            return

        print(output, flush=True)

    def _spin(self) -> None:
        while not self._stop_event.wait(0.2):
            with self._lock:
                if self._active_phase_index is None:
                    continue
                self._spinner_index = (self._spinner_index + 1) % len(SPINNER_FRAMES)
            self.render()

    @staticmethod
    def _format_steps(value: Optional[int]) -> str:
        return "--" if value is None else str(int(value))

    @staticmethod
    def _format_reward(value: Optional[float]) -> str:
        return "--" if value is None else f"{value:.2f}"

    @staticmethod
    def _format_team_line(label: str, stats: Dict[str, Optional[int]]) -> str:
        def value_or_dash(value: Optional[int]) -> str:
            return "--" if value is None else str(int(value))

        return (
            f"{label}: goals={value_or_dash(stats['goals'])} "
            f"shots={value_or_dash(stats['shots'])} "
            f"one-timers={value_or_dash(stats['one_timers'])} "
            f"cross checks={value_or_dash(stats['cross_checks'])}"
        )

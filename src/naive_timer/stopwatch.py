"""Core stopwatch/timer model.

Deliberately UI-free and dependency-free so it can be unit-tested headlessly
(no display, no Qt). The GUI layer drives this model and renders its state.

Time is injected via a ``clock`` callable (defaults to ``time.monotonic``) so
tests can advance time deterministically instead of sleeping.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, List


class State(Enum):
    STOPPED = "stopped"
    RUNNING = "running"
    PAUSED = "paused"


@dataclass
class Stopwatch:
    """A start/pause/reset stopwatch with lap support.

    Elapsed time accumulates while RUNNING. Pausing banks the elapsed time;
    starting again resumes from there. Reset returns to a clean STOPPED state.
    """

    clock: Callable[[], float] = time.monotonic
    _state: State = field(default=State.STOPPED, init=False)
    _accumulated: float = field(default=0.0, init=False)
    _started_at: float = field(default=0.0, init=False)
    _laps: List[float] = field(default_factory=list, init=False)

    @property
    def state(self) -> State:
        return self._state

    @property
    def is_running(self) -> bool:
        return self._state is State.RUNNING

    def elapsed(self) -> float:
        """Total elapsed seconds, live while running."""
        if self._state is State.RUNNING:
            return self._accumulated + (self.clock() - self._started_at)
        return self._accumulated

    def start(self) -> None:
        """Start or resume. No-op if already running."""
        if self._state is State.RUNNING:
            return
        self._started_at = self.clock()
        self._state = State.RUNNING

    def pause(self) -> None:
        """Pause and bank elapsed time. No-op unless running."""
        if self._state is not State.RUNNING:
            return
        self._accumulated += self.clock() - self._started_at
        self._state = State.PAUSED

    def toggle(self) -> None:
        """Convenience: start if not running, otherwise pause."""
        if self._state is State.RUNNING:
            self.pause()
        else:
            self.start()

    def reset(self) -> None:
        """Return to a clean stopped state, clearing laps."""
        self._state = State.STOPPED
        self._accumulated = 0.0
        self._started_at = 0.0
        self._laps.clear()

    def lap(self) -> float:
        """Record and return the current elapsed time as a lap."""
        marker = self.elapsed()
        self._laps.append(marker)
        return marker

    @property
    def laps(self) -> List[float]:
        return list(self._laps)


def format_elapsed(seconds: float) -> str:
    """Format seconds as HH:MM:SS.cs (centiseconds)."""
    if seconds < 0:
        seconds = 0.0
    total_cs = int(round(seconds * 100))
    cs = total_cs % 100
    total_s = total_cs // 100
    s = total_s % 60
    m = (total_s // 60) % 60
    h = total_s // 3600
    return f"{h:02d}:{m:02d}:{s:02d}.{cs:02d}"

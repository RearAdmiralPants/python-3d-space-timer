"""Countdown / alarm model and input parsing.

Like ``stopwatch``, this is UI-free and dependency-free so it can be fully
unit-tested without a display. The model counts a fixed number of seconds down
to zero; whether that duration came from "12 minutes from now" or "alarm at
02:54" is a parsing concern handled by :func:`parse_duration` /
:func:`parse_alarm`, not the model.

Time advances via an injectable ``clock`` (defaults to ``time.monotonic``) so
tests can move time deterministically.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Callable, Optional


class Phase(Enum):
    IDLE = "idle"          # configured but not started
    RUNNING = "running"
    PAUSED = "paused"
    FINISHED = "finished"  # reached zero; alert may be active


@dataclass
class Countdown:
    """Counts ``total`` seconds down to zero, then enters an alert window.

    While RUNNING, ``remaining()`` shrinks. On reaching zero it transitions to
    FINISHED and ``alert_active()`` stays true for ``alert_duration`` seconds
    (or until :meth:`dismiss`).
    """

    total: float = 0.0
    alert_duration: float = 120.0
    clock: Callable[[], float] = time.monotonic

    _phase: Phase = field(default=Phase.IDLE, init=False)
    _accumulated: float = field(default=0.0, init=False)
    _started_at: float = field(default=0.0, init=False)
    _finished_at: float = field(default=0.0, init=False)
    _dismissed: bool = field(default=False, init=False)

    # -- configuration -----------------------------------------------------
    def configure(self, total_seconds: float) -> None:
        """Set the countdown length and return to a clean IDLE state."""
        if total_seconds <= 0:
            raise ValueError("countdown duration must be positive")
        self.total = float(total_seconds)
        self.reset()

    # -- controls ----------------------------------------------------------
    def start(self) -> None:
        """Start or resume. No-op when running, finished, or nothing left."""
        if self._phase in (Phase.RUNNING, Phase.FINISHED):
            return
        if self.total - self._accumulated <= 0:
            return
        self._started_at = self.clock()
        self._phase = Phase.RUNNING

    def pause(self) -> None:
        """Pause and bank elapsed time. No-op unless running."""
        if self._phase is not Phase.RUNNING:
            return
        self._accumulated += self.clock() - self._started_at
        self._phase = Phase.PAUSED

    def toggle(self) -> None:
        if self._phase is Phase.RUNNING:
            self.pause()
        else:
            self.start()

    def reset(self) -> None:
        """Return to IDLE, keeping ``total`` so it can be restarted."""
        self._phase = Phase.IDLE
        self._accumulated = 0.0
        self._started_at = 0.0
        self._finished_at = 0.0
        self._dismissed = False

    def dismiss(self) -> None:
        """Silence/clear an active alert."""
        self._dismissed = True

    # -- queries -----------------------------------------------------------
    @property
    def phase(self) -> Phase:
        self._check_finish()
        return self._phase

    @property
    def is_running(self) -> bool:
        return self.phase is Phase.RUNNING

    @property
    def is_finished(self) -> bool:
        return self.phase is Phase.FINISHED

    def elapsed(self) -> float:
        if self._phase is Phase.RUNNING:
            return self._accumulated + (self.clock() - self._started_at)
        return self._accumulated

    def remaining(self) -> float:
        self._check_finish()
        return max(0.0, self.total - self.elapsed())

    def progress(self) -> float:
        """Fraction elapsed in ``[0, 1]`` (0 when total is 0)."""
        if self.total <= 0:
            return 0.0
        return min(1.0, self.elapsed() / self.total)

    def alert_active(self) -> bool:
        """True while the finished alert should still be signalling."""
        if self.phase is not Phase.FINISHED or self._dismissed:
            return False
        return (self.clock() - self._finished_at) < self.alert_duration

    # -- internals ---------------------------------------------------------
    def _check_finish(self) -> None:
        if self._phase is Phase.RUNNING and self.total - self.elapsed() <= 0:
            # Record the exact monotonic instant zero was crossed so the alert
            # window is measured from then, not from when we noticed.
            self._finished_at = self._started_at + (self.total - self._accumulated)
            self._accumulated = self.total
            self._phase = Phase.FINISHED


# ---------------------------------------------------------------------------
# Input parsing
# ---------------------------------------------------------------------------

_UNIT_RE = re.compile(r"(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>[hms])", re.IGNORECASE)
_UNIT_SECONDS = {"h": 3600.0, "m": 60.0, "s": 1.0}


def parse_duration(text: str) -> float:
    """Parse a timespan into seconds.

    Accepts:
      * unit form: ``"90s"``, ``"12m"``, ``"1h"``, ``"1h30m"``, ``"1h 30m 15s"``
      * colon form: ``"12:34"`` (mm:ss), ``"1:02:03"`` (hh:mm:ss)
      * a bare number: ``"12"`` → 12 minutes (timer convention)

    Raises ``ValueError`` on empty, malformed, or non-positive input.
    """
    raw = text.strip().lower()
    if not raw:
        raise ValueError("empty duration")

    if ":" in raw:
        parts = raw.split(":")
        if len(parts) not in (2, 3) or not all(p.strip() for p in parts):
            raise ValueError(f"bad clock-format duration: {text!r}")
        try:
            nums = [float(p) for p in parts]
        except ValueError:
            raise ValueError(f"bad clock-format duration: {text!r}")
        if len(parts) == 2:
            m, s = nums
            seconds = m * 60 + s
        else:
            h, m, s = nums
            seconds = h * 3600 + m * 60 + s
    elif _UNIT_RE.search(raw):
        # Sum all unit tokens; reject stray characters outside the tokens.
        if _UNIT_RE.sub("", raw).strip():
            raise ValueError(f"unrecognized duration: {text!r}")
        seconds = sum(
            float(m.group("value")) * _UNIT_SECONDS[m.group("unit").lower()]
            for m in _UNIT_RE.finditer(raw)
        )
    else:
        try:
            seconds = float(raw) * 60.0  # bare number = minutes
        except ValueError:
            raise ValueError(f"unrecognized duration: {text!r}")

    if seconds <= 0:
        raise ValueError("duration must be positive")
    return seconds


_ALARM_RE = re.compile(
    r"^(?P<h>\d{1,2}):(?P<m>\d{2})(?::(?P<s>\d{2}))?\s*(?P<ampm>am|pm)?$",
    re.IGNORECASE,
)


def parse_alarm(text: str, now: Optional[datetime] = None) -> float:
    """Parse a future clock time and return seconds from ``now`` until it.

    Accepts ``"HH:MM"`` / ``"H:MM"`` (24h), optional ``":SS"``, and an optional
    ``am``/``pm`` suffix. If the resulting time is at or before ``now``, it
    rolls forward to the next day. ``now`` defaults to the current local time.
    """
    if now is None:
        now = datetime.now()

    m = _ALARM_RE.match(text.strip())
    if not m:
        raise ValueError(f"unrecognized alarm time: {text!r}")

    hour = int(m.group("h"))
    minute = int(m.group("m"))
    second = int(m.group("s") or 0)
    ampm = m.group("ampm")

    if ampm:
        if not 1 <= hour <= 12:
            raise ValueError(f"bad 12-hour time: {text!r}")
        if ampm.lower() == "pm" and hour != 12:
            hour += 12
        elif ampm.lower() == "am" and hour == 12:
            hour = 0
    if not (0 <= hour <= 23 and 0 <= minute <= 59 and 0 <= second <= 59):
        raise ValueError(f"time out of range: {text!r}")

    target = now.replace(hour=hour, minute=minute, second=second, microsecond=0)
    if target <= now:
        target += timedelta(days=1)
    return (target - now).total_seconds()

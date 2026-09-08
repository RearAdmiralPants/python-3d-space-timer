"""naive-linux-timer-gui: a stopwatch + countdown timer with naive animation."""

from .countdown import Countdown, Phase, parse_alarm, parse_duration
from .stopwatch import State, Stopwatch, format_elapsed

__all__ = [
    "State",
    "Stopwatch",
    "format_elapsed",
    "Countdown",
    "Phase",
    "parse_duration",
    "parse_alarm",
]
__version__ = "0.2.0"

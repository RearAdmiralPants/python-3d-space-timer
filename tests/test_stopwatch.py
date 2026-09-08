"""Headless tests for the stopwatch model.

Uses a fake clock so time advances deterministically — no sleeping, and no
display required. Runs with either pytest or plain ``unittest``.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from naive_timer.stopwatch import State, Stopwatch, format_elapsed


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class StopwatchTest(unittest.TestCase):
    def setUp(self) -> None:
        self.clock = FakeClock()
        self.sw = Stopwatch(clock=self.clock)

    def test_starts_stopped_at_zero(self) -> None:
        self.assertEqual(self.sw.state, State.STOPPED)
        self.assertEqual(self.sw.elapsed(), 0.0)

    def test_running_accumulates_time(self) -> None:
        self.sw.start()
        self.clock.advance(5.0)
        self.assertTrue(self.sw.is_running)
        self.assertAlmostEqual(self.sw.elapsed(), 5.0)

    def test_pause_banks_and_freezes(self) -> None:
        self.sw.start()
        self.clock.advance(3.0)
        self.sw.pause()
        self.assertEqual(self.sw.state, State.PAUSED)
        self.clock.advance(10.0)  # time passes while paused
        self.assertAlmostEqual(self.sw.elapsed(), 3.0)

    def test_resume_continues_from_banked(self) -> None:
        self.sw.start()
        self.clock.advance(3.0)
        self.sw.pause()
        self.sw.start()
        self.clock.advance(2.0)
        self.assertAlmostEqual(self.sw.elapsed(), 5.0)

    def test_toggle(self) -> None:
        self.sw.toggle()
        self.assertTrue(self.sw.is_running)
        self.sw.toggle()
        self.assertEqual(self.sw.state, State.PAUSED)

    def test_double_start_is_noop(self) -> None:
        self.sw.start()
        self.clock.advance(1.0)
        self.sw.start()  # should not reset the started_at marker
        self.clock.advance(1.0)
        self.assertAlmostEqual(self.sw.elapsed(), 2.0)

    def test_reset_clears_everything(self) -> None:
        self.sw.start()
        self.clock.advance(4.0)
        self.sw.lap()
        self.sw.reset()
        self.assertEqual(self.sw.state, State.STOPPED)
        self.assertEqual(self.sw.elapsed(), 0.0)
        self.assertEqual(self.sw.laps, [])

    def test_laps_record_elapsed(self) -> None:
        self.sw.start()
        self.clock.advance(2.0)
        self.assertAlmostEqual(self.sw.lap(), 2.0)
        self.clock.advance(3.0)
        self.assertAlmostEqual(self.sw.lap(), 5.0)
        self.assertEqual(len(self.sw.laps), 2)


class FormatTest(unittest.TestCase):
    def test_zero(self) -> None:
        self.assertEqual(format_elapsed(0), "00:00:00.00")

    def test_subsecond(self) -> None:
        self.assertEqual(format_elapsed(1.23), "00:00:01.23")

    def test_minutes_and_hours(self) -> None:
        self.assertEqual(format_elapsed(3661.5), "01:01:01.50")

    def test_negative_clamps(self) -> None:
        self.assertEqual(format_elapsed(-5), "00:00:00.00")


if __name__ == "__main__":
    unittest.main()

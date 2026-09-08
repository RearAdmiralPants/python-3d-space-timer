"""Headless tests for the countdown model and input parsing."""

import os
import sys
import unittest
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from naive_timer.countdown import (
    Countdown,
    Phase,
    parse_alarm,
    parse_duration,
)


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class CountdownTest(unittest.TestCase):
    def setUp(self) -> None:
        self.clock = FakeClock()
        self.cd = Countdown(clock=self.clock, alert_duration=120.0)
        self.cd.configure(10.0)

    def test_configure_sets_idle(self) -> None:
        self.assertEqual(self.cd.phase, Phase.IDLE)
        self.assertAlmostEqual(self.cd.remaining(), 10.0)

    def test_configure_rejects_nonpositive(self) -> None:
        with self.assertRaises(ValueError):
            self.cd.configure(0)
        with self.assertRaises(ValueError):
            self.cd.configure(-5)

    def test_counts_down(self) -> None:
        self.cd.start()
        self.clock.advance(4.0)
        self.assertTrue(self.cd.is_running)
        self.assertAlmostEqual(self.cd.remaining(), 6.0)

    def test_pause_freezes_remaining(self) -> None:
        self.cd.start()
        self.clock.advance(3.0)
        self.cd.pause()
        self.clock.advance(100.0)
        self.assertEqual(self.cd.phase, Phase.PAUSED)
        self.assertAlmostEqual(self.cd.remaining(), 7.0)

    def test_resume(self) -> None:
        self.cd.start()
        self.clock.advance(3.0)
        self.cd.pause()
        self.cd.start()
        self.clock.advance(2.0)
        self.assertAlmostEqual(self.cd.remaining(), 5.0)

    def test_reaches_zero_and_finishes(self) -> None:
        self.cd.start()
        self.clock.advance(10.0)
        self.assertTrue(self.cd.is_finished)
        self.assertEqual(self.cd.remaining(), 0.0)

    def test_remaining_never_negative(self) -> None:
        self.cd.start()
        self.clock.advance(999.0)
        self.assertEqual(self.cd.remaining(), 0.0)

    def test_alert_active_window(self) -> None:
        self.cd.start()
        self.clock.advance(10.0)  # finishes exactly at zero
        self.assertTrue(self.cd.alert_active())
        self.clock.advance(119.0)
        self.assertTrue(self.cd.alert_active())
        self.clock.advance(2.0)  # past the 120s window
        self.assertFalse(self.cd.alert_active())

    def test_alert_measured_from_zero_crossing_not_poll(self) -> None:
        # Overshoot the finish by a lot before anyone reads state; the alert
        # window must still be measured from the true zero-crossing instant.
        self.cd.start()
        self.clock.advance(60.0)  # 50s past zero before first observation
        self.assertTrue(self.cd.is_finished)
        # 50s already elapsed in the alert window, 70s should remain.
        self.assertTrue(self.cd.alert_active())
        self.clock.advance(71.0)
        self.assertFalse(self.cd.alert_active())

    def test_dismiss_silences_alert(self) -> None:
        self.cd.start()
        self.clock.advance(10.0)
        self.assertTrue(self.cd.alert_active())
        self.cd.dismiss()
        self.assertFalse(self.cd.alert_active())

    def test_reset_returns_to_idle_keeping_total(self) -> None:
        self.cd.start()
        self.clock.advance(5.0)
        self.cd.reset()
        self.assertEqual(self.cd.phase, Phase.IDLE)
        self.assertAlmostEqual(self.cd.remaining(), 10.0)

    def test_start_noop_when_finished(self) -> None:
        self.cd.start()
        self.clock.advance(10.0)
        self.cd.start()  # should not restart
        self.assertTrue(self.cd.is_finished)

    def test_progress(self) -> None:
        self.cd.start()
        self.clock.advance(5.0)
        self.assertAlmostEqual(self.cd.progress(), 0.5)


class ParseDurationTest(unittest.TestCase):
    def test_units(self) -> None:
        self.assertEqual(parse_duration("90s"), 90)
        self.assertEqual(parse_duration("12m"), 720)
        self.assertEqual(parse_duration("1h"), 3600)
        self.assertEqual(parse_duration("1h30m"), 5400)
        self.assertEqual(parse_duration("1h 30m 15s"), 5415)

    def test_colon_forms(self) -> None:
        self.assertEqual(parse_duration("12:34"), 12 * 60 + 34)
        self.assertEqual(parse_duration("1:02:03"), 3723)

    def test_bare_number_is_minutes(self) -> None:
        self.assertEqual(parse_duration("12"), 720)

    def test_case_and_whitespace(self) -> None:
        self.assertEqual(parse_duration("  1H30M  "), 5400)

    def test_rejects_garbage(self) -> None:
        for bad in ["", "   ", "abc", "12x", "1:2:3:4", ":", "1:"]:
            with self.assertRaises(ValueError, msg=bad):
                parse_duration(bad)

    def test_rejects_nonpositive(self) -> None:
        with self.assertRaises(ValueError):
            parse_duration("0")
        with self.assertRaises(ValueError):
            parse_duration("0:00")


class ParseAlarmTest(unittest.TestCase):
    def setUp(self) -> None:
        self.now = datetime(2026, 7, 9, 2, 0, 0)  # 02:00

    def test_later_today(self) -> None:
        secs = parse_alarm("02:54", now=self.now)
        self.assertAlmostEqual(secs, 54 * 60)

    def test_rolls_to_tomorrow_when_past(self) -> None:
        secs = parse_alarm("01:00", now=self.now)  # already passed
        self.assertAlmostEqual(secs, 23 * 3600)

    def test_equal_time_rolls_forward(self) -> None:
        secs = parse_alarm("02:00", now=self.now)
        self.assertAlmostEqual(secs, 24 * 3600)

    def test_with_seconds(self) -> None:
        secs = parse_alarm("02:00:30", now=self.now)
        self.assertAlmostEqual(secs, 30)

    def test_am_pm(self) -> None:
        secs = parse_alarm("6:30pm", now=self.now)
        self.assertAlmostEqual(secs, (18 - 2) * 3600 + 30 * 60)
        # 12am == midnight -> tomorrow 00:00
        secs = parse_alarm("12:00am", now=self.now)
        self.assertAlmostEqual(secs, 22 * 3600)
        # 12pm == noon
        secs = parse_alarm("12:00pm", now=self.now)
        self.assertAlmostEqual(secs, 10 * 3600)

    def test_rejects_garbage(self) -> None:
        for bad in ["", "2pm", "25:00", "02:60", "abc", "2"]:
            with self.assertRaises(ValueError, msg=bad):
                parse_alarm(bad, now=self.now)


if __name__ == "__main__":
    unittest.main()

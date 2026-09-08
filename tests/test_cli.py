"""Headless tests for CLI argument parsing and JSON parameter loading.

No display or Qt widgets required -- the CLI parser and the load/apply path are
pure data transforms.
"""

import contextlib
import io
import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from naive_timer.shard import ShardParams
from naive_timer.tuning import apply_json_dict, load_params_file, ParamsError
from naive_timer.countdown import parse_duration, parse_alarm
from naive_timer.app import _parse_cli

# Path to real params files in the repo root.
_REPO_ROOT = os.path.join(os.path.dirname(__file__), "..")
_DEFAULT_PARAMS = os.path.join(_REPO_ROOT, "default-params.json")
_GREEN_NEBULA = os.path.join(_REPO_ROOT, "green-nebula.json")


class ParseCliTest(unittest.TestCase):
    """_parse_cli takes an explicit argv list and returns an argparse.Namespace.

    argparse reports usage errors and --help by raising SystemExit; the error
    tests swallow its stderr/stdout so a passing run stays quiet.
    """

    def _expect_exit(self, *args: str) -> SystemExit:
        buf = io.StringIO()
        with contextlib.redirect_stderr(buf), contextlib.redirect_stdout(buf):
            with self.assertRaises(SystemExit) as cm:
                _parse_cli(list(args))
        return cm.exception

    # -- happy paths --

    def test_no_args(self) -> None:
        cli = _parse_cli([])
        self.assertIsNone(cli.json)
        self.assertFalse(cli.no_panel)
        self.assertIsNone(cli.timer)

    def test_json_space(self) -> None:
        cli = _parse_cli(["--json", "params.json"])
        self.assertEqual(cli.json, "params.json")
        self.assertFalse(cli.no_panel)

    def test_json_equals(self) -> None:
        cli = _parse_cli(["--json=params.json"])
        self.assertEqual(cli.json, "params.json")

    def test_json_equals_with_path(self) -> None:
        cli = _parse_cli(["--json=./configs/green-nebula.json"])
        self.assertEqual(cli.json, "./configs/green-nebula.json")

    def test_no_panel_flag(self) -> None:
        cli = _parse_cli(["--no-panel"])
        self.assertTrue(cli.no_panel)
        self.assertIsNone(cli.json)

    def test_json_and_no_panel(self) -> None:
        cli = _parse_cli(["--json", "p.json", "--no-panel"])
        self.assertEqual(cli.json, "p.json")
        self.assertTrue(cli.no_panel)

    def test_parse_does_not_touch_sys_argv(self) -> None:
        before = list(sys.argv)
        _parse_cli(["--no-panel"])
        self.assertEqual(sys.argv, before)

    # -- timer flag --

    def test_timer_space(self) -> None:
        cli = _parse_cli(["--timer", "30m"])
        self.assertEqual(cli.timer, "30m")
        self.assertIsNone(cli.json)
        self.assertFalse(cli.no_panel)

    def test_timer_equals(self) -> None:
        cli = _parse_cli(["--timer=1h30m"])
        self.assertEqual(cli.timer, "1h30m")

    def test_timer_colon_format(self) -> None:
        cli = _parse_cli(["--timer", "25:00"])
        self.assertEqual(cli.timer, "25:00")

    def test_timer_alarm_format(self) -> None:
        cli = _parse_cli(["--timer", "02:54"])
        self.assertEqual(cli.timer, "02:54")

    def test_timer_with_json(self) -> None:
        cli = _parse_cli(["--timer", "10m", "--json", "p.json"])
        self.assertEqual(cli.timer, "10m")
        self.assertEqual(cli.json, "p.json")

    def test_timer_missing_value(self) -> None:
        self._expect_exit("--timer")

    # -- error paths (argparse exits) --

    def test_json_missing_value(self) -> None:
        self._expect_exit("--json")

    def test_unknown_option(self) -> None:
        self._expect_exit("--foo")

    def test_positional_is_error(self) -> None:
        self._expect_exit("my-params.json")

    def test_help_exits_zero(self) -> None:
        exc = self._expect_exit("--help")
        self.assertEqual(exc.code, 0)


class LoadParamsFileTest(unittest.TestCase):
    """load_params_file reads a JSON file and returns a dict."""

    def test_default_params(self) -> None:
        data = load_params_file(_DEFAULT_PARAMS)
        self.assertIsInstance(data, dict)
        self.assertIn("light_x", data)
        self.assertIn("glass_color", data)

    def test_green_nebula(self) -> None:
        data = load_params_file(_GREEN_NEBULA)
        self.assertIn("gravity", data)
        self.assertEqual(data["gravity"], 0.0)

    def test_missing_file(self) -> None:
        with self.assertRaises(ParamsError) as cm:
            load_params_file("/nonexistent/path/params.json")
        self.assertIn("not found", str(cm.exception))

    def test_directory_rejected(self) -> None:
        with self.assertRaises(ParamsError) as cm:
            load_params_file(_REPO_ROOT)
        self.assertIn("not a file", str(cm.exception))

    def test_invalid_json(self) -> None:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            try:
                f.write("{not valid json}")
                f.flush()
                with self.assertRaises(ParamsError):
                    load_params_file(f.name)
            finally:
                os.unlink(f.name)

    def test_non_object_top_level(self) -> None:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            try:
                json.dump([1, 2, 3], f)
                f.flush()
                with self.assertRaises(ParamsError) as cm:
                    load_params_file(f.name)
                self.assertIn("object", str(cm.exception))
            finally:
                os.unlink(f.name)

    def test_empty_object_is_valid(self) -> None:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            try:
                json.dump({}, f)
                f.flush()
                data = load_params_file(f.name)
            finally:
                os.unlink(f.name)
        self.assertEqual(data, {})


class ApplyJsonDictTest(unittest.TestCase):
    """apply_json_dict folds a dict into ShardParams in place."""

    def test_overwrites_known_field(self) -> None:
        params = ShardParams()
        apply_json_dict(params, {"light_x": 99.0})
        self.assertEqual(params.light_x, 99.0)

    def test_tuple_fields_coerced_from_list(self) -> None:
        params = ShardParams()
        apply_json_dict(params, {"glass_color": [1.0, 0.5, 0.0]})
        self.assertEqual(params.glass_color, (1.0, 0.5, 0.0))
        self.assertIsInstance(params.glass_color, tuple)

    def test_ignores_unknown_fields(self) -> None:
        params = ShardParams()
        before = params.light_x
        apply_json_dict(params, {"__nonexistent_key__": 123})
        self.assertEqual(params.light_x, before)

    def test_partial_update(self) -> None:
        params = ShardParams()
        orig_nebula = params.nebula
        apply_json_dict(params, {"gravity": 2.0})
        self.assertEqual(params.gravity, 2.0)
        self.assertEqual(params.nebula, orig_nebula)  # unchanged

    def test_loads_real_file_and_applies(self) -> None:
        params = ShardParams()
        data = load_params_file(_GREEN_NEBULA)
        apply_json_dict(params, data)

        # green-nebula.json has distinctive values
        self.assertEqual(params.light_x, 2.87)
        self.assertEqual(params.gravity, 0.0)
        self.assertEqual(params.nebula, 1.17)
        self.assertEqual(params.glass_color, (0.2, 0.2, 0.2))

    def test_string_fields(self) -> None:
        params = ShardParams()
        apply_json_dict(params, {"font_family": "serif", "font_bold": False})
        self.assertEqual(params.font_family, "serif")
        self.assertFalse(params.font_bold)


class TimerValueParsingTest(unittest.TestCase):
    """Verify the --timer value parsing strategy used in main().

    The main() function tries parse_duration first, then parse_alarm as a
    fallback. These tests verify that strategy end-to-end without involving
    the GUI.
    """

    def _parse_timer(self, raw: str) -> tuple[float, str]:
        """Replicate the parsing logic from main()."""
        try:
            seconds = parse_duration(raw)
            return (seconds, "Duration")
        except ValueError:
            seconds = parse_alarm(raw)
            return (seconds, "Alarm at")

    # -- duration forms (parsed as Duration) --

    def test_unit_seconds(self) -> None:
        secs, mode = self._parse_timer("90s")
        self.assertEqual(secs, 90.0)
        self.assertEqual(mode, "Duration")

    def test_unit_minutes(self) -> None:
        secs, mode = self._parse_timer("30m")
        self.assertEqual(secs, 1800.0)
        self.assertEqual(mode, "Duration")

    def test_unit_hours(self) -> None:
        secs, mode = self._parse_timer("1h")
        self.assertEqual(secs, 3600.0)
        self.assertEqual(mode, "Duration")

    def test_combined_units(self) -> None:
        secs, mode = self._parse_timer("1h30m")
        self.assertEqual(secs, 5400.0)
        self.assertEqual(mode, "Duration")

    def test_colon_mm_ss(self) -> None:
        secs, mode = self._parse_timer("25:00")
        self.assertEqual(secs, 1500.0)
        self.assertEqual(mode, "Duration")

    def test_colon_hh_mm_ss(self) -> None:
        secs, mode = self._parse_timer("1:30:00")
        self.assertEqual(secs, 5400.0)
        self.assertEqual(mode, "Duration")

    def test_bare_number_is_minutes(self) -> None:
        secs, mode = self._parse_timer("12")
        self.assertEqual(secs, 720.0)
        self.assertEqual(mode, "Duration")

    # -- alarm forms (parsed as Alarm at since duration fails) --
    # Note: colon formats are consumed by parse_duration (mm:ss), so only
    # am/pm suffixed times reach parse_alarm.

    def test_alarm_12h_pm(self) -> None:
        secs, mode = self._parse_timer("6:30pm")
        self.assertEqual(mode, "Alarm at")
        self.assertGreater(secs, 0)

    def test_alarm_12h_am(self) -> None:
        secs, mode = self._parse_timer("7:00am")
        self.assertEqual(mode, "Alarm at")
        self.assertGreater(secs, 0)

    # -- invalid forms raise ValueError --

    def test_invalid_raises(self) -> None:
        with self.assertRaises(ValueError):
            self._parse_timer("not-a-duration")

    def test_empty_raises(self) -> None:
        with self.assertRaises(ValueError):
            self._parse_timer("")

    def test_negative_raises(self) -> None:
        with self.assertRaises(ValueError):
            self._parse_timer("-5")


if __name__ == "__main__":
    unittest.main()

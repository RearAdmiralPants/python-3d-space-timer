"""PySide6 GUI: a tabbed Stopwatch + Countdown Timer.

Thin view layer over the ``Stopwatch`` and ``Countdown`` models (both of which
are unit-tested headlessly). This file only renders state, forwards button
presses, and drives the audible/visual alert.

Run with:  python -m naive_timer
"""

from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Audio backend: force Qt off its PipeWire backend.  MUST precede any import
# that can initialise QtMultimedia.
# ---------------------------------------------------------------------------
#
# Qt 6.11 defaults to a native PipeWire audio backend on Linux. Against
# PipeWire 1.0.5 (Ubuntu 24.04) that backend crashes the *whole process*
# whenever the audio sink it is attached to disappears -- Bluetooth headphones
# dropping, an HDMI sink going away with the monitor, a card re-profiling.
# Two signatures, both from PipeWire's own threads, neither catchable here:
#
#   SIGSEGV  null deref at +0x1c in libpipewire-module-protocol-native.so
#   SIGABRT  "corrupted size vs. prev_size" in pw_stream_new_simple, i.e. the
#            heap was already corrupt by the time the rebuild allocated
#
# It needs no alarm and no user action: constructing a QSoundEffect opens a
# stream, so an idle app that merely *might* ring later is fully exposed. This
# is what the "segfaults after ~45 minutes" report was -- 45 minutes is just
# how long it took for something in the session to touch the device list. The
# four "QSocketNotifier: Socket notifiers cannot be enabled or disabled from
# another thread" warnings that preceded each crash are the same backend, and
# they disappear entirely under PulseAudio.
#
# Reproduced deterministically (crash on the first attempt, PipeWire; 20/20
# clean, PulseAudio) by playing on a null sink and then unloading it.
#
# PulseAudio here means pipewire-pulse in practice -- the audio still goes
# through PipeWire, just via a client library that survives its server
# rearranging devices. setdefault, so `QT_AUDIO_BACKEND=PipeWire` still lets
# you check whether a newer PipeWire has fixed it.
if sys.platform.startswith("linux"):
    os.environ.setdefault("QT_AUDIO_BACKEND", "PulseAudio")

from PySide6.QtCore import Qt, QElapsedTimer, QTimer, QUrl
from PySide6.QtGui import QGuiApplication, QSurfaceFormat, QIcon, QSurfaceFormat
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from .countdown import Countdown, parse_alarm, parse_duration
from .shard import ShardParams, ShardWidget, default_surface_format
from .stopwatch import State, Stopwatch, format_elapsed
from . import sound, tuning


# ---------------------------------------------------------------------------
# Stay on top: X11 EWMH _NET_WM_STATE_ABOVE
# ---------------------------------------------------------------------------
#
# There is deliberately no Qt fallback here. The documented Qt approach --
# setWindowFlag(Qt.WindowStaysOnTopHint) followed by show() -- *destroys this
# window* under PySide6 6.11 + QOpenGLWidget on X11: the native window is torn
# down and never remapped, so the app vanishes from the screen and from
# _NET_CLIENT_LIST_STACKING, leaking a fresh window id on every toggle.
# hide()-first behaves no better. Asking the window manager to change the
# state, which is what EWMH is for, leaves the window (and its GL context)
# untouched.
#
# When EWMH is unavailable -- a Wayland-native session, no libX11, or a window
# manager that does not advertise the ABOVE state -- the checkbox is disabled
# with a reason in its tooltip. A control that visibly cannot work beats one
# that silently does nothing, which is exactly the failure this replaced.

# Xlib constants (X.h / the EWMH spec).
_CLIENT_MESSAGE = 33
_SUBSTRUCTURE_NOTIFY = 1 << 19
_SUBSTRUCTURE_REDIRECT = 1 << 20
_XA_ATOM = 4
_STATE_REMOVE = 0
_STATE_ADD = 1
_SOURCE_APPLICATION = 1  # _NET_WM_STATE data.l[3]


class _XEvent(ctypes.Structure):
    """XClientMessageEvent, padded out to a full XEvent union.

    Field order matches the C struct and ctypes inserts the natural alignment
    padding, so this must not be hand-padded. Two details are easy to get
    wrong and silently fatal:

    * ``data`` is the union's ``long l[5]``, so its members are 8 bytes wide on
      LP64 -- *not* 32-bit, despite ``format = 32`` (which describes the wire
      format, not the client-side array).
    * ``XSendEvent`` takes an ``XEvent *``, a union padded to ``long pad[24]``
      = 192 bytes. Handing it a short buffer invites a read past the end.
    """

    _fields_ = [
        ("type", ctypes.c_int),
        ("serial", ctypes.c_ulong),
        ("send_event", ctypes.c_int),
        ("display", ctypes.c_void_p),
        ("window", ctypes.c_ulong),
        ("message_type", ctypes.c_ulong),
        ("format", ctypes.c_int),
        ("data", ctypes.c_long * 5),
        ("_pad", ctypes.c_long * 12),  # to sizeof(XEvent) == 192
    ]


class _StayOnTop:
    """Asks the window manager to add/remove _NET_WM_STATE_ABOVE.

    Construct only after the QApplication exists: availability depends on the
    Qt platform plugin, and under a non-xcb plugin ``winId()`` is not an X11
    window id, so sending it to the X server would be meaningless.
    """

    def __init__(self) -> None:
        self.available = False
        self.reason = ""
        self._display: int | None = None
        self._root = 0

        platform = QGuiApplication.platformName()
        if platform != "xcb":
            self.reason = (
                f"needs an X11 (xcb) session; Qt is running on {platform!r}"
            )
            return

        lib_path = ctypes.util.find_library("X11")
        if lib_path is None:
            self.reason = "libX11 not found"
            return
        try:
            self._x11 = ctypes.CDLL(lib_path, use_errno=True)
        except OSError as exc:
            self.reason = f"cannot load {lib_path}: {exc}"
            return

        self._declare_signatures()

        self._display = self._x11.XOpenDisplay(None)
        if not self._display:
            self.reason = "cannot open the X display"
            return

        self._root = self._x11.XDefaultRootWindow(self._display)
        self._atom_state = self._intern("_NET_WM_STATE")
        self._atom_above = self._intern("_NET_WM_STATE_ABOVE")
        if not (self._atom_state and self._atom_above):
            self.reason = "the X server refused to intern the EWMH atoms"
            return

        # EWMH requires a compliant WM to advertise every state it honours.
        # Checking once here is what makes a silent no-op impossible: without
        # it, XSendEvent happily succeeds while nothing changes.
        supported = self._read_atoms(self._root, self._intern("_NET_SUPPORTED"))
        if self._atom_above not in supported:
            self.reason = (
                "the window manager does not advertise _NET_WM_STATE_ABOVE"
                if supported
                else "no EWMH-compliant window manager is running"
            )
            return

        self.available = True

    # -- ctypes plumbing ----------------------------------------------------

    def _declare_signatures(self) -> None:
        """Every restype matters: the default is ``c_int``, which truncates a
        64-bit ``Display *`` to 32 bits and only appears to work while the
        allocation happens to land in the low 2 GB."""
        x11 = self._x11

        x11.XOpenDisplay.restype = ctypes.c_void_p
        x11.XOpenDisplay.argtypes = [ctypes.c_char_p]

        x11.XDefaultRootWindow.restype = ctypes.c_ulong
        x11.XDefaultRootWindow.argtypes = [ctypes.c_void_p]

        x11.XInternAtom.restype = ctypes.c_ulong
        x11.XInternAtom.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int]

        x11.XSendEvent.restype = ctypes.c_int
        x11.XSendEvent.argtypes = [
            ctypes.c_void_p,   # display
            ctypes.c_ulong,    # w
            ctypes.c_int,      # propagate
            ctypes.c_long,     # event_mask
            ctypes.c_void_p,   # XEvent *
        ]

        x11.XGetWindowProperty.restype = ctypes.c_int
        x11.XGetWindowProperty.argtypes = [
            ctypes.c_void_p,                        # display
            ctypes.c_ulong,                         # w
            ctypes.c_ulong,                         # property
            ctypes.c_long,                          # long_offset
            ctypes.c_long,                          # long_length
            ctypes.c_int,                           # delete
            ctypes.c_ulong,                         # req_type
            ctypes.POINTER(ctypes.c_ulong),         # actual_type_return
            ctypes.POINTER(ctypes.c_int),           # actual_format_return
            ctypes.POINTER(ctypes.c_ulong),         # nitems_return
            ctypes.POINTER(ctypes.c_ulong),         # bytes_after_return
            ctypes.POINTER(ctypes.POINTER(ctypes.c_ulong)),  # prop_return
        ]

        x11.XFree.restype = ctypes.c_int
        x11.XFree.argtypes = [ctypes.c_void_p]

        x11.XFlush.restype = ctypes.c_int
        x11.XFlush.argtypes = [ctypes.c_void_p]

        x11.XSync.restype = ctypes.c_int
        x11.XSync.argtypes = [ctypes.c_void_p, ctypes.c_int]

    def _intern(self, name: str) -> int:
        return self._x11.XInternAtom(self._display, name.encode(), False)

    def _read_atoms(self, window: int, prop: int) -> frozenset[int]:
        """Read an ATOM-typed property. Empty set on any failure.

        Xlib returns format-32 properties as an array of C ``long``, so the
        items are 8 bytes wide here even though the wire format is 32-bit.
        """
        actual_type = ctypes.c_ulong()
        actual_format = ctypes.c_int()
        nitems = ctypes.c_ulong()
        bytes_after = ctypes.c_ulong()
        data = ctypes.POINTER(ctypes.c_ulong)()

        status = self._x11.XGetWindowProperty(
            self._display, window, prop,
            0, 4096, False, _XA_ATOM,
            ctypes.byref(actual_type), ctypes.byref(actual_format),
            ctypes.byref(nitems), ctypes.byref(bytes_after),
            ctypes.byref(data),
        )
        if status != 0 or not data:
            return frozenset()
        try:
            if actual_type.value != _XA_ATOM or actual_format.value != 32:
                return frozenset()
            return frozenset(data[i] for i in range(nitems.value))
        finally:
            self._x11.XFree(data)

    # -- public -------------------------------------------------------------

    def set_above(self, window_id: int, above: bool) -> None:
        """Ask the WM to add or remove ABOVE on a *mapped* window.

        A _NET_WM_STATE ClientMessage is only honoured while the window is
        mapped, so call this from a user action rather than before show().
        """
        if not self.available:
            return

        event = _XEvent()
        event.type = _CLIENT_MESSAGE
        event.send_event = True
        event.display = self._display
        event.window = window_id
        event.message_type = self._atom_state
        event.format = 32
        event.data[0] = _STATE_ADD if above else _STATE_REMOVE
        event.data[1] = self._atom_above
        event.data[2] = 0  # no second property
        event.data[3] = _SOURCE_APPLICATION

        # The WM listens on the root window for these.
        self._x11.XSendEvent(
            self._display,
            self._root,
            False,
            _SUBSTRUCTURE_NOTIFY | _SUBSTRUCTURE_REDIRECT,
            ctypes.byref(event),
        )
        self._x11.XFlush(self._display)

    def is_above(self, window_id: int) -> bool:
        """Whether the WM currently lists ABOVE for this window.

        The WM owns _NET_WM_STATE, so this is the only honest answer to
        "did it work?" -- and it is what the tests assert on. XSync first,
        since the state change is asynchronous.
        """
        if not self.available:
            return False
        self._x11.XSync(self._display, False)
        return self._atom_above in self._read_atoms(window_id, self._atom_state)


_stay_on_top: _StayOnTop | None = None


def stay_on_top() -> _StayOnTop:
    """The process-wide handle, built on first use.

    Lazy on purpose: constructing it needs a live QApplication, and importing
    this module must not open an X connection (the headless tests import it).
    """
    global _stay_on_top
    if _stay_on_top is None:
        _stay_on_top = _StayOnTop()
    return _stay_on_top

# ~60 FPS refresh for smooth animation.
FRAME_MS = 16

# How long before the alarm to build the sound objects. The app holds no audio
# resources outside this window and the alert itself -- see
# TimerWidget._sync_alert_player() for why that matters.
#
# Note this opens the client stream early; it does not resume a suspended sink,
# which is what a Bluetooth sink actually needs to not clip the first fraction
# of a second. That needs a silent play, and is not implemented yet.
ALERT_WARMUP_S = 1.0

_ICON_DIR = Path(__file__).parent / "icons"

# Desktop identity. This must match the installed .desktop file's basename,
# because that is how the panel maps a window back to its icon. Without it Qt
# derives WM_CLASS from the interpreter, so every Python tool on the system
# shows the same generic snake.
APP_ID = "naive-timer"


def _app_icon() -> QIcon:
    """The window icon -- deliberately a single bitmap, not a multi-size set.

    On xcb, Qt re-renders every entry of a multi-size QIcon at the screen's
    device pixel ratio, so on a HiDPI display the small hand-tuned sizes come
    back upscaled and blurry; worse, a 256px entry pushes _NET_WM_ICON past
    X11's ~256KB property limit and the icon is dropped entirely, silently.
    Both verified with xprop.

    The crisp per-size art reaches the panel by a different route: the desktop
    entry's Icon= key, resolved through the installed icon theme, which
    Cinnamon prefers over _NET_WM_ICON for any window it can match to a
    .desktop file. See tools/install-desktop.sh.
    """
    return QIcon(str(_ICON_DIR / "timer-icon-128.png"))


def _parse_cli(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Pure with respect to process state: tests pass an explicit ``argv`` (the
    args after the program name); ``None`` reads ``sys.argv``. argparse itself
    handles ``--help`` and usage errors, exiting 0 and 2 respectively.

    Returns a namespace with ``json`` (str | None), ``no_panel`` (bool), and
    ``timer`` (str | None).
    """
    parser = argparse.ArgumentParser(
        prog="naive-timer",
        description="A visually-appealing Linux stopwatch/timer.",
    )
    parser.add_argument(
        "--json",
        metavar="FILE",
        help="Load shard parameters from a JSON file and apply them on startup.",
    )
    parser.add_argument(
        "--no-panel",
        action="store_true",
        help="Never open the dev/tuning panel, even when NAIVE_TIMER_TUNE=1.",
    )
    parser.add_argument(
        "--timer",
        metavar="VALUE",
        help="Open on the Timer tab and start a countdown immediately. "
        "VALUE is parsed as a duration (e.g. '12m', '1h30m', '25:00') or, "
        "if that fails, as an alarm time (e.g. '02:54', '6:30pm'). "
        "On parse error the application exits with a non-zero status.",
    )
    return parser.parse_args(argv)


# Longest step we will hand the animation in one frame. A stall, a drag of the
# window, or a laptop resume can leave an arbitrarily large gap; without a clamp
# the camera would teleport rather than sway.
MAX_FRAME_S = 0.25


class FrameClock:
    """Real elapsed time between frames.

    The animation used to advance by a fixed FRAME_MS regardless of how long
    the frame actually took, so on a GPU that could not hold 60 FPS the sway,
    twinkle and drift all ran in slow motion -- at 27 ms/frame, 60% speed. The
    displayed time was always correct (that comes from the models, which read
    the wall clock); it was the animation that lagged. Measuring the interval
    makes animation speed the same on every machine.
    """

    def __init__(self) -> None:
        self._timer = QElapsedTimer()
        self._timer.start()

    def tick(self) -> float:
        """Seconds since the previous call, clamped."""
        return min(self._timer.restart() / 1000.0, MAX_FRAME_S)


# QtMultimedia is optional; if unavailable we degrade to a visual-only alert.
try:
    from PySide6.QtMultimedia import QSoundEffect

    _HAVE_AUDIO = True
except Exception:  # pragma: no cover - depends on platform Qt build
    _HAVE_AUDIO = False


class StopwatchWidget(QWidget):
    def __init__(self, params: ShardParams | None = None) -> None:
        super().__init__()
        self._sw = Stopwatch()
        self._shard = ShardWidget(self._sw, params)
        # True from the Reset shatter until the pieces clear and we zero out.
        self._resetting = False

        self._start_btn = QPushButton("Start")
        self._start_btn.clicked.connect(self._on_toggle)
        reset_btn = QPushButton("Reset")
        reset_btn.clicked.connect(self._on_reset)

        buttons = QHBoxLayout()
        buttons.addWidget(self._start_btn)
        buttons.addWidget(reset_btn)

        layout = QVBoxLayout(self)
        layout.addWidget(self._shard, stretch=1)
        layout.addLayout(buttons)

        self._clock = FrameClock()
        timer = QTimer(self)
        timer.timeout.connect(self._tick)
        timer.start(FRAME_MS)

    def _on_toggle(self) -> None:
        # Ignore Start/Pause while the shard is mid-shatter; the readout is
        # frozen and the model is being reset out from under it.
        if self._resetting:
            return
        self._sw.toggle()
        self._start_btn.setText(
            "Pause" if self._sw.state is State.RUNNING else "Start"
        )

    def _on_reset(self) -> None:
        # Reset shatters the shard rather than snapping to zero. Pausing first
        # freezes the readout at its final value so the numerals fly apart
        # showing that time; the model is zeroed only once the pieces clear
        # (see _tick), after which the shard reassembles at 00:00:00.
        if self._resetting:
            return
        self._resetting = True
        self._sw.pause()
        self._shard.set_alarm(True)
        self._start_btn.setText("Start")

    def _tick(self) -> None:
        if self._resetting and self._shard.pieces_have_cleared:
            self._sw.reset()
            self._shard.set_alarm(False)  # reassemble at zero
            self._resetting = False
        self._shard.set_text(format_elapsed(self._sw.elapsed()))
        self._shard.advance(self._clock.tick())


class TimerWidget(QWidget):
    """Countdown that accepts a duration ('12m') or an alarm time ('02:54')."""

    def __init__(self, params: ShardParams | None = None) -> None:
        super().__init__()
        self._cd = Countdown()
        self._shard = ShardWidget(self._cd, params)
        self._alerting = False
        # Built on demand, never at startup -- see _sync_alert_player().
        self._alert: "_AlertPlayer | None" = None

        self._mode = QComboBox()
        self._mode.addItems(["Duration", "Alarm at"])
        self._input = QLineEdit()
        self._input.setPlaceholderText("e.g. 12m  ·  1h30m  ·  25:00")
        self._input.returnPressed.connect(self._on_start)
        self._mode.currentIndexChanged.connect(self._update_placeholder)

        input_row = QHBoxLayout()
        input_row.addWidget(self._mode)
        input_row.addWidget(self._input, stretch=1)

        self._status = QLabel("")
        self._status.setAlignment(Qt.AlignCenter)

        self._start_btn = QPushButton("Start")
        self._start_btn.clicked.connect(self._on_start)
        reset_btn = QPushButton("Reset")
        reset_btn.clicked.connect(self._on_reset)
        self._dismiss_btn = QPushButton("Dismiss")
        self._dismiss_btn.clicked.connect(self._on_dismiss)
        self._dismiss_btn.setVisible(False)

        buttons = QHBoxLayout()
        buttons.addWidget(self._start_btn)
        buttons.addWidget(reset_btn)
        buttons.addWidget(self._dismiss_btn)

        layout = QVBoxLayout(self)
        layout.addLayout(input_row)
        layout.addWidget(self._shard, stretch=1)
        layout.addWidget(self._status)
        layout.addLayout(buttons)

        self._clock = FrameClock()
        timer = QTimer(self)
        timer.timeout.connect(self._tick)
        timer.start(FRAME_MS)

    def _update_placeholder(self) -> None:
        if self._mode.currentText() == "Duration":
            self._input.setPlaceholderText("e.g. 12m  ·  1h30m  ·  25:00")
        else:
            self._input.setPlaceholderText("e.g. 02:54  ·  6:30pm  ·  14:00")

    def _on_start(self) -> None:
        # If paused mid-countdown, Start just resumes.
        from .countdown import Phase

        if self._cd.phase is Phase.PAUSED:
            self._cd.start()
            self._start_btn.setText("Pause")
            return
        if self._cd.is_running:
            self._cd.pause()
            self._start_btn.setText("Start")
            return

        text = self._input.text().strip()
        try:
            if self._mode.currentText() == "Duration":
                seconds = parse_duration(text)
            else:
                seconds = parse_alarm(text)
        except ValueError as exc:
            self._status.setText(f"⚠ {exc}")
            return

        self._cd.configure(seconds)
        self._cd.start()
        self._status.setText("")
        self._start_btn.setText("Pause")

    def start_with(self, seconds: float, display_text: str, mode: str) -> None:
        """Programmatically configure and start a countdown.

        Bypasses input parsing — ``seconds`` is already a validated, parsed
        value. ``display_text`` is shown in the text box so the user sees what
        was set. ``mode`` is one of "Duration" or "Alarm at" and controls
        the combo box selection.

        Called from ``main()`` when the ``--timer`` CLI flag is used.
        """
        self._input.setText(display_text)
        idx = self._mode.findText(mode)
        if idx >= 0:
            self._mode.setCurrentIndex(idx)
        self._cd.configure(seconds)
        self._cd.start()
        self._status.setText("")
        self._start_btn.setText("Pause")

    def _on_reset(self) -> None:
        # The countdown has to be reset *before* _stop_alert(), which re-checks
        # whether the alarm is still imminent: called the other way round, a
        # reset inside the last ALERT_WARMUP_S sees a still-running countdown
        # and keeps the sound objects alive. Only until the next frame, but the
        # point of the exercise is not to hold streams we have no use for.
        self._cd.reset()
        self._stop_alert()
        self._start_btn.setText("Start")
        self._status.setText("")

    def _on_dismiss(self) -> None:
        self._cd.dismiss()
        self._stop_alert()

    def _stop_alert(self) -> None:
        self._alerting = False
        self._dismiss_btn.setVisible(False)
        self._shard.set_alarm(False)
        self._sync_alert_player()

    def _sync_alert_player(self) -> None:
        """Hold audio resources only while the alarm is imminent or ringing.

        Constructing a ``QSoundEffect`` opens a client stream on the default
        sink and keeps it there until the object dies -- uncorked, at that,
        with nothing ever written to it. Building ``_AlertPlayer`` in
        ``__init__`` therefore parked a permanent stream on the sink from
        launch to exit, in *both* tabs, for an app that might never ring.

        That is the same exposure that produced the "segfaults after ~45
        minutes" bug (see docs/HANDOFF.md): pinning the PulseAudio backend
        stopped the crash but left the stream. It also makes the app a
        participant in every device-list change and every re-route the session
        manager performs -- and on PipeWire 1.0.5 a re-route can leave a stream
        orphaned, linked to no sink and corked, which is silent and permanent.

        So: no audio objects until the countdown is nearly up, and they are
        released as soon as it is not. Declarative rather than event-driven
        because pause, resume, reset and dismiss all have to be covered, and an
        invariant checked once a frame cannot miss one of them.
        """
        if not _HAVE_AUDIO:
            return
        imminent = self._alerting or (
            self._cd.is_running and self._cd.remaining() <= ALERT_WARMUP_S
        )
        if imminent and self._alert is None:
            self._alert = _AlertPlayer()
        elif not imminent and self._alert is not None:
            self._alert.stop()
            self._alert = None

    def _tick(self) -> None:
        self._shard.set_text(format_elapsed(self._cd.remaining()))
        self._shard.advance(self._clock.tick())
        self._sync_alert_player()

        # The alert is carried entirely by the shard: it fractures, then
        # breathes dark red. No background flash -- that read as jarring.
        if self._cd.alert_active():
            if not self._alerting:
                self._begin_alert()
        elif self._alerting:
            # Alert window elapsed on its own.
            self._stop_alert()
            self._start_btn.setText("Start")

    def _begin_alert(self) -> None:
        self._alerting = True
        # Normally _tick's warm-up built this ALERT_WARMUP_S ago. This covers
        # the cases it cannot: a countdown configured with less than that left,
        # or start_with() from the --timer flag firing almost immediately.
        self._sync_alert_player()
        self._status.setText("⏰ Time's up!")
        self._start_btn.setText("Start")
        self._dismiss_btn.setVisible(True)
        self._shard.set_alarm(True)
        if self._alert is not None:
            self._alert.play()


class _AlertPlayer:
    """The shatter, once, then the chime looping quietly until stopped.

    Both are synthesized at runtime (see ``sound.py``); no binary assets. Note
    that QSoundEffect decodes only uncompressed WAV -- it errors on FLAC.
    """

    def __init__(self) -> None:
        self._effect = QSoundEffect()
        self._effect.setSource(QUrl.fromLocalFile(sound.default_chime_path()))
        self._effect.setLoopCount(QSoundEffect.Loop.Infinite.value)
        self._effect.setVolume(0.35)

        # One-shot, played the instant the shard breaks. Its own file is
        # already ~7 dB below the reference recordings, so this stays near
        # unity; turn the WAV's `amplitude` down rather than this, so the
        # headroom is baked in.
        self._shatter = QSoundEffect()
        self._shatter.setSource(QUrl.fromLocalFile(sound.default_shatter_path()))
        self._shatter.setLoopCount(1)
        self._shatter.setVolume(0.9)

    def play(self) -> None:
        self._shatter.play()
        self._effect.play()

    def stop(self) -> None:
        self._shatter.stop()
        self._effect.stop()


class MainWindow(QTabWidget):
    def __init__(self, *, timer_value: tuple[float, str, str] | None = None) -> None:
        super().__init__()
        self.setWindowTitle("Naive Linux Timer")
        # One ShardParams shared by both tabs, so the tuning panel moves both
        # shards at once. Tuning only the Timer tab looked like a dead slider
        # whenever the Stopwatch tab was in front.
        self.shard_params = ShardParams()
        self.stopwatch_tab = StopwatchWidget(self.shard_params)
        self.timer_tab = TimerWidget(self.shard_params)
        self.addTab(self.stopwatch_tab, "Stopwatch")
        self.addTab(self.timer_tab, "Timer")

        # "Stay on top" toggle in the top-right corner (right of the tabs).
        self._above = stay_on_top()
        self._stay_on_top = QCheckBox("Stay on top")
        if self._above.available:
            self._stay_on_top.toggled.connect(self._on_stay_on_top)
        else:
            self._stay_on_top.setEnabled(False)
            self._stay_on_top.setToolTip(
                f"Stay on top is unavailable: {self._above.reason}."
            )
        self.setCornerWidget(self._stay_on_top, Qt.Corner.TopRightCorner)

        # --timer CLI flag: pre-configured countdown, jump to Timer tab.
        if timer_value is not None:
            seconds, display_text, mode = timer_value
            self.timer_tab.start_with(seconds, display_text, mode)
            self.setCurrentWidget(self.timer_tab)

    def shards(self) -> list[ShardWidget]:
        return [self.stopwatch_tab._shard, self.timer_tab._shard]

    def _on_stay_on_top(self, checked: bool) -> None:
        """Ask the WM to raise the window above others, or stop.

        Connected only when EWMH is available, so there is no fallback branch
        here -- see the module header for why Qt's setWindowFlags path is not
        one.
        """
        self._above.set_above(int(self.winId()), checked)


def main() -> int:
    # Parse CLI args before QApplication. argparse handles --help and usage
    # errors itself (exiting 0 / 2); we only handle our own concerns below.
    cli = _parse_cli()

    # Load JSON params if requested, before building any GL widgets.
    params_data: dict | None = None
    if cli.json is not None:
        try:
            params_data = tuning.load_params_file(cli.json)
        except tuning.ParamsError as exc:
            print(f"naive-timer: {exc}", file=sys.stderr)
            return 1
        print(f"[timer] loaded params from {cli.json}")

    # Parse --timer value before the GUI starts so errors exit cleanly.
    timer_value: tuple[float, str, str] | None = None
    if cli.timer is not None:
        raw = cli.timer.strip()
        try:
            seconds = parse_duration(raw)
            mode = "Duration"
        except ValueError:
            try:
                seconds = parse_alarm(raw)
                mode = "Alarm at"
            except ValueError as exc:
                print(f"naive-timer: cannot parse timer value {cli.timer!r}: {exc}", file=sys.stderr)
                return 1
        timer_value = (seconds, raw, mode)

    # Must precede QApplication: the GL context is chosen at widget creation.
    QSurfaceFormat.setDefaultFormat(default_surface_format())

    app = QApplication(sys.argv)

    # Must precede window creation: WM_CLASS is stamped on the X11 window when
    # it is created, and the panel reads it exactly once.
    app.setApplicationName(APP_ID)
    app.setApplicationDisplayName("Naive Timer")
    app.setDesktopFileName(APP_ID)

    app.setWindowIcon(_app_icon())

    window = MainWindow(timer_value=timer_value)

    # Apply CLI-loaded params to the shared ShardParams instance.
    if params_data is not None:
        tuning.apply_json_dict(window.shard_params, params_data)
        # Refresh both shards so the new params take effect immediately.
        for shard in window.shards():
            shard.refresh_params()

    window.resize(420, 620)
    window.show()

    # The dev/tuning panel is opt-in via NAIVE_TIMER_TUNE, and --no-panel
    # suppresses it -- e.g. to load a look cleanly without the debug window,
    # or to load one and keep tweaking it (env set, --no-panel absent).
    if tuning.enabled() and not cli.no_panel:
        panel = tuning.TuningPanel(
            window.shards(),
            skip_autoload=cli.json is not None,
        )
        panel.show()

    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())

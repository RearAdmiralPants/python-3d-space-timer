"""Headless smoke tests for the Qt layer.

Why this file exists: every model test passed while the app aborted on startup,
because ``_AlertPlayer`` passed a ``QSoundEffect`` enum where PySide6 6.11
wants an int. Compile-checking ``app.py`` could not catch that. Constructing
the objects can.

Two tiers, because Qt's ``offscreen`` platform has **no OpenGL at all** --
constructing a ``QOpenGLWidget`` under it segfaults, it does not raise:

* Always: things that need no GL context -- the alert player, the shard
  geometry, the text texture image.
* Only with a GL-capable platform: ``MainWindow``, which builds a
  ``ShardWidget``.

To get the GL tier headlessly (in CI, or a cloud container), run under a
virtual X server, which gives real GL via Mesa's software rasteriser:

    xvfb-run -a python -m unittest discover -s tests

Bare ``python -m unittest discover -s tests`` skips the GL tier and says so.
"""

import math
import os
import time
import unittest

# Must be set before any QApplication is created. Respect an existing DISPLAY:
# if the developer has a screen, use it and get the GL tier for free.
if not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    from PySide6.QtWidgets import QApplication
except ImportError:  # pragma: no cover - exercised only in Qt-less envs
    QApplication = None

_app = None


def setUpModule():
    global _app
    if QApplication is not None:
        from PySide6.QtGui import QSurfaceFormat

        from naive_timer.shard import default_surface_format

        QSurfaceFormat.setDefaultFormat(default_surface_format())
        _app = QApplication.instance() or QApplication([])


def _has_opengl() -> bool:
    """True when the running Qt platform can host a QOpenGLWidget.

    The ``offscreen`` plugin cannot, and finds out by crashing the process, so
    this must be checked *before* constructing one.
    """
    if QApplication is None:
        return False
    from PySide6.QtGui import QGuiApplication

    return QGuiApplication.platformName() != "offscreen"


needs_qt = unittest.skipIf(QApplication is None, "PySide6 not installed")


@needs_qt
class NoGlTest(unittest.TestCase):
    """Everything that must hold without a GL context."""

    def test_alert_player_loops_forever(self):
        from naive_timer import app

        if not app._HAVE_AUDIO:
            self.skipTest("QtMultimedia unavailable; alert is visual-only")

        from PySide6.QtMultimedia import QSoundEffect

        player = app._AlertPlayer()
        self.assertEqual(
            player._effect.loopCount(), QSoundEffect.Loop.Infinite.value
        )

    def test_the_pipewire_audio_backend_is_not_in_use(self):
        """Qt's PipeWire audio backend takes the whole process down with it.

        Against PipeWire 1.0.5 (Ubuntu 24.04), Qt 6.11's native PipeWire audio
        backend segfaults or corrupts the heap whenever the sink it is attached
        to disappears -- Bluetooth headphones dropping, an HDMI sink leaving
        with the monitor. Constructing a QSoundEffect is enough to be exposed;
        nothing has to be playing. That is the "segfaults after ~45 minutes"
        bug, and it was reproduced on the first attempt by playing to a null
        sink and unloading it, against 20/20 clean under PulseAudio.

        So ``app`` pins the backend at import. Asserting on the environment is
        the only cheap check: which backend Qt picked is not exposed through
        any API, and actually crashing the process to find out is not a test.

        This fails if you deliberately export ``QT_AUDIO_BACKEND=PipeWire``,
        which is the intended answer -- that is the configuration that crashes.
        """
        import sys

        if not sys.platform.startswith("linux"):
            self.skipTest("the PipeWire backend is Linux-only")

        from naive_timer import app  # noqa: F401  -- the import is what sets it

        backend = os.environ.get("QT_AUDIO_BACKEND")
        self.assertTrue(backend, "app must pin an audio backend, not take Qt's default")
        self.assertNotEqual(backend, "PipeWire")

    def test_shatter_plays_once_and_the_chime_loops(self):
        """An infinitely looping shatter would be unbearable."""
        from naive_timer import app

        if not app._HAVE_AUDIO:
            self.skipTest("QtMultimedia unavailable; alert is visual-only")

        from PySide6.QtMultimedia import QSoundEffect

        player = app._AlertPlayer()
        self.assertEqual(player._shatter.loopCount(), 1)
        self.assertEqual(
            player._effect.loopCount(), QSoundEffect.Loop.Infinite.value
        )

    def test_the_x11_event_struct_matches_the_c_layout(self):
        """A wrong ctypes layout corrupts the message with no error anywhere.

        The first implementation declared the ``data`` union as ``c_int32 * 5``.
        On LP64 it is ``long l[5]``, so every field after the first landed at
        the wrong offset: XSendEvent still returned success and the window
        manager silently ignored the request. Sizes are the only cheap way to
        pin this down, and they need neither a display nor a WM.
        """
        import ctypes

        from naive_timer.app import _XEvent

        # sizeof(XEvent): a union padded to `long pad[24]`.
        self.assertEqual(ctypes.sizeof(_XEvent), 24 * ctypes.sizeof(ctypes.c_long))

        fields = dict(_XEvent._fields_)
        self.assertEqual(fields["data"]._type_, ctypes.c_long)
        self.assertEqual(ctypes.sizeof(fields["data"]), 5 * ctypes.sizeof(ctypes.c_long))

        # data must start where C puts it: after `format` plus its alignment pad.
        self.assertEqual(_XEvent.data.offset, 7 * ctypes.sizeof(ctypes.c_long))

    def test_geometry_is_a_solid_of_flat_shaded_facets(self):
        from naive_timer.shard import (
            _FLOATS_PER_VERTEX as floats_per_vertex,
            _tris_per_wedge,
            _OUTLINE,
            _build_geometry,
        )

        data = _build_geometry()
        self.assertEqual(len(data) % floats_per_vertex, 0)

        vertices = len(data) // floats_per_vertex
        # _build_geometry defaults to front subdivision 0, which is the
        # original single front triangle per wedge: 16 triangles in all.
        self.assertEqual(vertices, 3 * _tris_per_wedge(0) * len(_OUTLINE))

        for i in range(vertices):
            base = i * floats_per_vertex
            nx, ny, nz = data[base + 3 : base + 6]
            self.assertAlmostEqual(
                math.sqrt(nx * nx + ny * ny + nz * nz), 1.0, places=5,
                msg="facet normals must be unit length",
            )
            # The rim projects outside [0,1] on purpose, so the numerals stay
            # inside the bevel; ClampToEdge samples the transparent border.
            u, v = data[base + 6 : base + 8]
            self.assertTrue(-0.5 <= u <= 1.5 and -0.5 <= v <= 1.5)

    def test_hex_colors_parse(self):
        from naive_timer.shard import parse_hex_color

        self.assertEqual(parse_hex_color("#000000"), (0.0, 0.0, 0.0))
        self.assertEqual(parse_hex_color("ffffff"), (1.0, 1.0, 1.0))
        self.assertEqual(parse_hex_color("  #FFFFFF  "), (1.0, 1.0, 1.0))
        self.assertEqual(parse_hex_color("#f00"), (1.0, 0.0, 0.0))

        r, g, b = parse_hex_color("#ff8800")
        self.assertAlmostEqual(r, 1.0)
        self.assertAlmostEqual(g, 136 / 255)
        self.assertAlmostEqual(b, 0.0)

    def test_hex_colors_reject_garbage(self):
        """The entry field sees every keystroke, including half-typed values."""
        from naive_timer.shard import parse_hex_color

        for bad in ("", "#", "#ff", "#fffff", "#gggggg", "12345678", "red"):
            with self.assertRaises(ValueError, msg=f"{bad!r} should not parse"):
                parse_hex_color(bad)

    def test_hex_color_round_trip(self):
        from naive_timer.shard import format_hex_color, parse_hex_color

        for text in ("#000000", "#ffffff", "#ff8800", "#1a2b3c"):
            self.assertEqual(format_hex_color(parse_hex_color(text)), text)

    def test_format_hex_color_clamps(self):
        from naive_timer.shard import format_hex_color

        self.assertEqual(format_hex_color((-1.0, 0.5, 2.0)), "#0080ff")

    def test_numerals_stay_inside_the_bevel(self):
        """Ink must land on the front face, never spill onto the chamfer.

        The invariant is about the *ink*, not the ring: the numerals occupy
        only the central _TEXT_FIT of the texture, so the inset ring may
        legitimately project past u=1.0 into the transparent margin.
        """
        from naive_timer.shard import (
            _BEVEL_INSET, _OUTLINE, _TEXT_FIT, _face_uv,
        )

        # Rightmost edge of the ink, in texture coordinates.
        ink_edge_u = 0.5 + _TEXT_FIT / 2.0

        # Where the inset ring (the front face boundary) lands, at its widest.
        widest_x = max(abs(x) for x, _ in _OUTLINE)
        ring_u, _ = _face_uv(_BEVEL_INSET * widest_x, 0.0)

        self.assertGreater(
            ring_u, ink_edge_u,
            "the front face must extend past the ink, or numerals hit the bevel",
        )

        # And the silhouette rim samples the transparent border, not the text.
        rim_u, _ = _face_uv(widest_x, 0.0)
        self.assertGreater(rim_u, 1.0, "the rim must fall off the texture")

    def test_every_facet_normal_points_outward(self):
        """Two-pass transparency culls by winding, so orientation must hold.

        Measured against the *wedge's* centroid, not the shard's. The shard's
        axis lies inside both of a wedge's radial cut planes, so a cap normal
        is near-perpendicular to the direction from the shard centre and the
        sign of that dot product is meaningless.
        """
        from naive_timer.shard import (
            _FLOATS_PER_VERTEX as stride,
            _tris_per_wedge, _build_geometry,
        )

        data = _build_geometry()
        for i in range(0, len(data) // stride, 3):
            tri = [data[(i + k) * stride : (i + k) * stride + 3] for k in range(3)]
            nx, ny, nz = data[i * stride + 3 : i * stride + 6]
            centre = data[i * stride + 8 : i * stride + 11]
            cx = sum(v[0] for v in tri) / 3.0 - centre[0]
            cy = sum(v[1] for v in tri) / 3.0 - centre[1]
            cz = sum(v[2] for v in tri) / 3.0 - centre[2]
            self.assertGreater(
                nx * cx + ny * cy + nz * cz, 0.0,
                msg=f"triangle {i // 3} is wound inward",
            )

    def test_each_wedge_is_a_closed_solid(self):
        """Open shells look hollow the instant a tumbling piece turns edge-on.

        A closed triangle mesh has every edge shared by exactly two facets.
        Before the radial cut faces existed, a wedge's cut boundary edges
        appeared only once.
        """
        from collections import Counter

        from naive_timer.shard import (
            _FLOATS_PER_VERTEX as stride,
            _tris_per_wedge, _OUTLINE, _build_geometry,
        )

        data = _build_geometry()
        verts_per_wedge = 3 * _tris_per_wedge(0)

        def key(v):  # quantise, so shared corners compare equal
            return tuple(round(c, 5) for c in v)

        for wedge in range(len(_OUTLINE)):
            edges = Counter()
            start = wedge * verts_per_wedge
            for t in range(_tris_per_wedge(0)):
                tri = [
                    key(data[(start + t * 3 + k) * stride : (start + t * 3 + k) * stride + 3])
                    for k in range(3)
                ]
                for a, b in ((0, 1), (1, 2), (2, 0)):
                    edges[frozenset((tri[a], tri[b]))] += 1

            unshared = [e for e, n in edges.items() if n != 2]
            self.assertEqual(
                unshared, [], f"wedge {wedge} is an open shell, not a solid"
            )

    def test_only_the_cut_faces_are_flagged_as_caps(self):
        """The cap flag drives the fragment discard while the shard is whole."""
        from naive_timer.shard import (
            _FLOATS_PER_VERTEX as stride,
            _tris_per_wedge, _OUTLINE, _build_geometry,
        )

        data = _build_geometry()
        caps = [data[v * stride + 17] for v in range(len(data) // stride)]
        # 8 cap triangles of the 16 per wedge, 3 vertices each.
        self.assertEqual(sum(caps), 3 * 8 * len(_OUTLINE))
        self.assertTrue(all(c in (0.0, 1.0) for c in caps))

    def test_a_wedge_shares_one_rigid_body(self):
        """Front face, walls and back of one wedge must tumble together."""
        from naive_timer.shard import (
            _FLOATS_PER_VERTEX as stride,
            _tris_per_wedge, _OUTLINE, _build_geometry,
        )

        data = _build_geometry()
        verts_per_wedge = 3 * _tris_per_wedge(0)

        for wedge in range(len(_OUTLINE)):
            bodies = {
                tuple(data[v * stride + 8 : v * stride + 17])
                for v in range(
                    wedge * verts_per_wedge, (wedge + 1) * verts_per_wedge
                )
            }
            self.assertEqual(
                len(bodies), 1, f"wedge {wedge} has a split rigid body"
            )

    def test_each_wedge_pivots_on_its_own_centroid(self):
        """The pinwheel bug: pieces rotating about the shard's centre.

        Each wedge's pivot must sit inside that wedge, offset from the axis --
        not at the origin, which is what made every crack radiate from the
        middle.
        """
        import math

        from naive_timer.shard import (
            _FLOATS_PER_VERTEX as stride,
            _tris_per_wedge, _OUTLINE, _build_geometry,
        )

        data = _build_geometry()
        verts_per_wedge = 3 * _tris_per_wedge(0)

        centres = []
        for wedge in range(len(_OUTLINE)):
            base = wedge * verts_per_wedge * stride
            centres.append(tuple(data[base + 8 : base + 11]))

        for wedge, (cx, cy, _cz) in enumerate(centres):
            self.assertGreater(
                math.hypot(cx, cy), 0.15,
                f"wedge {wedge} pivots on the shard's axis, not its own",
            )

        # And no two wedges share a pivot.
        self.assertEqual(len(set(centres)), len(_OUTLINE))

    def test_rigid_bodies_are_deterministic(self):
        """The break must look identical on every run, so bugs reproduce."""
        from naive_timer.shard import _build_geometry

        self.assertEqual(_build_geometry(), _build_geometry())

    def test_pieces_are_gone_by_the_declared_clear_time(self):
        """The default clear time stops the draw calls; a piece must not outlive it.

        Integrates the same trajectory the vertex shader uses, at the default
        gravity and clear time. If someone retunes the velocities and a wedge
        lingers, this fails rather than letting a frozen shard sit on screen.
        """
        import math

        from naive_timer.shard import (
            _OUTLINE, _GRAVITY_1G, ShardParams, _build_geometry,
            _FLOATS_PER_VERTEX as stride,
            _tris_per_wedge,
        )

        params = ShardParams()
        data = _build_geometry()
        verts_per_wedge = 3 * _tris_per_wedge(0)
        t = params.shatter_clear_s
        gravity_y = -params.gravity * _GRAVITY_1G

        for wedge in range(len(_OUTLINE)):
            base = wedge * verts_per_wedge * stride
            centre = data[base + 8 : base + 11]
            vel = data[base + 11 : base + 14]

            # Where the pivot ends up. Tumbling only swings vertices about
            # this point, by at most the wedge's radius (~1.2 units).
            x = centre[0] + vel[0] * t
            y = centre[1] + vel[1] * t + 0.5 * gravity_y * t * t
            self.assertGreater(
                math.hypot(x, y), 1.2,
                f"wedge {wedge} pivot still near frame at t={t}s",
            )

    def test_every_wedge_tumbles_and_travels(self):
        """No piece may hang motionless in frame while the others leave.

        Read from the real geometry, and assert only that each piece is
        genuinely moving -- not some tuned magnitude, which changes whenever
        the break is retuned.
        """
        import math

        from naive_timer.shard import (
            _FLOATS_PER_VERTEX as stride,
            _tris_per_wedge, _OUTLINE, _build_geometry,
        )

        data = _build_geometry()
        verts_per_wedge = 3 * _tris_per_wedge(0)

        for wedge in range(len(_OUTLINE)):
            base = wedge * verts_per_wedge * stride
            vel = data[base + 11 : base + 14]
            axis = data[base + 14 : base + 17]

            self.assertGreater(
                math.sqrt(sum(v * v for v in vel)), 0.05,
                f"wedge {wedge} never leaves: no linear velocity",
            )
            self.assertGreater(
                math.sqrt(sum(a * a for a in axis)), 0.05,
                f"wedge {wedge} never turns: no angular velocity",
            )

    def test_wedge_bounds_enclose_every_vertex(self):
        """Each wedge's bounding radius must actually contain its geometry.

        The early-clear check relies on the sphere (centre, radius) enclosing
        the whole wedge for all time. Tumbling only rotates a vertex about the
        centre, so it's enough to prove the radius covers every rest vertex --
        if it does, no rotation can push a vertex outside it.
        """
        import math

        from naive_timer.shard import (
            _FLOATS_PER_VERTEX as stride,
            _OUTLINE, _build_geometry, _wedge_bounds,
        )

        data = _build_geometry()
        bounds = _wedge_bounds(data)
        self.assertEqual(len(bounds), len(_OUTLINE))

        verts_per_wedge = (len(data) // stride) // len(_OUTLINE)
        for wedge, (centre, _vel, radius) in enumerate(bounds):
            self.assertGreater(radius, 0.0, f"wedge {wedge} has zero radius")
            base = wedge * verts_per_wedge * stride
            for v in range(verts_per_wedge):
                off = base + v * stride
                d = math.dist(data[off : off + 3], centre)
                self.assertLessEqual(
                    d, radius + 1e-6,
                    f"wedge {wedge} vertex {v} sits outside its bounding radius",
                )

    def test_frustum_test_distinguishes_on_and_off_screen(self):
        """The bounding-sphere frustum test is the heart of early-clear.

        A sphere parked far to the side is off screen; one sitting at the origin
        (dead centre of a camera that always looks there) is on screen. No GL or
        real geometry needed -- the check is pure math over _wedge_bounds.
        """
        import types

        from naive_timer.shard import ShardParams

        # A minimal stand-in: the check only touches these attributes/methods.
        from naive_timer.shard import ShardWidget

        probe = ShardWidget.__new__(ShardWidget)
        probe.params = ShardParams(shatter_clear_s=60.0, gravity=1.0)
        probe._spin_at_break = 0.0
        probe._shatter_t = 1.0
        probe._elapsed = 1.0
        probe.width = types.MethodType(lambda self: 420, probe)
        probe.height = types.MethodType(lambda self: 620, probe)

        probe._wedge_bounds = [((100.0, 0.0, 0.0), (0.0, 0.0, 0.0), 0.5)]
        self.assertTrue(probe._all_pieces_offscreen(), "far-off sphere is gone")

        probe._wedge_bounds = [((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), 0.5)]
        self.assertFalse(
            probe._all_pieces_offscreen(), "sphere at the focus is on screen"
        )

    def test_early_clear_does_not_latch(self):
        """A piece that re-enters must un-clear, so the shard redraws.

        The check is intentionally not sticky: a full orbit (or a wide sway) can
        sweep the camera back toward a piece that had left the frame. Feeding the
        refresh an off-screen set then an on-screen set must flip the verdict
        back, not hold the stale 'cleared'.
        """
        import types

        from naive_timer.shard import ShardParams, ShardWidget

        probe = ShardWidget.__new__(ShardWidget)
        probe.params = ShardParams(shatter_clear_s=60.0, gravity=1.0)
        probe._spin_at_break = 0.0
        probe._early_cleared = False
        probe._next_clear_check = 0.0
        probe.width = types.MethodType(lambda self: 420, probe)
        probe.height = types.MethodType(lambda self: 620, probe)

        probe._shatter_t = probe._elapsed = 1.0
        probe._wedge_bounds = [((100.0, 0.0, 0.0), (0.0, 0.0, 0.0), 0.5)]
        probe._refresh_early_clear()
        self.assertTrue(probe.pieces_have_cleared)

        probe._shatter_t = probe._elapsed = 1.5
        probe._next_clear_check = 0.0  # force the throttle open
        probe._wedge_bounds = [((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), 0.5)]
        probe._refresh_early_clear()
        self.assertFalse(
            probe.pieces_have_cleared, "verdict must un-latch when a piece returns"
        )

    def test_settings_round_trip_through_json(self):
        """Save then load must reproduce every field, colours included.

        JSON has no tuples, so the colour fields are the ones at risk: they go
        out as arrays and must come back as tuples with their values intact.
        """
        import json

        from naive_timer.shard import ShardParams
        from naive_timer.tuning import apply_json_dict, params_to_json

        original = ShardParams()
        original.glow = 1.75
        original.sway_degrees = 123.0
        original.font_bold = not original.font_bold
        original.font_family = "Courier New"
        original.nebula_color_a = (0.5, 0.25, 0.1)

        restored = ShardParams()
        apply_json_dict(restored, json.loads(params_to_json(original)))

        self.assertEqual(restored, original)
        self.assertIsInstance(restored.nebula_color_a, tuple)

    def test_load_ignores_unknown_and_keeps_defaults_for_absent(self):
        """A file from another build must load what it can, not crash.

        Unknown keys are dropped; fields the file omits keep their current
        value rather than reverting or erroring.
        """
        from naive_timer.shard import ShardParams
        from naive_timer.tuning import apply_json_dict

        params = ShardParams()
        params.glow = 0.5
        apply_json_dict(params, {"glow": 1.2, "no_such_field": 99})

        self.assertEqual(params.glow, 1.2)
        self.assertFalse(hasattr(params, "no_such_field"))
        # A field absent from the dict is left alone.
        self.assertEqual(params.spec_power, ShardParams().spec_power)

    def test_text_image_has_ink_where_the_numerals_are(self):
        from naive_timer.shard import ShardParams, render_text_image

        blank = render_text_image("", ShardParams())
        drawn = render_text_image("00:12:34.56", ShardParams())

        self.assertEqual(blank.size(), drawn.size())

        def ink(image):
            return sum(
                image.pixelColor(x, y).alpha() > 0
                for y in range(0, image.height(), 8)
                for x in range(0, image.width(), 8)
            )

        self.assertEqual(ink(blank), 0, "empty text must draw nothing")
        self.assertGreater(ink(drawn), 0, "numerals must leave ink")


class CurvedFrontTest(unittest.TestCase):
    """The front-face subdivision and bulge sliders.

    The invariants the flat model already had -- watertight wedges, outward
    unit normals -- are the ones most likely to break when the cap curves, so
    they are re-checked at every level rather than only at the default.
    """

    LEVELS = range(6)
    BULGES = (0.0, 0.65, 1.0)

    def _wedges(self, data, subdiv):
        from naive_timer.shard import (
            _FLOATS_PER_VERTEX as stride, _OUTLINE, _tris_per_wedge,
        )

        per = _tris_per_wedge(subdiv)
        for wedge in range(len(_OUTLINE)):
            start = wedge * per * 3
            yield [
                [data[(start + t * 3 + k) * stride:(start + t * 3 + k) * stride + 3]
                 for k in range(3)]
                for t in range(per)
            ]

    def test_level_zero_reproduces_the_original_six_triangles(self):
        """The slider's origin must be a true no-op, not merely a close one.

        If level 0 differed from the shipped model, every existing tuned
        parameter set in default-params.json would render subtly differently
        the moment this feature landed.
        """
        from naive_timer.shard import _build_geometry

        self.assertEqual(_build_geometry(), _build_geometry(0, 0.0))

    def test_every_wedge_stays_a_closed_solid_at_every_level(self):
        """Subdividing the cap subdivides two edges it shares with neighbours.

        The cap's inset edge is shared with the front bevel and its two radial
        chains with the cut faces. Miss either and the wedge becomes an open
        shell with T-junctions -- which is invisible while the shard is whole
        and glaringly hollow the instant it shatters.
        """
        from collections import Counter

        from naive_timer.shard import _build_geometry

        def key(v):
            return tuple(round(c, 5) for c in v)

        for subdiv in self.LEVELS:
            for bulge in self.BULGES:
                data = _build_geometry(subdiv, bulge)
                for wedge, tris in enumerate(self._wedges(data, subdiv)):
                    edges = Counter()
                    for tri in tris:
                        pts = [key(v) for v in tri]
                        for a, b in ((0, 1), (1, 2), (2, 0)):
                            edges[frozenset((pts[a], pts[b]))] += 1
                    self.assertEqual(
                        [e for e, n in edges.items() if n != 2], [],
                        f"wedge {wedge} open at subdiv={subdiv} bulge={bulge}",
                    )

    def test_normals_stay_unit_and_outward_at_every_level(self):
        """A zero-area facet has no normal, and the transparency passes cull
        by winding -- so a degenerate triangle silently corrupts draw order.

        This caught the first implementation: the cut faces were fanned from
        the apex, and at bulge 0 the cap's radial chain is exactly collinear
        with the apex, so every fan triangle had zero area.
        """
        import math

        from naive_timer.shard import (
            _FLOATS_PER_VERTEX as stride, _build_geometry,
        )

        for subdiv in self.LEVELS:
            for bulge in self.BULGES:
                data = _build_geometry(subdiv, bulge)
                for i in range(0, len(data) // stride, 3):
                    base = i * stride
                    nx, ny, nz = data[base + 3:base + 6]
                    self.assertAlmostEqual(
                        math.sqrt(nx * nx + ny * ny + nz * nz), 1.0, places=4,
                        msg=f"subdiv={subdiv} bulge={bulge} tri={i // 3}",
                    )
                    tri = [
                        data[(i + k) * stride:(i + k) * stride + 3]
                        for k in range(3)
                    ]
                    centre = data[base + 8:base + 11]
                    dot = sum(
                        (sum(v[j] for v in tri) / 3.0 - centre[j]) * n
                        for j, n in enumerate((nx, ny, nz))
                    )
                    self.assertGreater(
                        dot, 0.0,
                        f"inward at subdiv={subdiv} bulge={bulge} tri={i // 3}",
                    )

    def test_the_cap_is_smooth_shaded_not_faceted(self):
        """The whole point of the exercise.

        Flat normals would make a subdivided dome trade six big facets for
        thousands of small ones -- a smooth silhouette with a visibly faceted
        highlight. Adjacent front-face triangles must therefore share a normal
        at a shared vertex, which flat shading can never do.
        """
        from naive_timer.shard import (
            _FLOATS_PER_VERTEX as stride, _build_geometry,
        )

        data = _build_geometry(3, 1.0)
        by_position = {}
        for v in range(len(data) // stride):
            base = v * stride
            if data[base + 17]:       # cap face, not the front
                continue
            pos = tuple(round(c, 5) for c in data[base:base + 3])
            by_position.setdefault(pos, set()).add(
                tuple(round(c, 4) for c in data[base + 3:base + 6])
            )

        # Interior cap vertices are shared by six facets; each must agree.
        shared = [n for pos, n in by_position.items() if pos[2] > 0.06]
        self.assertTrue(shared, "no interior cap vertices found")
        self.assertTrue(
            all(len(n) == 1 for n in shared),
            "front-face vertices carry per-facet normals: still flat shaded",
        )

    def test_bulge_raises_the_apex_monotonically(self):
        """The bulge slider must actually curve the face, and pin the rim."""
        from naive_timer.shard import (
            _BEVEL_Z, _FLOATS_PER_VERTEX as stride, _PEAK_Z, _build_geometry,
        )

        peaks = []
        for bulge in (0.0, 0.25, 0.5, 0.75, 1.0):
            data = _build_geometry(3, bulge)
            zs = [data[v * stride + 2] for v in range(len(data) // stride)]
            peaks.append(max(zs))

        self.assertAlmostEqual(peaks[0], _PEAK_Z, places=5)
        for lower, higher in zip(peaks, peaks[1:]):
            self.assertGreater(higher, lower)

        # The inset ring is where the cap meets the bevel; it must not move,
        # or the tangent blend is riding the whole surface up instead of
        # curving it.
        for bulge in (0.0, 1.0):
            data = _build_geometry(3, bulge)
            ring = [
                data[v * stride + 2]
                for v in range(len(data) // stride)
                # z > 0 excludes the *back* inset ring, which sits at the same
                # radius and would otherwise fail this as a false positive.
                if data[v * stride + 2] > 0.0
                and abs(math.hypot(*data[v * stride:v * stride + 2]) - 0.855) < 0.02
            ]
            self.assertTrue(ring)
            for z in ring:
                self.assertAlmostEqual(z, _BEVEL_Z, places=2)

    def test_the_crease_at_the_rim_closes_as_bulge_rises(self):
        """Tangent continuity is the visible payoff: no hard edge where the
        cap meets the chamfer. Measure it as the angle between the outermost
        cap facet and the bevel facet it abuts -- that angle must shrink
        toward zero as bulge goes to 1.
        """
        import math

        from naive_timer.shard import (
            _FLOATS_PER_VERTEX as stride, _build_geometry,
        )

        def facet_normal(tri):
            (ax, ay, az), (bx, by, bz), (cx, cy, cz) = tri
            ux, uy, uz = bx - ax, by - ay, bz - az
            vx, vy, vz = cx - ax, cy - ay, cz - az
            nx, ny, nz = (
                uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx,
            )
            if nz < 0.0:
                nx, ny, nz = -nx, -ny, -nz
            length = math.sqrt(nx * nx + ny * ny + nz * nz) or 1.0
            return (nx / length, ny / length, nz / length)

        def worst_crease(bulge):
            """Largest dihedral across a cap/bevel shared edge.

            Measured from the *geometry*, on facets that genuinely abut --
            comparing a cap normal against every bevel normal in the model
            mixes unrelated wedges and measures the outline's irregularity
            instead of the join.
            """
            data = _build_geometry(4, bulge)
            cap_edges, bevel_edges = {}, {}
            for i in range(0, len(data) // stride, 3):
                base = i * stride
                if data[base + 17]:          # radial cut face, not a surface
                    continue
                tri = [
                    tuple(round(c, 5) for c in
                          data[(i + k) * stride:(i + k) * stride + 3])
                    for k in range(3)
                ]
                zs = [v[2] for v in tri]
                if min(zs) >= 0.0499:        # on or above the inset ring: cap
                    bucket = cap_edges
                elif max(zs) <= 0.0501:      # on or below it: front bevel
                    bucket = bevel_edges
                else:
                    continue
                normal = facet_normal(tri)
                for a, b in ((0, 1), (1, 2), (2, 0)):
                    bucket.setdefault(frozenset((tri[a], tri[b])), []).append(normal)

            joins = [
                (cn, bn)
                for edge, caps in cap_edges.items()
                for cn in caps
                for bn in bevel_edges.get(edge, ())
            ]
            assert joins, "found no cap/bevel shared edges to measure"
            return max(
                math.degrees(math.acos(max(-1.0, min(1.0, sum(
                    a * b for a, b in zip(cn, bn)
                )))))
                for cn, bn in joins
            )

        creased = worst_crease(0.0)
        smooth = worst_crease(1.0)
        self.assertGreater(
            creased, 20.0, "the flat model should have a real crease"
        )
        self.assertLess(
            smooth, creased * 0.5,
            f"bulge=1 must close the crease: {creased:.1f} -> {smooth:.1f} deg",
        )

    def test_triangle_growth_stays_within_reason(self):
        """The ceiling is set by visual return, not by frame rate.

        If someone raises _FRONT_SUBDIV_MAX expecting the slider to reach a
        frame-rate wall, this is the note that says it will not: the app is
        fragment-bound, and level 5 is already past the point where the
        silhouette visibly improves.
        """
        from naive_timer.shard import (
            _FRONT_SUBDIV_MAX, _OUTLINE, _tris_per_wedge,
        )

        self.assertEqual(_tris_per_wedge(0) * len(_OUTLINE), 96)
        top = _tris_per_wedge(_FRONT_SUBDIV_MAX) * len(_OUTLINE)
        self.assertLess(top, 10_000)

    def test_geometry_is_deterministic_at_every_level(self):
        from naive_timer.shard import _build_geometry

        for subdiv in self.LEVELS:
            self.assertEqual(
                _build_geometry(subdiv, 0.7), _build_geometry(subdiv, 0.7)
            )


@needs_qt
class CrystalFrontTest(unittest.TestCase):
    """The crystal field on the front cap.

    The properties worth pinning here are the ones that are invisible in a
    screenshot and expensive to rediscover: that switching crystals off is a
    true no-op, that a wedge is still a closed solid once its cap is a field
    of spikes, and that every wedge still contributes the same number of
    vertices -- ``_wedge_bounds`` slices the vertex buffer into equal blocks
    to find each piece's pivot, so a wedge with its own count would silently
    hand the shatter the wrong centres.

    How it *looks* is not testable and is not tested. That is what
    ``tools/render_still.py`` is for.
    """

    # subdiv, crystal, density, vary, clear, scale, spike
    CASES = (
        (0, 0.04, 0, 0.5, 0.00, 3.0, 0.0),   # plates only
        (2, 0.04, 1, 0.7, 0.00, 3.0, 0.0),
        (2, 0.05, 1, 0.7, 0.45, 4.5, 0.0),   # plates, numerals spared
        (2, 0.00, 1, 0.7, 0.00, 3.0, 1.0),   # spikes only
        (1, 0.06, 2, 1.0, 0.00, 6.0, 0.8),   # both
        (3, 0.02, 0, 0.0, 0.00, 1.0, 0.0),
    )

    def _build(self, case):
        from naive_timer.shard import _build_geometry

        subdiv, crystal, density, vary, clear, scale, spike = case
        return _build_geometry(
            subdiv, 0.65, crystal, density, vary, clear, scale, spike
        )

    def test_crystal_zero_ignores_every_other_crystal_parameter(self):
        """The off switch has to be off, not nearly off.

        crystal is the only one of the four that gates the code path; the
        other three feed the spike maths. If any of them leaked into the
        buffer at crystal = 0, every tuned parameter set already on disk would
        render differently the moment this feature landed -- the same contract
        front_subdiv's level 0 has to keep.
        """
        from naive_timer.shard import _build_geometry

        self.assertEqual(_build_geometry(), _build_geometry(0, 0.0, 0.0))
        self.assertEqual(
            _build_geometry(3, 0.65),
            _build_geometry(3, 0.65, 0.0, 2, 1.0, 0.5),
        )

    def test_geometry_is_deterministic(self):
        """Spike heights and tip positions come from hashes, not random."""
        from naive_timer.shard import _build_geometry

        for case in self.CASES:
            self.assertEqual(self._build(case), self._build(case))

    def test_every_wedge_stays_a_closed_solid(self):
        """Spikes must not open the cap.

        This is the whole reason the tip is the only point that moves. The
        cap's inset chain is shared vertex-for-vertex with the front bevel,
        and its two radial chains with the cut faces and with the neighbouring
        wedge; displacing any of them cracks the solid along every cut. Raising
        a point strictly inside a triangle leaves all three of its edges where
        they were, so the wedge closes for exactly the reason it did before.
        """
        from collections import Counter

        from naive_timer.shard import (
            _FLOATS_PER_VERTEX as stride, _OUTLINE, _build_geometry,
        )

        for case in self.CASES:
            subdiv, crystal, density = case[0], case[1], case[2]
            data = self._build(case)
            verts = len(data) // stride
            per = verts // len(_OUTLINE)
            for wedge in range(len(_OUTLINE)):
                edges = Counter()
                start = wedge * per
                for t in range(per // 3):
                    pts = [
                        tuple(round(c, 5) for c in
                              data[(start + t * 3 + k) * stride:
                                   (start + t * 3 + k) * stride + 3])
                        for k in range(3)
                    ]
                    for a, b in ((0, 1), (1, 2), (2, 0)):
                        edges[frozenset((pts[a], pts[b]))] += 1
                self.assertEqual(
                    [e for e, n in edges.items() if n != 2], [],
                    f"wedge {wedge} open at crystal={crystal} subdiv={subdiv}",
                )

    def test_wedges_contribute_equal_vertex_counts(self):
        """_wedge_bounds divides the buffer by wedge count and trusts it."""
        from naive_timer.shard import (
            _FLOATS_PER_VERTEX as stride, _OUTLINE, _build_geometry,
            _tris_per_wedge,
        )

        for case in self.CASES:
            subdiv, crystal, density, spike = case[0], case[1], case[2], case[6]
            data = self._build(case)
            verts = len(data) // stride
            self.assertEqual(
                verts,
                3 * _tris_per_wedge(subdiv, crystal, density, spike) * len(_OUTLINE),
            )
            per = verts // len(_OUTLINE)
            for wedge in range(len(_OUTLINE)):
                base = wedge * per * stride
                centre = tuple(data[base + 8:base + 11])
                for v in range(per):
                    off = base + v * stride
                    self.assertEqual(tuple(data[off + 8:off + 11]), centre)

    def test_normals_are_unit_and_face_the_camera(self):
        """Flat facets, but the winding contract is unchanged.

        The two-pass transparency in paintGL culls by orientation to draw back
        surfaces before front ones, and the cap is the surface where "outward"
        is unambiguously +z. A spike tips its facets hard -- at crystal 2.0
        some of them are nearly vertical -- but never past the horizon, or the
        cap would start sorting itself into the wrong pass.
        """
        import math

        from naive_timer.shard import (
            _FLOATS_PER_VERTEX as stride, _OUTLINE, _build_geometry,
            _tris_per_wedge,
        )

        for case in self.CASES:
            subdiv, crystal, density, spike = case[0], case[1], case[2], case[6]
            data = self._build(case)
            per = _tris_per_wedge(subdiv, crystal, density, spike)
            cap_tris = 3 * (1 << min(5, subdiv + density)) ** 2
            for wedge in range(len(_OUTLINE)):
                for t in range(cap_tris):
                    for k in range(3):
                        off = ((wedge * per + t) * 3 + k) * stride
                        nx, ny, nz = data[off + 3:off + 6]
                        self.assertAlmostEqual(
                            math.sqrt(nx * nx + ny * ny + nz * nz), 1.0, places=5
                        )
                        self.assertGreater(
                            nz, 0.0,
                            f"cap normal turned away from +z at crystal={crystal}",
                        )

    def test_clear_zone_flattens_the_spike_completely(self):
        """The middle of the face must come back exactly flat, not nearly.

        A pit under a glyph is worse than a crystal on it: the etch reads the
        text through the surface normal, so a tilted facet drags the numeral
        sideways. Inside the clear radius the weight is 0, and a weight of 0
        has to mean zero lift *and* the cap's own smooth normals back --
        the original surface, merely cut into three coplanar pieces.

        Tapering rather than skipping is not a stylistic choice: skipping
        would give the wedges unequal vertex counts, which _wedge_bounds
        cannot survive.
        """
        from naive_timer.shard import _crystal_facets, _crystal_weight

        self.assertEqual(_crystal_weight(0.0, 0.0, 0.45), 0.0)
        self.assertEqual(_crystal_weight(0.9, 0.0, 0.45), 1.0)
        self.assertEqual(_crystal_weight(0.0, 0.0, 0.0), 1.0)  # no clear zone

        points = [(0.0, 0.0, 0.1), (0.1, 0.0, 0.1), (0.0, 0.1, 0.1)]
        normals = [(0.0, 0.0, 1.0)] * 3
        facets = list(_crystal_facets(points, normals, 1.6, 0.7, 0.9, 17))
        self.assertEqual(len(facets), 3)
        for tri, _uvs, tri_normals in facets:
            for vertex in tri:
                self.assertAlmostEqual(vertex[2], 0.1, places=9)
            for normal in tri_normals:
                self.assertEqual(normal, (0.0, 0.0, 1.0))

    def test_jitter_never_inverts_a_cap_triangle(self):
        """The one real hazard of moving the mesh around.

        _add_triangle_smooth fixes winding from the sign of the projected area,
        and paintGL culls by orientation to order its two transparency passes.
        Let a vertex overtake a neighbour and a triangle projects inside out:
        the winding fix then "corrects" it the wrong way and the facet sorts
        into the wrong pass. _CRYSTAL_JITTER_MAX is set below half the vertex
        spacing so this cannot happen; checked past the top of the slider,
        because the margin is the thing being tested, not the setting.
        """
        from naive_timer.shard import (
            _BEVEL_INSET, _OUTLINE, _front_patch, _patch_triangles,
        )

        def patch(i, jitter):
            ax, ay = _OUTLINE[i]
            bx, by = _OUTLINE[(i + 1) % len(_OUTLINE)]
            return _front_patch(
                (ax * _BEVEL_INSET, ay * _BEVEL_INSET, 0.05),
                (bx * _BEVEL_INSET, by * _BEVEL_INSET, 0.05),
                (ax, ay, 0.0), (bx, by, 0.0), 16, 0.65,
                0.045, 4.0, 0.0, jitter,
            )[0]

        for jitter in (0.0, 0.5, 1.0, 1.5):
            for i in range(len(_OUTLINE)):
                rings = patch(i, jitter)
                signs = set()
                for tri in _patch_triangles(rings):
                    a, b, c = (rings[j][k] for j, k in tri)
                    area = ((b[0] - a[0]) * (c[1] - a[1])
                            - (b[1] - a[1]) * (c[0] - a[0]))
                    self.assertNotEqual(area, 0.0, "degenerate cap triangle")
                    signs.add(area > 0.0)
                self.assertEqual(
                    len(signs), 1,
                    f"cap triangle inverted at jitter={jitter} wedge={i}",
                )

    def test_jitter_keeps_neighbouring_wedges_in_agreement(self):
        """Jitter is a continuous function of position, and has to stay one.

        A hash of the ring and slot indices would be the obvious way to jitter
        a mesh and it is the way that tears this one: wedge i numbers its
        shared chain as slot j while its neighbour numbers the same chain slot
        0. Only something keyed on where the vertex *is* gives both the same
        answer -- and it has to be continuous as well as positional, since the
        two wedges reach that position by different arithmetic and land an ulp
        apart.
        """
        from naive_timer.shard import (
            _BEVEL_INSET, _OUTLINE, _ends_fade, _front_patch,
        )

        self.assertEqual(_ends_fade(0.0), 0.0)
        self.assertEqual(_ends_fade(1.0), 0.0)

        def patch(i):
            ax, ay = _OUTLINE[i]
            bx, by = _OUTLINE[(i + 1) % len(_OUTLINE)]
            return _front_patch(
                (ax * _BEVEL_INSET, ay * _BEVEL_INSET, 0.05),
                (bx * _BEVEL_INSET, by * _BEVEL_INSET, 0.05),
                (ax, ay, 0.0), (bx, by, 0.0), 16, 0.65,
                0.045, 4.0, 0.0, 1.0,
            )[0]

        for i in range(len(_OUTLINE)):
            here, nxt = patch(i), patch((i + 1) % len(_OUTLINE))
            for j in range(len(here)):
                for axis in range(3):
                    self.assertAlmostEqual(
                        here[j][j][axis], nxt[j][0][axis], places=9,
                        msg=f"wedges {i}/{i + 1} disagree at ring {j}",
                    )

    def test_panel_shading_moves_no_geometry(self):
        """crystal_panel reassigns normals; it must not touch a position.

        Stated as a test because the temptation when a panel does not read as
        flat enough is to flatten the mesh under it, and that would put
        displacement back on shared vertices for a purely cosmetic reason.
        """
        from naive_timer.shard import _FLOATS_PER_VERTEX as stride
        from naive_timer.shard import _build_geometry

        flat = _build_geometry(2, 0.65, 0.045, 1, 0.5, 0.0, 4.0, 0.0, 0.7, 0.0)
        panelled = _build_geometry(
            2, 0.65, 0.045, 1, 0.5, 0.0, 4.0, 0.0, 0.7, 1.0
        )
        self.assertEqual(len(flat), len(panelled))
        moved = normals_changed = 0
        for v in range(len(flat) // stride):
            off = v * stride
            if flat[off:off + 3] != panelled[off:off + 3]:
                moved += 1
            if flat[off + 3:off + 6] != panelled[off + 3:off + 6]:
                normals_changed += 1
        self.assertEqual(moved, 0, "crystal_panel displaced geometry")
        self.assertGreater(normals_changed, 0, "crystal_panel did nothing")

    def test_the_inset_ring_is_never_displaced(self):
        """The fracture field must die before it reaches the bevel.

        The cap's outer ring is shared with the front bevel, which is drawn as
        one chamfer carrying a single averaged normal, and with the rim that
        draws the silhouette. Let the plates reach it and the chamfer's normal
        becomes the mean of facets pointing in every direction while the
        outline goes visibly ragged. _rim_fade is what prevents that, and it
        has to reach exactly zero, not nearly zero.
        """
        from naive_timer.shard import _BEVEL_INSET, _front_patch, _rim_fade

        self.assertEqual(_rim_fade(1.0), 0.0)
        self.assertEqual(_rim_fade(0.5), 1.0)

        corners = ((-0.95, 0.26), (-0.52, 0.88))
        args = (
            (corners[0][0] * _BEVEL_INSET, corners[0][1] * _BEVEL_INSET, 0.05),
            (corners[1][0] * _BEVEL_INSET, corners[1][1] * _BEVEL_INSET, 0.05),
            (corners[0][0], corners[0][1], 0.0),
            (corners[1][0], corners[1][1], 0.0),
            8, 0.65,
        )
        smooth, _ = _front_patch(*args)
        plated, _ = _front_patch(*args, 0.06, 3.0, 0.0)
        self.assertEqual(smooth[-1], plated[-1])
        self.assertNotEqual(smooth[4], plated[4])  # the interior did move

    def test_neighbouring_wedges_agree_on_their_shared_chain(self):
        """The seam test. A wedge's two radial chains belong to two wedges.

        The plate field is a pure function of planar position for exactly this
        reason: wedge i reaches its shared edge at t = 1 and wedge i+1 reaches
        the same edge at t = 0, by different arithmetic. If the height depended
        on anything but where the point is -- a per-wedge index, an
        accumulator, a parameter-space coordinate -- the two would disagree and
        the shard would open along every cut line.
        """
        from naive_timer.shard import _BEVEL_INSET, _OUTLINE, _front_patch

        def patch(i):
            ax, ay = _OUTLINE[i]
            bx, by = _OUTLINE[(i + 1) % len(_OUTLINE)]
            rings, _ = _front_patch(
                (ax * _BEVEL_INSET, ay * _BEVEL_INSET, 0.05),
                (bx * _BEVEL_INSET, by * _BEVEL_INSET, 0.05),
                (ax, ay, 0.0), (bx, by, 0.0), 8, 0.65,
                0.06, 3.0, 0.0,
            )
            return rings

        for i in range(len(_OUTLINE)):
            here = patch(i)
            nxt = patch((i + 1) % len(_OUTLINE))
            for j in range(len(here)):
                mine = here[j][j]        # my chain along the b edge
                theirs = nxt[j][0]       # their chain along the a edge
                for axis in range(3):
                    self.assertAlmostEqual(
                        mine[axis], theirs[axis], places=9,
                        msg=f"wedges {i}/{i + 1} disagree at ring {j}",
                    )

    def test_spikes_alone_leave_the_base_surface_alone(self):
        """With plates off, the tip is still the only new point.

        Worth keeping separate from the plate cases: spikes displace nothing
        that anything else references, so this mode has no seam exposure at
        all, and if that ever changes it should fail here rather than in a
        screenshot.
        """
        from naive_timer.shard import (
            _FLOATS_PER_VERTEX as stride, _OUTLINE, _build_geometry,
            _tris_per_wedge,
        )

        def cap_vertices(data, subdiv, crystal, density, spike):
            per = _tris_per_wedge(subdiv, crystal, density, spike)
            cap = (1 << (subdiv + density)) ** 2 if (crystal or spike) else \
                (1 << subdiv) ** 2
            cap *= 3 if (crystal or spike) else 1
            found = set()
            for wedge in range(len(_OUTLINE)):
                for t in range(cap):
                    for k in range(3):
                        off = ((wedge * per + t) * 3 + k) * stride
                        found.add(tuple(round(c, 5) for c in data[off:off + 3]))
            return found

        for subdiv, density in ((2, 0), (2, 1), (1, 2)):
            smooth = cap_vertices(
                _build_geometry(subdiv + density, 0.65),
                subdiv + density, 0.0, 0, 0.0,
            )
            spiked = cap_vertices(
                _build_geometry(subdiv, 0.65, 0.0, density, 0.7, 0.0, 3.0, 1.0),
                subdiv, 0.0, density, 1.0,
            )
            self.assertTrue(
                smooth <= spiked,
                f"{len(smooth - spiked)} cap vertices moved at "
                f"subdiv={subdiv} density={density}",
            )


class GlTest(unittest.TestCase):
    """Needs a real GL context. Run under xvfb-run when there's no display."""

    def setUp(self):
        # Must be checked here, not in a class decorator: decorators are
        # evaluated at import time, before setUpModule() has constructed the
        # QApplication, and platformName() is empty until then. Getting this
        # wrong lets the GL test run under `offscreen`, where constructing a
        # QOpenGLWidget segfaults rather than raising.
        if not _has_opengl():
            self.skipTest(
                "no OpenGL on this Qt platform; "
                "run under `xvfb-run -a` for the GL tier"
            )

    def test_main_window_constructs_and_shows(self):
        from naive_timer.app import MainWindow

        window = MainWindow()
        window.show()
        self.assertTrue(window.windowTitle())

    def test_a_modal_dialog_stops_the_scene_repainting(self):
        """The frame timers must not repaint while a modal dialog is up.

        The Save/Load choosers are GTK3 dialogs running in-process, so
        ``QDialog::exec()`` ends in ``gtk_dialog_run()`` -- a nested GLib main
        loop that goes on dispatching Qt's posted paint events. Repainting from
        inside it spent ~80% of the main thread on the HDR chain (plus a
        blocking Mesa DRI3 buffer wait) and left the dialog's own input
        handling a sliver: clicks took seconds to minutes to register while the
        scene behind ran at a full 60 FPS. See docs/HANDOFF.md.

        Animation *time* must keep advancing, or the scene would jump on
        dismissal instead of resuming.
        """
        from PySide6.QtWidgets import QDialog

        from naive_timer.app import MainWindow

        window = MainWindow()
        window.show()
        shard = window.shards()[0]

        repaints = []
        shard.update = lambda *args: repaints.append(1)  # shadows the bound method

        def repaints_from_one_frame() -> int:
            """Repaints caused by a single advance(), and nothing else.

            Measured tightly around the call because the window's own 16 ms
            frame timers advance this same shard whenever the event loop runs,
            which would otherwise be counted here as well.
            """
            before = len(repaints)
            shard.advance(0.016)
            return len(repaints) - before

        self.assertEqual(repaints_from_one_frame(), 1, "the idle scene stopped repainting")

        dialog = QDialog(window)
        dialog.setModal(True)
        dialog.show()
        _app.processEvents()
        self.assertIsNotNone(
            QApplication.activeModalWidget(),
            "Qt no longer reports a shown modal dialog; the guard in "
            "ShardWidget.advance() has nothing to test against",
        )

        before = shard._elapsed
        self.assertEqual(
            repaints_from_one_frame(),
            0,
            "repainted while a modal dialog was open -- that is what starves "
            "the file chooser and makes it look locked up",
        )
        self.assertGreater(
            shard._elapsed, before, "animation time stalled instead of the paint"
        )

        dialog.hide()
        _app.processEvents()
        self.assertEqual(
            repaints_from_one_frame(),
            1,
            "the scene never resumed once the dialog closed",
        )

    def test_startup_opens_no_audio_stream(self):
        """An idle app must hold no client stream on the default sink.

        Constructing a QSoundEffect opens one -- uncorked, never written to --
        and holds it until the object dies. Building the alert player in
        TimerWidget.__init__ therefore parked a stream on the sink from launch
        to exit, in both tabs, for an app that might never ring. Verified with
        `pactl list sink-inputs` against an app that had played nothing.

        That is the unresolved half of the "segfaults after ~45 minutes" bug
        (docs/HANDOFF.md): pinning the PulseAudio backend stopped the crash but
        left the exposure. It also puts the app in the path of every re-route
        the session manager does, and on PipeWire 1.0.5 a re-route can strand a
        stream with no sink and no way back.
        """
        from naive_timer.app import MainWindow

        window = MainWindow()
        self.assertIsNone(
            window.timer_tab._alert,
            "the timer built its sound objects at startup; that parks a "
            "stream on the sink for the life of the process",
        )

    def test_the_alarm_still_gets_its_sound_in_time(self):
        """Lazy is worthless if it is late. The warm-up must actually fire."""
        from naive_timer import app
        from naive_timer.app import ALERT_WARMUP_S, MainWindow

        if not app._HAVE_AUDIO:
            self.skipTest("QtMultimedia unavailable; alert is visual-only")

        window = MainWindow()
        timer = window.timer_tab

        # Well outside the warm-up window: still nothing open.
        timer._cd.configure(ALERT_WARMUP_S + 30.0)
        timer._cd.start()
        timer._sync_alert_player()
        self.assertIsNone(timer._alert, "opened a stream far too early")

        # Inside it: the player exists before the alarm needs it.
        timer._cd.configure(ALERT_WARMUP_S / 2.0)
        timer._cd.start()
        timer._sync_alert_player()
        self.assertIsNotNone(timer._alert, "no sound ready when the alarm fires")

        # Reset must hand the stream back rather than hold it forever.
        timer._on_reset()
        self.assertIsNone(timer._alert, "held the stream open after reset")

    def test_stay_on_top_reaches_the_window_manager(self):
        """The checkbox must change the WM's mind, not merely send a message.

        The first implementation sent a malformed ClientMessage with the wrong
        event mask: XSendEvent returned success, the WM ignored it, and the
        checkbox silently did nothing. Only reading _NET_WM_STATE back from the
        WM catches that -- so this asserts on the state, not on a return value.

        Skips where there is no window manager to ask (xvfb, offscreen,
        Wayland-native), which is also exactly when the checkbox is disabled.
        """
        from PySide6.QtWidgets import QApplication

        from naive_timer.app import MainWindow, stay_on_top

        above = stay_on_top()
        if not above.available:
            self.skipTest(f"stay-on-top unavailable: {above.reason}")

        window = MainWindow()
        window.show()
        QApplication.processEvents()
        window_id = int(window.winId())

        # A _NET_WM_STATE ClientMessage is only honoured once mapped.
        deadline = time.monotonic() + 5.0
        while not window.isVisible() and time.monotonic() < deadline:
            QApplication.processEvents()
            time.sleep(0.02)

        def wait_for_above(want: bool) -> bool:
            end = time.monotonic() + 5.0
            while time.monotonic() < end:
                QApplication.processEvents()
                if above.is_above(window_id) == want:
                    return True
                time.sleep(0.05)
            return False

        self.assertTrue(window._stay_on_top.isEnabled())

        window._stay_on_top.setChecked(True)
        self.assertTrue(
            wait_for_above(True),
            "the WM never added _NET_WM_STATE_ABOVE",
        )

        window._stay_on_top.setChecked(False)
        self.assertTrue(
            wait_for_above(False),
            "the WM never removed _NET_WM_STATE_ABOVE",
        )

        # The EWMH path must leave the window alone: Qt's setWindowFlags path
        # destroyed and never remapped it, which is why this one exists.
        self.assertEqual(int(window.winId()), window_id, "the window was recreated")
        self.assertTrue(window.isVisible())

    def test_shatter_starts_from_the_current_pose(self):
        """The shard must not snap back to its rest angle as it breaks."""
        from naive_timer.shard import ShardWidget

        class Model:
            is_running = True

        shard = ShardWidget(Model())
        shard._spin = 1.0
        shard.set_alarm(True)
        self.assertEqual(shard._spin_at_break, 1.0)

        shard.set_alarm(False)
        self.assertEqual(shard._shatter_t, 0.0, "reset reassembles the shard")

    def test_early_clear_beats_the_timeout_but_not_the_pieces(self):
        """The pieces stop being drawn when they're gone, before the hard cap.

        With a long timeout the wedges leave the frame long before it expires,
        so pieces_have_cleared must trip early -- yet it must never trip while
        the shard is still intact and centred in view.
        """
        from naive_timer.shard import ShardParams, ShardWidget

        class Model:
            is_running = True

        params = ShardParams(gravity=1.0, shatter_clear_s=20.0)
        shard = ShardWidget(Model(), params)
        shard.resize(420, 620)
        shard.set_alarm(True)

        # Freshly broken: the shard fills the frame, nothing has cleared.
        shard.advance(0.05)
        self.assertFalse(shard.pieces_have_cleared)

        cleared_at = None
        t = 0.05
        while t < params.shatter_clear_s:
            shard.advance(1 / 60.0)
            t += 1 / 60.0
            if shard.pieces_have_cleared:
                cleared_at = t
                break

        self.assertIsNotNone(cleared_at, "pieces never cleared before the cap")
        self.assertLess(
            cleared_at, params.shatter_clear_s - 2.0,
            "early clear should beat the 20s cap by a wide margin",
        )

    def test_camera_never_swings_behind_the_numerals(self):
        """This is a timer. The readout must stay legible at every phase.

        A full 360 degree orbit leaves the front face edge-on or mirrored for
        half of each cycle, so the camera sways instead. Sampled across a whole
        period, the front face (+z in world space) must stay well toward the
        camera.
        """
        import math

        from PySide6.QtGui import QVector3D

        from naive_timer.shard import ShardWidget

        class Model:
            is_running = False

        shard = ShardWidget(Model())
        period = 2 * math.pi / shard.params.orbit_speed

        worst = 1.0
        for i in range(64):
            shard._elapsed = period * i / 64
            eye, _right, _up, _forward = shard.camera()
            facing = QVector3D.dotProduct(QVector3D(0, 0, 1), eye.normalized())
            worst = min(worst, facing)

        # cos(60 deg) = 0.5. Anything less and the readout is badly raked.
        self.assertGreater(
            worst, 0.5, f"camera rakes the numerals too far (facing {worst:.3f})"
        )

    def test_sky_and_shard_share_one_camera(self):
        """If the two passes disagree, the backdrop slides against the glass."""
        import math

        from naive_timer.shard import ShardWidget

        class Model:
            is_running = False

        shard = ShardWidget(Model())
        shard._elapsed = 4.0
        eye, right, up, forward = shard.camera()

        # An orthonormal, right-handed basis pointing at the origin.
        self.assertAlmostEqual(forward.length(), 1.0, places=5)
        self.assertAlmostEqual(right.length(), 1.0, places=5)
        self.assertAlmostEqual(up.length(), 1.0, places=5)
        for a, b in ((right, up), (right, forward), (up, forward)):
            from PySide6.QtGui import QVector3D

            self.assertAlmostEqual(QVector3D.dotProduct(a, b), 0.0, places=5)

        # forward really does point from the eye at the shard
        from PySide6.QtGui import QVector3D

        expected = (QVector3D(0, 0, 0) - eye).normalized()
        self.assertAlmostEqual(QVector3D.dotProduct(forward, expected), 1.0, places=5)

    def test_etched_numerals_have_no_2x2_quad_structure(self):
        """The engraving must not be stair-stepped by screen-space derivatives.

        dFdx/dFdy are evaluated once per 2x2 pixel quad. If the coverage
        gradient is taken that way, pixels inside a quad share a value and jump
        at quad boundaries -- visible as speckled, stair-stepped glyph edges.
        A texture-space central difference has no such structure.

        Measured as: mean |luminance difference| between horizontally adjacent
        pixels, split by whether the pair straddles a quad boundary. Equal means
        no quad structure.
        """
        from PySide6.QtWidgets import QApplication

        from naive_timer.shard import ShardParams, ShardWidget

        class Model:
            is_running = False

        params = ShardParams()
        params.etch = 1.0
        params.glow = 0.2
        params.orbit_speed = 0.0     # hold the camera still
        params.orbit_radius = 1.55   # close in, so the glyphs are magnified
        params.orbit_height = 0.0

        shard = ShardWidget(Model(), params)
        shard.resize(360, 360)
        shard.set_text("00:00:00.00")
        shard.show()
        QApplication.processEvents()

        shard.makeCurrent()
        shard.paintGL()
        image = shard.grabFramebuffer()

        def luminance(x, y):
            r, g, b = image.pixelColor(x, y).getRgb()[:3]
            return 0.2126 * r + 0.7152 * g + 0.0722 * b

        inside, across = [], []
        for y in range(140, 240, 3):
            previous = luminance(100, y)
            for x in range(101, 260):
                current = luminance(x, y)
                delta = abs(current - previous)
                # 2x2 quads align to even x: (even, odd) is inside one quad.
                (inside if (x - 1) % 2 == 0 else across).append(delta)
                previous = current

        mean_inside = sum(inside) / len(inside)
        mean_across = sum(across) / len(across)
        if mean_inside < 1e-6:
            self.fail("degenerate render: no variation along the scanlines")

        ratio = mean_across / mean_inside
        self.assertLess(
            ratio, 1.6,
            f"etched edges show 2x2 quad structure (ratio {ratio:.2f}); "
            "the coverage gradient is coming from dFdx/dFdy again",
        )

    def test_light_color_reaches_the_shader(self):
        """Render twice under different lights; the pixels must differ.

        Uniform wiring fails silently in PySide6 -- a float bound to the int
        overload of setUniformValue truncates to 0 with no error. Only looking
        at the framebuffer catches that.
        """
        from PySide6.QtWidgets import QApplication

        from naive_timer.shard import ShardWidget, parse_hex_color

        class Model:
            is_running = False

        shard = ShardWidget(Model())
        shard.resize(160, 160)
        shard.set_text("00:00:00.00")
        shard.show()
        QApplication.processEvents()

        def render(hex_color):
            shard.params.light_color = parse_hex_color(hex_color)
            shard.makeCurrent()
            shard.paintGL()
            return shard.grabFramebuffer()

        white = render("#ffffff")
        warm = render("#ff5522")

        differing = sum(
            1
            for y in range(0, white.height(), 4)
            for x in range(0, white.width(), 4)
            if max(
                abs(
                    white.pixelColor(x, y).getRgb()[i]
                    - warm.pixelColor(x, y).getRgb()[i]
                )
                for i in range(3)
            )
            > 8
        )
        self.assertGreater(differing, 20, "light colour never reached the shader")

    def test_sky_still_draws_after_the_pieces_have_cleared(self):
        """paintGL returns early once the shard is gone. The sky must precede
        that return, or the backdrop vanishes for the rest of the alert."""
        from PySide6.QtWidgets import QApplication

        from naive_timer.shard import ShardWidget

        class Model:
            is_running = False

        shard = ShardWidget(Model())
        shard.resize(160, 160)
        shard.set_text("00:00:00.00")
        shard.show()
        QApplication.processEvents()

        shard.set_alarm(True)
        shard._shatter_t = 60.0        # long past _SHATTER_CLEAR_S
        shard._elapsed = 3.0
        self.assertTrue(shard.pieces_have_cleared)

        shard.makeCurrent()
        shard.paintGL()
        image = shard.grabFramebuffer()

        lit = sum(
            1
            for y in range(0, image.height(), 3)
            for x in range(0, image.width(), 3)
            if max(image.pixelColor(x, y).getRgb()[:3]) > 14
        )
        self.assertGreater(lit, 50, "the sky went dark when the shard left")

    def test_idle_rotation_ignores_whether_the_model_runs(self):
        """Speeding up on start drew the eye away from the numerals."""
        from naive_timer.shard import ShardWidget

        class Model:
            is_running = False

        stopped = ShardWidget(Model())
        stopped.advance(1.0)

        Model.is_running = True
        running = ShardWidget(Model())
        running.advance(1.0)

        self.assertEqual(stopped._spin, running._spin)


if __name__ == "__main__":
    unittest.main()

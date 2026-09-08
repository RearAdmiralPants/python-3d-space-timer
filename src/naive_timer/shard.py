"""OpenGL glass-shard view: the timer text as a texture on an angled facet.

Replaces the old ``SpinnerWidget`` and keeps its interface (``advance``,
``set_alarm``) so it drops into both tabs unchanged. The one addition is
``set_text``, since the numerals now live on the shard rather than in a
``QLabel`` beneath it.

The models stay UI-free: this widget is handed a formatted string and renders
it. Nothing here knows what a stopwatch is.

Shaders live in ``shaders/*.{vert,frag}`` and are **hot-reloaded** on save --
edit, save, watch the running app change. A shader that fails to compile prints
its error and leaves the previous program in place.

Launch with ``NAIVE_TIMER_TUNE=1`` to get live sliders for every uniform.
"""

from __future__ import annotations

import array
import math
from dataclasses import dataclass, field
from pathlib import Path

from PySide6.QtCore import QFileSystemWatcher, QSize, Qt
from PySide6.QtGui import (
    QColor,
    QFont,
    QImage,
    QMatrix4x4,
    QPainter,
    QSurfaceFormat,
    QVector2D,
    QVector3D,
)
from PySide6.QtOpenGL import (
    QOpenGLFramebufferObject,
    QOpenGLFramebufferObjectFormat,
    QOpenGLBuffer,
    QOpenGLShader,
    QOpenGLShaderProgram,
    QOpenGLTexture,
    QOpenGLVertexArrayObject,
)
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QtWidgets import QApplication

_SHADER_DIR = Path(__file__).parent / "shaders"

# GL enums we need but QOpenGLFunctions does not re-export.
_GL_FLOAT = 0x1406
_GL_TRIANGLES = 0x0004
_GL_DEPTH_TEST = 0x0B71
_GL_BLEND = 0x0BE2
_GL_SRC_ALPHA = 0x0302
_GL_ONE_MINUS_SRC_ALPHA = 0x0303
_GL_COLOR_BUFFER_BIT = 0x4000
_GL_DEPTH_BUFFER_BIT = 0x0100
_GL_CULL_FACE = 0x0B44
_GL_FRONT = 0x0404
_GL_BACK = 0x0405
_GL_FALSE = 0
_GL_TRUE = 1
_GL_FRAMEBUFFER = 0x8D40
_GL_COLOR_ATTACHMENT0 = 0x8CE0
_GL_TEXTURE_CUBE_MAP_POSITIVE_X = 0x8515  # the other five faces follow it
_GL_TEXTURE_CUBE_MAP_SEAMLESS = 0x884F
_GL_TEXTURE0 = 0x84C0
_GL_TEXTURE1 = 0x84C1
_GL_TEXTURE2 = 0x84C2
_GL_TEXTURE_2D = 0x0DE1
_GL_TEXTURE_MIN_FILTER = 0x2801
_GL_TEXTURE_MAG_FILTER = 0x2800
_GL_TEXTURE_WRAP_S = 0x2802
_GL_TEXTURE_WRAP_T = 0x2803
_GL_LINEAR = 0x2601
_GL_CLAMP_TO_EDGE = 0x812F

# Half-float, not 8-bit. This is the format that makes the whole lighting chain
# work: an 8-bit target clamps every highlight to 1.0 at write time, so a
# specular carrying radiance 20 and one carrying radiance 1.1 land in the buffer
# as the same white and the bright-pass can no longer tell them apart. RGBA16F
# keeps the real value all the way to the tonemap.
_GL_RGBA16F = 0x881A

# Multisample count for the offscreen scene target. The widget's default
# framebuffer gets its MSAA from QSurfaceFormat (see default_surface_format),
# but we no longer draw the scene there -- so the samples have to be requested
# again here or the geometry's edges would come back aliased.
_SCENE_SAMPLES = 4

# The bloom chain runs at a *fraction* of the scene's resolution, and the
# fraction is chosen from the window size rather than fixed.
#
# A fixed half was measured and is wrong at the top end. Glare and flare are
# low-frequency by construction -- they are the output of a wide blur -- so
# their cost scales with resolution while their *content* does not. At 420x620
# the whole post chain costs 0.84 ms; at half of 4K it costs 15.3 ms, which is
# the entire frame budget before the scene has drawn a single triangle.
#
# So: halve until the chain is no wider than this, which caps the cost of every
# pass downstream of the bright-pass regardless of how large the window gets.
# Small windows still get the full half-resolution chain and stay crisp.
_BLOOM_MAX_WIDTH = 960
_BLOOM_MAX_DIVISOR = 8


def _bloom_divisor(width: int) -> int:
    """Downscale factor for the glare chain at a given scene width."""
    divisor = 2
    while width // divisor > _BLOOM_MAX_WIDTH and divisor < _BLOOM_MAX_DIVISOR:
        divisor *= 2
    return divisor

# Separable blur iterations, and the texel step each one takes. Growing the step
# geometrically is what produces a wide, soft falloff from a small kernel: three
# passes at 1/2/4 texels reach far further than one nine-tap pass ever could,
# for six cheap draws over a half-resolution buffer.
_BLOOM_STEPS = (1.0, 2.0, 4.0)

# Post-process fragment stages, each paired with sky.vert's fullscreen triangle
# and each hot-reloaded like everything else. The uniforms listed are cached by
# location for the same reason the shard's are -- see _load_program.
_POST_STAGES = {
    "post_bright": ("uScene", "uTexel", "uExposure", "uThreshold"),
    "post_blur": ("uSource", "uDir"),
    "post_flare": ("uSource", "uTexel", "uStreak", "uLength", "uThreshold"),
    "post_composite": (
        "uScene", "uBloom", "uFlare",
        "uExposure", "uBloomStrength", "uFlareStrength", "uRolloff",
    ),
}

# Half-resolution scratch buffers. Three, not two: the bloom blur ping-pongs
# between the first two and would overwrite the bright-pass output, but the
# flare needs that same output as *its* source. The third holds the flare while
# the bloom chain runs.
_SCRATCH_BUFFERS = 3

# Cube face size for the baked nebula. The nebula is low-frequency by
# construction -- a domain-warped fbm thresholded into wisps -- so it survives
# this comfortably; the stars, which do not, are never baked. Six faces of
# RG16F at this size is 3 MB.
_NEBULA_CUBE_SIZE = 512

# Camera bases for the six cube faces, as (forward, right, up).
#
# These are not arbitrary: they reproduce OpenGL's own cubemap face convention
# (GL spec, the (ma, sc, tc) table), so that the direction this pass rasterises
# into a texel is exactly the direction a runtime texture(cube, dir) lookup
# resolves back to that texel. Get one axis sign wrong and the nebula comes
# back mirrored across a face boundary. Rendered with a 90 degree FOV and
# aspect 1, which is what makes the six frusta tile the sphere without overlap.
_CUBE_FACE_BASES = (
    ((1.0, 0.0, 0.0), (0.0, 0.0, -1.0), (0.0, -1.0, 0.0)),   # +X
    ((-1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, -1.0, 0.0)),   # -X
    ((0.0, 1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),     # +Y
    ((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, -1.0)),   # -Y
    ((0.0, 0.0, 1.0), (1.0, 0.0, 0.0), (0.0, -1.0, 0.0)),    # +Z
    ((0.0, 0.0, -1.0), (-1.0, 0.0, 0.0), (0.0, -1.0, 0.0)),  # -Z
)

# Texture the numerals are drawn into. Wide, because the readout is wide, and
# oversized because the glyphs get magnified across the face of the shard.
_TEX_W, _TEX_H = 1536, 512
_TEX_PT = 192

# Fraction of the texture width the numerals are allowed to occupy. The rest
# is transparent margin, which is what the bevel and side walls sample.
_TEXT_FIT = 0.92

# Irregular outline, traced counter-clockwise. Deliberately not symmetric --
# a regular polygon reads as a gem, not as a shard.
_OUTLINE = [
    (-0.95, 0.26),
    (-0.52, 0.88),
    (0.70, 0.64),
    (0.97, -0.30),
    (0.12, -0.92),
    (-0.82, -0.54),
]

# The shard is a solid, not a flat fan. Cross-section through one edge:
#
#            apex                     <- _PEAK_Z, front face peak
#           /    \
#      inset      inset               <- _BEVEL_Z, at _BEVEL_INSET scale
#     /                \
#   rim                rim            <- z = 0, the silhouette edge
#    |                  |             <- side wall, _THICKNESS deep
#   rim                rim            <- z = -_THICKNESS
#     \                /
#      inset      inset               <- back bevel, shallower than the front
#           \    /
#          back apex
#
# The bevel is what actually reads as "glass": it catches a highlight along
# the silhouette, which a zero-thickness polygon can never do.
_PEAK_Z = 0.14
# Depth of the straight middle slab, between the two bevels. Keep this thin:
# the bevels are what read as glass, and a fat middle makes the shard look
# like a slab of plastic.
_THICKNESS = 0.09
_BEVEL_INSET = 0.90
_BEVEL_Z = 0.05
_BACK_BEVEL_Z = 0.03
_BACK_PEAK_Z = 0.05

# Centre of mass of the whole shard.
_CENTER = (0.0, 0.0, -_THICKNESS / 2.0)

# pos(3) normal(3) uv(2) pieceCenter(3) pieceVel(3) pieceAxis(3) cap(1)
_FLOATS_PER_VERTEX = 18

# Front-face subdivision. Level n splits each wedge's front triangle into a fan
# of n**2... see _tris_per_wedge below. Level 0 is exactly the original single
# triangle per wedge, so the slider's origin is a true no-op.
#
# The ceiling is set by where the silhouette stops visibly improving, NOT by
# frame rate: this app is fragment-bound (nebula cubemap, glass refraction), so
# even level 5 -- ~6k front triangles -- costs no measurable frame time. A
# slider whose useful range is the first 0.5% of its travel is a bad slider.
_FRONT_SUBDIV_MAX = 5

# Crystal density. front_subdiv + crystal_density is clamped to
# _CRYSTAL_SUBDIV_MAX, which is deliberately the same ceiling as the smooth
# cap: at level 5 the cap is 32 rings, so the crystal field is 1024 spikes per
# wedge -- 18k flat facets across the shard. The limit here is *build* time,
# not frame time. _build_geometry is Python appending floats one triangle at a
# time; a level higher would quadruple that into a visible stall every time
# the slider moves, to produce facets small enough to alias, which is the one
# thing the whole feature is trying not to do.
_CRYSTAL_SUBDIV_MAX = 5
_CRYSTAL_DENSITY_MAX = 2

# How far a spike's tip may wander off its facet's centroid, as a fraction of
# the barycentric weights. Zero puts every tip dead centre, and a field of
# perfectly centred spikes on a regular ring grid reads as a machined pattern
# rather than as crystal. Held at a constant instead of exposed: it is a
# texture, not a look, and it has no useful setting other than "some".
_CRYSTAL_TIP_JITTER = 1.2

# Where the fracture field fades out along the radius, as a fraction of the
# way from apex to inset ring. The cap's outer edge is shared with the front
# bevel, which is drawn as a single flat chamfer with one averaged normal; let
# the plates reach it and the chamfer's normal becomes the mean of a set of
# facets pointing everywhere, and the silhouette goes ragged into the bargain.
# Fading the last sixth costs a band of smooth glass just inside the bevel and
# buys a clean outline.
_CRYSTAL_RIM_FADE = 0.84

# Domain warp applied to the plate lookup, in cells. Without it a jittered
# grid still reads as a grid: the cells come out much the same size and much
# the same roundness, and six of them in a row is a pattern the eye finds
# instantly. Bending the lookup position first stretches some cells into
# slivers and lets others swell, which is what real fracture looks like. Past
# about 0.6 the warp folds the field back on itself and cells start to
# reappear in two places at once.
_CRYSTAL_WARP = 0.45

# Mesh jitter frequencies, in cycles per scene unit. Two octaves rather than
# one: the low one stretches whole regions of the tessellation, the high one
# decorrelates neighbouring vertices. One octave of either alone reads as
# uniform noise on a regular grid, which is still a regular grid.
_CRYSTAL_JITTER_LOW = 3.0
_CRYSTAL_JITTER_HIGH = 11.0

# Mesh jitter amplitude at crystal_jitter = 1, as a fraction of the local
# vertex spacing. Below 0.5 a vertex cannot cross its neighbour, so no triangle
# can invert; the margin below that is for the two octaves landing in phase.
_CRYSTAL_JITTER_MAX = 0.42

# Plate tilt, as a height change across one cell. Zero gives flat slabs at
# random heights, which reads as a staircase; this puts each plate on its own
# slope so adjacent ones meet at a genuine angle.
_CRYSTAL_TILT = 0.45

# Aspect of the crystal_clear ellipse. The numerals are drawn into a 3:1
# texture that _face_uv projects onto a square patch of the face, so the glyphs
# occupy the full width but only the middle third of the height; a circular
# clear zone big enough to spare them would spare most of the cap with it.
_CRYSTAL_CLEAR_ASPECT = 0.34
# Planar distance over which the clear zone ramps back up to full crystal.
_CRYSTAL_CLEAR_FEATHER = 0.35

# Salt for the per-spike hashes, and a per-wedge stride so wedge 3's tenth
# triangle does not draw the same numbers as wedge 0's.
_CRYSTAL_SALT = 37
_CRYSTAL_STRIDE = 1013


def _rings(subdiv: int, ceiling: int = _FRONT_SUBDIV_MAX) -> int:
    """Radial rings between the apex and the inset ring. Level 0 -> 1 ring."""
    return 1 << max(0, min(ceiling, int(subdiv)))


def _cap_level(subdiv: int, crystal: float, density: int,
               spike: float = 0.0) -> int:
    """Subdivision level the cap is actually built at.

    Crystals are one-per-cap-triangle, so density is just more cap. Folding it
    in here rather than splitting triangles afterwards is what keeps the whole
    feature confined to the cap emission loop: the finer chain along the inset
    edge flows into the bevel strip and the radial cut faces exactly as the
    coarser one did, so the solid stays closed with no seam work at all.
    """
    if crystal <= 0.0 and spike <= 0.0:
        return max(0, min(_FRONT_SUBDIV_MAX, int(subdiv)))
    return max(0, min(_CRYSTAL_SUBDIV_MAX, int(subdiv) + max(0, int(density))))


def _tris_per_wedge(subdiv: int, crystal: float = 0.0, density: int = 0,
                    spike: float = 0.0) -> int:
    """Triangles one wedge contributes, at a given front subdivision.

    n**2 front patch + (n+1) front bevel + 2 wall + 2 back bevel + 1 back face
    + 2*(n+3) radial caps.  n=1 gives 16, the original hand-counted total.

    With crystals the patch term triples -- every cap triangle is emitted as
    three facets around a centre tip -- and n comes from _cap_level rather than
    front_subdiv alone. It triples whether or not crystal_spike is raising the
    tip, so that the buffer layout depends on one predicate instead of two;
    with the tip at zero height the three facets are coplanar and the result is
    the flat-shaded triangle you would have emitted anyway. A few thousand
    redundant triangles on a fragment-bound renderer is a cheaper thing to
    spend than a second way for the vertex count to be wrong.

    The count stays identical across wedges either way, which _wedge_bounds
    depends on: it slices the buffer into equal blocks.
    """
    n = _rings(_cap_level(subdiv, crystal, density, spike), _CRYSTAL_SUBDIV_MAX)
    faces = n * n * (3 if (crystal > 0.0 or spike > 0.0) else 1)
    return faces + 3 * n + 12

# Vertical field of view, degrees. Shared by the projection and by the sky's
# per-pixel ray reconstruction -- they must agree or the backdrop will not line
# up with the geometry.
_FOV_DEGREES = 38.0

# Downward acceleration (scene units/s^2) that ShardParams.gravity == 1.0 maps
# to -- the value the fall was originally tuned against. The slider scales this,
# so gravity=2.0 is twice as heavy and gravity=0.0 lets the pieces drift flat.
_GRAVITY_1G = 0.32


def parse_hex_color(text: str) -> tuple:
    """``"#ff8800"`` or ``"ff8800"`` -> ``(1.0, 0.533, 0.0)``.

    Raises ``ValueError`` on anything else. Shorthand (``#f80``) is accepted
    because people type it.
    """
    value = text.strip().lstrip("#")
    if len(value) == 3:
        value = "".join(c * 2 for c in value)
    if len(value) != 6:
        raise ValueError(f"expected RRGGBB or RGB, got {text!r}")
    try:
        channels = tuple(int(value[i : i + 2], 16) for i in (0, 2, 4))
    except ValueError:
        raise ValueError(f"{text!r} is not hexadecimal") from None
    return tuple(c / 255.0 for c in channels)


def format_hex_color(rgb) -> str:
    """``(1.0, 0.533, 0.0)`` -> ``"#ff8800"``. Inverse of parse_hex_color."""
    return "#" + "".join(
        f"{max(0, min(255, round(c * 255.0))):02x}" for c in rgb
    )


@dataclass
class ShardParams:
    """Everything the fragment shader can be tuned by."""

    light_x: float = 2.4
    light_y: float = 2.2
    light_z: float = 2.0
    # Scalar radiance multiplier on everything the light drives. 1.0 is the
    # look every saved preset was tuned against, so it must stay the default.
    #
    # It is *not* a blend toward white: the whitening you get by pushing it
    # comes out of the per-channel tonemap, which saturates the strongest
    # channel first. See the roll-off note in shard.frag.
    light_intensity: float = 1.0
    spec_power: float = 48.0
    spec_strength: float = 0.85
    fresnel: float = 0.55
    glow: float = 0.90
    etch: float = 0.0
    # Tilt applied to the normal per unit of coverage gradient. The gradient is
    # now a texture-space central difference (magnitude up to ~0.5), where it
    # used to be a screen-space derivative (far smaller), so this is a much
    # smaller number than the old hardcoded 18.0.
    etch_depth: float = 3.0
    base_alpha: float = 0.55
    glass_color: tuple = (0.16, 0.34, 0.52)
    text_color: tuple = (0.75, 0.93, 1.00)
    light_color: tuple = (1.00, 1.00, 1.00)
    font_family: str = "monospace"
    font_bold: bool = True

    # Shatter physics. gravity is in g: 0 leaves the pieces drifting flat, 1 is
    # the tuned fall, 2 is heavy. shatter_clear_s is how long the tumbling
    # wedges keep being drawn before the shard stops rasterising -- and, on the
    # Stopwatch, how long the Reset shatter runs before it reassembles at zero.
    gravity: float = 1.0
    shatter_clear_s: float = 5.5

    # Front-face curvature. Unlike everything above, these two rebuild the
    # vertex buffer rather than setting a uniform -- see _geometry_dirty.
    # front_subdiv is an integer level 0..5 (triangle count), front_bulge is
    # how far the cap swells from flat toward the tangent-continuous dome.
    # They are deliberately independent: a smooth shallow curve and a coarse
    # steep one are both things you might want to look at.
    front_subdiv: float = 3.0
    front_bulge: float = 0.65

    # Crystal facets on the front cap. Like the curvature pair above these
    # retessellate rather than set a uniform -- see _rebuild_geometry.
    #
    # front_subdiv/front_bulge exist to make the cap *smooth*; this is the
    # opposite instrument. Every cap triangle is raised into a three-sided
    # spike and flat-shaded, so the face becomes a field of hard-edged facets
    # that each catch the light at their own angle as the shard turns. The two
    # are not exclusive: subdivision is what sets how many crystals there are,
    # and bulge still shapes the dome they stand on.
    #
    # crystal is the fracture depth in scene units: how far the plates ride up
    # and down from the smooth cap. The cap's whole z-range is only about 0.09,
    # so the useful range is small -- past ~0.06 the crust stops reading as ice
    # under strain and starts reading as scrap metal. 0.0 must stay a
    # byte-identical no-op (the level-0 buffer is pinned by test).
    crystal: float = 0.0
    # How much of a plate shades as one panel, 0 = every mesh triangle is its
    # own facet, 1 = the Voronoi cell is. Fracture walls keep their own normal
    # either way. This is the knob for facet *size* variance, where
    # crystal_jitter is the one for facet shape.
    crystal_panel: float = 0.0
    # Irregularity of the tessellation itself, 0 = the plain ring-and-slot
    # grid, 1 = as far as a vertex can move without overtaking a neighbour.
    # The plate field randomises the facets' *heights*; this randomises their
    # shapes. Without it the creases still run in concentric rings and the
    # crust reads as quilted however chaotic the heights are.
    crystal_jitter: float = 0.0
    # Plates per scene unit. Low gives a few broad slabs, high a gravel of
    # small ones. This is the knob that decides how big a fragment reads as;
    # crystal_density only decides how finely each one is triangulated.
    crystal_scale: float = 3.0
    # Spikes, the other shape this can make: raise a tip inside every cap
    # triangle, height as a multiple of that triangle's own size. 0 leaves the
    # plates alone. The two compose -- spikes stand perpendicular to whatever
    # plate they are on -- but they are different objects: plates are a
    # fractured connected crust, spikes are a bed of nails.
    crystal_spike: float = 0.0
    # Extra cap subdivision levels used only when crystal > 0: one crystal per
    # cap triangle, so this is the density knob. Added on top of front_subdiv
    # and clamped to _CRYSTAL_SUBDIV_MAX -- see _CRYSTAL_DENSITY_MAX for why
    # the ceiling is where it is.
    crystal_density: float = 1.0
    # Spread of the per-spike height (crystal_spike only), 0 = every spike the
    # same, 1 = anything from a pit to two and a half times nominal.
    crystal_vary: float = 0.5
    # Radius of a flat, smooth-shaded ellipse left in the middle of the face,
    # in planar units, with the numerals' own aspect. 0 puts crystals
    # everywhere -- including straight through the readout, which is the point
    # at which you find out whether you can still read it.
    crystal_clear: float = 0.0

    # Camera. It sways back and forth across the front of the shard, always
    # looking at it -- rather than orbiting all the way round, which would
    # leave the numerals edge-on or mirrored for half of every cycle. This is a
    # timer: the readout has to stay readable.
    #
    # Set sway_degrees to 180 for a full orbit, if you want the sculpture
    # rather than the clock.
    orbit_speed: float = 0.22      # radians/sec of *phase*, not of angle
    orbit_radius: float = 3.2
    orbit_height: float = 0.55
    sway_degrees: float = 30.0     # half-width of the arc, either side of front
    orbit_bob: float = 0.30        # how far the eye rises and falls

    # The shard's own idle rotation, on top of the orbit. Zero by default now
    # that the camera moves -- two rotations at once is a lot of motion.
    idle_spin: float = 0.0

    # Tonemap and glare. These are frame-wide, not shard-wide: everything the
    # scene draws goes through them, which is the point of doing the tonemap
    # once at the end rather than per-object.
    #
    # exposure is a plain stop adjustment on the whole frame. rolloff is the
    # tonemap's shoulder -- the curve clips at radiance 1/(1 - rolloff), so 0.55
    # (the value this was hardcoded to inside shard.frag) runs out of headroom
    # at 2.22. Raise it when raising light_intensity.
    exposure: float = 1.0
    rolloff: float = 0.55
    # Glare. bloom_threshold is the radiance a pixel has to clear before it
    # throws any; at 1.0 that is roughly "brighter than white", so only genuine
    # highlights bloom and the sky stays clean. bloom_radius scales the blur's
    # reach.
    bloom: float = 0.35
    bloom_threshold: float = 1.0
    bloom_radius: float = 1.0
    # Lens flare: ghosts, halo and streaks. Needs no gate of its own -- it is
    # built from the same bright-pass buffer as the glare, so at a low
    # light_intensity there is simply nothing over the threshold for it to be
    # built out of, and it fades in by itself as the highlights get hot.
    # flare_threshold is *on top of* bloom_threshold: the flare is built from
    # whatever clears both, so it keys off only the searing cores while the
    # glare still picks up everything merely bright. Set it to 0 and the two
    # share a source, which turns a blown-out facet into a blown-out ghost.
    # Deliberately subtle by default. A strong flare washes ghosts straight
    # across the numerals, and this is a timer -- the readout is the point (the
    # same reasoning that made the camera sway rather than orbit). The dramatic
    # setting belongs to the shatter, when there is no longer a readout to
    # protect.
    flare: float = 0.25
    flare_threshold: float = 1.5
    flare_streak: float = 1.0
    flare_length: float = 4.0

    # Procedural backdrop. star_density counts cells across the whole celestial
    # sphere now that the sky is 3D, so it needs to be far larger than the
    # screen-space version wanted.
    nebula: float = 0.40
    nebula_color_a: tuple = (0.10, 0.16, 0.42)
    nebula_color_b: tuple = (0.42, 0.13, 0.34)
    star_density: float = 90.0
    star_brightness: float = 1.0


def _sky_fragment_source(defines: tuple[str, ...] = ()) -> str:
    """sky.frag with #defines injected straight after its #version line.

    GLSL requires #version to come first, so the defines cannot simply be
    prepended. Read fresh each call, which is what makes hot-reload work.
    """
    lines = (_SHADER_DIR / "sky.frag").read_text().splitlines(keepends=True)
    for index, line in enumerate(lines):
        if line.lstrip().startswith("#version"):
            head = index + 1
            break
    else:
        raise RuntimeError("sky.frag has no #version line")
    injected = "".join(f"#define {name} 1\n" for name in defines)
    return "".join(lines[:head]) + injected + "".join(lines[head:])


def default_surface_format() -> QSurfaceFormat:
    """A 3.3 core profile with multisampling. Must be set before QApplication."""
    fmt = QSurfaceFormat()
    fmt.setVersion(3, 3)
    fmt.setProfile(QSurfaceFormat.CoreProfile)
    fmt.setDepthBufferSize(24)
    fmt.setSamples(4)
    return fmt


# Shrinks the text projection so the numerals stay inside the bevel's inset
# ring instead of spilling onto the chamfer. >1 makes the text smaller; the rim
# then projects outside [0,1] and clamps to the texture's transparent border.
_FACE_UV_SCALE = 1.22


def _face_uv(x: float, y: float) -> tuple:
    """Planar-project a front-face point into the text texture."""
    u = x * 0.5 * _FACE_UV_SCALE + 0.5
    v = 1.0 - (y * 0.5 * _FACE_UV_SCALE + 0.5)
    return (u, v)


# Corner of the text image, which is always transparent. Side walls and the
# back face sample here so no numerals bleed onto them.
_NO_TEXT_UV = (0.002, 0.002)


def _add_triangle(data: array.array, tri, uvs, body, cap: float = 0.0) -> None:
    """Append one flat-shaded triangle, wound counter-clockwise from outside.

    Consistent outward winding is not cosmetic: the two-pass transparency in
    paintGL culls by face orientation to draw back surfaces before front ones.
    Get a triangle backwards and its wall renders in the wrong order.

    ``body`` is the wedge's (centre, velocity, axis), repeated on every vertex.
    Outwardness is judged against that centre -- the wedge's own interior point
    -- and *not* against the shard's centre of mass. The shard's axis lies
    inside both of a wedge's radial cut planes, so a cap triangle's normal is
    very nearly perpendicular to the direction from the shard centre, and the
    sign of that dot product is noise.

    ``cap`` marks the radial cut faces, which are interior surfaces while the
    shard is whole and are discarded by the fragment shader until it breaks.
    """
    ux, uy, uz = (tri[1][j] - tri[0][j] for j in range(3))
    vx, vy, vz = (tri[2][j] - tri[0][j] for j in range(3))
    nx = uy * vz - uz * vy
    ny = uz * vx - ux * vz
    nz = ux * vy - uy * vx

    centre, velocity, axis = body

    # Does the normal point away from the wedge's interior? If not, the
    # triangle is wound backwards: swap two vertices and flip the normal.
    cx = sum(v[0] for v in tri) / 3.0 - centre[0]
    cy = sum(v[1] for v in tri) / 3.0 - centre[1]
    cz = sum(v[2] for v in tri) / 3.0 - centre[2]
    if nx * cx + ny * cy + nz * cz < 0.0:
        tri = [tri[0], tri[2], tri[1]]
        uvs = [uvs[0], uvs[2], uvs[1]]
        nx, ny, nz = -nx, -ny, -nz

    length = math.sqrt(nx * nx + ny * ny + nz * nz) or 1.0
    normal = (nx / length, ny / length, nz / length)

    for (px, py, pz), (u, v) in zip(tri, uvs):
        data.extend(
            (px, py, pz, *normal, u, v, *centre, *velocity, *axis, cap)
        )


def _add_triangle_smooth(data, tri, uvs, normals, body, cap: float = 0.0) -> None:
    """Append one triangle carrying *per-vertex* normals.

    The curved front face is the one surface where flat shading defeats the
    purpose: subdivide it into thousands of facets with a normal each and the
    silhouette smooths while the specular highlight stays visibly faceted --
    you have simply traded six big facets for six thousand small ones.

    Winding still has to obey the same contract as _add_triangle, because
    paintGL culls by orientation to order the two transparency passes. The cap
    is convex and faces the camera, so "outward" here is unambiguously +z.
    """
    ux, uy, uz = (tri[1][j] - tri[0][j] for j in range(3))
    vx, vy, vz = (tri[2][j] - tri[0][j] for j in range(3))
    if ux * vy - uy * vx < 0.0:  # geometric normal points -z: wound backwards
        tri = [tri[0], tri[2], tri[1]]
        uvs = [uvs[0], uvs[2], uvs[1]]
        normals = [normals[0], normals[2], normals[1]]

    centre, velocity, axis = body
    for (px, py, pz), (u, v), normal in zip(tri, uvs, normals):
        data.extend(
            (px, py, pz, *normal, u, v, *centre, *velocity, *axis, cap)
        )


def _dome_apex_z() -> float:
    """Apex height of the fully tangent-continuous dome (bulge = 1).

    For one radial direction, the arc that meets the inset ring at the bevel's
    own slope is a circle centred on the axis: with s the bevel's steepness
    (dz/dr, outward and negative) and r the inset radius,

        rho**2 = r**2 * (1 + 1/s**2)     z_centre = _BEVEL_Z - r/s

    The outline is irregular, so every direction yields a slightly different
    apex height (they span about 5%). The surface can only have one apex, so
    take the mean and let the cubic in _front_profile_z absorb the residual --
    which is why the profile is a cubic and not literally a circle. A circle
    has two free parameters and four constraints here; a cubic has four.
    """
    total = 0.0
    for x, y in _OUTLINE:
        radius = math.hypot(x, y)
        inset_r = radius * _BEVEL_INSET
        steepness = _BEVEL_Z / (radius * (1.0 - _BEVEL_INSET))
        rho = inset_r * math.sqrt(1.0 + 1.0 / (steepness * steepness))
        total += (_BEVEL_Z - inset_r / steepness) + rho
    return total / len(_OUTLINE)


_DOME_APEX_Z = _dome_apex_z()


def _front_profile_z(u: float, inset_r: float, steepness: float, bulge: float):
    """Height of the front surface at radial parameter ``u``.

    ``u`` runs 0 at the apex to 1 at the inset ring, along one radial
    direction. ``bulge`` blends between the original flat triangle (0) and the
    tangent-continuous dome (1); because both profiles pin the same value at
    the ring, the blend stays exactly on the ring at every bulge, and the
    slope there is a blend of the two slopes -- so the crease closes smoothly
    rather than only at bulge = 1.
    """
    flat = _PEAK_Z + u * (_BEVEL_Z - _PEAK_Z)
    if bulge <= 0.0:
        return flat

    # Cubic Hermite: value/tangent at the apex (u=0) and at the ring (u=1).
    # Apex tangent is 0 -- a nonzero one would put a cone point back at the
    # centre, which is the very artefact we are removing.
    uu = u * u
    uuu = uu * u
    h00 = 2.0 * uuu - 3.0 * uu + 1.0
    h01 = -2.0 * uuu + 3.0 * uu
    h11 = uuu - uu
    # dz/du at the ring = dz/dr * dr/du = -steepness * inset_r.
    curved = h00 * _DOME_APEX_Z + h01 * _BEVEL_Z + h11 * (-steepness * inset_r)
    return flat + bulge * (curved - flat)


def _front_patch(inset_a, inset_b, rim_a, rim_b, n: int, bulge: float,
                 crystal: float = 0.0, scale: float = 3.0, clear: float = 0.0,
                 jitter: float = 0.0):
    """Ring-subdivided front cap for one wedge, ``n`` rings deep.

    With ``crystal`` > 0 each point is pushed along z by the plate field. Along
    z and not along the surface normal, which is the cheaper-looking option and
    the better one: the numerals are projected onto the cap planarly, so a
    displacement with any x/y component drags them sideways -- worst exactly
    where the cap is steepest -- while a pure-z one cannot move a UV at all.
    On a cap whose whole rise is 0.09 units the two are near enough
    indistinguishable anyway.

    With ``jitter`` > 0 the sample parameters are perturbed as well, which
    randomises the shape of every triangle. It is done in (u, t) rather than in
    x/y on purpose, because the two boundaries that matter are parameter lines:
    a radial jitter slides a point along its own ray, so the chains at t = 0
    and t = 1 stay exactly in the cut planes the radial faces are built from,
    and a tangential jitter slides a point along its own ring, so the chain at
    u = 1 stays exactly on the straight inset edge the bevel needs. Each
    perturbation is faded out at the two ends of its own parameter, so no
    corner of the patch moves at all.

    The perturbation is a function of *position*, never of (u, t) or of the
    ring and slot indices. Wedge i meets its neighbour at t = 1 while the
    neighbour meets it at t = 0; anything keyed on the parameter would give the
    two different answers and tear the seam open.

    The numerals do not care. _face_uv is a planar *projection*: a vertex that
    moves sideways simply picks up the coordinate of whatever is at its new
    position, so the text stays exactly where it was in space and only the
    triangulation under it changes.

    The analytic normals returned alongside are the *undisplaced* surface's.
    They are correct wherever the field is faded out -- the rim band and the
    clear zone -- which is the only place the caller still uses them; the
    fractured part is flat-shaded off the facets themselves.

    ``rings[j]`` holds j+1 points, ring 0 being the shared apex and ring n the
    chain along the inset edge. The two radial chains ``rings[j][0]`` and
    ``rings[j][j]`` are the wedge's cut boundaries; the radial cut faces must
    reuse them vertex-for-vertex or the solid opens up along every cut.
    """
    rings, normals = [], {}
    for j in range(n + 1):
        u = j / n
        row = []
        for k in range(j + 1):
            t = k / j if j else 0.0
            su, st = u, t
            if jitter > 0.0 and 0 < j:
                # Sample the noise at where this vertex *would* have been, so
                # that two wedges asking about their shared edge ask about the
                # same place. Only the planar part of that point is needed
                # and it is two multiplies, so take it directly rather than
                # running _cap_point_and_normal's cubic profile and central
                # difference for a z and a normal that get thrown away.
                bx = inset_a[0] + t * (inset_b[0] - inset_a[0])
                by = inset_a[1] + t * (inset_b[1] - inset_a[1])
                px, py = bx * u, by * u
                amp = jitter * _CRYSTAL_JITTER_MAX
                su = u + amp * _ends_fade(u) / n * _signed_noise2(
                    px, py, _CRYSTAL_SALT + 13
                )
                st = t + amp * _ends_fade(t) / j * _signed_noise2(
                    px, py, _CRYSTAL_SALT + 15
                )
            point, normal = _cap_point_and_normal(
                su, st, inset_a, inset_b, rim_a, rim_b, bulge
            )
            if crystal > 0.0:
                lift = (
                    crystal
                    * _plate_height(point[0], point[1], scale)
                    * _rim_fade(u)
                    * _crystal_weight(point[0], point[1], clear)
                )
                point = (point[0], point[1], point[2] + lift)
            row.append(point)
            normals[_normal_key(point)] = normal
        rings.append(row)
    return rings, normals


def _normal_key(point) -> tuple:
    """Quantised position, so the same point reached from two wedges matches."""
    return tuple(round(c, 5) for c in point)


def _mean_facet_normal(triangles) -> tuple:
    """Area-weighted mean normal of a group of facets, oriented outward (+z).

    The weight is the cross product's own length, left un-normalised, so wide
    facets count for more than slivers.
    """
    sx = sy = sz = 0.0
    for (ax, ay, az), (bx, by, bz), (cx, cy, cz) in triangles:
        ux, uy, uz = bx - ax, by - ay, bz - az
        vx, vy, vz = cx - ax, cy - ay, cz - az
        nx = uy * vz - uz * vy
        ny = uz * vx - ux * vz
        nz = ux * vy - uy * vx
        if nz < 0.0:
            nx, ny, nz = -nx, -ny, -nz
        sx, sy, sz = sx + nx, sy + ny, sz + nz
    length = math.sqrt(sx * sx + sy * sy + sz * sz) or 1.0
    return (sx / length, sy / length, sz / length)


def _profile_dz_du(u, inset_r, steepness, bulge):
    """d/du of _front_profile_z. Analytic, because a difference here shows."""
    flat = _BEVEL_Z - _PEAK_Z
    if bulge <= 0.0:
        return flat
    curved = (
        (6.0 * u * u - 6.0 * u) * _DOME_APEX_Z
        + (-6.0 * u * u + 6.0 * u) * _BEVEL_Z
        + (3.0 * u * u - 2.0 * u) * (-steepness * inset_r)
    )
    return flat + bulge * (curved - flat)


def _cap_point_and_normal(u, t, inset_a, inset_b, rim_a, rim_b, bulge):
    """Surface point and its exact normal at parameters (u, t).

    Normals are analytic rather than averaged from the facets. Facet averaging
    was the first implementation and it beaded the specular highlight: ring
    subdivision alternates upward- and downward-pointing triangles, so the
    accumulated normals zigzag from slot to slot, and above subdivision 3 that
    ripple beats against the specular lobe and breaks the highlight into a
    dashed line. The artefact scaled with triangle count -- the opposite of
    what a smoothing slider is supposed to do. The surface is parametric and
    its derivatives are known, so there is no reason to estimate them.
    """
    bx = inset_a[0] + t * (inset_b[0] - inset_a[0])
    by = inset_a[1] + t * (inset_b[1] - inset_a[1])

    def profile(tt):
        rx = rim_a[0] + tt * (rim_b[0] - rim_a[0])
        ry = rim_a[1] + tt * (rim_b[1] - rim_a[1])
        radius = math.hypot(rx, ry)
        return radius * _BEVEL_INSET, _BEVEL_Z / (radius * (1.0 - _BEVEL_INSET))

    inset_r, steepness = profile(t)
    z = _front_profile_z(u, inset_r, steepness, bulge)
    point = (bx * u, by * u, z)

    # The apex is a pole: dP/dt vanishes there and the cross product is
    # undefined. Its normal is +z by construction -- the profile's apex
    # tangent is horizontal, which is exactly what removes the cone point.
    if u <= 0.0:
        return point, (0.0, 0.0, 1.0)

    # dP/du along the radius.
    du = (bx, by, _profile_dz_du(u, inset_r, steepness, bulge))
    # dP/dt around the ring. z varies with t only through the outline's local
    # radius; a central difference on that is exact to rounding and far
    # clearer than differentiating hypot through the lerp.
    h = 1e-4
    lo_r, lo_s = profile(t - h)
    hi_r, hi_s = profile(t + h)
    dz_dt = (
        _front_profile_z(u, hi_r, hi_s, bulge)
        - _front_profile_z(u, lo_r, lo_s, bulge)
    ) / (2.0 * h)
    dt = (
        u * (inset_b[0] - inset_a[0]),
        u * (inset_b[1] - inset_a[1]),
        dz_dt,
    )

    nx = du[1] * dt[2] - du[2] * dt[1]
    ny = du[2] * dt[0] - du[0] * dt[2]
    nz = du[0] * dt[1] - du[1] * dt[0]
    if nz < 0.0:
        nx, ny, nz = -nx, -ny, -nz
    length = math.sqrt(nx * nx + ny * ny + nz * nz) or 1.0
    return point, (nx / length, ny / length, nz / length)


def _accumulate_patch_normals(normals_by_slot, accum: dict) -> None:
    """Fold one wedge's analytic normals into a *shared* vertex accumulator.

    Shared across wedges on purpose, and this is the only averaging that
    happens: within a wedge each position occurs once, so the analytic normal
    survives untouched. Only the two radial seams (shared by two wedges) and
    the apex (shared by all six) get averaged.

    That seam averaging is the point. The outline is a polygon, so the surface
    genuinely has an angular crease along every cut line; left alone it reads
    as six smooth petals with six seams rather than one continuous dome.
    """
    for key, normal in normals_by_slot.items():
        px, py, pz = accum.get(key, (0.0, 0.0, 0.0))
        accum[key] = (px + normal[0], py + normal[1], pz + normal[2])


def _resolve_normals(accum: dict) -> dict:
    resolved = {}
    for key, (nx, ny, nz) in accum.items():
        length = math.sqrt(nx * nx + ny * ny + nz * nz) or 1.0
        resolved[key] = (nx / length, ny / length, nz / length)
    return resolved


def _patch_triangles(rings):
    """Index triples ``(ring, slot)`` tiling the cap, apex ring outward."""
    for j in range(len(rings) - 1):
        for k in range(j + 1):
            yield ((j, k), (j + 1, k), (j + 1, k + 1))
        for k in range(j):
            yield ((j, k), (j + 1, k + 1), (j, k + 1))


def _hash01(i: int, salt: int) -> float:
    """Deterministic pseudo-random in [0, 1).

    Deliberately not ``random``: the break must look the same on every run, so
    a bad-looking tumble is reproducible and a test can pin the values.
    """
    x = math.sin(i * 12.9898 + salt * 78.233) * 43758.5453
    return x - math.floor(x)


def _hash2i(gx: int, gy: int, salt: int) -> float:
    """Deterministic pseudo-random in [0, 1) for a 2D lattice cell.

    Folds the cell into one integer with the usual coprime multipliers, then
    takes it mod 2**20 before handing it to _hash01. The modulo is not
    cosmetic: _hash01 is a sine hash, and feeding sine an argument of 1e9
    spends every bit of a double's mantissa on the integer part, so
    neighbouring cells start returning correlated values -- visible as plates
    that line up in rows.
    """
    return _hash01(((gx * 374761393) ^ (gy * 668265263)) % 1048576, salt)


def _value_noise2(x: float, y: float, salt: int) -> float:
    """Smooth lattice noise in [0, 1). Continuous, which is the whole point.

    Everything that perturbs a shared vertex has to be continuous in position,
    not merely deterministic. Two wedges reach their common cut edge by
    different arithmetic and land up to an ulp apart; a hash of that position
    would return two unrelated numbers and tear the seam open, while a
    continuous field returns two numbers an ulp apart.
    """
    ix = math.floor(x)
    iy = math.floor(y)
    fx = x - ix
    fy = y - iy
    sx = fx * fx * (3.0 - 2.0 * fx)
    sy = fy * fy * (3.0 - 2.0 * fy)

    n00 = _hash2i(ix, iy, salt)
    n10 = _hash2i(ix + 1, iy, salt)
    n01 = _hash2i(ix, iy + 1, salt)
    n11 = _hash2i(ix + 1, iy + 1, salt)
    return (
        (n00 * (1.0 - sx) + n10 * sx) * (1.0 - sy)
        + (n01 * (1.0 - sx) + n11 * sx) * sy
    )


def _signed_noise2(x: float, y: float, salt: int) -> float:
    """Two octaves of _value_noise2, remapped to [-1, 1)."""
    return 2.0 * (
        0.65 * _value_noise2(x * _CRYSTAL_JITTER_LOW, y * _CRYSTAL_JITTER_LOW, salt)
        + 0.35 * _value_noise2(
            x * _CRYSTAL_JITTER_HIGH, y * _CRYSTAL_JITTER_HIGH, salt + 1
        )
    ) - 1.0


# Per-cell constants for the plate field, memoised. Every vertex looks at the
# nine cells around it and each cell costs five sine hashes, so neighbouring
# vertices were re-deriving the same forty-five numbers over and over. The
# entries are a pure function of the cell index, so this changes no output --
# and the field only ever spans the cap, which at the top of crystal_scale is a
# few hundred cells, so it does not grow.
_PLATE_CELLS: dict = {}


def _plate_cell(gx: int, gy: int) -> tuple:
    """Site position, base height and slope for one cell of the plate field."""
    cell = _PLATE_CELLS.get((gx, gy))
    if cell is None:
        cell = (
            gx + _hash2i(gx, gy, _CRYSTAL_SALT + 4),
            gy + _hash2i(gx, gy, _CRYSTAL_SALT + 5),
            2.0 * _hash2i(gx, gy, _CRYSTAL_SALT + 6) - 1.0,
            _CRYSTAL_TILT * (2.0 * _hash2i(gx, gy, _CRYSTAL_SALT + 7) - 1.0),
            _CRYSTAL_TILT * (2.0 * _hash2i(gx, gy, _CRYSTAL_SALT + 8) - 1.0),
        )
        _PLATE_CELLS[(gx, gy)] = cell
    return cell


def _plate_height(x: float, y: float, scale: float) -> float:
    """Just the height. See _plate_sample."""
    return _plate_sample(x, y, scale)[0]


def _plate_sample(x: float, y: float, scale: float) -> tuple:
    """Fractured-plate field: ``(height, tilt_x, tilt_y)`` at a planar point.

    Height is roughly in [-1, 1]; the tilts are the winning cell's own slope,
    per unit of cell space, and are what let a caller shade a whole plate as
    one panel instead of shading each triangle it happens to be cut into.

    A jittered-grid Voronoi: the plane is cut into cells, each cell gets its
    own height and its own slope, and a point takes the plane of whichever
    cell's site is nearest. Cell boundaries are therefore *discontinuities* --
    which is the whole point. Smooth noise displaced and flat-shaded gives
    faceted dunes; what fractured ice actually looks like is flat plates that
    do not line up, and the step where they meet is the fracture.

    A pure function of position, with no state and no reliance on which wedge
    is asking. That is what lets three surfaces that share a vertex displace it
    identically and keep the solid closed.
    """
    # Domain warp first: bend the plane, then cut it into cells. Warping the
    # lookup rather than the result is what varies the cells' *shapes* -- a
    # warp applied to the height afterwards would leave the same round,
    # same-sized cells and merely wobble their tops.
    px = x * scale + _CRYSTAL_WARP * _signed_noise2(x, y, _CRYSTAL_SALT + 9)
    py = y * scale + _CRYSTAL_WARP * _signed_noise2(x, y, _CRYSTAL_SALT + 11)
    ix, iy = math.floor(px), math.floor(py)

    best = 1e30
    height = 0.0
    tilt_x = tilt_y = 0.0
    for gy in range(iy - 1, iy + 2):
        for gx in range(ix - 1, ix + 2):
            sx, sy, base, tx, ty = _plate_cell(gx, gy)
            dx, dy = px - sx, py - sy
            d = dx * dx + dy * dy
            if d < best:
                best = d
                tilt_x, tilt_y = tx, ty
                height = base + tx * dx + ty * dy
    return height, tilt_x, tilt_y


def _ends_fade(x: float) -> float:
    """1 in the middle of a parameter's range, 0 at both ends.

    Pinning the ends is what keeps the patch's boundary where the rest of the
    solid expects it: the apex, the inset chain and the two radial chains are
    all parameter lines, and every one of them is shared with a surface that
    is not this one.
    """
    return 4.0 * x * (1.0 - x)


def _rim_fade(u: float) -> float:
    """Fade the fracture field out before it reaches the inset ring."""
    if u <= _CRYSTAL_RIM_FADE:
        return 1.0
    t = (1.0 - u) / (1.0 - _CRYSTAL_RIM_FADE)
    return t * t * (3.0 - 2.0 * t)


def _crystal_weight(x: float, y: float, clear: float) -> float:
    """How much crystal a point on the cap gets, 0 (flat) to 1 (full spike).

    One weight per spike, taken at its base point, and it drives both the
    height and the normal blend -- so a weight of 0 emits a spike of zero
    height carrying the cap's own smooth normals, which is the original
    surface exactly, just cut into three coplanar pieces. That matters more
    than it sounds: the clear zone cannot *skip* spikes, because every wedge
    has to end up with the same vertex count for _wedge_bounds to slice the
    buffer into equal blocks. Tapering to nothing is the way to skip.
    """
    if clear <= 0.0:
        return 1.0
    r = math.hypot(x, y / _CRYSTAL_CLEAR_ASPECT)
    lo = clear
    hi = clear + _CRYSTAL_CLEAR_FEATHER
    if r <= lo:
        return 0.0
    if r >= hi:
        return 1.0
    t = (r - lo) / (hi - lo)
    return t * t * (3.0 - 2.0 * t)


def _normalize(v) -> tuple:
    length = math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]) or 1.0
    return (v[0] / length, v[1] / length, v[2] / length)


def _mix_normal(smooth, flat, w: float) -> tuple:
    """Normalised blend from a smooth vertex normal toward its facet normal."""
    if w >= 1.0:
        return flat
    nx = smooth[0] + w * (flat[0] - smooth[0])
    ny = smooth[1] + w * (flat[1] - smooth[1])
    nz = smooth[2] + w * (flat[2] - smooth[2])
    length = math.sqrt(nx * nx + ny * ny + nz * nz) or 1.0
    return (nx / length, ny / length, nz / length)


def _panel_normal(points, normals, crystal, scale):
    """Normal of the *plate* a triangle sits on, not of the triangle.

    This is the difference between a crust of uniform little facets and one of
    panels. The mesh is a ring-and-slot grid, so every triangle on it is
    roughly the same size; shade each independently and the eye reads that
    regularity straight through however chaotic the heights are. Give every
    triangle inside one Voronoi cell the same normal and the *plate* becomes
    the visible facet -- and plates come in whatever sizes and shapes the cell
    structure produced, which is the point.

    The cap underneath is curved, so the plate's plane is not simply its tilt.
    Recover the smooth cap's own gradient from the analytic normal it already
    carries (for z = f(x, y) the normal runs parallel to (-f_x, -f_y, 1)), add
    the plate's, and rebuild.
    """
    cx = sum(p[0] for p in points) / 3.0
    cy = sum(p[1] for p in points) / 3.0

    smooth = _normalize((
        sum(n[0] for n in normals),
        sum(n[1] for n in normals),
        sum(n[2] for n in normals),
    ))
    nz = smooth[2] if abs(smooth[2]) > 1e-6 else 1e-6
    _h, tilt_x, tilt_y = _plate_sample(cx, cy, scale)
    return _normalize((
        smooth[0] / nz - crystal * tilt_x * scale,
        smooth[1] / nz - crystal * tilt_y * scale,
        1.0,
    ))


def _crystal_facets(points, normals, height, vary, clear, index,
                    panel_normal=None, panel: float = 0.0):
    """One cap triangle, emitted as three flat facets around a centre tip.

    Yields ``(triangle, uvs, normals)``. This is where the cap stops being
    smooth: the three facets carry the base triangle's own geometric normal
    rather than the interpolated vertex normals, so every triangle edge on the
    cap becomes a visible crease. Flat shading is not a shortcut here, it is
    the effect -- a displaced surface with interpolated normals reads as
    eroded dunes, and the same surface flat-shaded reads as fractured plate.

    ``height`` (crystal_spike) additionally lifts the tip out of the base
    plane, along the base facet's *own* normal so the three sides come off it
    symmetrically, by a multiple of the facet's size. That scaling matters:
    cap triangles near the apex ring are far smaller than the ones near the
    rim, and an absolute lift would turn those into needles while barely
    dimpling these. ``vary`` spreads it per spike -- far enough at 1.0 to go
    negative, which pits the facet instead of spiking it.

    At height 0 the tip stays in the plane and the three facets are coplanar:
    a flat-shaded copy of the original triangle, which is exactly what the
    plate crust wants. Only the tip ever leaves the base triangle, so all
    three of its edges stay where the cap put them.
    """
    (ax, ay, az), (bx, by, bz), (cx, cy, cz) = points

    # Facet normal, oriented outward, and the facet's own size. twice_area is
    # the cross product's length, which is 2*area -- the normalisation divisor
    # and the size measure fall out of the same square root.
    ux, uy, uz = bx - ax, by - ay, bz - az
    vx, vy, vz = cx - ax, cy - ay, cz - az
    nx = uy * vz - uz * vy
    ny = uz * vx - ux * vz
    nz = ux * vy - uy * vx
    if nz < 0.0:
        nx, ny, nz = -nx, -ny, -nz
    twice_area = math.sqrt(nx * nx + ny * ny + nz * nz) or 1.0
    facet_n = (nx / twice_area, ny / twice_area, nz / twice_area)
    size = math.sqrt(0.5 * twice_area)

    # Adopt the plate's normal -- but only if this triangle is plausibly *in*
    # the plate. A cell boundary is a step in the height field, so the
    # triangles that straddle one are near-vertical fracture walls whose real
    # normal has nothing to do with either plate's plane; shading those as if
    # they lay flat turns every fracture into a smear. Where the two normals
    # already broadly agree the triangle is interior and takes the panel's;
    # where they do not, it is a wall and keeps its own.
    if panel_normal is not None and panel > 0.0:
        agree = (facet_n[0] * panel_normal[0] + facet_n[1] * panel_normal[1]
                 + facet_n[2] * panel_normal[2])
        t = min(1.0, max(0.0, (agree - 0.55) / 0.30))
        facet_n = _mix_normal(
            facet_n, panel_normal, panel * t * t * (3.0 - 2.0 * t)
        )

    # Barycentric tip position. Weights are perturbed about 1/3 and then
    # renormalised, so they stay strictly positive and the tip stays strictly
    # inside the triangle -- which is what guarantees the three side facets
    # are non-degenerate and that the tip's planar position is inside the
    # base's, so its UV cannot escape the face.
    w0 = 1.0 + _CRYSTAL_TIP_JITTER * (_hash01(index, _CRYSTAL_SALT) - 0.5)
    w1 = 1.0 + _CRYSTAL_TIP_JITTER * (_hash01(index, _CRYSTAL_SALT + 1) - 0.5)
    w2 = 1.0 + _CRYSTAL_TIP_JITTER * (_hash01(index, _CRYSTAL_SALT + 2) - 0.5)
    total = w0 + w1 + w2
    w0, w1, w2 = w0 / total, w1 / total, w2 / total

    base = (
        w0 * ax + w1 * bx + w2 * cx,
        w0 * ay + w1 * by + w2 * cy,
        w0 * az + w1 * bz + w2 * cz,
    )
    weight = _crystal_weight(base[0], base[1], clear)

    # 1.5 rather than 1.0 so that vary can reach past a flat facet into a pit
    # before the slider tops out; at vary = 1 the amplitude spans [-0.5, 2.5].
    amp = 1.0 + vary * 1.5 * (2.0 * _hash01(index, _CRYSTAL_SALT + 3) - 1.0)
    lift = height * amp * size * weight

    tip = (
        base[0] + facet_n[0] * lift,
        base[1] + facet_n[1] * lift,
        base[2] + facet_n[2] * lift,
    )
    # UV from the *base* point, not the tip. The tip is displaced along the
    # facet normal, which on a bulged cap has real x/y components; reading the
    # numerals from the tip's own position would shear them outward, worst
    # exactly where the cap is steepest.
    tip_uv = _face_uv(base[0], base[1])
    tip_n = _mix_normal(
        _normalize((
            w0 * normals[0][0] + w1 * normals[1][0] + w2 * normals[2][0],
            w0 * normals[0][1] + w1 * normals[1][1] + w2 * normals[2][1],
            w0 * normals[0][2] + w1 * normals[1][2] + w2 * normals[2][2],
        )),
        facet_n, weight,
    )

    uvs = [_face_uv(p[0], p[1]) for p in points]
    blended = [_mix_normal(n, facet_n, weight) for n in normals]

    for p, q in ((0, 1), (1, 2), (2, 0)):
        yield (
            [points[p], points[q], tip],
            [uvs[p], uvs[q], tip_uv],
            [blended[p], blended[q], tip_n],
        )


def _wedge_rigid_body(i: int, corners) -> tuple:
    """Pivot, linear velocity and angular velocity for one wedge.

    The pivot is the wedge's own centroid -- that is the whole point. Rotating
    every piece about the *shard's* centre is what produced the pinwheel.
    """
    n = float(len(corners))
    centre = (
        sum(c[0] for c in corners) / n,
        sum(c[1] for c in corners) / n,
        sum(c[2] for c in corners) / n,
    )

    # Drift outward from the axis, with a little scatter so the break is not a
    # symmetric bloom. Negative z: the pieces recede and shrink, as falling
    # glass does. Positive z threw them at the camera, where they ballooned.
    radial = math.hypot(centre[0], centre[1]) or 1.0
    speed = 0.26 + 0.16 * _hash01(i, 1)
    velocity = (
        centre[0] / radial * speed + (_hash01(i, 2) - 0.5) * 0.12,
        centre[1] / radial * speed + (_hash01(i, 3) - 0.5) * 0.12 + 0.14,
        -(0.10 + 0.20 * _hash01(i, 4)),
    )

    # Angular velocity: direction is the axis, magnitude is rad/sec. Keep this
    # slow -- a wedge's far corner is ~1 unit from its pivot, so even 2 rad/s
    # sweeps it across the frame.
    ax = _hash01(i, 5) - 0.5
    ay = _hash01(i, 6) - 0.5
    az = _hash01(i, 7) - 0.5
    alen = math.sqrt(ax * ax + ay * ay + az * az) or 1.0
    rate = 0.7 + 1.2 * _hash01(i, 8)
    axis = (ax / alen * rate, ay / alen * rate, az / alen * rate)

    return centre, velocity, axis


def _wedge_bounds(data: array.array) -> list:
    """Per-wedge (centre, launch velocity, radius) read back from the buffer.

    A wedge's vertices are one contiguous block, and its pivot centre and
    launch velocity are stored identically on every one of them (see
    ``_build_geometry``). The radius is the farthest any vertex sits from that
    centre -- and since the shatter only *rotates* a vertex about the centre
    (tumble) and *translates* the centre (drift + gravity), that radius bounds
    the whole piece for all time. So a sphere of this radius around the moving
    centre contains the wedge no matter how far it has tumbled: exactly what the
    early-clear frustum test needs, with no per-vertex work at check time.

    Returns ``[(centre, velocity, radius), ...]``, one entry per wedge, in the
    shard's rest frame (the idle-spin at the break is applied later).
    """
    stride = _FLOATS_PER_VERTEX
    n_wedges = len(_OUTLINE)
    total_verts = len(data) // stride
    verts_per_wedge = total_verts // n_wedges
    bounds = []
    for w in range(n_wedges):
        base = w * verts_per_wedge * stride
        centre = tuple(data[base + 8 : base + 11])
        velocity = tuple(data[base + 11 : base + 14])
        radius = 0.0
        for v in range(verts_per_wedge):
            off = base + v * stride
            dx = data[off] - centre[0]
            dy = data[off + 1] - centre[1]
            dz = data[off + 2] - centre[2]
            radius = max(radius, math.sqrt(dx * dx + dy * dy + dz * dz))
        bounds.append((centre, velocity, radius))
    return bounds


def _build_geometry(subdiv: int = 0, bulge: float = 0.0, crystal: float = 0.0,
                    crystal_density: int = 0, crystal_vary: float = 0.5,
                    crystal_clear: float = 0.0, crystal_scale: float = 3.0,
                    crystal_spike: float = 0.0,
                    crystal_jitter: float = 0.0,
                    crystal_panel: float = 0.0) -> array.array:
    """Extrude the outline into a solid, one wedge per edge.

    Each wedge contributes a front face, a front bevel, a side wall, a back
    bevel and a back face -- and all of them share one rigid-body state. That
    makes a wedge a real chunk: the whole solid piece tumbles together, rather
    than the front skin peeling off its own side wall.

    ``subdiv`` and ``bulge`` curve the front face. Defaults reproduce the
    original six flat triangles byte for byte.

    Interleaved per vertex:
        pos(3) normal(3) uv(2) pieceCenter(3) pieceVel(3) pieceAxis(3) = 17
    """
    data = array.array("f")
    # One cap tessellation for every wedge, so the per-wedge vertex count stays
    # uniform -- see _tris_per_wedge. Crystals ride on a finer cap than the
    # smoothing slider alone would ask for; everything downstream of the patch
    # (bevel strip, radial cuts) reads the chains generically and does not care.
    cap_rings = _rings(
        _cap_level(subdiv, crystal, crystal_density, crystal_spike),
        _CRYSTAL_SUBDIV_MAX,
    )
    crystalline = crystal > 0.0 or crystal_spike > 0.0
    # Jitter is only meaningful once the cap is flat-shaded: on the smooth cap
    # the creases it would randomise are not drawn at all, and moving the
    # vertices would achieve nothing but a different set of interpolation
    # errors under the numerals.
    jitter = max(0.0, min(1.0, crystal_jitter)) if crystalline else 0.0
    apex_z = _front_profile_z(0.0, 0.0, 0.0, bulge)
    front_apex = (0.0, 0.0, apex_z)
    back_apex = (0.0, 0.0, -_THICKNESS - _BACK_PEAK_Z)
    n = len(_OUTLINE)

    # Front caps first, in their own pass: the vertex normals are averaged
    # across wedge boundaries, so no wedge can be emitted until every wedge's
    # facets have been accumulated.
    patches = []
    accum: dict = {}
    for i in range(n):
        ax, ay = _OUTLINE[i]
        bx, by = _OUTLINE[(i + 1) % n]
        rings, slot_normals = _front_patch(
            (ax * _BEVEL_INSET, ay * _BEVEL_INSET, _BEVEL_Z),
            (bx * _BEVEL_INSET, by * _BEVEL_INSET, _BEVEL_Z),
            (ax, ay, 0.0),
            (bx, by, 0.0),
            cap_rings,
            bulge,
            crystal,
            crystal_scale,
            crystal_clear,
            jitter,
        )
        _accumulate_patch_normals(slot_normals, accum)
        patches.append(rings)
    cap_normals = _resolve_normals(accum)

    for i in range(n):
        ax, ay = _OUTLINE[i]
        bx, by = _OUTLINE[(i + 1) % n]
        rings = patches[i]

        rim_a = (ax, ay, 0.0)
        rim_b = (bx, by, 0.0)
        inset_a = (ax * _BEVEL_INSET, ay * _BEVEL_INSET, _BEVEL_Z)
        inset_b = (bx * _BEVEL_INSET, by * _BEVEL_INSET, _BEVEL_Z)

        back_rim_a = (ax, ay, -_THICKNESS)
        back_rim_b = (bx, by, -_THICKNESS)
        back_inset_a = (
            ax * _BEVEL_INSET, ay * _BEVEL_INSET, -_THICKNESS - _BACK_BEVEL_Z
        )
        back_inset_b = (
            bx * _BEVEL_INSET, by * _BEVEL_INSET, -_THICKNESS - _BACK_BEVEL_Z
        )

        # One rigid body for the whole wedge, pivoting on its own centroid.
        # Built first: its centre is the interior reference _add_triangle uses
        # to orient every facet of this wedge outward.
        piece = _wedge_rigid_body(
            i,
            [
                rim_a, rim_b, inset_a, inset_b,
                back_rim_a, back_rim_b, back_inset_a, back_inset_b,
                front_apex, back_apex,
            ],
        )

        uv_rim_a = _face_uv(ax, ay)
        uv_rim_b = _face_uv(bx, by)
        blank = [_NO_TEXT_UV] * 3

        # Front face: carries the numerals. _face_uv is a planar projection, so
        # every subdivided vertex gets its text coordinate for free and the
        # numerals map identically however far the cap is curved -- the etching
        # needed no changes at all.
        for tri_index, tri in enumerate(_patch_triangles(rings)):
            points = [rings[j][k] for j, k in tri]
            normals = [cap_normals[_normal_key(p)] for p in points]
            panel_n = None
            if crystal_panel > 0.0 and crystal > 0.0:
                panel_n = _panel_normal(points, normals, crystal, crystal_scale)
            if not crystalline:
                _add_triangle_smooth(
                    data,
                    points,
                    [_face_uv(p[0], p[1]) for p in points],
                    normals,
                    piece,
                )
                continue
            for facet, uvs, facet_normals in _crystal_facets(
                points, normals, crystal_spike, crystal_vary, crystal_clear,
                i * _CRYSTAL_STRIDE + tri_index, panel_n, crystal_panel,
            ):
                _add_triangle_smooth(data, facet, uvs, facet_normals, piece)

        # Front bevel: the chamfer that catches the edge highlight. Its inset
        # edge is shared with the cap, so it has to follow the cap's
        # subdivision of that edge -- otherwise the T-junctions leave the
        # wedge an open shell and it reads as hollow when it tumbles.
        chain = rings[-1]
        strip = [
            ([rim_a, chain[k], chain[k + 1]],
             [uv_rim_a, _face_uv(*chain[k][:2]), _face_uv(*chain[k + 1][:2])])
            for k in range(len(chain) - 1)
        ]
        strip.append((
            [rim_a, chain[-1], rim_b],
            [uv_rim_a, _face_uv(*chain[-1][:2]), uv_rim_b],
        ))
        # One normal for the whole chamfer, not one per sliver.
        #
        # The strip is a fan from rim_a, so at subdivision 5 it is 33 long thin
        # triangles across a band only ~0.1 units wide. Flat-shading each of
        # them independently banded the chamfer, and a specular highlight
        # crossing the band broke into a dashed line whose dash count tracked
        # the subdivision level -- a smoothing slider that visibly got worse
        # the further you pushed it. The chamfer is one facet conceptually, so
        # it gets one normal: the area-weighted mean of the strip.
        bevel_normal = _mean_facet_normal(tri for tri, _ in strip)
        for tri, uvs in strip:
            _add_triangle_smooth(
                data, tri, uvs, [bevel_normal] * 3, piece
            )
        # Side wall: the thickness you can actually see.
        _add_triangle(data, [rim_a, back_rim_a, back_rim_b], blank, piece)
        _add_triangle(data, [rim_a, back_rim_b, rim_b], blank, piece)
        # Back bevel, shallower than the front.
        _add_triangle(
            data, [back_rim_a, back_inset_a, back_inset_b], blank, piece
        )
        _add_triangle(data, [back_rim_a, back_inset_b, back_rim_b], blank, piece)
        # Back face.
        _add_triangle(data, [back_apex, back_inset_b, back_inset_a], blank, piece)

        # Radial cut faces. Without these a wedge is an open shell, and the
        # tumbling pieces read as hollow the moment they turn edge-on. Each cut
        # is a convex polygon in the plane through the shard's axis and one rim
        # point.
        #
        # The front run of each profile is the cap's own radial chain, reused
        # vertex-for-vertex: a straight apex-to-inset segment against a curved
        # cap would leave a crescent gap down every cut, visible the moment the
        # pieces separate.
        #
        # Fanned from the *rim*, not the apex. The chain's points are collinear
        # with the apex whenever bulge is 0, so an apex fan produced zero-area
        # facets there -- and a zero-area facet has no normal, which breaks the
        # winding contract the two-pass transparency depends on. The rim lies
        # off that line at every bulge, and the cross-section is convex, so a
        # fan from it is always well formed.
        for chain_a, rim, back_rim, back_inset in (
            ([rings[j][0] for j in range(len(rings))],
             rim_a, back_rim_a, back_inset_a),
            ([rings[j][j] for j in range(len(rings))],
             rim_b, back_rim_b, back_inset_b),
        ):
            profile = chain_a + [rim, back_rim, back_inset, back_apex]
            pivot = len(chain_a)  # index of `rim`
            count = len(profile)
            for step in range(1, count - 1):
                _add_triangle(
                    data,
                    [
                        profile[pivot],
                        profile[(pivot + step) % count],
                        profile[(pivot + step + 1) % count],
                    ],
                    blank,
                    piece,
                    cap=1.0,
                )

    return data


def render_text_image(text: str, params: ShardParams) -> QImage:
    """Draw the numerals into an RGBA image. No GL context required.

    Kept a free function rather than a widget method so it can be unit-tested
    without a GL context -- the ``offscreen`` Qt platform has no GL at all.
    """
    image = QImage(_TEX_W, _TEX_H, QImage.Format_RGBA8888)
    image.fill(Qt.transparent)
    if not text:
        return image

    painter = QPainter(image)
    painter.setRenderHint(QPainter.Antialiasing)
    painter.setRenderHint(QPainter.TextAntialiasing)

    font = QFont(params.font_family)
    font.setBold(params.font_bold)
    font.setPixelSize(_TEX_PT)
    painter.setFont(font)

    # Shrink to fit rather than clip: the readout widens by a character once
    # the hour rolls over.
    width = painter.fontMetrics().horizontalAdvance(text)
    if width > _TEX_W * _TEXT_FIT:
        font.setPixelSize(int(_TEX_PT * (_TEX_W * _TEXT_FIT) / width))
        painter.setFont(font)

    # White; the shader tints by uTextColor so colour stays a live uniform.
    painter.setPen(QColor(255, 255, 255))
    painter.drawText(image.rect(), Qt.AlignCenter, text)
    painter.end()
    return image


class ShardWidget(QOpenGLWidget):
    """Drop-in replacement for SpinnerWidget, plus ``set_text``."""

    def __init__(self, model, params: ShardParams | None = None) -> None:
        super().__init__()
        self._model = model
        self.params = params or ShardParams()

        self._alarm = False
        self._alarm_phase = 0.0
        self._shatter_t = 0.0  # seconds since the break; 0 while intact
        self._spin = 0.0
        self._spin_at_break = 0.0
        self._text = ""

        self._program: QOpenGLShaderProgram | None = None
        self._uniforms: dict[str, int] = {}
        self._sky_program: QOpenGLShaderProgram | None = None
        self._sky_uniforms: dict[str, int] = {}
        self._sky_vao = QOpenGLVertexArrayObject()
        self._nebula_cube: QOpenGLTexture | None = None

        # Offscreen HDR chain. _scene_fbo is multisampled and cannot be sampled
        # from, so it is blit-resolved into _resolve_fbo; the bloom pair
        # ping-pongs at half resolution. All four are rebuilt on resize.
        self._scene_fbo: QOpenGLFramebufferObject | None = None
        self._resolve_fbo: QOpenGLFramebufferObject | None = None
        self._bloom_fbos: list[QOpenGLFramebufferObject] = []
        self._target_size: tuple[int, int] = (0, 0)
        self._post_programs: dict[str, QOpenGLShaderProgram] = {}
        self._post_uniforms: dict[str, dict[str, int]] = {}
        self._elapsed = 0.0
        self._texture: QOpenGLTexture | None = None
        self._text_dirty = True
        self._shaders_dirty = False

        # Early-clear state: the shatter's fixed timeout is only an upper bound,
        # so we stop drawing the instant the pieces are actually gone. See
        # _refresh_early_clear / pieces_have_cleared.
        self._wedge_bounds: list = []
        self._early_cleared = False
        self._next_clear_check = 0.0

        self._vao = QOpenGLVertexArrayObject()
        self._vbo = QOpenGLBuffer(QOpenGLBuffer.VertexBuffer)
        self._vertex_data = array.array("f")
        self._vertex_count = 0
        self._geometry_key = None
        self._geometry_dirty = True
        self._rebuild_geometry()

        self.setMinimumSize(280, 280)

        self._watcher = QFileSystemWatcher(self)
        for name in (
            "shard.vert", "shard.frag", "sky.vert", "sky.frag",
            *(f"{stage}.frag" for stage in _POST_STAGES),
        ):
            self._watcher.addPath(str(_SHADER_DIR / name))
        self._watcher.fileChanged.connect(self._on_shader_changed)

    # -- public interface, mirrors the old spinner ------------------------

    def set_text(self, text: str) -> None:
        if text != self._text:
            self._text = text
            self._text_dirty = True

    def set_alarm(self, active: bool) -> None:
        if active == self._alarm:
            return
        self._alarm = active
        if active:
            # Break from wherever the idle rotation happens to be, not from
            # the rest pose. Otherwise the shard visibly snaps back to its
            # start angle on the frame it shatters.
            self._spin_at_break = self._spin
            self._early_cleared = False
            self._next_clear_check = 0.0
        else:
            # Reset reassembles the shard.
            self._shatter_t = 0.0
            self._alarm_phase = 0.0
            self._early_cleared = False

    def advance(self, dt: float) -> None:
        # The sky drifts and twinkles in every state, including after the
        # shard has shattered and stopped being drawn.
        self._elapsed += dt

        if self._alarm:
            # The pieces tumble away and keep going; the red pulse continues
            # long after they have left the frame, until the user resets.
            self._shatter_t += dt
            self._alarm_phase += dt * 1.5 * 2.0 * math.pi
            self._refresh_early_clear()
        else:
            # One idle speed, whether or not the model is running. Speeding up
            # while the timer ran drew the eye to the rotation instead of the
            # numerals.
            self._spin += dt * self.params.idle_spin

        # Animation time keeps flowing above; only the *repaint* is held back.
        #
        # A modal dialog runs a nested event loop, and Qt's posted paint events
        # are dispatched from inside it -- so the frame timers keep driving the
        # HDR chain while the user is looking at a file chooser. Under Mesa that
        # paint costs so much of the main thread (~80% of it: the post chain
        # plus a DRI3 back-buffer wait in makeCurrent) that the dialog's own
        # input handling is left a sliver, and clicks take seconds to land. The
        # NVIDIA driver is fast enough here to hide the problem, which is why
        # this only ever showed up on the integrated GPU.
        #
        # Note the dialog need not be a Qt one: the native GTK file chooser
        # still reports through Qt as the active modal widget, which is what
        # makes this a reliable test rather than a guess about what is on top.
        if QApplication.activeModalWidget() is not None:
            return

        self.update()

    # How often the early-clear geometry check runs, in seconds of shatter time.
    # A few times a second is ample: the pieces do not teleport.
    _CLEAR_CHECK_INTERVAL_S = 0.25
    # How far ahead each check looks. It only has to bridge the gap to the next
    # check with margin -- the check re-runs continuously, so a re-entry is
    # caught then rather than having to be foreseen now. Kept > the interval so
    # consecutive look-aheads overlap and leave no instant uncovered.
    _CLEAR_HORIZON_S = 0.5

    def _refresh_early_clear(self) -> None:
        """Re-test, a few times a second, whether the pieces are gone right now.

        Deliberately *not* latched. A piece that has left the frame can re-enter
        later -- a wide sway, or a full orbit (sway_degrees=180), sweeps the
        camera back toward it -- and when it does we must resume drawing. Because
        the check looks only a short horizon ahead (not the whole timeout), it is
        cheap enough to run for the shatter's full duration, and its cost no
        longer grows with shatter_clear_s.
        """
        if self._shatter_t < self._next_clear_check:
            return
        self._next_clear_check = self._shatter_t + self._CLEAR_CHECK_INTERVAL_S
        self._early_cleared = self._all_pieces_offscreen()

    def _all_pieces_offscreen(self) -> bool:
        """True if no wedge is on screen across the next look-ahead window.

        Walks each wedge's bounding sphere along the exact trajectory the vertex
        shader integrates (frozen idle-spin, drift, gravity) and tests it against
        the view frustum -- re-evaluating the still-swaying camera at each sample
        so a piece the camera pans *toward* isn't missed. Conservative: a sphere
        that only clips a frustum corner counts as visible, so this can lag the
        true clear but never triggers while anything is on screen.
        """
        bounds = self._wedge_bounds
        if not bounds:
            return False

        now = self._shatter_t
        timeout = self.params.shatter_clear_s
        if now >= timeout:
            return True

        gy_half = 0.5 * self.params.gravity * _GRAVITY_1G
        spin = self._spin_at_break
        cs, sn = math.cos(spin), math.sin(spin)
        elapsed_at_break = self._elapsed - self._shatter_t

        # Idle-spin (about Y) was frozen at the break and rotates each wedge's
        # rest-frame centre and launch velocity once. Neither depends on t or on
        # the camera, so spin them here -- not once per sample per wedge.
        spun = []
        for centre, vel, radius in bounds:
            cx = cs * centre[0] + sn * centre[2]
            cz = -sn * centre[0] + cs * centre[2]
            vx = cs * vel[0] + sn * vel[2]
            vz = -sn * vel[0] + cs * vel[2]
            spun.append((cx, centre[1], cz, vx, vel[1], vz, radius))

        aspect = max(self.width(), 1) / max(self.height(), 1)
        tan_v = math.tan(math.radians(_FOV_DEGREES) / 2.0)
        tan_h = tan_v * aspect
        sec_v = math.sqrt(1.0 + tan_v * tan_v)
        sec_h = math.sqrt(1.0 + tan_h * tan_h)
        near = 0.1

        # A short, fixed look-ahead (capped at the timeout), sampled finely
        # enough that a piece cannot cross the frame between two samples.
        horizon = min(timeout, now + self._CLEAR_HORIZON_S)
        step = 0.12
        t = now
        while True:
            eye, right, up, forward = self.camera(elapsed_at_break + t)
            # Pull the camera basis into plain floats once per sample: the
            # frustum test below is then pure arithmetic, with no per-wedge
            # PySide binding calls (which dominated the old inner loop).
            ex, ey, ez = eye.x(), eye.y(), eye.z()
            fx, fy, fz = forward.x(), forward.y(), forward.z()
            rx, ry, rz = right.x(), right.y(), right.z()
            ux, uy, uz = up.x(), up.y(), up.z()
            tt = t * t
            for cx, cy, cz, vx, vy, vz, radius in spun:
                px = cx + vx * t
                py = cy + vy * t - gy_half * tt
                pz = cz + vz * t
                dx = px - ex
                dy = py - ey
                dz = pz - ez
                dv = dx * fx + dy * fy + dz * fz
                xv = dx * rx + dy * ry + dz * rz
                yv = dx * ux + dy * uy + dz * uz
                dvth = dv * tan_h
                dvtv = dv * tan_v
                rh = radius * sec_h
                rv = radius * sec_v
                # Fully outside the frustum iff beyond any single plane by more
                # than the radius. If no plane rejects it, treat it as visible.
                if not (
                    dv < near - radius
                    or xv - dvth > rh    # past right
                    or -xv - dvth > rh   # past left
                    or yv - dvtv > rv    # past top
                    or -yv - dvtv > rv   # past bottom
                ):
                    return False
            if t >= horizon:
                return True
            t = min(t + step, horizon)

    @property
    def pieces_have_cleared(self) -> bool:
        """True while the tumbling wedges are off screen (so paintGL can skip).

        Tracks the latest early-clear check (see _all_pieces_offscreen), so it
        can flip back to False if a piece re-enters -- the shard resumes drawing
        for that window. ``shatter_clear_s`` is the hard upper bound past which
        the shard is considered gone regardless.
        """
        return self._early_cleared or self._shatter_t > self.params.shatter_clear_s

    def refresh_params(self) -> None:
        """Call after mutating ``params`` from the tuning panel."""
        self._text_dirty = True  # font/colour may have changed
        self._rebuild_geometry()
        self.update()

    def _rebuild_geometry(self) -> bool:
        """Rebuild the vertex data if the curvature parameters moved.

        Keyed on the values rather than rebuilt unconditionally: refresh_params
        fires on *every* slider, and re-tessellating plus re-uploading the
        buffer because someone nudged the specular power would be silly.
        Returns True if the data changed, so initializeGL/paintGL know whether
        the VBO needs reallocating.
        """
        subdiv = max(0, min(_FRONT_SUBDIV_MAX, int(round(self.params.front_subdiv))))
        bulge = max(0.0, min(1.0, float(self.params.front_bulge)))
        crystal = max(0.0, float(self.params.crystal))
        density = max(0, min(_CRYSTAL_DENSITY_MAX,
                             int(round(self.params.crystal_density))))
        vary = max(0.0, min(1.0, float(self.params.crystal_vary)))
        clear = max(0.0, float(self.params.crystal_clear))
        scale = max(0.1, float(self.params.crystal_scale))
        spike = max(0.0, float(self.params.crystal_spike))
        jitter = max(0.0, min(1.0, float(self.params.crystal_jitter)))
        panel = max(0.0, min(1.0, float(self.params.crystal_panel)))
        key = (subdiv, bulge, crystal, density, vary, clear, scale, spike,
               jitter, panel)
        if key == self._geometry_key:
            return False

        self._geometry_key = key
        self._vertex_data = _build_geometry(
            subdiv, bulge, crystal, density, vary, clear, scale, spike,
            jitter, panel,
        )
        self._vertex_count = len(self._vertex_data) // _FLOATS_PER_VERTEX
        self._geometry_dirty = True
        self._wedge_bounds = _wedge_bounds(self._vertex_data)
        return True

    # -- shader loading ---------------------------------------------------

    def _on_shader_changed(self, path: str) -> None:
        # Editors often replace rather than modify; re-add to keep watching.
        if path not in self._watcher.files():
            self._watcher.addPath(path)
        self._shaders_dirty = True
        self.update()

    def _load_program(self) -> None:
        """Compile the shader pair. Keep the old program if this fails."""
        program = QOpenGLShaderProgram()
        ok = program.addShaderFromSourceFile(
            QOpenGLShader.Vertex, str(_SHADER_DIR / "shard.vert")
        ) and program.addShaderFromSourceFile(
            QOpenGLShader.Fragment, str(_SHADER_DIR / "shard.frag")
        )
        if ok:
            program.bindAttributeLocation("aPos", 0)
            program.bindAttributeLocation("aNormal", 1)
            program.bindAttributeLocation("aUV", 2)
            program.bindAttributeLocation("aPieceCenter", 3)
            program.bindAttributeLocation("aPieceVel", 4)
            program.bindAttributeLocation("aPieceAxis", 5)
            program.bindAttributeLocation("aCap", 6)
            ok = program.link()

        if not ok:
            log = program.log().strip()
            if self._program is None:
                raise RuntimeError(f"initial shader compile failed:\n{log}")
            print(f"[shard] shader reload failed, keeping previous:\n{log}")
            return

        self._program = program
        # Cache uniform locations: PySide6's setUniformValue only accepts a
        # name as bytes, and only the int-location overloads cover plain
        # floats. Looking them up once is both correct and cheaper.
        program.bind()
        self._uniforms = {
            name: program.uniformLocation(name)
            for name in (
                "uText", "uModel", "uView", "uProj", "uNormalMat",
                "uCamPos", "uLightPos", "uLightColor", "uLightIntensity",
                "uGlassColor", "uTextColor",
                "uSpecPower", "uSpecStrength", "uFresnel", "uGlow",
                "uEtch", "uEtchDepth", "uBaseAlpha",
                "uAlarm", "uShatterT",
                "uGravity",
                "uSpin",
                "uSpinAtBreak",
            )
        }
        program.release()
        self._bind_attributes()
        print("[shard] shaders reloaded")

    def _set(self, name: str, value) -> None:
        """Set a uniform by name, skipping ones the compiler optimised out."""
        location = self._uniforms.get(name, -1)
        if location >= 0:
            self._program.setUniformValue(location, value)

    def _build_sky_program(self, defines: tuple[str, ...] = ()) -> QOpenGLShaderProgram | None:
        """Compile sky.{vert,frag}, optionally with preprocessor defines.

        The bake pass and the runtime pass are the same file compiled twice, so
        the noise cannot drift between what is baked and what would have been
        drawn. Returns None on failure; the caller decides whether that is
        fatal or merely a bad hot-reload.
        """
        program = QOpenGLShaderProgram()
        ok = program.addShaderFromSourceFile(
            QOpenGLShader.Vertex, str(_SHADER_DIR / "sky.vert")
        ) and program.addShaderFromSourceCode(
            QOpenGLShader.Fragment, _sky_fragment_source(defines)
        )
        if ok and program.link():
            return program
        print(f"[sky] compile failed{list(defines)}:\n{program.log().strip()}")
        return None

    def _load_sky_program(self) -> None:
        """Compile the backdrop shaders. Keep the old program if this fails."""
        program = self._build_sky_program()
        if program is None:
            if self._sky_program is None:
                raise RuntimeError("initial sky shader compile failed")
            print("[sky] shader reload failed, keeping previous")
            return

        program.bind()
        self._sky_uniforms = {
            name: program.uniformLocation(name)
            for name in (
                "uTime", "uNebula", "uNebulaColorA", "uNebulaColorB",
                "uStarDensity", "uStarBrightness",
                "uCamRight", "uCamUp", "uCamForward", "uTanHalfFov", "uAspect",
                "uNebulaCube",
            )
        }
        # The cube lives on texture unit 1; unit 0 is the shard's text atlas.
        loc = self._sky_uniforms.get("uNebulaCube", -1)
        if loc >= 0:
            program.setUniformValue1i(loc, 1)
        program.release()
        self._sky_program = program
        print("[sky] shaders reloaded")

        # The shape the runtime pass samples comes from this same source, so a
        # hot-reload that changes the noise must re-bake or the two disagree.
        self._bake_nebula()

    def _load_post_programs(self) -> None:
        """Compile the post-process stages. Keep the old ones if any fails.

        They all share sky.vert -- the fullscreen triangle built from
        gl_VertexID, with no vertex buffer -- because that is exactly what each
        of them needs and a second identical copy of it would only be one more
        file to keep in sync.
        """
        for stage, names in _POST_STAGES.items():
            program = QOpenGLShaderProgram()
            ok = program.addShaderFromSourceFile(
                QOpenGLShader.Vertex, str(_SHADER_DIR / "sky.vert")
            ) and program.addShaderFromSourceFile(
                QOpenGLShader.Fragment, str(_SHADER_DIR / f"{stage}.frag")
            )
            if ok:
                ok = program.link()
            if not ok:
                log = program.log().strip()
                if stage not in self._post_programs:
                    raise RuntimeError(f"initial {stage} compile failed:\n{log}")
                print(f"[post] {stage} reload failed, keeping previous:\n{log}")
                continue

            program.bind()
            locations = {
                name: program.uniformLocation(name) for name in names
            }
            # Samplers are bound to fixed units and never move, so they are set
            # here once rather than every frame.
            for name, unit in (
                ("uScene", 0), ("uSource", 0), ("uBloom", 1), ("uFlare", 2),
            ):
                location = locations.get(name, -1)
                if location >= 0:
                    program.setUniformValue1i(location, unit)
            program.release()

            self._post_uniforms[stage] = locations
            self._post_programs[stage] = program
        print("[post] shaders reloaded")

    def _post_set(self, stage: str, name: str, value, *, is_float=False) -> None:
        program = self._post_programs[stage]
        location = self._post_uniforms[stage].get(name, -1)
        if location < 0:
            return
        if is_float:
            program.setUniformValue1f(location, float(value))
        else:
            program.setUniformValue(location, value)

    # -- offscreen targets ------------------------------------------------

    def _ensure_targets(self, width: int, height: int) -> None:
        """Allocate the HDR chain, reallocating only when the size changes.

        Every buffer here is RGBA16F. The scene target additionally carries
        multisampling and a depth attachment; the rest are single-sample colour
        only, since nothing after the resolve has geometry to antialias or
        depth to test.
        """
        if self._target_size == (width, height) and self._scene_fbo is not None:
            return
        self._target_size = (width, height)

        scene_format = QOpenGLFramebufferObjectFormat()
        scene_format.setInternalTextureFormat(_GL_RGBA16F)
        scene_format.setAttachment(QOpenGLFramebufferObject.CombinedDepthStencil)
        scene_format.setSamples(_SCENE_SAMPLES)
        self._scene_fbo = QOpenGLFramebufferObject(
            QSize(width, height), scene_format
        )

        colour_format = QOpenGLFramebufferObjectFormat()
        colour_format.setInternalTextureFormat(_GL_RGBA16F)
        self._resolve_fbo = QOpenGLFramebufferObject(
            QSize(width, height), colour_format
        )

        divisor = _bloom_divisor(width)
        bloom_size = QSize(max(1, width // divisor), max(1, height // divisor))
        self._bloom_fbos = [
            QOpenGLFramebufferObject(bloom_size, colour_format)
            for _ in range(_SCRATCH_BUFFERS)
        ]

        # Qt does not promise a filter mode on an FBO's texture, and every one
        # of these is sampled with bilinear taps at offsets that fall between
        # texel centres -- the 2x2 box in the bright-pass, the growing steps in
        # the blur, the half-to-full upscale in the composite. Left on NEAREST
        # the glare would come back blocky and would crawl as the camera moves.
        fns = self.context().functions()
        for fbo in (self._resolve_fbo, *self._bloom_fbos):
            fns.glBindTexture(_GL_TEXTURE_2D, fbo.texture())
            for pname, value in (
                (_GL_TEXTURE_MIN_FILTER, _GL_LINEAR),
                (_GL_TEXTURE_MAG_FILTER, _GL_LINEAR),
                # Clamped, so the blur's taps near an edge repeat the border
                # pixel instead of wrapping the glare round to the far side.
                (_GL_TEXTURE_WRAP_S, _GL_CLAMP_TO_EDGE),
                (_GL_TEXTURE_WRAP_T, _GL_CLAMP_TO_EDGE),
            ):
                fns.glTexParameteri(_GL_TEXTURE_2D, pname, value)
        fns.glBindTexture(_GL_TEXTURE_2D, 0)

        got = self._scene_fbo.format().samples()
        print(
            f"[post] targets {width}x{height} RGBA16F, {got}x MSAA, "
            f"glare at {bloom_size.width()}x{bloom_size.height()} (1/{divisor})"
            + ("" if got else "  <-- multisampling unavailable")
        )

    def _bake_nebula(self) -> None:
        """Render the nebula's direction-only factors into a cubemap, once.

        This is the whole optimisation: the domain-warped fbm stops being
        per-pixel per-frame work and becomes a texture fetch. Six faces at 512
        is 1.6 Mpx, about a sixth of a single 4K frame, so it is cheap enough
        to redo on every shader reload rather than caching to disk.
        """
        fns = self.context().functions()
        program = self._build_sky_program(("SKY_PROCEDURAL_NEBULA", "SKY_BAKE"))
        if program is None:
            print("[sky] nebula bake skipped; keeping previous cube")
            return

        if self._nebula_cube is None:
            cube = QOpenGLTexture(QOpenGLTexture.TargetCubeMap)
            cube.setFormat(QOpenGLTexture.RG16F)
            cube.setSize(_NEBULA_CUBE_SIZE, _NEBULA_CUBE_SIZE)
            # No mipmaps: nothing minifies a skybox far enough to need them,
            # and generating them would only soften the wisps.
            cube.setMipLevels(1)
            cube.allocateStorage()
            cube.setMinificationFilter(QOpenGLTexture.Linear)
            cube.setMagnificationFilter(QOpenGLTexture.Linear)
            cube.setWrapMode(QOpenGLTexture.ClampToEdge)
            self._nebula_cube = cube

        # Without this, bilinear taps at a face edge clamp instead of reaching
        # into the neighbouring face, drawing a visible seam across the sky.
        fns.glEnable(_GL_TEXTURE_CUBE_MAP_SEAMLESS)

        # QOpenGLFramebufferObject owns the FBO name and frees it; PySide6's
        # glGenFramebuffers wants an out-array, and this needs no cleanup path.
        # Its own colour texture goes unused -- each face is attached over it.
        scratch = QOpenGLFramebufferObject(
            QSize(_NEBULA_CUBE_SIZE, _NEBULA_CUBE_SIZE)
        )
        scratch.bind()
        fns.glViewport(0, 0, _NEBULA_CUBE_SIZE, _NEBULA_CUBE_SIZE)
        fns.glDisable(_GL_DEPTH_TEST)
        fns.glDisable(_GL_BLEND)

        program.bind()
        p = self.params
        for name, value in (("uTanHalfFov", 1.0), ("uAspect", 1.0), ("uTime", 0.0)):
            loc = program.uniformLocation(name)
            if loc >= 0:
                program.setUniformValue1f(loc, float(value))
        self._sky_vao.bind()

        for index, (forward, right, up) in enumerate(_CUBE_FACE_BASES):
            for name, vec in (
                ("uCamForward", forward), ("uCamRight", right), ("uCamUp", up)
            ):
                loc = program.uniformLocation(name)
                if loc >= 0:
                    program.setUniformValue(loc, QVector3D(*vec))
            fns.glFramebufferTexture2D(
                _GL_FRAMEBUFFER,
                _GL_COLOR_ATTACHMENT0,
                _GL_TEXTURE_CUBE_MAP_POSITIVE_X + index,
                self._nebula_cube.textureId(),
                0,
            )
            fns.glDrawArrays(_GL_TRIANGLES, 0, 3)

        self._sky_vao.release()
        program.release()

        # scratch.release() would bind FBO 0; a QOpenGLWidget draws into its own.
        fns.glBindFramebuffer(_GL_FRAMEBUFFER, self.defaultFramebufferObject())
        # Qt sizes the viewport in device pixels, not logical ones; restoring
        # with self.width() would shrink the scene on a HiDPI screen.
        ratio = self.devicePixelRatio()
        fns.glViewport(0, 0, int(self.width() * ratio), int(self.height() * ratio))
        fns.glEnable(_GL_DEPTH_TEST)
        fns.glEnable(_GL_BLEND)
        print(f"[sky] nebula baked into {_NEBULA_CUBE_SIZE}^2 cubemap")

    def camera(self, elapsed: float | None = None) -> tuple:
        """Where the camera is and which way it faces.

        Sways across the front of the shard rather than orbiting it, and always
        looks at the origin. A sine sweep eases naturally at the extremes, so
        the reversal has no visible corner.

        ``elapsed`` defaults to the live clock; passing a value evaluates the
        camera at that animation time instead, which is how the early-clear
        check sees where the still-swaying camera will point a moment from now.

        Returns ``(eye, right, up, forward)`` in world space. Both passes use
        this: the shard for its view matrix, the sky to rebuild a view ray per
        pixel. They must agree, or the backdrop slides against the geometry.
        """
        p = self.params
        if elapsed is None:
            elapsed = self._elapsed
        phase = elapsed * p.orbit_speed
        angle = math.radians(p.sway_degrees) * math.sin(phase)

        # Bob at a different rate from the sway, so the two never resynchronise
        # into an obvious figure-of-eight.
        height = p.orbit_height + p.orbit_bob * math.sin(phase * 0.61)

        eye = QVector3D(
            p.orbit_radius * math.sin(angle),
            height,
            p.orbit_radius * math.cos(angle),
        )

        forward = (QVector3D(0.0, 0.0, 0.0) - eye).normalized()
        world_up = QVector3D(0.0, 1.0, 0.0)
        right = QVector3D.crossProduct(forward, world_up).normalized()
        up = QVector3D.crossProduct(right, forward).normalized()
        return eye, right, up, forward

    def _draw_sky(self, fns) -> None:
        """Fullscreen backdrop. No depth, no blending, no vertex buffer."""
        program = self._sky_program
        if program is None:
            return

        p = self.params
        _eye, right, up, forward = self.camera()
        aspect = max(self.width(), 1) / max(self.height(), 1)
        tan_half_fov = math.tan(math.radians(_FOV_DEGREES) / 2.0)
        fns.glDisable(_GL_DEPTH_TEST)
        fns.glDisable(_GL_BLEND)
        fns.glDepthMask(_GL_FALSE)

        program.bind()
        for name, value in (
            ("uTime", float(self._elapsed)),
            ("uNebula", float(p.nebula)),
            ("uStarDensity", float(p.star_density)),
            ("uStarBrightness", float(p.star_brightness)),
            ("uTanHalfFov", float(tan_half_fov)),
            ("uAspect", float(aspect)),
        ):
            loc = self._sky_uniforms.get(name, -1)
            if loc >= 0:
                program.setUniformValue1f(loc, value)
        for name, value in (
            ("uCamRight", right),
            ("uCamUp", up),
            ("uCamForward", forward),
            ("uNebulaColorA", QVector3D(*p.nebula_color_a)),
            ("uNebulaColorB", QVector3D(*p.nebula_color_b)),
        ):
            loc = self._sky_uniforms.get(name, -1)
            if loc >= 0:
                program.setUniformValue(loc, value)

        if self._nebula_cube is not None:
            self._nebula_cube.bind(1)

        self._sky_vao.bind()
        fns.glDrawArrays(_GL_TRIANGLES, 0, 3)
        self._sky_vao.release()
        program.release()

        if self._nebula_cube is not None:
            self._nebula_cube.release(1)
            # The shard pass samples its text atlas from unit 0 and never
            # re-selects the unit, so leaving unit 1 active would send its
            # binding to the wrong place.
            fns.glActiveTexture(_GL_TEXTURE0)

        fns.glDepthMask(_GL_TRUE)
        fns.glEnable(_GL_DEPTH_TEST)
        fns.glEnable(_GL_BLEND)

    def _set_float(self, name: str, value: float) -> None:
        """Set a float uniform.

        Not the same as ``_set``: PySide6 lists setUniformValue's int overload
        before its float one, so a Python float binds to the int overload and
        silently truncates -- 0.55 arrives in the shader as 0. That zeroed the
        alpha and the whole shard rendered invisible. setUniformValue1f is
        unambiguous.
        """
        location = self._uniforms.get(name, -1)
        if location >= 0:
            self._program.setUniformValue1f(location, value)

    def _upload_geometry(self) -> None:
        """(Re)allocate the VBO from _vertex_data.

        allocate() resizes as well as fills, which is what makes the
        subdivision slider work: the buffer grows from 288 vertices at level 0
        to ~19k at level 5. The attribute pointers are unaffected -- they are
        offsets into whatever buffer is bound, and the stride never changes.
        """
        self._geometry_dirty = False
        if not self._vbo.isCreated():
            return
        self._vbo.bind()
        self._vbo.allocate(
            self._vertex_data.tobytes(), len(self._vertex_data) * 4
        )
        self._vbo.release()

    def _bind_attributes(self) -> None:
        assert self._program is not None
        self._vao.bind()
        self._vbo.bind()
        stride = _FLOATS_PER_VERTEX * 4
        for loc, size, offset in (
            (0, 3, 0),        # aPos
            (1, 3, 3 * 4),    # aNormal
            (2, 2, 6 * 4),    # aUV
            (3, 3, 8 * 4),    # aPieceCenter
            (4, 3, 11 * 4),   # aPieceVel
            (5, 3, 14 * 4),   # aPieceAxis
            (6, 1, 17 * 4),   # aCap
        ):
            self._program.enableAttributeArray(loc)
            self._program.setAttributeBuffer(loc, _GL_FLOAT, offset, size, stride)
        self._vbo.release()
        self._vao.release()

    # -- text texture -----------------------------------------------------

    def _upload_text(self) -> None:
        if self._texture is not None:
            self._texture.destroy()
        # No .mirrored() here: the UV mapping already puts v=0 at the top of
        # the shard, which is where QOpenGLTexture puts the QImage's first row.
        self._texture = QOpenGLTexture(render_text_image(self._text, self.params))
        # Sample the base level: the glyphs are minified onto the face, so
        # mipmapping picks a blurry level and the numerals go soft. Anisotropy
        # keeps them crisp at the shard's oblique angle.
        self._texture.setMinificationFilter(QOpenGLTexture.Linear)
        self._texture.setMagnificationFilter(QOpenGLTexture.Linear)
        self._texture.setMaximumAnisotropy(16.0)
        self._texture.setWrapMode(QOpenGLTexture.ClampToEdge)
        self._text_dirty = False

    # -- QOpenGLWidget ----------------------------------------------------

    def initializeGL(self) -> None:  # noqa: N802 (Qt naming)
        fns = self.context().functions()
        fns.glEnable(_GL_DEPTH_TEST)
        fns.glEnable(_GL_BLEND)
        fns.glBlendFunc(_GL_SRC_ALPHA, _GL_ONE_MINUS_SRC_ALPHA)
        # Black, not a dark blue-grey. The sky pass covers every pixel, so this
        # is only ever seen if the backdrop fails to draw -- and it should look
        # unmistakably broken when that happens, not like a slightly dim sky.
        fns.glClearColor(0.0, 0.0, 0.0, 1.0)

        self._vao.create()
        self._vbo.create()
        self._upload_geometry()

        # Core profile requires a bound VAO for any draw, even one that fetches
        # no attributes. The sky triangle builds itself from gl_VertexID.
        self._sky_vao.create()

        self._load_program()
        self._load_sky_program()
        self._load_post_programs()

    def paintGL(self) -> None:  # noqa: N802
        if self._shaders_dirty:
            self._shaders_dirty = False
            self._load_program()
            self._load_sky_program()
            self._load_post_programs()
        if self._text_dirty:
            self._upload_text()
        if self._geometry_dirty:
            self._upload_geometry()

        fns = self.context().functions()

        # Qt sizes the widget's framebuffer in device pixels, not logical ones,
        # and so must every target in the chain -- otherwise the composite
        # resamples the scene on a HiDPI screen and the numerals go soft.
        ratio = self.devicePixelRatio()
        width = max(1, int(self.width() * ratio))
        height = max(1, int(self.height() * ratio))
        self._ensure_targets(width, height)

        # --- scene, into the HDR target -------------------------------------
        self._scene_fbo.bind()
        fns.glViewport(0, 0, width, height)
        fns.glClear(_GL_COLOR_BUFFER_BIT | _GL_DEPTH_BUFFER_BIT)

        # Backdrop first, and unconditionally: the shard may be gone, but the
        # sky is still there.
        self._draw_sky(fns)
        self._draw_shard(fns)

        # Multisampled targets cannot be sampled from, so resolve down to a
        # plain RGBA16F texture the post chain can read.
        QOpenGLFramebufferObject.blitFramebuffer(
            self._resolve_fbo, self._scene_fbo
        )

        # --- glare, then the single tonemap ---------------------------------
        self._render_bloom(fns, width, height)
        self._composite(fns, width, height)

    def _draw_shard(self, fns) -> None:
        """Rasterise the shard into whatever target is bound. Linear radiance.

        Split out of paintGL when the tonemap moved to the composite: this
        stage has two perfectly good reasons to draw nothing at all (no program
        yet, or the wedges have tumbled out of frame), and once there are passes
        that must run *afterwards*, an early return from paintGL would take the
        tonemap down with it and the window would go black.
        """
        program = self._program
        if program is None or self._texture is None:
            return

        # The wedges have tumbled out of frame. The alert still has ~115 s to
        # run; there is nothing left of the shard to rasterise.
        if self.pieces_have_cleared:
            return

        p = self.params
        aspect = max(self.width(), 1) / max(self.height(), 1)

        # The shard sits still in world space; the camera moves around it. The
        # old fixed model rotation is gone -- tilting the object *and* orbiting
        # the eye fights itself.
        model = QMatrix4x4()

        cam, _right, _up, _forward = self.camera()
        view = QMatrix4x4()
        view.lookAt(cam, QVector3D(0, 0, 0), QVector3D(0, 1, 0))

        proj = QMatrix4x4()
        proj.perspective(_FOV_DEGREES, aspect, 0.1, 100.0)

        alarm = 0.0
        if self._alarm:
            alarm = 0.5 + 0.5 * math.sin(self._alarm_phase)

        program.bind()
        self._texture.bind(0)
        self._set("uText", 0)
        self._set("uModel", model)
        self._set("uView", view)
        self._set("uProj", proj)
        self._set("uNormalMat", model.normalMatrix())
        self._set("uCamPos", cam)
        self._set("uLightPos", QVector3D(p.light_x, p.light_y, p.light_z))
        self._set("uLightColor", QVector3D(*p.light_color))
        self._set("uGlassColor", QVector3D(*p.glass_color))
        self._set("uTextColor", QVector3D(*p.text_color))
        self._set_float("uLightIntensity", float(p.light_intensity))
        self._set_float("uSpecPower", float(p.spec_power))
        self._set_float("uSpecStrength", float(p.spec_strength))
        self._set_float("uFresnel", float(p.fresnel))
        self._set_float("uGlow", float(p.glow))
        self._set_float("uEtch", float(p.etch))
        self._set_float("uEtchDepth", float(p.etch_depth))
        self._set_float("uBaseAlpha", float(p.base_alpha))
        self._set_float("uAlarm", float(alarm))
        self._set_float("uShatterT", float(self._shatter_t))
        self._set_float("uGravity", float(p.gravity * _GRAVITY_1G))
        self._set_float("uSpin", float(self._spin))
        self._set_float("uSpinAtBreak", float(self._spin_at_break))

        # Two passes, back surfaces first. The shard is translucent, so blend
        # order matters: draw the inside of the solid, then the outside over
        # it. Culling by winding gives that ordering for free, with no
        # per-triangle depth sort. Depth *writes* stay on so the numerals on
        # the front face still occlude the far wall behind them.
        self._vao.bind()
        fns.glEnable(_GL_CULL_FACE)
        for cull in (_GL_FRONT, _GL_BACK):
            fns.glCullFace(cull)
            fns.glDrawArrays(_GL_TRIANGLES, 0, self._vertex_count)
        fns.glDisable(_GL_CULL_FACE)
        self._vao.release()
        self._texture.release(0)
        program.release()

    # -- post-process passes ----------------------------------------------

    def _begin_post(self, fns) -> None:
        """State every fullscreen pass wants: no depth, no blend, VAO bound."""
        fns.glDisable(_GL_DEPTH_TEST)
        fns.glDisable(_GL_BLEND)
        fns.glDepthMask(_GL_FALSE)
        fns.glActiveTexture(_GL_TEXTURE0)
        self._sky_vao.bind()

    def _render_bloom(self, fns, width: int, height: int) -> None:
        """Bright-pass, then separable blur. Leaves the glare in _bloom_fbos[0].

        The two bloom targets ping-pong: each pass reads one and writes the
        other, never both, since sampling a texture that is attached to the
        bound framebuffer is undefined. An even number of blur draws per
        iteration (one horizontal, one vertical) is what puts the result back in
        slot 0 regardless of how many iterations run.
        """
        p = self.params
        # Read back from the target rather than recomputed, so the viewport can
        # never disagree with the buffer that was actually allocated.
        bloom_w = self._bloom_fbos[0].width()
        bloom_h = self._bloom_fbos[0].height()

        self._begin_post(fns)

        self._bloom_fbos[0].bind()
        fns.glViewport(0, 0, bloom_w, bloom_h)
        bright = self._post_programs["post_bright"]
        bright.bind()
        fns.glBindTexture(_GL_TEXTURE_2D, self._resolve_fbo.texture())
        # The box tap is in *full*-resolution texels: this pass is reading the
        # full-res scene even though it writes at half res.
        self._post_set(
            "post_bright", "uTexel", QVector2D(1.0 / width, 1.0 / height)
        )
        self._post_set("post_bright", "uExposure", p.exposure, is_float=True)
        self._post_set(
            "post_bright", "uThreshold", p.bloom_threshold, is_float=True
        )
        fns.glDrawArrays(_GL_TRIANGLES, 0, 3)
        bright.release()

        bloom_texel = QVector2D(1.0 / bloom_w, 1.0 / bloom_h)

        # Flare next, and before the bloom blur, because both read the raw
        # bright-pass in slot 0 and the blur is about to overwrite it.
        self._bloom_fbos[2].bind()
        fns.glViewport(0, 0, bloom_w, bloom_h)
        flare = self._post_programs["post_flare"]
        flare.bind()
        fns.glBindTexture(_GL_TEXTURE_2D, self._bloom_fbos[0].texture())
        self._post_set("post_flare", "uTexel", bloom_texel)
        self._post_set("post_flare", "uStreak", p.flare_streak, is_float=True)
        self._post_set("post_flare", "uLength", p.flare_length, is_float=True)
        self._post_set(
            "post_flare", "uThreshold", p.flare_threshold, is_float=True
        )
        fns.glDrawArrays(_GL_TRIANGLES, 0, 3)
        flare.release()

        blur = self._post_programs["post_blur"]
        blur.bind()

        # One light pass over the flare, to take the banding out of the streaks'
        # geometrically-spaced taps and soften the ghosts' edges.
        for dx, dy, source, target in (
            (1.0 / bloom_w, 0.0, 2, 1),
            (0.0, 1.0 / bloom_h, 1, 2),
        ):
            self._blur_pass(fns, bloom_w, bloom_h, dx, dy, source, target)

        # Then the bloom itself, ping-ponging 0 <-> 1. An even number of draws
        # per iteration is what leaves the result back in slot 0 however many
        # iterations run.
        for step in _BLOOM_STEPS:
            reach = step * max(0.0, p.bloom_radius)
            for dx, dy, source, target in (
                (reach / bloom_w, 0.0, 0, 1),
                (0.0, reach / bloom_h, 1, 0),
            ):
                self._blur_pass(fns, bloom_w, bloom_h, dx, dy, source, target)
        blur.release()

        self._sky_vao.release()

    def _blur_pass(self, fns, width, height, dx, dy, source, target) -> None:
        """One separable blur draw, scratch[source] -> scratch[target].

        Source and target must differ: sampling a texture that is attached to
        the bound framebuffer is undefined, and on some drivers it silently
        works right up until it does not.
        """
        self._bloom_fbos[target].bind()
        fns.glViewport(0, 0, width, height)
        fns.glBindTexture(_GL_TEXTURE_2D, self._bloom_fbos[source].texture())
        self._post_set("post_blur", "uDir", QVector2D(dx, dy))
        fns.glDrawArrays(_GL_TRIANGLES, 0, 3)

    def _composite(self, fns, width: int, height: int) -> None:
        """Tonemap scene + glare into the widget's own framebuffer.

        The last pass, and the only one that writes something a display can
        show. Everything before it is unbounded linear radiance.
        """
        p = self.params

        # release() on an FBO would bind 0, which is not where a QOpenGLWidget
        # draws -- it owns its own framebuffer and Qt composites that.
        fns.glBindFramebuffer(_GL_FRAMEBUFFER, self.defaultFramebufferObject())
        fns.glViewport(0, 0, width, height)

        self._begin_post(fns)

        program = self._post_programs["post_composite"]
        program.bind()
        fns.glBindTexture(_GL_TEXTURE_2D, self._resolve_fbo.texture())
        fns.glActiveTexture(_GL_TEXTURE1)
        fns.glBindTexture(_GL_TEXTURE_2D, self._bloom_fbos[0].texture())
        fns.glActiveTexture(_GL_TEXTURE2)
        fns.glBindTexture(_GL_TEXTURE_2D, self._bloom_fbos[2].texture())

        self._post_set("post_composite", "uExposure", p.exposure, is_float=True)
        self._post_set(
            "post_composite", "uBloomStrength", p.bloom, is_float=True
        )
        self._post_set(
            "post_composite", "uFlareStrength", p.flare, is_float=True
        )
        self._post_set("post_composite", "uRolloff", p.rolloff, is_float=True)
        fns.glDrawArrays(_GL_TRIANGLES, 0, 3)

        program.release()
        self._sky_vao.release()

        # Unit 0 is where the shard's text atlas and the sky's cube expect to
        # find themselves next frame; leaving a higher unit selected would send
        # their bindings somewhere else entirely.
        for unit in (_GL_TEXTURE2, _GL_TEXTURE1, _GL_TEXTURE0):
            fns.glActiveTexture(unit)
            fns.glBindTexture(_GL_TEXTURE_2D, 0)

        fns.glDepthMask(_GL_TRUE)
        fns.glEnable(_GL_DEPTH_TEST)
        fns.glEnable(_GL_BLEND)

    def resizeGL(self, w: int, h: int) -> None:  # noqa: N802
        pass  # projection and the HDR targets are both rebuilt from paintGL

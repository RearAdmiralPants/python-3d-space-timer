"""Time the real sky.frag at a given resolution, offscreen.

The sky is the app's dominant per-pixel cost: two 5-octave 3D fbm calls plus
three star layers, evaluated for every pixel of every frame. It is therefore
the thing that decides whether a given GPU can hold the 16 ms frame budget.

Renders FRAMES frames into an FBO with glFinish() on each, so the number is
GPU work rather than queue-submit latency. uTime advances per frame so the
driver cannot hoist anything out of the loop.

    python tools/bench_sky.py [WIDTH HEIGHT [FRAMES]]

With no arguments it uses the primary display's resolution. Note this measures
the shader in isolation: it excludes the shard, and it excludes PRIME's
per-frame copy back to the display, which a windowed app does pay.

It times four builds of the same shader, so the frame cost can be attributed:

    baked    what ships now: stars procedural, nebula from the cubemap
    full     the pre-bake shader: both evaluated per pixel
    stars    star layers only
    nebula   the domain-warped fbm only, unbaked
    empty    neither -- ray setup and raw fill rate, the floor

`baked` reads an unpopulated cube here (the bench does no bake pass), so its
noise cost is right but its colours are not. Timing is the point.

The parts do not sum to the whole; a GPU overlaps them. Read the deltas against
`empty` as "what would baking this away actually buy", which is the question
that decides whether the cubemap is worth building.

Uniforms come from default-params.json when it exists, because the shipped
star_density (178) is double the ShardParams default (90) and the cost scales
with it. Benchmark what actually runs.
"""
import json
import os
import math
import pathlib
import subprocess
import sys
import time

from PySide6.QtCore import QSize
from PySide6.QtGui import (
    QGuiApplication, QOffscreenSurface, QOpenGLContext, QSurfaceFormat, QVector3D,
)
from PySide6.QtOpenGL import (
    QOpenGLFramebufferObject, QOpenGLShader, QOpenGLShaderProgram,
    QOpenGLVertexArrayObject,
)

from naive_timer.shard import ShardParams, _FOV_DEGREES

GL_TRIANGLES = 0x0004
GL_RENDERER = 0x1F01

ROOT = pathlib.Path(__file__).resolve().parent.parent
SHADERS = ROOT / "src" / "naive_timer" / "shaders"

# label -> preprocessor symbols to define
VARIANTS = [
    ("baked", []),
    ("full", ["SKY_PROCEDURAL_NEBULA"]),
    ("stars", ["SKY_SKIP_NEBULA"]),
    ("nebula", ["SKY_PROCEDURAL_NEBULA", "SKY_SKIP_STARS"]),
    ("empty", ["SKY_SKIP_NEBULA", "SKY_SKIP_STARS"]),
]


def frag_source(defines):
    """sky.frag with #defines injected after the #version line.

    They have to go after #version -- GLSL requires it to be the first thing in
    the file bar comments and whitespace.
    """
    text = (SHADERS / "sky.frag").read_text()
    lines = text.splitlines(keepends=True)
    for i, line in enumerate(lines):
        if line.lstrip().startswith("#version"):
            head = i + 1
            break
    else:
        raise SystemExit("sky.frag has no #version line")
    # Accepts bare "NAME" and "NAME=VALUE".
    injected = "".join(
        f"#define {d.replace('=', ' ', 1) if '=' in d else d + ' 1'}\n" for d in defines
    )
    return "".join(lines[:head]) + injected + "".join(lines[head:])


def shipped_params():
    """ShardParams overridden by default-params.json, which is what the app loads."""
    p = ShardParams()
    path = ROOT / "default-params.json"
    if not path.exists():
        return p, False
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return p, False
    for key, value in data.items():
        if hasattr(p, key):
            setattr(p, key, value)
    return p, True


def primary_resolution(fallback=(1920, 1080)):
    """Ask xrandr for the current mode of the primary display."""
    try:
        out = subprocess.run(["xrandr"], capture_output=True, text=True, timeout=5).stdout
    except (OSError, subprocess.SubprocessError):
        return fallback
    for line in out.splitlines():
        if "*" in line:
            w, _, h = line.split()[0].partition("x")
            return int(w), int(h)
    return fallback


def main(argv):
    # SKY_BENCH_OCTAVES=3 prices the fbm octave count without editing the
    # shader. One process per value: Qt allows only one QGuiApplication.
    octaves = os.environ.get("SKY_BENCH_OCTAVES")
    if octaves:
        for _, defines in VARIANTS:
            defines.append(f"SKY_FBM_OCTAVES={int(octaves)}")

    if len(argv) >= 3:
        width, height = int(argv[1]), int(argv[2])
    else:
        width, height = primary_resolution()
    frames = int(argv[3]) if len(argv) >= 4 else 60

    fmt = QSurfaceFormat()
    fmt.setVersion(3, 3)
    fmt.setProfile(QSurfaceFormat.CoreProfile)
    QSurfaceFormat.setDefaultFormat(fmt)

    app = QGuiApplication(argv)  # noqa: F841  (must outlive the GL objects)

    surface = QOffscreenSurface()
    surface.setFormat(fmt)
    surface.create()

    ctx = QOpenGLContext()
    ctx.setFormat(fmt)
    if not ctx.create():
        return "could not create an OpenGL context"
    ctx.makeCurrent(surface)
    gl = ctx.functions()

    vao = QOpenGLVertexArrayObject()
    vao.create()

    fbo = QOpenGLFramebufferObject(QSize(width, height))
    fbo.bind()
    gl.glViewport(0, 0, width, height)
    vao.bind()

    p, from_file = shipped_params()

    def time_variant(defines):
        prog = QOpenGLShaderProgram()
        prog.addShaderFromSourceFile(QOpenGLShader.Vertex, str(SHADERS / "sky.vert"))
        if not prog.addShaderFromSourceCode(QOpenGLShader.Fragment, frag_source(defines)):
            raise SystemExit(f"fragment compile failed: {prog.log()}")
        if not prog.link():
            raise SystemExit(f"shader link failed: {prog.log()}")
        prog.bind()

        prog.setUniformValue1f("uTanHalfFov", math.tan(math.radians(_FOV_DEGREES) / 2.0))
        prog.setUniformValue1f("uAspect", width / height)
        prog.setUniformValue1f("uNebula", float(p.nebula))
        prog.setUniformValue1f("uStarDensity", float(p.star_density))
        prog.setUniformValue1f("uStarBrightness", float(p.star_brightness))
        prog.setUniformValue("uNebulaColorA", QVector3D(*p.nebula_color_a))
        prog.setUniformValue("uNebulaColorB", QVector3D(*p.nebula_color_b))
        prog.setUniformValue("uCamRight", QVector3D(1.0, 0.0, 0.0))
        prog.setUniformValue("uCamUp", QVector3D(0.0, 1.0, 0.0))
        prog.setUniformValue("uCamForward", QVector3D(0.0, 0.0, -1.0))

        def draw(t):
            prog.setUniformValue1f("uTime", t)
            gl.glDrawArrays(GL_TRIANGLES, 0, 3)

        # Warm up: shader compile, first-use driver paths, clocks spinning up.
        for i in range(10):
            draw(i * 0.016)
        gl.glFinish()

        times = []
        for i in range(frames):
            start = time.perf_counter()
            draw(100.0 + i * 0.016)
            gl.glFinish()
            times.append((time.perf_counter() - start) * 1000.0)
        times.sort()
        return times

    print(f"renderer : {gl.glGetString(GL_RENDERER)}")
    print(f"size     : {width}x{height} ({width * height / 1e6:.1f} Mpx)")
    print(f"params   : {'default-params.json' if from_file else 'ShardParams defaults'}"
          f"  (star_density={p.star_density:g}, nebula={p.nebula:g})")
    print(f"frames   : {frames} per variant\n")

    # Measure every variant before printing: the "vs empty" column needs the
    # floor, and the floor is measured last.
    timings = {label: time_variant(defines) for label, defines in VARIANTS}
    results = {label: t[len(t) // 2] for label, t in timings.items()}

    print(f"{'variant':8} {'median':>9} {'min':>8} {'max':>8} {'vs empty':>10}")
    for label, _ in VARIANTS:
        times = timings[label]
        delta = results[label] - results["empty"]
        print(f"{label:8} {results[label]:8.2f}ms {times[0]:7.2f} {times[-1]:7.2f} "
              f"{'' if label == 'empty' else f'{delta:9.2f}ms'}")

    full, empty = results["full"], results["empty"]
    # The verdict is about what ships, which is the baked path.
    shipped = results.get("baked", full)
    print(f"\nceiling  : {1000.0 / shipped:.1f} FPS   [16 ms budget => "
          f"{'OK' if shipped <= 16.0 else 'MISSED'}]  (baked)")
    if "baked" in results:
        print(f"bake won : {full - shipped:.2f} ms/frame, {full / shipped:.1f}x")
    if full > empty and {"nebula", "stars"} <= results.keys():
        nebula_share = (results["nebula"] - empty) / (full - empty)
        stars_share = (results["stars"] - empty) / (full - empty)
        print(f"share    : nebula {nebula_share * 100:.0f}%, stars {stars_share * 100:.0f}%"
              f"  (of the {full - empty:.2f} ms above the fill-rate floor)")
    print("\nRun this several times -- dynamic clocks make single samples noisy.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))

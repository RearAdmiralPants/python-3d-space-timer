#!/usr/bin/env python3
"""Render still frames of the shard to PNG, with no desktop session.

Exists because the interesting properties of the front face -- how many
distinct highlights it throws, whether the numerals survive -- are not
things a geometry assertion can see. Runs the real ShardWidget through the
real pipeline (sky, glass, HDR post) and grabs the finished framebuffer, so
what lands in the PNG is what the window would show.

    xvfb-run -s "-screen 0 1280x960x24" tools/render_still.py out.png \
        --json crystal-shard.json --angles 0,1.4,2.8

Software GL (llvmpipe under Xvfb) is slow but exact; a few seconds a frame.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from PySide6.QtCore import QCoreApplication, Qt  # noqa: E402
from PySide6.QtGui import QSurfaceFormat  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from naive_timer.shard import ShardParams, ShardWidget, default_surface_format  # noqa: E402
from naive_timer.tuning import apply_json_dict  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("out", help="output PNG; one file per angle, suffixed")
    ap.add_argument("--json", help="params file to load")
    ap.add_argument("--set", action="append", default=[],
                    help="name=value override, repeatable")
    ap.add_argument("--text", default="12:34")
    ap.add_argument("--angles", default="0",
                    help="comma-separated elapsed times to sample the sway at")
    ap.add_argument("--spin", default="0",
                    help="comma-separated idle-spin angles, one per --angles entry")
    ap.add_argument("--size", default="1280x960")
    args = ap.parse_args()

    QCoreApplication.setAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts)
    QSurfaceFormat.setDefaultFormat(default_surface_format())
    app = QApplication(sys.argv[:1])

    params = ShardParams()
    if args.json:
        with open(args.json) as fh:
            apply_json_dict(params, json.load(fh))
    for item in args.set:
        name, _, value = item.partition("=")
        if not hasattr(params, name):
            raise SystemExit(f"no such param: {name}")
        setattr(params, name, float(value))

    width, height = (int(v) for v in args.size.split("x"))
    widget = ShardWidget(None, params)
    widget.resize(width, height)
    widget.show()
    app.processEvents()
    widget.set_text(args.text)

    stem, ext = os.path.splitext(args.out)
    angles = [float(a) for a in args.angles.split(",")]
    spins = [float(a) for a in args.spin.split(",")]
    for i, elapsed in enumerate(angles):
        widget._elapsed = elapsed
        widget._spin = spins[i] if i < len(spins) else spins[-1]
        widget.update()
        app.processEvents()
        widget.repaint()
        image = widget.grabFramebuffer()
        path = args.out if len(angles) == 1 else f"{stem}-{i}{ext}"
        image.save(path)
        print(f"wrote {path}  (elapsed={elapsed})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

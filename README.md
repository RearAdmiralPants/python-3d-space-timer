# 3d-space-timer
(old name "naive-linux-timer-gui")

A simple stopwatch/timer running in Linux with naive 3D/animation features.

Built with [PySide6](https://doc.qt.io/qtforpython/) (Qt for Python). The
timekeeping logic is kept UI-free and fully unit-tested, so it can be developed
and verified without a display; the Qt layer is a thin view on top.

## Why

I just wanted to know how long it'd been since {thing} and sometimes I wanted to
know when it'd been {number} minutes.

I also like stars, nebulae, lighting, lens flares, and geometry.

## Features

**Stay-on-top toggle**: A simple checkbox keeps the window on top

**Stopwatch tab**
- Start / pause / resume / reset stopwatch
- `HH:MM:SS.cs` readout at ~60 FPS
- Lap support in the core model

**Timer tab (countdown / alarm)**
- **Duration** mode: count down a timespan — `12m`, `1h30m`, `90s`, or `25:00`
- **Alarm at** mode: count down to a clock time — `02:54`, `14:00`, `6:30pm`
  (rolls to tomorrow if the time has already passed today)
- Or start a countdown straight from the shell: `naive-timer --timer 30m`
- On reaching zero: the shard shatters, its pieces tumbling away under gravity,
  plus a gentle chime that loops quietly for ~2 minutes (configurable) or until
  you hit **Dismiss**
- Both the chime and the shatter are synthesized at runtime (no binary asset);
  point the alert at your own WAV to customize

**The readout itself**

Both tabs render the numerals as a texture on the face of a lit glass shard,
floating in a procedural starfield. It is drawn in **HDR** — the scene goes into
a floating-point buffer and is tonemapped once at the end — so the highlights on
the shard's bevels can carry far more light than the display can show, and
spill it back as glare and lens flare. See *Tuning the shard*, below.

## Prerequisites

Qt 6.5+ needs a system library that the PySide6 wheels do **not** bundle.
Without it the app aborts at startup with `Could not load the Qt platform
plugin "xcb"`. On Debian / Ubuntu / Mint:

```bash
sudo apt install libxcb-cursor0
```

This is the only thing that belongs to `sudo`. Everything Python-side goes in a
virtualenv (below) — on Mint and other distros with an *externally managed*
system Python, `sudo pip install` is both blocked and a good way to break
`apt`-managed tooling.

## Run it

```bash
./launch.sh
```

That creates `.venv` on first run, installs the project into it, and starts the
app. It re-installs whenever `pyproject.toml` changes, so a new dependency never
leaves you on a stale environment. Arguments are forwarded to the app.

On a laptop with both an integrated and a discrete GPU, OpenGL uses the
integrated one unless asked otherwise — and at 4K that can be the difference
between 17 and 140 FPS. Pick explicitly:

```bash
./launch.sh --gpu list      # what GPUs does this machine have?
./launch.sh --gpu nvidia    # the discrete GPU, via PRIME offload
./launch.sh --gpu intel     # the integrated GPU
```

Whether the discrete GPU is actually faster depends on the pairing, so measure
rather than assume — `tools/bench-gpu.sh` times the real sky shader on each GPU
at your display's resolution. See [docs/HANDOFF.md](docs/HANDOFF.md) for what
the numbers mean.

Creating the virtualenv needs `ensurepip`, which Debian and Ubuntu split into a
separate package. If `launch.sh` tells you it is missing:

```bash
sudo apt install python3-venv
```

The equivalent by hand, if you would rather not use the script:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e .
.venv/bin/python -m naive_timer
```

To install it as a real command (`naive-timer`) without touching the system
Python, use [pipx](https://pipx.pypa.io/) — it builds a private virtualenv per
application and puts the entry point on your `PATH`:

```bash
pipx install .
naive-timer
```

## Menu entry and panel icon

By default a Qt app started as `python -m ...` inherits its window identity from
the interpreter, so the panel shows the generic Python icon — indistinguishable
from every other Python tool you have running. To give it its own:

```bash
./tools/install-desktop.sh              # ~/.local/share, no root needed
./tools/install-desktop.sh --uninstall
```

That installs the icon at every size into the `hicolor` theme and a desktop
entry whose `StartupWMClass` matches the app's `WM_CLASS`, which is what lets
the panel map the running window back to its icon. Already-running instances
keep the old icon until restarted.

The menu entry launches with `--no-panel` (no dev tuning window); edit the
`Exec` line in `tools/install-desktop.sh` if you'd rather have it.

## Starting a timer from the command line

Kick off a countdown without touching the GUI:

```bash
naive-timer --timer 30m                       # 30 minutes
naive-timer --timer 1h30m                     # 1.5 hours
naive-timer --timer 25:00                     # 25 minutes (mm:ss)
naive-timer --timer 6:30pm                    # alarm at 6:30 PM
naive-timer --timer 10m --json green-nebula.json   # timer + custom look
```

The app opens on the Timer tab with the countdown already running, as though the
value had been typed into the text box and **Start** clicked. The value is parsed
as a duration first (`12m`, `1h30m`, `25:00`, bare number = minutes), then as an
alarm time (`02:54`, `6:30pm`) — the same parser the Timer tab uses. If neither
works, an error is written to stderr and the app exits without opening.

## Develop / test

The stopwatch, countdown, and sound models have no GUI dependency and can be
tested anywhere, with no third-party packages:

```bash
python -m unittest discover -s tests -v
```

`tests/test_gui_smoke.py` additionally constructs the Qt objects. It exists
because the model tests can be entirely green while the app crashes on startup
— which is exactly what happened once.

It has two tiers, because Qt's `offscreen` platform has **no OpenGL**, and
constructing a `QOpenGLWidget` under it *segfaults* rather than raising:

```bash
# Tier 1 only. Alert player, shard geometry, text texture. Skips the GL tier.
.venv/bin/python -m unittest discover -s tests -v

# Both tiers, still headless: a virtual X server gives real GL via Mesa.
xvfb-run -a .venv/bin/python -m unittest discover -s tests -v
```

With a display attached, both tiers run automatically. Use the `xvfb-run` form
in CI and cloud containers — otherwise the shard is never actually constructed.

## Tuning the shard

The timer text is rendered as a texture on the face of a lit glass shard. Every
shader in `src/naive_timer/shaders/` is **hot-reloaded**: edit one while the app
runs and it recompiles on save. A shader that fails to compile prints the error
and leaves the previous one running, so you cannot break the app from there.

| file | what it draws |
|------|---------------|
| `shard.vert` / `shard.frag` | the glass, its numerals, and the shatter |
| `sky.vert` / `sky.frag` | the starfield and nebula (also the cubemap bake) |
| `post_bright.frag` | picks out the pixels brighter than white |
| `post_blur.frag` | separable Gaussian, run repeatedly at a growing step |
| `post_flare.frag` | lens flare: ghosts, halo and streaks |
| `post_composite.frag` | adds the glare and applies the one tonemap |

`sky.vert` is the fullscreen triangle, so every post stage shares it.

```bash
./launch.sh                      # the panel is on by default
NAIVE_TIMER_TUNE=0 ./launch.sh   # ... without it
```

That adds a scrollable panel with live sliders for every uniform — light
position and intensity, specular power, Fresnel, glow, an etched↔emissive blend,
camera and sky, and the frame-wide tonemap and glare controls — plus font and
colour pickers. **Print params** dumps the current values to the console in a
form you can paste back into `ShardParams`.

Two of those sliders are a pair worth knowing about. `light_intensity` scales
the light's radiance, and `rolloff` sets the tonemap's shoulder — the curve
saturates at radiance `1 / (1 - rolloff)` and clips beyond, so raising intensity
without also raising rolloff turns the highlight into a flat white blob rather
than a bright one. `exposure` re-seats the whole frame once both have moved.

The lens flare needs no switch of its own: it is built from the same
brighter-than-white buffer as the glare, so it fades in by itself as the
highlights get hot enough to clear `bloom_threshold` and `flare_threshold`.

Sliders ignore the mouse wheel unless you click one first — otherwise scrolling
the panel would rewrite every value the pointer crossed.

A look saved from that panel is just a JSON file, and `--json` loads one at
startup — the shard wears it from the first frame:

```bash
./launch.sh --json green-nebula.json          # load a saved look
./launch.sh --no-panel                         # tuning env set, but no panel
NAIVE_TIMER_TUNE=1 ./launch.sh --json green-nebula.json  # load it, then tweak
```

`--json` only loads params; it no longer touches the panel. The panel is
governed solely by `NAIVE_TIMER_TUNE`, and `--no-panel` forces it off — so you
can load a look for a clean screenshot, or load one as a starting point and keep
dragging sliders. Run `naive-timer --help` for the full list. (Unknown keys in
the file are ignored, so a look saved by an older or newer build still loads what
it can.)

## Layout

```
src/naive_timer/
  stopwatch.py   # pure stopwatch model (no Qt) — unit-tested
  countdown.py   # pure countdown model + duration/alarm parsing — unit-tested
  sound.py       # runtime chime + shatter WAV synthesis (stdlib only) — unit-tested
  app.py         # PySide6 tabbed window (thin view)
  shard.py       # the OpenGL widget: geometry, HDR passes, ShardParams
  tuning.py      # dev-only live slider panel (NAIVE_TIMER_TUNE=1)
  shaders/       # GLSL, hot-reloaded on save — see "Tuning the shard"
  __main__.py    # enables `python -m naive_timer`
tests/
  test_stopwatch.py
  test_countdown.py
  test_sound.py
  test_cli.py        # CLI parsing + JSON param loading (headless, no Qt)
  test_gui_smoke.py  # constructs the Qt widgets offscreen (needs PySide6)
```

The separation is deliberate: logic changes can be verified headlessly (e.g. in
a cloud/CI session), while the visual layer is reviewed by running it locally.

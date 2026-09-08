"""Runtime generation of the alert chime and the shatter sound.

Both are synthesized into WAV files at runtime rather than committed as binary
assets. That is not only about repo hygiene: the reference shatter recordings
were captured from YouTube, and a processed copy of a copyrighted recording is
still a derivative work. Synthesis has no provenance problem, and it turns
"quieter" and "longer" into parameters rather than ffmpeg passes.

Pure stdlib (``wave``/``math``/``struct``/``random``), so it can be unit-tested
headlessly with no audio device and no Qt.

Note: ``QSoundEffect`` decodes **only uncompressed WAV**. It errors on FLAC.
"""

from __future__ import annotations

import math
import os
import random
import struct
import tempfile
import wave
from typing import Sequence

# Match the PipeWire graph's native format exactly, so the alert stream needs
# neither a resampler nor a channel upmix. 48 kHz stereo is the near-universal
# PipeWire default and is what both of this project's dev machines run; at
# 44100 mono every alert dragged a rate conversion and an upmix behind it.
#
# Synthesis stays mono internally -- only the WAV writers duplicate to stereo.
# Nothing about the sound changes, just the container.
SAMPLE_RATE = 48000
CHANNELS = 2
_DEFAULT_NOTES = (587.33, 880.0)  # D5, A5 — a soft, non-jarring interval

# Bump when the synthesis changes, so cached files in the temp dir are not
# reused after an edit. Without this, tweaking a shatter parameter appears to
# do nothing until you delete the WAV by hand. (v4: 48 kHz stereo.)
_CACHE_VERSION = 4


def generate_chime_wav(
    path: str,
    notes: Sequence[float] = _DEFAULT_NOTES,
    note_seconds: float = 0.45,
    tail_silence: float = 1.1,
    amplitude: float = 0.28,
    sample_rate: int = SAMPLE_RATE,
) -> str:
    """Write a looping-friendly chime to ``path`` and return it.

    The clip is: each note played in sequence with a smooth fade in/out,
    followed by ``tail_silence`` seconds of quiet — so looping it yields a
    gentle, spaced repetition rather than a continuous drone.
    """
    frames = bytearray()
    max_amp = int(amplitude * 32767)

    for freq in notes:
        n = int(note_seconds * sample_rate)
        for i in range(n):
            # Raised-cosine envelope: no clicks at note boundaries.
            env = 0.5 * (1 - math.cos(2 * math.pi * i / n))
            sample = env * math.sin(2 * math.pi * freq * i / sample_rate)
            value = int(sample * max_amp)
            frames += struct.pack("<" + "h" * CHANNELS, *([value] * CHANNELS))

    frames += b"\x00" * (2 * CHANNELS) * int(tail_silence * sample_rate)

    with wave.open(path, "wb") as wav:
        wav.setnchannels(CHANNELS)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(bytes(frames))
    return path


def _write_wav(path: str, samples: Sequence[float], sample_rate: int) -> str:
    """Write float samples in [-1, 1] as 16-bit PCM, duplicated to CHANNELS.

    The synthesis above is mono; the duplication happens here so the file
    matches the audio graph's format and needs no upmix at playback.
    """
    frames = bytearray()
    for value in samples:
        clipped = max(-1.0, min(1.0, value))
        packed = int(clipped * 32767)
        frames += struct.pack("<" + "h" * CHANNELS, *([packed] * CHANNELS))

    with wave.open(path, "wb") as wav:
        wav.setnchannels(CHANNELS)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(bytes(frames))
    return path


def _add_resonant_grain(
    buf: list,
    onset: int,
    freq: float,
    decay_samples: float,
    gain: float,
    burst_samples: int,
    rng: random.Random,
    sample_rate: int,
) -> None:
    """Mix one noise-excited resonator into ``buf``, in place.

    A two-pole resonator, ``y[n] = x[n] + a1*y[n-1] + a2*y[n-2]``, driven by a
    short burst of white noise. This is the single most important thing about
    the whole shatter, so it is worth being precise about why.

    Excited by an *impulse* (``burst_samples == 1``) the resonator's output is
    an exponentially decaying sine -- a pure tone, a struck bar, a wind chime.
    Excited by a short *noise burst* it produces the same spectral peak, but
    smeared: a click with a colour rather than a pitch. A glass fragment is the
    second thing. It is a broadband tick that happens to resonate somewhere,
    not a note.

    The previous version of this file used pure decaying sines. Measured with a
    Wiener-entropy (spectral flatness) estimate, it scored 0.002 -- essentially
    a chord -- and it sounded like one.

    Excitation is scaled so output RMS is roughly ``gain`` regardless of decay
    or burst length: a white-noise-driven resonator has output RMS proportional
    to ``sqrt(burst) / sqrt(1 - r^2)``, so divide it back out. Without this,
    short grains vanish and long ones dominate.

    Costs one multiply-add per sample and stops as soon as the ring-down is
    inaudible, so several hundred grains stay well inside a tenth of a second.
    """
    total = len(buf)
    if onset >= total or decay_samples <= 0.0 or freq >= 0.5 * sample_rate:
        return

    r = math.exp(-1.0 / decay_samples)
    omega = 2.0 * math.pi * freq / sample_rate
    a1 = 2.0 * r * math.cos(omega)
    a2 = -(r * r)

    burst = max(1, burst_samples)

    # The two excitations need different scaling, and getting this wrong is
    # silent rather than loud -- so it is spelled out.
    #
    # Impulse: the response is `excite * r^n * sin((n+1)w) / sin(w)`, so peak
    # amplitude is `excite / sin(w)`. Multiply it back to make `gain` mean peak.
    # (Using the noise scaling here would make a long decay near-inaudible: it
    # carries a `sqrt(1 - r^2)` factor, which for a 0.4 s ring is about 0.01.)
    #
    # Noise: output RMS goes as `sqrt(burst) / sqrt(1 - r^2)`, so divide that
    # out and `gain` means RMS. Without it, short grains vanish under long ones.
    impulse = burst == 1
    if impulse:
        excite = gain * math.sin(omega)
    else:
        excite = gain * math.sqrt(max(1e-9, 1.0 - r * r)) / math.sqrt(burst)

    # exp(-9.2) ~ 1e-4: past here the grain is inaudible under everything else.
    stop = min(total, onset + burst + int(9.2 * decay_samples))

    y1 = 0.0
    y2 = 0.0
    for i in range(onset, stop):
        if i - onset >= burst:
            x = 0.0
        elif impulse:
            x = excite  # deterministic: a random sign would randomise the note
        else:
            x = excite * rng.uniform(-1.0, 1.0)
        y = x + a1 * y1 + a2 * y2
        y2 = y1
        y1 = y
        buf[i] += y


def generate_shatter_wav(
    path: str,
    seconds: float = 3.0,
    amplitude: float = 0.22,
    seed: int = 7,
    resonances: int = 22,
    fragments: int = 720,
    strike_hz: float = 1760.0,
    strike_decay: float = 0.42,
    strike_gain: float = 0.16,
    sample_rate: int = SAMPLE_RATE,
) -> str:
    """Synthesize breaking glass and write it to ``path``.

    Four layers, which is roughly what breaking glass actually is:

    0. A **strike tone**: one pure decaying sine, the note of the pane being
       struck, ringing on underneath the break. This is the only deliberately
       tonal thing in the clip -- an impulse-excited resonator, where every
       other grain is noise-excited (see ``_add_resonant_grain``). It is
       ``strike_hz``, defaulting to A6: an octave above the A5 in the alert
       chime's D5/A5, so the two sounds are related rather than merely adjacent.

       ``strike_gain`` is small because it is a *peak* amplitude while the
       clatter grains are scaled to RMS -- 0.45 here sounds roughly four times
       louder than a fragment of nominal gain 0.45, not equal to it. Above about
       0.16 the tone swamps the clatter and the clip drifts back toward a chime:
       measured, spectral flatness falls from 0.09 to 0.02 and the zero-crossing
       rate drops out of the bottom of the 6.2-9.1 kHz glass band.
    1. An **impact transient**: a very short broadband noise burst, one-zero
       high-passed so it reads as a sharp crack rather than a thud.
    2. **Resonant partials**: the plate ringing as it fractures. Inharmonic
       matters -- harmonic partials sound like a bell, not like glass.
    3. **Fragment tinkles**: a long, thinning scatter of tiny grains as the
       pieces fall and settle. This is what makes it *sound* long rather than
       merely be long, and it is why time-stretching a recording sounds
       underwater: stretching smears the grains instead of spacing them.

    ``fragments`` must scale with ``seconds``. Density is what makes the scatter
    fuse into clatter; hold the count fixed while lengthening the clip and the
    tail thins until the grains resolve individually and chime again. Roughly
    240 fragments per second holds.

    Deterministic for a given ``seed`` -- the same alert every time, and a test
    can pin the bytes. ``amplitude`` is the peak after normalisation, so it maps
    directly to headroom: 0.22 is about -13 dBFS.
    """
    rng = random.Random(seed)
    total = int(seconds * sample_rate)
    buf = [0.0] * total

    # 0. Strike tone. burst_samples=1 is an impulse, so the resonator rings as a
    # pure sine -- the one place in this file where that is what we want.
    if strike_gain > 0.0 and strike_hz > 0.0:
        _add_resonant_grain(
            buf,
            onset=0,
            freq=strike_hz,
            decay_samples=strike_decay * sample_rate,
            gain=strike_gain,
            burst_samples=1,
            rng=rng,
            sample_rate=sample_rate,
        )

    # 1a. Impact transient: the crack.
    burst = int(0.05 * sample_rate)
    previous = 0.0
    for i in range(min(burst, total)):
        noise = rng.uniform(-1.0, 1.0)
        highpassed = noise - previous  # one-zero high-pass: kills the thud
        previous = noise
        buf[i] += highpassed * math.exp(-i / (0.012 * sample_rate))

    # 1b. Impact body: the crunch. Low-passed noise over a longer decay. The
    # crack alone is a click; this is what gives the break some weight.
    body = int(0.18 * sample_rate)
    lowpassed = 0.0
    for i in range(min(body, total)):
        noise = rng.uniform(-1.0, 1.0)
        lowpassed += 0.10 * (noise - lowpassed)  # one-pole low-pass
        buf[i] += 2.2 * lowpassed * math.exp(-i / (0.045 * sample_rate))

    # 2. Resonant partials: the pane ringing as it fractures.
    #
    # A third sit low, in the body of the pane. Without those the clip is all
    # treble and reads as a cymbal: an all-high partial set measured a
    # zero-crossing rate of 13 kHz where real breaking glass sits between 6 and
    # 9 kHz. But the low band starts at 380 Hz, not 180 -- a thin pane has very
    # little energy below that, and what was down there hummed.
    #
    # Decays are short. Anything ringing for half a second reads as a struck
    # bar, however inharmonic its neighbours are.
    for index in range(resonances):
        low_body = index % 3 == 0
        _add_resonant_grain(
            buf,
            onset=int(rng.uniform(0.0, 0.10) * sample_rate),
            freq=(
                rng.uniform(380.0, 1400.0) if low_body
                else rng.uniform(1400.0, 6500.0)
            ),
            decay_samples=rng.uniform(0.02, 0.16) * sample_rate,
            gain=rng.uniform(0.25, 0.7) if low_body else rng.uniform(0.15, 0.6),
            burst_samples=int(rng.uniform(0.0010, 0.0035) * sample_rate),
            rng=rng,
            sample_rate=sample_rate,
        )

    # 3. Fragment tinkles: shards striking each other and settling.
    #
    # Density is the point. A hundred grains resolve individually and chime; a
    # few hundred fuse into clatter. Each is 2-18 ms, which is a tick, not a
    # note -- long enough to carry a resonance, too short to carry a pitch.
    for _ in range(fragments):
        # Bias onsets early: most fragments land soon, a few keep skittering.
        when = rng.random() ** 1.6 * 0.90
        _add_resonant_grain(
            buf,
            onset=int(when * seconds * sample_rate),
            freq=rng.uniform(1800.0, 9200.0),
            decay_samples=rng.uniform(0.002, 0.018) * sample_rate,
            gain=rng.uniform(0.05, 0.35) * (1.0 - when * 0.7),
            burst_samples=int(rng.uniform(0.0003, 0.0012) * sample_rate),
            rng=rng,
            sample_rate=sample_rate,
        )

    # Fade the first 1.5 ms. The clip opens on a full-amplitude noise sample --
    # a step discontinuity, which is a click on top of the crack. The previous
    # version had no fade here and passed its no-click test only because the
    # seeded first draw happened to land at -0.013; other seeds clicked.
    #
    # 1.5 ms is below the ~2 ms where an attack stops being heard as immediate,
    # so the crack keeps its edge.
    #
    # This has to happen *before* normalisation: the clip's peak sits inside
    # the impact transient, so fading afterwards would scale the peak down and
    # `amplitude` would no longer mean peak level.
    attack = min(int(0.0015 * sample_rate), total)
    for i in range(attack):
        buf[i] *= 0.5 * (1.0 - math.cos(math.pi * i / attack))

    # Normalise to unity, then scale. Doing it in this order makes `amplitude`
    # mean peak level regardless of how the layers happened to sum.
    peak = max(abs(v) for v in buf) or 1.0
    scale = amplitude / peak
    for i in range(total):
        buf[i] *= scale

    # Fade the last 60 ms so the clip cannot click when it stops.
    fade = min(int(0.06 * sample_rate), total)
    for i in range(fade):
        buf[total - fade + i] *= 0.5 * (1.0 + math.cos(math.pi * i / fade))

    return _write_wav(path, buf, sample_rate)


def _cached(name: str, generate) -> str:
    """A per-user temp path, generated once.

    The uid is in the filename because /tmp is shared: the second user to run
    the app would otherwise collide with the first user's file and be unable to
    overwrite it.
    """
    uid = getattr(os, "getuid", lambda: 0)()
    path = os.path.join(
        tempfile.gettempdir(), f"naive_timer_{name}_{uid}_v{_CACHE_VERSION}.wav"
    )
    if not os.path.exists(path):
        generate(path)
    return path


def default_chime_path() -> str:
    """Generate (once) and return a temp-file path to the default chime."""
    return _cached("chime", generate_chime_wav)


def default_shatter_path() -> str:
    """Generate (once) and return a temp-file path to the shatter sound."""
    return _cached("shatter", generate_shatter_wav)

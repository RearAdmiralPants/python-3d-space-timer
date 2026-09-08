"""Headless tests for the synthesized chime and shatter.

Pure stdlib synthesis, so these need no audio device, no Qt and no display.
"""

import math
import os
import struct
import sys
import tempfile
import unittest
import wave

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from naive_timer.sound import CHANNELS, SAMPLE_RATE, generate_chime_wav, generate_shatter_wav


def _read(path):
    """Return one channel's samples plus the rate.

    The files are written at CHANNELS wide with every channel identical (the
    synthesis is mono), so the analysis tests below only need channel 0.
    """
    with wave.open(path, "rb") as wav:
        frames = wav.getnframes()
        rate = wav.getframerate()
        channels = wav.getnchannels()
        raw = struct.unpack(f"<{frames * channels}h", wav.readframes(frames))
    return [v / 32768.0 for v in raw[::channels]], rate


class SoundTest(unittest.TestCase):
    def test_generates_valid_wav(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "chime.wav")
            generate_chime_wav(path, note_seconds=0.1, tail_silence=0.1)
            self.assertTrue(os.path.exists(path))
            with wave.open(path, "rb") as wav:
                self.assertEqual(wav.getnchannels(), CHANNELS)
                self.assertEqual(wav.getsampwidth(), 2)
                self.assertEqual(wav.getframerate(), SAMPLE_RATE)
                self.assertGreater(wav.getnframes(), 0)


class ShatterTest(unittest.TestCase):
    def test_generates_valid_wav_of_the_requested_length(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "shatter.wav")
            generate_shatter_wav(path, seconds=0.4, resonances=4, fragments=6)
            with wave.open(path, "rb") as wav:
                self.assertEqual(wav.getnchannels(), CHANNELS)
                self.assertEqual(wav.getsampwidth(), 2)
                self.assertEqual(wav.getframerate(), SAMPLE_RATE)
                self.assertEqual(wav.getnframes(), int(0.4 * SAMPLE_RATE))

    def test_amplitude_is_the_peak_after_normalisation(self) -> None:
        """`amplitude` must mean headroom, whatever the layers summed to."""
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "shatter.wav")
            generate_shatter_wav(path, seconds=0.5, amplitude=0.22)
            samples, _rate = _read(path)
            peak = max(abs(v) for v in samples)
            self.assertAlmostEqual(peak, 0.22, delta=0.01)

    def test_quieter_than_the_reference_recordings(self) -> None:
        """The captured clips peaked near -6 dBFS and were 'a bit jarring'."""
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "shatter.wav")
            generate_shatter_wav(path, seconds=0.5)
            samples, _rate = _read(path)
            peak_db = 20 * math.log10(max(abs(v) for v in samples))
            self.assertLess(peak_db, -10.0, "shatter is louder than intended")

    def test_is_deterministic_for_a_seed(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            a = generate_shatter_wav(os.path.join(d, "a.wav"), seconds=0.3, seed=5)
            b = generate_shatter_wav(os.path.join(d, "b.wav"), seconds=0.3, seed=5)
            c = generate_shatter_wav(os.path.join(d, "c.wav"), seconds=0.3, seed=6)
            self.assertEqual(open(a, "rb").read(), open(b, "rb").read())
            self.assertNotEqual(open(a, "rb").read(), open(c, "rb").read())

    def test_does_not_click_at_either_end(self) -> None:
        """A clip that starts or stops mid-waveform pops on playback.

        Swept across seeds on purpose. This assertion used to hold for the
        default seed alone, by luck: there was no fade-in, so the first sample
        was just whatever the RNG drew, and most draws click.
        """
        for seed in range(8):
            with tempfile.TemporaryDirectory() as d:
                path = os.path.join(d, "shatter.wav")
                generate_shatter_wav(path, seconds=0.6, seed=seed)
                samples, _rate = _read(path)
                self.assertLess(abs(samples[0]), 0.02, f"click at onset, seed={seed}")
                self.assertLess(
                    abs(samples[-1]), 0.005, f"no fade-out at the tail, seed={seed}"
                )

    def test_energy_decays_rather_than_sustaining(self) -> None:
        """Breaking glass rings down. A drone would mean the envelopes broke."""
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "shatter.wav")
            generate_shatter_wav(path, seconds=1.2)
            samples, rate = _read(path)

            def rms(chunk):
                return math.sqrt(sum(v * v for v in chunk) / max(len(chunk), 1))

            head = rms(samples[: int(0.1 * rate)])
            tail = rms(samples[-int(0.2 * rate) :])
            self.assertGreater(head, tail * 5, "the shatter does not ring down")

    def test_is_bright_like_glass_not_dull_like_a_thud(self) -> None:
        """Zero-crossing rate is a cheap brightness proxy.

        The reference recordings measured 6.2-9.1 kHz. An all-high-partial
        version of this synth measured 13.2 kHz and read as a cymbal, which is
        why a third of the resonances now sit in the body of the pane.
        """
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "shatter.wav")
            generate_shatter_wav(path, seconds=1.5)
            samples, rate = _read(path)
            crossings = sum(
                1
                for i in range(1, len(samples))
                if (samples[i - 1] < 0) != (samples[i] < 0)
            )
            zcr = crossings / (len(samples) / rate)
            self.assertGreater(zcr, 4000, "too dull to be glass")
            self.assertLess(zcr, 12000, "too bright; reads as a cymbal")


if __name__ == "__main__":
    unittest.main()

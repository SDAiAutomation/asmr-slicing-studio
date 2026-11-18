"""Musique de fond synthwave générée par code : 100 % libre de droits.

Jamais la radio ni la BO du jeu (Content ID).
"""
import wave
from pathlib import Path

import numpy as np

SR = 44100


def _note(freq, dur, sr=SR):
    t = np.arange(int(dur * sr)) / sr
    # dent de scie adoucie + sinus = pad chaud
    saw = 2 * (t * freq - np.floor(0.5 + t * freq))
    sig = 0.55 * np.sin(2 * np.pi * freq * t) + 0.25 * saw
    env = np.minimum(1, t / 0.4) * np.minimum(1, (dur - t) / 0.5)
    return sig * env


def make_music(duration: float, dest: Path, bpm: int = 100, seed: int = 7) -> Path:
    rng = np.random.default_rng(seed)
    n = int(duration * SR) + SR
    mix = np.zeros(n)
    beat = 60 / bpm
    bar = 4 * beat
    # Am - F - C - G
    chords = [[220.0, 261.6, 329.6], [174.6, 220.0, 261.6], [196.0, 261.6, 329.6], [196.0, 246.9, 293.7]]
    basses = [55.0, 43.65, 65.41, 49.0]
    t0, k = 0.0, 0
    while t0 < duration + bar:
        i0 = int(t0 * SR)
        pad = sum(_note(f, bar) for f in chords[k % 4]) * 0.12
        seg = mix[i0:i0 + len(pad)]
        seg += pad[:len(seg)]
        # basse en croches
        for j in range(8):
            ts = t0 + j * beat / 2
            b = _note(basses[k % 4], beat / 2 * 0.9) * 0.22
            a = int(ts * SR)
            s = mix[a:a + len(b)]
            s += b[:len(s)]
        t0 += bar
        k += 1
    # kick sur chaque temps + charley
    tk = np.arange(int(0.25 * SR)) / SR
    kick = np.sin(2 * np.pi * (55 + 90 * np.exp(-tk * 30)) * tk) * np.exp(-tk * 12) * 0.6
    hat = rng.standard_normal(int(0.05 * SR)) * np.exp(-np.arange(int(0.05 * SR)) / SR * 80) * 0.06
    b = 0.0
    while b < duration + 1:
        a = int(b * SR)
        s = mix[a:a + len(kick)]
        s += kick[:len(s)]
        h = int((b + beat / 2) * SR)
        s2 = mix[h:h + len(hat)]
        s2 += hat[:len(s2)]
        b += beat
    mix = mix[: int(duration * SR)]
    fade = int(1.5 * SR)
    mix[-fade:] *= np.linspace(1, 0, fade)
    mix = mix / (np.abs(mix).max() + 1e-9) * 0.8
    stereo = np.repeat((mix * 32767).astype(np.int16)[:, None], 2, axis=1)
    with wave.open(str(dest), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(stereo.tobytes())
    return dest

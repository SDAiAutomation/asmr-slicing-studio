"""Procedural ASMR soundtrack built from the same plan.json as the animation.

Six event layers per cut (contact, penetration, traversal, end of cut,
separation, surface contact), material-specific, seeded variation, short room
reverb, five mixable buses with three presets (audio_preset), loudness-normalised below clipping.

Usage:
    python make_audio.py plan.json out_dir
Writes out_dir/soundtrack.wav (48 kHz stereo 16-bit) and out_dir/audio_events.json.
Optional: put real recordings in assets/samples/<profile>/<event>_*.wav to
replace a procedural layer (mono or stereo, any rate is resampled).
"""
import json
import os
import sys
import wave

import numpy as np
from scipy import ndimage, signal

SR = 48000

# Per-material sound design. Frequencies in Hz, durations in seconds.
PROFILES = {
    "steak": {   # dense, moist meat: thick squelch, little crackle, deep board knock
        "contact": dict(band=(2000, 7000), dur=0.005, gain=0.25),
        "body": dict(band=(500, 2500), gain=0.45, crackle=15, crackle_gain=0.15,
                     squish=(90, 420), squish_gain=0.60),
        "end": dict(f0=190, decay=0.05, gain=0.50, click=0.25),
        "separate": dict(band=(200, 1200), dur=0.08, gain=0.30),
        "land": dict(f0=110, decay=0.07, gain=0.28),
    },
    "cake": {    # soft sponge: airy, quiet
        "contact": dict(band=(2500, 8000), dur=0.004, gain=0.15),
        "body": dict(band=(800, 4000), gain=0.35, crackle=30, crackle_gain=0.20,
                     squish=(150, 600), squish_gain=0.25),
        "end": dict(f0=230, decay=0.04, gain=0.40, click=0.20),
        "separate": dict(band=(400, 2000), dur=0.06, gain=0.20),
        "land": dict(f0=140, decay=0.05, gain=0.20),
    },
    "mango": {   # soft wet chop: juicy squelch, fibrous crackle, soft board tap
        "contact": dict(band=(2500, 9000), dur=0.006, gain=0.35),
        "body": dict(band=(700, 3800), gain=0.55, crackle=55, crackle_gain=0.35,
                     squish=(120, 520), squish_gain=0.45),
        "end": dict(f0=210, decay=0.05, gain=0.45, click=0.25),
        "separate": dict(band=(300, 1800), dur=0.07, gain=0.30),
        "land": dict(f0=120, decay=0.06, gain=0.22),
    },
    "soap": {    # dry scrape: steady bright noise, no squish
        "contact": dict(band=(3000, 10000), dur=0.004, gain=0.30),
        "body": dict(band=(1800, 7000), gain=0.50, crackle=20, crackle_gain=0.15,
                     squish=(400, 900), squish_gain=0.10),
        "end": dict(f0=260, decay=0.035, gain=0.40, click=0.35),
        "separate": dict(band=(800, 3000), dur=0.04, gain=0.20),
        "land": dict(f0=180, decay=0.04, gain=0.25),
    },
    "jelly": {
        "contact": dict(band=(1500, 6000), dur=0.004, gain=0.15),
        "body": dict(band=(300, 1800), gain=0.40, crackle=8, crackle_gain=0.05,
                     squish=(80, 380), squish_gain=0.70),
        "end": dict(f0=170, decay=0.05, gain=0.30, click=0.10),
        "separate": dict(band=(150, 900), dur=0.12, gain=0.45),
        "land": dict(f0=90, decay=0.09, gain=0.30),
    },
    "chocolate": {  # crisp snap
        "contact": dict(band=(3000, 12000), dur=0.005, gain=0.40),
        "body": dict(band=(1500, 9000), gain=0.25, crackle=140, crackle_gain=0.70,
                     squish=(400, 900), squish_gain=0.0),
        "end": dict(f0=300, decay=0.03, gain=0.45, click=0.50),
        "separate": dict(band=(1500, 6000), dur=0.03, gain=0.35),
        "land": dict(f0=220, decay=0.03, gain=0.30),
    },
}


def bandnoise(rng, n, lo, hi):
    x = rng.standard_normal(n + 512)
    sos = signal.butter(4, [lo, min(hi, SR / 2 - 100)], btype="band", fs=SR, output="sos")
    return signal.sosfilt(sos, x)[512:]


def env_ad(n, attack=0.1, curve=2.0):
    t = np.linspace(0, 1, n, endpoint=False)
    a = max(1, int(n * attack))
    e = np.ones(n)
    e[:a] = np.linspace(0, 1, a)
    e[a:] = (1 - (t[a:] - t[a]) / (1 - t[a] + 1e-9)) ** curve
    return e


def tone(rng, f0, decay, n_extra=0.0):
    n = int(SR * decay * 6)
    t = np.arange(n) / SR
    f = f0 * rng.uniform(0.9, 1.1)
    y = (np.sin(2 * np.pi * f * t) + 0.4 * np.sin(2 * np.pi * f * 2.7 * t)) * np.exp(-t / decay)
    return y


def load_samples(profile):
    root = os.path.join(os.path.dirname(__file__), "..", "assets", "samples", profile)
    out = {}
    if not os.path.isdir(root):
        return out
    for fn in sorted(os.listdir(root)):
        if not fn.lower().endswith(".wav"):
            continue
        ev = fn.split("_")[0]
        with wave.open(os.path.join(root, fn)) as w:
            sr, ch, sw = w.getframerate(), w.getnchannels(), w.getsampwidth()
            raw = w.readframes(w.getnframes())
        if sw != 2:
            continue
        x = np.frombuffer(raw, np.int16).astype(float) / 32768
        x = x.reshape(-1, ch).mean(1)
        if sr != SR:
            x = signal.resample_poly(x, SR, sr)
        out.setdefault(ev, []).append(x)
    return out


def place(buf, start_s, y, pan, gain=1.0):
    i = int(start_s * SR)
    if i >= buf.shape[0] or i < 0:
        return
    y = y[: buf.shape[0] - i] * gain
    l = np.cos((pan + 1) * np.pi / 4)
    r = np.sin((pan + 1) * np.pi / 4)
    buf[i:i + len(y), 0] += y * l
    buf[i:i + len(y), 1] += y * r


def pick(samples, ev, rng):
    lst = samples.get(ev)
    if not lst:
        return None
    y = lst[rng.integers(len(lst))]
    rate = rng.uniform(0.94, 1.06)  # pitch/time variation so repeats are not identical
    return signal.resample(y, max(8, int(len(y) / rate)))


def chop(rng, P, samples, dur, nh=1):
    """contact + body (penetration/traversal) + board tap, as one buffer."""
    c, b, e = P["contact"], P["body"], P["end"]
    n = int(SR * (dur + 0.25))
    out = np.zeros(n)
    y = pick(samples, "contact", rng)
    if y is None:
        m = int(SR * c["dur"])
        y = bandnoise(rng, m, *c["band"]) * env_ad(m, 0.05, 3)
    out[:len(y)] += y[:n] * c["gain"]
    m = int(SR * dur)
    y = pick(samples, "body", rng)
    if y is None:
        body = bandnoise(rng, m, *b["band"]) * b["gain"] + bandnoise(rng, m, *b["squish"]) * b["squish_gain"]
        cr = np.zeros(m)
        for pos in rng.integers(0, max(1, m - 200), size=rng.poisson(b["crackle"] * dur * (1 + 0.5 * np.log2(nh)))):
            k = int(rng.integers(40, 160))
            cr[pos:pos + k] += rng.standard_normal(len(cr[pos:pos + k])) * np.exp(-np.arange(len(cr[pos:pos + k])) / 25)
        y = (body + cr * b["crackle_gain"]) * np.minimum(1, np.linspace(0, 1, m) * 4)
    i0 = int(SR * 0.004)
    out[i0:i0 + len(y)] += y[: n - i0]
    y = pick(samples, "end", rng)
    if y is None:
        y = tone(rng, e["f0"], e["decay"])
    y = y[: int(SR * 0.09)] * np.linspace(1, 0, min(len(y), int(SR * 0.09))) ** 2
    out[m:m + len(y)] += y[: n - m] * e["gain"]
    return out[: m + len(y)]


def whoosh(rng, dur=0.16):
    """Knife approach: a soft air swell, quiet (ASMR stays in front)."""
    n = int(SR * dur)
    y = bandnoise(rng, n, 500, 3500)
    return y * np.sin(np.linspace(0, np.pi, n)) ** 2 * 0.25


# Sound presets: one gain per bus. The cutting sounds always lead; 0 mutes a bus.
BUSES = ("cutting_asmr", "piece_impacts", "musical_notes", "ambient_layer", "reward_fx")
AUDIO_PRESETS = {
    "asmr_pure": {"cutting_asmr": 1.0, "piece_impacts": 1.0, "musical_notes": 0.0, "ambient_layer": 0.35, "reward_fx": 0.0},
    "musical_asmr": {"cutting_asmr": 1.0, "piece_impacts": 0.95, "musical_notes": 0.42, "ambient_layer": 0.5, "reward_fx": 0.45},
    "rhythmic_satisfying": {"cutting_asmr": 0.85, "piece_impacts": 0.95, "musical_notes": 0.95, "ambient_layer": 0.9, "reward_fx": 0.8},
}
LOUDNESS_TARGET = -16.0     # LUFS, the usual level for short-form social video

# Musical layer: everything is placed on the plan's own frames, so it stays locked to the blade.
PROGRESSION = [(0, 3, 7, 10), (-4, 0, 3, 7), (3, 7, 10, 14), (-2, 2, 5, 9)]  # Am7 Fmaj7 Cmaj7 G7 (semitones)
ROOTS = [110.0, 98.0, 123.47, 87.31]                                         # low on purpose: never arcade
MOTIFS = [(0, 1, 2, 3), (0, 2, 1, 3), (1, 2, 3, 2), (0, 3, 2, 1), (2, 1, 3, 0)]


def pluck(f, dur=0.5, bright=0.5):
    """Warm felt mallet: mostly fundamental, a whisper of overtone, gentle low-passed attack."""
    t = np.arange(int(SR * dur)) / SR
    y = (np.sin(2 * np.pi * f * t) * np.exp(-t / 0.22)
         + 0.22 * bright * np.sin(2 * np.pi * f * 2 * t) * np.exp(-t / 0.1)
         + 0.05 * bright * np.sin(2 * np.pi * f * 3.98 * t) * np.exp(-t / 0.03))
    a = int(SR * 0.006)
    y[:a] *= np.linspace(0, 1, a)
    return signal.sosfilt(signal.butter(2, 2600, "low", fs=SR, output="sos"), y)


def pad(freqs, dur):
    """Slow chord bed under a level, gently moving so it never sits still."""
    n = int(SR * dur)
    t = np.arange(n) / SR
    y = sum(np.sin(2 * np.pi * f * t) + 0.5 * np.sin(2 * np.pi * f * 1.004 * t) for f in freqs) / len(freqs)
    y *= 0.8 + 0.2 * np.sin(2 * np.pi * 0.7 * t)
    fade = max(1, min(n // 3, int(SR * 0.25)))
    e = np.ones(n)
    e[:fade] = np.linspace(0, 1, fade)
    e[-fade:] = np.linspace(1, 0, fade)
    return y * e


def music_layer(plan, B, rng, t_of):
    """Chord bed per level, sparse rising notes on the blade, rewards only on key steps."""
    root = ROOTS[int(rng.integers(len(ROOTS)))]
    note = lambda semis: root * 2 ** (semis / 12)
    last = len(plan["levels"]) - 1
    blades = {}
    for b in plan.get("blade_strokes", []):
        blades.setdefault(b["level"], []).append(b)
    motif_shift = int(rng.integers(len(MOTIFS)))
    for lvl in plan["levels"]:
        li = lvl["level"]
        chord = PROGRESSION[li % len(PROGRESSION)]
        lift = li / max(1, last)      # the whole track swells towards the last level
        octave = 12 if lift >= 0.5 else 0
        t0, t1 = t_of(lvl["start_frame"]), t_of(lvl["collapse_end"])
        place(B["ambient_layer"], t0, pad([note(s) for s in chord[:3]], t1 - t0 + 0.3), 0.0, 0.05 + 0.07 * lift)
        motif = MOTIFS[(li + motif_shift) % len(MOTIFS)]
        passes = blades.get(li, [])
        every = max(1, len(passes) // 16)  # dense levels: a phrase, not a note per cut
        for i, b in enumerate(passes[::every]):
            semis = chord[motif[i % 4]] + 12 + octave * (i // 8 % 2)
            pan = float(np.clip(b["anchor"][0] / 0.12, -0.4, 0.4))
            v = 0.32 + 0.14 * lift + float(rng.uniform(-0.03, 0.03))
            place(B["musical_notes"], t_of(b["time"]["exit"]), pluck(note(semis)), pan, v)
        cs, ce = t_of(lvl["collapse_start"]), t_of(lvl["collapse_end"])
        land = cs + (ce - cs) * 0.68
        key_step = li == last or li % 3 == 2
        if key_step:   # discreet rise into the landing, then a short warm chord
            for k in range(4):
                place(B["musical_notes"], cs + (land - cs) * k / 4, pluck(note(chord[k] + 12), 0.3), 0.0, 0.1 + 0.03 * k)
            for k, s in enumerate(chord[:3]):
                place(B["reward_fx"], land + k * 0.03, pluck(note(s + 12), 0.8, 0.8), (k - 1) * 0.3, 0.3 + 0.15 * (li == last))
        # soft bass exactly on the main landing, level-dependent, never a boom
        place(B["reward_fx"], land, pluck(note(chord[0] - 12), 0.7, 0.2), 0.0, 0.3 + 0.2 * lift)


def loudness_lufs(x):
    """Integrated loudness (ITU-R BS.1770: K-weighting, 400 ms blocks, gating)."""
    b1 = signal.butter(2, 1500, "high", fs=SR, output="sos")
    b2 = signal.butter(2, 38, "high", fs=SR, output="sos")
    y = signal.sosfilt(b2, signal.sosfilt(b1, x, axis=0), axis=0)
    blk = int(0.4 * SR)
    hop = int(0.1 * SR)
    p = np.array([float((y[i:i + blk] ** 2).sum(1).mean()) for i in range(0, max(1, len(y) - blk), hop)])
    p = p[p > 10 ** (-70 / 10)]
    if not len(p):
        return -70.0
    gate = 10 * np.log10(p.mean()) - 10
    g = p[10 * np.log10(p) > gate]
    return float(-0.691 + 10 * np.log10(g.mean())) if len(g) else -70.0


def limit(x, ceiling):
    """Fast look-ahead peak limiter: keeps transients under the ceiling without clipping."""
    env = np.abs(x).max(1)
    w = int(SR * 0.004)
    env = ndimage.maximum_filter1d(env, 2 * w + 1)
    g = np.minimum(1.0, ceiling / np.maximum(env, 1e-9))
    g = ndimage.minimum_filter1d(g, 2 * w + 1)
    g = ndimage.uniform_filter1d(g, 2 * w + 1)
    return x * g[:, None]


def grid_events(plan, B, P, rng, samples, events, t_of):
    """Grid format: a chop per stroke frame, a crumble cascade when the object collapses."""
    fps = plan["fps"]
    # one sound per visible blade pass, placed on the frames where the edge enters and
    # leaves the object: the audio cannot drift from the animation
    for bs in plan.get("blade_strokes", []):
        tm = bs["time"]
        c, x = tm["contact"], tm["exit"]
        nh = bs["group_cuts"]
        place(B["cutting_asmr"], max(0.0, t_of(c) - 0.16), whoosh(rng), 0.0, rng.uniform(0.5, 0.8))
        events.append({"t": max(0.0, t_of(c) - 0.16), "type": "approach", "frame": c})
        pos = bs["anchor"][0]
        pan = float(np.clip(pos / 0.12, -0.6, 0.6))
        dur = max(0.03, (x - c) / fps)
        gain = rng.uniform(0.85, 1.05) * (1.0 if nh == 1 else min(1.3, 0.85 + 0.12 * np.log2(nh)))
        place(B["cutting_asmr"], t_of(c), chop(rng, P, samples, dur, nh), pan, gain)
        events.append({"t": t_of(c), "type": "contact", "frame": c, "strokes": nh})
        events.append({"t": t_of(x), "type": "end_of_cut", "frame": x, "strokes": nh})
    for lvl in plan["levels"]:
        t0, t1 = t_of(lvl["collapse_start"]), t_of(lvl["collapse_end"])
        n = int(SR * (t1 - t0 + 0.4))
        dens = 60 + 90 * np.log10(max(10, lvl["cubes"]))      # grains per second
        y = np.zeros(n)
        for pos in rng.integers(0, max(1, int(SR * (t1 - t0))), size=int(dens * (t1 - t0))):
            k = int(rng.integers(60, 400))
            seg = rng.standard_normal(k) * np.exp(-np.arange(k) / rng.uniform(15, 60))
            y[pos:pos + k] += seg[: n - pos] * rng.uniform(0.2, 1.0)
        y = signal.sosfilt(signal.butter(2, [600, 7000], "band", fs=SR, output="sos"), y)
        y *= np.concatenate([np.linspace(0.3, 1, n // 3), np.linspace(1, 0, n - n // 3) ** 1.5])
        place(B["piece_impacts"], t0, y, 0.0, 0.5)
        thud = tone(rng, P["land"]["f0"], P["land"]["decay"] * 1.5)
        place(B["piece_impacts"], t0 + (t1 - t0) * 0.68, thud, 0.0, P["land"]["gain"] * 1.4)
        events.append({"t": t0, "type": "separation", "level": lvl["level"]})
        events.append({"t": t1, "type": "surface_contact", "level": lvl["level"]})


def render(plan, out_dir):
    cfg = plan["config"]
    prof_name = cfg["audio_profile"]
    P = PROFILES[prof_name]
    fps = plan["fps"]
    rng = np.random.default_rng(cfg["random_seed"] + 7)
    samples = load_samples(prof_name)
    total_s = plan["total_frames"] / fps
    buf_shape = (int(total_s * SR) + SR, 2)
    B = {name: np.zeros(buf_shape) for name in BUSES}
    buf = B["cutting_asmr"]
    preset = cfg.get("audio_preset") or "asmr_pure"
    gains = dict(AUDIO_PRESETS[preset], **(cfg.get("audio_bus_gains") or {}))
    events = []
    t_of = lambda f: (f - 1) / fps  # frame 1 is t = 0

    # scene width for stereo placement
    if plan.get("kind") == "grid":
        grid_events(plan, B, P, rng, samples, events, t_of)
        if gains["musical_notes"] > 0 or gains["reward_fx"] > 0:
            music_layer(plan, B, rng, t_of)
    xs = [h["plane_point"][0] + h["offset"][0] for h in plan.get("hits", [])] or [0]
    span = max(0.05, max(abs(min(xs)), abs(max(xs))))

    # Dense strokes cut several pieces in the same frame: one richer sound per frame
    # instead of hundreds of stacked transients (which would turn into white noise).
    groups = {}
    for h in plan.get("hits", []):
        groups.setdefault(h["contact_frame"], []).append(h)
    root_vol = max(1e-9, plan["pieces"][0]["volume"]) if plan.get("pieces") else 1.0
    for cf in sorted(groups):
        g = groups[cf]
        h = g[0]
        nh = len(g)
        t0, t1 = t_of(h["contact_frame"]), t_of(h["exit_frame"])
        dur = max(0.03, t1 - t0)
        px = float(np.mean([x["plane_point"][0] + x["offset"][0] for x in g]))
        pan = float(np.clip(px / span * 0.6, -0.7, 0.7))
        size = (sum(x["piece_volume"] for x in g) / root_vol) ** (1 / 3)
        loud = min(1.0, 0.55 + 0.45 * size)   # small pieces are quieter
        loud *= 1.0 if nh == 1 else min(1.25, 0.8 + 0.12 * np.log2(nh))
        var = rng.uniform(0.85, 1.15)

        # 1. blade contact
        c = P["contact"]
        y = pick(samples, "contact", rng)
        if y is None:
            n = int(SR * c["dur"] * rng.uniform(0.8, 1.3))
            y = bandnoise(rng, n, *c["band"]) * env_ad(n, 0.05, 3)
        place(buf, t0, y, pan, c["gain"] * loud * var)
        events.append({"t": t0, "type": "contact", "hit": h["piece"]})

        # 2+3. penetration then traversal (one continuous body, swelling in)
        b = P["body"]
        y = pick(samples, "body", rng)
        if y is None:
            n = int(SR * dur)
            body = bandnoise(rng, n, *b["band"]) * b["gain"]
            sq = bandnoise(rng, n, *b["squish"]) * b["squish_gain"]
            sq *= 0.6 + 0.4 * np.sin(np.linspace(0, np.pi * rng.uniform(2, 5), n)) ** 2
            cr = np.zeros(n)
            k = rng.poisson(b["crackle"] * dur * (1 + 0.5 * np.log2(nh)))
            for pos in rng.integers(0, max(1, n - 200), size=k):
                m = int(rng.integers(40, 160))
                cr[pos:pos + m] += rng.standard_normal(m) * np.exp(-np.arange(m) / 25)
            cr = signal.sosfilt(signal.butter(2, [1500, 6500], "band", fs=SR, output="sos"), cr)
            y = (body + sq + cr * b["crackle_gain"])
            e = np.minimum(1, np.linspace(0, 1, n) * 4) * (0.7 + 0.3 * np.linspace(1, 0.6, n))
            y = y * e
        place(buf, t0 + 0.004, y, pan, loud * var)
        events.append({"t": t0 + 0.004, "type": "penetration", "hit": h["piece"]})
        events.append({"t": t0 + dur * 0.3, "type": "traversal", "hit": h["piece"]})

        # 4. end of cut: blade meets the board
        e_ = P["end"]
        y = pick(samples, "end", rng)
        if y is None:
            y = tone(rng, e_["f0"], e_["decay"])
            m = int(SR * 0.004)
            y[:m] += rng.standard_normal(m) * e_["click"] * env_ad(m, 0.05, 2)
        place(buf, t1, y, pan, e_["gain"] * (0.7 + 0.3 * loud) * var)
        events.append({"t": t1, "type": "end_of_cut", "hit": h["piece"]})

        # 5. separation of the two (or more) parts
        s = P["separate"]
        y = pick(samples, "separate", rng)
        if y is None:
            n = int(SR * s["dur"] * rng.uniform(0.8, 1.2))
            y = bandnoise(rng, n, *s["band"]) * env_ad(n, 0.2, 2)
        place(buf, t1 + 0.02, y, pan, s["gain"] * loud)
        events.append({"t": t1 + 0.02, "type": "separation", "hit": h["piece"]})

        # 6. parts settle on the board
        l = P["land"]
        y = pick(samples, "land", rng)
        if y is None:
            y = tone(rng, l["f0"], l["decay"]) * 0.8
            y = signal.sosfilt(signal.butter(2, 900, "low", fs=SR, output="sos"), y)
        land_t = t1 + 0.3 * rng.uniform(0.7, 1.1)
        place(B["piece_impacts"], land_t, y, pan, l["gain"] * loud)
        events.append({"t": land_t, "type": "surface_contact", "hit": h["piece"]})

    # small, close "kitchen" room: short decaying-noise impulse response
    ir_n = int(SR * 0.09)
    ir = rng.standard_normal((ir_n, 2)) * np.exp(-np.arange(ir_n) / (SR * 0.018))[:, None]
    ir /= np.abs(ir).sum(0)
    def room(x):
        wet = np.stack([signal.fftconvolve(x[:, i], ir[:, i])[: len(x)] for i in range(2)], 1)
        return x + wet * 2.2
    # very quiet room tone so silence does not sound "dead"
    B["ambient_layer"] += signal.sosfilt(signal.butter(2, 400, "low", fs=SR, output="sos"),
                                         rng.standard_normal(buf_shape), axis=0) * 0.004
    for name in ("cutting_asmr", "piece_impacts"):
        B[name] = room(B[name])
    for name in ("musical_notes", "reward_fx"):
        B[name] = B[name] + room(B[name]) * 0.25
    n_out = int(total_s * SR)
    hp = signal.butter(2, 35, "high", fs=SR, output="sos")
    bus_stats = {}
    mix = np.zeros((n_out, 2))
    for name in BUSES:
        y = signal.sosfilt(hp, B[name][:n_out], axis=0) * gains[name]
        bus_stats[name] = {"gain": gains[name], "peak": round(float(np.abs(y).max()), 4)}
        mix += y
    ceiling = 10 ** (-1.0 / 20)
    mix *= 10 ** ((LOUDNESS_TARGET - loudness_lufs(mix)) / 20)   # loudness first, then transients
    mix = limit(mix, ceiling)
    pcm = np.clip(mix, -1, 1)
    clipped = int(np.count_nonzero(np.abs(pcm) >= 0.9999))

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "soundtrack.wav")
    with wave.open(path, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((pcm * 32767).astype("<i2").tobytes())
    rms = float(np.sqrt((pcm ** 2).mean()))
    info = {"path": path, "duration_s": len(pcm) / SR, "sample_rate": SR, "channels": 2,
            "peak_dbfs": round(20.0 * np.log10(np.abs(pcm).max() + 1e-12), 2),
            "rms_dbfs": round(20 * np.log10(rms + 1e-12), 2),
            "clipped_samples": clipped, "profile": prof_name,
            "used_samples": sorted(samples), "music": preset != "asmr_pure", "preset": preset,
            "loudness_lufs": round(loudness_lufs(pcm), 1), "buses": bus_stats, "events": len(events)}
    with open(os.path.join(out_dir, "audio_events.json"), "w", encoding="utf-8") as f:
        json.dump({"info": info, "events": events}, f)
    return info


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(2)
    with open(sys.argv[1], encoding="utf-8") as f:
        plan = json.load(f)
    print(json.dumps(render(plan, sys.argv[2])))


if __name__ == "__main__":
    main()

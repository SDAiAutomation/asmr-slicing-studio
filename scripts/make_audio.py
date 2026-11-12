"""Procedural ASMR soundtrack built from the same plan.json as the animation.

Grid format: a whoosh + chop per visible blade pass (placed on the frames where the
edge enters and leaves the object), a crumble cascade and a thud when each level
collapses into cubes. Material-specific, seeded variation, short room reverb,
peak-normalised below clipping.

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
from scipy import signal

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


def grid_events(plan, buf, P, rng, samples, events, t_of):
    """A chop per blade pass, a crumble cascade when the object collapses."""
    fps = plan["fps"]
    # one sound per visible blade pass, placed on the frames where the edge enters and
    # leaves the object: the audio cannot drift from the animation
    for bs in plan.get("blade_strokes", []):
        tm = bs["time"]
        c, x = tm["contact"], tm["exit"]
        nh = bs["group_cuts"]
        gap = tm.get("gap", 99.0)            # frames to the neighbouring stroke (chop rhythm)
        if gap >= 3:                         # a whoosh only while strokes are still separate
            place(buf, max(0.0, t_of(c) - 0.16), whoosh(rng), 0.0, rng.uniform(0.5, 0.8))
            events.append({"t": max(0.0, t_of(c) - 0.16), "type": "approach", "frame": c})
        pos = bs["anchor"][0]
        pan = float(np.clip(pos / 0.12, -0.6, 0.6))
        dur = max(0.03, (x - c) / fps)
        gain = rng.uniform(0.85, 1.05) * (1.0 if nh == 1 else min(1.3, 0.85 + 0.12 * np.log2(nh)))
        gain *= float(np.clip(np.sqrt(gap / 6.0), 0.18, 1.0))   # dense chopping blends into a roll
        place(buf, t_of(c), chop(rng, P, samples, dur, nh), pan, gain)
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
        place(buf, t0, y, 0.0, 0.5)
        thud = tone(rng, P["land"]["f0"], P["land"]["decay"] * 1.5)
        place(buf, t1 - 0.05, thud, 0.0, P["land"]["gain"] * 1.4)
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
    buf = np.zeros((int(total_s * SR) + SR, 2))
    events = []
    t_of = lambda f: (f - 1) / fps  # frame 1 is t = 0

    grid_events(plan, buf, P, rng, samples, events, t_of)

    # small, close "kitchen" room: short decaying-noise impulse response
    ir_n = int(SR * 0.09)
    ir = rng.standard_normal((ir_n, 2)) * np.exp(-np.arange(ir_n) / (SR * 0.018))[:, None]
    ir /= np.abs(ir).sum(0)
    wet = np.stack([signal.fftconvolve(buf[:, i], ir[:, i])[: len(buf)] for i in range(2)], 1)
    mix = buf + wet * 2.2
    # very quiet room tone so silence does not sound "dead"
    mix += signal.sosfilt(signal.butter(2, 400, "low", fs=SR, output="sos"),
                          rng.standard_normal((len(mix), 2)), axis=0) * 0.0015
    mix = mix[: int(total_s * SR)]
    mix = signal.sosfilt(signal.butter(2, 35, "high", fs=SR, output="sos"), mix, axis=0)

    peak = float(np.abs(mix).max()) or 1.0
    target = 10 ** (-1.0 / 20)  # -1 dBFS
    mix *= target / peak
    pcm = np.clip(mix, -1, 1)
    clipped = int(np.count_nonzero(np.abs(pcm) >= 0.999))

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "soundtrack.wav")
    with wave.open(path, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((pcm * 32767).astype("<i2").tobytes())
    rms = float(np.sqrt((pcm ** 2).mean()))
    info = {"path": path, "duration_s": len(pcm) / SR, "sample_rate": SR, "channels": 2,
            "peak_dbfs": round(20 * np.log10(np.abs(pcm).max() + 1e-12), 2),
            "rms_dbfs": round(20 * np.log10(rms + 1e-12), 2),
            "clipped_samples": clipped, "profile": prof_name,
            "used_samples": sorted(samples), "music": False, "events": len(events)}
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

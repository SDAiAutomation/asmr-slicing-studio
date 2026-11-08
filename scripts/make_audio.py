"""Procedural ASMR soundtrack built from the same plan.json as the animation.

Six event layers per cut (contact, penetration, traversal, end of cut,
separation, surface contact), material-specific, seeded variation, short room
reverb, no music by default, peak-normalised below clipping.

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

    # scene width for stereo placement
    xs = [h["plane_point"][0] + h["offset"][0] for h in plan["hits"]] or [0]
    span = max(0.05, max(abs(min(xs)), abs(max(xs))))

    # Dense strokes cut several pieces in the same frame: one richer sound per frame
    # instead of hundreds of stacked transients (which would turn into white noise).
    groups = {}
    for h in plan["hits"]:
        groups.setdefault(h["contact_frame"], []).append(h)
    root_vol = max(1e-9, plan["pieces"][0]["volume"])
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
        place(buf, land_t, y, pan, l["gain"] * loud)
        events.append({"t": land_t, "type": "surface_contact", "hit": h["piece"]})

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

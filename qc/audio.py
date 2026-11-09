"""Procedural sound for a Contrôle Qualité episode, driven by the engine timeline.

    python qc/audio.py timeline.json out.wav

Events (from qc/engine.html buildTimeline): round, approach, ramp, ok, question,
break (material/kind), stamp (green/red), card, explain, cta.
"""
import json
import sys

import numpy as np
from scipy.signal import lfilter

SR = 44100


def lp(x, a):
    return lfilter([a], [1, -(1 - a)], x)


def env(n, att, dec):
    t = np.arange(n) / SR
    return np.minimum(1, t / max(att, 1e-4)) * np.exp(-t / dec)


def sm(a, b, x):
    k = np.clip((x - a) / (b - a), 0, 1)
    return k * k * (3 - 2 * k)


def main(tl_path, out_path):
    tl = json.load(open(tl_path, encoding="utf-8"))
    dur = float(tl["duration"])
    n_all = int(dur * SR)
    out = np.zeros(n_all)
    tt = np.arange(n_all) / SR
    rng = np.random.default_rng(11)

    def add(sig, t0, gain=1.0):
        i = int(t0 * SR)
        j = min(n_all, i + len(sig))
        if 0 <= i < n_all:
            out[i:j] += gain * sig[: j - i]

    def noise(sec):
        return rng.standard_normal(int(sec * SR))

    out += lp(rng.standard_normal(n_all), 0.02) * 0.02          # room tone
    load = np.zeros(n_all)                                        # normalised load 0..1 over time
    motor_on = np.zeros(n_all)
    for e in tl["events"]:
        if e["type"] == "approach":
            m = (tt >= e["t0"]) & (tt < e["t1"])
            motor_on[m] = 1
        if e["type"] == "ramp":
            m = (tt >= e["t0"]) & (tt < e["t1"])
            motor_on[m] = 1
            prev = 0.0 if e["from"] == 0 else e["frac"] * e["from"] / max(e["to"], 1e-9)
            load[m] = prev + (e["frac"] - prev) * sm(e["t0"], e["t1"], tt[m])
    # hold the load between ramps until the break
    breaks = [e["t"] for e in tl["events"] if e["type"] == "break"]
    rounds = [e["t"] for e in tl["events"] if e["type"] == "round"]
    for i, r0 in enumerate(rounds):
        b = breaks[i] if i < len(breaks) else dur
        seg = (tt >= r0) & (tt < b)
        idx = np.where(seg)[0]
        cur = 0.0
        for k in idx:
            if load[k] > 0:
                cur = load[k]
            else:
                load[k] = cur
    act = lp(motor_on, 0.0015)
    f = 50 + 45 * load
    ph = 2 * np.pi * np.cumsum(f) / SR
    motor = np.sin(ph) + 0.5 * np.sin(2 * ph) + 0.25 * np.sin(3 * ph + 0.3)
    hiss = lp(rng.standard_normal(n_all), 0.25) * 0.35
    out += act * (motor * 0.15 + hiss * 0.1)
    held = lp((load > 0).astype(float), 0.001)
    out += held * np.sin(2 * np.pi * 46 * tt) * 0.05 * (0.4 + load)

    for e in tl["events"]:
        ty = e["type"]
        if ty == "round":
            s = lp(noise(0.5), 0.12) * np.sin(np.pi * np.arange(int(0.5 * SR)) / (0.5 * SR))
            add(s, e["t"], 0.3)
        elif ty == "approach":
            n = int(0.4 * SR)
            t = np.arange(n) / SR
            add(np.sin(2 * np.pi * (110 - 40 * t) * t) * env(n, 0.002, 0.08) + rng.standard_normal(n) * env(n, 0.0005, 0.01) * 0.5, e["t1"], 0.5)
        elif ty == "ok":
            n = int(0.6 * SR)
            t = np.arange(n) / SR
            add((np.sin(2 * np.pi * 1318.5 * t) + 0.6 * np.sin(2 * np.pi * 1975.5 * t)) * env(n, 0.003, 0.16), e["t"], 0.17)
        elif ty == "question":
            k = 0
            t0 = e["t0"] + 0.05
            while t0 < e["t1"] - 0.1:
                n = int(0.05 * SR)
                t = np.arange(n) / SR
                add(np.sin(2 * np.pi * (2400 if k % 2 == 0 else 1800) * t) * env(n, 0.0005, 0.008), t0, 0.33)
                t0 += 0.25
                k += 1
            q = (tt > e["t0"]) & (tt < e["t1"])
            out[q] += np.sin(2 * np.pi * (60 + 30 * sm(e["t0"], e["t1"], tt[q])) * tt[q]) * 0.08 * sm(e["t0"], e["t1"], tt[q])
        elif ty == "go":                                           # start gate
            m = int(0.25 * SR)
            tm = np.arange(m) / SR
            add(np.sin(2 * np.pi * 1760 * tm) * env(m, 0.002, 0.08), e["t"], 0.3)
            add(lp(rng.standard_normal(int(0.05 * SR)), 0.5) * env(int(0.05 * SR), 0.0005, 0.01), e["t"], 0.5)
        elif ty == "race":                                         # balls rolling: rumble that grows with speed
            t0, t1, sp = e["t0"], e["t1"], e.get("speed", 1)
            m = (tt >= t0) & (tt < t1 + 0.05)
            k = np.clip((tt[m] - t0) / max(t1 - t0, 1e-3), 0, 1)
            a = 0.02 if sp == 1 else 0.008                         # slow motion: deeper rumble
            out[m] += lp(rng.standard_normal(int(m.sum())), a) * (0.25 + 0.75 * k) * 2.2
            for j, tf in enumerate(sorted(e["finishes"])):
                mm = int(0.5 * SR)
                tm = np.arange(mm) / SR
                f0 = (1318.5, 1046.5, 880.0)[min(j, 2)] * (1 if sp == 1 else 0.5)
                add(np.sin(2 * np.pi * f0 * tm) * env(mm, 0.002, 0.15) + 0.4 * np.sin(4 * np.pi * f0 * tm) * env(mm, 0.002, 0.06), tf, 0.22)
        elif ty == "drop":                                         # trap doors snap open
            m = int(0.06 * SR)
            add(lp(rng.standard_normal(m), 0.6) * env(m, 0.0005, 0.01), e["t"], 0.5)
        elif ty == "break" and e.get("kind") == "drop":            # small impact: tick + one shell crack
            n = int(0.4 * SR)
            t = np.arange(n) / SR
            add(np.sin(2 * np.pi * (180 - 60 * t) * t) * env(n, 0.001, 0.05), e["t"], 0.6)
            add(lp(rng.standard_normal(n), 0.7) * env(n, 0.0003, 0.018), e["t"], 0.8)
        elif ty == "break":
            t_b = e["t"]
            for k in range(14):                                       # stress creaks
                t0 = t_b - 0.95 + (k / 14) ** 0.7 * 0.93
                m = int(0.03 * SR)
                add(lp(rng.standard_normal(m), 0.5) * env(m, 0.0005, 0.006), t0, 0.12 + 0.3 * k / 14)
            concrete = e.get("material") == "concrete"
            n = int(0.9 * SR)
            t = np.arange(n) / SR
            if concrete:
                boom = np.sin(2 * np.pi * (70 - 35 * t) * t) * env(n, 0.002, 0.18)
                crunch = rng.standard_normal(n) * env(n, 0.0004, 0.07)
                for _ in range(40):
                    i = int(rng.uniform(0, 0.35) * SR)
                    m = int(0.01 * SR)
                    if i + m < n:
                        crunch[i:i + m] += rng.standard_normal(m) * np.exp(-np.arange(m) / (0.0025 * SR)) * rng.uniform(0.4, 1.0)
                snap = 1.3 if e.get("kind") == "traction" else 1.0
                add(boom, t_b, 1.0)
                add(lp(crunch, 0.35), t_b, 0.9 * snap)
                for _ in range(14):                                  # debris landing
                    m = int(0.025 * SR)
                    add(lp(rng.standard_normal(m), 0.4) * np.exp(-np.arange(m) / (0.004 * SR)), t_b + 0.25 + rng.uniform(0, 0.8), 0.35)
            else:
                crack = rng.standard_normal(n) * env(n, 0.0003, 0.03)
                add(crack, t_b, 0.9)
                add(np.sin(2 * np.pi * (90 - 50 * t) * t) * env(n, 0.002, 0.12), t_b, 0.8)
                m = int(0.45 * SR)
                tm = np.arange(m) / SR
                add(lp(rng.standard_normal(m), 0.08) * env(m, 0.01, 0.12) * (1 + 0.6 * np.sin(2 * np.pi * 23 * tm)), t_b + 0.05, 1.2)
        elif ty == "replay":
            # rewind whoosh, then the break again, pitched down and stretched (slow motion)
            m = int(0.5 * SR)
            tm = np.arange(m) / SR
            wn = rng.standard_normal(m)
            k = tm / 0.5                                   # filter opens up as the whoosh rises
            add(((1 - k) * lp(wn, 0.05) + k * lp(wn, 0.3)) * k ** 2, e["t0"] - 0.1, 0.35)
            if "tbreak" not in e:                          # physics replays: the race sound is its own event
                continue
            n = int(2.0 * SR)
            t = np.arange(n) / SR
            boom = np.sin(2 * np.pi * (40 - 15 * t) * t) * env(n, 0.01, 0.55)
            rumble = lp(rng.standard_normal(n), 0.03) * env(n, 0.02, 0.6)
            add(boom, e["tbreak"], 0.9)
            add(rumble, e["tbreak"], 1.4)
            for _ in range(10):
                mm = int(0.08 * SR)
                add(lp(rng.standard_normal(mm), 0.08) * np.exp(-np.arange(mm) / (0.02 * SR)), e["tbreak"] + rng.uniform(0.1, 0.9), 0.5)
        elif ty == "stamp":
            t0 = e["t"] + 0.17
            n = int(0.6 * SR)
            t = np.arange(n) / SR
            add(np.sin(2 * np.pi * (70 - 20 * t) * t) * env(n, 0.001, 0.1) + lp(rng.standard_normal(n), 0.3) * env(n, 0.0005, 0.025) * 0.8, t0, 1.0)
            if e.get("color") == "green":
                for k, fr in enumerate((784, 1046.5)):
                    m = int(0.35 * SR)
                    tm = np.arange(m) / SR
                    add(np.sin(2 * np.pi * fr * tm) * env(m, 0.003, 0.12), t0 + 0.1 + 0.11 * k, 0.2)
            else:
                m = int(0.5 * SR)
                tm = np.arange(m) / SR
                add(lp(np.sign(np.sin(2 * np.pi * 180 * tm)) * env(m, 0.005, 0.25), 0.2), t0 + 0.08, 0.2)
        elif ty == "card":
            m = int(0.35 * SR)
            add(lp(rng.standard_normal(m), 0.1) * np.sin(np.pi * np.arange(m) / m), e["t"], 0.22)
        elif ty == "explain":
            m = int(0.5 * SR)
            add(lp(rng.standard_normal(m), 0.12) * np.sin(np.pi * np.arange(m) / m), e["t0"], 0.3)
            for k in range(3):                                       # soft ticks as lines appear
                mm = int(0.12 * SR)
                tm = np.arange(mm) / SR
                add(np.sin(2 * np.pi * 880 * tm) * env(mm, 0.002, 0.04), e["t0"] + 0.8 + 2.0 * k, 0.15)
        elif ty == "cta":
            m = int(0.3 * SR)
            tm = np.arange(m) / SR
            add(np.sin(2 * np.pi * (500 + 900 * tm) * tm) * env(m, 0.002, 0.07), e["t"], 0.3)

    out *= np.minimum(1, tt / 0.05) * np.minimum(1, (dur - tt) / 0.4)
    out = np.tanh(out * 1.4)
    out /= np.max(np.abs(out)) * 1.12
    pcm = (out * 32767).astype(np.int16)
    import wave
    with wave.open(out_path, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(np.stack([pcm, pcm], 1).tobytes())
    print(f"audio {dur:.2f} s -> {out_path}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])

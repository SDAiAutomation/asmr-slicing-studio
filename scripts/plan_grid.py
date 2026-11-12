"""Grid / levels planner ("N CUTS -> X CUBES" format).

Each level: a fresh object, C straight cuts split over the X, Y and Z axes
(sweeping across the object, axis by axis), then the scored object collapses
into its cubes. Piece counts are computed from the real cell occupancy of the
shape after every cut, so both counters are exact:
  CUTS   = blade strokes done in this level
  PIECES = non-empty boxes of the current partial grid (= cubes at the end)

Usage: python plan_grid.py config.json plan.json
"""
import json
import math
import sys
import time

import numpy as np

import knife_traj as KT
from common import OBJECTS, grid_occupancy, load_config


def split_cuts(ext, C):
    """Distribute C cuts over 3 axes so that the cells are as cubic as possible."""
    s = ext.sum() / (C + 3)
    for _ in range(60):
        cuts = np.maximum(0, np.round(ext / s) - 1).astype(int)
        diff = C - cuts.sum()
        if diff == 0:
            break
        s *= (cuts.sum() + 3) / (C + 3) if cuts.sum() else 0.9
    cuts = np.maximum(0, np.round(ext / s) - 1).astype(int)
    while cuts.sum() != C:  # exact fix-up on the axis with the largest/smallest cells
        cell = ext / (cuts + 1)
        if cuts.sum() < C:
            cuts[np.argmax(cell)] += 1
        else:
            i = int(np.argmin(np.where(cuts > 0, cell, np.inf)))
            cuts[i] -= 1
    return cuts


FIRST_CUT_FRAMES = 6.0   # every level opens on one stroke every 6 frames (calm, readable)
CALM_CUTS = 4            # strokes at the calm rhythm before the acceleration
AXIS_PAUSE_S = 0.35      # the knife turns to the next cutting direction
MIN_FWD_FRAMES = 0.12    # fastest stroke through the object (a blur streak)
MAX_FWD_FRAMES = 4.0     # slowest stroke: a deliberate, clean chop


def level_seconds(C):
    """Cutting time of a level: grows slowly with the cut count (the rate does the rest)."""
    return 1.3 + 0.8 * math.log2(C + 1)


def cut_schedule(n, frames, first, ramp=0.6):
    """Frame offsets at which cut 1..n completes. Each level opens on one cut every
    `first` frames (calm, clearly separate strokes); the rate then grows exponentially
    for the first `ramp` of the level and holds its top speed until the end, so the
    counter accelerates and then races, like a chef speeding up."""
    if n < 2:
        return [float(frames)]
    r0 = 1.0 / first
    t = np.linspace(0.0, frames, 4001)

    def done(a):
        t1 = ramp * frames
        up = r0 * np.expm1(a * np.minimum(t, t1)) / a
        return up + r0 * math.exp(a * t1) * np.maximum(0.0, t - t1)

    if r0 * frames >= n:          # small level: even pacing already fits
        return (np.arange(1, n + 1) * frames / n).tolist()
    lo_a, hi_a = 1e-6, 40.0 / (ramp * frames)
    for _ in range(60):
        a = (lo_a + hi_a) / 2
        if done(a)[-1] < n:
            lo_a = a
        else:
            hi_a = a
    c = done(hi_a)
    return np.interp(np.arange(1, n + 1), c, t).tolist()


_OCC = {}


def occupancy(shape, lo, ext, n):
    key = (id(shape), tuple(int(v) for v in n))
    if key not in _OCC:  # duration fitting re-plans several times: compute each grid once
        _OCC[key] = grid_occupancy(shape, lo, ext, n)
    return _OCC[key]


def boxes_count(occ, active):
    """Non-empty boxes when only the first active[a] cuts of each axis are made."""
    r = occ
    for a in range(3):
        n = r.shape[a]
        k = int(active[a])
        if k >= n - 1:
            continue
        starts = np.arange(k + 1)
        r = np.logical_or.reduceat(r, starts, axis=a)
    return int(np.count_nonzero(r))


def _plan(cfg, ts=1.0):
    t0 = time.time()
    shape = OBJECTS[cfg["object_type"]]
    fps = cfg["fps"]
    lo, hi = shape.bbox()
    # tight extents without bbox padding
    ext = hi - lo
    rng = np.random.default_rng(cfg["random_seed"])

    levels, strokes, cuts_ev, pieces_ev = [], [], [], []
    blade_strokes = []
    height = float(ext[2])
    centre = np.array([0.0, 0.0, height / 2])
    radius = float(np.linalg.norm(ext)) / 2 * 1.15
    # the blade grows with the object so its edge always covers the whole section
    width_needed = float(np.hypot(ext[0], ext[1]))
    knife_scale = float(min(1.3, max(0.85, width_needed * 1.05 / (KT.EDGE_X[-1] - KT.EDGE_X[0]))))
    resistance = KT.RESISTANCE.get(cfg["object_type"], 0.5)
    safe_r = radius * 1.25 + 0.03
    knife_keys = []
    prev_blade = first_blade = None
    frame = 1
    intro = int(0.4 * fps)   # first cut within half a second: the hook is the action itself
    for li, C in enumerate(cfg["cuts_sequence"]):
        cuts = split_cuts(ext, C)
        n = cuts + 1
        if int(np.prod(n)) > 4 * cfg["max_cubes"]:
            break
        occ, centres = occupancy(shape, lo, ext, n)
        cubes = int(occ.sum())
        if cubes > cfg["max_cubes"]:
            break
        start = frame
        enter = max(6, int(0.45 * fps * ts)) if li else 0
        f = frame + enter + (intro if li == 0 else int(0.25 * fps))
        # stroke order: X cuts left->right, then Y cuts front->back, then Z (horizontal)
        order = [(0, k) for k in range(1, cuts[0] + 1)] + \
                [(1, k) for k in range(1, cuts[1] + 1)] + \
                [(2, k) for k in range(cuts[2], 0, -1)]
        n_cuts = max(1, len(order))
        pose = KT.Pose(0.0, centre, height / 2)

        def stroke_for(j):
            ax, k = order[j]
            p_obj = (lo + ext / 2).copy()
            p_obj[ax] = lo[ax] + ext[ax] * k / n[ax]
            normal = np.zeros(3)
            normal[ax] = 1.0
            st = KT.plan_stroke(shape, pose, p_obj, normal, radius, knife_scale, 0.0)
            inward = 1.0 if k < n[ax] / 2 else -1.0
            for nudge in (0.005, 0.01, 0.02, 0.04, 0.07, 0.1):   # plane grazing the outline:
                if st is not None:                    # move the blade inwards (fraction of the
                    break                             # object), a real stroke still lands
                p_obj[ax] = lo[ax] + ext[ax] * (k / n[ax] + inward * nudge)
                st = KT.plan_stroke(shape, pose, p_obj, normal, radius, knife_scale, 0.0)
            return st

        # ONE REAL KNIFE STROKE PER CUT. The blade marches plane by plane through the object;
        # the first strokes come every FIRST_CUT_FRAMES (calm, readable), then the rhythm
        # accelerates until several strokes fall inside one frame: the knife then renders as a
        # motion-blurred streak (real Cycles motion blur on sub-frame keys), every stroke at
        # its own exact time, and the counter climbs by exactly one per stroke.
        # the first strokes keep the calm guide rhythm, then the ramp starts
        calm = min(n_cuts, CALM_CUTS)
        targets = [FIRST_CUT_FRAMES * (i + 1) for i in range(calm)]
        if n_cuts > calm:
            # never slower than the calm rhythm: the stroke rate always climbs after the opening
            rest = min(0.75 * FIRST_CUT_FRAMES * (n_cuts - calm),
                       max(2 * MIN_FWD_FRAMES * (n_cuts - calm), level_seconds(C) * ts * fps - targets[-1]))
            targets += [targets[-1] + x for x in cut_schedule(n_cuts - calm, rest, FIRST_CUT_FRAMES)]
        t_contact, shift = [], 0.0
        for j in range(n_cuts):
            if j and order[j][0] != order[j - 1][0]:
                shift += AXIS_PAUSE_S * fps * max(ts, 0.6)   # the knife turns to the next direction
            t_contact.append(f + targets[j] + shift)
        planned = [stroke_for(j) for j in range(n_cuts)]
        lvl_blades = []
        for j, st in enumerate(planned):
            if st is None:
                continue
            gap_prev = t_contact[j] - t_contact[j - 1] if j else FIRST_CUT_FRAMES
            gap_next = t_contact[j + 1] - t_contact[j] if j + 1 < n_cuts else gap_prev
            gap = min(gap_prev, gap_next)
            fwd = min(MAX_FWD_FRAMES, max(MIN_FWD_FRAMES, 0.5 * gap))
            travel = st["s_end"] - st["s_ready"]
            v = travel / fwd
            a = t_contact[j] - (st["s_in"] - st["s_ready"]) / v
            st.update(level=li, axis=int(order[j][0]), group_cuts=1, resistance=resistance,
                      time={"ready": a, "contact": t_contact[j],
                            "exit": a + (st["s_out"] - st["s_ready"]) / v,
                            "sep_end": a + fwd, "gap": gap})
            lvl_blades.append(st)
        knife_keys.extend(KT.chop_keys(lvl_blades, centre, safe_r, prev=prev_blade))
        if lvl_blades:
            first_blade = first_blade or lvl_blades[0]
            prev_blade = lvl_blades[-1]
        cut_frames = [int(math.ceil(t)) for t in t_contact]
        cut_group = list(range(n_cuts))
        dur = int(math.ceil(t_contact[-1])) + 2 - f
        blade_strokes.extend(lvl_blades)
        active = [0, 0, 0]
        z_done = 0
        last_eval_frame = None
        lvl_strokes = []
        for j, (ax, k) in enumerate(order):
            sf = cut_frames[j]
            pos = lo[ax] + ext[ax] * k / n[ax]
            if ax == 2:
                z_done += 1
                active[2] = cuts[2]  # horizontal cuts come from the top: count them as they land
            else:
                active[ax] = k
            st = {"level": li, "axis": int(ax), "index": int(k), "pos": float(pos),
                  "frame": int(sf), "n_in_level": j + 1, "group": cut_group[j]}
            lvl_strokes.append(st)
            cuts_ev.append((int(sf), j + 1, li))
            if sf != last_eval_frame or j == len(order) - 1:
                if ax == 2:
                    # first z_done horizontal cuts from the top = last indices
                    zc = occ
                    z_starts = sorted(set([0] + [cuts[2] + 1 - m for m in range(1, z_done + 1)]))
                    zc = np.logical_or.reduceat(zc, z_starts, axis=2)
                    cnt = boxes_count(zc, [active[0], active[1], zc.shape[2]])
                else:
                    cnt = boxes_count(occ, active)
                pieces_ev.append((int(sf), cnt, li))
                last_eval_frame = sf
        f_end = f + dur + 2
        collapse = max(int(0.45 * fps), int(0.75 * fps * ts))
        hold = max(int(0.35 * fps), int(0.7 * fps * ts))
        exit_f = max(int(0.3 * fps), int(0.4 * fps * ts))
        lvl = {"level": li, "cuts": int(C), "cuts_xyz": cuts.tolist(), "cells_xyz": n.tolist(),
               "cubes": cubes, "start_frame": start, "enter_frames": enter,
               "strokes_start": f, "strokes_end": f + dur, "collapse_start": f_end,
               "collapse_end": f_end + collapse, "exit_start": f_end + collapse + hold,
               "end_frame": f_end + collapse + hold + exit_f,
               "cell_size": (ext / n).tolist(), "lo": lo.tolist(), "ext": ext.tolist()}
        pieces_ev.append((f_end, cubes, li))
        levels.append(lvl)
        strokes.extend(lvl_strokes)
        frame = lvl["end_frame"]
    # seamless loop: a fresh, uncut object slides back into the exact opening pose,
    # so the last frame flows into the first one when TikTok/Reels replays the video
    loop_in = frame
    total = frame + int(1.0 * fps)
    keys = KT.close_loop(knife_keys, first_blade, prev_blade, total, centre, safe_r) if first_blade else []
    return {
        "version": 3, "kind": "grid", "config": cfg, "fps": fps,
        "total_frames": int(total), "duration_s": total / fps,
        "levels": levels, "strokes": strokes, "loop_in": int(loop_in),
        "blade_strokes": blade_strokes, "knife_scale": knife_scale,
        "object_centre": centre.tolist(),
        "knife_keys": keys,
        "knife_track": KT.sample_keys(keys, range(1, int(total) + 1)),
        "counters": {"cuts": cuts_ev, "pieces": pieces_ev},
        "stats": {"levels": len(levels), "blade_strokes": len(strokes),
                  "final_cubes": levels[-1]["cubes"] if levels else 0,
                  "plan_seconds": round(time.time() - t0, 2)},
    }


def plan(cfg):
    """Fits the configured video_duration (seconds) when given, by scaling the pacing."""
    p = _plan(cfg)
    target = cfg.get("video_duration")
    if target:
        ts = 1.0
        for _ in range(6):
            ts *= target / p["duration_s"]
            ts = min(2.5, max(0.25, ts))
            p = _plan(cfg, ts)
            if abs(p["duration_s"] - target) < 1.0:
                break
        p["stats"]["time_scale"] = round(ts, 3)
    return p


def main():
    cfg = load_config(sys.argv[1])
    p = plan(cfg)
    with open(sys.argv[2], "w", encoding="utf-8") as f:
        json.dump(p, f)
    print(json.dumps({**p["stats"], "total_frames": p["total_frames"],
                      "duration_s": round(p["duration_s"], 1),
                      "levels": [(l["cuts"], l["cubes"]) for l in p["levels"]]}))


if __name__ == "__main__":
    main()

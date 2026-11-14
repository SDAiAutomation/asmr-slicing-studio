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


FIRST_CUT_S = 0.5   # every level opens on calm, clearly separate strokes


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
    lo_a, hi_a = 1e-6, 2.0
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
    prev_ready = None
    last_end = 0
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
        targets = cut_schedule(n_cuts, level_seconds(C) * ts * fps, FIRST_CUT_S * fps)
        pose = KT.Pose(0.0, centre, height / 2)
        cut_frames = [f] * n_cuts
        cut_group = [0] * n_cuts
        lvl_blades = []

        def stroke_for(j):
            ax, k = order[j]
            p_obj = (lo + ext / 2).copy()
            p_obj[ax] = lo[ax] + ext[ax] * k / n[ax]
            normal = np.zeros(3)
            normal[ax] = 1.0
            return KT.plan_stroke(shape, pose, p_obj, normal, radius, knife_scale, 0.0)

        # One real blade pass per cut while the rhythm allows it. The cut rate accelerates
        # through the level; once it outruns the fastest possible stroke, each pass takes
        # every cut that is due before it leaves the object (the counter still climbs 1 by 1).
        cursor, j, gi = f, 0, 0
        passes = []
        single_pass = cfg.get("progressive_dicing", False)
        while j < n_cuts:
            st = stroke_for(j)
            if st is None:   # plane grazing the outline: the cut lands without its own pass
                if single_pass:
                    raise ValueError(f"Cannot animate cut {j + 1} at level {li + 1}")
                cut_frames[j] = int(round(max(cursor, f + targets[j])))
                cut_group[j] = gi
                cursor = cut_frames[j]
                j, gi = j + 1, gi + 1
                continue
            p1, q1 = KT.ready_pose(st)
            fly = KT.transit_frames(prev_ready[0], prev_ready[1], p1, q1, centre, safe_r) if prev_ready else 0
            ready = max(cursor, last_end + fly)   # the blade flies in during the transition
            end = max(ready + KT.fast_frames(st), int(round(f + targets[j])) + 3)
            if single_pass:
                end = max(end, ready + KT.body_frames(st), ready + round(0.46 * fps))
            if end - ready >= KT.body_frames(st):
                tm = KT.timing(ready, end, resistance, KT.retract_frames(st))
            else:
                tm = KT.quick_timing(ready, end, st)
            k = j + 1
            while not single_pass and k < n_cuts and f + targets[k] <= end and order[k][0] == order[j][0]:
                k += 1
            g = list(range(j, k))
            st.update(level=li, axis=int(order[j][0]), group_cuts=len(g),
                      resistance=resistance, time=tm)
            lvl_blades.append(st)
            passes.append((g, tm))
            for jj in g:
                cut_group[jj] = gi
            prev_ready = (p1, q1)
            cursor, j, gi, last_end = end, k, gi + 1, end
        # The counter climbs one cut at a time: a single cut lands as the edge leaves the
        # object; a fast pass carrying many cuts keeps them coming, one per frame or more,
        # until the next pass touches the object, so the count never jumps by a block.
        for i, (g, tm) in enumerate(passes):
            c0, x0 = tm["contact"], tm["exit"]
            if single_pass:
                cut_frames[g[0]] = c0
                continue
            if len(g) <= x0 - c0 + 1:
                span_end = x0
            else:
                span_end = passes[i + 1][1]["contact"] - 1 if i + 1 < len(passes) else tm["retract_end"]
            for q, jj in enumerate(g):
                cut_frames[jj] = int(round(c0 + (span_end - c0) * (q + 1) / len(g)))
        cut_frames = np.maximum.accumulate(np.asarray(cut_frames, int)).tolist()
        dur = cursor - f
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
    return {
        "version": 3, "kind": "grid", "config": cfg, "fps": fps,
        "total_frames": int(total), "duration_s": total / fps,
        "levels": levels, "strokes": strokes, "loop_in": int(loop_in),
        "blade_strokes": blade_strokes, "knife_scale": knife_scale,
        "object_centre": centre.tolist(),
        "knife_track": KT.build_track(blade_strokes, int(total), centre, safe_r),
        "counters": {"cuts": cuts_ev, "pieces": pieces_ev},
        "stats": {"levels": len(levels), "blade_strokes": len(blade_strokes),
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


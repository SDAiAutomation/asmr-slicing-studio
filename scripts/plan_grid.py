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


FIRST_CUT_S = 0.5   # every level opens on a calm, clearly separate pass


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
_STROKES = {}


def occupancy(shape, lo, ext, n):
    key = (id(shape), tuple(int(v) for v in n))
    if key not in _OCC:  # duration fitting re-plans several times: compute each grid once
        _OCC[key] = grid_occupancy(shape, lo, ext, n)
    return _OCC[key]


def boxes_count(occ, made):
    """Non-empty boxes when only the cuts in made[a] (cut index k separates cells k-1 and k,
    1 <= k <= n-1) are done on each axis."""
    r = occ
    for a in range(3):
        n = r.shape[a]
        ks = sorted(int(k) for k in made[a] if 0 < k < n)
        if len(ks) >= n - 1:
            continue
        r = np.logical_or.reduceat(r, np.array([0] + ks), axis=a)
    return int(np.count_nonzero(r))


PASS_RATE = (1.2, 4.6)   # visible blade passes per second: first level -> last level
PASS_CAP = (6, 22)       # most passes a level may use: first level -> last level


def pass_budget(C, li, n_levels):
    """Visible passes of a level and their mean rate (passes/s). Both climb from level to
    level, so the tool visibly speeds up as the cut count grows. While the cuts fit in the
    budget, one knife makes one cut per pass; beyond it each pass carries a comb of
    parallel blades (allocate_passes), so every announced cut is really made."""
    u = li / max(1, n_levels - 1)
    rate = PASS_RATE[0] + (PASS_RATE[1] - PASS_RATE[0]) * u ** 0.9
    cap = round(PASS_CAP[0] + (PASS_CAP[1] - PASS_CAP[0]) * u)
    return int(min(C, cap)), rate


def allocate_passes(cuts, P):
    """Passes per axis, proportional to the cuts on that axis (>= 1 where there are cuts)."""
    C = int(cuts.sum())
    p = np.array([0 if c == 0 else max(1, min(c, round(P * c / C))) for c in cuts])
    while p.sum() > max(P, int((cuts > 0).sum())):
        i = int(np.argmax(np.where(p > 1, cuts / np.maximum(p, 1), -1)))
        if p[i] <= 1:
            break
        p[i] -= 1
    while p.sum() < P:
        room = np.where(p < cuts, cuts / np.maximum(p, 1), -1)
        i = int(np.argmax(room))
        if room[i] < 0:
            break
        p[i] += 1
    return p


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
    n_levels = len(cfg["cuts_sequence"])
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
        # Passes, axis by axis: X left->right, Y front->back, Z bottom->top (a comb's spare
        # blades then pass above the object, never through the board). With p passes
        # for c cuts on an axis, pass j makes the cuts j+1, j+1+p, j+1+2p... at once: a comb of
        # ceil(c/p) blades spaced p cells apart. p == c is the single knife, one cut per pass.
        P, rate = pass_budget(C, li, n_levels)
        cut_s = max(0.8, P / rate) * ts
        p_ax = allocate_passes(cuts, P)
        passes_def = []
        for ax in range(3):
            pa, ca = int(p_ax[ax]), int(cuts[ax])
            for j in range(pa):
                passes_def.append((ax, list(range(j + 1, ca + 1, pa)), pa))
        n_pass = len(passes_def)
        targets = cut_schedule(n_pass, cut_s * fps, FIRST_CUT_S * fps)
        pose = KT.Pose(0.0, centre, height / 2)
        lvl_blades, cut_rows = [], []

        def plane_pos(ax, k):
            return lo[ax] + ext[ax] * k / n[ax]

        def stroke_for(ax, ks, stride):
            normal = np.zeros(3)
            normal[ax] = 1.0
            # the comb's whole span: sample a few of its planes, keep the earliest contact
            # and the latest exit, anchor the trajectory on the first blade
            blades = int(math.ceil(cuts[ax] / stride))
            span = [ks[0] + stride * i for i in range(blades)]
            span = [k for k in span if 0 < k < n[ax]] or [ks[0]]
            picks = sorted({span[0], span[len(span) // 2], span[-1], span[len(span) // 4], span[3 * len(span) // 4]},
                           key=span.index)
            sts = []
            for k in picks:
                key = (cfg["object_type"], ax, round(float(plane_pos(ax, k)), 6))
                if key not in _STROKES:   # duration fitting re-plans several times
                    p_obj = (lo + ext / 2).copy()
                    p_obj[ax] = plane_pos(ax, k)
                    _STROKES[key] = KT.plan_stroke(shape, pose, p_obj, normal, radius, knife_scale, 0.0)
                if _STROKES[key] is not None:
                    sts.append((k, _STROKES[key]))
            if not sts:
                return None
            mid = sts[len(sts) // 2][1]
            first = ks[0]
            st = dict(mid)
            shift = plane_pos(ax, first) - plane_pos(ax, sts[len(sts) // 2][0])
            st["anchor"] = (np.asarray(mid["anchor"]) + np.asarray(mid["n"]) * shift * float(np.sign(mid["n"][ax]))).tolist()
            st["s_in"] = min(x["s_in"] for _, x in sts)
            st["s_out"] = max(x["s_out"] for _, x in sts)
            st["s_end"] = max(x["s_end"] for _, x in sts)
            st["s_ready"] = st["s_in"] - KT.CLEARANCE
            st["draw"] = min(x["draw"] for _, x in sts)
            # blade offsets along the world normal, measured from the first blade
            st["blades"] = [float(plane_pos(ax, ks[0] + stride * i) - plane_pos(ax, ks[0]))
                            for i in range(blades)]
            return st

        cursor, prev_ax = f, None
        for j, (ax, ks, stride) in enumerate(passes_def):
            st = stroke_for(ax, ks, stride)
            if st is None:   # plane grazing the outline: the cuts land without a visible pass
                fr = int(round(max(cursor, f + targets[j])))
                cut_rows.append((ax, ks, fr, [fr] * len(ks), None))
                cursor = fr
                continue
            p1, q1 = KT.ready_pose(st)
            fly = KT.transit_frames(prev_ready[0], prev_ready[1], p1, q1, centre, safe_r) if prev_ready else 0
            if prev_ax is not None and ax != prev_ax:
                fly += 2   # the comb is swapped above the object between axes
            ready = max(cursor, last_end + fly)
            end = max(ready + KT.fast_frames(st), int(round(f + targets[j])) + 3)
            if end - ready >= KT.body_frames(st):
                tm = KT.timing(ready, end, resistance, KT.retract_frames(st))
            else:
                tm = KT.quick_timing(ready, end, st)
            # blade offsets along the knife's own local Z (the comb is one rigid tool)
            zsign = float(KT.quat_to_mat(np.asarray(st["q"]))[:, 2] @ np.asarray(st["n"]))
            st.update(level=li, axis=int(ax), group_cuts=len(ks), stride=int(stride),
                      resistance=resistance, time=tm, blade_z=[o * zsign for o in st["blades"]])
            lvl_blades.append(st)
            # every blade of the comb goes through between contact and exit: the counter
            # climbs over that window
            c0, x0 = tm["contact"], tm["exit"]
            frs = [int(round(c0 + (x0 - c0) * (q + 1) / len(ks))) for q in range(len(ks))]
            cut_rows.append((ax, ks, x0, frs, len(lvl_blades) - 1))
            prev_ready, prev_ax = (p1, q1), ax
            cursor, last_end = end, end
        dur = cursor - f
        blade_strokes.extend(lvl_blades)
        made = [[], [], []]
        lvl_strokes, done = [], 0
        for g, (ax, ks, x0, frs, bi) in enumerate(cut_rows):
            for k, fr in zip(ks, frs):
                done += 1
                lvl_strokes.append({"level": li, "axis": int(ax), "index": int(k), "pos": float(plane_pos(ax, k)),
                                    "frame": int(fr), "n_in_level": done, "group": g})
                cuts_ev.append((int(fr), done, li))
            made[ax].extend(ks)
            pieces_ev.append((int(x0), boxes_count(occ, made), li))
        lvl_passes = [{"axis": int(ax), "frame": int(x0), "count": len(ks),
                       "stride": int(p_ax[ax]), "cuts_on_axis": int(cuts[ax])}
                      for ax, ks, x0, frs, bi in cut_rows]
        f_end = f + dur + 2
        collapse = max(int(0.45 * fps), int(0.75 * fps * ts))
        hold = max(int(0.35 * fps), int(0.7 * fps * ts))
        if li == n_levels - 1:   # the payoff: the last collapse plays in slow motion, then lingers
            collapse, hold = int(collapse * 2.2), int(hold * 1.8)
        exit_f = max(int(0.3 * fps), int(0.4 * fps * ts))
        lvl = {"level": li, "cuts": int(C), "cuts_xyz": cuts.tolist(), "cells_xyz": n.tolist(),
               "cubes": cubes, "start_frame": start, "enter_frames": enter,
               "strokes_start": f, "strokes_end": f + dur, "collapse_start": f_end,
               "collapse_end": f_end + collapse, "exit_start": f_end + collapse + hold,
               "end_frame": f_end + collapse + hold + exit_f,
               "cell_size": (ext / n).tolist(), "lo": lo.tolist(), "ext": ext.tolist(),
               "passes": lvl_passes, "passes_xyz": p_ax.tolist()}
        pieces_ev.append((f_end, cubes, li))
        levels.append(lvl)
        strokes.extend(lvl_strokes)
        frame = lvl["end_frame"]
    # seamless loop: a fresh, uncut object slides back into the exact opening pose,
    # so the last frame flows into the first one when TikTok/Reels replays the video
    loop_in = frame
    total = frame + int(1.0 * fps)
    tools = tool_windows(blade_strokes, int(total))
    return {
        "version": 3, "kind": "grid", "config": cfg, "fps": fps,
        "total_frames": int(total), "duration_s": total / fps,
        "levels": levels, "strokes": strokes, "loop_in": int(loop_in),
        "blade_strokes": blade_strokes, "knife_scale": knife_scale,
        "object_centre": centre.tolist(),
        "knife_track": KT.build_track(blade_strokes, int(total), centre, safe_r),
        "tools": tools,
        "counters": {"cuts": cuts_ev, "pieces": pieces_ev},
        "stats": {"levels": len(levels), "blade_strokes": len(strokes),
                  "final_cubes": levels[-1]["cubes"] if levels else 0,
                  "plan_seconds": round(time.time() - t0, 2)},
    }


def tool_windows(blade_strokes, total):
    """Which tool is in hand when: the single knife or a comb of parallel blades.
    One entry per run of passes sharing the same blades; the swap happens halfway through
    the flight between two runs, high above the object."""
    runs = []
    for i, b in enumerate(blade_strokes):
        key = (b["level"], b["axis"], tuple(round(z, 7) for z in b["blade_z"]))
        if runs and runs[-1]["key"] == key:
            runs[-1]["last"] = i
        else:
            runs.append({"key": key, "first": i, "last": i})
    out = []
    for r_i, r in enumerate(runs):
        first, last = blade_strokes[r["first"]]["time"], blade_strokes[r["last"]]["time"]
        start = 1 if r_i == 0 else out[-1]["hide"]
        if r_i + 1 < len(runs):
            nxt = blade_strokes[runs[r_i + 1]["first"]]["time"]
            hide = (last["retract_end"] + nxt["ready"] + 1) // 2
        else:
            hide = total + 1
        out.append({"level": r["key"][0], "axis": r["key"][1], "blade_z": list(r["key"][2]),
                    "show": int(start), "hide": int(hide), "strokes": [r["first"], r["last"]]})
    return out


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

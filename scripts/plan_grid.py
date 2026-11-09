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
from common import OBJECTS, MATERIALS, grid_occupancy, load_config, levitate_lift


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
    levitate = cfg["format"] == "levitate"
    span = float(max(ext[0], ext[1]))
    lift = levitate_lift(cfg, shape) if levitate else 0.0
    height = float(ext[2])
    centre = np.array([0.0, 0.0, lift + height / 2])
    radius = float(np.linalg.norm(ext)) / 2 * 1.15
    # the blade grows with the object so its edge always covers the whole section
    width_needed = float(np.hypot(ext[0], ext[1]))
    knife_scale = float(min(1.3, max(0.85, width_needed * 1.05 / (KT.EDGE_X[-1] - KT.EDGE_X[0]))))
    resistance = KT.RESISTANCE.get(cfg["object_type"], 0.5)
    yaws = [-0.32, 0.42, -0.5, 0.24]
    safe_r = radius * 1.25 + 0.03
    prev_ready = None
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
        dur = max(int(0.5 * fps), int(round(ts * fps * min(4.5, max(1.0, 1.0 + 0.55 * math.log2(C / 4 + 1))))))
        # stroke order: X cuts left->right, then Y cuts front->back, then Z (horizontal)
        order = [(0, k) for k in range(1, cuts[0] + 1)] + \
                [(1, k) for k in range(1, cuts[1] + 1)] + \
                [(2, k) for k in range(cuts[2], 0, -1)]
        # Visible blade passes: one per cut while it stays readable, then each pass stands for
        # a group of neighbouring cuts (they appear while the blade crosses the object).
        min_slot = max(8, int(0.36 * fps))
        n_cuts = max(1, len(order))
        readable = n_cuts <= 20   # every cut gets its own, unhurried blade pass
        if readable:
            min_slot = max(min_slot, int(0.4 * fps))
        visible = n_cuts if readable else min(n_cuts, max(1, dur // min_slot))
        yaw = yaws[li % 4] if levitate else 0.0
        pose = KT.Pose(yaw, centre, height / 2)
        cut_frames = [f] * n_cuts
        cut_group = [0] * n_cuts
        groups = [g for g in np.array_split(np.arange(n_cuts), visible) if len(g)]
        lvl_blades, planned = [], []
        for gi, g in enumerate(groups):
            mid = int(g[len(g) // 2])
            ax, k = order[mid]
            p_obj = (lo + ext / 2).copy()
            p_obj[ax] = lo[ax] + ext[ax] * k / n[ax]
            normal = np.zeros(3)
            normal[ax] = 1.0
            st = KT.plan_stroke(shape, pose, p_obj, normal, radius, knife_scale, 0.0)
            if st is None:   # a plane grazing the outline: try the group's other planes
                for alt in g:
                    ax, k = order[int(alt)]
                    p_obj = (lo + ext / 2).copy()
                    p_obj[ax] = lo[ax] + ext[ax] * k / n[ax]
                    normal = np.zeros(3)
                    normal[ax] = 1.0
                    st = KT.plan_stroke(shape, pose, p_obj, normal, radius, knife_scale, 0.0)
                    if st is not None:
                        break
            if st is not None:
                st.update(level=li, axis=int(ax), group_cuts=len(g), resistance=resistance)
            planned.append((g, st))
        # timing: each pass gets the frames it needs (flight between planes + its own stroke),
        # any spare time of the level is shared out so the pacing stays even
        needs = []
        for g, st in planned:
            if st is None:
                needs.append((0, min_slot))
                continue
            p1, q1 = KT.ready_pose(st)
            fly = KT.transit_frames(prev_ready[0], prev_ready[1], p1, q1, centre, safe_r) if prev_ready else 0
            body = max(min_slot, KT.body_frames(st))
            needs.append((fly, body))
            prev_ready = (p1, q1)
        # readable levels take exactly the time their passes need: a brisk, steady blade
        dur = sum(a + b for a, b in needs) if readable else max(dur, sum(a + b for a, b in needs))
        spare = dur - sum(a + b for a, b in needs)
        cursor = f
        for gi, ((g, st), (fly, body)) in enumerate(zip(planned, needs)):
            body += spare // len(planned) + (1 if gi < spare % len(planned) else 0)
            ready = cursor + fly
            end = ready + body
            if st is not None:
                st["time"] = KT.timing(ready, end, resistance, KT.retract_frames(st))
                lvl_blades.append(st)
                c0, x0 = st["time"]["contact"], st["time"]["exit"]
            else:
                c0, x0 = ready, ready
            for q, j in enumerate(g):
                cut_frames[int(j)] = int(round(c0 + (x0 - c0) * (q + 1) / len(g)))
                cut_group[int(j)] = gi
            cursor = end
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
        collapse = min(int(0.85 * fps), max(int(0.45 * fps), int(0.75 * fps * ts)))  # gravity does not slow down
        hold = max(int(0.35 * fps), int(0.7 * fps * ts))
        exit_f = max(int(0.3 * fps), int(0.4 * fps * ts))
        lvl = {"level": li, "cuts": int(C), "cuts_xyz": cuts.tolist(), "cells_xyz": n.tolist(),
               "cubes": cubes, "start_frame": start, "enter_frames": enter,
               "strokes_start": f, "strokes_end": f + dur, "collapse_start": f_end,
               "collapse_end": f_end + collapse, "exit_start": f_end + collapse + hold,
               "end_frame": f_end + collapse + hold + exit_f,
               "cell_size": (ext / n).tolist(), "lo": lo.tolist(), "ext": ext.tolist(), "yaw": yaw}
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
        "blade_strokes": blade_strokes, "knife_scale": knife_scale, "lift": lift,
        "object_centre": centre.tolist(),
        "knife_track": KT.build_track(blade_strokes, int(total), centre, safe_r),
        "object_reaction": KT.reaction_track(blade_strokes, int(total), resistance),
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

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

from common import OBJECTS, MATERIALS, load_config


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


def occupancy(shape, lo, ext, n):
    """Bool grid of cells whose centre lies inside the shape (computed in x-slabs)."""
    axes = [lo[a] + (np.arange(n[a]) + 0.5) * ext[a] / n[a] for a in range(3)]
    occ = np.zeros(tuple(n), bool)
    Y, Z = np.meshgrid(axes[1], axes[2], indexing="ij")
    for i, x in enumerate(axes[0]):
        occ[i] = shape.inside(np.full_like(Y, x), Y, Z)
    return occ, axes


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


def plan(cfg):
    t0 = time.time()
    shape = OBJECTS[cfg["object_type"]]
    fps = cfg["fps"]
    lo, hi = shape.bbox()
    # tight extents without bbox padding
    ext = hi - lo
    rng = np.random.default_rng(cfg["random_seed"])

    levels, strokes, cuts_ev, pieces_ev = [], [], [], []
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
        enter = int(0.45 * fps) if li else 0
        f = frame + enter + (intro if li == 0 else int(0.25 * fps))
        dur = int(round(fps * min(4.5, max(1.0, 1.0 + 0.55 * math.log2(C / 4 + 1)))))
        # stroke order: X cuts left->right, then Y cuts front->back, then Z (horizontal)
        order = [(0, k) for k in range(1, cuts[0] + 1)] + \
                [(1, k) for k in range(1, cuts[1] + 1)] + \
                [(2, k) for k in range(cuts[2], 0, -1)]
        active = [0, 0, 0]
        z_done = 0
        last_eval_frame = None
        lvl_strokes = []
        for j, (ax, k) in enumerate(order):
            sf = f + (j * dur) // max(1, len(order))
            pos = lo[ax] + ext[ax] * k / n[ax]
            if ax == 2:
                z_done += 1
                active[2] = cuts[2]  # horizontal cuts come from the top: count them as they land
            else:
                active[ax] = k
            st = {"level": li, "axis": int(ax), "index": int(k), "pos": float(pos),
                  "frame": int(sf), "n_in_level": j + 1}
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
        collapse = int(0.75 * fps)
        hold = int(0.7 * fps)
        exit_f = int(0.4 * fps)
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
        "counters": {"cuts": cuts_ev, "pieces": pieces_ev},
        "stats": {"levels": len(levels), "blade_strokes": len(strokes),
                  "final_cubes": levels[-1]["cubes"] if levels else 0,
                  "plan_seconds": round(time.time() - t0, 2)},
    }


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

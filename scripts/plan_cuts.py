"""Progressive cut planner.

Turns a config into a deterministic plan.json: every blade stroke, every cutting
plane, every piece (with its parent), keyframe-ready timings, camera framing and
an expected piece-count timeline.

Geometry is tracked on a voxelised copy of the exact same shape Blender builds,
so each split is checked for real connectivity (a cut that produces 3 parts is
counted as 3, never assumed to be 2). Blender later re-checks on the meshes and
remains the final source of truth for the on-screen counter.

Usage:
    python plan_cuts.py config.json plan.json
"""
import json
import math
import sys
import time

import numpy as np
from scipy import ndimage

from common import OBJECTS, MATERIALS, direction_vector, load_config

MIN_VOX = 6          # smallest voxel count we allow a piece to have
NOISE_VOX = 3        # connected bits smaller than this are voxel noise
DENSE_K = 12         # above this many cuts in one stroke, cuts overlap in time


class Grid:
    def __init__(self, shape, res):
        lo, hi = shape.bbox()
        size = hi - lo
        self.h = float(size.max() / res)
        self.dims = np.ceil(size / self.h).astype(int) + 1
        self.lo = lo
        ax = [lo[i] + (np.arange(self.dims[i]) + 0.5) * self.h for i in range(3)]
        X, Y, Z = np.meshgrid(*ax, indexing="ij")
        inside = shape.inside(X, Y, Z)
        self.flat_inside = np.flatnonzero(inside.ravel())
        self.coords = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
        self.vox_vol = self.h ** 3

    def pts(self, idx):
        return self.coords[idx]

    def components(self, idx):
        """Split a voxel index set into 26-connected components."""
        ijk = np.stack(np.unravel_index(idx, self.dims), axis=1)
        mn = ijk.min(0)
        sub = ijk - mn
        g = np.zeros(sub.max(0) + 1, bool)
        g[tuple(sub.T)] = True
        lab, n = ndimage.label(g, structure=np.ones((3, 3, 3), bool))
        if n <= 1:
            return [idx]
        labels = lab[tuple(sub.T)]
        comps = [idx[labels == k] for k in range(1, n + 1)]
        comps.sort(key=len, reverse=True)
        # fragments under 4 % of the half are voxel staircase artefacts, not real parts
        thr = max(NOISE_VOX, int(0.04 * len(idx)), 2 * MIN_VOX)
        big = [c for c in comps if len(c) >= thr] or comps[:1]
        noise = [c for c in comps if len(c) < thr and c is not big[0]]
        if noise:  # voxel staircase noise -> attach to the largest part
            big[0] = np.concatenate([big[0]] + noise)
        return big


class Piece:
    _next = 0

    def __init__(self, idx, grid, parent=None, level=0):
        self.id = Piece._next
        Piece._next += 1
        self.idx = idx
        self.parent = parent
        self.level = level
        p = grid.pts(idx)
        self.centroid = p.mean(0)
        self.volume = len(idx) * grid.vox_vol
        self.born_frame = None
        self.cut_frame = None
        self.keys = []  # [(frame, [dx,dy,dz])]
        self.children = []

    def offset_at(self, frame):
        if not self.keys:
            return np.zeros(3)
        ks = self.keys
        if frame <= ks[0][0]:
            return np.asarray(ks[0][1])
        for (f0, o0), (f1, o1) in zip(ks, ks[1:]):
            if f0 <= frame <= f1:
                t = 0.0 if f1 == f0 else (frame - f0) / (f1 - f0)
                t = t * t * (3 - 2 * t)  # matches Blender bezier ease closely
                return np.asarray(o0) + (np.asarray(o1) - np.asarray(o0)) * t
        return np.asarray(ks[-1][1])

    def add_key(self, frame, off):
        off = [float(x) for x in off]
        if self.keys and self.keys[-1][0] >= frame:
            # keep keys strictly increasing; overwrite a key at the same frame
            self.keys = [k for k in self.keys if k[0] < frame]
        self.keys.append((int(frame), off))


def explode_factor(n_pieces, sep):
    return sep * (1.5 + 2.0 * math.log10(max(n_pieces, 1)))


def principal_axis(points):
    c = points - points.mean(0)
    w, v = np.linalg.eigh(c.T @ c)
    return v[:, -1]


def choose_plane(piece, grid, n_dir):
    """Plane through the piece centroid with the stroke direction; falls back to
    the piece's longest axis when the requested slice would be a sliver."""
    p = grid.pts(piece.idx)
    for attempt, n in enumerate([n_dir, principal_axis(p)]):
        d = (p - piece.centroid) @ n
        neg, pos = np.count_nonzero(d < 0), np.count_nonzero(d >= 0)
        if min(neg, pos) >= max(MIN_VOX, 0.22 * len(d)):
            return n / np.linalg.norm(n), attempt == 1
    return None, True


def travel_direction(n):
    """Blade moves downwards inside the cutting plane when it can."""
    down = np.array([0.0, 0.0, -1.0])
    d = down - (down @ n) * n
    if np.linalg.norm(d) < 0.25:  # horizontal slice: blade slides sideways
        side = np.array([1.0, 0.0, 0.0])
        d = side - (side @ n) * n
        if np.linalg.norm(d) < 0.25:
            side = np.array([0.0, 1.0, 0.0])
            d = side - (side @ n) * n
    return d / np.linalg.norm(d)


def speed_seconds(cfg):
    return {"slow": 0.55, "medium": 0.38, "auto": 0.42, "fast": 0.24}[cfg["cutting_speed"]]


def stroke_targets(seq, start=1):
    """Expand the progression into strokes; one stroke can at most double."""
    out, cur = [], start
    for tgt in seq:
        while cur < tgt:
            nxt = min(tgt, cur * 2)
            out.append((nxt, tgt))
            cur = nxt
    return out


def plan(cfg):
    t0 = time.time()
    rng = np.random.default_rng(cfg["random_seed"])
    shape = OBJECTS[cfg["object_type"]]
    mat = MATERIALS[cfg["material_profile"]]
    fps = cfg["fps"]
    grid = Grid(shape, cfg["grid_resolution"])
    Piece._next = 0

    s0 = speed_seconds(cfg)
    min_space = 2
    intro = int(round(1.2 * fps))
    pre = int(round(0.35 * fps))
    post = int(round(0.45 * fps))
    outro = int(round(1.8 * fps))
    sep_frames = int(round(0.3 * fps))

    seq = cfg["progression_sequence"]
    levels = [seq] if cfg["cutting_mode"] == "cumulative" else [[t] for t in seq]

    all_pieces, hits, strokes, counter, camera, level_info = [], [], [], [], [], []
    frame = 1
    dirs = cfg["cutting_directions"]
    stroke_no = 0
    fallbacks = 0
    level_shift = np.array([0.0, 0.0, 0.0])

    for li, lvl_seq in enumerate(levels):
        root = Piece(grid.flat_inside.copy(), grid, level=li)
        root.born_frame = frame
        enter = np.array([0.45, 0.0, 0.0]) if li > 0 else np.zeros(3)
        root.add_key(frame, enter)
        if li > 0:
            root.add_key(frame + int(0.6 * fps), [0, 0, 0])
            frame += int(0.6 * fps)
        alive = {root.id: root}
        all_pieces.append(root)
        counter.append({"frame": frame if li else 1, "count": 1, "level": li})
        lvl_start = frame
        frame += intro if li == 0 else int(0.5 * fps)

        target_strokes = stroke_targets(lvl_seq if cfg["cutting_mode"] == "cumulative"
                                        else _level_substeps(lvl_seq[0]))
        final_target = lvl_seq[-1]
        si = 0
        while True:
            if si < len(target_strokes):
                stroke_target, milestone = target_strokes[si]
            elif len(alive) < final_target and si < len(target_strokes) + 3:
                stroke_target, milestone = final_target, final_target   # top-up stroke
            else:
                break
            si += 1
            if len(alive) >= stroke_target:
                continue
            dname = dirs[stroke_no % len(dirs)]
            n_dir = direction_vector(dname, stroke_no, rng)
            need = stroke_target - len(alive)
            cand = sorted([p for p in alive.values() if len(p.idx) >= 2 * MIN_VOX],
                          key=lambda p: -len(p.idx))
            if not cand:
                break
            chosen = cand[:need + need // 4 + 3]   # spares in case a piece refuses a clean cut
            # sweep order: knife walks across the board instead of jumping around
            sweep = np.cross(np.array([0, 0, 1.0]), n_dir)
            if np.linalg.norm(sweep) < 1e-3:
                sweep = np.array([1.0, 0, 0])
            chosen.sort(key=lambda p: float((p.centroid + p.offset_at(frame)) @ sweep))

            k = min(len(chosen), need)
            # sparse strokes: one visible cut after another; dense strokes: a slicing
            # wave across the board, several pieces per frame, bounded stroke length
            dense = k > DENSE_K
            if dense:
                stroke_frames = int(min(fps * 2.4, max(fps * 0.9, fps * (0.5 + 0.2 * math.log2(k)))))
                spacing, traverse = 0, 2
            else:
                spacing = max(min_space, int(round(fps * max(0.07, s0 * k ** -0.45))))
                traverse = max(2, int(round(spacing * 0.6)))
            frame += pre
            stroke_start = frame
            stroke = {"index": stroke_no, "direction": dname, "normal": n_dir.tolist(),
                      "start_frame": frame, "milestone": milestone, "hits": []}
            e_now = explode_factor(len(alive) + k, mat["separation"])
            for i_hit, p in enumerate(chosen):
                if len(alive) >= stroke_target:
                    break
                if dense:
                    frame = stroke_start + (min(i_hit, k - 1) * stroke_frames) // k
                n, fb = choose_plane(p, grid, n_dir)
                if n is None:
                    continue
                fallbacks += int(fb)
                pts = grid.pts(p.idx)
                d = (pts - p.centroid) @ n
                halves = [p.idx[d < 0], p.idx[d >= 0]]
                parts = []
                for hidx in halves:
                    parts.extend(grid.components(hidx))
                contact = frame
                exit_f = frame + traverse
                off_parent = p.offset_at(contact)
                ext = float(np.abs(d).max())
                travel = travel_direction(n)
                ext_travel = float(np.abs((pts - p.centroid) @ travel).max())
                p.cut_frame = exit_f
                kids = []
                for part in parts:
                    c = Piece(part, grid, parent=p.id, level=li)
                    c.born_frame = exit_f
                    c.add_key(exit_f, off_parent)
                    tgt = np.array([*(c.centroid[:2] * e_now), 0.0]) + level_shift
                    c.add_key(exit_f + sep_frames, tgt)
                    kids.append(c)
                    all_pieces.append(c)
                p.children = [c.id for c in kids]
                del alive[p.id]
                for c in kids:
                    alive[c.id] = c
                hit = {"stroke": stroke_no, "piece": p.id, "children": p.children,
                       "plane_point": p.centroid.tolist(), "plane_normal": n.tolist(),
                       "fallback_axis": bool(fb), "travel": travel.tolist(),
                       "extent": ext, "extent_travel": ext_travel,
                       "contact_frame": contact, "exit_frame": exit_f,
                       "offset": off_parent.tolist(),
                       "piece_volume": p.volume, "parts_created": len(kids),
                       "count_after": len(alive), "level": li}
                hits.append(hit)
                stroke["hits"].append(len(hits) - 1)
                counter.append({"frame": exit_f, "count": len(alive), "level": li})
                frame += spacing
            if dense:
                frame = stroke_start + stroke_frames
            stroke["dense"] = dense
            frame += traverse + post
            # settle every piece of this level to the new explode factor
            e_after = explode_factor(len(alive), mat["separation"])
            for q in alive.values():
                q.add_key(frame - post // 2, q.offset_at(frame - post // 2))
                q.add_key(frame, np.array([*(q.centroid[:2] * e_after), 0.0]) + level_shift)
            stroke["end_frame"] = frame
            stroke["count_after"] = len(alive)
            strokes.append(stroke)
            camera.append(_frame_camera(list(alive.values()), grid, frame, stroke["start_frame"]))
            stroke_no += 1

        level_info.append({"level": li, "start": lvl_start, "end": frame,
                           "pieces": len(alive), "target": lvl_seq[-1]})
        if cfg["cutting_mode"] == "levels" and li < len(levels) - 1:
            # slide the finished level off the board, nothing just vanishes
            out_f = frame + int(0.6 * fps)
            for q in alive.values():
                q.add_key(frame, q.offset_at(frame))
                q.add_key(out_f, q.offset_at(frame) + np.array([-0.6, 0, 0]))
                q.hide_frame = out_f + 1
            frame = out_f

    total = frame + outro
    if cfg["video_duration"]:
        # report only: retiming is done by changing cutting_speed
        pass

    pieces_json = []
    for p in all_pieces:
        pieces_json.append({
            "id": p.id, "parent": p.parent, "level": p.level,
            "voxels": int(len(p.idx)), "volume": p.volume,
            "centroid": p.centroid.tolist(), "born_frame": p.born_frame,
            "cut_frame": p.cut_frame, "children": p.children,
            "hide_frame": getattr(p, "hide_frame", None), "keys": p.keys})

    final_alive = [p for p in all_pieces if p.cut_frame is None and p.level == len(levels) - 1]
    return {
        "version": 2,
        "config": cfg,
        "fps": fps,
        "total_frames": int(total),
        "duration_s": total / fps,
        "grid": {"res": cfg["grid_resolution"], "voxel": grid.h,
                 "inside_voxels": int(len(grid.flat_inside))},
        "pieces": pieces_json,
        "hits": hits,
        "strokes": strokes,
        "counter": counter,
        "camera": camera,
        "levels": level_info,
        "stats": {
            "blade_strokes": len(strokes),
            "cut_planes": len(hits),
            "expected_final_pieces": len(final_alive),
            "target": cfg["target_piece_count"],
            "fallback_axis_cuts": fallbacks,
            "plan_seconds": round(time.time() - t0, 3),
        },
    }


def _level_substeps(target):
    # inside a level: double each stroke, finish with a partial stroke
    return [target]


def _frame_camera(alive, grid, frame, start):
    pts = []
    for p in alive:
        lo = grid.pts(p.idx).min(0) + p.offset_at(frame)
        hi = grid.pts(p.idx).max(0) + p.offset_at(frame)
        pts += [lo, hi]
    pts = np.array(pts)
    lo, hi = pts.min(0), pts.max(0)
    return {"frame": int(frame), "start_frame": int(start),
            "center": ((lo + hi) / 2).tolist(), "radius": float(np.linalg.norm(hi - lo) / 2)}


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(2)
    cfg = load_config(sys.argv[1])
    result = plan(cfg)
    with open(sys.argv[2], "w", encoding="utf-8") as f:
        json.dump(result, f)
    s = result["stats"]
    print(json.dumps({**s, "total_frames": result["total_frames"],
                      "duration_s": round(result["duration_s"], 2),
                      "progression": [st["count_after"] for st in result["strokes"]]}))


if __name__ == "__main__":
    main()

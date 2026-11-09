"""How the cut pieces fall: a contact-driven deposition, computed per piece (no Blender needed).

This is not a rigid-body solver (it has to handle up to ~1.2 M instances on free CPU runners).
Pieces settle column by column on a height field, so they cannot end inside each other or
under the board; a piece cannot land before the one that supports it; release follows the
geometry (free outer/upper pieces go first). Output drives the CubeCollapse node group.
"""
import numpy as np

# spread: how far the pile slumps outwards; spin: rad of tumble; bounce: rebound height (share
# of the fall, in the node-group curve); squash: flattening at impact; crumbs: decorative only
FALL = {
    "cake":      dict(spread=1.0, spin=0.55, bounce=0.02, squash=0.30, crumbs=True),
    "steak":     dict(spread=0.7, spin=0.30, bounce=0.012, squash=0.16, crumbs=False),
    "mango":     dict(spread=0.85, spin=0.60, bounce=0.04, squash=0.18, crumbs=False),
    "jelly":     dict(spread=1.0, spin=0.50, bounce=0.11, squash=0.40, crumbs=False),
    "soap":      dict(spread=0.9, spin=1.10, bounce=0.10, squash=0.04, crumbs=False),
    "chocolate": dict(spread=1.0, spin=1.30, bounce=0.12, squash=0.0, crumbs=False),
}
DEFAULT = dict(spread=0.85, spin=0.7, bounce=0.05, squash=0.1, crumbs=False)
LAND_U = 0.66     # local fall progress at which the curve first touches down
PACK = 0.94       # stacked height of a piece relative to its cell


def _ranges(counts):
    """Concatenated 0..c-1 for every c, without a Python loop."""
    ends = np.cumsum(counts)
    out = np.arange(ends[-1])
    out[counts[0]:] -= np.repeat(ends[:-1], counts[1:])
    return out


def params(material):
    return FALL.get(material, DEFAULT)


def fall_curve(bounce):
    """(progress, fallen share): free fall to a hard landing, then a damped rebound."""
    return [(0.0, 0.0), (0.2, 0.09), (0.4, 0.37), (0.55, 0.69), (LAND_U, 1.0),
            (0.76, 1.0 - bounce), (0.86, 1.0), (0.93, 1.0 - bounce * 0.25), (1.0, 1.0)]


def simulate(co, cell, ground_z, material, seed, max_foot=None, release=None, fall_frac=1.0):
    """co: Nx3 local centres, cell: piece size. Returns target (Nx3), delay, span, spin (N).
    release: optional per-piece start in the window (0..1): what is released first lands lowest;
    fall_frac: share of the window one fall lasts when pieces are released progressively."""
    P = params(material)
    rng = np.random.default_rng(seed)
    co = np.asarray(co, np.float64)
    n = len(co)
    cell = np.asarray(cell, np.float64)
    z0, h = co[:, 2].min(), max(float(np.ptp(co[:, 2])), 1e-6)
    rel = (co[:, 2] - z0) / h
    centre = co[:, :2].mean(0)
    r = co[:, :2] - centre
    ext = np.maximum(np.abs(r).max(0), cell[:2])
    dist = np.hypot(r[:, 0] / ext[0], r[:, 1] / ext[1])
    rmax = max(float(dist.max()), 1e-6)
    # the settled pile is a dome wider than the object; every piece gets its own slot in it,
    # so nothing can overlap or sink through the board
    foot = ext * (1.0 + 0.8 * P["spread"])
    limit_foot = None if max_foot is None else np.maximum(np.asarray(max_foot, float) - cell[:2] * 0.7, cell[:2])
    if limit_foot is not None:      # the whole pile stays on the board
        foot = np.minimum(foot, limit_foot)
    step = max(cell[0], cell[1]) * 1.02
    dz_cell = cell[2] * PACK
    limit = max(dz_cell, 0.72 * (h + cell[2]))       # a settled pile is lower than the object
    while True:
        nx, ny = (np.ceil(foot / step).astype(int) + 1)
        gx, gy = np.meshgrid(np.arange(-nx, nx + 1) * step, np.arange(-ny, ny + 1) * step, indexing="ij")
        keep = (gx / foot[0]) ** 2 + (gy / foot[1]) ** 2 <= 1.0
        if keep.sum() < 1:
            keep[nx, ny] = True
        gx, gy = gx[keep], gy[keep]
        rho2 = np.minimum(1.0, (gx / foot[0]) ** 2 + (gy / foot[1]) ** 2)
        top = max(dz_cell, 1.2 * n * dz_cell / len(gx))
        while True:
            cap = np.maximum(1, np.floor(top * np.sqrt(1.0 - rho2) / dz_cell)).astype(np.int64)
            if cap.sum() >= n:
                break
            top *= 1.05
        wider = foot * 1.06 if limit_foot is None else np.minimum(foot * 1.06, limit_foot)
        if top <= limit or np.allclose(wider, foot):
            break
        foot = wider
    col = np.repeat(np.arange(len(gx)), cap)
    lvl = np.concatenate([np.arange(c) for c in cap]) if len(cap) < 4096 else _ranges(cap)
    slot_r = np.hypot(gx[col] / foot[0], gy[col] / foot[1]) + lvl * 1e-6
    slot_order = np.argsort(slot_r, kind="stable")[:n]
    col, lvl = col[slot_order], lvl[slot_order]
    # radial order is kept (with some noise and the top thrown further), then azimuth inside rings
    key = dist / rmax * rng.uniform(0.82, 1.18, n) + 0.28 * rel
    pieces = np.argsort(key, kind="stable")
    n_rings = max(1, min(48, n // 12))
    azim_p = np.arctan2(r[:, 1] / ext[1], r[:, 0] / ext[0])
    azim_s = np.arctan2(gy[col] / foot[1], gx[col] / foot[0])
    assigned_col = np.empty(n, np.int64)
    assigned_lvl = np.empty(n, np.int64)
    for chunk_p, chunk_s in zip(np.array_split(pieces, n_rings), np.array_split(np.arange(n), n_rings)):
        # height first (low pieces take low slots), azimuth inside each height band
        def arrange(idx, height, azim):
            band = np.floor(np.argsort(np.argsort(height)) * 6 / max(1, len(idx)))
            return idx[np.lexsort((azim, band))]
        early_low = co[chunk_p, 2] if release is None else release[chunk_p] + 1e-3 * rel[chunk_p]
        p = arrange(chunk_p, early_low, azim_p[chunk_p])
        q = arrange(chunk_s, lvl[chunk_s] + 1e-3 * np.arange(len(chunk_s)), azim_s[chunk_s])
        assigned_col[p] = col[q]
        assigned_lvl[p] = lvl[q]
    jit = (rng.random((n, 2)) - 0.5) * step * 0.32
    xy = centre + np.column_stack((gx[assigned_col], gy[assigned_col])) + jit
    tz = ground_z + (assigned_lvl + 0.5) * dz_cell
    target = np.column_stack((xy, tz))
    dz = np.maximum(0.0, co[:, 2] - target[:, 2])
    # release: outside and top first, the inner core follows once it has room
    delay = 0.24 * (1.0 - dist / rmax) + 0.08 * (1.0 - rel) + 0.05 * rng.random(n)
    fall = np.sqrt(dz / max(float(dz.max()), 1e-6))      # free fall: time ~ sqrt(height)
    span = np.maximum(0.35, 0.8 * fall)
    if release is not None:
        delay = np.asarray(release, float) + delay * fall_frac
        span = span * fall_frac
    land = delay + span * LAND_U
    # contact order: within a column a piece cannot land before the one under it
    order = np.lexsort((assigned_lvl, assigned_col))
    gid = assigned_col[order]
    ls = land[order] + gid * 10.0
    ls = np.maximum.accumulate(ls) - gid * 10.0
    land[order] = ls
    span = np.maximum(span, (land - delay) / LAND_U)
    k = float((delay + span).max())
    if release is not None:      # a slice never leaves before its blade: shorten the fall instead
        span = np.minimum(span, np.maximum(1.0 - delay, 0.01))
    elif k > 1.0:
        delay, span = delay / k, span / k
    spin = P["spin"] * np.minimum(1.0, dz / max(1e-6, 4.0 * float(cell[2]))) * rng.uniform(0.4, 1.0, n)
    return {"target": target.astype(np.float32), "delay": delay.astype(np.float32),
            "span": span.astype(np.float32), "spin": spin.astype(np.float32),
            "scale": np.ones(n, np.float32)}


def add_crumbs(co, kinds, fall, cell, ground_z, seed, max_crumbs=1500):
    """Decorative crumbs thrown off the surface (the piece counters never include them)."""
    rng = np.random.default_rng(seed + 991)
    m = int(min(max_crumbs, len(co) * 0.5))
    if m < 8:
        return co, kinds, fall
    idx = rng.integers(0, len(co), m)
    centre = co[:, :2].mean(0)
    target = fall["target"][idx].copy()
    target[:, :2] += (target[:, :2] - centre) * rng.uniform(0.05, 0.35, m)[:, None]
    target[:, 2] = ground_z + float(cell[2]) * 0.06
    out = dict(fall)
    out["target"] = np.vstack((fall["target"], target)).astype(np.float32)
    out["delay"] = np.concatenate((fall["delay"], fall["delay"][idx] * rng.uniform(0.6, 1.0, m))).astype(np.float32)
    out["span"] = np.concatenate((fall["span"], fall["span"][idx])).astype(np.float32)
    out["spin"] = np.concatenate((fall["spin"], rng.uniform(0.5, 2.0, m))).astype(np.float32)
    # a crumb is at most ~4 mm whatever the level, and never larger than the piece it left
    s = float(min(0.2, 0.004 / max(float(np.max(cell)), 1e-6)))
    out["scale"] = np.concatenate((fall["scale"], rng.uniform(0.5, 1.0, m) * s)).astype(np.float32)
    return np.vstack((co, co[idx])).astype(np.float32), np.concatenate((kinds, kinds[idx])).astype(np.float32), out

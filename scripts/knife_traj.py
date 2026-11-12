"""Blade trajectories driven by the object's real geometry (no Blender needed).

Every visible blade stroke follows approach -> contact -> penetration -> traversal ->
separation -> retract, sampled per frame. The stroke is computed from the actual
intersection of the cut plane with the shape, so "contact" is the frame where the curved
cutting edge really enters the object and "exit" the frame where it has left it.
Retraction goes back up its own kerf; moves between strokes arc around the object.
"""
import math

import numpy as np

# Knife geometry (metres, before scale). Local axes: X along the blade (tip at -x, heel at
# +x, handle beyond the heel), Y from the edge to the spine, Z the blade normal.
EDGE_X = np.linspace(-0.112, 0.088, 61)
EDGE_CENTRE_X = -0.012
HANDLE_PREF = np.array([0.8, -0.5, 0.2])     # handle towards the camera's right
RESISTANCE = {"cake": 0.25, "jelly": 0.1, "mango": 0.5, "soap": 0.7, "steak": 0.9, "chocolate": 1.0}
CLEARANCE = 0.03                             # gap between ready pose and first contact
EXIT_OVERTRAVEL = 0.006
BOARD_MARGIN = 0.0004


def edge_y(x):
    """Height of the cutting edge above y=0: flat at the heel, curved belly towards the tip."""
    return 0.013 * (np.maximum(0.0, -np.asarray(x, float)) / 0.115) ** 2.2


def _unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def travel_direction(n):
    """Direction of the cut inside the plane: downwards, or front-to-back for flat planes."""
    n = _unit(n)
    ref = np.array([0.0, 1.0, 0.0]) if abs(n[2]) > 0.85 else np.array([0.0, 0.0, -1.0])
    return _unit(ref - n * (ref @ n))


def blade_matrix(n, d):
    """Rotation whose local -Y (the edge) leads along d and whose Z is the plane normal."""
    n, d = _unit(n), _unit(d)
    y, z = -d, n
    x = np.cross(y, z)
    if x @ HANDLE_PREF < 0:
        z, x = -z, -x
    return np.column_stack((x, y, z))


def mat_to_quat(m):
    t = np.trace(m)
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        q = [0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s]
    else:
        i = int(np.argmax(np.diag(m)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = math.sqrt(1.0 + m[i, i] - m[j, j] - m[k, k]) * 2
        q = [0.0] * 4
        q[0] = (m[k, j] - m[j, k]) / s
        q[i + 1] = 0.25 * s
        q[j + 1] = (m[j, i] + m[i, j]) / s
        q[k + 1] = (m[k, i] + m[i, k]) / s
    return _unit(q)


def quat_to_mat(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def slerp(a, b, t):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a @ b < 0:
        b = -b
    d = min(1.0, float(a @ b))
    th = math.acos(d)
    if th < 1e-5:
        return _unit(a + (b - a) * t)
    return _unit((math.sin((1 - t) * th) * a + math.sin(t * th) * b) / math.sin(th))


def vec_slerp(a, b, t):
    """Spherical interpolation of unit directions (no antipodal flip, unlike quaternions)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    th = math.acos(max(-1.0, min(1.0, float(a @ b))))
    if th < 1e-5:
        return _unit(a + (b - a) * t)
    if abs(th - math.pi) < 1e-3:   # exactly opposite: go over the top
        b = _unit(b + np.array([0, 0, 1e-2]))
        th = math.acos(max(-1.0, min(1.0, float(a @ b))))
    return _unit((math.sin((1 - t) * th) * a + math.sin(t * th) * b) / math.sin(th))


def edge_points(pos, m, scale):
    """World positions of the cutting edge samples."""
    x = EDGE_X * scale
    y = edge_y(EDGE_X) * scale
    return pos + np.outer(x, m[:, 0]) + np.outer(y, m[:, 1])


def blade_outline(pos, m, scale):
    """Edge plus spine samples, used for clearance checks."""
    spine = np.where(EDGE_X > -0.02, 0.048, 0.013 + 0.035 * np.clip((EDGE_X + 0.115) / 0.095, 0, 1) ** 0.75)
    return np.vstack((edge_points(pos, m, scale),
                      pos + np.outer(EDGE_X * scale, m[:, 0]) + np.outer(spine * scale, m[:, 1])))


class Pose:
    """Rigid placement of the object: world = R @ (obj - pivot_obj) + centre."""

    def __init__(self, yaw, centre, pivot_z):
        c, s = math.cos(yaw), math.sin(yaw)
        self.R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])
        self.centre = np.asarray(centre, float)
        self.pivot = np.array([0.0, 0.0, pivot_z])

    def to_world(self, p):
        return self.R @ (np.asarray(p, float) - self.pivot) + self.centre

    def to_object(self, w):
        return (np.asarray(w, float) - self.centre) @ self.R + self.pivot

    def direction(self, v):
        return self.R @ np.asarray(v, float)


def inside_world(shape, pose, pts):
    o = pose.to_object(pts)
    return shape.inside(o[..., 0], o[..., 1], o[..., 2])


def section_extent(shape, pose, point, n, d, radius, res=220):
    """(umin, umax) of the object's cross-section along the blade axis u = d x n."""
    u_axis = _unit(np.cross(d, n))
    g = np.linspace(-radius, radius, res)
    U, V = np.meshgrid(g, g, indexing="ij")
    pts = point + U[..., None] * u_axis + V[..., None] * d
    mask = inside_world(shape, pose, pts)
    if not mask.any():
        return None
    us = U[mask]
    return float(us.min()), float(us.max())


def plan_stroke(shape, pose, p_obj, n_obj, radius, scale, floor_z=0.0):
    """Geometry of one blade pass through the plane (object-space point and normal).
    Returns the blade frame, the anchor line and the s-coordinates of contact/exit."""
    n = _unit(pose.direction(n_obj))
    point = pose.to_world(p_obj)
    d = travel_direction(n)
    m = blade_matrix(n, d)
    ext = section_extent(shape, pose, point, n, d, radius)
    if ext is None:
        return None
    u_axis = m[:, 0]
    # section_extent's u may be flipped relative to the blade axis: measure along m[:,0]
    sign = float(_unit(np.cross(d, n)) @ u_axis)
    lo, hi = sorted((ext[0] * sign, ext[1] * sign))
    width = hi - lo
    centre_u = (lo + hi) / 2 - EDGE_CENTRE_X * scale
    anchor = point + u_axis * centre_u
    # march the real curved edge through the shape
    ss = np.arange(-radius * 1.4, radius * 1.4, 0.0005)
    pos = anchor + ss[:, None] * d
    ex = EDGE_X * scale
    ey = edge_y(EDGE_X) * scale
    offs = np.outer(ex, u_axis) + np.outer(ey, m[:, 1])
    pts = pos[:, None, :] + offs[None, :, :]
    hit = inside_world(shape, pose, pts).any(axis=1)
    if not hit.any():
        return None
    idx = np.flatnonzero(hit)
    s_in, s_out = float(ss[idx[0]]), float(ss[idx[-1]]) + 0.0005
    s_end = s_out + EXIT_OVERTRAVEL
    # never let the edge reach the board
    edge_low = lambda s: float((anchor + d * s + offs)[:, 2].min())
    if d[2] < -1e-6:
        limit = s_in + (edge_low(s_in) - (floor_z + BOARD_MARGIN)) / -d[2]
        s_end = min(s_end, limit)
        s_out = min(s_out, s_end)
    free = max(0.0, scale * (EDGE_X[-1] - EDGE_X[0]) - width)
    return {"n": n.tolist(), "d": d.tolist(), "q": mat_to_quat(m).tolist(), "anchor": anchor.tolist(),
            "u_axis": u_axis.tolist(), "s_in": s_in, "s_out": s_out, "s_end": s_end,
            "s_ready": s_in - CLEARANCE, "width": float(width), "draw": float(min(0.024, free)), "covered": bool(free > 0)}


VMAX = 0.075      # metres per frame: fast swing, still believable
ANG_MAX = 0.55    # radians per frame


def retract_frames(st):
    return int(math.ceil(1.6 * (st["s_end"] - st["s_ready"]) / VMAX))


def body_frames(st):
    """Frames a stroke needs: retract under VMAX + 2 of separation + at least 8 to cut."""
    return retract_frames(st) + 2 + 8


def fast_frames(st):
    """Shortest stroke the blade can make at full speed (fast phases of a level)."""
    travel = st["s_end"] - st["s_ready"]
    back = max(2, int(math.ceil(1.5 * travel / VMAX)))
    fwd = max(4, int(math.ceil(1.5 * travel / VMAX)) + 2)
    return fwd + back


def quick_timing(a, b, st):
    """Compressed stroke in [a, b] when there is no room for the full, natural one:
    the blade goes straight through at full speed and snaps back (cut rate > ~2/s)."""
    travel = st["s_end"] - st["s_ready"]
    back = max(2, int(math.ceil(1.5 * travel / VMAX)))
    r = b - 1
    sep_end = r - back
    x = sep_end - 1
    c = a + max(1, round((x - a) * 0.3))
    c = min(c, x - 1)
    pen = c + max(0, (x - c) // 3)
    pen = min(pen, x - 1)
    return {"ready": a, "contact": c, "pen_end": pen, "exit": x,
            "sep_end": sep_end, "retract_end": r, "slot": [a, b], "fast": True}


def transit_frames(p0, q0, p1, q1, centre, safe_r):
    """Frames needed to fly between two ready poses without exceeding the speed limits."""
    centre = np.asarray(centre, float)
    pts = [_transit(p0, q0, p1, q1, centre, safe_r, u)[0] for u in np.linspace(0, 1, 40)]
    length = float(sum(np.linalg.norm(b - a) for a, b in zip(pts, pts[1:])))
    q0, q1 = np.asarray(q0, float), np.asarray(q1, float)
    ang = 2 * math.acos(min(1.0, abs(float(q0 @ q1))))
    return max(0, int(math.ceil(1.6 * max(length / VMAX, ang / ANG_MAX))))


def timing(a, b, resistance, retract):
    """Integer frames of a stroke leaving the ready pose at frame a and ending before b."""
    r = b - 1
    sep_end = r - max(retract, (b - a) // 5)
    x = sep_end - 2
    cutting = x - a
    slow = 0.7 + resistance
    approach = max(3, round(cutting * (0.55 - 0.15 * resistance)))
    c = a + approach
    pen = max(1, round((x - c) * 0.3 * slow))
    return {"ready": a, "contact": c, "pen_end": c + min(pen, x - c - 1), "exit": x,
            "sep_end": sep_end, "retract_end": r, "slot": [a, b]}


def _hermite(t, t0, t1, s0, s1, v0, v1):
    h = t1 - t0
    u = (t - t0) / h
    return ((2 * u**3 - 3 * u**2 + 1) * s0 + (u**3 - 2 * u**2 + u) * h * v0 +
            (-2 * u**3 + 3 * u**2) * s1 + (u**3 - u**2) * h * v1)


def stroke_s(st, frame):
    """Blade progress along d at a frame: natural speed profile, slowed by the material."""
    t = st["time"]
    res = st["resistance"]
    trav_len = st["s_out"] - st["s_in"]
    mean = trav_len / max(1, t["exit"] - t["contact"])
    v_c = mean * (1.0 - 0.3 * res)
    v_p = mean * (0.75 - 0.4 * res)
    # a cut that ends on the board has no exit run: the blade slows into the wood instead
    # of overshooting the board and bouncing back
    v_x = 0.0 if st["s_end"] - st["s_out"] < 1e-6 else mean * 1.15
    depth = st["s_in"] + 0.22 * trav_len
    knots = [(t["ready"], st["s_ready"], 0.0), (t["contact"], st["s_in"], v_c),
             (t["pen_end"], depth, v_p), (t["exit"], st["s_out"], v_x),
             (t["sep_end"], st["s_end"], 0.0), (t["retract_end"], st["s_ready"], 0.0)]
    if frame <= knots[0][0]:
        return st["s_ready"]
    for (t0, s0, v0), (t1, s1, v1) in zip(knots, knots[1:]):
        if t0 < frame <= t1:
            if t1 <= t0:
                return s1
            slope = (s1 - s0) / (t1 - t0)
            if slope > 0:   # keeps the forward motion monotonic
                v0 = min(v0, 3 * slope)
                v1 = min(v1, 3 * slope)
            return float(_hermite(frame, t0, t1, s0, s1, v0, v1))
    return st["s_ready"]


def stroke_pos(st, s):
    prog = (s - st["s_in"]) / max(1e-6, st["s_out"] - st["s_in"])
    prog = min(1.0, max(0.0, prog))
    draw = st["draw"] * (prog * prog * (3 - 2 * prog) - 0.5)
    return np.asarray(st["anchor"]) + np.asarray(st["d"]) * s + np.asarray(st["u_axis"]) * draw


def ready_pose(st):
    return stroke_pos(st, st["s_ready"]), np.asarray(st["q"], float)


def _transit(p0, q0, p1, q1, centre, safe_r, t):
    """Smooth flight between two ready poses, bulging around the object when needed."""
    t = t * t * (3 - 2 * t)
    o0, o1 = p0 - centre, p1 - centre
    r0, r1 = np.linalg.norm(o0), np.linalg.norm(o1)
    ang = math.acos(max(-1.0, min(1.0, float(_unit(o0) @ _unit(o1)))))
    if ang < 0.45:
        pos = p0 + (p1 - p0) * t
    else:
        bulge = max(0.0, safe_r - min(r0, r1)) * math.sin(math.pi * t)
        dir_ = vec_slerp(_unit(o0), _unit(o1), t)
        pos = centre + dir_ * (r0 + (r1 - r0) * t + bulge)
    return pos, slerp(q0, q1, t)


def build_track(strokes, total_frames, centre, safe_r):
    """Per-frame pose [x, y, z, qw, qx, qy, qz] of the working knife, frames 1..total."""
    if not strokes:
        return []
    track = [None] * (total_frames + 1)
    centre = np.asarray(centre, float)
    first_p, first_q = ready_pose(strokes[0])
    prev_p, prev_q, prev_end = first_p, first_q, 1
    for st in strokes:
        t = st["time"]
        p1, q1 = ready_pose(st)
        for f in range(prev_end, t["ready"] + 1):
            u = (f - prev_end) / max(1, t["ready"] - prev_end)
            track[f] = _transit(prev_p, prev_q, p1, q1, centre, safe_r, u)
        q = np.asarray(st["q"], float)
        for f in range(t["ready"], t["retract_end"] + 1):
            track[f] = (stroke_pos(st, stroke_s(st, f)), q)
        prev_p, prev_q, prev_end = p1, q1, t["retract_end"]
    # closing flight back to the first pose: the video loops without a jump
    for f in range(prev_end, total_frames + 1):
        u = (f - prev_end) / max(1, total_frames - prev_end)
        track[f] = _transit(prev_p, prev_q, first_p, first_q, centre, safe_r, u)
    out, last = [], None
    for f in range(1, total_frames + 1):
        p, q = track[f]
        q = np.asarray(q, float)
        if last is not None and q @ last < 0:   # shortest-path quaternion signs
            q = -q
        last = q
        out.append([round(float(x), 6) for x in (*p, *q)])
    return out


# ---- chop format: one real stroke per cut, sub-frame keys (motion blur does the rest) ----
def _key(t, p, q):
    return [float(t), *[float(x) for x in p], *[float(x) for x in q]]


def _stroke_poses(st):
    q = np.asarray(st["q"], float)
    return (stroke_pos(st, st["s_ready"]), stroke_pos(st, st["s_in"]),
            stroke_pos(st, st["s_out"]), stroke_pos(st, st["s_end"]), q)


def chop_keys(blades, centre, safe_r, prev=None):
    """Pose keys [t, x, y, z, qw, qx, qy, qz] (t in frames, may be fractional) for a chain of
    strokes. Each stroke runs from its ready pose through the object at constant speed; between
    two strokes the blade lifts back up its own kerf and steps to the next plane (or, when the
    direction changes, flies around the object)."""
    centre = np.asarray(centre, float)
    keys = []
    for st in blades:
        t = st["time"]
        R, I, O, E, q = _stroke_poses(st)
        if prev is None:
            keys.append(_key(max(1.0, t["ready"] - 1.0), R, q))
        else:
            pt = prev["time"]
            pR, _, _, pE, pq = _stroke_poses(prev)
            gap = max(1e-3, t["ready"] - pt["sep_end"])
            back = pt["sep_end"] + 0.6 * gap
            keys.append(_key(back, pR, pq))               # back up through its own cut
            same = prev.get("level") == st.get("level") and prev["axis"] == st["axis"] \
                and abs(float(np.dot(pq, q))) > 0.999
            if not same:                                   # turn: fly around the object
                for u in np.linspace(0, 1, 10)[1:-1]:
                    p, qq = _transit(pR, pq, R, q, centre, safe_r, float(u))
                    keys.append(_key(back + (t["ready"] - back) * u, p, qq))
            keys.append(_key(t["ready"], R, q))
        keys.append(_key(t["contact"], I, q))
        keys.append(_key(t["exit"], O, q))
        keys.append(_key(t["sep_end"], E, q))
        prev = st
    return keys


def close_loop(keys, first, last, total, centre, safe_r, fps=24):
    """After the last stroke: lift out of its cut, park the knife high out of frame while the
    pieces fall (it never covers the result or the counters), then come back down to the
    opening pose for a seamless loop."""
    R0, _, _, _, q0 = _stroke_poses(first)
    R, _, _, _, q = _stroke_poses(last)
    t0 = last["time"]["sep_end"]
    out = list(keys)
    out.append(_key(t0 + 3, R, q))
    up = np.array([0.0, 0.0, 0.45])
    park_in = t0 + 3 + 0.35 * fps
    park_out = max(park_in + 1, total - 0.9 * fps)
    out.append(_key(park_in, R + up, q))
    out.append(_key(park_out, R0 + up, q0))
    out.append(_key(total, R0, q0))
    out.sort(key=lambda k: k[0])
    clean, last_t = [], -1e9
    for k in out:                          # strictly increasing times, continuous quaternions
        if k[0] <= last_t + 1e-4:
            k = [last_t + 1e-3] + k[1:]
        if clean and np.dot(clean[-1][4:], k[4:]) < 0:
            k = k[:4] + [-x for x in k[4:]]
        clean.append([round(x, 6) for x in k])
        last_t = k[0]
    return clean


def sample_keys(keys, frames):
    """Pose at the given (possibly fractional) frames, linear between keys."""
    ts = np.array([k[0] for k in keys])
    vals = np.array([k[1:] for k in keys])
    out = []
    for f in frames:
        i = int(np.searchsorted(ts, f, side="right"))
        if i <= 0:
            row = vals[0]
        elif i >= len(ts):
            row = vals[-1]
        else:
            u = (f - ts[i - 1]) / max(1e-9, ts[i] - ts[i - 1])
            p = vals[i - 1, :3] * (1 - u) + vals[i, :3] * u
            row = np.concatenate([p, slerp(vals[i - 1, 3:], vals[i, 3:], u)])
        out.append([round(float(x), 6) for x in row])
    return out

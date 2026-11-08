"""Shared definitions for the ASMR slicing studio.

Pure Python + numpy so it can be imported both by the planner (system Python)
and by Blender's bundled Python. Never import bpy here.
"""
import json
import math
import copy

import numpy as np

# --------------------------------------------------------------------------
# Default configuration (short vertical ASMR video)
# --------------------------------------------------------------------------
DEFAULTS = {
    "object_type": "mango",
    "material_profile": None,          # None -> same as object_type
    "target_piece_count": 15,
    "progression_sequence": None,      # None -> auto (2,3,4,5,8,10,15,25,50,...)
    "cutting_mode": "cumulative",      # "cumulative" (Mode A) | "levels" (Mode B)
    "cutting_directions": ["vertical", "horizontal", "diagonal", "cross", "angled", "random3d"],
    "cutting_speed": "auto",           # "auto" | "slow" | "medium" | "fast"
    "camera_style": "follow",          # "follow" | "static"
    "audio_profile": None,             # None -> same as material_profile
    "music": False,
    "video_duration": None,            # seconds; None -> auto from piece count
    "resolution": [1080, 1920],
    "preview_resolution": [540, 960],
    "fps": 30,
    "render_engine": "eevee",          # "eevee" | "cycles"
    "cycles_samples": 64,
    "output_path": "runs",
    "random_seed": 42,
    "watermark": "",                   # e.g. "@mychannel"; empty -> none
    "show_counter": True,
    "hook_text": "",                   # big question shown during the opening seconds
    "end_text": "",                    # call to action shown at the end (e.g. a comment prompt)
    "format": "slices",                # "slices" (progressive halving) | "grid" (levels of grid cuts -> cubes)
    "cuts_sequence": None,             # grid format: cuts per level; None -> auto
    "max_cubes": 20000000,             # grid format: safety cap on the final level
    "display_cubes_max": 1200000,      # grid format: cubes actually instanced per level (render budget)
    "label_cuts": "CUTS",
    "label_pieces": None,              # None -> "PIECES" (slices) / "CUBES" (grid)
    "grid_resolution": None,           # None -> auto from target count
}

GRID_AUTO_CUTS = [4, 10, 15, 25, 40, 60, 100, 160, 250, 400, 600, 1000]
AUTO_SEQUENCE = [2, 3, 4, 5, 8, 10, 15, 25, 50, 100, 200, 400, 1000, 2500, 5000, 10000, 25000, 50000]

DIRECTION_NAMES = ["vertical", "horizontal", "diagonal", "cross", "angled", "random3d"]


def load_config(path_or_dict):
    if isinstance(path_or_dict, dict):
        user = path_or_dict
    else:
        with open(path_or_dict, "r", encoding="utf-8") as f:
            user = json.load(f)
    cfg = copy.deepcopy(DEFAULTS)
    unknown = [k for k in user if k not in DEFAULTS]
    if unknown:
        raise ValueError(f"Unknown config keys: {unknown}")
    cfg.update(user)
    return validate_config(cfg)


def validate_config(cfg):
    n = cfg["target_piece_count"]
    if not isinstance(n, int) or n < 2:
        raise ValueError("target_piece_count must be an integer >= 2")
    if n > 50000:
        raise ValueError("target_piece_count above 50000 is not supported (render time/memory)")
    if cfg["object_type"] not in OBJECTS:
        raise ValueError(f"object_type must be one of {list(OBJECTS)}")
    cfg["material_profile"] = cfg["material_profile"] or cfg["object_type"]
    cfg["audio_profile"] = cfg["audio_profile"] or cfg["material_profile"]
    if cfg["material_profile"] not in MATERIALS:
        raise ValueError(f"material_profile must be one of {list(MATERIALS)}")
    if cfg["cutting_mode"] not in ("cumulative", "levels"):
        raise ValueError("cutting_mode must be 'cumulative' or 'levels'")
    for d in cfg["cutting_directions"]:
        if d not in DIRECTION_NAMES:
            raise ValueError(f"unknown cutting direction '{d}', use {DIRECTION_NAMES}")
    seq = cfg["progression_sequence"]
    if seq is None:
        seq = [k for k in AUTO_SEQUENCE if k < n] + [n]
    seq = [int(k) for k in seq]
    if seq[-1] != n:
        seq.append(n)
    if any(b <= a for a, b in zip(seq, seq[1:])) or seq[0] < 2:
        raise ValueError(f"progression_sequence must be strictly increasing and >= 2: {seq}")
    cfg["progression_sequence"] = seq
    if cfg["format"] not in ("slices", "grid"):
        raise ValueError("format must be 'slices' or 'grid'")
    if cfg["label_pieces"] is None:
        cfg["label_pieces"] = "CUBES" if cfg["format"] == "grid" else "PIECES"
    if cfg["format"] == "grid":
        cs = cfg["cuts_sequence"] or GRID_AUTO_CUTS
        cs = [int(c) for c in cs]
        if any(c < 1 for c in cs):
            raise ValueError("cuts_sequence values must be >= 1")
        cfg["cuts_sequence"] = cs
    if cfg["grid_resolution"] is None:
        # ~60 voxels per final piece on average (inside volume ~ 0.23 * res^3)
        cfg["grid_resolution"] = int(max(72, min(240, round((60 * n / 0.23) ** (1 / 3)))))
    return cfg


# --------------------------------------------------------------------------
# Object shapes: an invertible deformation of the unit ball.
#   forward(u,v,w) -> (x,y,z) builds the mesh (Blender)
#   inside(x,y,z)  -> bool   samples the volume (planner)
# Both use exactly the same maths, so planner centroids match Blender meshes.
# Units are metres. Y = long axis, Z = up (object rests on its flat side).
# --------------------------------------------------------------------------
class Shape:
    def __init__(self, a_pos, a_neg, b, c, bend=0.0, taper=0.0, beak=0.0, z_flat=0.0, z_lift=0.0):
        self.a_pos, self.a_neg, self.b, self.c = a_pos, a_neg, b, c
        self.bend, self.taper, self.beak, self.z_flat = bend, taper, beak, z_flat
        self.z_lift = z_lift  # distance from board to object centre

    # helper profiles, all functions of v in [-1, 1] (long axis)
    def _taper(self, v):
        # wider at the stem shoulder (v>0), narrower towards the tip (v<0)
        return 1.0 + self.taper * v

    def _bend(self, v):
        # kidney curvature + small beak near the tip (v ~ -0.75)
        beak = self.beak * np.exp(-((v + 0.78) / 0.16) ** 2)
        return self.bend * (v ** 2 - 0.33) + beak

    def _zscale(self, v):
        return 1.0 - self.z_flat * v ** 2

    def forward(self, u, v, w):
        u, v, w = np.asarray(u, float), np.asarray(v, float), np.asarray(w, float)
        a = np.where(u >= 0, self.a_pos, self.a_neg)
        x = a * u * self._taper(v) + self._bend(v)
        y = self.b * v
        z = self.c * w * self._zscale(v) * self._taper(v) ** 0.5
        return x, y, z + self.z_lift

    def inside(self, x, y, z):
        x, y, z = np.asarray(x, float), np.asarray(y, float), np.asarray(z, float) - self.z_lift
        v = y / self.b
        t = self._taper(v)
        xr = x - self._bend(v)
        a = np.where(xr >= 0, self.a_pos, self.a_neg)
        u = xr / (a * t)
        w = z / (self.c * self._zscale(v) * t ** 0.5)
        return u * u + v * v + w * w <= 1.0

    def bbox(self):
        lat = np.linspace(-np.pi / 2, np.pi / 2, 181)[:, None]
        lon = np.linspace(0, 2 * np.pi, 361)[None, :]
        x, y, z = self.forward(np.cos(lat) * np.cos(lon), np.sin(lat) + 0 * lon, np.cos(lat) * np.sin(lon))
        lo = np.array([x.min(), y.min(), z.min()])
        hi = np.array([x.max(), y.max(), z.max()])
        pad = 0.01 * (hi - lo).max()
        return lo - pad, hi + pad


# Real mango (Kent/Keitt-like): ~13 cm long, ~9.5 cm wide, ~8 cm thick,
# slightly kidney-shaped, broader shoulder at the stem end, small beak.
MANGO = Shape(a_pos=0.050, a_neg=0.044, b=0.066, c=0.040,
              bend=0.010, taper=0.10, beak=0.004, z_flat=0.18, z_lift=0.0405)
# Soap bar: rounded box approximated by a superellipsoid-ish flat ball
SOAP = Shape(a_pos=0.045, a_neg=0.045, b=0.065, c=0.022, z_lift=0.0225)
JELLY = Shape(a_pos=0.050, a_neg=0.050, b=0.050, c=0.040, z_flat=-0.15, z_lift=0.0405)

class Extruded:
    """Flat object (steak, cake slice...): an irregular outline extruded upwards.
    Bottom rests on the board (z = 0)."""

    kind = "extruded"

    def __init__(self, ax, ay, h, notch=0.10, fat=0.10, seed_phase=0.6, wobble=0.0):
        self.ax, self.ay, self.h = ax, ay, h
        self.notch, self.fat, self.ph, self.wobble = notch, fat, seed_phase, wobble

    def radius(self, th):
        th = np.asarray(th, float)
        e = 1.0 / np.sqrt((np.cos(th) / self.ax) ** 2 + (np.sin(th) / self.ay) ** 2)
        d = np.angle(np.exp(1j * (th - 1.95)))
        mod = (1 + 0.05 * np.cos(2 * th + self.ph) + 0.045 * np.cos(3 * th + 1.3)
               + 0.02 * np.cos(5 * th + 0.4) - self.notch * np.exp(-(d / 0.28) ** 2))
        return e * mod

    def outline(self, n=240):
        th = np.linspace(0, 2 * np.pi, n, endpoint=False)
        r = self.radius(th)
        return np.stack([r * np.cos(th), r * np.sin(th)], 1)

    def rel(self, x, y):
        """0 at the centre, 1 on the outline."""
        x, y = np.asarray(x, float), np.asarray(y, float)
        return np.hypot(x, y) / self.radius(np.arctan2(y, x))

    def top(self, x, y):
        """Organic thickness: gentle swells, thinner rounded edge (real meat is not a slab)."""
        x, y = np.asarray(x, float), np.asarray(y, float)
        u, v = x / self.ax, y / self.ay
        swell = 1 + self.wobble * (0.6 * np.sin(2.3 * u + 1.1) * np.sin(1.9 * v + 0.4)
                                   + 0.4 * np.sin(4.1 * u - 3.3 * v + 0.7))
        r = np.clip(self.rel(x, y), 0, 1)
        edge = 1 - 0.28 * np.clip((r - 0.82) / 0.18, 0, 1) ** 2
        return self.h * swell * edge

    def inside(self, x, y, z):
        x, y, z = np.asarray(x, float), np.asarray(y, float), np.asarray(z, float)
        return (self.rel(x, y) <= 1) & (z >= 0) & (z <= self.top(x, y))

    def is_fat(self, x, y, z):
        """Fat cap along the outline. Marbling inside the muscle is a 3D shader field
        evaluated at each cube's original position, so it stays continuous across cuts."""
        # a cube is drawn as fat only when its centre lies in the outer half of the cap
        return self.rel(x, y) > 1 - 0.55 * self.fat

    def bbox(self):
        o = self.outline(720)
        lo = np.array([o[:, 0].min(), o[:, 1].min(), 0.0])
        hi = np.array([o[:, 0].max(), o[:, 1].max(), self.h * (1 + self.wobble) * 1.01])
        return lo, hi


# ribeye-like: ~21 x 14 cm, 2.5-3.2 cm thick, thin fat cap
STEAK = Extruded(ax=0.105, ay=0.068, h=0.027, fat=0.06, wobble=0.12)
CAKE = Extruded(ax=0.085, ay=0.085, h=0.055, notch=0.0, fat=0.07, seed_phase=0.0)

OBJECTS = {"mango": MANGO, "soap": SOAP, "jelly": JELLY, "chocolate": SOAP,
           "steak": STEAK, "cake": CAKE}


# --------------------------------------------------------------------------
# Material profiles (visual + mechanical + sound). Colours are linear RGB.
# --------------------------------------------------------------------------
MATERIALS = {
    "mango": {
        "label": "MANGO",
        "skin_base": (0.55, 0.42, 0.02), "skin_stem": (0.10, 0.28, 0.02),
        "skin_blush": (0.70, 0.08, 0.02), "skin_roughness": 0.42,
        "flesh": (0.93, 0.42, 0.02), "flesh_deep": (0.85, 0.30, 0.01),
        "flesh_roughness": 0.16, "flesh_subsurface": 0.35, "fibrous": True,
        "rim": (0.80, 0.55, 0.05), "rim_width": 0.0016,
        "separation": 0.10,  # explode factor scale
        "pit_strategy": "seedless_flesh",
    },
    "soap": {
        "label": "SOAP",
        "skin_base": (0.55, 0.25, 0.60), "skin_stem": (0.55, 0.25, 0.60),
        "skin_blush": (0.55, 0.25, 0.60), "skin_roughness": 0.55,
        "flesh": (0.62, 0.30, 0.66), "flesh_deep": (0.58, 0.27, 0.62),
        "flesh_roughness": 0.65, "flesh_subsurface": 0.05, "fibrous": False,
        "rim": (0.60, 0.29, 0.64), "rim_width": 0.0,
        "separation": 0.08, "pit_strategy": "none",
    },
    "jelly": {
        "label": "JELLY",
        "skin_base": (0.80, 0.03, 0.10), "skin_stem": (0.80, 0.03, 0.10),
        "skin_blush": (0.80, 0.03, 0.10), "skin_roughness": 0.05,
        "flesh": (0.85, 0.05, 0.12), "flesh_deep": (0.70, 0.02, 0.08),
        "flesh_roughness": 0.04, "flesh_subsurface": 0.8, "fibrous": False,
        "rim": (0.85, 0.05, 0.12), "rim_width": 0.0,
        "separation": 0.12, "pit_strategy": "none",
    },
    "steak": {
        "label": "STEAK",
        "skin_base": (0.92, 0.86, 0.78), "skin_stem": (0.92, 0.86, 0.78),
        "skin_blush": (0.92, 0.86, 0.78), "skin_roughness": 0.45,
        "flesh": (0.32, 0.028, 0.032), "flesh_deep": (0.19, 0.012, 0.018),
        "flesh_roughness": 0.35, "flesh_subsurface": 0.15, "fibrous": False,
        "rim": (0.92, 0.86, 0.78), "rim_width": 0.0,
        "separation": 0.10, "pit_strategy": "none",
    },
    "cake": {
        "label": "CAKE",
        "skin_base": (0.85, 0.55, 0.65), "skin_stem": (0.85, 0.55, 0.65),
        "skin_blush": (0.85, 0.55, 0.65), "skin_roughness": 0.5,
        "flesh": (0.80, 0.55, 0.25), "flesh_deep": (0.70, 0.45, 0.18),
        "flesh_roughness": 0.7, "flesh_subsurface": 0.05, "fibrous": False,
        "rim": (0.85, 0.55, 0.65), "rim_width": 0.0,
        "separation": 0.10, "pit_strategy": "none",
    },
    "chocolate": {
        "label": "CHOCOLATE",
        "skin_base": (0.08, 0.03, 0.01), "skin_stem": (0.08, 0.03, 0.01),
        "skin_blush": (0.08, 0.03, 0.01), "skin_roughness": 0.30,
        "flesh": (0.10, 0.04, 0.015), "flesh_deep": (0.07, 0.03, 0.01),
        "flesh_roughness": 0.55, "flesh_subsurface": 0.0, "fibrous": False,
        "rim": (0.10, 0.04, 0.015), "rim_width": 0.0,
        "separation": 0.08, "pit_strategy": "none",
    },
}


def direction_vector(name, step_index, rng):
    """Unit normal of a cutting plane for a named direction."""
    ang = rng.uniform(-0.25, 0.25)  # natural hand variation
    if name == "vertical":        # knife comes down, slices across the long axis
        n = (0.0, 1.0, 0.0)
        n = (math.sin(ang * 0.4), math.cos(ang * 0.4), 0.0)
    elif name == "horizontal":    # flat slice parallel to the board
        n = (0.0, 0.0, 1.0)
    elif name == "cross":         # vertical, perpendicular to the previous slices
        n = (math.cos(ang * 0.4), math.sin(ang * 0.4), 0.0)
    elif name == "diagonal":
        a = math.pi / 4 + ang
        n = (math.cos(a), math.sin(a), 0.0)
    elif name == "angled":        # tilted blade
        a = rng.uniform(0, 2 * math.pi)
        tilt = math.radians(rng.uniform(25, 50))
        n = (math.cos(a) * math.cos(tilt), math.sin(a) * math.cos(tilt), math.sin(tilt))
    elif name == "random3d":
        v = rng.normal(size=3)
        n = tuple(v)
    else:
        raise ValueError(name)
    n = np.asarray(n, float)
    return n / np.linalg.norm(n)


def grid_occupancy(shape, lo, ext, n):
    """Cells of an n[0] x n[1] x n[2] grid that contain part of the shape.
    A cell is a real piece as soon as any part of the object lies in it, so each cell
    is tested on a sub-grid (3^3 samples, 2^3 above 3 M cells) rather than only at its
    centre - otherwise coarse levels would under-count (2 cuts on a steak = 4 pieces)."""
    lo, ext, n = np.asarray(lo, float), np.asarray(ext, float), np.asarray(n, int)
    k = 3 if int(np.prod(n)) <= 3_000_000 else 2
    cell = ext / n
    offs = (np.arange(k) + 0.5) / k
    occ = np.zeros(tuple(n), bool)
    ys = lo[1] + (np.arange(n[1])[:, None] + offs[None, :]).ravel() * cell[1]
    zs = lo[2] + (np.arange(n[2])[:, None] + offs[None, :]).ravel() * cell[2]
    Y, Z = np.meshgrid(ys, zs, indexing="ij")
    for i in range(n[0]):
        hit = np.zeros((n[1] * k, n[2] * k), bool)
        for ox in offs:
            x = lo[0] + (i + ox) * cell[0]
            hit |= shape.inside(np.full_like(Y, x), Y, Z)
        occ[i] = hit.reshape(n[1], k, n[2], k).any(axis=(1, 3))
    centres = [lo[a] + (np.arange(n[a]) + 0.5) * cell[a] for a in range(3)]
    return occ, centres

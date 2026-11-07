"""Automated checks. Run:  python -m pytest tests -q   (or: python tests/test_pipeline.py)

Blender/FFmpeg tests run only when BLENDER_PATH (or blender on PATH) is available.
"""
import json
import os
import subprocess
import sys
import tempfile
import wave

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from common import load_config  # noqa: E402
from plan_cuts import plan  # noqa: E402
import make_audio  # noqa: E402

_cache = {}


def P(**kw):
    key = json.dumps(kw, sort_keys=True)
    if key not in _cache:
        _cache[key] = plan(load_config(kw))
    return _cache[key]


def leaves(p, level=None):
    return [x for x in p["pieces"] if x["cut_frame"] is None and x["hide_frame"] is None
            and (level is None or x["level"] == level)]


# ---- progression ---------------------------------------------------------
def test_progression_reaches_target_and_is_progressive():
    for n in (15, 100, 400):
        p = P(target_piece_count=n)
        counts = [s["count_after"] for s in p["strokes"]]
        assert counts == sorted(counts) and len(set(counts)) == len(counts), counts
        assert counts[-1] == n == len(leaves(p))
        assert len(p["strokes"]) >= 5, "a single global cut is not a progression"


def test_milestones_of_auto_sequence_are_visible():
    p = P(target_piece_count=15)
    counts = {s["count_after"] for s in p["strokes"]}
    for m in (2, 3, 4, 5, 8, 10, 15):
        assert m in counts


def test_counter_is_monotonic_and_matches_parts():
    p = P(target_piece_count=50)
    prev = 1
    for h in p["hits"]:
        assert h["parts_created"] >= 2
        assert h["count_after"] == prev + h["parts_created"] - 1
        prev = h["count_after"]


def test_strokes_vs_planes_vs_pieces_are_distinct_metrics():
    s = P(target_piece_count=100)["stats"]
    assert s["blade_strokes"] < s["cut_planes"] <= s["expected_final_pieces"]


# ---- directions ------------------------------------------------------------
def test_multidirectional_cuts():
    p = P(target_piece_count=15)
    names = {s["direction"] for s in p["strokes"]}
    assert len(names) >= 4
    normals = np.array([h["plane_normal"] for h in p["hits"]])
    assert np.allclose(np.linalg.norm(normals, axis=1), 1, atol=1e-6)
    # at least three clearly different plane orientations
    uniq = []
    for n in normals:
        if all(abs(n @ u) < 0.95 for u in uniq):
            uniq.append(n)
    assert len(uniq) >= 3


def test_each_named_direction_alone():
    for d in ["vertical", "horizontal", "diagonal", "cross", "angled", "random3d"]:
        p = P(target_piece_count=8, cutting_directions=[d])
        assert len(leaves(p)) == 8, d


# ---- geometry ----------------------------------------------------------------
def test_volume_conservation_and_partition():
    p = P(target_piece_count=100)
    assert sum(x["voxels"] for x in leaves(p)) == p["grid"]["inside_voxels"]
    for x in p["pieces"]:
        if x["children"]:
            kids = [p["pieces"][c] for c in x["children"]]
            assert sum(k["voxels"] for k in kids) == x["voxels"]


def test_no_piece_vanishes_or_floats():
    p = P(target_piece_count=50)
    for x in p["pieces"]:
        if x["parent"] is not None:
            par = p["pieces"][x["parent"]]
            assert x["born_frame"] == par["cut_frame"], "children appear exactly when parent hides"
        for _, off in x["keys"]:
            assert abs(off[2]) < 1e-9, "pieces only slide on the board, never lift"


def test_mango_shape_is_not_an_ellipsoid():
    from common import MANGO
    lo, hi = MANGO.bbox()
    size = hi - lo
    assert 1.2 < size[1] / size[0] < 1.6 and size[2] < size[0]  # long, flattened
    assert 0.11 < size[1] < 0.16  # a real mango is ~11-15 cm long
    ys = np.linspace(-0.05, 0.05, 11)
    x0 = MANGO.forward(np.zeros(11), ys / MANGO.b, np.zeros(11))[0]
    assert np.ptp(x0) > 0.003  # kidney curvature: centre line is bent


# ---- reproducibility ---------------------------------------------------------
def test_seed_reproducible():
    a = plan(load_config({"target_piece_count": 25, "random_seed": 7}))
    b = plan(load_config({"target_piece_count": 25, "random_seed": 7}))
    c = plan(load_config({"target_piece_count": 25, "random_seed": 8}))
    strip = lambda d: json.dumps({k: v for k, v in d.items() if k != "stats"}, sort_keys=True)
    assert strip(a) == strip(b)
    assert strip(a) != strip(c)


# ---- levels mode -------------------------------------------------------------
def test_levels_mode():
    p = P(target_piece_count=25, cutting_mode="levels", progression_sequence=[2, 5, 10, 25])
    assert [lv["pieces"] for lv in p["levels"]] == [2, 5, 10, 25]
    assert len(p["levels"]) == 4


# ---- audio -------------------------------------------------------------------
def test_audio_sync_and_levels():
    p = P(target_piece_count=15)
    with tempfile.TemporaryDirectory() as d:
        info = make_audio.render(p, d)
        ev = json.load(open(os.path.join(d, "audio_events.json")))["events"]
        with wave.open(os.path.join(d, "soundtrack.wav")) as w:
            assert w.getframerate() == 48000 and w.getnchannels() == 2
            dur = w.getnframes() / 48000
        assert abs(dur - p["total_frames"] / p["fps"]) < 0.05
        assert info["clipped_samples"] == 0 and info["peak_dbfs"] <= -0.9
        contacts = sorted(e["t"] for e in ev if e["type"] == "contact")
        expected = sorted((h["contact_frame"] - 1) / p["fps"] for h in p["hits"])
        assert np.allclose(contacts, expected, atol=1e-6)
        types = {e["type"] for e in ev}
        assert types == {"contact", "penetration", "traversal", "end_of_cut",
                         "separation", "surface_contact"}


def test_audio_variation_not_identical_hits():
    p = P(target_piece_count=8)
    with tempfile.TemporaryDirectory() as d:
        make_audio.render(p, d)
        with wave.open(os.path.join(d, "soundtrack.wav")) as w:
            x = np.frombuffer(w.readframes(w.getnframes()), np.int16)[::2].astype(float)
        segs = []
        for h in p["hits"][:4]:
            i = int((h["contact_frame"] - 1) / p["fps"] * 48000)
            segs.append(x[i:i + 4800])
        c = np.corrcoef(segs)
        assert np.all(c[np.triu_indices(4, 1)] < 0.9)


# ---- Blender (optional) ------------------------------------------------------
def _blender():
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    from run import tools
    return tools()["blender"]


def test_blender_geometry_15():
    b = _blender()
    if not b:
        print("SKIP blender test (no Blender found)")
        return
    with tempfile.TemporaryDirectory() as d:
        pl = os.path.join(d, "plan.json")
        with open(pl, "w") as f:
            json.dump(P(target_piece_count=15, render_engine="cycles"), f)
        r = subprocess.run([b, "-b", "-P", os.path.join(ROOT, "scripts", "blender_build.py"),
                            "--", "--plan", pl, "--out", d, "--stage", "build"],
                           capture_output=True, text=True)
        assert r.returncode == 0, r.stdout[-3000:]
        rep = json.load(open(os.path.join(d, "blender_report.json")))
        assert rep["final_pieces_validated"] == 15
        assert rep["failures"] == [] and rep["non_manifold"] == 0
        assert rep["volume_error_pct"] < 1.0
        assert len(os.listdir(os.path.join(d, "previews"))) >= 4


if __name__ == "__main__":
    fails = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print("PASS", name)
            except AssertionError as e:  # noqa: PERF203
                fails += 1
                print("FAIL", name, e)
    sys.exit(1 if fails else 0)

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
import make_audio  # noqa: E402

def test_only_grid_format_exists():
    cfg = load_config({"object_type": "cake", "format": "grid", "target_piece_count": 2,
                       "cutting_mode": "levels", "cuts_sequence": [2, 8]})
    assert cfg["cuts_sequence"] == [2, 8] and "format" not in cfg
    for old in ("levitate", "slices"):
        try:
            load_config({"format": old})
        except ValueError:
            continue
        raise AssertionError(f"format {old} should be refused")


# ---- grid format ("N CUTS -> X CUBES") ------------------------------------------
def test_grid_levels_counts_are_exact_and_growing():
    import plan_grid
    p = plan_grid.plan(load_config({"object_type": "steak",
                                    "cuts_sequence": [4, 10, 25, 100, 400]}))
    cubes = [l["cubes"] for l in p["levels"]]
    assert cubes == sorted(cubes) and cubes[-1] > 100000
    for l in p["levels"]:
        assert sum(l["cuts_xyz"]) == l["cuts"]
        n = len([s for s in p["strokes"] if s["level"] == l["level"]])
        assert n == l["cuts"], "one blade stroke per announced cut"
    # live piece counter never decreases inside a level and ends on the cube count
    by = {}
    for f, c, li in p["counters"]["pieces"]:
        by.setdefault(li, []).append(c)
    for li, v in by.items():
        assert v == sorted(v) and v[-1] == p["levels"][li]["cubes"]


def test_grid_audio_rhythm_and_levels():
    import plan_grid
    p = plan_grid.plan(load_config({"object_type": "steak", "cuts_sequence": [4, 60]}))
    with tempfile.TemporaryDirectory() as d:
        info = make_audio.render(p, d)
        assert info["clipped_samples"] == 0
        ev = json.load(open(os.path.join(d, "audio_events.json")))["events"]
        ts = sorted(e["t"] for e in ev if e["type"] == "contact")
        assert min(b - a for a, b in zip(ts, ts[1:])) >= 0.09   # <= ~10 chops per second
        assert sum(e["type"] == "surface_contact" for e in ev) == 2


# ---- Blender (optional) ------------------------------------------------------
def _blender():
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    from run import tools
    return tools()["blender"]


def test_blender_grid_build():
    b = _blender()
    if not b:
        print("SKIP blender grid test (no Blender found)")
        return
    import plan_grid
    with tempfile.TemporaryDirectory() as d:
        pl = os.path.join(d, "plan.json")
        p = plan_grid.plan(load_config({"object_type": "steak",
                                        "cuts_sequence": [4, 25], "render_engine": "cycles"}))
        with open(pl, "w") as f:
            json.dump(p, f)
        r = subprocess.run([b, "-b", "-P", os.path.join(ROOT, "scripts", "blender_build.py"),
                            "--", "--plan", pl, "--out", d, "--stage", "build"],
                           capture_output=True, text=True)
        assert r.returncode == 0, r.stdout[-3000:]
        rep = json.load(open(os.path.join(d, "blender_report.json")))
        assert rep["failures"] == []
        assert [l["cubes_built"] for l in rep["levels"]] == [l["cubes"] for l in p["levels"]]


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

"""Blade trajectory checks: geometry, phases, continuity, audio sync. No Blender needed."""
import json
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import knife_traj as KT  # noqa: E402
from common import OBJECTS, load_config  # noqa: E402
from plan_grid import plan  # noqa: E402


def make(**kw):
    base = dict(format="levitate", object_type="cake", cuts_sequence=[1, 8], target_piece_count=2,
                cutting_mode="levels", random_seed=1, video_duration=10, fps=24)
    base.update(kw)
    return plan(load_config(base))


@pytest.fixture(scope="module")
def p():
    return make()


def pose_at(p, f):
    r = p["knife_track"][f - 1]
    return np.array(r[:3]), KT.quat_to_mat(np.array(r[3:]))


def scene_pose(p, level):
    shape = OBJECTS[p["config"]["object_type"]]
    lo, hi = shape.bbox()
    return KT.Pose(p["levels"][level]["yaw"], p["object_centre"], (hi[2] - lo[2]) / 2), shape


def test_three_orientations_and_phase_order(p):
    assert {b["axis"] for b in p["blade_strokes"]} >= {0, 1, 2}
    for b in p["blade_strokes"]:
        t = b["time"]
        assert t["ready"] < t["contact"] <= t["pen_end"] < t["exit"] < t["sep_end"] < t["retract_end"]


def test_no_teleport_no_snap(p):
    tr = np.array(p["knife_track"])
    assert len(tr) == p["total_frames"]
    assert np.linalg.norm(np.diff(tr[:, :3], axis=0), axis=1).max() < 0.08
    dots = np.abs(np.sum(tr[1:, 3:] * tr[:-1, 3:], axis=1)).clip(0, 1)
    assert np.degrees(2 * np.arccos(dots)).max() < 50


def test_no_jerk_spikes_inside_strokes(p):
    for b in p["blade_strokes"]:
        t = b["time"]
        ss = np.array([KT.stroke_s(b, f) for f in range(t["ready"], t["sep_end"] + 1)])
        v = np.diff(ss)
        assert (v > -1e-9).all(), "the blade never backs up while cutting"
        assert np.abs(np.diff(v)).max() < 3 * max(v.max(), 1e-6)


def test_loop_closes(p):
    tr = np.array(p["knife_track"])
    assert np.linalg.norm(tr[0, :3] - tr[-1, :3]) < 1e-3


def test_edge_leads_and_contact_is_real(p):
    for b in p["blade_strokes"]:
        pose, shape = scene_pose(p, b["level"])
        c = b["time"]["contact"]
        pos, m = pose_at(p, c)
        assert np.dot(-m[:, 1], b["d"]) > 0.999, "edge, not spine, travels first"
        pts = KT.edge_points(pos, m, p["knife_scale"])
        assert KT.inside_world(shape, pose, pts + np.array(b["d"]) * 0.001).any(), "edge touches the object"
        before, m0 = pose_at(p, c - 1)
        assert not KT.inside_world(shape, pose, KT.edge_points(before, m0, p["knife_scale"])).any()


def test_blade_sweeps_the_planned_cut_plane(p):
    for b in p["blade_strokes"]:
        pos, m = pose_at(p, b["time"]["contact"])
        n, anchor = np.array(b["n"]), np.array(b["anchor"])
        assert abs((pos - anchor) @ n) < 1e-4
        assert abs(abs(m[:, 2] @ n) - 1) < 1e-4


def test_edge_covers_section_and_exits(p):
    for b in p["blade_strokes"]:
        assert b["covered"], b["width"]
        pose, shape = scene_pose(p, b["level"])
        pos, m = pose_at(p, b["time"]["sep_end"])
        assert not KT.inside_world(shape, pose, KT.edge_points(pos, m, p["knife_scale"])).any()


def test_never_touches_board(p):
    k = p["knife_scale"]
    for row in p["knife_track"]:
        m = KT.quat_to_mat(np.array(row[3:]))
        assert KT.blade_outline(np.array(row[:3]), m, k)[:, 2].min() > 0.0


def test_blade_only_in_object_inside_its_kerf(p):
    k = p["knife_scale"]
    spans = {}
    for b in p["blade_strokes"]:
        for f in range(b["time"]["contact"], b["time"]["retract_end"] + 1):
            spans[f] = b
    hits = 0
    for f in range(1, p["total_frames"] + 1):
        lvl = next((l for l in p["levels"] if l["start_frame"] <= f < l["collapse_start"]), None)
        if lvl is None or f in spans or f < lvl["start_frame"] + lvl["enter_frames"]:
            continue
        pos, m = pose_at(p, f)
        pose, shape = scene_pose(p, lvl["level"])
        hits += int(KT.inside_world(shape, pose, KT.blade_outline(pos, m, k)).any())
    assert hits == 0


def test_any_3d_plane_gets_an_edge_first_blade():
    rng = np.random.default_rng(3)
    shape = OBJECTS["mango"]
    lo, hi = shape.bbox()
    pose = KT.Pose(0.3, (0, 0, 0.2), (hi[2] - lo[2]) / 2 + lo[2])
    for _ in range(40):
        n = KT._unit(rng.normal(size=3))
        st = KT.plan_stroke(shape, pose, (lo + hi) / 2, n, 0.12, 1.0)
        assert st is not None
        m = KT.quat_to_mat(np.array(st["q"]))
        assert np.dot(-m[:, 1], st["d"]) > 0.999 and abs(abs(m[:, 2] @ np.array(st["n"])) - 1) < 1e-6
        assert st["s_out"] > st["s_in"]


def test_cuts_appear_while_their_blade_crosses(p):
    for lvl in p["levels"]:
        blades = [b for b in p["blade_strokes"] if b["level"] == lvl["level"]]
        windows = [(b["time"]["contact"], b["time"]["exit"]) for b in blades]
        for st in (s for s in p["strokes"] if s["level"] == lvl["level"]):
            assert any(a <= st["frame"] <= b for a, b in windows), st


def test_high_levels_stay_readable():
    q = make(cuts_sequence=[500], video_duration=None)
    assert 1 <= len(q["blade_strokes"]) <= 25
    assert sum(b["group_cuts"] for b in q["blade_strokes"]) == 500
    t = [b["time"] for b in q["blade_strokes"]]
    assert all(b["contact"] - a["retract_end"] >= 0 for a, b in zip(t, t[1:]))


def test_audio_contacts_equal_blade_contacts(p, tmp_path):
    import make_audio
    make_audio.render(p, str(tmp_path))
    found = [x for x in os.listdir(tmp_path) if x.endswith(".json")]
    evs = None
    for name in found:
        d = json.load(open(tmp_path / name))
        if isinstance(d, dict) and "events" in d:
            evs = d["events"]
        elif isinstance(d, list) and d and "type" in d[0]:
            evs = d
    assert evs is not None, found
    fps = p["fps"]
    got = sorted(round(e["t"] * fps) for e in evs if e["type"] == "contact")
    want = sorted(b["time"]["contact"] - 1 for b in p["blade_strokes"])
    want2 = sorted(b["time"]["contact"] for b in p["blade_strokes"])
    assert got == want or got == want2


def test_reaction_is_small(p):
    a = np.array(p["object_reaction"])
    assert len(a) == p["total_frames"]
    assert 0 < np.linalg.norm(a, axis=1).max() <= 0.0031


# ---- shared blade system on the slices format --------------------------------------------
@pytest.fixture(scope="module")
def slices():
    import plan_cuts
    return plan_cuts.plan(load_config(dict(object_type="mango", target_piece_count=32, progression_sequence=[2, 4, 8, 16, 32],
                                           cutting_mode="cumulative", random_seed=1, fps=24)))


def test_slices_uses_the_shared_track(slices):
    tr = np.array(slices["knife_track"])
    assert len(tr) == slices["total_frames"] and slices["blade_strokes"]
    assert np.linalg.norm(np.diff(tr[:, :3], axis=0), axis=1).max() < 0.08


def test_slices_blade_crosses_each_passed_cut_plane(slices):
    tr = np.array(slices["knife_track"])
    for st in slices["blade_strokes"]:
        t = st["time"]
        m = KT.quat_to_mat(np.array(st["q"]))
        for f in range(t["contact"], t["exit"] + 1):
            p = tr[f - 1, :3]
            assert abs((p - np.array(st["anchor"])) @ np.array(st["n"])) < 1e-3
        assert t["ready"] < t["contact"] < t["exit"] < t["sep_end"] < t["retract_end"]


def test_slices_every_cut_is_covered_by_a_pass(slices):
    assert slices["sparse_cuts_without_pass"] == 0   # a cut never happens without its blade
    assert sum(s["group_cuts"] for s in slices["blade_strokes"]) == len(slices["hits"])


def test_slices_keeps_real_piece_progression(slices):
    assert [s["count_after"] for s in slices["strokes"]] == sorted(s["count_after"] for s in slices["strokes"])
    assert slices["strokes"][-1]["count_after"] == 32

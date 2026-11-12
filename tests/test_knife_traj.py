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
    base = dict(object_type="cake", cuts_sequence=[1, 8], random_seed=1, video_duration=10, fps=24)
    base.update(kw)
    return plan(load_config(base))


@pytest.fixture(scope="module")
def p():
    return make()


def pose_at(p, t):
    r = KT.sample_keys(p["knife_keys"], [t])[0]
    return np.array(r[:3]), KT.quat_to_mat(np.array(r[3:]))


def scene_pose(p, level):
    shape = OBJECTS[p["config"]["object_type"]]
    lo, hi = shape.bbox()
    return KT.Pose(0.0, p["object_centre"], (hi[2] - lo[2]) / 2), shape


def test_exactly_one_stroke_per_cut_and_phase_order(p):
    for lvl in p["levels"]:
        bl = [b for b in p["blade_strokes"] if b["level"] == lvl["level"]]
        assert len(bl) == lvl["cuts"], "N cuts = N real knife strokes"
        assert all(b["group_cuts"] == 1 for b in bl)
    assert {b["axis"] for b in p["blade_strokes"]} >= {0, 1, 2}
    for b in p["blade_strokes"]:
        t = b["time"]
        assert t["ready"] < t["contact"] < t["exit"] <= t["sep_end"]
    c = [b["time"]["contact"] for b in p["blade_strokes"]]
    assert all(b > a for a, b in zip(c, c[1:])), "strokes never overlap"


def test_counter_climbs_by_one_at_each_contact(p):
    for lvl in p["levels"]:
        bl = [b for b in p["blade_strokes"] if b["level"] == lvl["level"]]
        ev = [(f, n) for f, n, li in p["counters"]["cuts"] if li == lvl["level"]]
        assert [n for _, n in ev] == list(range(1, lvl["cuts"] + 1))
        for (f, _), b in zip(ev, bl):
            assert b["time"]["contact"] <= f < b["time"]["contact"] + 1


def test_keys_are_continuous(p):
    k = np.array(p["knife_keys"])
    assert (np.diff(k[:, 0]) > 0).all()
    step = np.linalg.norm(np.diff(k[:, 1:4], axis=0), axis=1)
    dt = np.diff(k[:, 0])
    assert ((step < 0.35) | (step / dt < 0.06)).all(), "no jump: long moves are slow ones"
    assert (np.sum(k[1:, 4:] * k[:-1, 4:], axis=1) > 0).all(), "shortest-path quaternions"


def test_loop_closes(p):
    k = p["knife_keys"]
    assert k[-1][0] <= p["total_frames"] + 1e-6
    first = KT.ready_pose(p["blade_strokes"][0])[0]
    assert np.linalg.norm(np.array(k[-1][1:4]) - first) < 1e-3


def test_edge_leads_and_contact_is_real(p):
    for b in p["blade_strokes"]:
        pose, shape = scene_pose(p, b["level"])
        c = b["time"]["contact"]
        pos, m = pose_at(p, c)
        assert np.dot(-m[:, 1], b["d"]) > 0.999, "edge, not spine, travels first"
        pts = KT.edge_points(pos, m, p["knife_scale"])
        assert KT.inside_world(shape, pose, pts + np.array(b["d"]) * 0.001).any(), "edge touches the object"
        before, m0 = pose_at(p, b["time"]["ready"])
        assert not KT.inside_world(shape, pose, KT.edge_points(before, m0, p["knife_scale"])).any()


def test_blade_sweeps_the_planned_cut_plane(p):
    for b in p["blade_strokes"]:
        n, anchor = np.array(b["n"]), np.array(b["anchor"])
        for t in (b["time"]["contact"], (b["time"]["contact"] + b["time"]["exit"]) / 2, b["time"]["exit"]):
            pos, m = pose_at(p, t)
            assert abs((pos - anchor) @ n) < 1e-4
            assert abs(abs(m[:, 2] @ n) - 1) < 1e-4


def test_edge_covers_section_and_exits(p):
    for b in p["blade_strokes"]:
        assert b["covered"], b["width"]
        pose, shape = scene_pose(p, b["level"])
        pos, m = pose_at(p, b["time"]["sep_end"])
        edge = KT.edge_points(pos, m, p["knife_scale"])
        if b["s_end"] - b["s_out"] < 1e-6:
            # downward cut on a board-resting object: the edge must reach the wood
            assert edge[:, 2].min() < 0.0015, "the blade cuts all the way down to the board"
        else:
            assert not KT.inside_world(shape, pose, edge).any()


def test_never_touches_board(p):
    k = p["knife_scale"]
    ts = np.arange(1, p["total_frames"], 0.25)
    for row in KT.sample_keys(p["knife_keys"], ts):
        m = KT.quat_to_mat(np.array(row[3:]))
        assert KT.blade_outline(np.array(row[:3]), m, k)[:, 2].min() > 0.0


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


def test_rhythm_starts_calm_then_accelerates_to_a_blur():
    q = make(object_type="mango", cuts_sequence=[400], video_duration=None, fps=24)
    bl = q["blade_strokes"]
    assert len(bl) == 400
    c = np.array([b["time"]["contact"] for b in bl])
    gaps = np.diff(c)
    assert abs(gaps[0] - 6.0) < 1e-6 and abs(gaps[2] - 6.0) < 1e-6, "guide rhythm: 1 stroke / 6 frames"
    assert gaps[-1] < 0.5, "several strokes per frame at the end (motion blur streak)"
    same_axis = [g for g, a, b in zip(gaps, bl, bl[1:]) if a["axis"] == b["axis"]]
    assert all(b <= a + 1e-6 for a, b in zip(same_axis, same_axis[1:])), "never slows down"
    fwd = [b["time"]["sep_end"] - b["time"]["ready"] for b in bl]
    assert fwd[-1] < fwd[0] / 5, "the stroke itself gets faster"


def test_audio_contacts_equal_blade_contacts(p, tmp_path):
    import make_audio
    make_audio.render(p, str(tmp_path))
    d = json.load(open(tmp_path / "audio_events.json"))
    fps = p["fps"]
    got = sorted(round(e["t"] * fps, 3) for e in d["events"] if e["type"] == "contact")
    want = sorted(round(b["time"]["contact"] - 1, 3) for b in p["blade_strokes"])
    assert got == want

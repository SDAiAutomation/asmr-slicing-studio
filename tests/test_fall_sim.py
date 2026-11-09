import os
import sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import fall_sim as F


def grid(n=14):
    a = (np.arange(n) + 0.5) / n * 0.1
    x, y, z = np.meshgrid(a, a, a, indexing="ij")
    return np.column_stack((x.ravel(), y.ravel(), z.ravel())), np.full(3, 0.1 / n)


def test_lands_on_the_board_never_through_it():
    co, c = grid()
    f = F.simulate(co, c, -0.05, "cake", 1)
    assert f["target"][:, 2].min() >= -0.05


def columns(f, c, co):
    size = max(c[0], c[1]) * 1.02
    centre = co[:, :2].mean(0)
    out = {}
    for i, k in enumerate(map(tuple, np.rint((np.asarray(f["target"][:, :2], float) - centre) / size).astype(int))):
        out.setdefault(k, []).append(i)
    return out


def test_stacked_pieces_do_not_overlap_and_barely_rise():
    co, c = grid()
    f = F.simulate(co, c, 0.0, "steak", 2)
    gaps = np.concatenate([np.diff(np.sort(f["target"][v, 2])) for v in columns(f, c, co).values() if len(v) > 1])
    assert (gaps >= c[2] * F.PACK * 0.99).mean() > 0.99     # float32 column-boundary rounding aside
    rise = f["target"][:, 2] - co[:, 2]
    assert (rise > c[2]).mean() < 0.15


def test_trajectories_differ_and_fit_the_window():
    co, c = grid()
    f = F.simulate(co, c, 0.0, "mango", 3)
    assert np.std(f["delay"]) > 0.02 and np.std(f["span"]) > 0.01
    assert (f["delay"] >= 0).all() and ((f["delay"] + f["span"]) <= 1.0 + 1e-5).all()


def test_support_lands_first():
    co, c = grid()
    f = F.simulate(co, c, 0.0, "cake", 4)
    land = f["delay"] + f["span"] * F.LAND_U
    pairs = bad = 0
    for v in columns(f, c, co).values():
        v = sorted(v, key=lambda i: f["target"][i, 2])
        for a, b in zip(v, v[1:]):
            pairs += 1
            bad += land[b] < land[a] - 1e-4
    assert pairs > 100 and bad / pairs < 0.005              # float32 column-boundary rounding aside


def test_materials_behave_differently():
    assert F.params("steak")["bounce"] < F.params("chocolate")["bounce"]
    assert F.params("steak")["spin"] < F.params("soap")["spin"]
    assert F.params("cake")["crumbs"] and not F.params("steak")["crumbs"]


def test_crumbs_are_small_and_on_the_board():
    co, c = grid()
    k = np.zeros(len(co), np.float32)
    f = F.simulate(co, c, 0.0, "cake", 5)
    co2, k2, f2 = F.add_crumbs(co, k, f, c, 0.0, 5)
    assert len(co2) > len(co) and len(k2) == len(co2) == len(f2["scale"])
    assert (f2["scale"][len(co):] <= 0.2 + 1e-6).all()


def test_pile_stays_inside_the_board_limit():
    a = (np.arange(8) + 0.5) / 8 * 0.24 - 0.12
    x, y, z = np.meshgrid(a, a, [0.02, 0.06], indexing="ij")
    co = np.column_stack((x.ravel(), y.ravel(), z.ravel()))
    cell = np.array([0.03, 0.03, 0.04])
    f = F.simulate(co, cell, 0.0, "cake", 7, max_foot=(0.17, 0.115))
    assert np.abs(f["target"][:, 0]).max() < 0.17 + 0.01
    assert np.abs(f["target"][:, 1]).max() < 0.115 + 0.01


def test_progressive_release_lands_in_release_order():
    co, c = grid()
    rel = (co[:, 0] - co[:, 0].min()) / np.ptp(co[:, 0]) * 0.7
    f = F.simulate(co, c, -0.05, "cake", 9, release=rel, fall_frac=0.3)
    assert (f["delay"] >= rel - 1e-6).all() and ((f["delay"] + f["span"]) <= 1.0 + 1e-5).all()
    land = f["delay"] + f["span"] * F.LAND_U
    assert np.corrcoef(rel, land)[0, 1] > 0.8

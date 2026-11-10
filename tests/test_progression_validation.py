import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from validate_progression import validate_plan


class PlanQATests(unittest.TestCase):
    def test_grid_valid(self):
        p = {"kind": "grid", "levels": [{"level": 0, "cuts": 2, "visible_blade_passes": 1, "grouped_cut_count": 2}],
             "strokes": [{"level": 0}, {"level": 0}], "blade_strokes": [{"level": 0}],
             "stats": {"blade_strokes": 1, "planned_cuts": 2}}
        self.assertEqual(validate_plan(p), [])

    def test_grid_misreported_passes(self):
        p = {"kind": "grid", "levels": [{"level": 0, "cuts": 2, "visible_blade_passes": 2}],
             "strokes": [{"level": 0}, {"level": 0}], "blade_strokes": [{"level": 0}],
             "stats": {"blade_strokes": 2, "planned_cuts": 2}}
        self.assertTrue(validate_plan(p))

    def test_slices_valid(self):
        p = {"hits": [{}, {}, {}], "strokes": [
            {"hits": [0, 1], "cut_planes": 2, "grouped_hits": 1},
            {"hits": [2], "cut_planes": 1, "grouped_hits": 0}],
            "stats": {"cut_planes": 3, "grouped_cut_planes": 1}}
        self.assertEqual(validate_plan(p), [])

    def test_slices_duplicate_hit(self):
        p = {"hits": [{}, {}], "strokes": [{"hits": [0, 0], "cut_planes": 2, "grouped_hits": 1}],
             "stats": {"cut_planes": 2, "grouped_cut_planes": 1}}
        self.assertTrue(validate_plan(p))

    def test_slices_overreported_blade_coverage(self):
        p = {"hits": [{}], "strokes": [{"hits": [0], "cut_planes": 1, "grouped_hits": 0}],
             "blade_strokes": [{"group_cuts": 2}], "stats": {"cut_planes": 1, "grouped_cut_planes": 0}}
        self.assertTrue(validate_plan(p))


if __name__ == "__main__":
    unittest.main()

from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from common import load_config
from plan_grid import plan


def test_papaya_guide_timeline():
    cfg=load_config({"format":"papaya_slices"})
    p=plan(load_config(cfg))
    assert [s["frame"] for s in p["strokes"]] == list(range(29,324,6))
    assert len(p["blade_strokes"]) == 50
    assert p["levels"][0]["cubes"] == 100
    assert p["levels"][0]["collapse_start"] == 343
    assert p["total_frames"] == 420
    assert p["duration_s"] == 17.5

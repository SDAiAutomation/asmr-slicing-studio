import os
import sys
import numpy as np
import wave
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from common import load_config
import plan_grid
import make_audio as A


def plan(preset):
    return plan_grid.plan(load_config({"object_type": "cake", "format": "grid", "cuts_sequence": [3, 12, 40],
                                       "random_seed": 3, "fps": 24, "audio_preset": preset}))


def audio(tmp, preset):
    info = A.render(plan(preset), str(tmp))
    with wave.open(info["path"]) as w:
        x = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(float) / 32768
    return info, x.reshape(-1, 2)


def test_default_stays_without_music():
    assert load_config({"object_type": "cake"})["audio_preset"] == "asmr_pure"


def test_pure_has_no_notes_and_presets_differ(tmp_path):
    p, xp = audio(tmp_path / "a", "asmr_pure")
    m, xm = audio(tmp_path / "b", "musical_asmr")
    assert p["buses"]["musical_notes"]["peak"] == 0 and p["buses"]["reward_fx"]["peak"] == 0
    assert m["buses"]["musical_notes"]["peak"] > 0
    assert np.abs(xp - xm).max() > 0.01


def test_no_clipping_and_loudness_on_target(tmp_path):
    for preset in A.AUDIO_PRESETS:
        i, _ = audio(tmp_path / preset, preset)
        assert i["clipped_samples"] == 0
        assert i["peak_dbfs"] <= -0.9
        assert abs(i["loudness_lufs"] - A.LOUDNESS_TARGET) < 2.5


def test_cutting_sounds_lead_musical_preset():
    g = A.AUDIO_PRESETS["musical_asmr"]
    assert g["cutting_asmr"] > 2 * g["musical_notes"]

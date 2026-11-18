"""Spec JSON -> vidéo MP4 + fiche de publication."""
import json
import shutil
from pathlib import Path

from . import config
from .checks import check_spec
from .music import make_music
from .render import render_video
from .spec import load_spec
from .tts import voice_scenes


def build(spec_path, use_tts=True, force=False):
    spec = load_spec(spec_path)
    outdir = config.OUT_DIR / spec["id"]
    work = outdir / "_work"
    work.mkdir(parents=True, exist_ok=True)

    voices, provider = voice_scenes(spec, work, use_tts)
    tts_on = provider != "none"
    # durée mini : on prolonge la scène finale (CTA) de quelques secondes si besoin
    missing = config.MIN_VIDEO_SEC - sum(v["duration"] for v in voices)
    if 0 < missing <= config.MAX_TAIL_PAD_SEC:
        voices[-1]["duration"] += missing
    report = check_spec(spec, [v["duration"] for v in voices])
    report["voice"] = provider if tts_on else "aucune (musique seule)"
    (outdir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if not report["ok"] and not force:
        return None, report

    total = sum(v["duration"] for v in voices)
    music = make_music(total + 0.5, work / "music.wav", seed=abs(hash(spec["id"])) % 1000)
    video = outdir / f"{spec['id']}.mp4"
    render_video(spec, voices, work, music, video, tts_on)

    tags = " ".join("#" + h.lstrip("#") for h in spec["hashtags"])
    meta = {
        "titre": spec.get("title", ""),
        "description": (spec.get("description", "") + "\n\n" + tags).strip(),
        "sources": spec.get("sources", ""),
        "duree_s": round(total, 1),
        "fichier": video.name,
    }
    (outdir / "publication.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    shutil.rmtree(work, ignore_errors=True)
    return video, report

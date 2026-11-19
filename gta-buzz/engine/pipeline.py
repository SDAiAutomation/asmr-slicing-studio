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

    meta = publication_meta(spec, total, video.name)
    (outdir / "publication.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    (outdir / f"{spec['id']}.txt").write_text(publication_txt(meta, spec["lang"]), encoding="utf-8")
    shutil.rmtree(work, ignore_errors=True)
    return video, report


def default_pinned(spec):
    """Commentaire épinglé par défaut : la question finale + relance."""
    cta = next((sc for sc in reversed(spec["scenes"]) if sc["type"] == "cta"), None)
    q = (cta.get("small") or cta.get("big")) if cta else ""
    if spec["lang"] == "fr":
        return f"{q} Réponds en commentaire 👇 D'autres vidéos GTA 6 arrivent, abonne-toi pour ne rien rater."
    return f"{q} Drop your answer below 👇 More GTA 6 videos are coming, follow so you don't miss them."


def publication_meta(spec, total, filename):
    tags = " ".join("#" + h.lstrip("#") for h in spec["hashtags"])
    return {
        "titre": spec.get("title", ""),
        "description": spec.get("description", ""),
        "hashtags": tags,
        "commentaire_epingle": spec.get("pinned_comment") or default_pinned(spec),
        "sources": spec.get("sources", ""),
        "duree_s": round(total, 1),
        "fichier": filename,
    }


def publication_txt(meta, lang):
    """Fichier texte prêt à copier-coller sur YouTube / TikTok / Instagram."""
    fr = lang == "fr"
    src = ("Sources : " if fr else "Sources: ") + meta["sources"] if meta["sources"] else ""
    desc = "\n\n".join(x for x in (meta["description"], src, meta["hashtags"]) if x)
    return (
        f"=== {'TITRE' if fr else 'TITLE'} ===\n{meta['titre']}\n\n"
        f"=== DESCRIPTION ===\n{desc}\n\n"
        f"=== HASHTAGS ===\n{meta['hashtags']}\n\n"
        f"=== {'COMMENTAIRE ÉPINGLÉ' if fr else 'PINNED COMMENT'} ===\n{meta['commentaire_epingle']}\n\n"
        f"=== INFOS ===\n{meta['fichier']} · {meta['duree_s']} s\n"
    )

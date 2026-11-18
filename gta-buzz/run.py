"""GTA Buzz Engine — Shorts sur l'actu GTA 6, sans images du jeu.

Exemples :
  python run.py veille                         # sujets du jour -> out/topics_DATE.json
  python run.py script "GTA 6 bat le record de Twitch" --facts "6,5 millions de spectateurs" --lang fr
  python run.py render specs/economie_fr.json  # une vidéo
  python run.py render specs/                  # toutes les specs du dossier
  python run.py check specs/economie_fr.json   # garde-fous seulement
"""
import argparse
import json
import sys
from pathlib import Path

from engine import config
from engine.checks import check_spec
from engine.spec import load_spec
from engine.tts import voice_scenes


def cmd_veille(_):
    from engine.veille import run_veille
    path, topics, errors = run_veille()
    print(f"{len(topics)} sujets -> {path}")
    for t in topics[:15]:
        print(f"  [{t['source']}] {t['title']}")
    for e in errors:
        print("  ! source en échec :", e)


def cmd_script(a):
    from engine.script import generate_spec
    for lang in a.lang.split(","):
        p = generate_spec(a.topic, a.facts, lang.strip(), Path("specs/auto"), a.sources)
        print("Brouillon :", p, "-> relis les chiffres puis : python run.py render", p)


def _specs(target):
    p = Path(target)
    return sorted(p.glob("*.json")) if p.is_dir() else [p]


def cmd_check(a):
    for s in _specs(a.target):
        spec = load_spec(s)
        voices, _ = voice_scenes(spec, Path("."), use_tts=False)
        print(s.name, json.dumps(check_spec(spec, [v["duration"] for v in voices]), ensure_ascii=False, indent=2))


def cmd_render(a):
    from engine.pipeline import build
    failed = 0
    for s in _specs(a.target):
        print(f"▶ {s.name}")
        video, report = build(s, use_tts=not a.no_voice, force=a.force)
        for w in report["warnings"]:
            print("   ⚠", w)
        if video is None:
            failed += 1
            for e in report["errors"]:
                print("   ✖", e)
            print("   bloqué par les garde-fous (--force pour ignorer)")
        else:
            print(f"   ✔ {video} ({report['duration']} s, voix : {report['voice']})")
    sys.exit(1 if failed else 0)


def main():
    ap = argparse.ArgumentParser(description="GTA Buzz Engine")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("veille").set_defaults(fn=cmd_veille)
    sp = sub.add_parser("script")
    sp.add_argument("topic")
    sp.add_argument("--facts", default="")
    sp.add_argument("--sources", default="")
    sp.add_argument("--lang", default="fr,en")
    sp.set_defaults(fn=cmd_script)
    rp = sub.add_parser("render")
    rp.add_argument("target")
    rp.add_argument("--no-voice", action="store_true", help="ne pas appeler ElevenLabs")
    rp.add_argument("--force", action="store_true", help="rendre même si les garde-fous bloquent")
    rp.set_defaults(fn=cmd_render)
    cp = sub.add_parser("check")
    cp.add_argument("target")
    cp.set_defaults(fn=cmd_check)
    a = ap.parse_args()
    config.OUT_DIR.mkdir(exist_ok=True)
    a.fn(a)


if __name__ == "__main__":
    main()

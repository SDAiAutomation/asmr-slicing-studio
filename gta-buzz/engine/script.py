"""Génère une spec vidéo (JSON) à partir d'un sujet, via l'API Claude.

La spec produite est un BROUILLON : relis les chiffres avant de rendre.
"""
import json
import re
import urllib.request
from pathlib import Path

from . import config

SYSTEM = """Tu écris des scripts de Shorts de 65 à 75 secondes (170 à 200 mots au total, jamais moins de 60 s) sur l'actualité de GTA 6.
Règles absolues :
- Aucun spoiler d'histoire, aucune fuite, aucune fin de mission, aucun piratage.
- Aucune violence dans l'accroche (7 premières secondes) : accroche par un chiffre, un mystère ou une question.
- N'utilise QUE les chiffres présents dans les faits fournis. Si aucun chiffre n'est fourni, n'invente rien : utilise des scènes 'text'.
- Phrases courtes, langage parlé, une idée par scène, 9 à 12 scènes, dernière scène de type 'cta' qui pose une question.
- Le titre contient « GTA 6 » et fait moins de 80 caractères.
Réponds UNIQUEMENT avec un objet JSON valide au format :
{"id": str, "lang": "fr"|"en", "theme": "vice"|"money", "tag": str, "title": str, "description": str,
 "hashtags": [str], "sources": str, "pinned_comment": str,
 "scenes": [
   {"type": "hook", "big": str, "small": str, "voice": str},
   {"type": "stat", "value": number, "decimals": int, "prefix": str, "suffix": str, "label": str, "voice": str},
   {"type": "text", "big": str, "small": str, "voice": str},
   {"type": "versus", "left": {"top": str, "big": str}, "right": {"top": str, "big": str}, "small": str, "voice": str},
   {"type": "cta", "big": str, "small": str, "voice": str}
 ]}
'pinned_comment' = commentaire à épingler : une question qui fait répondre + une relance pour s'abonner (2 phrases max).\n'voice' = le texte lu par la voix off (il sert aussi de sous-titres). 'big' = 2 à 6 mots max."""


def _call_claude(prompt: str) -> str:
    if not config.ANTHROPIC_API_KEY:
        raise RuntimeError("ANTHROPIC_API_KEY manquante : écris la spec à la main (voir specs/) ou ajoute la clé")
    body = json.dumps({
        "model": config.ANTHROPIC_MODEL,
        "max_tokens": 2000,
        "system": SYSTEM,
        "messages": [{"role": "user", "content": prompt}],
    }).encode()
    req = urllib.request.Request("https://api.anthropic.com/v1/messages", data=body, headers={
        "x-api-key": config.ANTHROPIC_API_KEY,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    })
    with urllib.request.urlopen(req, timeout=120) as r:
        data = json.loads(r.read())
    return "".join(b.get("text", "") for b in data["content"])


def generate_spec(topic: str, facts: str, lang: str, out_dir: Path, sources: str = "") -> Path:
    prompt = (f"Langue : {lang}\nSujet : {topic}\n"
              f"Faits vérifiés (seuls chiffres autorisés) :\n{facts or '(aucun)'}\n"
              f"Sources à afficher : {sources or '(à compléter)'}")
    raw = _call_claude(prompt)
    m = re.search(r"\{.*\}", raw, re.S)
    if not m:
        raise ValueError("Réponse sans JSON :\n" + raw[:500])
    spec = json.loads(m.group(0))
    spec["lang"] = lang
    spec["status"] = "draft"
    if sources and not spec.get("sources"):
        spec["sources"] = sources
    slug = re.sub(r"[^a-z0-9]+", "-", spec.get("id") or topic.lower())[:40].strip("-")
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{slug}_{lang}.json"
    path.write_text(json.dumps(spec, ensure_ascii=False, indent=2), encoding="utf-8")
    return path

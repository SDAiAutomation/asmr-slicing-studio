"""Lecture et validation d'un fichier spec (une vidéo = un JSON)."""
import json
from pathlib import Path

SCENE_TYPES = {"hook", "stat", "text", "versus", "cta"}


def load_spec(path) -> dict:
    spec = json.loads(Path(path).read_text(encoding="utf-8"))
    spec.setdefault("lang", "fr")
    spec.setdefault("theme", "vice")
    spec.setdefault("tag", "GTA 6 • ACTU")
    spec.setdefault("hashtags", [])
    spec.setdefault("sources", "")
    if not spec.get("id"):
        spec["id"] = Path(path).stem
    for i, sc in enumerate(spec.get("scenes", [])):
        if sc.get("type") not in SCENE_TYPES:
            raise ValueError(f"Scène {i} : type inconnu {sc.get('type')!r} (attendu : {sorted(SCENE_TYPES)})")
        if not sc.get("voice"):
            raise ValueError(f"Scène {i} : champ 'voice' manquant (texte dit + sous-titres)")
    if not spec.get("scenes"):
        raise ValueError("Aucune scène dans la spec")
    return spec


def fmt_number(value: float, decimals: int, lang: str) -> str:
    """1234567.5 -> '1 234 567,5' (fr) ou '1,234,567.5' (en)."""
    s = f"{value:,.{decimals}f}"
    if lang == "fr":
        s = s.replace(",", " ").replace(".", ",")
    return s

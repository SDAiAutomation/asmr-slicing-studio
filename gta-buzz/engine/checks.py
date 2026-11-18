"""Garde-fous avant publication : copyright, spoilers, monétisation YouTube."""
import re

# Mots à proscrire partout (spoilers, fuites, piratage)
BANNED = [
    "leak", "leaked", "fuite", "fuité", "spoiler", "ending", "fin du jeu", "dernière mission",
    "final mission", "crack", "téléchargement gratuit", "free download", "early access gratuit",
]
# Violence à éviter dans l'accroche (7 premières secondes : zone sans pub YouTube)
VIOLENT = [
    "tue", "tuer", "tué", "meurtre", "massacre", "fusillade", "sang", "arme", "flingue", "braquage sanglant",
    "kill", "killing", "murder", "massacre", "shooting", "blood", "gun", "torture",
]
TITLE_KEYS = ["gta 6", "gta vi", "gta6"]


def _words(text):
    return re.findall(r"[\wÀ-ÿ']+", text.lower())


def check_spec(spec, durations):
    errors, warnings = [], []
    total = sum(durations)
    if total < 12:
        errors.append(f"Vidéo trop courte ({total:.1f} s < 12 s)")
    if total > 60:
        warnings.append(f"Vidéo longue ({total:.1f} s) : TikTok/Reels préfèrent moins de 60 s")

    title = spec.get("title", "")
    if not any(k in title.lower() for k in TITLE_KEYS):
        errors.append("Le titre doit contenir « GTA 6 » (référencement)")
    if len(title) > 100:
        errors.append(f"Titre trop long ({len(title)} caractères > 100)")

    all_text = " ".join([title, spec.get("description", "")] +
                        [" ".join(str(v) for v in sc.values() if isinstance(v, str)) for sc in spec["scenes"]]).lower()
    for b in BANNED:
        if b in all_text:
            errors.append(f"Terme interdit détecté : « {b} » (spoiler / fuite / piratage)")

    # accroche : texte affiché ou dit pendant les 7 premières secondes
    t, hook_text = 0.0, []
    for sc, d in zip(spec["scenes"], durations):
        if t >= 7:
            break
        hook_text.append(" ".join(str(v) for v in sc.values() if isinstance(v, str)))
        t += d
    hw = set(_words(" ".join(hook_text)))
    for v in VIOLENT:
        if v in hw:
            errors.append(f"Mot violent « {v} » dans les 7 premières secondes (perte de pub YouTube)")

    if any(sc["type"] == "stat" for sc in spec["scenes"]) and not spec.get("sources"):
        errors.append("Chiffres affichés sans sources : renseigne le champ 'sources'")

    if spec["scenes"][0]["type"] not in ("hook", "stat", "versus"):
        warnings.append("La 1re scène n'est pas une accroche (hook/stat/versus)")
    if spec["scenes"][-1]["type"] != "cta":
        warnings.append("Pas d'appel à l'action en fin de vidéo (scène 'cta')")
    return {"ok": not errors, "errors": errors, "warnings": warnings, "duration": round(total, 2)}

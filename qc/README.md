# Format « Contrôle Qualité »

Un objet passe sur un banc d'essai 3D (presse ou traction). La charge monte par paliers, la vidéo
se fige sur une question (« Ça tient à 100 t ? »), puis ça casse à la **vraie valeur** et un tampon
tombe (CONFORME / NON CONFORME…). Vidéo de ~60 s : 2 manches + explication + source à l'écran.

## Règle n° 1 : tout chiffre est vrai
- Chaque config a un champ `source` (affiché en fin de vidéo) et `sources_detail` (calcul complet).
  Le workflow refuse une config sans `source`.
- Valeur de casse = valeur publiée (norme, étude). Si on simule, on l'écrit (« Essai simulé aux valeurs de la norme »).
- Pas de mesure fiable pour un objet → on ne le teste pas.

## Fichiers
| Fichier | Rôle |
|---|---|
| `qc/engine.html` | Scène 3D raymarchée (WebGL) + textes, pilotée par la config. Toutes les durées sont dans `D` (timeline). |
| `qc/render.js` | Rendu image par image dans Chromium headless : `timeline`, `stills`, `frames a-b`, `video`. |
| `qc/audio.py` | Son procédural synchronisé sur les événements de la timeline. |
| `assets/configs/qc/*.json` | Un épisode par fichier. |
| `.github/workflows/qc-render.yml` | Rendu parallélisé sur GitHub Actions → Google Drive (`Controle_Qualite/`). |

## Config
```jsonc
{
  "id": "qc-001-beton", "series": "CONTRÔLE QUALITÉ · TEST 001",
  "hook": ["ligne 1", "ligne 2"], "hook_sub": "Ton estimation en commentaire ↓",
  "rounds": [{
    "title": "MANCHE 1 · ON ÉCRASE",
    "kind": "compression" | "traction",
    "specimen": "cube" | "prism" | "egg_side" | "egg_up",
    "material": "concrete" | "egg",
    "label": "CHARGE", "unit": "t", "decimals": 0, "gauge_max": 120,
    "steps": [20, 50, 80], "question_value": 100, "question": "ÇA TIENT À 100 t ?", "question_sub": "…",
    "break_value": 85, "threshold": {"value": 85, "label": "seuil C30/37"} | null,
    "stamp": {"text": "CONFORME", "color": "green" | "red"}, "result_chip": "85 t",
    "fact": ["ligne 1", "ligne 2"]
  }],
  "explain": {"title": "…", "lines": ["…", "…", "…"]},
  "source": ["Valeurs : …", "…"], "cta": "PROCHAIN TEST ? Écris l'objet ↓",
  "caption": "…", "sources_detail": ["calcul…"]
}
```
Le passage `explain` montre aujourd'hui une poutre en béton armé (épisode béton). Pour un autre
objet, il faudra ajouter sa scène d'explication dans `engine.html` (`mode 2`).

## Local
```bash
node qc/render.js timeline assets/configs/qc/beton_c30_37.json /tmp/tl.json
node qc/render.js stills   assets/configs/qc/beton_c30_37.json /tmp/st 1.5,19.4,41
node qc/render.js video    assets/configs/qc/beton_c30_37.json /tmp/v.mp4      # ~3 s/frame sur CPU
python qc/audio.py /tmp/tl.json /tmp/a.wav
ffmpeg -i /tmp/v.mp4 -i /tmp/a.wav -c:v copy -c:a aac -shortest out.mp4
```
Textes à l'écran : dans la zone sûre TikTok/Reels (rien d'important au-dessus de y=250 ni sous y=1440 sur 1920).

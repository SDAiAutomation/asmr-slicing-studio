---
name: asmr-slicing-studio
description: Génère des vidéos ASMR verticales (Shorts/TikTok/Reels) au format « N CUTS -> X CUBES » : un objet 3D réaliste (steak, gâteau, mangue…) est coupé en grille niveau par niveau puis s'effondre en cubes, avec compteurs CUTS et CUBES exacts et sons synchronisés, rendu avec Blender + FFmpeg (GitHub Actions). Utilise ce skill dès que l'utilisateur parle de vidéo ASMR de découpe, slicing satisfaisant, « cut into N cubes », compteur de coupes/cubes, SliceLoop, ou veut refaire/améliorer une vidéo de découpe Blender, même s'il ne dit pas « skill ».
---

# ASMR Slicing Studio — format unique : grid

```
config.json ─► plan_grid.py ─► plan.json ─┬─► make_audio.py ─► soundtrack.wav
               (coupes, comptage exact     │
                des cubes, trajectoire     └─► blender_build.py + blender_grid.py
                du couteau)                       ─► scene.blend + previews ─► frames ─► MP4
```

Une seule source de vérité temporelle (`plan.json`) pilote le couteau, les coutures, les cubes, les compteurs et l'audio.

## Le format

Chaque niveau : un objet neuf glisse sur la planche, l'outil fait C coupes droites (X de gauche à droite, puis Y d'avant en arrière, puis Z de bas en haut), chaque coupe laisse une couture visible, puis l'objet s'effondre en ses cubes réels. Plus il y a de coupes, plus l'outil va vite : un couteau seul aux premiers niveaux, puis un peigne de lames parallèles de plus en plus fourni (jusqu'à ~50 lames), chaque lame faisant réellement sa coupe. Compteurs : CUTS (haut, « coupes faites / coupes du niveau ») et CUBES (sous l'objet), calculés sur l'occupation réelle de la forme, jamais estimés. La vidéo boucle : elle finit sur l'objet intact dans la pose d'ouverture.

Objets validés : `steak`, `cake`, `mango`. `soap`, `jelly`, `chocolate` existent mais ne sont pas validés à l'œil.

## Utilisation

1. `python scripts/run.py check` (Blender/FFmpeg ; sinon `BLENDER_PATH`, `FFMPEG_PATH`, `FFPROBE_PATH`).
2. Config, ex. : `{"object_type": "cake", "cuts_sequence": [2, 4, 8, 14, 24, 40, 64, 100, 160, 250], "video_duration": 62, "hook_text": "Guess the cubes at 250"}`
3. `python scripts/run.py preview config.json` → dossier de run (`plan.json`, `soundtrack.wav`, `scene.blend`, `blender_report.json`, `previews/`).
4. Regarder les previews avant tout rendu complet.
5. `python scripts/run.py render RUN_DIR [--final]` puis `python scripts/run.py export RUN_DIR`.

Sur GitHub : Actions → *Render ASMR video* → `config_json` (prioritaire) ou les champs objet / coupes / qualité.

## Paramètres (`config.json`)

| clé | défaut | rôle |
|---|---|---|
| `object_type` | `steak` | `steak`, `cake`, `mango` (+ `soap`, `jelly`, `chocolate`) |
| `cuts_sequence` | `[4, 10, 15, 25, 40, 60, 100, 160, 250, 400, 600, 1000]` | coupes par niveau (≤ 3000) |
| `video_duration` | auto (62 s en CI) | le rythme est mis à l'échelle pour tenir la durée |
| `fps`, `resolution`, `preview_resolution`, `export_resolution` | 30, 1080×1920, 540×960, — | 9:16 |
| `render_engine` | `eevee` (CI : `cycles`) | |
| `hook_text`, `end_text` | `""` | texte d'ouverture / d'appel aux commentaires (anglais, sans emoji) |
| `board_watermark` | `"@sliceloop"` | pseudo gravé sur la planche (anti-repost) |
| `display_cubes_max` | 1 200 000 | cubes réellement instanciés par niveau (le compteur reste exact) |
| `max_cubes` | 30 000 000 | garde-fou sur le dernier niveau |
| `random_seed` | 42 | même seed = même vidéo |

Les anciennes clés `format`, `target_piece_count`, `cutting_mode`, `progression_sequence`… sont ignorées ; `format` autre que `grid` est refusé.

## Revue (obligatoire avant de dire « prêt »)

- `previews/01_opening.png` : l'objet est-il reconnaissable ?
- `previews/02_cut_in_progress.png` : coutures visibles, lame dans le plan de coupe.
- `previews/06_comb_last_level.png` : le peigne du dernier niveau, lames dans l'objet.
- `previews/05_progression_sheet.png` : les cubes augmentent à chaque niveau.
- `blender_report.json` : `failures == []`, `cubes_built == cubes_planned` à chaque niveau.
- Audio : écouter au casque ; le son est procédural (échantillons réels possibles dans `assets/samples/<profil>/`).

## Tests

`python -m pytest tests -q` : comptes exacts et croissants, une coupe annoncée = une coupe faite, rythme qui accélère, trajectoire du couteau (pas de saut, jamais dans la planche, jusqu'à la planche), synchro audio, et — si Blender est trouvé — construction réelle d'une scène.

Architecture et limites : `references/architecture.md`. Rôle de la tâche quotidienne : `references/daily_planner.md`.

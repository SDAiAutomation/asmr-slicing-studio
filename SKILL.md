---
name: asmr-slicing-studio
description: Génère des vidéos ASMR verticales (Shorts/TikTok/Reels) où un objet 3D réaliste (mangue, savon, gelée, chocolat) est découpé PROGRESSIVEMENT dans plusieurs directions (2 → 3 → 5 → 15 → 100 → 400 morceaux), avec un compteur vérifié par la géométrie et des sons de découpe synchronisés, en local avec Blender + FFmpeg. Utilise ce skill dès que l'utilisateur parle de vidéo ASMR de découpe, slicing satisfaisant, "cut into N pieces", couteau qui coupe un fruit en 3D, compteur de morceaux, ou veut refaire/améliorer une vidéo de découpe Blender, même s'il ne dit pas "skill".
---

# ASMR Slicing Studio

Pipeline local, déterministe, sans API payante :

```
config.json ─► plan_cuts.py ─► plan.json ─┬─► make_audio.py ─► soundtrack.wav
                (découpes réelles,        │
                 comptage voxel)          └─► blender_build.py ─► scene.blend + previews
                                                     │ (rejoue chaque coupe sur les meshes,
                                                     │  valide, pilote le compteur)
                                                     └─► frames PNG ─► export (MP4 H.264+AAC)
```

Une seule source de vérité temporelle (`plan.json`) pilote le couteau, les morceaux, le compteur et l'audio : ils ne peuvent pas se désynchroniser.

## Pourquoi ce design (lire avant de modifier)

Les guides type « Pehle se pieces banao aur seams reveal karo » pré-découpent tout l'objet puis révèlent les coutures. Résultat : une découpe globale déguisée, un compteur qui compte des coups de couteau et non des morceaux, et aucune progression réelle. Ici :

- **Chaque coupe part de l'état précédent.** Blender coupe le mesh du morceau parent (bisect + capuchon + séparation des îlots) au moment exact du passage de la lame. Les enfants apparaissent à la frame où le parent disparaît : rien ne disparaît, rien n'apparaît de nulle part.
- **Coups de lame ≠ plans de coupe ≠ morceaux.** Un coup (stroke) = une direction appliquée à plusieurs morceaux ; un plan peut créer 2 morceaux ou plus. Le compteur affiche le nombre de **parties de mesh validées** par Blender, jamais une estimation.
- **Objet reconnaissable.** La mangue est une forme paramétrique inversible (réniforme, épaule côté pédoncule, petit bec, aplatie) de ~13,5 × 9,7 × 8,3 cm. Planificateur et Blender utilisent la même équation (`scripts/common.py`), donc les centroïdes concordent.

## Deux formats (compteurs CUTS + PIECES/CUBES dans les deux)

- `format: "slices"` (défaut) : découpe progressive réelle, morceau par morceau (planificateur `plan_cuts.py`).
- `format: "grid"` : niveaux « N CUTS → X CUBES » (`plan_grid.py` + `blender_grid.py`) : coupes en grille sur X, Y puis Z, coutures visibles, effondrement en cubes instanciés (Geometry Nodes), jusqu'à ~12 millions de cubes. `cuts_sequence` fixe les coupes par niveau. Objets : `steak`, `cake`, `mango`, `soap`…

## Utilisation

1. **Vérifier les outils** (une fois) : `python scripts/run.py check`. Si Blender/FFmpeg introuvables, demander le chemin complet et définir `BLENDER_PATH`, `FFMPEG_PATH`, `FFPROBE_PATH`.
2. **Traduire la demande en config** (`config.json`). Exemple : *« realistic mango progressively sliced into 400 pieces, multiple directions »* →
   ```json
   {"object_type": "mango", "target_piece_count": 400,
    "cutting_directions": ["vertical","horizontal","diagonal","cross","angled","random3d"],
    "random_seed": 42}
   ```
   Ne renseigner que ce que l'utilisateur précise ; les défauts (voir ci-dessous) sont faits pour un Short.
3. **Preview** : `python scripts/run.py preview config.json` → nouveau dossier de run avec `plan.json`, `soundtrack.wav`, `scene.blend`, `impact_frames.json`, `blender_report.json`, `previews/` (ouverture, coupe en cours, mi-progression, final, planche de progression `05_progression_sheet.png`).
4. **Inspecter réellement** avant tout rendu complet (voir « Revue »). Montrer les previews à l'utilisateur et lister la direction de chaque coup.
5. **Rendu** après validation : `python scripts/run.py render RUN_DIR --final` (reprend là où il s'est arrêté ; sans `--final` = résolution preview).
6. **Export** : `python scripts/run.py export RUN_DIR` → MP4 + `export_report.json` (dimensions, frames, durée, flux AAC, décodage complet sans erreur).

`python scripts/run.py all config.json --final` enchaîne tout, mais ne l'utiliser que si l'utilisateur a déjà validé un preview.

## Paramètres (`config.json`)

| clé | défaut | rôle |
|---|---|---|
| `object_type` | `mango` | `mango`, `soap`, `jelly`, `chocolate` |
| `material_profile` / `audio_profile` | = objet | visuel+mécanique / son |
| `target_piece_count` | 15 | nombre final, sans plafond artificiel (testé jusqu'à 400) |
| `progression_sequence` | auto `2,3,4,5,8,10,15,25,50,100,200,400…` | paliers visibles ; un coup peut au plus doubler, les grands sauts sont découpés en plusieurs coups |
| `cutting_mode` | `cumulative` | `cumulative` (mode A) ou `levels` (mode B : un objet neuf par niveau, l'ancien glisse hors du plan) |
| `cutting_directions` | les 6 | `vertical`, `horizontal`, `diagonal`, `cross`, `angled`, `random3d` ; tournent à chaque coup |
| `cutting_speed` | `auto` | `slow`/`medium`/`fast` ; les phases denses accélèrent automatiquement |
| `camera_style` | `follow` | `follow` (recadre à chaque palier) ou `static` |
| `fps`, `resolution`, `preview_resolution` | 30, 1080×1920, 540×960 | 9:16 |
| `render_engine` | `eevee` | `cycles` = plus réaliste, beaucoup plus lent |
| `watermark`, `show_counter`, `music` | `""`, true, false | |
| `board_watermark` | `"@WonderLoopTV-sd"` | pseudo gravé sur la planche (coin avant gauche), anti-repost ; `""` pour l'enlever |
| `random_seed` | 42 | même seed = même vidéo |
| `output_path` | `runs` | chaque run crée un dossier neuf horodaté |

Chaque plan de coupe = point + normale 3D dans `plan.json` (`hits[].plane_point/plane_normal`) : pour des coupes sur mesure, éditer ces champs ou ajouter une direction dans `common.direction_vector`.

## Revue (obligatoire avant de dire « prêt »)

Les tests unitaires ne suffisent pas. Ouvrir et regarder :
- `previews/01_opening.png` : la mangue est-elle immédiatement reconnaissable ? (forme réniforme, dégradé vert→jaune→rouge, lenticelles)
- `previews/02_cut_in_progress.png` : la lame est dans le plan de coupe, sur le bon morceau.
- `previews/05_progression_sheet.png` : le nombre de morceaux augmente visiblement à chaque vignette, directions variées.
- `blender_report.json` : `final_pieces_validated == target`, `failures == []`, `non_manifold == 0`, `volume_error_pct < 1`.
- Audio : écouter au casque le 1er, un coup du milieu et le dernier. Ne jamais affirmer qu'un son est réaliste sans écoute humaine ; le skill génère du son procédural, la qualité finale dépend d'une validation à l'oreille ou d'échantillons réels déposés dans `assets/samples/<profil>/<événement>_*.wav` (`contact`, `body`, `end`, `separate`, `land`).

Si un critère échoue : corriger la cause (forme dans `common.py`, matériaux dans `blender_build.py`, plan dans `plan_cuts.py`), relancer un **nouveau** preview, ne pas maquiller le compteur.

## Tests

`python tests/test_pipeline.py` (ou `pytest tests`) : progression, paliers, compteur = parties créées, multidirectionnel, conservation du volume (partition exacte des voxels), aucun morceau qui flotte/disparaît, forme non-ellipsoïde, reproductibilité seed, mode levels, synchro audio à l'échantillon près, absence de clipping, variation entre coups, et — si Blender est trouvé — reconstruction réelle de 15 morceaux sans mesh non-manifold.

## Référence

- `references/architecture.md` : choix techniques, comptage, performances mesurées, limites connues, pistes d'optimisation. Le lire avant de toucher à la géométrie ou d'annoncer des performances.

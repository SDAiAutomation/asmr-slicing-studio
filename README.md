# ASMR Slicing Studio

Vidéos ASMR verticales « N CUTS -> X CUBES » (Blender + FFmpeg), rendues sur GitHub Actions pour SliceLoop (@sliceloop).

**Lancer une vidéo :** onglet *Actions* → *Render ASMR video* → *Run workflow* (objet, coupes par niveau, qualité).
Le MP4 est déposé sur Google Drive (sinon dans les *Artifacts* du run).

Détails du skill : [SKILL.md](SKILL.md) · architecture et limites : [references/architecture.md](references/architecture.md)


## Papaye en tranches

Dans **Render ASMR video**, utiliser `config_json` avec `{"format":"papaya_slices","object_type":"papaya","cuts_sequence":[50],"fps":24}`. En local, enregistrer cette configuration et lancer `python scripts/run.py preview config.json`. Les 50 impacts commencent à l’image 29, espacés de six images à 24 i/s. La simulation libère les 100 demi-tranches à l’image 343. Le dossier du rendu contient la scène éditable, `impact_frames.json`, les aperçus et la piste stéréo. Le format accepte un seul niveau. Les autres objets utilisent le format grille.

# GTA Buzz Engine

Moteur de Shorts verticaux (1080×1920) qui surfe sur l'actu GTA 6 **sans aucune image du jeu** : chiffres animés, décor néon généré, sous-titres mot à mot, musique synthwave générée par code. Zéro risque Content ID.

## Installation

Il faut Python 3.10 ou plus récent, et `ffmpeg` dans le PATH.

```bash
pip install -r requirements.txt
cp .env.example .env      # puis remplis tes clés (facultatif)
set -a; source .env; set +a
```

Voix off : `pip install -r requirements-tts.txt` et `espeak-ng` installent Kokoro, une voix gratuite. Sans Kokoro ni clé ElevenLabs, les vidéos sortent avec la musique seule.

## Workflow quotidien (10 min)

```bash
python run.py veille                     # 1. sujets GTA 6 du jour (Google News FR/EN + Reddit), filtrés anti-spoiler
python run.py script "GTA 6 bat le record de Twitch" \
   --facts "6 521 115 spectateurs simultanés le 18/11" --sources "GatorStreams"   # 2. brouillon FR + EN (clé Claude)
# 3. relis les chiffres dans specs/auto/*.json
python run.py render specs/auto/          # 4. rendu + garde-fous
```

Chaque vidéo sort dans `out/<id>/` avec trois fichiers :

- `<id>.mp4` : la vidéo à publier.
- `<id>.txt` : le titre, la description (avec les sources), les hashtags et le commentaire à épingler, prêts à copier-coller.
- `publication.json` : les mêmes infos, au format machine.
- `report.json` : le résultat des garde-fous.

## Format d'une spec

Une vidéo correspond à un fichier JSON dans `specs/`. Types de scènes disponibles :

| Type | Champs | Usage |
| --- | --- | --- |
| `hook` | `big`, `small` | Accroche des 7 premières secondes |
| `stat` | `value`, `decimals`, `prefix`, `suffix`, `label`, `top` | Chiffre avec compteur animé |
| `text` | `big`, `small` | Une idée forte |
| `versus` | `left{top,big}`, `right{top,big}`, `small` | Comparaison |
| `cta` | `big`, `small` | Question finale qui fait commenter |

Champ facultatif `pinned_comment` : le commentaire à épingler. S'il est absent, il est construit à partir de la question finale.

Chaque scène a un champ `voice` : c'est le texte lu par la voix off, qui sert aussi de sous-titres. Deux thèmes : `vice` (néon rose) et `money` (vert, pour Argent Simple).

## Garde-fous (bloquants)

- Le titre contient « GTA 6 » et fait moins de 100 caractères.
- Aucun terme lié aux spoilers, fuites ou piratage (`leak`, `fuite`, `spoiler`, `ending`…).
- Aucun mot violent dans les 7 premières secondes (règle de pub YouTube de septembre 2026).
- Toute vidéo qui affiche des chiffres a un champ `sources` renseigné.
- La durée fait au moins 61 s (seuil de rémunération TikTok). Si la voix finit un peu avant, la scène finale est prolongée de 6 s au maximum.

`--force` passe outre les garde-fous, à éviter.

## Règles à ne jamais casser

- Pas d'images du jeu ni des trailers, pas de clips d'autres créateurs, pas de logo GTA.
- Pas de « faux leaks » ni de trailers générés par IA : Take-Two les fait retirer.
- Les chiffres doivent venir de sources vérifiées, et jamais de Reddit seul.

## Rendu automatique sur GitHub Actions (branche `gta6-buzz`)

Chaque push d'un fichier dans `gta-buzz/specs/` lance le workflow **GTA Buzz render**, qui fait trois choses :

1. Il rend les specs ajoutées ou modifiées, avec la voix off ElevenLabs.
2. Il dépose le MP4 et `publication.json` sur Google Drive, dans `GTA6-Buzz/<id>/`, grâce au secret `RCLONE_CONF` déjà utilisé par l'ASMR.
3. Il garde une copie dans les Artifacts du run pendant 14 jours.

**Voix off gratuite par défaut : [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M)** (licence Apache 2.0, tourne sur le CPU du runner, sans clé ni crédits).

- FR : `ff_siwis`
- EN : `am_michael`

Pour changer de voix sur une vidéo, ajoute `"kokoro_voice"` dans sa spec. Pour changer le débit, ajoute `"speed"` (1.08 par défaut).

Option payante : si le secret `ELEVENLABS_API_KEY` existe, ElevenLabs remplace Kokoro automatiquement.

On peut aussi lancer un rendu à la main : onglet *Actions* → *GTA Buzz render* → *Run workflow*.

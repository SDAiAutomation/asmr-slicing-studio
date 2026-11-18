# GTA Buzz Engine

Moteur de Shorts verticaux (1080×1920) qui surfe sur l'actu GTA 6 **sans aucune image du jeu** : chiffres animés, décor néon généré, sous-titres mot à mot, musique synthwave générée par code. Zéro risque Content ID.

## Installation

Il faut Python 3.10 ou plus récent, et `ffmpeg` dans le PATH.

```bash
pip install -r requirements.txt
cp .env.example .env      # puis remplis tes clés (facultatif)
set -a; source .env; set +a
```

Sans clé, tout fonctionne : les vidéos sortent avec la musique seule. Avec `ELEVENLABS_API_KEY` et `VOICE_FR` / `VOICE_EN`, la voix off est ajoutée et les sous-titres se calent dessus.

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
- `publication.json` : le titre, la description et les hashtags à copier-coller.
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

Chaque scène a un champ `voice` : c'est le texte lu par la voix off, qui sert aussi de sous-titres. Deux thèmes : `vice` (néon rose) et `money` (vert, pour Argent Simple).

## Garde-fous (bloquants)

- Le titre contient « GTA 6 » et fait moins de 100 caractères.
- Aucun terme lié aux spoilers, fuites ou piratage (`leak`, `fuite`, `spoiler`, `ending`…).
- Aucun mot violent dans les 7 premières secondes (règle de pub YouTube de septembre 2026).
- Toute vidéo qui affiche des chiffres a un champ `sources` renseigné.
- La durée fait au moins 12 s. Au-delà de 60 s, c'est seulement un avertissement.

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

**À faire une fois :** ajouter la clé ElevenLabs dans *Settings → Secrets and variables → Actions*, sous le nom `ELEVENLABS_API_KEY`.

Voix par défaut, modifiables via les variables `VOICE_FR` / `VOICE_EN` :

- FR : Corentin (`IHngRooVccHyPqB4uQkG`)
- EN : Brian (`nPczCjzI2devNBz1zQrb`)

On peut aussi lancer un rendu à la main : onglet *Actions* → *GTA Buzz render* → *Run workflow*.

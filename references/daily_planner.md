# Planificateur quotidien (rôle de Claude)

Chaque jour, la tâche planifiée Claude décide **une** vidéo, la lance sur GitHub Actions et n'attend pas le rendu.

## Entrées
- `history/videos.jsonl` : une ligne par vidéo produite (config, hook, légende, statut, et — quand Saliou les ajoute — `views`, `likes`, `comments`).
- Ce fichier (leviers, garde-fous).

## Format (unique) : grid "N CUTS -> X CUBES"
- Un objet neuf par niveau, des coupes en grille qui augmentent, puis effondrement en cubes. Compteurs CUTS (haut, « coupes faites / coupes du niveau ») et CUBES (bas), exacts.
- Clés : `object_type`, `cuts_sequence` (ex. [3, 8, 20, 45, 90, 160, 300, 500, 750, 1000]), `hook_text`, `end_text`, `random_seed`, `quality` (`hd` par défaut).
- Durée **≥ 60 s** (TikTok ne rémunère que les vidéos de plus d'une minute) : `video_duration: 62` posé automatiquement par le workflow, 8 à 12 niveaux pour que la minute soit remplie d'action.
- Les formats `slices` et `levitate` ont été supprimés : ne plus les demander (le workflow refuse un `format` autre que `grid`).

## Ce qui nous distingue des chaînes concurrentes (à exploiter)
- **Comptes exacts** : nos chiffres sont calculés, pas « ≈ ». À dire dans la légende ("every cube counted").
- **Boucle parfaite** : la vidéo finit sur l'objet intact dans la pose d'ouverture → relectures automatiques.
- **Action dès 0,5 s** : pas d'intro, la première coupe tombe tout de suite.
- **Séries** : "Day N" avec un nombre qui grandit chaque jour, et la réponse à la devinette de la veille dans la légende.
- **Variété d'objets** avec le même format (steak, gâteau, mangue…) : c'est l'objet et la séquence qui changent chaque jour, pas le format.

## Leviers d'engagement (choisir 1 angle principal par jour)
1. **Nombre absurde** : 1 000 coupes, 10 millions de cubes. Le compteur est le héros. Hook du type "Can it reach 10 million cubes?".
2. **Devinette** : hook "Guess the final number" + end_text "Comment your guess for tomorrow" → appelle les commentaires.
3. **Niveaux** : format défi "Level 1: 4 cubes ... Level 10: ?" → rétention jusqu'à la fin.
4. **Petit nombre très lent et très net** (15–50) : ASMR pur, gros plans, son mis en avant.
5. **Série** : nombre de coupes final qui grandit chaque jour ("Day 6: 1,200 cuts") → abonnements pour voir la suite.
6. **Débat** : end_text "Mango or soap next?" → choix binaire, facile à commenter.

Règles :
- Chaque vidéo est publiée, donc aucune répétition : jamais le même objet que la vidéo précédente (et si possible aucun des 3 derniers), ni le même angle, ni la même séquence. Le workflow refuse une répétition de la dernière vidéo réussie (même objet ou même séquence ; `"allow_repeat": true` dans `config_json` pour un nouveau rendu volontaire).
- Si `views`/`comments` sont renseignés, privilégier les angles des 3 meilleures vidéos (70 %) et garder 30 % d'exploration.
- Textes à l'écran **en anglais**, courts (≤ 6 mots pour le hook), **sans emoji** (la police Blender ne les affiche pas). Les emojis sont permis dans la légende.
- Légende : 1 phrase d'accroche + 1 question qui appelle un commentaire + 5 à 8 hashtags (#asmr #satisfying #oddlysatisfying #asmrsounds #3danimation ...).

## Lisibilité sur téléphone
- Zone sûre : l'appli recouvre le haut (~13 %, barre de recherche) et le bas (~25 %, pseudo + légende) de l'écran. Les compteurs sont placés dedans (CUTS à ~20 % du haut, CUBES juste sous l'objet) : ne pas les redescendre.
- Au-delà de ~100 000 cubes, la pile finale devient une texture (on ne voit plus les cubes) : c'est le compteur qui porte le « waouh », garder 1 à 2 niveaux où les cubes restent visibles (≤ 5 000).

## Garde-fous techniques
- Objets : `steak`, `cake`, `mango` validés. `soap`, `jelly`, `chocolate` existent mais ne sont pas validés à l'œil : au plus 1 essai par semaine.
- Dernier niveau ≤ 1 000 coupes en `hd` (au-delà, le rendu CPU dépasse les limites GitHub) ; 3 000 coupes max par niveau.
- Si la dernière ligne de l'historique est `failed`, refaire une config sûre (steak ou cake, ≤ 250 coupes) avant de reprendre les gros nombres.
- `random_seed` = date du jour au format AAAAMMJJ.

## Hors périmètre
- Les vidéos Breakpoint Lab (Tests, Physics — @BreakpointLabTV) vivent dans le dépôt `breakpoint-lab-studio`. Ce dépôt ne fait que SliceLoop (@sliceloop).

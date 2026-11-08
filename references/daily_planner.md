# Planificateur quotidien (rôle de Claude)

Chaque jour, la tâche planifiée Claude décide **une** vidéo, la lance sur GitHub Actions et n'attend pas le rendu.

## Entrées
- `history/videos.jsonl` : une ligne par vidéo produite (config, hook, légende, statut, et — quand Saliou les ajoute — `views`, `likes`, `comments`).
- Ce fichier (leviers, garde-fous).

## Formats
- **`grid` (format principal, ~5 jours sur 7)** : niveaux "N CUTS -> X CUBES". Un objet neuf par niveau, des coupes en grille qui augmentent, puis effondrement en cubes. Compteurs CUTS (haut) et CUBES (bas), exacts. Clés : `format: "grid"`, `cuts_sequence` (ex. [4, 10, 25, 60, 160, 400, 1000]), objet `steak` (validé en priorité), `cake`, `mango`. Jusqu'à ~1 000 coupes (≈ 12 millions de cubes) ; garder la vidéo entre 35 et 70 s (6 à 12 niveaux).
- **`slices`** : découpe progressive en morceaux, compteurs CUTS et PIECES. Pour varier. Toujours en `cutting_mode: "levels"` : chaque niveau repart de l'objet intact (c'est la règle des formats qui marchent), `progression_sequence` = nombre de morceaux par niveau (ex. [2, 5, 15, 50, 200, 1000]).
- Règle commune : chaque niveau repart toujours de la forme initiale ; le compteur du haut affiche « coupes faites / coupes du niveau ».

## Ce qui nous distingue des chaînes concurrentes (à exploiter)
- **Comptes exacts** : nos chiffres sont calculés, pas « ≈ ». À dire dans la légende ("every cube counted").
- **Boucle parfaite** : la vidéo finit sur l'objet intact dans la pose d'ouverture → relectures automatiques.
- **Action dès 0,5 s** : pas d'intro, la première coupe tombe tout de suite.
- **Séries** : "Day N" avec un nombre qui grandit chaque jour, et la réponse à la devinette de la veille dans la légende.
- **Variété d'objets** avec le même format (steak, gâteau, mangue…), pas un seul objet.

## Leviers d'engagement (choisir 1 angle principal par jour)
1. **Nombre absurde** : 1 000, 5 000, 10 000, 25 000 morceaux. Le compteur est le héros. Hook du type "Can it reach 25,000 pieces?".
2. **Devinette** : hook "Guess the final number" + end_text "Comment your guess for tomorrow" → appelle les commentaires.
3. **Niveaux** (`cutting_mode: levels`) : format défi "Level 1: 2 pieces ... Level 5: ?" → rétention jusqu'à la fin.
4. **Petit nombre très lent et très net** (15–50) : ASMR pur, gros plans, son mis en avant.
5. **Série** : même objet, nombre qui double chaque jour ("Day 6: 3,200 pieces") → abonnements pour voir la suite.
6. **Débat** : end_text "Mango or soap next?" → choix binaire, facile à commenter.

Règles :
- Alterner : ne pas répéter le même angle 2 jours de suite, ni le même nombre de morceaux.
- Au moins 1 vidéo ≥ 5 000 morceaux par semaine, au plus 2 (rendu long).
- Si `views`/`comments` sont renseignés, privilégier les angles des 3 meilleures vidéos (70 %) et garder 30 % d'exploration.
- Textes à l'écran **en anglais**, courts (≤ 6 mots pour le hook), **sans emoji** (la police Blender ne les affiche pas). Les emojis sont permis dans la légende.
- Légende : 1 phrase d'accroche + 1 question qui appelle un commentaire + 5 à 8 hashtags (#asmr #satisfying #oddlysatisfying #asmrsounds #3danimation ...).

## Garde-fous techniques
- Objets : `steak` = objet phare du format grid ; `mango` validé en slices. `soap`, `jelly`, `chocolate` existent mais ne sont pas encore validés à l'œil : au plus 1 essai par semaine, et seulement après une vidéo mango réussie la veille.
- `target_piece_count` ≤ 25 000. Au-delà de 5 000, `quality: preview` (le rendu CPU dépasse sinon les 5 h de GitHub).
- Si la dernière ligne de l'historique est `failed`, refaire une config sûre (mango, ≤ 400, preview) avant de reprendre les gros nombres.
- `random_seed` = date du jour au format AAAAMMJJ.

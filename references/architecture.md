# Architecture et limites (format grid)

## 1. Planification (`plan_grid.py`, sans Blender)

- **Coupes par axe** : `split_cuts` répartit les C coupes d'un niveau sur X, Y, Z pour que les cellules soient le plus cubiques possible.
- **Comptage exact** : `common.grid_occupancy` teste chaque cellule sur une sous-grille (3³ points, 2³ au-delà de 3 M cellules) ; une cellule est un cube dès qu'une partie de l'objet s'y trouve. Le compteur CUBES en cours de niveau compte les boîtes non vides de la grille partielle (coupes déjà faites seulement).
- **Rythme** : `cut_schedule` ouvre chaque niveau sur des coupes calmes et séparées, puis accélère (exponentielle sur 60 % du niveau, puis vitesse max). `plan()` met l'ensemble à l'échelle pour tenir `video_duration`.
- **Couteau** (`knife_traj.py`) : chaque passage suit approche → contact → pénétration → traversée → séparation → retrait, calculé sur l'intersection réelle du plan avec la forme. Une coupe verticale s'arrête sur la planche (jamais dedans). Quand le rythme dépasse le geste le plus rapide possible, un passage emporte toutes les coupes dues avant sa sortie ; le compteur continue à monter une par une.

## 2. Scène Blender (`blender_grid.py`)

- Objet : mesh paramétrique (mangue) ou extrudé à contour irrégulier (steak, gâteau), matériaux procéduraux (persillé du steak continu à travers les cubes).
- **Coutures** : shader piloté par le nombre de coupes faites par axe (paramètres animés), donc des milliers de coupes ne coûtent rien en géométrie.
- **Cubes** : un point par cellule occupée, instancié en cube par Geometry Nodes ; effondrement (affaissement + étalement) animé par un paramètre `Collapse`. Au-delà de `display_cubes_max`, on garde toutes les cellules de surface et une part des cellules intérieures (invisibles).
- Caméra fixe, planche signature identique dans toutes les vidéos, compteurs attachés à la caméra dans la zone sûre TikTok/Reels.

## 3. Audio (`make_audio.py`)

Un « whoosh » + une coupe par passage de lame (aux frames de contact/sortie), une cascade de grains + un choc sourd à chaque effondrement. Procédural, normalisé à −1 dBFS.

## 4. Limites connues

- Les cubes sont des cellules pleines : au bord d'un objet arrondi, ils dépassent légèrement de la forme au moment de l'effondrement.
- Les coutures sont peintes (shader), pas de vraies entailles dans la géométrie.
- Pas de simulation rigid-body : l'effondrement est une animation procédurale (stable, reproductible).
- Au-delà de ~100 000 cubes, la pile finale devient une texture : c'est le compteur qui porte le « waouh ».

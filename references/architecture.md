# Architecture et limites (format grid)

## 1. Planification (`plan_grid.py`, sans Blender)

- **Coupes par axe** : `split_cuts` répartit les C coupes d'un niveau sur X, Y, Z pour que les cellules soient le plus cubiques possible.
- **Comptage exact** : `common.grid_occupancy` teste chaque cellule sur une sous-grille (3³ points, 2³ au-delà de 3 M cellules) ; une cellule est un cube dès qu'une partie de l'objet s'y trouve. Le compteur CUBES en cours de niveau compte les boîtes non vides de la grille partielle (coupes déjà faites seulement).
- **Rythme** : `cut_schedule` ouvre chaque niveau sur des coupes calmes et séparées, puis accélère (exponentielle sur 60 % du niveau, puis vitesse max). `plan()` met l'ensemble à l'échelle pour tenir `video_duration`.
- **Couteau** (`knife_traj.py`) : chaque passage suit approche → contact → pénétration → traversée → séparation → retrait, calculé sur l'intersection réelle du plan avec la forme. Une coupe verticale s'arrête sur la planche (jamais dedans). Quand le rythme dépasse le geste le plus rapide possible, un passage emporte toutes les coupes dues avant sa sortie ; le compteur continue à monter une par une.

## 2. Scène Blender (`blender_grid.py`)

- Objet : mesh paramétrique (mangue) ou extrudé à contour irrégulier (steak, gâteau), matériaux procéduraux (persillé du steak continu à travers les cubes).
- **Coutures** : shader piloté par le nombre de coupes faites par axe (paramètres animés), donc des milliers de coupes ne coûtent rien en géométrie. Couleur par matière (`MATERIALS[...]["seam"]`, contrastée avec la surface : claire sur le glaçage noir du gâteau, sombre sur la mangue et le steak) ; demi-largeur `min(0,9 mm, 0,3 × cellule)` : ~2 px aux premiers niveaux, puis un voile qui balaie l'objet quand les cellules passent sous le pixel.
- **Cubes** : un point par cellule occupée, instancié en cube par Geometry Nodes ; les cubes du bord sont réduits au morceau qu'ils contiennent vraiment (`common.grid_fit`, attribut `fit`), donc la pile ne déborde pas de la silhouette au moment du passage aux cubes ; effondrement (affaissement + étalement) animé par un paramètre `Collapse`. Au-delà de `display_cubes_max`, on garde toutes les cellules de surface et une part des cellules intérieures (invisibles).
- Caméra fixe, planche signature identique dans toutes les vidéos, compteurs attachés à la caméra dans la zone sûre TikTok/Reels.

## 3. Audio (`make_audio.py`)

Un « whoosh » + une coupe par passage de lame (aux frames de contact/sortie), un tic court pour chaque coupe supplémentaire d'un passage rapide, une cascade de grains + un choc sourd à chaque effondrement. Procédural, normalisé à −1 dBFS.

## 4. Limites connues

- Les cubes du bord sont des boîtes réduites (au tiers de cellule près), pas la forme exacte du morceau.
- Au-delà d'environ 4 coupes par seconde, un passage de lame emporte plusieurs coupes : le compteur continue de monter une par une, avec un petit « tic » sonore par coupe.
- Les coutures sont peintes (shader), pas de vraies entailles dans la géométrie.
- Pas de simulation rigid-body : l'effondrement est une animation procédurale (stable, reproductible).
- Au-delà de ~100 000 cubes, la pile finale devient une texture : c'est le compteur qui porte le « waouh ».

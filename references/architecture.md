# Architecture et limites (format grid)

## 1. Planification (`plan_grid.py`, sans Blender)

- **Coupes par axe** : `split_cuts` répartit les C coupes d'un niveau sur X, Y, Z pour que les cellules soient le plus cubiques possible.
- **Comptage exact** : `common.grid_occupancy` teste chaque cellule sur une sous-grille (3³ points, 2³ au-delà de 3 M cellules) ; une cellule est un cube dès qu'une partie de l'objet s'y trouve. Le compteur CUBES en cours de niveau compte les boîtes non vides de la grille partielle (coupes déjà faites seulement).
- **Vitesse qui monte avec le nombre de coupes** (`pass_budget`, `allocate_passes`) : chaque niveau reçoit un nombre de passages visibles et un rythme moyen qui augmentent de niveau en niveau (≈ 1,2 → 4,6 passages/s visés, 6 → 22 passages max). Tant que les coupes tiennent dans ce budget, un seul couteau fait une coupe par passage. Au-delà, chaque passage est fait par un **peigne de lames parallèles** : avec p passages pour c coupes sur un axe, le passage j fait les coupes j+1, j+1+p, j+1+2p… → ceil(c/p) lames espacées de p cellules. Chaque coupe annoncée est donc réellement faite par une lame (testé). Dans un niveau, les passages accélèrent aussi (`cut_schedule`). Ordre : X gauche→droite, Y avant→arrière, Z bas→haut (les lames en trop d'un peigne passent au-dessus de l'objet, jamais dans la planche).
- **Outils** (`tool_windows`) : couteau seul ou peigne, échangé en l'air entre deux séries de passages.
- **Couteau** (`knife_traj.py`) : chaque passage suit approche → contact → pénétration → traversée → séparation → retrait, calculé sur l'intersection réelle des plans avec la forme (plusieurs plans du peigne échantillonnés). Geste le plus court : 5 images (un « chop »). Une coupe verticale s'arrête sur la planche.

## 2. Scène Blender (`blender_grid.py`)

- Objet : mesh paramétrique (mangue) ou extrudé à contour irrégulier (steak, gâteau), matériaux procéduraux (persillé du steak continu à travers les cubes).
- **Coutures** : shader piloté par axe par le nombre de coupes, le pas du peigne et le nombre de passages faits (paramètres animés), donc des milliers de coupes ne coûtent rien en géométrie. Couleur par matière (`MATERIALS[...]["seam"]`, contrastée avec la surface : claire sur le glaçage noir du gâteau, sombre sur la mangue et le steak) ; demi-largeur `min(0,9 mm, 0,3 × cellule)` : ~2 px aux premiers niveaux, puis un voile qui balaie l'objet quand les cellules passent sous le pixel.
- **Cubes** : un point par cellule occupée, instancié en cube par Geometry Nodes ; les cubes du bord sont réduits au morceau qu'ils contiennent vraiment (`common.grid_fit`, attribut `fit`), donc la pile ne déborde pas de la silhouette au moment du passage aux cubes ; effondrement (affaissement + étalement) animé par un paramètre `Collapse`. Au-delà de `display_cubes_max`, on garde toutes les cellules de surface et une part des cellules intérieures (invisibles).
- Rétention : « LEVEL x/N » + barre de progression, compteur CUBES qui « pop » à chaque effondrement, dernier effondrement au ralenti avec travelling avant, flou de mouvement sur le couteau seul.
- Caméra fixe (sauf le travelling final), planche signature identique dans toutes les vidéos, compteurs attachés à la caméra dans la zone sûre TikTok/Reels.

## 3. Audio (`make_audio.py`)

Un « whoosh » + une coupe par passage de lame (aux frames de contact/sortie), plus épaisse pour un peigne (plusieurs lames qui mordent à quelques ms d'écart), une cascade de grains + un choc sourd à chaque effondrement. Procédural, normalisé à −1 dBFS.

## 4. Limites connues

- Les cubes du bord sont des boîtes réduites (au tiers de cellule près), pas la forme exacte du morceau.
- Les coutures sont peintes (shader), pas de vraies entailles dans la géométrie.
- Pas de simulation rigid-body : l'effondrement est une animation procédurale (stable, reproductible).
- Au-delà de ~100 000 cubes, la pile finale devient une texture : c'est le compteur qui porte le « waouh ».

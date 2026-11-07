# Architecture, mesures et limites

## 1. Diagnostic des défauts de l'approche « pré-découpe + révélation »

| Symptôme observé | Cause racine | Correction dans ce skill |
|---|---|---|
| Découpe globale au lieu d'une progression | Tous les morceaux sont créés au départ (N plans parallèles d'un coup), le couteau ne fait que « révéler » des coutures | `plan_cuts.py` construit un arbre parent → enfants ; `blender_build.py` coupe le mesh du parent au moment du passage de la lame |
| Compteur faux / sans lien avec la géométrie | Le compteur s'incrémente à chaque contact de lame (`0/N → N/N`), pas au nombre de morceaux | Compteur piloté par le nombre de parties de mesh validées après chaque coupe |
| Pas d'augmentation progressive | Coupes parallèles = N+1 tranches, croissance linéaire uniforme | Paliers configurables, chaque coup peut couper plusieurs morceaux (jusqu'à ×2) dans une nouvelle direction |
| Mangue peu crédible | Ellipsoïde/œuf + texture orange | Forme paramétrique réniforme mesurée (13,5 × 9,7 × 8,3 cm), peau multi-zones + lenticelles, chair fibreuse humide, liseré de peau sur les coupes |
| Son mécanique | Un même « chop » par coup | 6 couches par coupe, aléa contrôlé par seed, volume proportionnel à la taille du morceau, panoramique, petite réverb |

## 2. Géométrie

- **Planification** sur grille voxel (72³ à 128³ selon N) de la *même* forme que le mesh. Chaque split est vérifié en connectivité 26 → si un plan crée 3 parties (objet légèrement concave), le compteur passe de +2, jamais d'hypothèse « 1 coup = +1 ».
- **Plan par morceau** : passe par le centroïde du morceau avec la normale du coup ; si cela donnerait une lamelle (< 22 % d'un côté), repli sur l'axe principal du morceau (compté dans `fallback_axis_cuts`).
- **Blender** : `bmesh.ops.bisect_plane` sur une copie du parent (côté négatif puis positif) → `holes_fill` (capuchon = chair) → `inset_region` (liseré de peau 1,6 mm) → séparation des îlots → validation (arêtes non-manifold, volume signé). Les parties Blender sont appariées aux enfants planifiés par centroïde le plus proche ; les écarts sont rapportés (`mismatch_hits`, `failures`).
- **Séparation** : explosion radiale *horizontale* des centroïdes (facteur croissant avec log N). Pour une partition convexe, translater chaque morceau proportionnellement à son centroïde conserve la séparation → pas d'interpénétration ; les morceaux restent posés (z constant), rien ne flotte.
- **Noyau** : stratégie explicite `seedless_flesh` (chair pleine, stylisation courante des vidéos ASMR). Un noyau découpable est une évolution possible (voir §5).

## 3. Mesures (sandbox Linux, sans Blender — à refaire sur la machine cible)

| Morceaux | Coups de lame | Plans de coupe | Morceaux attendus | Planif. | Mémoire planif. | Audio | Durée vidéo |
|---|---|---|---|---|---|---|---|
| 15 | 7 | 14 | 15 | 0,1 s | 84 Mo | <1 s | 14,3 s |
| 100 | 10 | 98 | 100 | 0,4 s | 112 Mo | ~1 s | 23,9 s |
| 400 | 12 | 399 | 400 | 1,3 s | 178 Mo | 3,5 s | 45,8 s |

Le temps Blender (construction + rendu) dépend du PC : `python scripts/run.py bench 15,50,100,400` produit le tableau complet (objets, échecs, non-manifold, s/frame, taille .blend). Ordre de grandeur attendu : quelques secondes de construction pour 400 morceaux (~800 objets, ~100 k sommets), rendu EEVEE quelques dixièmes de seconde par frame en 540×960.

## 4. Limites connues

- La lame est un objet rigide qui traverse le morceau ; la couture n'est pas « ouverte » progressivement pendant la traversée (les enfants apparaissent à la sortie de lame, 2–5 frames).
- Une lame longue peut frôler visuellement un morceau voisin aligné dans le même plan pendant la traversée (le manche est orienté vers l'extérieur pour limiter ce cas).
- Pas de simulation rigid-body : les morceaux glissent sur la planche (animation procédurale). Plus stable et reproductible, moins chaotique qu'une vraie chute.
- Son 100 % procédural par défaut : synchronisé et sans clipping (vérifié), mais son réalisme doit être jugé à l'oreille ; des échantillons réels peuvent remplacer chaque couche.
- `video_duration` n'est pas imposé strictement : la durée découle du nombre de coups et de `cutting_speed`.

## 5. Pistes d'optimisation

- Au-delà de ~1 000 morceaux : décimer les meshes des morceaux minuscules, instancier les compteurs, rendre les phases denses en demi-résolution puis upscaler.
- Rigid-body optionnel uniquement pour la chute finale (bake après la dernière coupe).
- Noyau : second volume intérieur (fibreux, blanc-jaune) coupé avec les mêmes plans, matériau `pit` sur les capuchons qui le traversent.
- Ouverture progressive de la couture : booléen animé d'une fente fine suivant la lame.

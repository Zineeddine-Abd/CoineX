# CoineX — Comptage automatique de pièces de monnaie

Projet de traitement d'image réalisé dans le cadre du cours de traitement d'image (Licence 3).  
**Objectif** : compter automatiquement le nombre de pièces de monnaie présentes dans une photographie, sans utiliser de deep learning, uniquement avec des méthodes classiques vues en cours.

---

## Table des matières

1. [Description du projet](#1-description-du-projet)
2. [Lien avec le cours](#2-lien-avec-le-cours)
3. [Installation](#3-installation)
4. [Structure des fichiers](#4-structure-des-fichiers)
5. [Format des données](#5-format-des-données)
6. [Exécution détaillée de chaque script](#6-exécution-détaillée-de-chaque-script)
   - [main.py](#61-mainpy--point-dentrée-principal)
   - [evaluation.py](#62-evaluationpy--évaluation-sur-un-dataset)
   - [traitement.py](#63-traitementpy--fonctions-utilisables-seules)
   - [visualizer.py](#64-visualizerpy--visualiser-le-pipeline-étape-par-étape)
   - [preparer_dataset.py](#65-preparer_datasetpy--préparer-un-nouveau-dataset)
7. [Pipeline de traitement (étape par étape)](#7-pipeline-de-traitement-étape-par-étape)
8. [Explication détaillée de chaque concept](#8-explication-détaillée-de-chaque-concept)
   - [Représentation d'une image](#81-représentation-dune-image)
   - [Espaces couleur : RGB, HSL, niveaux de gris](#82-espaces-couleur--rgb-hsl-niveaux-de-gris)
   - [Redimensionnement bilinéaire](#83-redimensionnement-bilinéaire)
   - [Flou gaussien et convolution](#84-flou-gaussien-et-convolution)
   - [Zero padding](#85-zero-padding)
   - [Algorithme d'Otsu](#86-algorithme-dotsu)
   - [Morphologie binaire](#87-morphologie-binaire)
   - [Composantes connexes et BFS](#88-composantes-connexes-et-bfs)
   - [Descripteurs de forme](#89-descripteurs-de-forme)
   - [Soustraction du fond](#810-soustraction-du-fond)
   - [Métriques d'évaluation : MAE et MSE](#811-métriques-dévaluation--mae-et-mse)
9. [Hyperparamètres et réglages](#9-hyperparamètres-et-réglages)
10. [Évaluation et interprétation des résultats](#10-évaluation-et-interprétation-des-résultats)
11. [Cas difficiles gérés](#11-cas-difficiles-gérés)

---

## 1. Description du projet

CoineX prend en entrée une image (JPEG/PNG) contenant des pièces de monnaie posées sur une surface, et retourne un entier : le nombre de pièces détectées.

```
[image.jpg]  -->  compter_pieces()  -->  7
```

Le programme fonctionne entièrement depuis zéro (from scratch) : aucune bibliothèque de vision comme OpenCV n'est utilisée pour les calculs principaux. Seules `numpy` et `matplotlib` sont utilisées, ce qui rend chaque étape directement traçable aux formules du cours.

---

## 2. Lien avec le cours

Chaque étape du pipeline correspond à une ou plusieurs semaines du cours :

| Semaine | Notion du cours                        | Où c'est utilisé dans le code                          |
|---------|----------------------------------------|--------------------------------------------------------|
| S1-S2   | Représentation des images, pixels      | `lire_image_rgb()` — lecture, normalisation, redim.    |
| S3      | Espaces couleur (RGB, HSL, niveaux gris) | `rgb_vers_hsl()`, `rgb_vers_gris()`                  |
| S5      | Seuillage, algorithme d'Otsu           | `seuil_otsu()`, `histogramme_u8()`                     |
| S6      | Morphologie binaire                    | `erosion_binaire()`, `dilatation_binaire()`, `ouverture_binaire()`, `fermeture_binaire()` |
| S7      | Composantes connexes, descripteurs     | `composantes_connexes()`, `extraire_composantes_utiles()`, `estimer_nombre_depuis_composantes()` |
| S8      | Convolution, filtrage gaussien         | `noyau_gaussien_1d()`, `convolution_1d_lignes()`, `convolution_1d_colonnes()`, `flou_gaussien()` |
| S9      | Padding (zero padding)                 | `mode='same'` dans `np.convolve()`                     |
| S10     | Opérations morphologiques composées    | `fermeture_binaire()` — bouche les trous de reflets    |
| S10     | Métriques d'évaluation (MAE, MSE)      | `evaluation.py` — `evaluer_modele()`                   |

---

## 3. Installation

### Prérequis

- Python 3.8 ou supérieur
- pip

### Étapes

```bash
# 1. Se placer dans le dossier du projet
cd CoineX

# 2. (Optionnel mais recommandé) Créer un environnement virtuel
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows

# 3. Installer les dépendances
pip install -r requirements.txt
```

### Dépendances (`requirements.txt`)

```
numpy          # calculs matriciels (convolution, morphologie, etc.)
matplotlib     # lecture des images (matplotlib.image.imread)
pandas         # (optionnel, utilisé dans preparer_dataset.py)
opencv-python  # (optionnel, utilisé dans preparer_dataset.py pour l'annotation)
```

> Le pipeline principal (`traitement.py`) n'utilise que `numpy` et `matplotlib`.

---

## 4. Structure des fichiers

```
CoineX/
│
├── traitement.py          # Pipeline principal — toute la logique de détection
├── evaluation.py          # Calcul des métriques MAE et MSE sur un dataset
├── main.py                # Point d'entrée : lance la validation (et optionnellement le test)
├── visualizer.py          # Outil de visualisation des étapes du pipeline (8 sous-graphiques)
├── preparer_dataset.py    # Script de préparation/annotation d'un nouveau dataset
│
├── data/
│   ├── validation/        # Images du jeu de validation (img_001.jpg ... img_140.jpg)
│   ├── validation.json    # Vérité terrain pour la validation
│   ├── test/              # Images du jeu de test (ne pas regarder pendant le développement)
│   └── test.json          # Vérité terrain pour le test (règle d'or)
│
└── requirements.txt       # Dépendances Python
```

---

## 5. Format des données

### Images

Les images sont des fichiers JPEG ou PNG classiques placés dans `data/validation/` ou `data/test/`.  
Elles peuvent être de tailles variées — le pipeline les redimensionne automatiquement si elles dépassent 520 pixels sur le plus grand côté.

### Fichier JSON (vérité terrain)

Chaque dataset est accompagné d'un fichier JSON qui associe le nom de l'image au nombre réel de pièces :

```json
{
    "img_001.jpg": 2,
    "img_002.jpg": 10,
    "img_003.jpg": 16
}
```

- **Clé** : nom du fichier image (sans chemin)
- **Valeur** : nombre de pièces sur cette image (entier)

Le jeu de **validation** (`validation.json`) contient 140 images. Il sert à régler les hyperparamètres.  
Le jeu de **test** (`test.json`) ne doit être utilisé qu'une seule fois, à la fin, pour évaluer les performances réelles (règle d'or).

---

## 6. Exécution détaillée de chaque script

### 6.1 `main.py` — Point d'entrée principal

C'est le script à lancer au quotidien pour évaluer les performances de l'algorithme.

**Commande :**

```bash
python main.py
```

**Ce que ça fait, étape par étape :**

1. Importe `evaluer_modele` depuis `evaluation.py`
2. Définit les chemins vers les données (`data/validation/` et `data/validation.json`)
3. Lance l'évaluation complète sur les 140 images de validation
4. Affiche chaque résultat image par image, puis les métriques globales (MAE, MSE)

**Contenu du fichier :**

```python
from evaluation import evaluer_modele

def main():
    DOSSIER_VALIDATION = "data/validation"
    JSON_VALIDATION    = "data/validation.json"

    DOSSIER_TEST = "data/test"
    JSON_TEST    = "data/test.json"

    # Phase 1 : Validation (toujours active)
    print("PHASE DE VALIDATION (Réglage des hyperparamètres)")
    meilleur_flou = (7, 7)
    evaluer_modele(DOSSIER_VALIDATION, JSON_VALIDATION, taille_flou=meilleur_flou)

    # Phase 2 : Test (commentée — à décommenter une seule fois en fin de projet)
    # print("\n\nPHASE DE TEST (Évaluation finale)")
    # evaluer_modele(DOSSIER_TEST, JSON_TEST, taille_flou=meilleur_flou)
```

**Pour tester une autre valeur de flou :**

Changer `meilleur_flou = (7, 7)` par exemple en `(5, 5)` ou `(11, 11)`, puis relancer :

```bash
python main.py
```

**Pour activer le test final (une seule fois) :**

Décommenter les deux lignes de la phase de test, puis relancer :

```bash
python main.py
```

> **Règle absolue** : ne décommenter la phase de test qu'une seule fois, quand tous les hyperparamètres sont définitivement fixés.

---

### 6.2 `evaluation.py` — Évaluation sur un dataset

Ce fichier contient la fonction `evaluer_modele()`. Vous pouvez l'utiliser directement dans un script Python sans passer par `main.py`.

**Signature de la fonction :**

```python
evaluer_modele(dossier_images, fichier_json, taille_flou=(7, 7))
```

| Paramètre       | Type          | Description                                         |
|-----------------|---------------|-----------------------------------------------------|
| `dossier_images`| `str`         | Chemin vers le dossier contenant les images JPEG/PNG |
| `fichier_json`  | `str`         | Chemin vers le fichier JSON de vérité terrain       |
| `taille_flou`   | `tuple (int, int)` | Taille du noyau gaussien, ex: `(7, 7)` (défaut) |

**Exemple d'utilisation directe :**

```python
from evaluation import evaluer_modele

# Évaluation sur la validation avec un flou de 5×5
evaluer_modele("data/validation", "data/validation.json", taille_flou=(5, 5))

# Évaluation sur le test (uniquement en fin de projet)
evaluer_modele("data/test", "data/test.json", taille_flou=(7, 7))
```

**Ce que la fonction fait en interne :**

```
1. Ouvre le fichier JSON → charge le dictionnaire {nom_image: vrai_nombre}
2. Pour chaque image dans le JSON :
   a. Construit le chemin complet : dossier_images + "/" + nom_image
   b. Appelle compter_pieces(chemin, taille_flou)
   c. Calcule la différence : prédiction - vrai_nombre
   d. Accumule les erreurs absolues et quadratiques
   e. Affiche [OK] ou [ERREUR] avec le détail
3. Calcule et affiche MAE, MSE, et le pourcentage d'erreur
```

**Sortie typique :**

```
Évaluation sur le dataset : data/validation.json
----------------------------------------
[OK]     img_001.jpg | Prédit: 2  | Réel: 2
[ERREUR] img_002.jpg | Prédit: 9  | Réel: 10 | Diff: -1
[OK]     img_003.jpg | Prédit: 16 | Réel: 16
...
----------------------------------------
MAE (Erreur Absolue Moyenne)     : 1.23
MSE (Erreur Quadratique Moyenne) : 3.45
Nombre Moyen Réel de Pièces      : 4.12
Pourcentage d'Erreur (MAE/Mean)  : 29.85%
----------------------------------------
```

---

### 6.3 `traitement.py` — Fonctions utilisables seules

`traitement.py` est le cœur du projet. Il exporte toutes les fonctions du pipeline. Chaque fonction peut être utilisée indépendamment pour tester ou déboguer une étape précise.

**Importer et utiliser la fonction principale :**

```python
from traitement import compter_pieces

# Cas le plus simple : une image, le résultat
resultat = compter_pieces("data/validation/img_001.jpg")
print(resultat)   # ex: 2

# Avec un noyau de flou plus grand (plus de lissage)
resultat = compter_pieces("data/validation/img_001.jpg", taille_flou=(11, 11))
```

**Tester chaque étape du pipeline séparément :**

```python
from traitement import (
    lire_image_rgb,
    rgb_vers_hsl,
    rgb_vers_gris,
    flou_gaussien,
    seuil_otsu,
    ouverture_binaire,
    fermeture_binaire,
    composantes_connexes,
)
import matplotlib.pyplot as plt

# 1. Charger l'image
image = lire_image_rgb("data/validation/img_001.jpg")
print("Taille de l'image :", image.shape)   # ex: (400, 520, 3)

# 2. Extraire la saturation
luminosite, saturation = rgb_vers_hsl(image)
print("Min saturation :", saturation.min(), "Max :", saturation.max())

# 3. Flouter la saturation
saturation_floue = flou_gaussien(saturation, taille=7)

# 4. Calculer le seuil d'Otsu
seuil = seuil_otsu(saturation_floue)
print("Seuil Otsu :", round(seuil, 3))   # ex: 0.312

# 5. Binariser
masque = saturation_floue > seuil

# 6. Nettoyage morphologique
masque = ouverture_binaire(masque, taille=3)
masque = fermeture_binaire(masque, taille=3)

# 7. Composantes connexes
composantes = composantes_connexes(masque)
print(f"Nombre de composantes trouvées : {len(composantes)}")
for c in composantes:
    print(f"  aire={c['area']}, circularité={c['circularite']:.2f}, remplissage={c['remplissage']:.2f}")

# 8. Afficher le masque binaire
plt.imshow(masque, cmap='gray')
plt.title("Masque binaire après morphologie")
plt.show()
```

**Tester la voie alternative (niveaux de gris) :**

```python
from traitement import lire_image_rgb, rgb_vers_gris
import numpy as np
import matplotlib.pyplot as plt

image = lire_image_rgb("data/validation/img_001.jpg")
gris = rgb_vers_gris(image)

# Estimer le fond (médiane des bords)
marge = 15
bords = np.concatenate([
    gris[:marge, :].ravel(),
    gris[-marge:, :].ravel(),
    gris[:, :marge].ravel(),
    gris[:, -marge:].ravel(),
])
fond = float(np.median(bords))
print("Fond estimé :", round(fond, 3))

# Soustraction du fond
difference = np.abs(gris - fond)

plt.imshow(difference, cmap='hot')
plt.title("Différence avec le fond")
plt.colorbar()
plt.show()
```

---

### 6.4 `visualizer.py` — Visualiser le pipeline étape par étape

Ce script affiche une grille de 8 sous-graphiques montrant chaque transformation appliquée à l'image.

**Commande en ligne (la plus simple) :**

```bash
python visualizer.py data/validation/img_001.jpg
```

Remplacer `img_001.jpg` par n'importe quelle image du dataset.

**Utilisation depuis un script Python :**

```python
from visualizer import visualiser_pipeline

# Afficher dans une fenêtre interactive
visualiser_pipeline("data/validation/img_001.jpg")

# Avec un flou différent
visualiser_pipeline("data/validation/img_001.jpg", taille_flou=(11, 11))

# Sauvegarder la figure dans un fichier au lieu de l'afficher
visualiser_pipeline(
    "data/validation/img_001.jpg",
    taille_flou=(7, 7),
    save_path="resultats/pipeline_img001.png"
)
```

**Ce que chaque sous-graphique montre :**

| N° | Titre dans la figure              | Ce qu'on voit                                                   |
|----|-----------------------------------|-----------------------------------------------------------------|
| 1  | Image Originale (RGB)             | La photo brute, telle que lue depuis le disque                 |
| 2  | Saturation (HSL)                  | Canal S de HSL : pièces = zones claires, fond = zones sombres  |
| 3  | Gaussian Blur                     | Saturation après flou gaussien : bords adoucis, bruit réduit   |
| 4  | Seuillage Otsu                    | Masque binaire : blanc = pièce, noir = fond                    |
| 5  | Ouverture Morphologique           | Masque nettoyé : petites taches parasites supprimées           |
| 6  | Composantes Connexes              | Chaque région détectée colorée d'une couleur différente        |
| 7  | Descripteurs de Forme             | Bounding boxes vertes (pièce de référence) ou rouges (rejeté) |
| 8  | Résultat Final                    | Tableau récapitulatif : prédiction + hyperparamètres actifs    |

**Visualiser seulement la distribution des descripteurs :**

```python
from visualizer import visualiser_descripteurs

# Affiche 4 histogrammes : aire, circularité, remplissage, ratio
visualiser_descripteurs("data/validation/img_001.jpg")
```

---

### 6.5 `preparer_dataset.py` — Préparer un nouveau dataset

Ce script s'utilise **une seule fois** quand on veut créer un nouveau dataset depuis des images brutes. Il n'est pas nécessaire si les dossiers `data/validation/` et `data/test/` existent déjà.

**Ce que fait le script :**

1. `preparer_et_diviser()` : prend un dossier d'images brutes, les mélange aléatoirement, les renomme en `img_001.jpg`, `img_002.jpg`, etc., puis les divise en 70% validation / 30% test.
2. `annoter_dossier()` : affiche chaque image (avec OpenCV) et demande à l'utilisateur de taper le nombre de pièces visible. Sauvegarde le résultat dans un JSON.

**Commande :**

```bash
python preparer_dataset.py
```

**Ce que le script exécute automatiquement :**

```python
# Étape 1 : Diviser les images brutes
preparer_et_diviser("data_brute", "data/validation", "data/test")
# → Copie et renomme les images depuis "data_brute/"
# → 70% dans data/validation/, 30% dans data/test/

# Étape 2 : Annoter manuellement image par image
annoter_dossier("data/validation", "data/validation.json")
# → Affiche chaque image, tape le nombre dans la console
# → Sauvegarde dans data/validation.json

annoter_dossier("data/test", "data/test.json")
# → Même chose pour le jeu de test
```

**Interaction pendant l'annotation :**

```
Image img_001.jpg - Nombre de pièces ? : 2        ← taper un entier + Entrée
Image img_002.jpg - Nombre de pièces ? : 10
Image img_003.jpg - Nombre de pièces ? : q        ← taper 'q' pour quitter et sauvegarder
```

> Ce script nécessite OpenCV (`opencv-python`) pour afficher les images pendant l'annotation. Il n'est pas nécessaire pour faire tourner le pipeline de détection.

---

## 7. Pipeline de traitement (étape par étape)

Voici ce que fait `compter_pieces()` à l'intérieur, dans l'ordre :

```
Image JPEG/PNG
     |
     v
[1] lire_image_rgb()
     Lit l'image, normalise en uint8 RGB (3 canaux),
     redimensionne si > 520px (interpolation bilinéaire).
     |
     v
[2] rgb_vers_hsl()
     Convertit RGB → HSL, extrait le canal Saturation.
     La saturation est robuste aux variations de luminosité
     (ombres, reflets de lumière ambiante).
     |
     v
[3] flou_gaussien()
     Applique un flou gaussien sur la saturation.
     Réduit le bruit et les petites variations avant le seuillage.
     Implémenté avec 2 convolutions 1D séparables (horizontale puis verticale).
     |
     v
[4] seuil_otsu()
     Calcule automatiquement le meilleur seuil de binarisation.
     Maximise la variance entre les 2 classes : fond vs pièces.
     Pas de paramètre à régler à la main.
     |
     v
[5] ouverture_binaire()   (Érosion → Dilatation)
     Supprime les petits points blancs parasites dans le masque.
     |
     v
[6] fermeture_binaire()   (Dilatation → Érosion)
     Bouche les petits trous noirs à l'intérieur des pièces
     (causés par les reflets brillants au centre des pièces).
     |
     v
[7] composantes_connexes()
     Identifie chaque région blanche connectée dans le masque.
     Algorithme BFS (parcours en largeur), 8-connexité.
     Calcule pour chaque région : aire, circularité, remplissage, bbox.
     |
     v
[8] extraire_composantes_utiles()
     Filtre : ne garde que les composantes dont l'aire est plausible
     (ni trop petite = bruit, ni trop grande = fond mal segmenté)
     et qui ne touchent pas le bord de l'image.
     |
     v
[9] estimer_nombre_depuis_composantes()
     Compte les pièces en gérant le cas des pièces collées :
     si une composante a une aire ≈ 2× l'aire typique, on compte 2.
     |
     v
[10] detection_piece_unique()  (filet de sécurité — voie alternative)
     Voie alternative basée sur la soustraction du fond en niveaux de gris.
     Vérifie si l'image contient probablement exactement 1 pièce.
     |
     v
[11] Règles correctives finales
     Si la prédiction est aberrante (ex: 5 alors qu'on voit 1 grosse forme ronde),
     on corrige la valeur finale.
     |
     v
Résultat : nombre de pièces (entier)
```

Le pipeline utilise **deux voies complémentaires** :

| Voie                    | Entrée         | Méthode                       | Quand elle est utile                        |
|-------------------------|----------------|-------------------------------|---------------------------------------------|
| Principale (saturation) | Canal S de HSL | Otsu + morphologie + BFS      | La plupart des images (fond terne + pièces colorées) |
| Secondaire (gris)       | Niveaux de gris| Soustraction du fond + Otsu   | Cas d'une seule pièce mal segmentée par la voie principale |

---

## 8. Explication détaillée de chaque concept

### 8.1 Représentation d'une image

**Qu'est-ce qu'une image numérique ?**

Une image est un tableau 2D (ou 3D pour les couleurs) de nombres entiers appelés pixels.

```
Image en niveaux de gris (2D) :
┌─────────────────────────┐
│  12  45  78  200  255   │  ← chaque nombre = intensité d'un pixel (0 à 255)
│   0  30  90  150  100   │
│  55  80 120  180   60   │
└─────────────────────────┘
Shape : (3 lignes, 5 colonnes) = (3, 5)

Image couleur RGB (3D) :
Shape : (hauteur, largeur, 3 canaux)
         ↑              ↑  Rouge, Vert, Bleu
```

- **0** = noir / sombre
- **255** = blanc / très clair
- **3 canaux RGB** : un nombre pour le rouge, un pour le vert, un pour le bleu

**Normalisation dans notre code :**

Après la lecture, les pixels sont stockés en `uint8` (entiers 0 à 255) pour les opérations couleur, puis convertis en `float32` entre 0.0 et 1.0 pour les calculs mathématiques.

**Où c'est dans le code :** `lire_image_rgb()` dans `traitement.py`

---

### 8.2 Espaces couleur : RGB, HSL, niveaux de gris

#### RGB (Red, Green, Blue)

C'est la représentation native des écrans et des photos numériques. Chaque pixel = 3 valeurs (R, G, B) entre 0 et 255. C'est additif : R+G+B maximal = blanc.

**Problème avec RGB :** Si on change l'éclairage (ombre, lumière directe), les 3 valeurs changent toutes en même temps, ce qui rend la détection difficile.

#### HSL (Hue, Saturation, Lightness)

HSL sépare ce que l'œil perçoit :
- **H (teinte)** : la couleur en tant que telle (rouge, jaune, bleu...)
- **S (saturation)** : l'intensité de la couleur (gris pâle = 0, couleur vive = 1)
- **L (luminosité)** : clair ou sombre (noir = 0, blanc = 1)

**Pourquoi la saturation est utile ici :**

```
Fond (tissu gris)      → saturation ≈ 0.05  (presque pas de couleur)
Pièce de 1€ (or)      → saturation ≈ 0.45  (plus colorée)
Pièce de 50cts (cuivre) → saturation ≈ 0.60  (encore plus colorée)
```

La saturation fait ressortir les pièces par rapport au fond même si la lumière change, parce que la luminosité change mais la saturation, elle, reste stable.

**Formule mathématique (vue en cours) :**

```
delta = max(R,G,B) - min(R,G,B)
L = (max + min) / 2
S = delta / (1 - |2L - 1|)
```

**Où c'est dans le code :** `rgb_vers_hsl()` dans `traitement.py`

#### Niveaux de gris

Réduit les 3 canaux RGB à une seule valeur par pixel avec la formule de luminance :

```
Gray = 0.299×R + 0.587×G + 0.114×B
```

Les coefficients ne sont pas (1/3, 1/3, 1/3) parce que l'œil humain est plus sensible au vert qu'au rouge, et moins au bleu. Ces poids simulent cette sensibilité.

**Où c'est dans le code :** `rgb_vers_gris()` dans `traitement.py`

---

### 8.3 Redimensionnement bilinéaire

Quand une image est trop grande (> 520px), on la réduit. Pour cela, il faut "inventer" les valeurs des nouveaux pixels.

**Interpolation bilinéaire :** pour chaque pixel de la nouvelle image, on regarde les 4 pixels les plus proches dans l'image originale et on fait une moyenne pondérée selon la distance.

```
Image originale (4×4)       Image réduite (2×2)
┌──┬──┬──┬──┐               ┌──────┬──────┐
│10│20│30│40│               │ ???  │ ???  │   ← chaque pixel est une
├──┼──┼──┼──┤    →          ├──────┼──────┤     moyenne pondérée des
│15│25│35│45│               │ ???  │ ???  │     4 voisins dans l'original
├──┼──┼──┼──┤               └──────┴──────┘
│20│30│40│50│
└──┴──┴──┴──┘
```

**Pourquoi pas "plus proche voisin" ?**  
Le plus proche voisin choisit brutalement un pixel → rendu pixelisé avec effet d'escalier. La bilinéaire fait une transition douce → rendu plus naturel.

**Où c'est dans le code :** `redimensionner_bilineaire()` dans `traitement.py`

---

### 8.4 Flou gaussien et convolution

#### Qu'est-ce qu'une convolution ?

La convolution remplace chaque pixel par une combinaison linéaire de ses voisins, pondérée par un noyau (kernel).

```
Ligne de pixels : [10, 20, 30, 40, 50]
Noyau gaussien  : [0.25, 0.50, 0.25]   (somme = 1)

Pour le pixel central (30) :
nouvelle_valeur = 20×0.25 + 30×0.50 + 40×0.25
               = 5 + 15 + 10
               = 30   (ici c'est au centre donc peu de changement)
```

#### La gaussienne comme noyau

La gaussienne donne plus de poids aux voisins proches et moins aux voisins éloignés. C'est la forme naturelle du flou dans les appareils photo.

```
Formule : G(x) = exp(-x² / (2σ²))   [normalisée pour que la somme = 1]

Exemple de noyau 1D avec σ=1 et taille=5 :
[0.06, 0.24, 0.40, 0.24, 0.06]
   ↑                       ↑
peu de poids             peu de poids
(voisins lointains)      (voisins lointains)
```

#### Convolution séparable

Une gaussienne 2D peut être décomposée en deux convolutions 1D (horizontale puis verticale). Résultat identique, mais beaucoup plus rapide.

```
Flou 2D = Flou horizontal (ligne par ligne) → puis Flou vertical (colonne par colonne)
```

**Pourquoi flouter avant le seuillage ?**

Le seuillage est sensible au bruit : un tout petit pixel brillant peut être classé comme "pièce". Le flou lisse ces variations locales avant de binariser.

**Où c'est dans le code :** `noyau_gaussien_1d()`, `convolution_1d_lignes()`, `convolution_1d_colonnes()`, `flou_gaussien()` dans `traitement.py`

---

### 8.5 Zero padding

Quand on applique un noyau de taille 5 sur le bord d'une image, il y a un problème : les pixels imaginaires en dehors de l'image n'existent pas.

**Solution — Zero padding :** on imagine que les pixels en dehors de l'image valent 0.

```
Image originale (1D) : [10, 20, 30, 40, 50]
Avec zero padding    : [ 0,  0, 10, 20, 30, 40, 50,  0,  0]
                          ↑  ↑                        ↑  ↑
                      pixels imaginaires = 0       pixels imaginaires = 0
```

**Conséquence :** les pixels au bord de l'image seront légèrement assombris (ils intègrent des zéros dans leur moyenne). Ce n'est pas un problème ici car les composantes connexes qui touchent le bord sont rejetées.

**Dans notre code :** `np.convolve(..., mode='same')` applique le zero padding implicitement.

---

### 8.6 Algorithme d'Otsu

**Problème :** on veut binariser l'image (pixel = blanc si c'est une pièce, noir si c'est le fond). Mais quel seuil choisir ?

**Solution d'Otsu :** trouver automatiquement le seuil `t` qui sépare au mieux deux classes.

**Idée intuitive :**

```
Histogramme de saturation :

Nombre   |
de       |   ██         ██
pixels   |   ██   ·   · ██
         |  ███ ·     · ███
         └─────────────────→ intensité (0 à 1)
              ↑       ↑
             fond    pièces
              └───────┘
                 ↑
            Meilleur seuil = là où les deux pics sont les mieux séparés
```

**Formule mathématique :**

```
Pour chaque seuil t possible :
  w0(t) = proportion de pixels sous t       (fraction du fond)
  w1(t) = proportion de pixels au-dessus t  (fraction des pièces)
  μ0(t) = intensité moyenne du fond
  μ1(t) = intensité moyenne des pièces

  σ²_B(t) = w0 × w1 × (μ0 - μ1)²   ← variance inter-classes

Choisir t qui maximise σ²_B(t)
```

Plus les deux groupes sont éloignés et équilibrés, plus la variance inter-classes est grande.

**Pourquoi Otsu plutôt que K-Means ?**

Otsu est **optimal** pour exactement 2 classes : il teste tous les seuils possibles (0 à 255) et garantit le meilleur. K-Means pour k=2 dépend de son initialisation aléatoire et peut converger vers un mauvais résultat.

**Où c'est dans le code :** `histogramme_u8()` + `seuil_otsu()` dans `traitement.py`

---

### 8.7 Morphologie binaire

Après le seuillage, le masque binaire n'est pas parfait : il y a des petits points parasites et des trous. La morphologie sert à nettoyer ça.

#### Érosion

Un pixel blanc reste blanc **seulement si tous ses voisins** dans une fenêtre de taille `t` sont blancs.

```
Avant érosion :    Après érosion (taille 3) :
1 1 1 1 1          0 1 1 1 0
1 1 1 1 1          1 1 1 1 1
1 1 0 1 1    →     1 0 0 0 1   ← trou agrandi
1 1 1 1 1          1 1 1 1 1
1 1 1 1 1          0 1 1 1 0

Effet : rétrécit les objets, supprime les petits points isolés.
```

#### Dilatation

Un pixel noir devient blanc **si au moins un de ses voisins** dans la fenêtre est blanc.

```
Effet : agrandit les objets, referme les petits trous.
```

#### Ouverture = Érosion puis Dilatation

```
Ouverture(masque) = Dilatation( Érosion(masque) )

Utilité : Supprime les petits points parasites SANS trop modifier les grandes formes.
          Un grain de poussière blanc (3 pixels) disparaît.
          Une pièce (5000 pixels) reste quasi intacte.
```

#### Fermeture = Dilatation puis Érosion

```
Fermeture(masque) = Érosion( Dilatation(masque) )

Utilité : Bouche les petits trous SANS trop modifier les contours.
          Un reflet brillant au centre d'une pièce crée un trou → la fermeture le referme.
```

**Ordre dans notre pipeline :**

```
Ouverture d'abord → nettoie le bruit extérieur (petites taches)
Fermeture ensuite → bouche les trous intérieurs (reflets des pièces)
```

**Où c'est dans le code :** `erosion_binaire()`, `dilatation_binaire()`, `ouverture_binaire()`, `fermeture_binaire()` dans `traitement.py`

---

### 8.8 Composantes connexes et BFS

**Définition :** une composante connexe est un ensemble maximal de pixels blancs qui se touchent tous entre eux.

```
Masque binaire :               Composantes connexes détectées :
0 0 0 0 0 0 0 0                0 0 0 0 0 0 0 0
0 1 1 1 0 0 0 0                0 A A A 0 0 0 0
0 1 1 0 0 0 0 0    →           0 A A 0 0 0 0 0   (composante A)
0 0 0 0 0 1 1 0                0 0 0 0 0 B B 0   (composante B)
0 0 0 0 0 1 0 0                0 0 0 0 0 B 0 0
```

**Algorithme BFS (Breadth-First Search — parcours en largeur) :**

```
Pour chaque pixel blanc non encore étiqueté :
  1. Créer une nouvelle étiquette (couleur)
  2. Ajouter ce pixel dans une file (deque)
  3. Tant que la file n'est pas vide :
     a. Prendre le premier pixel de la file
     b. L'étiqueter
     c. Ajouter tous ses voisins blancs non étiquetés dans la file
  4. Tous les pixels explorés = une composante
```

**8-connexité vs 4-connexité :**

Avec la 4-connexité (haut/bas/gauche/droite seulement), ces deux pixels ne seraient **pas** dans la même composante :

```
0 1
1 0   ← les deux "1" sont en diagonale : 4-connexité = 2 composantes ≠ 8-connexité = 1 composante
```

On utilise la **8-connexité** (diagonales incluses) parce que les bords courbes des pièces peuvent se connecter par diagonale après le seuillage.

**Où c'est dans le code :** `composantes_connexes()` dans `traitement.py`

---

### 8.9 Descripteurs de forme

Une fois les composantes extraites, on calcule 4 descripteurs pour chacune :

#### Aire

```
Aire = nombre de pixels blancs dans la composante
```

Permet d'éliminer les très petites composantes (bruit) et les trop grandes (fond mal segmenté).

#### Boîte englobante (Bounding Box)

```
bbox = (y_min, x_min, y_max, x_max)
     = le plus petit rectangle qui contient tous les pixels de la composante
```

#### Remplissage (Fill ratio)

```
Remplissage = Aire / (hauteur_bbox × largeur_bbox)
```

Une pièce ronde a un remplissage ≈ 0.78 (π/4, un cercle dans un carré).  
Un objet allongé ou irrégulier a un remplissage plus faible.

#### Circularité

```
Circularité = 4π × Aire / Périmètre²

Cercle parfait  → circularité = 1.0
Carré           → circularité ≈ 0.785
Objet allongé   → circularité << 1
```

C'est le critère le plus discriminant : une pièce est ronde, donc sa circularité devrait être proche de 1.

#### Ratio hauteur/largeur (Aspect ratio)

```
Ratio = hauteur_bbox / largeur_bbox

Cercle vu de face → ratio ≈ 1.0
Cercle vu en angle → ratio peut s'éloigner de 1.0
```

**Où c'est dans le code :** calculé dans `composantes_connexes()`, utilisé dans `estimer_nombre_depuis_composantes()` et `detection_piece_unique()`.

---

### 8.10 Soustraction du fond

Utilisée dans la voie secondaire (`detection_piece_unique()`).

**Idée :** si le fond est uniforme, soustraire sa couleur fait ressortir les objets qui diffèrent.

**Estimation du fond :** on prend la médiane des pixels sur les bordures de l'image (on suppose que le fond est visible sur les bords).

```
Exemple :
  fond estimé = 0.72 (médiane des bords)
  
  pixel A (fond) = 0.70  →  |0.70 - 0.72| = 0.02  (faible, fond)
  pixel B (pièce) = 0.35  →  |0.35 - 0.72| = 0.37  (élevé, objet)
  pixel C (reflet) = 0.95  →  |0.95 - 0.72| = 0.23  (élevé, objet)
```

Pourquoi la **médiane** et pas la moyenne ? La médiane est robuste : si une pièce est partiellement visible sur le bord, elle ne fausse pas l'estimation du fond.

**Où c'est dans le code :** `detection_piece_unique()` dans `traitement.py`

---

### 8.11 Métriques d'évaluation : MAE et MSE

Après avoir prédit le nombre de pièces pour toutes les images, on mesure la qualité des prédictions.

Soit pour chaque image i : `y_i` = prédiction, `ŷ_i` = vrai nombre, `N` = nombre d'images.

#### MAE — Erreur Absolue Moyenne

```
MAE = (1/N) × Σ |y_i - ŷ_i|
```

**Interprétation :** en moyenne, on se trompe de MAE pièces par image.

```
Exemples :
  Prédit 3, réel 3   → erreur = 0
  Prédit 5, réel 3   → erreur = 2
  Prédit 1, réel 4   → erreur = 3
  
  MAE = (0 + 2 + 3) / 3 = 1.67 pièces d'erreur en moyenne
```

#### MSE — Erreur Quadratique Moyenne

```
MSE = (1/N) × Σ (y_i - ŷ_i)²
```

**Différence avec MAE :** le carré pénalise beaucoup plus les grosses erreurs.

```
  Erreur de 1  →  MSE contribution = 1
  Erreur de 3  →  MSE contribution = 9   (3 fois plus grande, mais 9 fois plus pénalisée)
```

Quand MSE >> MAE², cela signifie qu'on a quelques très grosses erreurs qui faussent tout.

#### Pourcentage d'erreur

```
% erreur = (MAE / Moyenne réelle) × 100
```

Permet de contextualiser : MAE=1.5 est bien si les images ont en moyenne 10 pièces (15%), mais mauvais si elles en ont en moyenne 2 (75%).

---

## 9. Hyperparamètres et réglages

Tous les seuils réglables sont définis **en haut de `traitement.py`**, regroupés par groupe logique.  
Ils ont été calibrés empiriquement sur le jeu de **validation uniquement**.

### Groupe 1 — Critères de la pièce de référence

```python
COIN_REFERENCE_CIRCULARITY_MIN = 0.45   # Circularité minimale (1.0 = cercle parfait)
COIN_REFERENCE_FILL_MIN        = 0.45   # Remplissage aire / bbox minimum
COIN_REFERENCE_ASPECT_MIN      = 0.65   # Ratio hauteur/largeur minimum
COIN_REFERENCE_ASPECT_MAX      = 1.55   # Ratio hauteur/largeur maximum
```

Ces seuils définissent ce qu'on considère comme une "vraie pièce de référence". Elle sert à estimer l'aire typique d'une pièce (médiane des aires de référence).

### Groupe 2 — Détection des pièces fusionnées

```python
MERGE_AREA_RATIO_THRESHOLD = 1.8   # Aire > 1.8× la typique → on compte 2 pièces
MERGE_FILL_MIN             = 0.35  # Remplissage minimum pour valider la fusion
```

Si deux pièces se touchent, elles forment une seule composante avec une aire double. Ce seuil corrige le sous-comptage.

### Groupe 3 — Détection "une seule pièce"

```python
SINGLE_COIN_FILL_MIN         = 0.62   # Remplissage strict
SINGLE_COIN_CIRCULARITY_MIN  = 0.40   # Circularité minimum
SINGLE_COIN_ASPECT_RATIO_MIN = 0.78   # Ratio hauteur/largeur minimum
SINGLE_COIN_ASPECT_RATIO_MAX = 1.28   # Ratio hauteur/largeur maximum
```

Critères stricts pour conclure "il y a exactement 1 pièce" via la voie secondaire.

### Groupe 4 et 5 — Règles correctives

```python
CORRECTION_LARGE_CIRCULARITY_MIN   = 0.48   # Circularité pour corriger à 1
CORRECTION_LARGE_FILL_MIN          = 0.68   # Remplissage pour corriger à 1
CORRECTION_LARGE_AREA_MULTIPLIER   = 8.0    # Doit être > 8× l'aire médiane
CORRECTION_RARE_CIRCULARITY_MIN    = 0.55   # (règle 2) circularité stricte
CORRECTION_RARE_FILL_MIN           = 0.72   # (règle 2) remplissage strict
```

Évitent les erreurs grossières comme prédire 5 pour une seule grande pièce.

### Groupe 6 — Performance système

```python
MAX_IMAGE_DIMENSION = 520   # Images > 520px sont redimensionnées avant traitement
```

### Comment ajuster les hyperparamètres

1. Lancer `main.py` et noter la MAE initiale.
2. Examiner les lignes `[ERREUR]` pour comprendre le type d'erreur (sur-comptage ou sous-comptage).
3. Ajuster un seul paramètre à la fois.
4. Relancer et comparer la MAE.
5. Ne jamais regarder ni toucher `test.json` pendant cette phase.

---

## 10. Évaluation et interprétation des résultats

### Métriques calculées

Après l'évaluation, `evaluer_modele()` affiche :

```
MAE (Erreur Absolue Moyenne)     : X.XX
MSE (Erreur Quadratique Moyenne) : X.XX
Nombre Moyen Réel de Pièces      : X.XX
Pourcentage d'Erreur (MAE/Mean)  : X.XX%
```

### Formules

Soit N le nombre d'images, `yi` la prédiction, `ŷi` la vraie valeur :

```
MAE = (1/N) × Σ |yi - ŷi|      (erreur absolue moyenne)
MSE = (1/N) × Σ (yi - ŷi)²     (erreur quadratique moyenne)
Pourcentage = MAE / Moyenne réelle × 100
```

### Comment lire ces métriques

| Métrique       | Ce qu'elle dit                                                         |
|----------------|------------------------------------------------------------------------|
| MAE            | En moyenne, on se trompe de X pièces par image (ex: MAE=1.2 → ±1.2 pièce) |
| MSE            | Comme MAE, mais pénalise plus fortement les grosses erreurs            |
| MAE/Mean %     | Erreur relative : si la moyenne est 4 pièces et MAE=1.2 → 30% d'erreur |

La **MAE** est la métrique principale à surveiller. Une MAE inférieure à 1.5 est un bon résultat pour ce type de problème avec un algorithme classique.

### Ligne par ligne dans la sortie

```
[OK]     img_001.jpg | Prédit: 2  | Réel: 2         ← prédiction correcte
[ERREUR] img_042.jpg | Prédit: 0  | Réel: 1 | Diff: -1   ← sous-comptage d'1
[ERREUR] img_076.jpg | Prédit: 15 | Réel: 13 | Diff: +2  ← sur-comptage de 2
```

---

## 11. Cas difficiles gérés

| Cas                              | Problème                                              | Solution dans le code                                  |
|----------------------------------|-------------------------------------------------------|--------------------------------------------------------|
| 2 pièces collées                 | Forment 1 seule composante → sous-comptage            | Ratio aire/aire_typique dans `estimer_nombre_depuis_composantes()` |
| Pièce avec reflet central        | Le reflet crée un trou dans le masque → sur-comptage  | `fermeture_binaire()` bouche les trous               |
| Petites taches de bruit          | Faux positifs après seuillage                         | `ouverture_binaire()` supprime les petits parasites  |
| Objet coupé par le bord          | Pièce partiellement visible                           | Filtre `touche_bord` dans `extraire_composantes_utiles()` |
| 1 seule pièce mal segmentée      | Saturation insuffisante → prédiction 0 ou 4           | `detection_piece_unique()` (voie gris + soustraction fond) |
| Image trop grande                | Temps de calcul excessif                              | Redimensionnement bilinéaire dans `lire_image_rgb()`  |

---

## Notes

- `traitement.py` est la version active, importée par `evaluation.py`.
- Toutes les valeurs ont été calibrées sur la validation. La phase de test (`data/test/`) ne doit être utilisée qu'une seule fois pour le rapport final.
- La règle d'or du machine learning : **on ne touche jamais aux données de test pendant le développement**.

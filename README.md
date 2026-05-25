# CoineX — Comptage Automatique de Pièces de Monnaie

Un projet universitaire de traitement d'images réalisé dans le cadre du cours **Image** de L3 .

**L'objectif est simple :** à partir d’une image contenant des pièces de monnaie posées sur une surface, prédire automatiquement le nombre exact de pièces détectées.


## Architecture du Projet

Le code a été structuré de manière professionnelle et modulaire pour que chaque approche (pipeline) soit séparée et facile à comprendre.

```text
CoineX/
│
├── main.py                  # Point d'entrée principal (votre télécommande)
├── evaluation.py            # Script d'évaluation commun à tous les algorithmes
│
├── pipelines/               # Les différents algorithmes développés
│   ├── morphologie/         # La méthode officielle (HSV + Otsu + Morphologie)
│   ├── contours/            # Méthode alternative (Canny / Sobel)
│   ├── opencv_test/         # Test de validation avec la vraie librairie OpenCV
│   ├── nn/                  # Approche Deep Learning (CNN VGG-like, 5 canaux)
│   └── archives/            # Anciennes versions du code
│
├── utils/                   # Outils partagés
│   ├── io_utils.py          # Lecture d'image et redimensionnement
│   └── color.py             # Conversion RGB -> HSL / Gris
│
├── scripts/                 
│   └── preparer_dataset.py  # Script pour annoter et créer de nouveaux datasets
│
└── data/                    # Le dataset
    ├── validation/          # Images de validation (pour régler les hyperparamètres)
    └── test/                # Images de test (à ne regarder qu'à la toute fin)
```

---

## Comment exécuter et tester le projet ?

### 1. Installation

Assurez-vous d'avoir Python installé, puis ouvrez un terminal dans le dossier du projet et installez les quelques dépendances requises :

```bash
pip install -r requirements.txt
```

### 2. Évaluer les performances (Le juge final)

Le fichier `main.py` est conçu pour évaluer vos algorithmes sur tout un dataset (140 images) et vous donner les scores finaux (Taux de réussite, MAE, MSE).

**Lancer la méthode principale (Morphologie) sur tout le dataset :**
```bash
python main.py
```

**Tester l'algorithme sur une SEULE image :**
```bash
python main.py --image data/validation/img_001.jpg
```
*(Vous pouvez combiner avec `--pipeline contours` par exemple pour tester une autre méthode sur cette image !)*

**Lancer la méthode par contours (sur tout le dataset) :**
```bash
python main.py --pipeline contours
```

**Lancer le test OpenCV :**
```bash
python main.py --pipeline opencv
```

**Lancer la méthode Deep Learning (CNN) :**
```bash
python main.py --pipeline nn
```
*Nécessite `meilleur_modele_nn.pth` à la racine du projet. Voir la section ci-dessous pour l'entraînement.*

**Lancer l'évaluation finale sur le dataset de test :**
```bash
python main.py --mode test
```

### 3. Visualiser le fonctionnement étape par étape

Si vous voulez comprendre comment l'algorithme "réfléchit" et voir l'image se transformer (passage en noir et blanc, flou, nettoyage, détection...), utilisez les **visualiseurs**.

Pour voir les étapes de la **Morphologie** sur une image précise :
```bash
python pipelines/morphologie/visualizer.py data/validation/img_001.jpg
```

Pour voir les étapes de la méthode par **Contours** :
```bash
python pipelines/contours/visualizer.py data/validation/img_001.jpg
```
*(Remplacez `img_001.jpg` par n'importe quelle autre image du dossier `data/validation/`)*

---

## Pipeline NN (Deep Learning)

Le pipeline `nn` utilise un CNN VGG-like (~1.19M paramètres) qui prend en entrée 5 canaux :
- **Luminance** Y (continu)
- **Saturation** HLS (continu)
- **Magnitude Sobel** (continu)
- **Masque Otsu + morphologie** (binaire)
- **Contours Canny** (binaire)

L'entraînement se fait sur Kaggle (GPU T4 gratuit) via le notebook fourni :

```
pipelines/nn/coinex-nn-pipeline.ipynb
```

Étapes :
1. Upload du notebook sur Kaggle, attacher le dataset d'entraînement
2. Lancer "Save & Run All" (~1.5 h sur T4)
3. Télécharger le checkpoint produit : `/kaggle/working/meilleur_modele_nn.pth`
4. Placer ce fichier **à la racine du projet** (à côté de `main.py`)
5. Lancer `python main.py --pipeline nn`

Le checkpoint contient les poids du modèle ET les statistiques de normalisation
(mean/std par canal calculées sur le train set), donc tout est self-contained.

À l'inférence, on applique **Test-Time Augmentation** : 8 passages du modèle
sur les 8 transformations du groupe diédral D4 (4 rotations × 2 flips), puis
moyenne des prédictions pour réduire la variance.

Si le checkpoint est absent, le pipeline `nn` bascule automatiquement vers
`morphologie` pour ne pas casser l'évaluation.

---

## Métriques d'Évaluation

Le script `evaluation.py` calcule les scores suivants :
- **Taux de prédictions exactes (Accuracy) :** Le pourcentage d'images où l'algorithme a trouvé le bon nombre exact de pièces.
- **MAE (Erreur Absolue Moyenne) :** En moyenne, de combien de pièces l'algorithme se trompe-t-il par image ?
- **MSE (Erreur Quadratique Moyenne) :** Pénalise très fortement les grosses erreurs (quand l'algorithme se trompe de 15 pièces d'un coup).
- **RMSE :** Racine carrée de la MSE, pour ramener le score à la même unité que la MAE.

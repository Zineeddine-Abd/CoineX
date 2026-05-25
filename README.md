# CoineX — Comptage Automatique de Pièces de Monnaie

Projet universitaire de traitement d'images — L3 Informatique, cours **Image**.

**Objectif :** à partir d'une photo de pièces de monnaie posées sur une surface, prédire automatiquement le nombre exact de pièces.

---

## Architecture du Projet

```text
CoineX/
│
├── main.py                        # Point d'entrée : évaluation et test image unique
├── evaluation.py                  # Calcul des métriques (MAE, MSE, RMSE, accuracy)
│
├── pipelines/
│   ├── morphologie/               # Pipeline principal (implémenté from scratch)
│   │   ├── traitement.py          # Exports publics du pipeline
│   │   ├── detection.py           # Fonction principale compter_pieces()
│   │   ├── morphology.py          # Érosion, dilatation, ouverture, fermeture, BFS
│   │   ├── segmentation.py        # Histogramme, Otsu, égalisation
│   │   ├── filters.py             # Flou gaussien séparable
│   │   ├── config.py              # Hyperparamètres (seuils de forme)
│   │   └── visualizer.py          # Visualisation étape par étape (8 panneaux)
│   │
│   ├── contours/                  # Pipeline alternatif (OpenCV)
│   │   ├── traitement.py          # Pipeline complet + compter_pieces()
│   │   └── visualizer.py          # Visualisation étape par étape
│   │
│   └── nn/                        # Pipeline Deep Learning (CNN VGG-like)
│       ├── traitement.py          # Inférence + TTA + fallback morphologie
│       ├── modele.py              # Architecture CNN (5 canaux, ~1.19M paramètres)
│       ├── pretraitement.py       # Construction des 5 canaux d'entrée
│       └── entrainement.py        # Script d'entraînement (utilisé sur Kaggle)
│
├── utils/
│   ├── io_utils.py                # Lecture d'image, redimensionnement bilinéaire
│   └── color.py                   # Conversion RGB → HSL / Niveaux de gris
│
├── scripts/
│   └── preparer_dataset.py        # Annotation et division du dataset
│
└── data/
    ├── validation.json            # Vérité terrain — 140 images de validation
    ├── test.json                  # Vérité terrain — 60 images de test
    ├── validation/                # Images de validation (gitignorées)
    └── test/                      # Images de test (gitignorées)
```

---

## Pipelines Disponibles

### Pipeline Morphologie (principal, from scratch)

Implémenté entièrement en NumPy sans librairie de vision.

**Chaîne de traitement :**
1. Lecture et normalisation de l'image (uint8 RGB)
2. Redimensionnement bilinéaire (max 520px)
3. Conversion RGB → HSL, extraction de la saturation
4. Flou gaussien séparable (noyau adaptatif)
5. Seuillage automatique d'Otsu
6. Ouverture + fermeture morphologique binaire
7. Extraction des composantes connexes (BFS 8-connexe)
8. Filtrage par aire et position (rejet des bords)
9. Estimation du nombre de pièces par ratio d'aire médiane
10. Détection secondaire "pièce unique" (par contraste sur fond)

### Pipeline Contours (alternatif, OpenCV)

Basé sur la détection de contours par gradient.

**Chaîne de traitement :**
1. Sous-échantillonnage (800px)
2. Niveaux de gris → flou fort → égalisation d'histogramme
3. Flou doux → Sobel (Gx, Gy, magnitude)
4. Seuillage d'Otsu sur la magnitude
5. Fermeture morphologique → remplissage des silhouettes
6. Ouverture → érosion circulaire (sépare les pièces tangentes)
7. Filtrage par circularité, solidité et ratio de forme

### Pipeline NN (Deep Learning, CNN)

CNN VGG-like (~1.19M paramètres) entraîné sur Kaggle (GPU T4).

**Entrée :** 5 canaux par image — Luminance Y, Saturation HSL, Magnitude Sobel, Masque Otsu+morphologie, Contours Canny.

**Inférence :** Test-Time Augmentation sur les 8 transformations du groupe diédral D4 (4 rotations × 2 flips), puis moyenne.

Si le fichier de poids `meilleur_modele_nn.pth` est absent, le pipeline bascule automatiquement sur la morphologie.

---

## Résultats

| Pipeline      | Dataset     | MAE  | MSE   | RMSE | Accuracy |
|---------------|-------------|------|-------|------|----------|
| Morphologie   | Validation  | 1.74 | 11.81 | 3.44 | 50.0 %   |
| Morphologie   | Test        | 2.12 | 20.92 | 4.57 | 53.3 %   |
| Contours      | Validation  | 2.95 | 19.39 | 4.40 | 17.9 %   |
| Contours      | Test        | 2.87 | 24.63 | 4.96 | 31.7 %   |

Le pipeline morphologie est meilleur sur les deux datasets. Le pipeline contours souffre des fonds non uniformes et des pièces peu saturées.

---

## Comment Exécuter

### 1. Installation

```bash
pip install -r requirements.txt
```

### 2. Évaluation sur le dataset complet

**Pipeline morphologie (méthode principale) :**
```bash
python main.py
python main.py --pipeline morphologie --mode validation
python main.py --pipeline morphologie --mode test
```

**Pipeline contours :**
```bash
python main.py --pipeline contours --mode validation
python main.py --pipeline contours --mode test
```

**Pipeline NN (nécessite le checkpoint — voir section ci-dessous) :**
```bash
python main.py --pipeline nn --mode validation
python main.py --pipeline nn --mode test
```

### 3. Tester sur une seule image

```bash
python main.py --image data/validation/img_001.jpg
python main.py --image data/validation/img_001.jpg --pipeline contours
python main.py --image data/validation/img_001.jpg --pipeline nn
```

### 4. Visualiser le pipeline étape par étape

**Morphologie (8 étapes : saturation → Otsu → morphologie → composantes) :**
```bash
python pipelines/morphologie/visualizer.py data/validation/img_001.jpg
```

**Contours (Sobel → seuillage → remplissage → filtrage) :**
```bash
python pipelines/contours/visualizer.py data/validation/img_001.jpg
```

---

## Pipeline NN — Entraînement

Le modèle est entraîné sur Kaggle (GPU T4, gratuit) via le notebook :
```
pipelines/nn/entrainement.py
```

**Étapes :**
1. Entraîner le modèle sur Kaggle (~1.5h sur T4)
2. Télécharger le checkpoint : `meilleur_modele_nn.pth`
3. Placer ce fichier **à la racine du projet** (à côté de `main.py`)
4. Lancer `python main.py --pipeline nn`

Le checkpoint contient les poids ET les statistiques de normalisation (mean/std par canal), tout est auto-suffisant.

---

## Métriques d'Évaluation

- **Accuracy :** pourcentage d'images où le nombre exact est prédit
- **MAE** (Erreur Absolue Moyenne) : erreur moyenne en nombre de pièces
- **MSE** (Erreur Quadratique Moyenne) : pénalise les grosses erreurs
- **RMSE** : racine de la MSE, même unité que la MAE

---

## Dataset

200 images annotées manuellement, divisées en :
- **140 images de validation** (réglage des hyperparamètres)
- **60 images de test** (évaluation finale)

Le dataset est gitignorié. Pour créer un nouveau dataset à partir de photos brutes :
```bash
python scripts/preparer_dataset.py
```

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

## Métriques d'Évaluation

Le script `evaluation.py` calcule les scores suivants :
- **Taux de prédictions exactes (Accuracy) :** Le pourcentage d'images où l'algorithme a trouvé le bon nombre exact de pièces.
- **MAE (Erreur Absolue Moyenne) :** En moyenne, de combien de pièces l'algorithme se trompe-t-il par image ?
- **MSE (Erreur Quadratique Moyenne) :** Pénalise très fortement les grosses erreurs (quand l'algorithme se trompe de 15 pièces d'un coup).
- **RMSE :** Racine carrée de la MSE, pour ramener le score à la même unité que la MAE.

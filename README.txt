==============================================================================
  CoineX — Comptage Automatique de Pièces de Monnaie
  Projet L3 Informatique — Cours Image
==============================================================================

OBJECTIF
--------
À partir d'une photo de pièces de monnaie posées sur une surface, prédire
automatiquement le nombre exact de pièces présentes.


STRUCTURE DU PROJET
-------------------
CoineX/
  main.py               Point d'entrée : évaluation et test image unique
  evaluation.py         Calcul des métriques (MAE, MSE, RMSE, accuracy)

  pipelines/
    morphologie/        Pipeline principal (implémenté from scratch en NumPy)
      traitement.py     Exports publics du pipeline
      detection.py      Fonction principale compter_pieces()
      morphology.py     Érosion, dilatation, ouverture, fermeture, BFS
      segmentation.py   Histogramme, Otsu, égalisation
      filters.py        Flou gaussien séparable
      config.py         Hyperparamètres (seuils de forme)
      visualizer.py     Visualisation étape par étape (8 panneaux)

    contours/           Pipeline alternatif (OpenCV)
      traitement.py     Pipeline complet + compter_pieces()
      visualizer.py     Visualisation étape par étape

    nn/                 Pipeline Deep Learning (CNN VGG-like)
      traitement.py     Inférence + TTA + fallback morphologie
      modele.py         Architecture CNN (5 canaux, ~1.19M paramètres)
      pretraitement.py  Construction des 5 canaux d'entrée
      entrainement.py   Script d'entraînement (utilisé sur Kaggle)

  utils/
    io_utils.py         Lecture d'image, redimensionnement bilinéaire
    color.py            Conversion RGB -> HSL / Niveaux de gris

  scripts/
    preparer_dataset.py Annotation et division du dataset

  data/
    validation.json     Vérité terrain — 140 images de validation
    test.json           Vérité terrain — 60 images de test
    validation/         Images de validation (non versionnées)
    test/               Images de test (non versionnées)


==============================================================================
  DESCRIPTION DES PIPELINES
==============================================================================

PIPELINE MORPHOLOGIE (principal, from scratch)
-----------------------------------------------
Implémenté entièrement en NumPy, sans librairie de vision (pas de OpenCV).

Étapes :
  1. Lecture et normalisation de l'image (uint8 RGB)
  2. Redimensionnement bilinéaire (max 520 pixels)
  3. Conversion RGB -> HSL, extraction de la saturation
  4. Flou gaussien séparable (noyau adaptatif)
  5. Seuillage automatique d'Otsu
  6. Ouverture + fermeture morphologique binaire
  7. Extraction des composantes connexes (BFS 8-connexe)
  8. Filtrage par aire et position (rejet des bords)
  9. Estimation du nombre de pièces par ratio d'aire médiane
 10. Détection secondaire "pièce unique" (contraste sur fond)

PIPELINE CONTOURS (alternatif, OpenCV)
---------------------------------------
Basé sur la détection de contours par gradient.

Étapes :
  1. Sous-échantillonnage (800px)
  2. Niveaux de gris -> flou fort -> égalisation d'histogramme
  3. Flou doux -> Sobel (Gx, Gy, magnitude)
  4. Seuillage d'Otsu sur la magnitude
  5. Fermeture morphologique -> remplissage des silhouettes
  6. Ouverture -> érosion circulaire (sépare les pièces tangentes)
  7. Filtrage par circularité, solidité et ratio de forme

PIPELINE NN (Deep Learning, CNN VGG-like)
------------------------------------------
CNN à 5 canaux d'entrée, ~1.19 million de paramètres, entraîné sur Kaggle.

  Canaux d'entrée : Luminance Y, Saturation HSL, Magnitude Sobel,
                    Masque Otsu+morphologie, Contours Canny

  Inférence avec Test-Time Augmentation : moyenne sur 8 transformations
  (4 rotations x 2 flips) pour réduire la variance de prédiction.

  Si le fichier de poids est absent, le pipeline utilise automatiquement
  le pipeline morphologie comme solution de repli.


==============================================================================
  RÉSULTATS
==============================================================================

  Pipeline      Dataset     MAE    MSE    RMSE   Accuracy
  -----------   ----------  -----  -----  -----  --------
  Morphologie   Validation  1.74   11.81  3.44   50.0 %
  Morphologie   Test        2.12   20.92  4.57   53.3 %
  Contours      Validation  2.95   19.39  4.40   17.9 %
  Contours      Test        2.87   24.63  4.96   31.7 %

Le pipeline morphologie est meilleur sur les deux datasets.
Le pipeline contours souffre des fonds non uniformes et des pièces peu saturées.

  Nombre moyen de pièces par image : 3.83 (validation), 3.65 (test)
  Dataset : 200 images annotées manuellement (140 validation + 60 test)


==============================================================================
  MÉTRIQUES D'ÉVALUATION
==============================================================================

  Accuracy  Pourcentage d'images où le nombre exact est prédit.

  MAE       Erreur Absolue Moyenne.
            En moyenne, de combien de pièces l'algorithme se trompe-t-il ?
            Exemple : MAE = 1.74 signifie que l'algorithme se trompe en
            moyenne de 1.74 pièce par image.

  MSE       Erreur Quadratique Moyenne.
            Pénalise beaucoup plus les grosses erreurs que les petites.
            Exemple : se tromper de 4 pièces compte pour 16, pas pour 4.

  RMSE      Racine carrée de la MSE.
            Même unité que la MAE (nombre de pièces), mais reflète les
            grosses erreurs. RMSE > MAE si des erreurs importantes existent.

  MAE/Moy   Pourcentage d'erreur par rapport au nombre moyen réel de pièces.
            Permet de juger si l'erreur est "grande" relativement au problème.


==============================================================================
  COMMANDES D'EXÉCUTION
==============================================================================

INSTALLATION
  pip install -r requirements.txt


ÉVALUATION SUR TOUT LE DATASET

  Morphologie sur validation :
    python main.py
    python main.py --pipeline morphologie --mode validation

  Morphologie sur test :
    python main.py --pipeline morphologie --mode test

  Contours sur validation :
    python main.py --pipeline contours --mode validation

  Contours sur test :
    python main.py --pipeline contours --mode test

  NN sur validation (bascule sur morphologie si poids absents) :
    python main.py --pipeline nn --mode validation

  NN sur test :
    python main.py --pipeline nn --mode test


TEST SUR UNE SEULE IMAGE

  python main.py --image data/validation/img_001.jpg
  python main.py --image data/validation/img_001.jpg --pipeline contours
  python main.py --image data/validation/img_001.jpg --pipeline nn


VISUALISATION ÉTAPE PAR ÉTAPE

  Morphologie (saturation -> Otsu -> morphologie -> composantes) :
    python pipelines/morphologie/visualizer.py data/validation/img_001.jpg

  Contours (Sobel -> seuillage -> remplissage -> filtrage) :
    python pipelines/contours/visualizer.py data/validation/img_001.jpg


==============================================================================
  PIPELINE NN — ENTRAÎNEMENT
==============================================================================

Le modèle est entraîné sur Kaggle (GPU T4, gratuit).

Étapes :
  1. Lancer l'entraînement avec pipelines/nn/entrainement.py sur Kaggle
  2. Télécharger le checkpoint : meilleur_modele_nn.pth
  3. Placer ce fichier à la racine du projet (à côté de main.py)
  4. Lancer : python main.py --pipeline nn

Le checkpoint contient les poids du modèle ET les statistiques de
normalisation par canal (calculées sur le dataset d'entraînement).
Tout est auto-suffisant dans un seul fichier .pth.


==============================================================================
  DATASET
==============================================================================

200 images annotées manuellement, divisées en :
  - 140 images de validation (réglage des hyperparamètres)
  -  60 images de test (évaluation finale, non consulté pendant le dev)

Les images sont gitignorées (non versionnées dans le dépôt).

Pour créer un nouveau dataset à partir de photos brutes :
  python scripts/preparer_dataset.py

==============================================================================

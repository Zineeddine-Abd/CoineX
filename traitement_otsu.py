import cv2
import numpy as np

def compter_pieces(chemin_image, taille_flou=(21, 21)): # Flou augmenté par défaut
    img = cv2.imread(chemin_image)
    if img is None:
        return 0

    # 1. Conversion en niveaux de gris
    gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 2. Flou Gaussien (plus efficace que le flou simple pour le bruit)
    # On utilise un noyau impair large pour lisser les textures du fond
    floute = cv2.GaussianBlur(gris, taille_flou, 0)

    # 3. Segmentation par Otsu
    _, binarisee = cv2.threshold(floute, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 4. Nettoyage morphologique
    noyau = np.ones((5,5), np.uint8)
    propre = cv2.morphologyEx(binarisee, cv2.MORPH_OPEN, noyau)

    # 5. Extraction avec filtrage de surface (La clé du problème !)
    nb_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(propre)
    
    compteur_reel = 0
    # On définit une surface minimale (en pixels) en dessous de laquelle on ignore l'objet
    # Tu pourras ajuster cette valeur (500 est un bon début)
    surface_min = 500 

    for i in range(1, nb_labels): # On commence à 1 pour ignorer le fond
        surface = stats[i, cv2.CC_STAT_AREA]
        if surface > surface_min:
            compteur_reel += 1

    return compteur_reel
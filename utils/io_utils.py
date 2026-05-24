import math
from collections import deque
import matplotlib.image as mpimg
import numpy as np

from pipelines.morphologie.config import MAX_IMAGE_DIMENSION

# =============================================================================
# LECTURE ET NORMALISATION DE L'IMAGE
# =============================================================================
def lire_image_rgb(chemin_image):
    """
    Lit une image depuis le disque, la convertit dans un format RGB propre,
    puis la redimensionne si elle est trop grande.
    
    Représentation des images numériques
    --------------------------------------------------------
    - Modèle mathématique : f(Ω) → X^c
    - Sampling (discrétisation spatiale en pixels)
    - Quantization (valeurs en bytes 0-255)
    - Normalisation en format RGB propre

    Ce que fait la fonction :
    -------------------------
    1) lit l'image avec matplotlib.image.imread
    2) si l'image est en niveaux de gris (2D), la convertit en RGB (pas tres utile mais on c jms)
    3) si l'image a un canal alpha (RGBA), on enlève alpha (pas tres utile ais on jms)
    4) si l'image n'est pas en uint8, on la convertit en [0,255] uint8
    5) si l'image est trop grande, on la redimensionne et avec interpolation bilinéaire horizontale puis verticale 

    Pourquoi cette fonction est importante :
    ---------------------------------------
    Elle garantit que le reste du pipeline reçoit toujours une image dans un
    format propre et homogène  (tous en uint8)

    Sans cela, on pourrait avoir :
    - une image en float entre 0 et 1
    - une autre en uint8 entre 0 et 255
    - une image en gris 2D
    - une image RGBA à 4 canaux

    Et cela complique les calculs.

    Exemple :
    ---------
    Une image PNG avec transparence peut être lue en (H, W, 4).
    Ici, on garde seulement les 3 premiers canaux RGB.

    Exemple 2 :
    -----------
    Une image en niveaux de gris de forme (480, 640) n'a pas d'axe 2.
    Or les fonctions couleur attendent souvent (H, W, 3).
    Donc on empile l'image 3 fois pour obtenir :
    (480, 640, 3)
    """
    try:
        image = mpimg.imread(chemin_image)
    except FileNotFoundError:
        return None

    # Si l'image est en niveaux de gris (2D), on la convertit en RGB en empilant les canaux.
    # Crée une 3e dimension et mets les copies dedans
    if image.ndim == 2:
        image = np.stack([image, image, image], axis=2)

   # Si l'image a un canal alpha (RGBA), on enlève le canal alpha pour ne garder que RGB.
   # Garde toutes les lignes et colonnes, mais seulement les 3 premiers canaux
    if image.shape[2] > 3:
        image = image[..., :3]

    # Si l'image n'est pas en uint8, on la convertit en [0,255] uint8.
    # rint arrondi à l'entier le plus proche, clip pour éviter les débordements, puis convertit en uint8
    if image.dtype != np.uint8:
        image = np.clip(np.rint(image * 255.0), 0, 255).astype(np.uint8)

    hauteur, largeur = image.shape[:2]

    # Calcul de l'échelle de redimensionnement.
    #
    # echelle <= 1 :
    # - si l'image est petite, echelle = 1 -> on ne change rien
    # - si l'image est grande, echelle < 1 -> on réduit
    echelle = min(1.0, MAX_IMAGE_DIMENSION / max(largeur, hauteur))

    # Redimensionnement si nécessaire.
    if echelle < 1.0:
        image = redimensionner_bilineaire(
            image,
            max(1, int(round(hauteur * echelle))),
            max(1, int(round(largeur * echelle))),
        )

    return image


import numpy as np

# =============================================================================
# REDIMENSIONNEMENT BILINÉAIRE
# =============================================================================
def redimensionner_bilineaire(image, nouvelle_hauteur, nouvelle_largeur):
    """
    Redimensionne l'image par interpolation bilinéaire.

    Notion d'interpolation :
    ------------------------
    Quand on change la taille d'une image, les nouveaux pixels n'existent pas
    dans l'image d'origine. Il faut donc "inventer" leur valeur.
    L'interpolation donne une règle pour cela.

    Interpolation bilinéaire :
    --------------------------
    Pour chaque pixel de sortie, on regarde les 4 pixels les plus proches
    dans l'image d'origine :
    - haut gauche
    - haut droite
    - bas gauche
    - bas droite

    Puis on fait une moyenne pondérée selon la position réelle du point.

    Pourquoi c'est mieux que le "plus proche voisin" :
    --------------------------------------------------
    Le plus proche voisin donne souvent un rendu brutal, avec effet d'escalier.
    L'interpolation bilinéaire donne des transitions plus douces.

    Exemple :
    ---------
    Si entre deux pixels on a 10 et 30,
    un pixel intermédiaire peut devenir environ 20,
    au lieu d'être brutalement 10 ou 30.
    
    Notre exemple (L'analogie de la mosaïque) :
    -------------------------------------------
    Imaginez l'ancienne image comme une vraie mosaïque de carreaux colorés, et la 
    nouvelle image comme une feuille transparente avec une nouvelle grille vide que 
    l'on superpose par-dessus.
    Pour chaque case vide de la feuille transparente (nouveau pixel), on pose notre doigt. 
    Ce doigt atterrit "à cheval" sur 4 vieux carreaux de la mosaïque en dessous. 
    On prend la couleur de ces 4 vieux carreaux, et on la mélange en fonction de 
    la proximité exacte du doigt avec chacun d'eux pour peindre la nouvelle case.
    """
    hauteur, largeur = image.shape[:2]

    # Si la feuille transparente a exactement la même taille que la mosaïque, 
    # on fait juste une copie directe.
    if nouvelle_hauteur == hauteur and nouvelle_largeur == largeur:
        return image.copy()

    # =========================================================================
    # ÉTAPE 1 : LA SUPERPOSITION (Où tombent nos doigts ?)
    # =========================================================================
    # crée nouvelle_hauteur valeurs uniformes entre 0 et hauteur-1
    # représentent les positions X et Y dans l’image source correspondant aux lignes de sortie
    y = np.linspace(0, hauteur - 1, nouvelle_hauteur, dtype=np.float32)
    x = np.linspace(0, largeur - 1, nouvelle_largeur, dtype=np.float32)
    # Crée une grille 2D de coordonnées (xx, yy) pour chaque pixel de sortie
    xx, yy = np.meshgrid(x, y)

    # =========================================================================
    # ÉTAPE 2 : IDENTIFIER LES 4 VIEUX CARREAUX SOUS CHAQUE DOIGT
    # =========================================================================
    # 'floor' (arrondi vers le bas) trouve le vieux carreau en Haut à Gauche.
    x0 = np.floor(xx).astype(np.int32)
    y0 = np.floor(yy).astype(np.int32)

    # On ajoute +1 pour trouver les carreaux de droite du bas.
    # 'clip' empêche de chercher un carreau qui n'existe pas en dehors de la table.
    x1 = np.clip(x0 + 1, 0, largeur - 1)
    y1 = np.clip(y0 + 1, 0, hauteur - 1)

    # =========================================================================
    # ÉTAPE 3 : CALCULER LA PROPORTION DE MÉLANGE (Le poids)
    # =========================================================================
    # On mesure à quel point notre doigt est décalé par rapport au carreau Haut-Gauche.
    # wx = 0.8 signifie qu'on est très proche de la droite (à 80%).
    wx = xx - x0
    wy = yy - y0

    image = image.astype(np.float32)

    # =========================================================================
    # ÉTAPE 4 : PRENDRE LA PEINTURE DES 4 VIEUX CARREAUX
    # =========================================================================
    haut_gauche = image[y0, x0]
    haut_droite = image[y0, x1]
    bas_gauche = image[y1, x0]
    bas_droite = image[y1, x1]

    # =========================================================================
    # ÉTAPE 5 : LE MÉLANGE DE PEINTURE
    # =========================================================================
    # On mélange d'abord la ligne du haut, puis la ligne du bas...
    haut = haut_gauche * (1.0 - wx)[..., None] + haut_droite * wx[..., None]
    bas = bas_gauche * (1.0 - wx)[..., None] + bas_droite * wx[..., None]

    # ...puis on mélange ces deux résultats verticalement pour avoir la couleur finale !
    resultat = haut * (1.0 - wy)[..., None] + bas * wy[..., None]

    # =========================================================================
    # ÉTAPE 6 : NETTOYAGE ET RENDU
    # =========================================================================
    # On arrondit nos mélanges à virgule en nombres entiers (0 à 255) propres.
    return np.clip(np.rint(resultat), 0, 255).astype(np.uint8)



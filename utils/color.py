import math
from collections import deque
import matplotlib.image as mpimg
import numpy as np

# =============================================================================
# CONVERSION COULEUR : RGB -> HSL (partiel : luminosité + saturation)
# =============================================================================
# la diff ici est que je retourne les 2 saturation et luminosité pas que la saturation
def rgb_vers_hsl(image_rgb):
    """
    Convertit une image RGB en HSL (Hue, Saturation, Luminosity).
    
    COURS : semaine 3 - Espaces couleur (Color Spaces)
    -----------------------------------------------
    Convertit RGB (additif, écrans) → HSL (perceptuel, robuste aux ombres)
    Formule mathématique de saturation HSL :
    S = delta / (1 - |2L - 1|)  où delta = max(R,G,B) - min(R,G,B)

    Pourquoi la saturation est importante :
    ---------------------------------------
    Dans certaines images, les pièces se distinguent mieux du fond par leur
    saturation que par leur intensité lumineuse.

    Exemple :
    ---------
    - fond terne / grisâtre
    - pièces plus "riches" en couleur
    La saturation permet alors de mieux faire ressortir les pièces.

    Détail mathématique :
    ---------------------
    On calcule :
    - le max des canaux RGB
    - le min des canaux RGB
    - delta = max - min
    - luminosité = (max + min)/2
    - saturation HSL avec la formule adaptée

    
    """
    image = image_rgb.astype(np.float32) / 255.0

    maximum = np.max(image, axis=2)
    minimum = np.min(image, axis=2)
    delta = maximum - minimum

    luminosite = (maximum + minimum) / 2.0
    saturation = np.zeros_like(luminosite)

    masque = delta > 1e-6
    denominateur = 1.0 - np.abs(2.0 * luminosite - 1.0)
    saturation[masque] = delta[masque] / (denominateur[masque] + 1e-6)

    return luminosite, np.clip(saturation, 0.0, 1.0)


# =============================================================================
# CONVERSION COULEUR : RGB -> GRIS
# =============================================================================
def rgb_vers_gris(image_rgb):
    """
    Convertit une image RGB en niveaux de gris normalisés entre 0 et 1.
    
    COURS : semaine 3 - Conversion en niveaux de gris
    -----------------------------------------------
    Formule de luminance standard :
    Gray = 0.299R + 0.587G + 0.114B
    
    Ces coefficients reflètent la sensibilité de l'œil humain :
    - Plus sensible au vert (0.587)
    - Moins sensible au bleu (0.114)
    - Sensibilité moyenne au rouge (0.299)

    Pourquoi cette fonction existe alors qu'on a déjà la saturation :
    -----------------------------------------------------------------
    Parce qu'une seule représentation ne suffit pas toujours.
    La saturation est bonne dans certains cas,
    mais pour le cas "une seule pièce", le contraste avec le fond
    en niveaux de gris peut être plus utile.

    Formule de luminance :
    ----------------------
    0.299 R + 0.587 G + 0.114 B

    Pourquoi ces coefficients :
    ---------------------------
    L'œil humain est plus sensible au vert qu'au rouge, et moins au bleu.
    Ce n'est donc pas une moyenne simple (R+G+B)/3. (hadi justif ida saksana)

    Exemple :
    ---------
    Une pièce métallique peu saturée peut rester bien visible en gris
    si elle contraste avec le fond.
    """
    image = image_rgb.astype(np.float32)
    return (
        0.299 * image[..., 0] + 0.587 * image[..., 1] + 0.114 * image[..., 2]
    ) / 255.0



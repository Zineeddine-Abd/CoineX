import math
from collections import deque
import matplotlib.image as mpimg
import numpy as np


# =============================================================================
# PIPELINE GLOBAL DU PROGRAMME
# =============================================================================
#
# 
#
# Chaîne globale de traitement :
# ------------------------------
# 1) lire l'image proprement et la normaliser (mettre ses valeurs de pixels dans une échelle standard pour faciliter les calculs)
# 2) éventuellement la redimensionner pour réduire le coût de calcul (parce qu’on traite moins de pixels)
# 3) Convertit RGB en HSL pour faire la détection à partir de la saturation (HSL)
# 4) Convertit en niveaux de gris avec la formule de luminance
# 4) lisser l'image avec un flou gaussien
# 5) segmenter automatiquement avec Otsu
# 6) nettoyer le masque avec une ouverture morphologique binaire
# 7) extraire les composantes connexes
# 8) mesurer leur forme (aire, circularité, remplissage, bbox, bord)
# 9) estimer le nombre de pièces à partir de ces composantes
# 10) lancer une détection secondaire spéciale "une seule pièce"
# 11) appliquer des règles correctives finales
#
#
# Exemples de cas difficiles que ce pipeline essaie de mieux gérer :
# ------------------------------------------------------------------
# - 2 pièces collées -> peuvent former une seule grosse composante
# - 1 seule pièce brillante -> peut être découpée en plusieurs régions
# - objet  au bord -> ne doit pas être compté
# - image très grande -> coût de calcul plus élevé si on ne redimensionne pas

# =============================================================================
# CONFIGURATION & HYPERPARAMÈTRES
# =============================================================================
# [RÈGLE D'OR] : Toutes les valeurs ci-dessous ont été réglées empiriquement
# sur la base de VALIDATION UNIQUEMENT, sans regarder la base de test.
# Cette approche garantit que les résultats finaux ne sont pas biaisés.
# =============================================================================

# ===== GROUPE 1 : DESCRIPTEURS DE FORMES POUR COIN DE RÉFÉRENCE =====
# Utilisé pour identifier une composante connexe qui ressemble vraiment à une pièce.
# Une pièce "de référence" nous permet d'estimer la taille typique d'une pièce.
#
# Cours : Semaine 7 - Descripteurs d'objets et composantes connexes
#
COIN_REFERENCE_CIRCULARITY_MIN = 0.45      # Circularité minimale (1.0 = cercle parfait)
COIN_REFERENCE_FILL_MIN = 0.45             # Remplissage minimum (aire / bbox)
COIN_REFERENCE_ASPECT_MIN = 0.65           # Ratio minimum hauteur/largeur (si < 1 : objet aplati)
COIN_REFERENCE_ASPECT_MAX = 1.55           # Ratio maximum hauteur/largeur (si > 1 : objet allongé)

# ===== GROUPE 2 : DÉTECTION DE PIÈCES FUSIONNÉES =====
# Quand deux pièces se touchent, elles forment une seule composante connexe.
# Ces paramètres permettent de détecter et de compter ces cas.
#
# Cours : Semaine 7 - Analyse d'objets connexes
#
MERGE_AREA_RATIO_THRESHOLD = 1.8           # Si aire > 1.8 × aire_typique, compte comme 2 pièces
MERGE_FILL_MIN = 0.35                      # Remplissage minimum pour détecter fusion

# ===== GROUPE 3 : DÉTECTION SPÉCIALE "UNE SEULE PIÈCE" =====
# Filet de sécurité : si le pipeline principal doute, on teste si c'est exactement 1 pièce.
# Critères très stricts pour ne faux-positif.
#
# Cours : Semaine 7 - Validation de composantes
#
SINGLE_COIN_FILL_MIN = 0.62                # Remplissage minimum pour une pièce unique
SINGLE_COIN_CIRCULARITY_MIN = 0.40         # Circularité minimum pour une pièce unique
SINGLE_COIN_ASPECT_RATIO_MIN = 0.78        # Ratio min hauteur/largeur
SINGLE_COIN_ASPECT_RATIO_MAX = 1.28        # Ratio max hauteur/largeur

# ===== GROUPE 4 : RÈGLES CORRECTIVES (TRÈS GRANDES PIÈCES) =====
# Si la prédiction principale est très grande mais qu'on observe une seule
# composante énorme et très circulaire, on corrige à 1.
#
# Cours : Semaine 7-8 - Analyse statistique des composantes
#
CORRECTION_LARGE_CIRCULARITY_MIN = 0.48    # Circularité pour très grande pièce
CORRECTION_LARGE_FILL_MIN = 0.68           # Remplissage pour très grande pièce
CORRECTION_LARGE_AREA_MULTIPLIER = 8.0     # Doit être > 8× aire typique

# ===== GROUPE 5 : RÈGLES CORRECTIVES (COMPOSANTES RARES) =====
# Si prediction est très grande mais peu de composantes, peut être 1 grosse pièce.
#
# Cours : Semaine 7 - Statistiques sur les composantes
#
CORRECTION_RARE_CIRCULARITY_MIN = 0.55     # Circularité stricte
CORRECTION_RARE_FILL_MIN = 0.72            # Remplissage strict

# ===== GROUPE 6 : PARAMÈTRES SYSTÈME =====
# Paramètres de performance et système, pas liés à la détection mathématique.
#
MAX_IMAGE_DIMENSION = 520                  # Redimensionner images > 520 pixels (performance)


# =============================================================================
# LECTURE ET NORMALISATION DE L'IMAGE
# =============================================================================
def lire_image_rgb(chemin_image):
    """
    Lit une image depuis le disque, la convertit dans un format RGB propre,
    puis la redimensionne si elle est trop grande.
    
    COURS : Semaine 1-2 - Représentation des images numériques
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
    r = image[..., 0]
    g = image[..., 1]
    b = image[..., 2]

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


# =============================================================================
# FLUO GAUSSIEN : NOYAU ET CONVOLUTIONS
# =============================================================================
def noyau_gaussien_1d(taille, sigma):
    """
    Crée un noyau gaussien 1D normalisé pour le filtrage.
    
    COURS : semaine 8 - Convolution & Filtrage Gaussien
    -----------------------------------------------
    Formule mathématique :
    G(x) = exp(-(x²) / (2σ²))  [normalisé : sum = 1]
    
    Plus σ est grand : plus de flou
    Plus σ est petit : moins de flou
    
    Utilisé dans le flou gaussien séparable (horizontal puis vertical)
    pour une efficacité O(n) au lieu d'O(n²)

    Notion de noyau :
    -----------------
    Un noyau est un petit tableau de poids utilisé pour filtrer localement
    une image.

    Exemple de noyau 1D gaussien :
    ------------------------------
    [0.05, 0.25, 0.40, 0.25, 0.05]

    Le centre compte plus que les bords.

    Pourquoi :
    ----------
    Dans un flou gaussien, les voisins proches influencent davantage
    la nouvelle valeur que les voisins éloignés

    Paramètres :
    ------------
    taille : nombre de coefficients du noyau
    sigma  : étalement de la gaussienne

    Plus sigma est grand :
    ----------------------
    plus le flou est étalé.
    """

    # calcul du rayon du noyau (distance du centre aux bords)
    rayon = taille // 2

    # création d’un axe centré sur 0 : ex [-2, -1, 0, 1, 2]
    axe = np.arange(-rayon, rayon + 1, dtype=np.float32)

    # calcul de la gaussienne pour chaque position de l’axe
    # donne des poids élevés au centre et faibles aux extrémités
    noyau = np.exp(-(axe * axe) / (2.0 * sigma * sigma))

    # somme des valeurs du noyau (avant normalisation)
    somme = np.sum(noyau)

    # normalisation pour que la somme des poids = 1
    # (évite de modifier la luminosité de l’image lors du filtrage)
    if somme > 0:
        noyau /= somme

    # retourne le noyau 1D prêt à être utilisé en convolution
    return noyau


def convolution_1d_lignes(image, noyau):
    """
    Applique la convolution 1D horizontalement (ligne par ligne).
    
    COURS : Week 8 - Convolution discrète
    -----------------------------------
    Convolution 1D discrète : (f * g)[n] = Σ f[m] * g[n-m]
    
    Mode 'same' : préserve la taille de l'image (padding aux bords) pour que:
    taille entrée = taille sortie
    
    np.convolve() implémente exactement la formule mathématique du cours,
    pas une approximation ou optimisation en black-box.

    Pourquoi une convolution 1D :
    -----------------------------
    Une gaussienne 2D peut être séparée en :
    - une convolution horizontale
    - puis une convolution verticale

    Cela donne le même résultat qu'une vraie gaussienne 2D,
    mais plus efficacement et plus rapide.

    Notion de fenêtre glissante :
    -----------------------------
    Pour chaque pixel, on regarde une petite fenêtre locale centrée autour
    de lui.

    Exemple sur une ligne :
    -----------------------
    Ligne de pixels= [10, 10, 20, 30, 30]
    Le noyau= [0.25, 0.5, 0.25]
    Fenêtre = petit morceau de l’image utilisé localement pour calculer un pixel filtré
    Fenêtre taille 3 :
    - [10, 10, 20]
    - [10, 20, 30]
    - [20, 30, 30]

    Ensuite on multiplie cette fenêtre par le noyau puis on somme.
    10 × 0.25 + 20 × 0.5 + 30 × 0.25
    = 2.5 + 10 + 7.5
    = 20
    Le pixel central (20) est remplacé par une nouvelle valeur (20 ici)

    Padding mode="edge" :
    ---------------------
    Aux bords, on prolonge la valeur du bord pour éviter de perdre des pixels.
    Image originale : [10, 10, 20, 30, 30]
    On prolonge (padding “edge”) : [10, 10, 10, 20, 30, 30, 30]
    
    [AJOUT POUR LA SOUTENANCE] :
    ----------------------------
    Le code d'origine (commenté ci-dessous) utilisait np.einsum et sliding_window_view
    qui sont très complexes à justifier. Le nouveau code utilise np.convolve qui 
    traduit exactement la combinaison linéaire vue en cours (Semaine 8).
    """
    # ----- ANCIEN CODE GARDÉ EN COMMENTAIRE -----
    # pad = len(noyau) // 2
    # image_pad = np.pad(image, ((0, 0), (pad, pad)), mode="edge")
    # fenetres = np.lib.stride_tricks.sliding_window_view(image_pad, len(noyau), axis=1)
    # return np.einsum("ijk,k->ij", fenetres, noyau[::-1], optimize=True)
    # ---------------------------------------------

    return np.apply_along_axis(
    # Fonction appliquée à chaque ligne de l'image
    lambda ligne: np.convolve(
        ligne,            # une ligne de pixels (1D)
        noyau,            # filtre (poids de convolution)
        mode='same'       # conserve la même taille que la ligne d'origine
    ),
    
    axis=1,              # 1 = on parcourt les lignes (horizontalement)
    
    arr=image            # image 2D (matrice de pixels)
)
    


def convolution_1d_colonnes(image, noyau):
    """
    Même principe que convolution_1d_lignes, mais verticalement.

    On applique maintenant le noyau sur les colonnes.
    """
    # ----- ANCIEN CODE GARDÉ EN COMMENTAIRE -----
    # pad = len(noyau) // 2
    # image_pad = np.pad(image, ((pad, pad), (0, 0)), mode="edge")
    # fenetres = np.lib.stride_tricks.sliding_window_view(image_pad, len(noyau), axis=0)
    # return np.einsum("ijk,k->ij", fenetres, noyau[::-1], optimize=True)
    # ---------------------------------------------
    
    return np.apply_along_axis(
    # Applique une fonction sur chaque colonne de l'image
    lambda colonne: np.convolve(
        colonne,        # une colonne de pixels (1D vertical)
        noyau,          # filtre (poids de convolution)
        mode='same'     # conserve la même taille que la colonne d'origine
    ),

    axis=0,             # 0 = on parcourt les colonnes (verticalement)
    
    arr=image           # image 2D (matrice de pixels)
)

def flou_gaussien(image, taille):
    """
    Applique un flou gaussien à une image 2D (2D Gaussian filtering).
    
    COURS : semaine 8 - Opérations locales & Filtrage Gaussien
    -------------------------------------------------------
    Combine deux convolutions 1D séparables pour efficacité.
    C'est une \"opération locale\" : chaque pixel dépend de ses voisins.
    
    Réduit :
    - Le bruit haute fréquence
    - Les petites variations locales
    - Les détails qui perturbent la segmentation

    Pourquoi le flou gaussien est important :
    ----------------------------------------
    Avant le seuillage, on veut réduire :
    - le bruit
    - les petites variations locales
    - les détails qui risquent de perturber Otsu
    comme a dit M. lobry

    Étapes :
    --------
    1) imposer une taille impaire 
    2) choisir sigma
    3) créer le noyau gaussien 1D
    4) convolver horizontalement
    5) convolver verticalement

    Pourquoi la taille doit être impaire :
    --------------------------------------
    Pour avoir un centre bien défini dans le noyau.

    Exemple :
    ---------
    - taille 5 -> centre clair
    - taille 4 -> pas de centre unique
    """
    taille = max(3, int(taille))
    if taille % 2 == 0:
        taille += 1

    sigma = max(1.0, taille / 3.0)
    noyau = noyau_gaussien_1d(taille, sigma)

    image = image.astype(np.float32)
    image = convolution_1d_lignes(image, noyau)
    return convolution_1d_colonnes(image, noyau)


# =============================================================================
# HISTOGRAMME ET SEUIL D'OTSU
# =============================================================================
def histogramme_u8(image):
    """
    Convertit une image normalisée [0,1] en uint8 [0,255], puis calcule
    l'histogramme des intensités.

    Pourquoi on passe en uint8 :
    ----------------------------
    Otsu est ici implémenté sur 256 niveaux de gris.
    Cela simplifie le calcul de l'histogramme.

    Exemple :
    ---------
    Un pixel 0.0 devient 0
    Un pixel 0.5 devient environ 128
    Un pixel 1.0 devient 255
    """
    # rint arrondi à l'entier le plus proche, clip pour éviter les débordements, puis convertit en uint8
    image_u8 = np.clip(np.rint(image * 255.0), 0, 255).astype(np.uint8)
    # np.bincount compte le nombre d'occurrences de chaque valeur de pixel (0 à 255)
    # ravel : aplati l'image 2D en 1D pour que bincount puisse compter tous les pixels
    return np.bincount(image_u8.ravel(), minlength=256).astype(np.float64)


def seuil_otsu(image):
    """
    Calcule automatiquement un seuil d'Otsu dans [0,1].
    
    COURS : semaine 5 - Segmentation par seuillage & Algorithme d'Otsu
    ================================================================
    
    PRINCIPE MATHÉMATIQUE :
    Maximise la variance entre-classes σ²_B(t) pour tous seuils t ∈ [0,255]
    
    σ²_B(t) = w0(t) * w1(t) * (μ0(t) - μ1(t))²
    
    Où :
    - w0(t) = fraction de pixels en dessous du seuil t (fond)
    - w1(t) = fraction de pixels au-dessus du seuil t (objets)
    - μ0(t) = intensité moyenne du fond   
    - μ1(t) = intensité moyenne des objets
    
    ALGORITHME :
    1. Calculer histogramme de l'image (256 niveaux)
    2. Pour chaque seuil t, calculer la variance inter-classes
    3. Retourner le seuil avec variance maximale
    
    AVANTAGE : Seuil déterministe, s'adapte à l'image (pas de paramètre à régler)
    Comparaison avec K-Means : Otsu est OPTIMAL pour 2 classes, K-Means dépend de l'initialisation

    Principe d'Otsu :
    -----------------
    On cherche le seuil qui sépare au mieux deux classes :
    - le fond
    - les objets

    Pour chaque seuil possible, on mesure la séparation entre les deux classes
    avec la variance inter-classes, puis on prend le meilleur.

    Pourquoi c'est utile :
    ----------------------
    On évite de choisir un seuil à la main.
    Le seuil s'adapte à l'image.

    Exemple :
    ---------
    Si l'histogramme a :
    - un pic autour de 20 pour le fond
    - un pic autour de 170 pour les pièces
    Otsu choisira un seuil intermédiaire.

    [AJOUT POUR LA SOUTENANCE] - Lien avec le cours (Semaine 6) :
    -------------------------------------------------------------
    Pourquoi Otsu et pas les K-Moyennes ?
    Ici, nous voulons séparer exactement 2 classes (Fond vs Pièce).
    L'algorithme d'Otsu teste de manière exhaustive tous les seuils pour 
    minimiser la variance intra-classe. Il garantit donc une solution mathématiquement 
    OPTIMALE pour ce cas précis, contrairement aux K-Moyennes qui dépendent 
    de leur initialisation aléatoire.
    """
    hist = histogramme_u8(image)
    total = hist.sum()
    if total == 0:
        return 0.5

    # transformons l'histogramme en probabilités
    probabilites = hist / total
    # calcul des proportion de pixels jusqu’au niveau de gris i
    cumul_prob = np.cumsum(probabilites)
    # calcul de la moyenne cumulée des intensités jusqu’au niveau i
    cumul_moy = np.cumsum(probabilites * np.arange(256))
    # la moyenne totale de l'image (intensité moyenne globale) c'est le dernier élément de cumul_moy
    moyenne_totale = cumul_moy[-1]

    variance_inter = (moyenne_totale * cumul_prob - cumul_moy) ** 2
    variance_inter /= cumul_prob * (1.0 - cumul_prob) + 1e-12

    # np.argmax trouve l'indice du seuil qui maximise la variance inter-classes
    seuil = int(np.argmax(variance_inter))
    return seuil / 255.0


# =============================================================================
# MORPHOLOGIE BINAIRE
# =============================================================================
def erosion_binaire(image_binaire, taille):
    """
    Érosion binaire (Binary Erosion Morphological Operation).
    
    COURS : semaine 6 - Opérations morphologiques binaires
    --------------------------------------------------
    Définition : Un pixel reste blanc (True) SIseul si TOUS les pixels
    dans la fenêtre locale autour de lui sont blancs.
    
    Formule logique : 
    O(y,x) = min{I(y+dy, x+dx) : (dy,dx) ∈ fenêtre}
    Pour l'image binaire = AND logique sur tous les voisins
    
    Effets visuels :
    - Supprime les petits bruits isolés
    - Réduit les objets (amincit les régions)
    - Enlève les petites excroissances
    - Sépare les objets proches

    Idée :
    ------
    Un pixel reste true seulement si toute la fenêtre autour de lui est true

    Effet :
    -------
    - supprime les petits bruits
    - réduit les objets
    - enlève les petites excroissances

    Exemple :
    ---------
    Un pixel blanc isolé disparaît après érosion.
    """
    taille = max(1, int(taille))
    if taille % 2 == 0:
        taille += 1

    pad = taille // 2
    image_pad = np.pad(
        image_binaire.astype(bool), ((pad, pad), (pad, pad)), mode="constant", constant_values=False
    )
    fenetres = np.lib.stride_tricks.sliding_window_view(image_pad, (taille, taille))

    return np.all(fenetres, axis=(2, 3))


def dilatation_binaire(image_binaire, taille):
    """
    Dilatation binaire (Binary Dilation Morphological Operation).
    
    COURS : semaine 6 - Opérations morphologiques binaires
    -------------------------------------------------
    Définition : Un pixel devient blanc (True) SI AU MOINS UN pixel
    dans la fenêtre locale autour de lui est blanc.
    
    Formule logique :
    O(y,x) = max{I(y+dy, x+dx) : (dy,dx) ∈ fenêtre}
    Pour l'image binaire = OR logique sur tous les voisins
    
    Effets visuels :
    - Agrandit les objets (épaissit les régions)
    - Referme les petits trous internes
    - Fusionne les objets proches
    - Comble les petites rides

    Idée :
    ------
    Un pixel devient vrai si au moins un pixel de sa fenêtre est vrai.

    Effet :
    -------
    - agrandit les objets
    - referme de petits trous

    Pourquoi c'est utile :
    ----------------------
    Elle complète l'érosion dans l'ouverture morphologique.
    """
    taille = max(1, int(taille))
    if taille % 2 == 0:
        taille += 1

    pad = taille // 2
    image_pad = np.pad(
        image_binaire.astype(bool), ((pad, pad), (pad, pad)), mode="constant", constant_values=False
    )
    fenetres = np.lib.stride_tricks.sliding_window_view(image_pad, (taille, taille))

    return np.any(fenetres, axis=(2, 3))


def ouverture_binaire(image_binaire, taille):
    """
    Ouverture binaire = Érosion suivi de Dilation (Opening Operation).
    
    COURS : semaine 6 - Composition d'opérations morphologiques
    -------------------------------------------------------
    Formule : O = Dilate(Erode(I))
    
    Propriétés mathématiques :
    - Élimine les objets plus petits que la fenêtre structurale
    - Lisse les contours (réduit les oscillations)
    - Conserve les gros objets presque intacts
    - Idempotente : O(O(I)) = O(I)
    
    Cas d'usage :
    - Nettoyage du bruit après seuillage
    - Suppression des petits parasites
    - Lissage des bords

    Pourquoi on l'utilise :
    -----------------------
    Après le seuillage, l'image binaire contient souvent :
    - des petits points parasites
    - des petits morceaux de bruit
    - des bords irréguliers

    L'ouverture permet de nettoyer ces défauts.

    Exemple :
    ---------
    Une vraie pièce + quelques petits points isolés :
    l'ouverture garde surtout la pièce.
    """
    return dilatation_binaire(erosion_binaire(image_binaire, taille), taille)


# =============================================================================
# COMPOSANTES CONNEXES ET DESCRIPTEURS DE FORME
# =============================================================================
def composantes_connexes(image_binaire):
    """
    Extrait toutes les composantes connexes d'un masque binaire et calcule
    plusieurs descripteurs de forme pour chacune.
    
    COURS : semaine 7 - Analyse d'objets & Composantes connexes
    ==========================================================
    
    DÉFINITIONS :
    - Composante connexe : ensemble maximal de pixels blancs reliés entre eux
    - 8-connexité : deux pixels sont voisins s'ils se touchent (incluant diagonales)
    
    ALGORITHME (Breadth-First Search - BFS) :
    1. Parcourir tous les pixels (y, x) de haut en bas, gauche à droite
    2. Si pixel blanc et non encore étiqueté :
       a. Créer nouvelle composante (nouvelle étiquette)
       b. BFS depuis ce pixel : explorer tous ses voisins 8-connexes récursivement
       c. Étiqueter tous les pixels trouvés avec même étiquette
    3. Calculer descripteurs de forme pour chaque composante
    
    DESCRIPTEURS CALCULÉS (Week 7 - Properties of objects) :
    - area           : nombre de pixels
    - bbox           : boîte englobante (y_min, x_min, y_max, x_max)
    - hauteur_bbox / largeur_bbox : dimensions de la boîte
    - remplissage    : ratio aire / aire_bbox (compacité)
    - circularite    : 4π*aire / périmètre² (1.0 = cercle parfait)
    - touche_bord    : booléen True si objet sort du cadre
    
    UTILITÉ EN DÉTECTION DE PIÈCES :
    Permet de distinguer une pièce d'un bruit :
    - Pièce : circularité ≈ 0.7-0.9, remplissage ≈ 0.6-0.9
    - Bruit : circularité < 0.5, remplissage aléatoire

    Qu'est-ce qu'une composante connexe :
    -------------------------------------
    C'est un ensemble de pixels blancs reliés entre eux.

    Ici on utilise la connexité 8 :
    -------------------------------
    On considère comme voisins :
    - haut, bas, gauche, droite
    - + les 4 diagonales

    Avec la 8 connexté c'est plus fiable pcq on peut avoir ca 0 1 ca peut etre considére comme 2 composante avec la 4 connexité alors que c une seule
                                                          1 0 

    Pourquoi la connexité 8 :
    -------------------------
    Une vraie région d'objet peut être connectée par diagonale.
    [AJOUT POUR LA SOUTENANCE - Semaine 8] : Contrairement à la 4-connexité, 
    la 8-connexité est indispensable ici pour palier aux bruits de discrétisation 
    sur les bords courbes des pièces de monnaie.

    Rôle de deque :
    ---------------
    On utilise une file (deque) pour faire un parcours en largeur BFS.
    Cela sert à explorer tous les pixels appartenant à la même composante.

    Descripteurs calculés :
    -----------------------
    area           : nombre de pixels de la composante
    bbox           : rectangle englobant
    hauteur_bbox   : hauteur du rectangle
    largeur_bbox   : largeur du rectangle
    remplissage    : aire / aire_bbox
    circularite    : 4πA / P²
    touche_bord    : True si la composante touche le bord de l'image

    Pourquoi c'est une grosse amélioration :
    ----------------------------------------
    Avant, on se contentait souvent de l'aire.
    Maintenant, on peut distinguer :
    - un objet rond et compact
    - un parasite allongé
    - un objet coupé par le bord

    Exemple :
    ---------
    Deux objets peuvent avoir la même aire :
    - un disque compact
    - une trace fine et allongée
    La circularité et le remplissage permettent de les différencier.
    """
    masque = image_binaire.astype(bool)
    hauteur, largeur = masque.shape
    etiquettes = np.zeros((hauteur, largeur), dtype=np.int32)
    composantes = []
    etiquette = 0
    voisins = (
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1),
    )

    for y in range(hauteur):
        for x in range(largeur):
            if not masque[y, x] or etiquettes[y, x] != 0:
                continue

            etiquette += 1
            file_pixels = deque([(y, x)])
            etiquettes[y, x] = etiquette
            pixels_y = []
            pixels_x = []
            touche_bord = False

            while file_pixels:
                cy, cx = file_pixels.popleft()
                pixels_y.append(cy)
                pixels_x.append(cx)

                if cy == 0 or cx == 0 or cy == hauteur - 1 or cx == largeur - 1:
                    touche_bord = True

                for dy, dx in voisins:
                    ny = cy + dy
                    nx = cx + dx
                    if (
                        0 <= ny < hauteur
                        and 0 <= nx < largeur
                        and masque[ny, nx]
                        and etiquettes[ny, nx] == 0
                    ):
                        etiquettes[ny, nx] = etiquette
                        file_pixels.append((ny, nx))

            pixels_y = np.asarray(pixels_y, dtype=np.int32)
            pixels_x = np.asarray(pixels_x, dtype=np.int32)
            area = int(pixels_y.size)
            y_min = int(pixels_y.min())
            y_max = int(pixels_y.max())
            x_min = int(pixels_x.min())
            x_max = int(pixels_x.max())
            hauteur_bbox = y_max - y_min + 1
            largeur_bbox = x_max - x_min + 1

            # OPTIMISATION : Calcul rapide du périmètre sans érosion coûteuse
            # Compte les pixels de contour (pixels blancs avec au moins un voisin noir)
            perimetre = 0
            for py, px in zip(pixels_y, pixels_x):
                # Vérifier si ce pixel est un pixel de contour
                is_border = False
                for dy, dx in voisins:
                    ny, py_check = py + dy, py
                    nx, px_check = px + dx, px
                    # Si voisin hors limites ou noir, c'est un pixel de contour
                    if not (0 <= ny < hauteur and 0 <= nx < largeur and masque[ny, nx]):
                        is_border = True
                        break
                if is_border:
                    perimetre += 1
            
            # Éviter division par zéro
            if perimetre == 0:
                perimetre = max(1, area)  # Sinon utiliser l'aire comme fallback
            
            remplissage = area / float(hauteur_bbox * largeur_bbox)
            circularite = 4.0 * math.pi * area / max(1.0, perimetre * perimetre)

            composantes.append(
                {
                    "label": etiquette,
                    "area": area,
                    "bbox": (y_min, x_min, y_max, x_max),
                    "hauteur_bbox": hauteur_bbox,
                    "largeur_bbox": largeur_bbox,
                    "remplissage": remplissage,
                    "circularite": circularite,
                    "touche_bord": touche_bord,
                }
            )

    return composantes


def extraire_composantes_utiles(masque, aire_min, aire_max):
    """
    Filtre les composantes détectées pour ne garder que les composantes plausibles.

    Étapes :
    --------
    1) extraire les composantes connexes du masque
    2) garder seulement celles dont :
       - l'aire est comprise entre aire_min et aire_max
       - la composante ne touche pas le bord

    Pourquoi ce filtre :
    --------------------
    Tout ce qui est segmenté en blanc n'est pas forcément une pièce.
    Il peut y avoir :
    - du bruit
    - une grande zone de fond mal segmentée
    - une pièce coupée au bord

    Exemple :
    ---------
    Si une composante a une aire plausible mais touche le bord,
    on la rejette, car elle peut être partiellement hors champ.
    """
    composantes = composantes_connexes(masque)
    return [
        comp
        for comp in composantes
        if aire_min <= comp["area"] <= aire_max and not comp["touche_bord"]
    ]


def estimer_nombre_depuis_composantes(composantes):
    """
    Estime le nombre de pièces réelles à partir des composantes détectées.
    
    COURS : Week 7 - Analyse statistique d'objets
    ============================================
    
    PROBLÈME À RÉSOUDRE :
    Une composante connexe ≠ une pièce dans 100% des cas :
    - Cas 1 : Deux pièces qui se touchent → 1 seule composante (sous-comptage)
    - Cas 2 : Une pièce brillante → 2-3 régions (sur-comptage)
    
    STRATÉGIE (exemple seuil + ratio) :
    1. Identifier des \"pièces de référence\" :
       - circularité >= COIN_REFERENCE_CIRCULARITY_MIN
       - remplissage >= COIN_REFERENCE_FILL_MIN
       - aspect ratio dans COIN_REFERENCE_ASPECT_{MIN,MAX}
       
    2. Calculer l'aire MÉDIANE de ces références = aire_typique
    
    3. Pour chaque composante :
       - Si aire < 1.8 * aire_typique : compter comme 1 pièce
       - Si aire >= 1.8 * aire_typique : compter comme round(aire/aire_typique) pièces
    
    EXEMPLE CONCRET :
    - Pièce typique = 12000 pixels
    - Composante trouvée = 24000 pixels
    - Ratio = 24000/12000 = 2.0 → compte comme 2 pièces
    
    AMÉLIORATIONS AU MAE :
    Réduit les grosses erreurs (prédire 1 au lieu de 4)

    Pourquoi cette fonction est cruciale :
    -------------------------------------
    Une composante n'est pas forcément une seule pièce.
    Si deux pièces se touchent, elles peuvent fusionner en une seule région.

    Idée :
    ------
    1) On choisit des composantes "de référence" qui ressemblent bien
       à des pièces normales :
       - circularité correcte
       - remplissage correct
       - ratio bbox raisonnable

    2) On calcule leur aire médiane :
       -> cela donne une aire typique d'une pièce

    3) Pour chaque composante :
       - si son aire vaut environ 2 fois l'aire typique, on compte 2
       - si elle vaut environ 3 fois, on peut compter 3
       - sinon on compte 1

    Exemple :
    ---------
    Si une pièce "typique" vaut 12000 pixels,
    une composante à 24000 pixels peut représenter 2 pièces collées.

    Pourquoi cela améliore le MAE :
    -------------------------------
    Parce que cela corrige des sous-comptages fréquents.
    """
    if not composantes:
        return 0

    # Utilisation des hyperparamètres définis en haut du fichier
    aires_reference = [
        comp["area"]
        for comp in composantes
        if comp["circularite"] >= COIN_REFERENCE_CIRCULARITY_MIN
        and comp["remplissage"] >= COIN_REFERENCE_FILL_MIN
        and COIN_REFERENCE_ASPECT_MIN
        <= comp["hauteur_bbox"] / max(1, comp["largeur_bbox"])
        <= COIN_REFERENCE_ASPECT_MAX
    ]

    if not aires_reference:
        return len(composantes)

    aire_reference = float(np.median(aires_reference))
    compteur = 0

    for comp in composantes:
        ratio = comp["area"] / max(1.0, aire_reference)
        if ratio >= MERGE_AREA_RATIO_THRESHOLD and comp["remplissage"] >= MERGE_FILL_MIN:
            compteur += max(1, int(round(ratio)))
        else:
            compteur += 1

    return compteur


# =============================================================================
# DÉTECTION PRINCIPALE
# =============================================================================
def detection_principale(image_rgb, taille_flou):
    """
    Détection principale basée sur la saturation.
    
    COURS : Semaines 3 à 8 - Pipeline complet de traitement d'image
    ===============================================================
    
    PIPELINE DÉTAILLÉ :
    
    Étape 1 : Transformation couleur (Week 3)
    ├─ RGB → HSL
    └─ Extraction du canal Saturation
    
    Étape 2 : Opération locale (Week 8)
    ├─ Flou gaussien à taille adaptative
    └─ Réduit le bruit avant segmentation
    
    Étape 3 : Seuillage (Week 5)
    ├─ Algorithme d'Otsu automatique
    └─ Crée image binaire sans paramètre manuel
    
    Étape 4 : Opérations morphologiques (Week 6)
    ├─ Ouverture binaire = Érosion puis Dilation
    └─ Nettoie les petits parasites
    
    Étape 5 : Analyse d'objets (Week 7)
    ├─ Extraction des composantes connexes
    ├─ Calcul des descripteurs de forme
    └─ Filtrage par aire et limites d'image
    
    Étape 6 : Estimation (Statistical analysis)
    └─ Compte le nombre de pièces probable

    But : Détecter les pièces via leur saturation (robuste aux ombres)

    Pipeline :
    ----------
    1) conversion RGB -> HSL, puis récupération de la saturation
    2) flou gaussien
    3) seuillage d'Otsu
    4) ouverture binaire
    5) calcul des seuils d'aire
    6) extraction des composantes utiles
    7) estimation du nombre de pièces

    Pourquoi cette détection :
    --------------------------
    La saturation est souvent utile pour faire ressortir les pièces
    par rapport au fond.

    Paramètre taille_flou :
    -----------------------
    Ici taille_flou est donné comme un tuple, par exemple (7,7).
    On en tire une taille moyenne locale.

    Pourquoi une taille locale adaptative :
    ---------------------------------------
    On combine :
    - la taille de flou demandée
    - la taille de l'image

    Cela permet d'éviter un flou trop faible sur une grande image
    ou trop fort sur une petite image.

    Exemple :
    ---------
    Si l'image est grande, on peut prendre un noyau un peu plus grand
    pour lisser correctement le bruit.
    """
    _, saturation = rgb_vers_hsl(image_rgb)
    taille_locale = max(
        int(round((taille_flou[0] + taille_flou[1]) / 2.0)),
        int(round(min(image_rgb.shape[:2]) / 90)),
        5,
    )
    if taille_locale % 2 == 0:
        taille_locale += 1

    saturation_floue = flou_gaussien(saturation, taille_locale)
    seuil = seuil_otsu(saturation_floue)
    masque = saturation_floue > seuil

    taille_morpho = max(3, int(round(min(image_rgb.shape[:2]) / 110)))
    if taille_morpho % 2 == 0:
        taille_morpho += 1
    masque = ouverture_binaire(masque, taille_morpho)

    aire_image = image_rgb.shape[0] * image_rgb.shape[1]
    aire_min = max(500, int(aire_image * 0.0010))
    aire_max = int(aire_image * 0.18)

    composantes = extraire_composantes_utiles(masque, aire_min, aire_max)
    return estimer_nombre_depuis_composantes(composantes), composantes


# =============================================================================
# DÉTECTION SPÉCIALE : CAS "UNE SEULE PIÈCE"
# =============================================================================
def detection_piece_unique(image_rgb):
    """
    Détection spécialisée : \"Y a-t-il probablement exactement 1 pièce ?\"
    
    COURS : Week 7 - Validation & stratégies de détection
    =====================================================
    
    BUT : Filet de sécurité si la détection principale doute
    
    Cas où c'est utile :
    - Pièce unique mal segmentée par voie saturation → prédiction 0 ou 4
    - Pièce brillante qui crée plusieurs régions → sur-comptage
    
    STRATÉGIE ALTERNATIVE (Contraste avec fond) :
    
    Au lieu d'utiliser la saturation :
    1. Convertir en niveaux de gris (Week 3)
    2. Estimer la couleur du fond (médiane des bords)
    3. Soustraire le fond à l'image
    4. Déterminer où la différence est grande
    
    Cette approche :
    - Déteste les reflets (très différents du fond)
    - Déteste les zones ombragées (différentes du fond)
    - Isole mieux une pièce unique sur fond assez uniforme
    
    CRITÈRES TRÈS STRICTS (pour éviter les faux positifs) :
    - remplissage >= SINGLE_COIN_FILL_MIN
    - circularité >= SINGLE_COIN_CIRCULARITY_MIN  
    - ratio hauteur/largeur dans [SINGLE_COIN_ASPECT_RATIO_MIN, MAX]
    
    Pourquoi la médiane des bords pour le fond :
    -------------------------------------
    La détection principale peut parfois :
    - prédire 0 alors qu'il y a 1 pièce
    - ou prédire 4 alors qu'il n'y a qu'une seule pièce mal segmentée

    Cette fonction sert donc de filet de sécurité.

    Pipeline détaillé :
    -------------------
    1) conversion en gris
    2) estimation du fond à partir des bords
    3) soustraction au fond
    4) flou gaussien
    5) seuillage d'Otsu
    6) ouverture binaire
    7) extraction des composantes utiles
    8) filtrage strict de candidats plausibles
    9) si on trouve exactement un candidat -> True

    Pourquoi la soustraction au fond :
    ----------------------------------
    Si le fond est assez uniforme, alors :
    - les pixels du fond sont proches de "fond"
    - la pièce diffère plus fortement

    Exemple :
    ---------
    fond = 0.7
    pixel fond = 0.72 -> différence = 0.02
    pixel pièce = 0.35 -> différence = 0.35

    Donc la pièce ressort.

    Pourquoi prendre la médiane des bords :
    ---------------------------------------
    On suppose que le fond est visible sur les bords de l'image.
    La médiane est robuste aux petites perturbations.

    Pourquoi des critères stricts à la fin :
    ----------------------------------------
    On veut éviter de conclure trop facilement qu'il y a une seule pièce.
    On impose donc :
    - remplissage élevé
    - circularité correcte
    - bbox proche d'un carré
    """
    gris = rgb_vers_gris(image_rgb)

    marge = max(8, int(round(min(gris.shape) * 0.03)))
    bord = np.concatenate(
        [
            gris[:marge, :].ravel(),
            gris[-marge:, :].ravel(),
            gris[:, :marge].ravel(),
            gris[:, -marge:].ravel(),
        ]
    )
    fond = float(np.median(bord))

    difference = np.abs(gris - fond)
    difference_floue = flou_gaussien(difference, max(5, marge | 1))
    seuil = max(0.08, seuil_otsu(difference_floue))
    masque = ouverture_binaire(difference_floue > seuil, 5)

    aire_image = image_rgb.shape[0] * image_rgb.shape[1]
    aire_min = max(600, int(aire_image * 0.003))
    aire_max = int(aire_image * 0.30)
    composantes = extraire_composantes_utiles(masque, aire_min, aire_max)

    # Utilisation des hyperparamètres
    candidates = [
        comp
        for comp in composantes
        if comp["remplissage"] >= SINGLE_COIN_FILL_MIN
        and comp["circularite"] >= SINGLE_COIN_CIRCULARITY_MIN
        and SINGLE_COIN_ASPECT_RATIO_MIN
        <= comp["hauteur_bbox"] / max(1, comp["largeur_bbox"])
        <= SINGLE_COIN_ASPECT_RATIO_MAX
    ]

    return len(candidates) == 1


# =============================================================================
# FONCTION PRINCIPALE : COMPTER LES PIÈCES
# =============================================================================
def compter_pieces(chemin_image, taille_flou=(7, 7)):
    """
    Fonction principale : compte le nombre de pièces dans une image.
    
    APERÇU GLOBAL :
    
    Ce programme implémente un PIPELINE COMPLET de traitement d'image,
    démontrant les concepts de chaque semaine du cours.
    
    SEMAINE 1-2  : Représentation (lire l'image, normaliser en RGB propre)
    SEMAINE 3    : Espaces couleur (HSL saturation pour la robustesse)
    SEMAINE 5    : Seuillage (Otsu automatique, diviseur 2 classes)
    SEMAINE 6    : Morphologie (ouverture = érosion + dilatation)
    SEMAINE 7    : Composantes connexes (8-connectivity, descripteurs)
    SEMAINE 8    : Convolution & Filtrage (Gaussian blur séparable)
    
    FLUX DE CONTRÔLE :
    
    1. ENTRÉE : chemin_image + taille_flou=[7,7]
    
    2. DÉTECTION PRINCIPALE
       ├─ Voie saturation (robuste aux ombres)
       ├─ Applique tout le pipeline Weeks 1-8
       └─ Retourne : prédiction + composantes détaillées
       
    3. DÉTECTION SECONDAIRE (Filet de sécurité)
       ├─ Voie contraste gris (alternative)
       ├─ Cherche si exactement 1 pièce probable
       └─ Retourne : booléen True/False
       
    4. RÈGLES CORRECTIVES (Post-traitement)
       ├─ Règle 1 : Si prédiction >= 4 mais 1 grosse pièce → corrige à 1
       ├─ Règle 2 : Si prédiction >= 4 mais peu de composantes → corrige à 1
       └─ Règle 3 : Si détection secondaire positive → applique correction
       
    5. SORTIE : nombre final de pièces
    
    RAISON DES CORRECTIONS :
    Une erreur de 4 au lieu de 1 = MAE +3
    Corriger à 1 = MAE +0
    → Important pour réduire le MAE (metrics Week 10)

    Étapes globales :
    -----------------
    1) lecture et normalisation de l'image
    2) détection principale
    3) détection spéciale "une seule pièce"
    4) règles correctives finales
    5) retour de la prédiction finale

    Pourquoi cette structure :
    --------------------------
    On sépare :
    - la prédiction principale
    - la logique de correction

    Cela rend le pipeline plus robuste et plus lisible.

    Idée de la correction finale :
    ------------------------------
    Même si la détection principale se trompe,
    on peut parfois détecter qu'il s'agit en réalité d'une seule pièce
    à partir de la forme globale observée.

    Exemple :
    ---------
    Cas difficile :
    - vraie valeur = 1
    - détection principale = 4
    - mais on observe une seule grande composante très circulaire
    -> on corrige à 1

    Pourquoi c'est bon pour le MAE :
    --------------------------------
    Une erreur de 4 au lieu de 1 donne une erreur absolue de 3.
    Si on corrige à 1, l'erreur devient 0.
    """
    image_rgb = lire_image_rgb(chemin_image)
    if image_rgb is None:
        return 0

    prediction_principale, composantes = detection_principale(image_rgb, taille_flou)
    piece_unique = detection_piece_unique(image_rgb)

    # Si on a des composantes, on peut appliquer des règles de cohérence globale.
    if composantes:
        aires = sorted(comp["area"] for comp in composantes)
        aire_mediane = float(np.median(aires))

        # On cherche des très grandes composantes bien rondes et bien remplies.
        #
        # Idée :
        # ------
        # Si la détection principale a beaucoup surcompté,
        # mais qu'on observe en réalité une seule grande forme circulaire,
        # cela suggère qu'il y a une seule grosse pièce.
        
        # Utilisation des hyperparamètres
        grandes_pieces_circulaires = [
            comp
            for comp in composantes
            if comp["circularite"] >= CORRECTION_LARGE_CIRCULARITY_MIN
            and comp["remplissage"] >= CORRECTION_LARGE_FILL_MIN
            and comp["area"] >= CORRECTION_LARGE_AREA_MULTIPLIER * max(1.0, aire_mediane)
        ]

        # Règle 1 :
        # Si la prédiction principale est très grande, mais qu'on voit une seule
        # énorme composante circulaire, on corrige à 1.
        if prediction_principale >= 4 and len(grandes_pieces_circulaires) == 1:
            return 1

        # Règle 2 :
        # Si la prédiction principale est grande, mais qu'il y a très peu
        # de composantes et qu'au moins l'une d'elles ressemble fortement à
        # une pièce, on corrige aussi à 1.
        
        # Utilisation des hyperparamètres
        if (
            prediction_principale >= 4
            and len(composantes) <= 2
            and any(comp["circularite"] >= CORRECTION_RARE_CIRCULARITY_MIN and comp["remplissage"] >= CORRECTION_RARE_FILL_MIN for comp in composantes)
        ):
            return 1

    # Règles liées à la détection spéciale "une seule pièce".
    if piece_unique:
        # Si la détection principale n'a rien vu, mais que la détection spéciale
        # voit clairement une seule pièce plausible, on corrige à 1.
        if prediction_principale == 0:
            return 1

        # Si la détection principale donne un nombre exagérément grand par rapport
        # au nombre de composantes, on corrige à 1.
        if prediction_principale >= 3 * max(1, len(composantes)):
            return 1

    return prediction_principale
import math
from collections import deque
import matplotlib.image as mpimg
import numpy as np

# =============================================================================
# FLUO GAUSSIEN : NOYAU ET CONVOLUTIONS
# =============================================================================
def noyau_gaussien_1d(taille, sigma):
    """
    Crée un noyau gaussien 1D normalisé pour le filtrage.
    
    Convolution & Filtrage Gaussien
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
    
    Convolution discrète
    -----------------------------------
    Convolution 1D discrète : (f * g)[n] = Σ f[m] * g[n-m]
    
    Mode 'same' : préserve la taille de l'image (padding aux bords) pour que:
    taille entrée = taille sortie
    
    np.convolve() implémente exactement la 
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
    
        ----------------------------
    Le code d'origine (commenté ci-dessous) utilisait np.einsum et sliding_window_view
    qui sont très complexes à justifier. Le nouveau code utilise np.convolve qui 
    traduit exactement la combinaison linéaire  .
    """
    # ----- ANCIEN CODE GARDÉ EN COMMENTAIRE -----
    # pad = len(noyau) // 2
    # image_pad = np.pad(image, ((0, 0), (pad, pad)), mode="edge")
    # fenetres = np.lib.stride_tricks.sliding_window_view(image_pad, len(noyau), axis=1)
    # return np.einsum("ijk,k->ij", fenetres, noyau[::-1], optimize=True)
    # ---------------------------------------------

    # Padding aux bords de l'image :
    # np.convolve avec mode='same' applique un ZERO PADDING implicitement.
    # Cela signifie que les pixels imaginaires en dehors de l'image valent 0.
    #
    # Conséquence : les pixels tout au bord de l'image seront légèrement
    # assombris (effet de vignettage), car la moyenne intègre des zéros.
    #
    # Ce n'est pas un problème ici, car on rejette de toute façon les
    # composantes connexes qui touchent le bord (voir touche_bord dans
    # composantes_connexes).
    return np.apply_along_axis(
        lambda ligne: np.convolve(
            ligne,        # une ligne de pixels (1D)
            noyau,        # filtre (poids de convolution)
            mode='same'   # conserve la même taille + zero padding implicite
        ),
        axis=1,           # 1 = on parcourt les lignes (horizontalement)
        arr=image         # image 2D (matrice de pixels)
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
    
    Opérations locales & Filtrage Gaussien
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
# GRADIENT DE SOBEL
# =============================================================================
def gradient_sobel(image):
    """
    Calcule le gradient de Sobel d'une image en niveaux de gris normalisée [0,1].

    Détection de contours, opérateur de Sobel
    -------------------------------------------------------------
    La détection de contours cherche les zones où l'intensité change brusquement.
    Un CONTOUR correspond à un fort gradient (variation rapide de l'intensité).

    FORMULE DU GRADIENT :
        G  = √(Gx² + Gy²)

    Où Gx et Gy sont les dérivées partielles (approximations discrètes) :
        Gx = image convoluée par le noyau horizontal de Sobel
        Gy = image convoluée par le noyau vertical de Sobel

    NOYAUX DE SOBEL ( :

        Noyau horizontal (détecte les changements de gauche à droite) :
            Kx = [[ 1,  0, -1],
                  [ 2,  0, -2],
                  [ 1,  0, -1]]

        Noyau vertical (détecte les changements de haut en bas) :
            Ky = [[ 1,  2,  1],
                  [ 0,  0,  0],
                  [-1, -2, -1]]

    SÉPARABILITÉ  :
    --------------------------
    Ces noyaux 2D peuvent être SÉPARÉS en deux convolutions 1D :

        Kx = colonne [-1, 0, 1]ᵀ × ligne [1, 2, 1]
        Ky = colonne [1, 2, 1]ᵀ  × ligne [-1, 0, 1]

    On réutilise nos fonctions de convolution 1D existantes (convolution_1d_lignes
    et convolution_1d_colonnes), ce qui rend le code cohérent avec le pipeline.

    ÉTAPES PAS-À-PAS :
    ------------------
    1) Appliquer le lissage horizontal (noyau [1,2,1] normalisé) sur les lignes,
       puis le gradient vertical (noyau [-1,0,1]) sur les colonnes → Gy
    2) Appliquer le gradient horizontal (noyau [-1,0,1]) sur les lignes,
       puis le lissage vertical (noyau [1,2,1] normalisé) sur les colonnes → Gx
    3) Calculer la magnitude : G = √(Gx² + Gy²)
    4) Normaliser G dans [0,1] pour rester compatible avec le pipeline

    UTILITÉ DANS LE PROJET :
    ------------------------
    La carte de gradient peut compléter la saturation :
    - Là où la saturation est faible (pièce peu colorée), le gradient
      peut encore révéler les contours de la pièce.
    - Utilisable comme canal alternatif ou en combinaison.

    Paramètre :
    -----------
    image : tableau 2D float [0,1]  (typiquement issu de rgb_vers_gris ou rgb_vers_hsl)

    Retour :
    --------
    gradient : tableau 2D float [0,1], valeurs élevées = contours détectés
    """
    image = image.astype(np.float32)

    # ---- Noyaux 1D de Sobel ----
    # Le noyau de Sobel 2D est séparable : on décompose en deux vecteurs 1D.
    # Noyau de lissage (poids binomial) :
    lissage  = np.array([1.0, 2.0, 1.0], dtype=np.float32) / 4.0   # somme = 1
    # Noyau de dérivée (différence centrée) :
    derivee  = np.array([-1.0, 0.0, 1.0], dtype=np.float32)         # pas de normalisation

    # ---- Gradient horizontal Gx ----
    # Kx détecte les variations de GAUCHE à DROITE.
    # Décomposition : d'abord lissage vertical (colonnes), puis dérivée horizontale (lignes).
    lisse_colonnes = convolution_1d_colonnes(image,  lissage)   # étape lissage
    gx             = convolution_1d_lignes(lisse_colonnes, derivee)   # étape dérivée

    # ---- Gradient vertical Gy ----
    # Ky détecte les variations de HAUT en BAS.
    # Décomposition : d'abord lissage horizontal (lignes), puis dérivée verticale (colonnes).
    lisse_lignes = convolution_1d_lignes(image,    lissage)   # étape lissage
    gy           = convolution_1d_colonnes(lisse_lignes, derivee)   # étape dérivée

    # ---- Magnitude du gradient : G = √(Gx² + Gy²) ----
    gradient = np.sqrt(gx * gx + gy * gy)

    # ---- Normalisation dans [0,1] ----
    # On divise par la valeur maximale pour garder la plage [0,1].
    val_max = gradient.max()
    if val_max > 0:
        gradient = gradient / val_max

    return gradient



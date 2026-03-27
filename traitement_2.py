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
# 1) lire l'image proprement et la normaliser
# 2) éventuellement la redimensionner pour réduire le coût de calcul
# 3) faire une détection principale à partir de la saturation (HSL)
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
#
# Exemples de cas difficiles que ce pipeline essaie de mieux gérer :
# ------------------------------------------------------------------
# - 2 pièces collées -> peuvent former une seule grosse composante
# - 1 seule pièce brillante -> peut être découpée en plusieurs régions
# - objet  au bord -> ne doit pas être compté
# - image très grande -> coût de calcul plus élevé si on ne redimensionne pas
#
# Pourquoi ce pipeline améliore le MAE :
# --------------------------------------
# Le MAE mesure l'erreur absolue moyenne entre prédiction et vérité terrain.
# Les grosses erreurs (par exemple prédire 4 au lieu de 1) augmentent beaucoup
# le MAE. Ici, le pipeline ajoute des mécanismes pour éviter ce type d'erreurs.
# =============================================================================


# Taille maximale autorisée pour le plus grand côté de l'image. apres on recalcule avec interpolation
# Pourquoi :
# ----------
# Traiter directement une grande image coûte cher.
# Exemple :
# - image 2000 x 1500 = 3 000 000 pixels
# - image ramenée à ~520 px sur le plus grand côté = beaucoup moins de pixels
#
# Donc :
# - flou plus rapide
# - morphologie plus rapide
# - composantes connexes plus rapides
# - temps global plus faible
MAX_DIMENSION = 520


# =============================================================================
# LECTURE ET NORMALISATION DE L'IMAGE
# =============================================================================
def lire_image_rgb(chemin_image):
    """
    Lit une image depuis le disque, la convertit dans un format RGB propre,
    puis la redimensionne si elle est trop grande.

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

    
    if image.ndim == 2:
        image = np.stack([image, image, image], axis=2)

   
    if image.shape[2] > 3:
        image = image[..., :3]


    if image.dtype != np.uint8:
        image = np.clip(np.rint(image * 255.0), 0, 255).astype(np.uint8)

    hauteur, largeur = image.shape[:2]

    # Calcul de l'échelle de redimensionnement.
    #
    # echelle <= 1 :
    # - si l'image est petite, echelle = 1 -> on ne change rien
    # - si l'image est grande, echelle < 1 -> on réduit
    echelle = min(1.0, MAX_DIMENSION / max(largeur, hauteur))

    # Redimensionnement si nécessaire.
    if echelle < 1.0:
        image = redimensionner_bilineaire(
            image,
            max(1, int(round(hauteur * echelle))),
            max(1, int(round(largeur * echelle))),
        )

    return image


# =============================================================================
# REDIMENSIONNEMENT BILINÉAIRE
# =============================================================================
#elle permet d'ameliorer les perf (plus rapide)
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
    """
    hauteur, largeur = image.shape[:2]

    # Si les dimensions ne changent pas, on renvoie une copie.
    if nouvelle_hauteur == hauteur and nouvelle_largeur == largeur:
        return image.copy()

    # Coordonnées de sortie ramenées dans le repère de l'image d'origine.
    #
    # y et x contiennent les positions réelles à échantillonner dans
    # l'image d'origine.
    y = np.linspace(0, hauteur - 1, nouvelle_hauteur, dtype=np.float32)
    x = np.linspace(0, largeur - 1, nouvelle_largeur, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)

    # Coordonnées entières du coin supérieur gauche.
    x0 = np.floor(xx).astype(np.int32)
    y0 = np.floor(yy).astype(np.int32)

    # Coordonnées du pixel voisin à droite / en bas.
    x1 = np.clip(x0 + 1, 0, largeur - 1)
    y1 = np.clip(y0 + 1, 0, hauteur - 1)

    # Poids d'interpolation.
    #
    # wx mesure où on est entre x0 et x1.
    # wy mesure où on est entre y0 et y1.
    wx = xx - x0
    wy = yy - y0

    image = image.astype(np.float32)

    # Les 4 voisins.
    haut_gauche = image[y0, x0]
    haut_droite = image[y0, x1]
    bas_gauche = image[y1, x0]
    bas_droite = image[y1, x1]

    # Interpolation horizontale sur la ligne du haut et sur la ligne du bas.
    haut = haut_gauche * (1.0 - wx)[..., None] + haut_droite * wx[..., None]
    bas = bas_gauche * (1.0 - wx)[..., None] + bas_droite * wx[..., None]

    # Interpolation verticale entre les deux résultats.
    resultat = haut * (1.0 - wy)[..., None] + bas * wy[..., None]

    return np.clip(np.rint(resultat), 0, 255).astype(np.uint8)


# =============================================================================
# CONVERSION COULEUR : RGB -> HSL (partiel : luminosité + saturation)
# =============================================================================
# la diff ici est que je retourne les 2 saturation et luminosité pas que la saturation
def rgb_vers_hsl(image_rgb):
    """
    Convertit une image RGB en deux composantes utiles ici :
    - luminosité
    - saturation

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

    Pourquoi cette fonction existe alors qu'on a déjà la saturation :
    -----------------------------------------------------------------
    Parce qu'une seule représentation ne suffit pas toujours.
    La saturation est bonne dans certains cas,
    mais pour le cas "une seule pièce", le contraste avec le fond
    en niveaux de gris peut être plus utile.

    Formule de luminance :
    ----------------------
    0.299 R + 0.587 G + 0.114 B (par chatgpt lgq yaeni)

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
    Crée un noyau gaussien 1D normalisé.

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
    rayon = taille // 2
    axe = np.arange(-rayon, rayon + 1, dtype=np.float32)
    noyau = np.exp(-(axe * axe) / (2.0 * sigma * sigma))
    somme = np.sum(noyau)
    if somme > 0:
        noyau /= somme
    return noyau


def convolution_1d_lignes(image, noyau):
    """
    Applique la convolution 1D horizontalement (ligne par ligne).

    Pourquoi une convolution 1D :
    -----------------------------
    Une gaussienne 2D peut être séparée en :
    - une convolution horizontale
    - puis une convolution verticale

    Cela donne le même résultat qu'une vraie gaussienne 2D,
    mais plus efficacement.

    Notion de fenêtre glissante :
    -----------------------------
    Pour chaque pixel, on regarde une petite fenêtre locale centrée autour
    de lui.

    Exemple sur une ligne :
    -----------------------
    Ligne = [10, 10, 20, 30, 30]
    Fenêtre taille 3 :
    - [10, 10, 20]
    - [10, 20, 30]
    - [20, 30, 30]

    Ensuite on multiplie cette fenêtre par le noyau puis on somme.

    Padding mode="edge" :
    ---------------------
    Aux bords, on prolonge la valeur du bord pour éviter de perdre des pixels.
    """
    pad = len(noyau) // 2
    image_pad = np.pad(image, ((0, 0), (pad, pad)), mode="edge")
    fenetres = np.lib.stride_tricks.sliding_window_view(image_pad, len(noyau), axis=1)
    return np.einsum("ijk,k->ij", fenetres, noyau[::-1], optimize=True)


def convolution_1d_colonnes(image, noyau):
    """
    Même principe que convolution_1d_lignes, mais verticalement.

    On applique maintenant le noyau sur les colonnes.
    """
    pad = len(noyau) // 2
    image_pad = np.pad(image, ((pad, pad), (0, 0)), mode="edge")
    fenetres = np.lib.stride_tricks.sliding_window_view(image_pad, len(noyau), axis=0)
    return np.einsum("ijk,k->ij", fenetres, noyau[::-1], optimize=True)


def flou_gaussien(image, taille):
    """
    Applique un flou gaussien à une image 2D.

    Pourquoi le flou gaussien est important :
    ----------------------------------------
    Avant le seuillage, on veut réduire :
    - le bruit
    - les petites variations locales
    - les détails qui risquent de perturber Otsu
    comme a dit lobruy

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
    image_u8 = np.clip(np.rint(image * 255.0), 0, 255).astype(np.uint8)
    return np.bincount(image_u8.ravel(), minlength=256).astype(np.float64)


def seuil_otsu(image):
    """
    Calcule automatiquement un seuil d'Otsu dans [0,1].

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
    """
    hist = histogramme_u8(image)
    total = hist.sum()
    if total == 0:
        return 0.5

    probabilites = hist / total
    cumul_prob = np.cumsum(probabilites)
    cumul_moy = np.cumsum(probabilites * np.arange(256))
    moyenne_totale = cumul_moy[-1]

    variance_inter = (moyenne_totale * cumul_prob - cumul_moy) ** 2
    variance_inter /= cumul_prob * (1.0 - cumul_prob) + 1e-12

    seuil = int(np.argmax(variance_inter))
    return seuil / 255.0


# =============================================================================
# MORPHOLOGIE BINAIRE
# =============================================================================
def erosion_binaire(image_binaire, taille):
    """
    Érosion binaire.

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
    Dilatation binaire.

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
    Ouverture binaire = érosion puis dilatation.

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
    plusieurs descripteurs de forme.

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

            sous_masque = etiquettes[y_min : y_max + 1, x_min : x_max + 1] == etiquette
            contour = sous_masque & ~erosion_binaire(sous_masque, 3)
            perimetre = int(np.count_nonzero(contour))
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
    Estime le nombre de pièces réelles à partir des composantes utiles.

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

    aires_reference = [
        comp["area"]
        for comp in composantes
        if comp["circularite"] >= 0.45
        and comp["remplissage"] >= 0.45
        and 0.65
        <= comp["hauteur_bbox"] / max(1, comp["largeur_bbox"])
        <= 1.55
    ]

    if not aires_reference:
        return len(composantes)

    aire_reference = float(np.median(aires_reference))
    compteur = 0

    for comp in composantes:
        ratio = comp["area"] / max(1.0, aire_reference)
        if ratio >= 1.8 and comp["remplissage"] >= 0.35:
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
    Détection spécialisée pour répondre à la question :
    "Y a-t-il probablement exactement une seule pièce ?"

    Pourquoi on a ajouté cette fonction :
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

    candidates = [
        comp
        for comp in composantes
        if comp["remplissage"] >= 0.62
        and comp["circularite"] >= 0.40
        and 0.78
        <= comp["hauteur_bbox"] / max(1, comp["largeur_bbox"])
        <= 1.28
    ]

    return len(candidates) == 1


# =============================================================================
# FONCTION PRINCIPALE : COMPTER LES PIÈCES
# =============================================================================
def compter_pieces(chemin_image, taille_flou=(7, 7)):
    """
    Fonction principale du programme.

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
        grandes_pieces_circulaires = [
            comp
            for comp in composantes
            if comp["circularite"] >= 0.48
            and comp["remplissage"] >= 0.68
            and comp["area"] >= 8.0 * max(1.0, aire_mediane)
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
        if (
            prediction_principale >= 4
            and len(composantes) <= 2
            and any(comp["circularite"] >= 0.55 and comp["remplissage"] >= 0.72 for comp in composantes)
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



#voici le resultat avec ça : 

# Évaluation sur le dataset : data/validation.json
# ----------------------------------------

# ----------------------------------------
# MAE (Erreur Absolue Moyenne)     : 1.79
# MSE (Erreur Quadratique Moyenne) : 12.26
# Nombre Moyen Réel de Pièces      : 3.83
# Pourcentage d'Erreur (MAE/Mean)  : 46.64%
# ----------------------------------------


# [ERREUR] img_001.jpg | Prédit: 0 | Réel: 2 | Diff: -2
# [ERREUR] img_002.jpg | Prédit: 1 | Réel: 10 | Diff: -9
# [ERREUR] img_003.jpg | Prédit: 0 | Réel: 16 | Diff: -16
# [ERREUR] img_004.jpg | Prédit: 6 | Réel: 7 | Diff: -1
# [ERREUR] img_005.jpg | Prédit: 5 | Réel: 6 | Diff: -1
# [ERREUR] img_006.jpg | Prédit: 0 | Réel: 1 | Diff: -1
# [OK] img_007.jpg | Prédit: 1 | Réel: 1
# [ERREUR] img_008.jpg | Prédit: 1 | Réel: 4 | Diff: -3
# [ERREUR] img_009.jpg | Prédit: 1 | Réel: 6 | Diff: -5
# [ERREUR] img_010.jpg | Prédit: 2 | Réel: 1 | Diff: 1
# [ERREUR] img_011.jpg | Prédit: 0 | Réel: 10 | Diff: -10
# [OK] img_012.jpg | Prédit: 3 | Réel: 3
# [OK] img_013.jpg | Prédit: 5 | Réel: 5
# [ERREUR] img_014.jpg | Prédit: 1 | Réel: 6 | Diff: -5
# [ERREUR] img_015.jpg | Prédit: 1 | Réel: 3 | Diff: -2
# [OK] img_016.jpg | Prédit: 1 | Réel: 1
# [OK] img_017.jpg | Prédit: 1 | Réel: 1
# [OK] img_018.jpg | Prédit: 1 | Réel: 1
# [ERREUR] img_019.jpg | Prédit: 8 | Réel: 6 | Diff: 2
# [OK] img_020.jpg | Prédit: 1 | Réel: 1
# [ERREUR] img_021.jpg | Prédit: 0 | Réel: 1 | Diff: -1
# [OK] img_022.jpg | Prédit: 8 | Réel: 8
# [OK] img_023.jpg | Prédit: 1 | Réel: 1
# [ERREUR] img_024.jpg | Prédit: 1 | Réel: 6 | Diff: -5
# [OK] img_025.jpg | Prédit: 1 | Réel: 1
# [ERREUR] img_026.jpg | Prédit: 1 | Réel: 2 | Diff: -1
# [OK] img_027.jpg | Prédit: 1 | Réel: 1
# [ERREUR] img_028.jpg | Prédit: 1 | Réel: 3 | Diff: -2
# [OK] img_029.jpg | Prédit: 4 | Réel: 4
# [ERREUR] img_030.jpg | Prédit: 0 | Réel: 4 | Diff: -4
# [OK] img_031.jpg | Prédit: 6 | Réel: 6
# [ERREUR] img_032.jpg | Prédit: 2 | Réel: 8 | Diff: -6
# [OK] img_033.jpg | Prédit: 1 | Réel: 1
# [OK] img_034.jpg | Prédit: 1 | Réel: 1
# [OK] img_035.jpg | Prédit: 10 | Réel: 10
# [OK] img_036.jpg | Prédit: 1 | Réel: 1
# [OK] img_037.jpg | Prédit: 1 | Réel: 1
# [ERREUR] img_038.jpg | Prédit: 8 | Réel: 7 | Diff: 1
# [OK] img_039.jpg | Prédit: 4 | Réel: 4
# [OK] img_040.jpg | Prédit: 1 | Réel: 1
# [ERREUR] img_041.jpg | Prédit: 1 | Réel: 3 | Diff: -2
# [OK] img_042.jpg | Prédit: 1 | Réel: 1
# [OK] img_043.jpg | Prédit: 1 | Réel: 1
# [OK] img_044.jpg | Prédit: 1 | Réel: 1
# [ERREUR] img_045.jpg | Prédit: 4 | Réel: 5 | Diff: -1
# [ERREUR] img_046.jpg | Prédit: 3 | Réel: 5 | Diff: -2
# [ERREUR] img_047.jpg | Prédit: 0 | Réel: 5 | Diff: -5
# [OK] img_048.jpg | Prédit: 1 | Réel: 1
# [OK] img_049.jpg | Prédit: 3 | Réel: 3
# [OK] img_050.jpg | Prédit: 1 | Réel: 1
# [OK] img_051.jpg | Prédit: 1 | Réel: 1
# [ERREUR] img_052.jpg | Prédit: 2 | Réel: 1 | Diff: 1
# [OK] img_053.jpg | Prédit: 1 | Réel: 1
# [ERREUR] img_054.jpg | Prédit: 8 | Réel: 10 | Diff: -2
# [OK] img_055.jpg | Prédit: 1 | Réel: 1
# [ERREUR] img_056.jpg | Prédit: 5 | Réel: 7 | Diff: -2
# [ERREUR] img_057.jpg | Prédit: 1 | Réel: 9 | Diff: -8
# [ERREUR] img_058.jpg | Prédit: 0 | Réel: 5 | Diff: -5
# [ERREUR] img_059.jpg | Prédit: 5 | Réel: 6 | Diff: -1
# [ERREUR] img_060.jpg | Prédit: 0 | Réel: 1 | Diff: -1
# [ERREUR] img_061.jpg | Prédit: 0 | Réel: 1 | Diff: -1
# [ERREUR] img_062.jpg | Prédit: 2 | Réel: 8 | Diff: -6
# [OK] img_063.jpg | Prédit: 1 | Réel: 1
# [ERREUR] img_064.jpg | Prédit: 1 | Réel: 15 | Diff: -14
# [OK] img_065.jpg | Prédit: 1 | Réel: 1
# [ERREUR] img_066.jpg | Prédit: 2 | Réel: 4 | Diff: -2
# [ERREUR] img_067.jpg | Prédit: 2 | Réel: 1 | Diff: 1
# [ERREUR] img_068.jpg | Prédit: 2 | Réel: 16 | Diff: -14
# [OK] img_069.jpg | Prédit: 1 | Réel: 1
# [OK] img_070.jpg | Prédit: 4 | Réel: 4
# [OK] img_071.jpg | Prédit: 1 | Réel: 1
# [ERREUR] img_072.jpg | Prédit: 0 | Réel: 5 | Diff: -5
# [ERREUR] img_073.jpg | Prédit: 1 | Réel: 7 | Diff: -6
# [ERREUR] img_074.jpg | Prédit: 0 | Réel: 1 | Diff: -1
# [ERREUR] img_075.jpg | Prédit: 0 | Réel: 1 | Diff: -1
# [ERREUR] img_076.jpg | Prédit: 8 | Réel: 13 | Diff: -5
# [OK] img_077.jpg | Prédit: 1 | Réel: 1
# [ERREUR] img_078.jpg | Prédit: 1 | Réel: 3 | Diff: -2
# [ERREUR] img_079.jpg | Prédit: 1 | Réel: 5 | Diff: -4
# [OK] img_080.jpg | Prédit: 4 | Réel: 4
# [OK] img_081.jpg | Prédit: 1 | Réel: 1
# [ERREUR] img_082.jpg | Prédit: 0 | Réel: 4 | Diff: -4
# [OK] img_083.jpg | Prédit: 3 | Réel: 3
# [ERREUR] img_084.jpg | Prédit: 14 | Réel: 17 | Diff: -3
# [ERREUR] img_085.jpg | Prédit: 0 | Réel: 6 | Diff: -6
# [OK] img_086.jpg | Prédit: 1 | Réel: 1
# [OK] img_087.jpg | Prédit: 1 | Réel: 1
# [OK] img_088.jpg | Prédit: 1 | Réel: 1
# [OK] img_089.jpg | Prédit: 1 | Réel: 1
# [ERREUR] img_090.jpg | Prédit: 0 | Réel: 1 | Diff: -1
# [ERREUR] img_091.jpg | Prédit: 0 | Réel: 10 | Diff: -10
# [OK] img_092.jpg | Prédit: 1 | Réel: 1
# [ERREUR] img_093.jpg | Prédit: 0 | Réel: 10 | Diff: -10
# [OK] img_094.jpg | Prédit: 3 | Réel: 3
# [ERREUR] img_095.jpg | Prédit: 2 | Réel: 1 | Diff: 1
# [ERREUR] img_096.jpg | Prédit: 0 | Réel: 1 | Diff: -1
# [ERREUR] img_097.jpg | Prédit: 1 | Réel: 2 | Diff: -1
# [OK] img_098.jpg | Prédit: 1 | Réel: 1
# [ERREUR] img_099.jpg | Prédit: 13 | Réel: 12 | Diff: 1
# [OK] img_100.jpg | Prédit: 1 | Réel: 1
# [OK] img_101.jpg | Prédit: 1 | Réel: 1
# [ERREUR] img_102.jpg | Prédit: 1 | Réel: 4 | Diff: -3
# [OK] img_103.jpg | Prédit: 11 | Réel: 11
# [OK] img_104.jpg | Prédit: 1 | Réel: 1
# [ERREUR] img_105.jpg | Prédit: 0 | Réel: 1 | Diff: -1
# [ERREUR] img_106.jpg | Prédit: 2 | Réel: 8 | Diff: -6
# [ERREUR] img_107.jpg | Prédit: 3 | Réel: 4 | Diff: -1
# [ERREUR] img_108.jpg | Prédit: 2 | Réel: 8 | Diff: -6
# [ERREUR] img_109.jpg | Prédit: 2 | Réel: 4 | Diff: -2
# [ERREUR] img_110.jpg | Prédit: 0 | Réel: 1 | Diff: -1
# [ERREUR] img_111.jpg | Prédit: 0 | Réel: 7 | Diff: -7
# [OK] img_112.jpg | Prédit: 1 | Réel: 1
# [OK] img_113.jpg | Prédit: 1 | Réel: 1
# [ERREUR] img_114.jpg | Prédit: 0 | Réel: 4 | Diff: -4
# [OK] img_115.jpg | Prédit: 1 | Réel: 1
# [ERREUR] img_116.jpg | Prédit: 1 | Réel: 5 | Diff: -4
# [ERREUR] img_117.jpg | Prédit: 0 | Réel: 1 | Diff: -1
# [ERREUR] img_118.jpg | Prédit: 0 | Réel: 1 | Diff: -1
# [OK] img_119.jpg | Prédit: 1 | Réel: 1
# [OK] img_120.jpg | Prédit: 1 | Réel: 1
# [ERREUR] img_121.jpg | Prédit: 0 | Réel: 1 | Diff: -1
# [OK] img_122.jpg | Prédit: 1 | Réel: 1
# [OK] img_123.jpg | Prédit: 4 | Réel: 4
# [OK] img_124.jpg | Prédit: 2 | Réel: 2
# [OK] img_125.jpg | Prédit: 12 | Réel: 12
# [OK] img_126.jpg | Prédit: 5 | Réel: 5
# [ERREUR] img_127.jpg | Prédit: 1 | Réel: 3 | Diff: -2
# [OK] img_128.jpg | Prédit: 4 | Réel: 4
# [OK] img_129.jpg | Prédit: 2 | Réel: 2
# [ERREUR] img_130.jpg | Prédit: 2 | Réel: 1 | Diff: 1
# [OK] img_131.jpg | Prédit: 6 | Réel: 6
# [OK] img_132.jpg | Prédit: 1 | Réel: 1
# [OK] img_133.jpg | Prédit: 11 | Réel: 11
# [OK] img_134.jpg | Prédit: 5 | Réel: 5
# [OK] img_135.jpg | Prédit: 6 | Réel: 6
# [OK] img_136.jpg | Prédit: 1 | Réel: 1
# [OK] img_137.jpg | Prédit: 1 | Réel: 1
# [OK] img_138.jpg | Prédit: 1 | Réel: 1
# [ERREUR] img_139.jpg | Prédit: 2 | Réel: 1 | Diff: 1
# [OK] img_140.jpg | Prédit: 1 | Réel: 1

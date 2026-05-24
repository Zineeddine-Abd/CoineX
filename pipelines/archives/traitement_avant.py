import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import deque
import math


# -------------------------------------------------------------------------
# Convertit une image RGB en niveaux de gris normalisés entre 0 et 1.
#
# Pourquoi je l'ai ajouté:
# pcq elle sert pour la detection spéciale "une seule pièce" pcq en niveau de gris c plus simple.
# dans certains cas la saturation ne suffit pas bien à séparer la pièce
# du fond ms le gris permet  de travailler sur le contraste 
#
# Formule utilisé :
# 0.299 R + 0.587 G + 0.114 B
# c une formule classique de luminance (3ami chatpgt)

#
# Exemple :
# Une pièce argentée sur fond clair peut être difficile à distinguer
# en saturation mais plus visible en niveaux de gris w la plupart des photos les table sont clair et certaine pièce tn 
# -------------------------------------------------------------------------
def rgb_vers_gris(image_rgb):
    image = image_rgb.astype(np.float32)
    return (
        0.299 * image[..., 0] + 0.587 * image[..., 1] + 0.114 * image[..., 2]
    ) / 255.0



# -------------------------------------------------------------------------
# là je vais divisé la fonction ouverture morph en trois fonctions plus simples : érosion, dilatation, ouverture binaire.
# Pourquoi :
# - c'est plus clair de voir les étapes séparées, et facile tn à expliquer pendant la présentation et aussi plus rapide à executer qu'avec pandas
#--------------------------------------------------------------------------


# -------------------------------------------------------------------------
# Réalise une érosion morphologique sur une image binaire sans pandas 
#
# Idée :
# Un pixel reste vrai (blanc) seulement si toute la fenêtre locale autour
# de lui est vraie. Cela permet de supprimer les petits bruits isolés
# et de rétrécir les objets
#
# Pourquoi je l'ai ajouté (vu qu'il existe deja ouverture_morph...) :
# pr remplacer les anciennes manipulations plus lourdes ou moins
# adaptées au binaire par une vraie morphologie binaire explicite.
#
# Exemple :
# Un pixel blanc isolé au milieu de noir disparaît après érosion.
# -------------------------------------------------------------------------
def erosion_binaire(image_binaire, taille):
    taille = max(1, int(taille))
    if taille % 2 == 0:
        taille += 1
    pad = taille // 2
    image_pad = np.pad(
        image_binaire.astype(bool), ((pad, pad), (pad, pad)), mode="constant", constant_values=False
    )
    fenetres = np.lib.stride_tricks.sliding_window_view(image_pad, (taille, taille))
    return np.all(fenetres, axis=(2, 3))


# -------------------------------------------------------------------------
# Réalise une dilatation morphologique sur une image binaire.
#
# Idée :
# Un pixel devient vrai blanc si au moins un pixel de sa fenêtre locale
# est vrai. Cela agrandit les objets et peut refermer de petits trous
#
# Pourquoi on l'a ajoutée :
# Elle complète l'érosion pour construire l'ouverture morphologique.
#
# Exemple :
# Une région blanche un peu amincie peut être regonflée après dilatation.
# -------------------------------------------------------------------------
def dilatation_binaire(image_binaire, taille):
    taille = max(1, int(taille))
    if taille % 2 == 0:
        taille += 1
    pad = taille // 2
    image_pad = np.pad(
        image_binaire.astype(bool),
        ((pad, pad), (pad, pad)),
        mode="constant",
        constant_values=False,
    )
    fenetres = np.lib.stride_tricks.sliding_window_view(image_pad, (taille, taille))
    return np.any(fenetres, axis=(2, 3))


# -------------------------------------------------------------------------
# Réalise une ouverture morphologique binaire :
# érosion puis dilatation.
#
# Pourquoi on l'a ajoutée :
# Cette opération nettoie le masque binaire après segmentation :
# - supprime les petits points parasites,
# - lisse un peu les formes,
# - conserve les objets importants.
#
# Exemple :
# Une pièce + quelques petits points de bruit :
# après ouverture, la pièce reste, les petits points disparaissent.
# -------------------------------------------------------------------------
def ouverture_binaire(image_binaire, taille):
    return dilatation_binaire(erosion_binaire(image_binaire, taille), taille)



def rgb_vers_saturation(img):
    """
    Remplace cv2.cvtColor(img, cv2.COLOR_BGR2HLS) et le split.
    Calcule mathématiquement le canal de Saturation depuis le RGB.
    """
    # Normalisation entre 0 et 1
    img = img.astype(float)
    if img.max() > 1.0:
        img /= 255.0
        
    cmax = np.max(img, axis=2)
    cmin = np.min(img, axis=2)
    delta = cmax - cmin
    luminosite = (cmax + cmin) / 2.0
    
    # Formule mathématique de la saturation HSL
    saturation = np.zeros_like(luminosite)
    mask = delta != 0
    saturation[mask] = delta[mask] / (1.0 - np.abs(2.0 * luminosite[mask] - 1.0))
    
    # Retour sur une échelle 0-255
    return (saturation * 255).astype(np.uint8)



def appliquer_flou_gaussien(img, ksize=15):
    """
    Remplace cv2.GaussianBlur. Utilise la convolution mathématique pure avec NumPy.
    """
    # 1. Création du noyau Gaussien 1D
    sigma = 0.3 * ((ksize - 1) * 0.5 - 1) + 0.8
    ax = np.linspace(-(ksize - 1) / 2., (ksize - 1) / 2., ksize)
    gauss = np.exp(-0.5 * np.square(ax) / np.square(sigma))
    kernel_1d = gauss / np.sum(gauss)
    
    # 2. Convolution séparable (horizontale puis verticale pour des performances optimales)
    flou_h = np.apply_along_axis(lambda m: np.convolve(m, kernel_1d, mode='same'), axis=1, arr=img)
    flou_v = np.apply_along_axis(lambda m: np.convolve(m, kernel_1d, mode='same'), axis=0, arr=flou_h)
    return flou_v



def seuillage_otsu(img):
    """
    Remplace cv2.threshold(..., cv2.THRESH_OTSU).
    """
    # Calcul de l'histogramme de l'image
    hist, _ = np.histogram(img.flatten(), 256, [0, 256])
    total = img.size
    sum_total = np.dot(np.arange(256), hist)
    
    sumB, wB, max_var, seuil_optimal = 0.0, 0, 0.0, 0
    
    # Parcours de tous les seuils de 0 à 255
    for i in range(256):
        wB += hist[i]  # Poids du fond
        if wB == 0: continue
        wF = total - wB  # Poids du premier plan
        if wF == 0: break
        
        sumB += i * hist[i]
        mB = sumB / wB  # Moyenne du fond
        mF = (sum_total - sumB) / wF  # Moyenne du premier plan
        
        # Variance inter-classe (Between-class variance)
        var_between = wB * wF * (mB - mF) ** 2
        
        # On cherche à maximiser cette variance
        if var_between > max_var:
            max_var = var_between
            seuil_optimal = i
            
    # Binarisation
    binarisee = np.zeros_like(img)
    binarisee[img > seuil_optimal] = 255
    return binarisee



def ouverture_morphologique(bin_img, ksize=15):
    """
    Remplace cv2.morphologyEx(..., cv2.MORPH_OPEN).
    Une astuce de Data Science consiste à utiliser Pandas pour l'érosion/dilatation !
    """
    df = pd.DataFrame(bin_img)
    
    # 1. ÉROSION = Recherche du Minimum local avec une fenêtre glissante (rolling)
    # Lignes
    eroded = df.rolling(window=ksize, center=True, min_periods=1).min()
    # Colonnes (on transpose, on applique, on re-transpose)
    eroded = eroded.T.rolling(window=ksize, center=True, min_periods=1).min().T
    
    # 2. DILATATION = Recherche du Maximum local sur l'image érodée
    dilated = eroded.rolling(window=ksize, center=True, min_periods=1).max()
    dilated = dilated.T.rolling(window=ksize, center=True, min_periods=1).max().T
    
    return dilated.values


# -------------------------------------------------------------------------
# Extrait les composantes connexes d'un masque binaire.
#
# C'est une des fonctions les plus importantes du nouveau code.
#
# Ce qu'elle fait :
# - repère chaque objet blanc séparé dans l'image,
# - lui attribue une étiquette,
# - calcule plusieurs descripteurs géométriques.
#
# Descripteurs calculés :
# - area : aire de l'objet
# - bbox : boîte englobante
# - hauteur_bbox / largeur_bbox
# - remplissage : à quel point l'objet remplit sa boîte
# - circularite : à quel point l'objet ressemble à un disque
# - touche_bord : indique si l'objet touche le bord de l'image
#
# Pourquoi elle a été modifiée :
# Avant, on comptait surtout à partir de l'aire.
# Maintenant, on veut analyser si une région ressemble vraiment à une pièce.
#
# Exemple :
# Deux objets peuvent avoir la même aire :
# - un disque compact
# - une grande trace allongée (reflet ou etc)
# Avec les nouveaux descripteurs, on peut les distinguer.
# -------------------------------------------------------------------------

def extraire_composantes_connexes(bin_img):
    """
    Remplace cv2.connectedComponentsWithStats.
    Algorithme de parcours en profondeur (DFS) pour trouver les objets connexes.
    """
    masque = bin_img.astype(bool)
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


# -------------------------------------------------------------------------
# Garde uniquement les composantes  comme pièces.
#
# Conditions :
# - aire entre aire_min et aire_max
# - ne touche pas le bord
#
# Pourquoi on l'a ajoutée :
# Après segmentation, tout ce qui est blanc n'est pas forcément une pièce
# Il faut donc filtrer les petits bruits, les très grosses zones aberrantes
# et les objets coupés par le bord.
#
# Exemple :
# On peut détecter :
# - une vraie pièce
# - un reflet 
# - une demi collée au bord
# On ne garde que l'objet crédible
# -------------------------------------------------------------------------
def extraire_composantes_utiles(composantes, aire_min, aire_max):
    return [
        comp
        for comp in composantes
        if aire_min <= comp["area"] <= aire_max and not comp["touche_bord"]
    ]


# -------------------------------------------------------------------------
# Estime le nombre réel de pièces à partir des composantes utiles.
#
# Pourquoi cette fonction est importante :
# Une composante détectée n'est pas forcément une seule pièce
# Si deux pièces se touchent elles etre représenter par une grande region 
#
# Principe :
# 1. On choisit des composantes de référence qui ressemblent bien à des pièces.
# 2. On calcule leur aire médiane = taille typique d'une pièce.
# 3. Pour chaque composante, on compare son aire à cette référence.
# 4. Si elle est environ 2 fois plus grande, on peut compter 2 pièces, etc.
#
# Exemple :
# Si une pièce normale vaut environ 12000 pixels
# et qu'une composante fait 24000 pixels,
# alors elle peut correspondre à 2 pièces fusionnées.
# -------------------------------------------------------------------------
def estimer_nombre_depuis_composantes(composantes):
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


# -------------------------------------------------------------------------
# Détection spéciale du cas "une seule pièce".
#
# Pourquoi on l'a ajoutée :
# La détection principale peut parfois surcompter ou rater une image
# contenant une seule pièce. Cette fonction ajoute une sécurité
#
# Étapes :
# 1. conversion en gris
# 2. estimation du fond à partir des bords pcq generalement les bord auront la meme couleur que le fond 
# 3. calcul de la différence au fond
# 4. flou + seuillage + ouverture
# 5. extraction des composantes
# 6. sélection stricte des candidats
#
# Si on trouve exactement un candidat on renvoie true
#
# Exemple :
# Une seule pièce brillante peut être découpée en plusieurs régions par
# la voie principale mais avec cette methode  c.a.d par contraste avec le fond on peut retrouver
# qu'il n'y a en réalité qu'une seule pièce
# -------------------------------------------------------------------------
def detection_piece_unique(image_rgb):
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

    diff_u8 = np.clip(np.rint(difference * 255.0), 0, 255).astype(np.uint8)
    difference_floue = appliquer_flou_gaussien(diff_u8, ksize=5)

    masque_u8 = seuillage_otsu(difference_floue)
    masque = ouverture_binaire(masque_u8 > 0, 5)

    aire_image = image_rgb.shape[0] * image_rgb.shape[1]
    aire_min = max(600, int(aire_image * 0.003))
    aire_max = int(aire_image * 0.30)

    composantes = extraire_composantes_connexes(masque)
    composantes_utiles = extraire_composantes_utiles(composantes, aire_min, aire_max)

    candidates = [
        comp
        for comp in composantes_utiles
        if comp["remplissage"] >= 0.62
        and comp["circularite"] >= 0.40
        and 0.78 <= comp["hauteur_bbox"] / max(1, comp["largeur_bbox"]) <= 1.28
    ]

    return len(candidates) == 1


# -------------------------------------------------------------------------
# Fonction principale : compte le nombre de pièces dans une image.
#
# C'est elle qui orchestre toute la chaîne de traitement.
#
# Étapes :
# 1. harmoniser la taille du flou
# 2. lire l'image
# 3. convertir en saturation
# 4. lisser par flou gaussien
# 5. segmenter par Otsu
# 6. nettoyer par ouverture binaire
# 7. extraire les composantes
# 8. garder seulement les composantes utiles
# 9. estimer le nombre de pièces à partir des formes
# 10. appliquer un correctif "pièce unique" si besoin
#
# Pourquoi cette fonction a été modifiée :
# L'ancien code faisait surtout un comptage simple par aires.
# Le nouveau code ajoute :
# - une vraie analyse de composantes,
# - la détection des objets fusionnés,
# - une correction des cas difficiles,
# - une logique spéciale pour les images contenant une seule pièce.
#
# Exemple :
# Si le pipeline principal prédit 4, mais qu'une seule grande composante
# circulaire est détectée et que la détection pièce unique est positive,
# on corrige la réponse finale à 1.
# -------------------------------------------------------------------------
def compter_pieces(chemin_image, taille_flou=5):
    # Si le flou est donné sous forme de tuple, on garde seulement la première valeur.
    # Exemple : (5,5) devient 5.
    if isinstance(taille_flou, tuple):
        taille_flou = taille_flou[0]

    # Le noyau du flou gaussien doit être impair  pour determiner facilmement centre
    if taille_flou % 2 == 0:
        taille_flou += 1

    img = plt.imread(chemin_image)

   
    saturation = rgb_vers_saturation(img)

    
    image_floue = appliquer_flou_gaussien(saturation, ksize=taille_flou)

   
    image_binaire = seuillage_otsu(image_floue)

    # Nettoyage du masque binaire.
    masque_bool = image_binaire > 0
    image_nettoyee = ouverture_binaire(masque_bool, 3)

    # Seuils d'aire proportionnels à la taille de l'image.
    aire_image = img.shape[0] * img.shape[1]
    aire_min = max(500, int(aire_image * 0.0010)) #meme explication que la ligne en bas
    aire_max = int(aire_image * 0.18) #les valeurs ont été regler manuellement pcq c la seule solution on ne peut pas les determiner automatiquement et il y a pas de valeur universelle 

    # Extraction et filtrage des composantes.
    composantes = extraire_composantes_connexes(image_nettoyee)
    composantes_utiles = extraire_composantes_utiles(composantes, aire_min, aire_max)

    # Prédiction principale.
    prediction_principale = estimer_nombre_depuis_composantes(composantes_utiles)

    # Détection de secours pour le cas "une seule pièce".
    piece_unique = detection_piece_unique(img)

    # Détection d'une éventuelle très grande pièce circulaire.
    grandes_pieces_circulaires = [
        comp for comp in composantes_utiles
        if comp["area"] >= max(aire_min * 2, int(aire_image * 0.02))
        and comp["circularite"] >= 0.60
        and comp["remplissage"] >= 0.55
    ]

    # Si le modèle a fortement surcompté mais qu'on voit surtout une grande pièce circulaire,
    # on corrige à 1. ( il peut y avoir une tres grande piece (une seule) mia que en fonction de l'aire elle compte pour plusieurs on corrige)
    if prediction_principale >= 4 and len(grandes_pieces_circulaires) == 1:
        return 1

    # Si très peu de composantes sont présentes mais qu'au moins l'une ressemble très fortement
    # à une vraie pièce, on corrige aussi à 1.
    if (
        prediction_principale >= 4
        and len(composantes_utiles) <= 2
        and any(
            comp["circularite"] >= 0.55 and comp["remplissage"] >= 0.72
            for comp in composantes_utiles
        )
    ):
        return 1

    # Si la détection spéciale "pièce unique" est positive, on corrige certains cas aberrants.
    if piece_unique:
        if prediction_principale == 0:
            return 1
        if prediction_principale >= 3 * max(1, len(composantes_utiles)):
            return 1

    # Sinon on garde la prédiction principale.
    return prediction_principale
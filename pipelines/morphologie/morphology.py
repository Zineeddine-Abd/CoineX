import math
from collections import deque
import matplotlib.image as mpimg
import numpy as np

# =============================================================================
# MORPHOLOGIE BINAIRE
# =============================================================================
def erosion_binaire(image_binaire, taille):
    """
    Érosion binaire (Binary Erosion Morphological Operation).
    
    Opérations morphologiques binaires
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
    
    Opérations morphologiques binaires
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

    Composition d'opérations morphologiques
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


def fermeture_binaire(image_binaire, taille):
    """
    Fermeture binaire = Dilatation suivie d'Érosion (Closing Operation).

    Morphologie mathématique (opérations composées)
    -------------------------------------------------------------------
    Formule : Fermeture = Erosion( Dilatation(I) )

    C'est l'opération INVERSE de l'ouverture :
    - L'ouverture  commence par éroder  -> supprime le bruit extérieur
    - La fermeture commence par dilater -> bouche les trous intérieurs

    Effet visuel :
    --------------
    Avant fermeture : un objet avec un petit trou noir à l'intérieur
    Après fermeture : le trou est bouché, l'objet est solide

    Pourquoi c'est utile pour notre projet (pièces de monnaie) :
    ------------------------------------------------------------
    Les pièces métalliques ont souvent des reflets lumineux en leur centre.
    Ces reflets, très clairs, peuvent être vus comme du "fond" par Otsu et
    créer un trou blanc à l'intérieur du masque de la pièce.

    Sans fermeture :
      masque = [1 1 1 1 1]   <- bord de la pièce = blanc (1)
               [1 1 0 1 1]   <- reflet central    = trou  (0) <- PROBLÈME
               [1 1 1 1 1]   <- bord de la pièce = blanc (1)

    Avec fermeture :
      masque = [1 1 1 1 1]
               [1 1 1 1 1]   <- le trou est rebouché
               [1 1 1 1 1]

    Ce trou, s'il n'est pas corrigé, peut faire croire à plusieurs
    composantes connexes au lieu d'une seule -> sur-comptage des pièces.

    Ordre dans le pipeline :
    ------------------------
    1. Ouverture  -> supprime les petits points parasites (bruit extérieur)
    2. Fermeture  -> bouche les petits trous internes (reflets)

    Les deux ensemble donnent un masque propre et solide.
    """
    # Dilatation d'abord : agrandit les zones blanches, bouche les petits trous
    # Érosion ensuite  : remet les objets à leur taille d'origine
    return erosion_binaire(dilatation_binaire(image_binaire, taille), taille)


# =============================================================================
# COMPOSANTES CONNEXES ET DESCRIPTEURS DE FORME
# =============================================================================
def composantes_connexes(image_binaire):
    """
    Extrait toutes les composantes connexes d'un masque binaire et calcule
    plusieurs descripteurs de forme pour chacune.
    
    Analyse d'objets & Composantes connexes
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
    
    DESCRIPTEURS CALCULÉS (Properties of objects) :
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
    Contrairement à la 4-connexité, 
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

            # -----------------------------------------------------------------
            # Périmètre par morphologie binaire
            # -----------------------------------------------------------------
            # Définition : le CONTOUR d'un objet = ses pixels qui ont au moins
            # un voisin en dehors de l'objet.
            #
            # Formule morphologique :
            #   Contour(A) = A  −  Érosion(A)
            #
            # Explication pas-à-pas :
            #   1) Érosion(A) : garde seulement les pixels "intérieurs"
            #      (ceux dont TOUS les voisins sont aussi dans l'objet).
            #   2) A − Érosion(A) : on enlève l'intérieur, il ne reste que
            #      les pixels de bord = le contour.
            #   3) Périmètre = nombre de pixels du contour.
            #
            # On extrait le sous-masque de la boîte englobante pour travailler
            # sur un petit tableau au lieu de l'image entière.
            # -----------------------------------------------------------------
            sous_masque = (etiquettes[y_min:y_max + 1, x_min:x_max + 1] == etiquette)
            interieur   = erosion_binaire(sous_masque, 3)
            contour     = sous_masque & ~interieur   # Contour = A − Érosion(A)
            perimetre   = int(np.sum(contour))        # périmètre = nb pixels du contour

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
